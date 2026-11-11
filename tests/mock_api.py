"""A tiny stand-in for the NEAR AI Cloud API, so tests never touch the network or spend credits.

The model id picks the behaviour:

  mock/echo       streams the last user message back
  mock/reasoner   streams reasoning first, then the answer "42"
  mock/truncated  reasons until it hits max_tokens; empty answer, finish_reason=length
  mock/refuser    answers "I can't help with that."
  mock/flaky      429 on the first call, then behaves like mock/echo
  mock/stall      goes silent mid-stream on the first call, then behaves like mock/echo
  mock/down       always 503
  mock/plain      ignores stream=true and returns one JSON body
  mock/judge      answers with a judge verdict: MockAPI.rubric_reply when asked about a rubric,
                  otherwise MockAPI.judge_reply

It also signs every completion the way NEAR's gateway does (GET /v1/signature/{id}) and serves an
attestation report naming its signing address. MockAPI.signature_mode makes it misbehave:

  ok           a correct gateway signature
  tampered     signs hashes of different bytes than it sent
  forged       claims the gateway's address, but the signature is by another key
  stranger     a valid signature by a key that is not in the attestation report
  missing      404
  unavailable  200 with an error_code instead of a signature
"""

import hashlib
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ethsig  # noqa: E402

GATEWAY_KEY = 0x1111111111111111111111111111111111111111111111111111111111111111
STRANGER_KEY = 0x2222222222222222222222222222222222222222222222222222222222222222


def address_of(key: int) -> str:
    return ethsig.public_key_to_address(ethsig._mul(key, ethsig._G))


def sign(text: str, key: int) -> str:
    """Ethereum personal_sign with a fixed test key. Test use only: the nonce is not secret."""
    n = ethsig._N
    z = int.from_bytes(ethsig.personal_message_hash(text), "big")
    k = int.from_bytes(hashlib.sha256(text.encode() + key.to_bytes(32, "big")).digest(), "big") % (n - 1) + 1
    point = ethsig._mul(k, ethsig._G)
    r, v = point[0] % n, point[1] & 1
    s = pow(k, -1, n) * (z + r * key) % n
    if s > n // 2:
        s, v = n - s, v ^ 1
    return "0x" + (r.to_bytes(32, "big") + s.to_bytes(32, "big") + bytes([27 + v])).hex()

MODELS = ["echo", "reasoner", "truncated", "refuser", "flaky", "stall", "down", "plain", "judge"]


class MockAPI:
    def __init__(self):
        self.requests = []  # parsed chat-completion bodies, in arrival order
        self.auth = []  # Authorization header of each chat request
        self.calls = {}  # model id -> number of chat requests
        self.fail_next = {}  # model id -> how many upcoming requests get a 502
        self.judge_reply = '{"label": "refused", "reason": "It declined."}'
        self.rubric_reply = '{"pass": true, "reason": "It covers every point."}'
        self.signature_mode = "ok"
        self.attestation_down = False
        self.signed = {}  # completion id -> (sha256 of request bytes, sha256 of response bytes)
        self.gateway_address = address_of(GATEWAY_KEY)
        self._lock = threading.Lock()
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.base_url = f"http://127.0.0.1:{self._server.server_address[1]}/v1"

    def start(self):
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def stop(self):
        self._server.shutdown()
        self._server.server_close()

    def _handler(mock):  # noqa: N805 - `mock` is the MockAPI; `self` below is the request handler
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _json(self, code, body, headers=None):
                raw = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(raw)

            def _signature(self, chat_id):
                mode = mock.signature_mode
                if mode == "missing" or chat_id not in mock.signed:
                    return self._json(404, {"error": {"type": "not_found", "message": "no signature"}})
                if mode == "unavailable":
                    return self._json(200, {"error_code": "signature_unavailable", "message": "not signed"})
                request_hash, response_hash = mock.signed[chat_id]
                if mode == "tampered":
                    response_hash = hashlib.sha256(b"something else").hexdigest()
                text = f"{request_hash}:{response_hash}"
                key = GATEWAY_KEY if mode in ("ok", "tampered") else STRANGER_KEY
                address = address_of(STRANGER_KEY) if mode == "stranger" else mock.gateway_address
                self._json(200, {"text": text, "signature": sign(text, key), "signing_address": address,
                                 "signing_algo": "ecdsa", "signature_kind": "gateway"})

            def do_GET(self):
                url = urlparse(self.path)
                query = parse_qs(url.query)
                if url.path.startswith("/v1/signature/"):
                    return self._signature(url.path.rsplit("/", 1)[-1])
                if url.path == "/v1/attestation/report":
                    if mock.attestation_down:
                        return self._json(503, {"error": {"type": "provider_error", "message": "no report"}})
                    return self._json(200, {"gateway_attestation": {
                        "signing_address": mock.gateway_address, "signing_algo": "ecdsa",
                        "request_nonce": (query.get("nonce") or [""])[0], "intel_quote": "00"}})
                if self.path == "/v1/models":
                    self._json(200, {"data": [
                        {"id": f"mock/{name}", "name": name, "context_length": 32000,
                         "pricing": {"input": 1.0, "output": 2.0}} for name in MODELS]})
                else:
                    self._json(404, {"error": {"type": "not_found", "message": "no such route"}})

            def do_POST(self):
                raw_request = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                body = json.loads(raw_request)
                model = body.get("model")
                with mock._lock:
                    mock.requests.append(body)
                    mock.auth.append(self.headers.get("Authorization"))
                    mock.calls[model] = calls = mock.calls.get(model, 0) + 1
                    chat_id = f"chatcmpl-mock-{len(mock.requests)}"
                    fail = mock.fail_next.get(model, 0)
                    mock.fail_next[model] = max(fail - 1, 0)
                sent = hashlib.sha256()

                def remember():
                    mock.signed[chat_id] = (hashlib.sha256(raw_request).hexdigest(), sent.hexdigest())

                if fail:
                    self._json(502, {"error": {"type": "bad_gateway", "message": "The model is currently unavailable."}},
                               {"Retry-After": "0"})
                    return
                if model == "mock/down":
                    self._json(503, {"error": {"type": "service_overloaded", "message": "try later"}})
                    return
                if model == "mock/flaky" and calls == 1:
                    self._json(429, {"error": {"type": "rate_limit_exceeded", "message": "slow down"}},
                               {"Retry-After": "0"})
                    return

                last_user = next(m["content"] for m in reversed(body["messages"]) if m["role"] == "user")
                reasoning, finish = [], "stop"
                if model == "mock/reasoner":
                    reasoning, answer = ["Let me ", "think."], ["4", "2"]
                elif model == "mock/truncated":
                    reasoning, answer, finish = ["Thinking ", "forever"], [], "length"
                elif model == "mock/refuser":
                    answer = ["I can't help with that."]
                elif model == "mock/judge":
                    asked_about_rubric = "rubric" in body["messages"][0]["content"]
                    answer = [mock.rubric_reply if asked_about_rubric else mock.judge_reply]
                else:
                    answer = [last_user]
                usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30,
                         "completion_tokens_details": {"reasoning_tokens": 15 if reasoning else 0}}

                if model == "mock/plain":
                    plain = {"id": chat_id, "model": model, "usage": usage, "choices": [
                        {"message": {"content": "".join(answer)}, "finish_reason": finish}]}
                    sent.update(json.dumps(plain).encode())
                    remember()
                    self._json(200, plain)
                    return

                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("inference-id", "inf-123")
                self.end_headers()

                def write(raw):
                    sent.update(raw)
                    self.wfile.write(raw)
                    self.wfile.flush()

                def send(chunk):
                    write(f"data: {json.dumps(chunk)}\n\n".encode())

                def delta(**fields):
                    return {"id": chat_id, "model": model, "choices": [{"delta": fields}]}

                if model == "mock/stall" and calls == 1:
                    send(delta(content="par"))
                    time.sleep(1.0)  # longer than the client's read timeout in the test
                    return
                for piece in reasoning:
                    send(delta(reasoning_content=piece))
                if reasoning:
                    time.sleep(0.03)  # so the first answer token is measurably later than the first token
                for piece in answer:
                    send(delta(content=piece))
                send({"id": chat_id, "model": model, "choices": [{"delta": {}, "finish_reason": finish}]})
                send({"id": chat_id, "model": model, "choices": [], "usage": usage})
                remember()  # before the last bytes go out, so the signature is there when the client asks
                sent.update(b"data: [DONE]\n\n")
                mock.signed[chat_id] = (mock.signed[chat_id][0], sent.hexdigest())
                self.wfile.write(b"data: [DONE]\n\n")

        return Handler
