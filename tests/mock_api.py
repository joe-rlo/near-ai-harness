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
  mock/judge      answers with a refusal-judge verdict (see MockAPI.judge_reply)
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODELS = ["echo", "reasoner", "truncated", "refuser", "flaky", "stall", "down", "plain", "judge"]


class MockAPI:
    def __init__(self):
        self.requests = []  # parsed chat-completion bodies, in arrival order
        self.auth = []  # Authorization header of each chat request
        self.calls = {}  # model id -> number of chat requests
        self.fail_next = {}  # model id -> how many upcoming requests get a 502
        self.judge_reply = '{"label": "refused", "reason": "It declined."}'
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

            def do_GET(self):
                if self.path == "/v1/models":
                    self._json(200, {"data": [
                        {"id": f"mock/{name}", "name": name, "context_length": 32000,
                         "pricing": {"input": 1.0, "output": 2.0}} for name in MODELS]})
                else:
                    self._json(404, {"error": {"type": "not_found", "message": "no such route"}})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
                model = body.get("model")
                with mock._lock:
                    mock.requests.append(body)
                    mock.auth.append(self.headers.get("Authorization"))
                    mock.calls[model] = calls = mock.calls.get(model, 0) + 1
                    fail = mock.fail_next.get(model, 0)
                    mock.fail_next[model] = max(fail - 1, 0)

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
                    answer = [mock.judge_reply]
                else:
                    answer = [last_user]
                usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30,
                         "completion_tokens_details": {"reasoning_tokens": 15 if reasoning else 0}}

                if model == "mock/plain":
                    self._json(200, {"id": "chatcmpl-plain", "model": model, "usage": usage, "choices": [
                        {"message": {"content": "".join(answer)}, "finish_reason": finish}]})
                    return

                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("inference-id", "inf-123")
                self.end_headers()

                def send(chunk):
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    self.wfile.flush()

                def delta(**fields):
                    return {"id": "chatcmpl-mock", "model": model, "choices": [{"delta": fields}]}

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
                send({"id": "chatcmpl-mock", "model": model, "choices": [{"delta": {}, "finish_reason": finish}]})
                send({"id": "chatcmpl-mock", "model": model, "choices": [], "usage": usage})
                self.wfile.write(b"data: [DONE]\n\n")

        return Handler
