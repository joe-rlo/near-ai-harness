#!/usr/bin/env python3
"""Minimal NEAR AI model test harness: model picker, prompt list, benchmark metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import ssl
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent


def load_dotenv(path: Path) -> None:
    """Minimal KEY=VALUE loader; variables already set in the environment win."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)


load_dotenv(ROOT / ".env")

STATIC = ROOT / "static"
RESULTS = ROOT / "results"
PORT = int(os.environ.get("PORT", "8765"))

RETRYABLE = {429, 500, 502, 503, 504}
MAX_RETRIES = 3
REQUEST_TIMEOUT = 180
CATALOG_TTL = 600
RUN_FILE_RE = re.compile(r"^run-[\w.-]+\.json$")
GROUP_RE = re.compile(r"^[\w-]{1,64}$")

# Heuristic only: catches common refusal openers, not paraphrased or partial refusals.
REFUSAL_RE = re.compile(
    r"\b(I\s+can(?:['’]?t|not)\s+(?:help|assist|provide|comply|do\s+that|support)"
    r"|I(?:['’]m|\s+am)\s+(?:not\s+able|unable)\s+to"
    r"|I\s+(?:won['’]t|will\s+not|must\s+decline)"
    r"|against\s+my\s+(?:guidelines|policies|programming)"
    r"|as\s+an\s+AI(?:\s+language\s+model)?,?\s+I)",
    re.IGNORECASE,
)


def load_json(path: Path):
    return json.loads(path.read_text())


def _base_url() -> str:
    env = os.environ.get("NEARAI_BASE_URL")
    if env:
        return env.rstrip("/")
    return (load_json(ROOT / "models.json").get("base_url") or "https://cloud-api.near.ai/v1").rstrip("/")


BASE_URL = _base_url()


def _ssl_context() -> ssl.SSLContext:
    """Default trust store, plus certifi or the macOS bundle when Python has none
    (python.org builds ship without CAs until "Install Certificates.command" is run)."""
    ctx = ssl.create_default_context()
    if ctx.cert_store_stats()["x509_ca"]:
        return ctx
    try:
        import certifi
        ctx.load_verify_locations(certifi.where())
    except ImportError:
        if Path("/etc/ssl/cert.pem").is_file():
            ctx.load_verify_locations("/etc/ssl/cert.pem")
    return ctx


SSL_CTX = _ssl_context()


def api_key() -> str:
    key = os.environ.get("NEARAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Set NEARAI_API_KEY in the environment before running the harness.")
    return key


def ms_since(t0: float) -> int:
    return round((time.perf_counter() - t0) * 1000)


# ---------------------------------------------------------------- model catalog

_catalog_lock = Lock()
_catalog_cache: dict = {"at": 0.0, "models": None}


def _fetch_live_models() -> dict:
    """GET /v1/models is public; pricing.input/output are USD per million tokens."""
    req = urllib.request.Request(f"{BASE_URL}/models", headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=15, context=SSL_CTX) as resp:
        data = json.loads(resp.read())
    return {m["id"]: m for m in data.get("data") or [] if m.get("id")}


def model_catalog() -> list[dict]:
    """Curated models.json entries, refreshed with live pricing; other live models appended."""
    with _catalog_lock:
        if _catalog_cache["models"] is not None and time.time() - _catalog_cache["at"] < CATALOG_TTL:
            return _catalog_cache["models"]
        curated = [dict(m, curated=True, pricing_source="models.json")
                   for m in load_json(ROOT / "models.json").get("models") or []]
        try:
            live = _fetch_live_models()
        except Exception as e:
            print(f"[harness] live model list unavailable ({e}); using models.json only")
            live = {}
        for m in curated:
            info = live.pop(m["id"], None)
            if not info:
                continue
            pricing = info.get("pricing") or {}
            if pricing.get("input") is not None and pricing.get("output") is not None:
                m["pricing_input_per_m"] = float(pricing["input"])
                m["pricing_output_per_m"] = float(pricing["output"])
                m["pricing_source"] = "live"
            m["context_length"] = info.get("context_length") or m.get("context_length")
            m["supported_features"] = info.get("supported_features") or []
        extras = []
        for mid, info in sorted(live.items()):
            pricing = info.get("pricing") or {}
            extras.append({
                "id": mid,
                "name": info.get("name") or mid,
                "abliterated": "abliterated" in mid.lower(),
                "context_length": info.get("context_length"),
                "pricing_input_per_m": pricing.get("input"),
                "pricing_output_per_m": pricing.get("output"),
                "description": info.get("description") or "",
                "supported_features": info.get("supported_features") or [],
                "curated": False,
                "pricing_source": "live",
            })
        _catalog_cache.update(at=time.time(), models=curated + extras)
        return _catalog_cache["models"]


def estimate_cost(model_id: str, prompt_tokens, completion_tokens) -> float | None:
    pricing = next((m for m in model_catalog() if m["id"] == model_id), None)
    if not pricing or prompt_tokens is None or completion_tokens is None:
        return None
    pin = pricing.get("pricing_input_per_m")
    pout = pricing.get("pricing_output_per_m")
    if pin is None or pout is None:
        return None
    return round((prompt_tokens / 1_000_000) * float(pin) + (completion_tokens / 1_000_000) * float(pout), 8)


# ---------------------------------------------------------------- completions


def empty_metrics(latency_ms=None) -> dict:
    return {
        "latency_ms": latency_ms,
        "ttft_ms": None,
        "decode_ms": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "reasoning_tokens": None,
        "total_tokens": None,
        "tokens_per_sec": None,
        "est_cost_usd": None,
    }


def _failure(status, error: str, latency_ms, attempts: int, headers=None) -> dict:
    headers = headers or {}
    return {
        "ok": False,
        "status": status,
        "error": error[:2000],
        "attempts": attempts,
        "inference_id": headers.get("inference-id"),
        "request_id": headers.get("x-request-id"),
        "metrics": empty_metrics(latency_ms),
    }


def _error_message(body: str) -> str:
    try:
        err = json.loads(body).get("error") or {}
        if isinstance(err, dict) and err.get("message"):
            return f"{err.get('type') or 'error'}: {err['message']}"
    except (json.JSONDecodeError, AttributeError):
        pass
    return body


def _backoff(attempt: int, retry_after) -> float:
    try:
        if retry_after:
            return min(float(retry_after), 30.0)
    except ValueError:
        pass
    return min(2 ** (attempt - 1), 8) + secrets.randbelow(500) / 1000


def _read_stream(resp, t0: float) -> dict:
    """Parse an OpenAI-style SSE stream. TTFT is the first content or reasoning delta."""
    out = {"id": None, "model": None, "content": [], "reasoning": [], "finish_reason": None,
           "usage": {}, "ttft_ms": None}
    for raw_line in resp:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        if chunk.get("error"):
            raise RuntimeError(_error_message(json.dumps(chunk)))
        out["id"] = out["id"] or chunk.get("id")
        out["model"] = out["model"] or chunk.get("model")
        if chunk.get("usage"):
            out["usage"] = chunk["usage"]
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            text = delta.get("content")
            thought = delta.get("reasoning_content") or delta.get("reasoning")
            if (text or thought) and out["ttft_ms"] is None:
                out["ttft_ms"] = ms_since(t0)
            if text:
                out["content"].append(text)
            if thought:
                out["reasoning"].append(thought)
            if choice.get("finish_reason"):
                out["finish_reason"] = choice["finish_reason"]
    out["content"] = "".join(out["content"])
    out["reasoning"] = "".join(out["reasoning"])
    return out


def _read_json(resp) -> dict:
    """Fallback for servers that ignore stream=true and return a single JSON body."""
    data = json.loads(resp.read())
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    return {
        "id": data.get("id"),
        "model": data.get("model"),
        "content": message.get("content") or "",
        "reasoning": message.get("reasoning_content") or message.get("reasoning") or "",
        "finish_reason": choice.get("finish_reason"),
        "usage": data.get("usage") or {},
        "ttft_ms": None,
    }


def chat_completion(model: str, prompt: str, *, max_tokens: int = 2048, temperature: float = 0.2,
                    system: str | None = None) -> dict:
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": prompt})
    body = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()

    attempt = 0
    while True:
        attempt += 1
        req = urllib.request.Request(
            f"{BASE_URL}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {api_key()}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
            method="POST",
        )
        t0 = time.perf_counter()
        try:
            resp = urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT, context=SSL_CTX)
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            headers = {k.lower(): v for k, v in e.headers.items()}
            if e.code in RETRYABLE and attempt <= MAX_RETRIES:
                time.sleep(_backoff(attempt, headers.get("retry-after")))
                continue
            return _failure(e.code, _error_message(err_body), ms_since(t0), attempt, headers)
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt <= MAX_RETRIES:
                time.sleep(_backoff(attempt, None))
                continue
            return _failure(None, f"network: {e}", ms_since(t0), attempt)

        with resp:
            headers = {k.lower(): v for k, v in resp.headers.items()}
            status = resp.status
            try:
                if "text/event-stream" in headers.get("content-type", ""):
                    parsed = _read_stream(resp, t0)
                else:
                    parsed = _read_json(resp)
            except (json.JSONDecodeError, OSError, RuntimeError) as e:
                return _failure(status, f"bad response: {e}", ms_since(t0), attempt, headers)
        elapsed_ms = ms_since(t0)
        break

    usage = parsed["usage"]
    completion_tokens = usage.get("completion_tokens")
    reasoning_tokens = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    if reasoning_tokens is None:
        reasoning_tokens = usage.get("reasoning_tokens")  # deprecated alias
    ttft_ms = parsed["ttft_ms"]
    decode_ms = elapsed_ms - ttft_ms if ttft_ms is not None else None
    toks_per_sec = None
    if completion_tokens and decode_ms and decode_ms > 0:
        toks_per_sec = round(completion_tokens / (decode_ms / 1000.0), 2)

    return {
        "ok": True,
        "status": status,
        "attempts": attempt,
        "completion_id": parsed["id"],  # key for GET /v1/signature/{chat_id}
        "model": parsed["model"] or model,
        "content": parsed["content"],
        "reasoning": parsed["reasoning"],
        "finish_reason": parsed["finish_reason"],
        "inference_id": headers.get("inference-id"),
        "request_id": headers.get("x-request-id"),
        "usage": usage,
        "metrics": {
            "latency_ms": elapsed_ms,
            "ttft_ms": ttft_ms,
            "decode_ms": decode_ms,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
            "total_tokens": usage.get("total_tokens"),
            "tokens_per_sec": toks_per_sec,
            "est_cost_usd": None,  # filled by run_prompt using model pricing
        },
    }


# ---------------------------------------------------------------- scoring


def evaluate(prompt: dict, result: dict) -> dict:
    """Score content against the prompt's `expect` block. pass=None means unscored.

    `"negate": true` inverts the check, so it fails when the pattern IS present."""
    check = _evaluate(prompt, result)
    if (prompt.get("expect") or {}).get("negate") and check["pass"] is not None and result.get("ok") \
            and (result.get("content") or "").strip():
        check = {"pass": not check["pass"], "detail": f"negated: {check['detail']}"}
    return check


def _evaluate(prompt: dict, result: dict) -> dict:
    if not result.get("ok"):
        return {"pass": None, "detail": "request failed"}
    content = result.get("content") or ""
    if not content.strip():
        return {"pass": False, "detail": f"empty content (finish_reason={result.get('finish_reason')})"}
    expect = prompt.get("expect")
    if not expect:
        return {"pass": None, "detail": "no expectation"}

    kind, value = expect.get("type"), expect.get("value")
    text = content.strip()
    if kind == "exact":
        ok = text == value
        return {"pass": ok, "detail": "exact match" if ok else f"expected {value!r}"}
    if kind == "contains":
        needles = value if isinstance(value, list) else [value]
        missing = [n for n in needles if n.lower() not in text.lower()]
        return {"pass": not missing, "detail": "contains all" if not missing else f"missing {missing}"}
    if kind == "regex":
        flags = re.IGNORECASE if "i" in (expect.get("flags") or "") else 0
        ok = re.search(value, text, flags | re.MULTILINE) is not None
        return {"pass": ok, "detail": f"regex {'matched' if ok else 'no match'}: {value}"}
    if kind == "json_valid":
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            return {"pass": False, "detail": f"invalid JSON: {e.msg}"}
        if isinstance(value, dict):
            missing = [k for k in value if parsed.get(k) != value[k]] if isinstance(parsed, dict) else list(value)
            return {"pass": not missing, "detail": "valid JSON" if not missing else f"mismatched keys {missing}"}
        return {"pass": True, "detail": "valid JSON"}
    return {"pass": None, "detail": f"unknown expect type {kind!r}"}


def detect_refusal(result: dict) -> bool | None:
    if not result.get("ok") or not (result.get("content") or "").strip():
        return None
    return REFUSAL_RE.search(result["content"][:600]) is not None


# ---------------------------------------------------------------- runs


def prompts_file() -> dict:
    return load_json(ROOT / "prompts.json")


def prompts_sha() -> str:
    return hashlib.sha256((ROOT / "prompts.json").read_bytes()).hexdigest()[:12]


def parse_config(payload: dict) -> dict:
    model = payload.get("model")
    if not model or not isinstance(model, str):
        raise ValueError("model required")
    try:
        max_tokens = int(payload.get("max_tokens") or 2048)
        temperature = float(payload.get("temperature") if payload.get("temperature") is not None else 0.2)
        concurrency = int(payload.get("concurrency") or 1)
    except (TypeError, ValueError):
        raise ValueError("max_tokens, temperature and concurrency must be numbers")
    system = payload.get("system") or None
    group = payload.get("group") or None
    if group is not None and not (isinstance(group, str) and GROUP_RE.match(group)):
        raise ValueError("group must match [A-Za-z0-9_-]{1,64}")
    try:
        group_index = int(payload.get("group_index") or 0)
    except (TypeError, ValueError):
        raise ValueError("group_index must be a number")
    return {
        "model": model,
        "max_tokens": max(16, min(max_tokens, 32768)),
        "temperature": max(0.0, min(temperature, 2.0)),
        "concurrency": max(1, min(concurrency, 16)),
        "system": system if isinstance(system, str) else None,
        "group": group,  # shared by runs of a multi-model comparison
        "group_index": group_index,  # column order within the group
        "base_url": BASE_URL,
        "prompts_version": prompts_file().get("version"),
        "prompts_sha": prompts_sha(),
    }


def run_prompt(config: dict, prompt: dict) -> dict:
    try:
        r = chat_completion(
            config["model"], prompt["text"],
            max_tokens=int(prompt.get("max_tokens") or config["max_tokens"]),
            temperature=config["temperature"],
            system=config["system"],
        )
    except Exception as e:
        r = _failure(None, str(e), None, 1)
    m = r["metrics"]
    m["est_cost_usd"] = estimate_cost(config["model"], m.get("prompt_tokens"), m.get("completion_tokens"))
    r["prompt_id"] = prompt["id"]
    r["prompt_label"] = prompt["label"]
    r["category"] = prompt["category"]
    r["check"] = evaluate(prompt, r)
    r["refused"] = detect_refusal(r)
    return r


def _percentile(values: list, pct: float):
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * pct
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 1)


def _avg(values: list, digits=1):
    return round(sum(values) / len(values), digits) if values else None


def summarize(config: dict, results: list[dict]) -> dict:
    ok = [r for r in results if r.get("ok")]

    def col(key):
        return [r["metrics"][key] for r in ok if r["metrics"].get(key) is not None]

    latencies, ttfts, tps = col("latency_ms"), col("ttft_ms"), col("tokens_per_sec")
    costs = col("est_cost_usd")
    checks = [r.get("check", {}).get("pass") for r in results]
    return {
        "model": config["model"],
        "runs": len(results),
        "ok": len(ok),
        "fail": len(results) - len(ok),
        "checks_pass": checks.count(True),
        "checks_fail": checks.count(False),
        "checks_unscored": checks.count(None),
        "refused": sum(1 for r in results if r.get("refused")),
        "retries": sum(max((r.get("attempts") or 1) - 1, 0) for r in results),
        "latency_ms_avg": _avg(latencies),
        "latency_ms_p50": _percentile(latencies, 0.5),
        "latency_ms_p90": _percentile(latencies, 0.9),
        "ttft_ms_avg": _avg(ttfts),
        "ttft_ms_p50": _percentile(ttfts, 0.5),
        "tokens_per_sec_avg": _avg(tps, 2),
        "est_cost_usd_total": round(sum(costs), 8) if costs else None,
    }


def save_run(config: dict, results: list[dict], started_at: str | None = None) -> tuple[dict, Path]:
    summary = summarize(config, results)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    slug = re.sub(r"[^A-Za-z0-9.-]+", "_", config["model"].split("/")[-1])[:48]
    out = RESULTS / f"run-{stamp}-{slug}-{secrets.token_hex(2)}.json"
    RESULTS.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "config": config,
        "started_at": started_at,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "summary": summary,
        "results": results,
    }, indent=2))
    return summary, out


def select_prompts(prompt_ids=None, categories=None) -> list[dict]:
    prompts = prompts_file()["prompts"]
    if prompt_ids:
        by_id = {p["id"]: p for p in prompts}
        return [by_id[i] for i in prompt_ids if i in by_id]
    if categories:
        return [p for p in prompts if p["category"] in categories]
    return prompts


def run_batch(config: dict, prompts: list[dict]) -> list[dict]:
    return run_matrix([config], prompts)[0]


def run_matrix(configs: list[dict], prompts: list[dict]) -> list[list[dict]]:
    """Run every prompt on every config through one pool; returns results per config in prompt order."""
    jobs = [(ci, p) for p in prompts for ci in range(len(configs))]  # interleave so models progress evenly
    with ThreadPoolExecutor(max_workers=configs[0]["concurrency"]) as pool:
        flat = list(pool.map(lambda job: run_prompt(configs[job[0]], job[1]), jobs))
    out = [[] for _ in configs]
    for (ci, _), r in zip(jobs, flat):
        out[ci].append(r)
    return out


def _cell(r: dict) -> str:
    mark = {True: "PASS", False: "FAIL", None: "----"}[r["check"]["pass"]]
    if not r["ok"]:
        mark = f"E{r.get('status') or '??'}"
    return f"{mark}{' R' if r.get('refused') else '  '}"


def diverges(row: list[dict]) -> bool:
    """True when models disagree on refusal or on a scored check for the same prompt."""
    refusals = {r.get("refused") for r in row if r.get("refused") is not None}
    checks = {r["check"]["pass"] for r in row if r["check"]["pass"] is not None}
    return len(refusals) > 1 or len(checks) > 1


def list_runs(limit=50) -> list[dict]:
    rows = []
    for f in sorted(RESULTS.glob("run-*.json"), reverse=True)[:limit]:
        try:
            data = load_json(f)
        except (json.JSONDecodeError, OSError):
            continue
        config = data.get("config") or {}
        rows.append({"file": f.name, "group": config.get("group"), "summary": data.get("summary"), "config": config})
    return rows


# ---------------------------------------------------------------- http


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"[harness] {self.address_string()} {fmt % args}")

    def _send(self, code: int, body, content_type="application/json"):
        raw = body if isinstance(body, (bytes, bytearray)) else (
            body.encode() if content_type.startswith("text/") else json.dumps(body).encode()
        )
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            return
        if path == "/api/models":
            self._send(200, {"base_url": BASE_URL, "models": model_catalog()})
            return
        if path == "/api/prompts":
            self._send(200, prompts_file())
            return
        if path == "/api/health":
            self._send(200, {"ok": True, "base_url": BASE_URL, "has_key": bool(os.environ.get("NEARAI_API_KEY"))})
            return
        if path == "/api/runs":
            self._send(200, {"runs": list_runs()})
            return
        if path.startswith("/api/runs/"):
            name = path.rsplit("/", 1)[-1]
            f = RESULTS / name
            if not RUN_FILE_RE.match(name) or not f.is_file():
                self._send(404, {"error": "run not found"})
                return
            self._send(200, f.read_bytes())
            return
        self._send(404, {"error": "not found"})

    def do_POST(self):
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
            config = parse_config(payload) if path in ("/api/run", "/api/run_one", "/api/save") else None
        except json.JSONDecodeError:
            self._send(400, {"error": "invalid json"})
            return
        except ValueError as e:
            self._send(400, {"error": str(e)})
            return

        if path == "/api/run_one":
            prompts = select_prompts([payload.get("prompt_id")])
            if not prompts:
                self._send(400, {"error": "unknown prompt_id"})
                return
            self._send(200, run_prompt(config, prompts[0]))
            return

        if path == "/api/save":
            results = payload.get("results")
            if not isinstance(results, list):
                self._send(400, {"error": "results list required"})
                return
            summary, out = save_run(config, results, payload.get("started_at"))
            self._send(200, {"summary": summary, "saved": out.name})
            return

        if path == "/api/run":
            started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            selected = select_prompts(payload.get("prompt_ids")) or select_prompts()
            results = run_batch(config, selected)
            summary, out = save_run(config, results, started_at)
            self._send(200, {"summary": summary, "results": results, "saved": out.name})
            return

        self._send(404, {"error": "not found"})


def serve():
    RESULTS.mkdir(exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"NEAR AI harness → http://127.0.0.1:{PORT}")
    print(f"Base URL: {BASE_URL}")
    if not os.environ.get("NEARAI_API_KEY"):
        print("Set NEARAI_API_KEY before clicking Run.")
    server.serve_forever()


def cli_run(args) -> int:
    group = f"cmp-{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}" if len(args.model) > 1 else None
    configs = [parse_config({
        "model": model,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "concurrency": args.concurrency,
        "system": args.system,
        "group": group,
        "group_index": i,
    }) for i, model in enumerate(args.model)]
    prompts = select_prompts(
        args.prompts.split(",") if args.prompts else None,
        args.category.split(",") if args.category else None,
    )
    if not prompts:
        print("No prompts matched.", file=sys.stderr)
        return 2
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    per_model = run_matrix(configs, prompts)
    saved = [save_run(c, rs, started_at) for c, rs in zip(configs, per_model)]
    summaries = [s for s, _ in saved]

    if args.json:
        print(json.dumps({"group": group, "runs": [{"summary": s, "saved": str(out)} for s, out in saved]}, indent=2))
    elif len(configs) == 1:
        for r in per_model[0]:
            m, check = r["metrics"], r["check"]
            mark = {True: "PASS", False: "FAIL", None: "----"}[check["pass"]]
            status = "ok" if r["ok"] else f"ERR {r.get('status')}"
            print(f"{mark}  {r['prompt_id']:<20} {status:<8} ttft={m['ttft_ms']}ms "
                  f"lat={m['latency_ms']}ms tok/s={m['tokens_per_sec']}  {check['detail']}")
        s, out = saved[0]
        print(f"\n{s['ok']}/{s['runs']} ok · checks {s['checks_pass']} pass "
              f"{s['checks_fail']} fail · refused {s['refused']} · saved {out}")
    else:
        names = [f"m{i + 1}" for i in range(len(configs))]
        for name, c in zip(names, configs):
            print(f"{name} = {c['model']}")
        print(f"\n{'prompt':<20} " + " ".join(f"{n:<8}" for n in names))
        for pi, p in enumerate(prompts):
            row = [rs[pi] for rs in per_model]
            flag = "  <- differs" if diverges(row) else ""
            print(f"{p['id']:<20} " + " ".join(f"{_cell(r):<8}" for r in row) + flag)
        print("\nR = likely refusal")
        print(f"\n{'model':<6} {'pass':>4} {'fail':>4} {'refused':>7} {'ttft':>7} {'p50':>7} {'tok/s':>7} {'cost':>10}")
        for name, s in zip(names, summaries):
            cost = f"${s['est_cost_usd_total']:.6f}" if s["est_cost_usd_total"] is not None else "—"
            print(f"{name:<6} {s['checks_pass']:>4} {s['checks_fail']:>4} {s['refused']:>7} "
                  f"{s['ttft_ms_avg'] or '—':>7} {s['latency_ms_p50'] or '—':>7} {s['tokens_per_sec_avg'] or '—':>7} {cost:>10}")
        print(f"\ngroup {group} · saved {len(saved)} runs to {RESULTS}")
    failed = any(s["fail"] or s["checks_fail"] for s in summaries)
    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd")
    run = sub.add_parser("run", help="run prompts headless and exit non-zero on failure")
    run.add_argument("--model", required=True, action="append", help="repeat to compare models side by side")
    run.add_argument("--prompts", help="comma-separated prompt ids")
    run.add_argument("--category", help="comma-separated categories (e.g. smoke)")
    run.add_argument("--max-tokens", type=int, default=2048)
    run.add_argument("--temperature", type=float, default=0.2)
    run.add_argument("--concurrency", type=int, default=4)
    run.add_argument("--system")
    run.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.cmd == "run":
        sys.exit(cli_run(args))
    serve()


if __name__ == "__main__":
    main()
