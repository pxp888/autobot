#!/usr/bin/env python3
"""autobot — an OpenAI-compatible inference gateway that routes each request
to the best provider/model using a Kev (System One) decision endpoint.

Run:  .venv/bin/python autobot.py            (PORT env, default 8000)
Test: .venv/bin/python autobot.py --selftest
"""
import asyncio
import json
import os
import time
from collections import deque
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

KEV_URL = os.environ.get("KEV_URL", "http://192.168.0.189:8009").rstrip("/")
KEV_API_KEY = os.environ.get("KEV_API_KEY", "")  # bearer key for keyed Kev servers; empty = open local server
CONFIG_PATH = Path(os.environ.get("CONFIG_PATH", str(Path(__file__).parent / "providers.json")))
LOG_PATH = Path(os.environ.get("KEV_LOG_PATH", str(Path(__file__).parent / "log.txt")))
# ponytail: ~4 chars/token heuristic instead of a real tokenizer. Routing looks at the LAST
# ~500 tokens — what's being asked now, not how the conversation started.
STATE_CHARS = int(os.environ.get("STATE_CHARS", "3000"))  # was 4000, then 2000
KEV_TIMEOUT = float(os.environ.get("KEV_TIMEOUT_SECS", "15"))
# ponytail: one global read timeout for upstreams (code-gen can be slow); per-provider override later if needed
UPSTREAM_TIMEOUT = httpx.Timeout(float(os.environ.get("TIMEOUT_SECS", "300")), connect=15)
MASK = "******"  # apiKey placeholder in /api/config responses; posting it back means "unchanged"

app = FastAPI(title="autobot")
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")
route_log: deque[dict] = deque(maxlen=200)
token_counts = {
    "prompt": {},       # model_key -> cumulative prompt tokens since startup
    "completion": {},   # model_key -> cumulative completion tokens since startup
    "requests": {},     # model_key -> cumulative request count since startup
}


def count_tokens(text: str) -> int:
    """ponytail: ~4 characters per token, matching the STATE_CHARS heuristic."""
    return max(1, len(text) // 4)

# ---------- config ----------

def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text())


def kev_url(cfg: dict) -> str:
    """Effective Kev endpoint: file setting wins, KEV_URL env is the default/fallback."""
    return (cfg.get("kevUrl") or KEV_URL).rstrip("/")


def kev_key(cfg: dict) -> str:
    """Effective Kev bearer key: file setting wins, KEV_API_KEY env is the default/fallback."""
    return cfg.get("kevApiKey") or KEV_API_KEY


def kev_headers(cfg: dict) -> dict:
    """Keyed servers (started with KEV_API_KEY) require Bearer auth on every /v1/* route; open ones ignore it."""
    key = kev_key(cfg)
    return {"authorization": f"Bearer {key}"} if key else {}


def resolve_api_key(posted, old):
    """MASK/empty post means 'unchanged'. Returns the value to persist, or None to drop (placeholder)."""
    if posted and posted != MASK:
        return posted
    if old and old != MASK:
        return old
    return None


def validate_config(cfg) -> None:
    if not isinstance(cfg, dict) or not isinstance(cfg.get("providers"), dict):
        raise ValueError('config must be an object with a "providers" object')
    for name, prov in cfg["providers"].items():
        if not isinstance(prov, dict) or not prov.get("baseUrl") or not isinstance(prov.get("models"), list):
            raise ValueError(f'provider "{name}" needs baseUrl and models[]')
        for m in prov["models"]:
            if not isinstance(m, dict) or not m.get("id"):
                raise ValueError(f'provider "{name}": every model needs an id')
            w = m.get("weight")
            if w is not None:
                if isinstance(w, bool) or not isinstance(w, (int, float)):
                    try:  # accept numeric strings from HTML forms
                        float(w)
                    except (TypeError, ValueError):
                        raise ValueError(f'provider "{name}"/{m["id"]}: weight must be a number')


# ---------- routing decision ----------

def flatten_content(c) -> str:
    """Message content may be a string or a list of typed parts."""
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join("" if p is None else (p.get("text", "") if isinstance(p, dict) else str(p)) for p in c)
    return str(c)


def build_state(messages) -> str:
    # ponytail: drop system messages — harness boilerplate (tools, rules, "think carefully")
    # made Kev route even a bare "hello" to the smart model (~3x prob, measured). Routing
    # decides on the conversation itself; the tail keeps state bounded.
    parts = [f"{m.get('role', 'user')}:\n{flatten_content(m.get('content'))}"
             for m in messages if m.get("role") != "system"]
    return ("\n\n".join(parts))[-STATE_CHARS:]  # tail: recent turns decide routing


def candidates(cfg) -> list[tuple[str, str, dict, dict]]:
    """[(option_key, provider_name, model_dict, provider_dict), ...]; keys unique even if ids repeat."""
    out, seen = [], set()
    for pname, prov in cfg.get("providers", {}).items():
        for m in prov.get("models", []):
            key = f"{pname}:{m['id']}" if m["id"] in seen else m["id"]
            seen.add(m["id"])
            out.append((key, pname, m, prov))
    return out


def model_weight(model: dict) -> float:
    """Per-model multiplier on routing probability (steering knob). 1 = neutral."""
    w = model.get("weight", 1)
    if isinstance(w, bool):
        return 1.0
    try:
        v = float(w)
        return v if not (v != v) else 1.0  # NaN guard
    except (TypeError, ValueError):
        return 1.0


def describe(pname: str, model: dict, prov: dict) -> str:
    bits = [model.get("description") or "(no description)", f"provider {pname}"]
    if prov.get("description"):
        bits.append(prov["description"])
    return "; ".join(bits)


def average_probs(answers: list[dict]) -> dict:
    """Per-option mean over multiple Kev passes."""
    probs = {}
    n = len(answers) or 1
    for ans in answers:
        for k, v in (ans.get("probabilities") or {}).items():
            probs[k] = probs.get(k, 0.0) + float(v) / n
    return probs


async def kev_question(client: httpx.AsyncClient, state: str, criteria: dict, url: str, headers=None) -> dict:
    r = await client.post(
        f"{url}/v1/systemone",
        json={
            "state": state,
            "questions": {
                "model": {
                    "type": "choice",
                    "instructions": "Which model is best suited for this request?",
                    "criteria": criteria,
                }
            },
        },
        headers=headers or {},
    )
    r.raise_for_status()
    return r.json()["answers"]["model"]


async def decide(cfg: dict, body: dict, state: str) -> tuple[dict, bool]:
    """Pick (candidate, was_fallback). Skips Kev when the client named a known model."""
    cands = candidates(cfg)
    if not cands:
        raise ValueError("no models configured")
    by_key = {c[0]: c for c in cands}
    req_model = body.get("model")
    if req_model and req_model in by_key:
        cand = by_key[req_model]
        log_route(req_model, state, {}, "explicit", req_model, None, False,
                  {"provider": cand[1], "model": cand[2]["id"]})  # pinned requests were invisible before
        return cand, False  # explicit model wins, no routing

    criteria = {k: describe(pname, m, prov) for k, pname, m, prov in cands}
    fallback_key = next((k for k, _, m, _ in cands if m["id"] == cfg.get("defaultModel")), cands[0][0])
    try:
        async with httpx.AsyncClient(timeout=KEV_TIMEOUT) as client:
            # ponytail: this Kev checkpoint has a strong last-option position bias (verified via /permute:
            # argmax flips between orders). Two concurrent passes — normal + reversed order — and averaging
            # the probabilities cancels it; wall time stays ~one pass.
            url, headers = kev_url(cfg), kev_headers(cfg)
            if len(criteria) > 1:
                answers = await asyncio.gather(
                    kev_question(client, state, criteria, url, headers),
                    kev_question(client, state, dict(reversed(list(criteria.items()))), url, headers),
                )
            else:
                answers = [await kev_question(client, state, criteria, url, headers)]
        probs = average_probs(list(answers))
        # apply per-model weight (steering knob) before picking argmax
        effective = {k: probs[k] * model_weight(by_key.get(k, (None,))[2]) for k in probs}
        key = max(effective, key=effective.get) if effective else fallback_key
        bad_answer = key not in by_key  # keys are ours, so this only fires on a malformed answer
        cand = by_key[key]
        print(f"[autobot] picked {cand[2]['id']} from {cand[1]}")
        log_route(req_model, state, probs, " / ".join(a.get("choice", "?") for a in answers),
                  key, None, bad_answer, {"provider": cand[1], "model": cand[2]["id"]},
                  effectives=effective)
        return cand, bad_answer
    except Exception as e:  # kev down / bad answer -> default model keeps the gateway alive
        cand = by_key[fallback_key]
        print(f"[autobot] fallback to {cand[2]['id']} from {cand[1]}: {e}")
        log_route(req_model, state, {}, f"fallback: {e.__class__.__name__}: {e}", fallback_key, None, True,
                  {"provider": cand[1], "model": cand[2]["id"]})
        return cand, True


def log_route(requested, state, probs, choice, picked, timing, fell_back, sent=None, effectives=None):
    entry = {
        "ts": time.strftime("%H:%M:%S"),
        "requested": requested or "(auto)",
        "picked": picked,
        "kev_choice": choice,
        "probabilities": {k: round(v, 3) for k, v in probs.items()},
        "timing": timing,
        "fallback": fell_back,
        "preview": state[:160].replace("\n", " "),
        "tokens": count_tokens(state),
    }
    if effectives is not None:
        entry["effective"] = {k: round(v, 3) for k, v in effectives.items()}
    if sent is not None:
        entry["sent"] = sent  # exactly what forward() POSTs upstream: {provider, model}
    route_log.appendleft(entry)
    with LOG_PATH.open("a") as f:  # ponytail: one JSON line per decision, append-only; rotate if it ever gets big
        f.write(json.dumps(entry) + "\n")


# ---------- upstream forwarding ----------

def upstream_url(prov: dict) -> str:
    return prov["baseUrl"].rstrip("/") + "/chat/completions"


def upstream_headers(prov: dict) -> dict:
    h = {"content-type": "application/json"}
    if prov.get("apiKey"):
        h["authorization"] = f"Bearer {prov['apiKey']}"
    return h


def sse_usage(buf: bytes) -> tuple[bytes, dict | None]:
    """ponytail: pull complete SSE data lines out of a rolling buffer; returns (leftover_tail, usage|None).
    A usage event can be split across TCP reads and share one read with [DONE], so only whole
    lines are parsed — the old per-chunk regex never produced valid JSON (it captured the key)."""
    usage = None
    while b"\n" in buf:
        line, buf = buf.split(b"\n", 1)
        s = line.lstrip()
        if not s.startswith(b"data:"):
            continue
        payload = s[5:].lstrip()
        if not payload or payload == b"[DONE]":
            continue
        try:
            ev = json.loads(payload)
        except ValueError:
            continue
        u = ev.get("usage") if isinstance(ev, dict) else None
        if isinstance(u, dict):
            usage = u
    return buf, usage


async def forward(cfg: dict, body: dict, cand: tuple):
    _key, pname, model, prov = cand
    key = _key
    url, headers = upstream_url(prov), upstream_headers(prov)
    fwd_body = {**body, "model": model["id"]}
    if fwd_body.get("stream"):
        # ponytail: OpenAI-compatible streams only emit a final usage chunk when asked; without
        # this most providers send none and stream token counts stay 0. setdefault respects clients who opt out.
        fwd_body.setdefault("stream_options", {"include_usage": True})

    def count_usage(resp: dict):
        usage = resp.get("usage", {})
        prompt = usage.get("prompt_tokens", usage.get("input_tokens", 0))
        completion = usage.get("completion_tokens", usage.get("output_tokens", 0))
        if prompt or completion:
            token_counts["prompt"][key] = token_counts["prompt"].get(key, 0) + prompt
            token_counts["completion"][key] = token_counts["completion"].get(key, 0) + completion
        token_counts["requests"][key] = token_counts["requests"].get(key, 0) + 1

    if not body.get("stream"):
        async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
            r = await client.post(url, json=fwd_body, headers=headers)
        try:
            data = r.json()
            count_usage(data)
            return JSONResponse(content=data, status_code=r.status_code)
        except ValueError:
            # response is not JSON, log raw for debugging
            print(f"[autobot] non-JSON response from {key}: {r.text[:500]}")
            return PlainTextResponse(r.text, status_code=r.status_code,
                                     media_type=r.headers.get("content-type", "text/plain"))

    async def gen():
        async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
            async with client.stream("POST", url, json=fwd_body, headers=headers) as r:
                if r.status_code != 200:
                    err = {"error": (await r.aread()).decode(errors="replace")}
                    token_counts["requests"][key] = token_counts["requests"].get(key, 0) + 1
                    yield f"data: {json.dumps(err)}\n\n"
                    return
                buf = b""
                usage = None
                async for chunk in r.aiter_bytes():
                    yield chunk
                    # parse SSE from a rolling buffer; keep the last usage event seen
                    buf, u = sse_usage(buf + chunk)
                    if u is not None:
                        usage = u
                if usage is not None:
                    count_usage({"usage": usage})
                else:
                    # provider sent no usage (stream_options unsupported?) — count the request only
                    token_counts["requests"][key] = token_counts["requests"].get(key, 0) + 1

    return StreamingResponse(gen(), media_type="text/event-stream")


# ---------- API ----------

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    if not isinstance(body.get("messages"), list) or not body["messages"]:
        return JSONResponse({"error": "messages[] is required"}, status_code=400)
    cfg = load_config()  # ponytail: re-read file per request; live config edits, no reload logic
    state = build_state(body["messages"])
    try:
        cand, _fell_back = await decide(cfg, body, state)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=503)

    return await forward(cfg, body, cand)


@app.get("/v1/models")
async def list_models():
    cfg = load_config()
    # 'auto' is advertised so harnesses that only pick from /v1/models can opt into Kev routing.
    # The second id embeds a capability keyword ON PURPOSE: Odysseus sniffs native tool-calling
    # support from the model string (agent_loop.py _model_supports_tools); "auto" matches nothing,
    # so it fell back to fenced-block prompt mode for every routed turn. Both ids route identically
    # through Kev (any unknown id is a routing request). Pick qwen3-auto in tool-using harnesses;
    # if you ever route a model that lacks function calling, pin it explicitly instead.
    data = [
        {"id": "qwen3-auto", "object": "model", "created": 0, "owned_by": "autobot (kev-routed, native tools)"},
        {"id": "auto", "object": "model", "created": 0, "owned_by": "autobot (kev-routed)"},
    ]
    data += [{"id": m["id"], "object": "model", "created": 0, "owned_by": pname}
             for _key, pname, m, _prov in candidates(cfg)]
    return {"object": "list", "data": data}


@app.get("/health")
async def health():
    cfg = load_config()
    kev_up = False
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            kev_up = (await client.get(f"{kev_url(cfg)}/v1/models", headers=kev_headers(cfg))).status_code == 200
    except Exception:
        pass
    return {"ok": True, "kev_url": kev_url(cfg), "kev_up": kev_up, "kev_key": bool(kev_key(cfg)),
            "config_path": str(CONFIG_PATH), "models": len(candidates(cfg))}


@app.get("/api/config")
async def get_config():
    cfg = load_config()
    # mask api keys in the returned config so the GUI never receives real values
    for p in cfg.get("providers", {}).values():
        if p.get("apiKey"):
            p["apiKey"] = MASK
    cfg["kevUrl"] = cfg.get("kevUrl") or KEV_URL
    if kev_key(cfg):  # never hand the real key to the GUI (env-sourced ones included)
        cfg["kevApiKey"] = MASK
    return cfg


@app.post("/api/config")
async def save_config(request: Request):
    try:
        cfg = await request.json()
        validate_config(cfg)
    except (ValueError, json.JSONDecodeError) as e:
        return JSONResponse({"error": f"invalid config: {e}"}, status_code=400)
    # masked/empty apiKey means "unchanged": keep the on-disk value, otherwise every GUI save
    # (the form always posts the placeholder back) would wipe real keys from providers.json
    old_cfg = load_config()
    old_provs = old_cfg.get("providers") or {}
    for name, p in cfg.get("providers", {}).items():
        key = resolve_api_key(p.pop("apiKey", None), (old_provs.get(name) or {}).get("apiKey"))
        if key is not None:
            p["apiKey"] = key
    kev = resolve_api_key(cfg.pop("kevApiKey", None), old_cfg.get("kevApiKey"))
    if kev is not None:
        cfg["kevApiKey"] = kev
    tmp = CONFIG_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(cfg, indent=2))  # atomic-ish swap so readers never see a half file
    tmp.replace(CONFIG_PATH)
    for p in cfg.get("providers", {}).values():  # never echo real keys back to the caller
        if p.get("apiKey"):
            p["apiKey"] = MASK
    if cfg.get("kevApiKey"):
        cfg["kevApiKey"] = MASK
    return cfg


@app.get("/api/token-counts")
async def get_token_counts():
    """In-memory cumulative prompt, completion tokens, and request counts per model since startup."""
    prompt_counts = dict(token_counts["prompt"])
    completion_counts = dict(token_counts["completion"])
    req_counts = dict(token_counts["requests"])
    # ensure all counted models appear in all maps
    for k in req_counts:
        prompt_counts.setdefault(k, 0)
        completion_counts.setdefault(k, 0)
    req_total = sum(req_counts.values())
    prompt_total = sum(prompt_counts.values())
    completion_total = sum(completion_counts.values())
    return {"prompt_counts": prompt_counts, "completion_counts": completion_counts, "total_prompt_tokens": prompt_total, "total_completion_tokens": completion_total, "requests": req_counts, "total_requests": req_total}


@app.get("/api/routes")
async def get_routes():
    return list(route_log)


# ---------- web UI (controls: edit the config) ----------

@app.get("/")
async def index():
    html = (Path(__file__).parent / "static" / "index.html").read_text()
    return PlainTextResponse(html.replace("{cfg_path}", str(CONFIG_PATH)), media_type="text/html")


# ---------- selftest: no server or kev needed ----------

def selftest() -> None:
    import tempfile

    cfg = {"defaultModel": "b", "providers": {
        "p1": {"baseUrl": "http://x/v1", "models": [{"id": "a", "description": "smart"}, {"id": "b"}]},
        "p2": {"baseUrl": "http://y", "apiKey": "k", "models": [{"id": "c", "description": "fast"}]}}}

    # state building: system boilerplate dropped (harness prompts skewed routing), long input truncated
    assert build_state([{"role": "system", "content": "boilerplate"}, {"role": "user", "content": "hello"}]) == "user:\nhello"
    st = build_state([{"role": "assistant", "content": ["a", None, 3]}])
    assert "assistant:\na3" in st, st
    long1 = build_state([{"role": "user", "content": "HARD C++ QUESTION\n" + "z" * 5000},
                         {"role": "assistant", "content": "ok"},
                         {"role": "user", "content": "rename foo to bar"}])
    assert len(long1) == STATE_CHARS and long1.endswith("rename foo to bar") and "HARD" not in long1, len(long1)
    assert build_state([{"role": "user", "content": "x" * (STATE_CHARS + 500)}]) == "x" * STATE_CHARS

    # multi-pass averaging: the bias-cancellation merge picks the true argmax of the mean
    merged = average_probs([{"probabilities": {"a": 0.3, "b": 0.7}},
                            {"probabilities": {"a": 0.8, "b": 0.2}}])
    assert max(merged, key=merged.get) == "a" and abs(merged["a"] - 0.55) < 1e-9, merged

    # candidate dedup: same id on two providers -> one gets a qualified key
    dup = {"providers": {
        "p1": {"baseUrl": "u", "models": [{"id": "m"}]},
        "p2": {"baseUrl": "v", "models": [{"id": "m"}, {"id": "n"}]}}}
    assert len(candidates(dup)) == 3

    # kev_question posts to the given url (regression: it read a non-existent global cfg ->
    # NameError swallowed into fallback on every request)
    try:
        asyncio.run(kev_question(httpx.AsyncClient(timeout=1), "s", {"a": "d"}, "http://127.0.0.1:9"))
        assert False, "should fail to connect"
    except httpx.ConnectError:
        pass

    # validation rejects garbage
    for bad in [{"providers": None}, {"providers": {"p": {}}}, {"providers": {"p": {"baseUrl": "u", "models": [{}]}}}]:
        try:
            validate_config(bad); assert False, f"should reject {bad}"
        except ValueError:
            pass

    # kev address: file setting wins (trailing slash stripped), env is the default
    assert kev_url({"kevUrl": "http://x:1/"}) == "http://x:1" and kev_url({}) == KEV_URL

    # kev api key: file setting wins over env; header only sent when a key is effective (keyed servers 401 without it)
    global KEV_API_KEY
    saved_env_key, KEV_API_KEY = KEV_API_KEY, ""
    try:
        assert kev_headers({}) == {} and kev_headers({"kevApiKey": "file"}) == {"authorization": "Bearer file"}
        seen = {}
        def handler(req):
            seen["auth"] = req.headers.get("authorization")
            return httpx.Response(200, json={"answers": {"model": {"choice": "a", "probabilities": {"a": 1.0}}}})
        async def _q():
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
                return await kev_question(c, "s", {"a": "d"}, "http://k/v1", kev_headers({"kevApiKey": "sekrit"}))
        ans = asyncio.run(_q())
        assert seen["auth"] == "Bearer sekrit" and ans["probabilities"] == {"a": 1.0}, (seen, ans)
    finally:
        KEV_API_KEY = saved_env_key

    # masked/empty post means unchanged; placeholders never persist to disk
    assert resolve_api_key(MASK, "real") == "real" and resolve_api_key("", "real") == "real"
    assert resolve_api_key("new", "real") == "new"
    assert resolve_api_key(MASK, None) is None and resolve_api_key(MASK, MASK) is None and resolve_api_key(None, None) is None

    # config round-trips through the file path
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "c.json"; p.write_text(json.dumps(cfg))
        global CONFIG_PATH
        old, CONFIG_PATH = CONFIG_PATH, p
        try:
            assert load_config()["providers"]["p1"]["models"][0]["id"] == "a"
        finally:
            CONFIG_PATH = old

    # regression: the GUI posts apiKeys back masked; saving must not clobber real keys on disk
    async def post_cfg(body):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            r = await c.post("/api/config", json=body)
            if r.status_code != 200:
                raise AssertionError((r.status_code, r.text))
            return r.json()

    with tempfile.TemporaryDirectory() as d3:
        p3 = Path(d3) / "c.json"
        p3.write_text(json.dumps({"kevApiKey": "disk-kev", "providers": {
            "p1": {"baseUrl": "u", "apiKey": "real-key", "models": [{"id": "a"}]},
            # p2 simulates the corrupted file: placeholder persisted on disk is not a real key
            "p2": {"baseUrl": "v", "apiKey": MASK, "models": [{"id": "b"}]}}}))
        old, CONFIG_PATH = CONFIG_PATH, p3
        try:
            got = asyncio.run(post_cfg({"kevApiKey": MASK, "providers": {
                # masked key -> unchanged; missing key stays missing
                "p1": {"baseUrl": "u", "apiKey": MASK, "models": [{"id": "a"}]},
                "p2": {"baseUrl": "v", "models": [{"id": "b"}]}}}))
            assert got["providers"]["p1"]["apiKey"] == MASK and "apiKey" not in got["providers"]["p2"], got
            assert got.get("kevApiKey") == MASK  # response masks it, like provider keys
            disk = json.loads(p3.read_text())
            assert disk["providers"].get("p2", {}).get("apiKey") is None  # placeholder dropped, not kept
            assert disk["providers"]["p1"]["apiKey"] == "real-key", disk  # the reported bug: mask persisted over real key
            assert disk["kevApiKey"] == "disk-kev", disk  # masked kev post keeps the on-disk key
            asyncio.run(post_cfg({"kevApiKey": "new-kev", "providers": {"p1": {"baseUrl": "u", "apiKey": "new-key", "models": [{"id": "a"}]}}}))
            d2 = json.loads(p3.read_text())
            assert d2["providers"]["p1"]["apiKey"] == "new-key" and d2["kevApiKey"] == "new-kev"  # explicit keys still apply
        finally:
            CONFIG_PATH = old

    # decisions land in log.txt (one JSON line each), including the fallback path
    global LOG_PATH
    with tempfile.TemporaryDirectory() as d2:
        old, LOG_PATH = LOG_PATH, Path(d2) / "log.txt"
        try:
            log_route(None, "user:\nhello", {"a": 0.5}, "a", None, None, False)
            assert json.loads(LOG_PATH.read_text())["kev_choice"] == "a"

            # explicit model bypasses routing (decide is async; check the fast path's contract)
            got = asyncio.run(decide(cfg, {"model": "c"}, "anything"))
            assert got[0][2]["id"] == "c" and got[0][3].get("apiKey") == "k", got

            # weight: multiplier on routing probability; default 1.0 for no-op models
            assert model_weight({}) == 1.0
            assert model_weight({"weight": 2.0}) == 2.0
            assert model_weight({"weight": "garbage"}) == 1.0
            assert model_weight({"weight": True}) == 1.0
            last = json.loads(LOG_PATH.read_text().splitlines()[-1])
            assert last["requested"] == "c" and last["picked"] == "c" and not last["fallback"], last
            assert last["sent"] == {"provider": "p2", "model": "c"}, last
        finally:
            LOG_PATH = old

    # streaming usage extraction (regression: old per-chunk regex captured '"usage": {...}', which
    # is not JSON — stream token counts were always 0)
    tail, u = sse_usage(b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: {"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":3}}\n\ndata: [DONE]\n')
    assert tail == b"" and u == {"prompt_tokens": 7, "completion_tokens": 3}, (tail, u)
    # usage event split across two reads: nothing until the line completes
    t1, u1 = sse_usage(b'data: {"choices":[],"us')
    assert u1 is None and t1 == b'data: {"choices":[],"us', (t1, u1)
    t2, u2 = sse_usage(t1 + b'age":{"prompt_tokens":7,"completion_tokens":3}}\n\ndata: [DONE]\n')
    assert u2 == {"prompt_tokens": 7, "completion_tokens": 3} and t2 == b"", (t2, u2)

    print("selftest ok")


if __name__ == "__main__":
    if "--selftest" in os.sys.argv[1:]:
        selftest()
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
