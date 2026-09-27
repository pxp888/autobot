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
CONFIG_PATH = Path(os.environ.get("CONFIG_PATH", str(Path(__file__).parent / "providers.json")))
LOG_PATH = Path(os.environ.get("KEV_LOG_PATH", str(Path(__file__).parent / "log.txt")))
# ponytail: ~4 chars/token heuristic instead of a real tokenizer. Routing looks at the LAST
# ~500 tokens — what's being asked now, not how the conversation started.
STATE_CHARS = int(os.environ.get("STATE_CHARS", "2000"))  # was 4000
KEV_TIMEOUT = float(os.environ.get("KEV_TIMEOUT_SECS", "15"))
# ponytail: one global read timeout for upstreams (code-gen can be slow); per-provider override later if needed
UPSTREAM_TIMEOUT = httpx.Timeout(float(os.environ.get("TIMEOUT_SECS", "300")), connect=15)

app = FastAPI(title="autobot")
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")
route_log: deque[dict] = deque(maxlen=200)
model_token_counts: dict[str, int] = {}  # in-memory: model_key -> cumulative prompt tokens since startup


def count_tokens(text: str) -> int:
    """ponytail: ~4 characters per token, matching the STATE_CHARS heuristic."""
    return max(1, len(text) // 4)

# ---------- config ----------

def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text())


def kev_url(cfg: dict) -> str:
    """Effective Kev endpoint: file setting wins, KEV_URL env is the default/fallback."""
    return (cfg.get("kevUrl") or KEV_URL).rstrip("/")


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


async def kev_question(client: httpx.AsyncClient, state: str, criteria: dict, url: str) -> dict:
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
            url = kev_url(cfg)
            if len(criteria) > 1:
                answers = await asyncio.gather(
                    kev_question(client, state, criteria, url),
                    kev_question(client, state, dict(reversed(list(criteria.items()))), url),
                )
            else:
                answers = [await kev_question(client, state, criteria, url)]
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


async def forward(cfg: dict, body: dict, cand: tuple):
    _key, pname, model, prov = cand
    url, headers = upstream_url(prov), upstream_headers(prov)
    fwd_body = {**body, "model": model["id"]}

    if not body.get("stream"):
        async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
            r = await client.post(url, json=fwd_body, headers=headers)
        try:
            return JSONResponse(content=r.json(), status_code=r.status_code)
        except ValueError:
            return PlainTextResponse(r.text, status_code=r.status_code,
                                     media_type=r.headers.get("content-type", "text/plain"))

    async def gen():
        async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
            async with client.stream("POST", url, json=fwd_body, headers=headers) as r:
                if r.status_code != 200:
                    err = {"error": (await r.aread()).decode(errors="replace")}
                    yield f"data: {json.dumps(err)}\n\n"
                    return
                async for chunk in r.aiter_bytes():
                    yield chunk

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

    # count prompt tokens for the model that was picked
    key = cand[0]
    model_token_counts[key] = model_token_counts.get(key, 0) + count_tokens(state)

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
            kev_up = (await client.get(f"{kev_url(cfg)}/v1/models")).status_code == 200
    except Exception:
        pass
    return {"ok": True, "kev_url": kev_url(cfg), "kev_up": kev_up, "config_path": str(CONFIG_PATH),
            "models": len(candidates(cfg))}


@app.get("/api/config")
async def get_config():
    cfg = load_config()
    # form shows the effective address; saving persists it into the file (env becomes fallback only)
    cfg["kevUrl"] = cfg.get("kevUrl") or KEV_URL
    return cfg


@app.post("/api/config")
async def save_config(request: Request):
    try:
        cfg = await request.json()
        validate_config(cfg)
    except (ValueError, json.JSONDecodeError) as e:
        return JSONResponse({"error": f"invalid config: {e}"}, status_code=400)
    tmp = CONFIG_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(cfg, indent=2))  # atomic-ish swap so readers never see a half file
    tmp.replace(CONFIG_PATH)
    return cfg


@app.get("/api/token-counts")
async def get_token_counts():
    """In-memory cumulative prompt tokens per model since startup."""
    return {"counts": dict(model_token_counts), "total": sum(model_token_counts.values())}


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

    # config round-trips through the file path
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "c.json"; p.write_text(json.dumps(cfg))
        global CONFIG_PATH
        old, CONFIG_PATH = CONFIG_PATH, p
        try:
            assert load_config()["providers"]["p1"]["models"][0]["id"] == "a"
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

    print("selftest ok")


if __name__ == "__main__":
    if "--selftest" in os.sys.argv[1:]:
        selftest()
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
