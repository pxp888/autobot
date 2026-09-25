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
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse

KEV_URL = os.environ.get("KEV_URL", "http://192.168.0.189:8009").rstrip("/")
CONFIG_PATH = Path(os.environ.get("CONFIG_PATH", str(Path(__file__).parent / "providers.json")))
# ponytail: ~4 chars/token heuristic instead of a real tokenizer. Routing looks at the LAST
# ~1000 tokens — what's being asked now, not how the conversation started.
STATE_CHARS = 4000
KEV_TIMEOUT = float(os.environ.get("KEV_TIMEOUT_SECS", "15"))
# ponytail: one global read timeout for upstreams (code-gen can be slow); per-provider override later if needed
UPSTREAM_TIMEOUT = httpx.Timeout(float(os.environ.get("TIMEOUT_SECS", "300")), connect=15)

app = FastAPI(title="autobot")
route_log: deque[dict] = deque(maxlen=200)


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


async def kev_question(client: httpx.AsyncClient, state: str, criteria: dict) -> dict:
    r = await client.post(
        f"{kev_url(cfg)}/v1/systemone",
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
        return by_key[req_model], False  # explicit model wins, no routing

    criteria = {k: describe(pname, m, prov) for k, pname, m, prov in cands}
    fallback_key = next((k for k, _, m, _ in cands if m["id"] == cfg.get("defaultModel")), cands[0][0])
    try:
        async with httpx.AsyncClient(timeout=KEV_TIMEOUT) as client:
            # ponytail: this Kev checkpoint has a strong last-option position bias (verified via /permute:
            # argmax flips between orders). Two concurrent passes — normal + reversed order — and averaging
            # the probabilities cancels it; wall time stays ~one pass.
            if len(criteria) > 1:
                answers = await asyncio.gather(
                    kev_question(client, state, criteria),
                    kev_question(client, state, dict(reversed(list(criteria.items())))),
                )
            else:
                answers = [await kev_question(client, state, criteria)]
        probs = average_probs(list(answers))
        key = max(probs, key=probs.get) if probs else fallback_key
        bad_answer = key not in by_key  # keys are ours, so this only fires on a malformed answer
        log_route(req_model, state, probs, " / ".join(a.get("choice", "?") for a in answers),
                  key, None, bad_answer)
        return by_key[key], bad_answer
    except Exception as e:  # kev down / bad answer -> default model keeps the gateway alive
        log_route(req_model, state, {}, f"fallback: {e.__class__.__name__}: {e}", fallback_key, None, True)
        return by_key[fallback_key], True


def log_route(requested, state, probs, choice, picked, timing, fell_back):
    route_log.appendleft({
        "ts": time.strftime("%H:%M:%S"),
        "requested": requested or "(auto)",
        "picked": picked,
        "kev_choice": choice,
        "probabilities": {k: round(v, 3) for k, v in probs.items()},
        "timing": timing,
        "fallback": fell_back,
        "preview": state[:160].replace("\n", " "),
    })


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
    try:
        cand, _fell_back = await decide(cfg, body, build_state(body["messages"]))
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=503)
    return await forward(cfg, body, cand)


@app.get("/v1/models")
async def list_models():
    cfg = load_config()
    # 'auto' is advertised so harnesses that only pick from /v1/models can opt into Kev routing
    data = [{"id": "auto", "object": "model", "created": 0, "owned_by": "autobot (kev-routed)"}]
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


@app.get("/api/routes")
async def get_routes():
    return list(route_log)


# ---------- web UI (controls: edit the config) ----------

UI = """<!doctype html>
<html><head><meta charset="utf-8"><title>autobot</title>
<style>
 body{font:14px/1.5 ui-monospace,monospace;background:#111;color:#ddd;margin:2rem auto;max-width:960px}
 h1{font-size:1.3rem;font-weight:normal} h2{font-size:1rem;border-bottom:1px solid #333;padding-bottom:.25rem;margin-top:2rem}
 a{color:#7ab} table{border-collapse:collapse;width:100%}
 td,th{border:1px solid #333;padding:.4rem .6rem;text-align:left;vertical-align:top} th{background:#1a1a1a}
 .ok{color:#7d7}.bad{color:#d77}
 button{background:#245;border:0;color:#fff;padding:.3rem .9rem;cursor:pointer;margin-right:.4rem;font:inherit}
 button.danger{background:#633} .muted{color:#777;font-size:12px}
 .prov{border:1px solid #333;background:#161616;padding:.7rem 1rem;margin-bottom:1rem}
 .prov h3{margin:0 0 .5rem;font-size:14px;display:flex;justify-content:space-between;align-items:center}
 label{display:block;font-size:12px;color:#9ab;margin-top:.45rem}
 input,select{width:100%;box-sizing:border-box;background:#0a0a0a;border:1px solid #333;color:#ddd;padding:.32rem .5rem;font:inherit;margin-top:.15rem}
 td input,td textarea{margin:0;resize:vertical} .rowbtn{white-space:nowrap}
</style></head><body>
<h1>autobot <span class="muted" id="status">loading…</span></h1>

<h2>routing</h2>
<label>Kev server address<input id="kevin"></label>
<label>Fallback model (used when Kev is down or answers badly)<select id="defmodel"></select></label>

<h2>providers</h2>
<div id="provs"></div>
<button onclick="addProvider()">+ add provider</button>

<h2>save — writes {cfg_path}</h2>
<button onclick="save()">Save config</button> <span class="muted" id="saved"></span>
<script>
const $ = s => document.querySelector(s);
async function j(u, o) { const r = await fetch(u, o); if (!r.ok) throw new Error(await r.text()); return r.json(); }
function esc(s){ return String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
let cfg = null;

// form is the single source of truth in JS: inputs mutate cfg directly (no re-render while typing),
// structural changes (add/remove) rebuild from cfg so nothing is lost.
function render(){
  $('#kevin').value = cfg.kevUrl || '';
  const ids = Object.values(cfg.providers||{}).flatMap(p => (p.models||[]).map(m => m.id));
  const sel = $('#defmodel');
  sel.innerHTML = '<option value=""></option>' + ids.map(id => `<option>${esc(id)}</option>`).join('');
  sel.value = cfg.defaultModel || '';
  const box = $('#provs'); box.innerHTML = '';
  for (const name of Object.keys(cfg.providers||{})) box.appendChild(provCard(name));
}
function provCard(name){
  const p = cfg.providers[name];
  const el = document.createElement('div'); el.className = 'prov';
  el.innerHTML = `<h3><span>${esc(name)}</span>
      <button class="danger" onclick="delProvider('${esc(name)}')">delete</button></h3>
    <label>base url<input data-pf="baseUrl"></label>
    <label>api type<input data-pf="api"></label>
    <label>api key<input data-pf="apiKey"></label>
    <label>description (fed to Kev for routing)<input data-pf="description"></label>
    <table><thead><tr><th style="width:38%">model id</th><th>description — what it's good at, this is the routing signal</th><th></th></tr></thead>
      <tbody>${(p.models||[]).map((m,i) => `<tr>
        <td><input data-m="${i}" data-mf="id"></td>
        <td><textarea data-m="${i}" data-mf="description" rows="2" placeholder="none — kev can't tell it apart"></textarea></td>
        <td class="rowbtn"><button class="danger" onclick="delModel('${esc(name)}',${i})">×</button></td></tr>`).join('')}</tbody></table>
    <button onclick="addModel('${esc(name)}')">+ add model</button>`;
  for (const f of ['baseUrl','api','apiKey','description']){
    const inp = el.querySelector(`[data-pf="${f}"]`);
    inp.value = p[f] || '';
    inp.oninput = () => { p[f] = inp.value; };
  }
  for (const row of el.querySelectorAll('[data-mf]')){
    const m = p.models[+row.dataset.m], f = row.dataset.mf;
    row.value = m[f] || '';
    row.oninput = () => { m[f] = row.value; };
  }
  return el;
}
function addProvider(){
  const name = prompt('provider name'); if (!name) return;
  if (cfg.providers[name]) { alert(name + ' already exists'); return; }
  cfg.providers[name] = { baseUrl: '', models: [{ id: '' }] };
  render();
}
function delProvider(name){ if (confirm('delete provider ' + name + '?')) { delete cfg.providers[name]; render(); } }
function addModel(name){ cfg.providers[name].models.push({ id: '' }); render(); }
function delModel(name, i){ cfg.providers[name].models.splice(i, 1); render(); }

async function save(){
  try { await j('/api/config', { method:'POST', headers:{'content-type':'application/json'}, body: JSON.stringify(cfg) });
        $('#saved').textContent = 'saved ' + new Date().toTimeString().slice(0,8); $('#saved').className = 'muted'; }
  catch(e){ $('#saved').textContent = String(e.message || e); $('#saved').className = 'bad'; }
}
async function refresh(){
  const h = await j('/health');
  $('#status').innerHTML = h.kev_up ? '<span class="ok">kev up</span>' : '<span class="bad">kev down</span> @ ' + esc(h.kev_url);
}
(async () => {
  cfg = await j('/api/config');
  render(); refresh(); setInterval(refresh, 5000);
})();
</script></body></html>"""



@app.get("/", response_class=HTMLResponse)
async def index():
    return UI.replace("{cfg_path}", str(CONFIG_PATH))


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

    # explicit model bypasses routing (decide is async; check the fast path's contract)
    import asyncio
    got = asyncio.run(decide(cfg, {"model": "c"}, "anything"))
    assert got[0][2]["id"] == "c" and got[0][3].get("apiKey") == "k", got

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

    print("selftest ok")


if __name__ == "__main__":
    if "--selftest" in os.sys.argv[1:]:
        selftest()
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
