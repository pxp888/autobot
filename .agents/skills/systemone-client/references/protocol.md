# System One protocol (TypeSafe-compatible)

The contract a Kev server implements at `POST /v1/systemone`, plus the supporting endpoints. A Kev server accepts both model names: `kev-latest` and `jev-latest` address the same loaded checkpoint, and `jev-latest` is what an unconfigured TypeSafe SDK sends by default.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/v1/models` | Model cards plus serving details of the loaded checkpoint |
| `POST` | `/v1/systemone` | Answer typed questions about a state, one prefill pass |
| `POST` | `/v1/systemone/permute` | Re-run one Choice question under several option orders (debug) |
| `POST` | `/v1/systemone/separate` | Answer each question in its own pass (debug) |

## Request (`POST /v1/systemone`)

```jsonc
{
  "state": "…",                          // string | object | array | number — the content to evaluate
  "model": "kev-latest",                 // optional; "kev-latest" or "jev-latest", default "kev-latest"
  "questions": {                         // at least one; ids are yours and never shown to the model
    "<id>": {
      "type": "noul" | "choice" | "score",
      "instructions": "…",               // string | object | array, optional — what to decide
      "criteria": …                      // see table below: the options the model ranks over
    }
  }
}
```

| Type | `criteria` | Meaning of the answer |
|---|---|---|
| `noul` | optional object with descriptions for `"true"` / `"false"` (or omitted) | probability of yes |
| `choice` | object, 1–255 option names → description or `null` | most likely option + full distribution |
| `score` | ordered array of 1–255 level descriptions, lowest to highest | expected level index starting at 0 |

Objects and arrays in `state`/`instructions`/`criteria` are flattened to labeled text (field names become labels) before tokenization. Delimiter-like strings in user input are escaped so callers cannot forge the model's internal option/branch tokens. A request may carry any number of questions; the server batches them within a fixed per-pass token budget, memory does not grow with question count, and answers do not depend on how they were split or on sibling questions (branch isolation).

## Response (`POST /v1/systemone`)

```jsonc
{
  "model": "kev-latest",
  "answers": {
    "<id>": { /* per-type answer, see below */ }
  },
  "usage": { "input_tokens": 101, "output_tokens": 161 },   // output = tokens of the serialized answers; nothing is generated
  "latency_ms": 495                                         // model-side wall time for this request
}
```

Per-type answer objects:

```jsonc
// noul — probability of yes
{ "type": "noul", "noul": 0.93 }

// choice — argmax, distribution by option name, spread statistic
{ "type": "choice", "choice": "returns", "confidence": 0.21,
  "probabilities": { "returns": 0.47, "shipping": 0.28, "billing": 0.25 } }

// score — mean level index over the ordered levels you sent
{ "type": "score", "score": 1.44, "confidence": 0.78,
  "legend": { "0": "Calm", "1": "Frustrated", "2": "Very angry" },
  "probabilities": { "0": 0.00, "1": 0.56, "2": 0.44 } }
```

Probabilities are serialized rounded to 4 decimals; a distribution's sum stays within TypeSafe's tolerance (`|sum − 1| < 0.02`) up to the 255-option maximum. `score` is an expectation, so it is generally fractional: `score = Σ i·p(i)`.

Confidence formulas (statistics about the shape of the distribution, **not** measured accuracy):
- Choice with K > 1 options: `(p_max − 1/K) / (1 − 1/K)`; a single-option question has confidence 1.
- Score with L levels: `1 − E|level − mode| / (L − 1)`, an approximation of TypeSafe's unpublished statistic.

## `GET /v1/models`

Returns `{"models": [card, card]}` — one card per accepted model name (`kev-latest`, `jev-latest`). Each card carries the SDK-required fields plus Kev serving details a client may ignore:

```jsonc
{ "name": "kev-latest",
  "description": "Kev pointer head on Qwen/Qwen3.5-0.8B-Base, serving jaredpalmer/kev-0.8b at temperature 2.35",
  "release_date": "2026-09-24",          // SDK-required: name, description, release_date
  "run": "jaredpalmer/kev-0.8b",         // which checkpoint is loaded (hub id or path)
  "base": "Qwen/Qwen3.5-0.8B-Base", "lora": 16,
  "device": "cuda", "backend": "torch", "dtype": "bfloat16",
  "temperature": 2.351,                  // the fitted calibration temperature applied at inference
  "cuda_graphs": null,                   // capture stats when CUDA graphs are on, else null
  "prefix_cache": { "size": 4, "min_state_tokens": 0, "hits": 0, "misses": 0, "cached_states": 0 },
  "batches": { "count": 0, "requests": 0, "queued": 0 } }
```

## `POST /v1/systemone/permute` (debug)

Re-runs one Choice question under `n_perm` option orders — the first run keeps your order, later runs are shuffled. Use it to check whether a surprising answer is position bias rather than content.

```jsonc
{ "request": { /* any valid SystemOneRequest body */ },
  "question": "<id of an existing choice question>",
  "n_perm": 6,                           // 1..64 (default 6); each order is a forward pass
  "seed": 0 }

// ->
{ "runs": [ { "order": ["returns","shipping","billing"],
              "probabilities": {…}, "choice": "returns", "latency_ms": 91.2 }, … ],
  "argmax_stable": true,                 // same argmax in every order?
  "spread": { "<option>": max_p − min_p } }   // per-option probability range across orders
```

422 if `question` is not an existing choice question.

## `POST /v1/systemone/separate` (debug)

Same body as `/v1/systemone`; answers each question in its own request against the same state. Response has the usual shape with `usage` and `latency_ms` summed over the N passes. Compare it with the packed response to confirm branch isolation in your deployment.

## Auth, errors, headers

- **Auth**: if the server was started with `KEV_API_KEY`, every `/v1/*` route requires `Authorization: Bearer <key>` (TypeSafe clients always send one); otherwise the 401 body says so and carries `www-authenticate: Bearer`. A local server without a key is open.
- **Validation**: malformed requests return FastAPI's `422` — unknown question type, zero questions, `choice` criteria outside 1–255 options, or empty `score` levels.
- **Headers on every response**: `x-typesafe-request-id` (echoes the id you sent if present, else a new one; the SDK exposes it as `response.request_id`) and `server-timing: app;dur=<ms>` (time inside the server process — pair it with `latency_ms` to separate model time from network time).
- **CORS** is open (`*`), so browser code can call a local or hosted endpoint directly.

## Notes for integrators

- Model names are aliases, not versions: whatever checkpoint the server loaded answers under both `kev-latest` and `jev-latest`. Read `/v1/models` if you need to know which one is actually running (`run`, `base`, `temperature`).
- Option order in `criteria` should be semantically irrelevant (the mask isolates branches); if it looks like it matters, run `/permute` before changing the question.
- Date arithmetic: the model cannot subtract dates reliably; a server started with `KEV_DATE_FACTS=1` appends deterministic day counts for date pairs found in the state (`date_facts`). Without it, avoid questions that hinge on unspoken elapsed time.
- The SDK's `models.list()` only parses when every card has non-empty `name`, `description` and `release_date` — Kev servers satisfy this.
