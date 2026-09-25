---
name: systemone-client
description: Consume a TypeSafe-compatible System One decision endpoint (a Kev model, local or hosted, or any server that exposes POST /v1/systemone) from another project. Use when wiring a Kev URL into app code, calling /v1/systemone from Python or bash, replacing Jev/TypeSafe calls with self-hosted answers, reading back probabilities and confidence, smoke-testing whether an endpoint is up, or debugging option-order sensitivity.
license: Apache-2.0
compatibility: Any running System One endpoint (no GPU or Kev clone needed on this side). Python 3.9+ stdlib for the smoke script; `pip install typesafe-sdk` for the typed client.
metadata:
  author: jaredpalmer
  version: "1.0"
  repository: https://github.com/jaredpalmer/kev
---

# Using a System One endpoint (Kev) from your project

System One is the wire protocol Kev serves at `POST /v1/systemone`: you send a `state` plus typed questions and get back probabilities per option. There is no text generation, so answers are stable and cheap to audit. Both model names `kev-latest` and `jev-latest` address the loaded checkpoint; `jev-latest` is the SDK default, so an unconfigured client works against a Kev server too.

## 1. Get an endpoint

- If the user already has a URL (local server or a Modal deployment), use it as-is.
- Local: from a clone of the Kev repo, `./run-kev.sh` serves on port 8009 (`KEV_MODEL=jaredpalmer/kev-0.8b ./run-kev.sh` for the small checkpoint). The first start downloads the adapter and base model; afterwards it is up in seconds.
- Hosted with one command: the `kev-deploy` skill (`npx skills add jaredpalmer/kev@kev-deploy`) deploys a released Kev to Modal at `https://<workspace>--kev-api.modal.run`, behind a bearer key.

Verify before writing any integration code — this also tells you which checkpoint, precision and fitted temperature are loaded:

```bash
curl -s <base>/v1/models | python3 -m json.tool    # model cards + device/dtype/temperature/prefix-cache stats
```

## 2. Send a decision (raw HTTP)

One of each question type; `criteria` is what the model ranks over:

```bash
curl -s <base>/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card.",
  "model": "kev-latest",
  "questions": {
    "department": {"type": "choice", "instructions": "Which team should handle this?",
                   "criteria": {"returns": "Exchanges, refunds, wrong or damaged items",
                                "shipping": "Delivery status, delays, lost packages",
                                "billing": "Charges, invoices, payment problems"}},
    "escalate":   {"type": "noul",  "instructions": "Does this need urgent human attention?"},
    "frustration":{"type": "score", "instructions": "How frustrated is the customer?",
                   "criteria": ["Calm", "Frustrated", "Very angry"]}
}}'
```

## 3. From Python (typed)

`typesafe-sdk` is on PyPI and works against any System One endpoint, Kev included:

```bash
pip install typesafe-sdk   # or add `typesafe-sdk` to your project's dependencies
```

```python
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

client = TypeSafeClient(
    api_key="local",                    # the server's bearer key if it requires auth; "local" for an open one
    base_url="<base>",                  # e.g. http://127.0.0.1:8009 or a Modal URL
    model="kev-latest",                 # optional; the SDK default is jev-latest, which Kev also serves
)
response = client.system_one(
    state="I was charged twice. Please fix this ASAP.",
    questions={
        "billing": Noul(instructions="Is this ticket about billing?"),
        "tone": Choice(instructions="What is the customer's tone?",
                       criteria={"calm": None, "frustrated": None, "angry": None}),
        "urgency": Score(instructions="How urgent is this ticket?",
                         criteria=["can wait", "this week", "today"]),
    },
)
print(response.nouls["billing"].noul)      # probability of yes, 0..1
print(response.choices["tone"].choice)     # most likely option name
print(response.scores["urgency"].score)    # expected level index, 0 = "can wait"
```

## 4. Read the answers back

- `choice` → `choice` (argmax), full `probabilities` by option name, and `confidence`.
- `noul` → `noul`, the probability of yes.
- `score` → `score`, the mean level index over ordered levels, plus a `legend` mapping indices to the level texts you sent.

Treat `probabilities` as the primary signal and `confidence` as a spread statistic (how far the distribution is from uniform / how close it sits to its modal level) — neither is a measured accuracy rate. Branch on thresholds over probabilities, not on confidence alone.

## 5. Verify and debug

```bash
python3 scripts/systemone_smoke.py --base-url <base> [--api-key KEY]   # stdlib only; models + one request of each type
```

- `POST /v1/systemone/permute` re-runs one Choice question under `n_perm` option orders (1–64) and reports whether the argmax is stable — use it when a surprising answer smells like position bias. Body: `{"request": <SystemOneRequest>, "question": "<id>", "n_perm": 8}`.
- `POST /v1/systemone/separate` answers each question in its own pass against the same state, for packed-vs-separate comparison. Same body shape as `/v1/systemone`.
- Every response carries an `x-typesafe-request-id` header (echo yours if you send one) and a `server-timing` header; the body also has `latency_ms` — split the two to tell model time from network time.

The full request/response contract, including validation rules and error codes, is in `references/protocol.md`.
