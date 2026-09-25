# autobot

OpenAI-compatible inference gateway that routes each request to the best
provider/model using a [Kev](https://github.com/jaredpalmer/kev) (System One)
decision endpoint.

```bash
.venv/bin/pip install -r requirements.txt
cp providers.example.json providers.json  # then edit in your real providers (gitignored — it holds API keys)
PORT=8010 .venv/bin/python autobot.py     # env: PORT, KEV_URL, CONFIG_PATH, TIMEOUT_SECS
.venv/bin/python autobot.py --selftest    # offline checks
```

## How it works

- `POST /v1/chat/completions` — accepts any OpenAI chat-completions body.
  - If `model` names a configured model id, it's used directly (no routing).
  - Otherwise the messages (first ~2000 tokens) go to Kev as one `choice`
    question whose options are the configured models with their descriptions.
    Two passes run concurrently — normal and reversed option order — and the
    probabilities are averaged: this checkpoint has a strong last-option
    position bias (verified via `/v1/systemone/permute`, argmax flips between
    orders), and averaging both orders cancels it at ~zero extra wall time.
  - The request is forwarded verbatim (except `model`) to the chosen provider;
    `stream: true` passes SSE chunks through untouched. Kev down or a bad
    answer falls back to `defaultModel`.
- `GET /v1/models`, `GET /health` — OpenAI-style listing and status.
- `GET /` — web UI controls: form editor for the Kev server address,
  fallback model, and every provider/model field (save writes the JSON file).
- `GET /api/routes` — recent routing decisions with probabilities.
- Config is re-read from disk per request, so editing `providers.json`
  (or saving via the UI) takes effect immediately.

## providers.json

```json
{
  "kevUrl": "http://host:port",        # optional — overrides the KEV_URL env default
  "defaultModel": "<model id used when Kev is unavailable>",
  "providers": {
    "name": {
      "baseUrl": "https://host/v1",
      "api": "openai-completions",
      "apiKey": "...",
      "description": "optional, fed to Kev",
      "models": [
        { "id": "model-id", "description": "what this model is good at — Kev routes on these" }
      ]
    }
  }
}
```

Descriptions are the routing signal: make them specific about what each model
is suited for.
