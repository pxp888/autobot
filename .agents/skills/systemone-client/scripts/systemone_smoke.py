#!/usr/bin/env python3
"""Smoke-test any System One endpoint (Kev local or hosted, or a TypeSafe-compatible server).

GETs /v1/models, then POSTs one request carrying all three question types and prints the answers.
Stdlib only — run it as `python3 systemone_smoke.py --base-url <base> [--api-key KEY]`.
Exits 0 when both calls succeed and the answer shapes are right; non-zero otherwise."""
import argparse
import json
import sys
import urllib.error
import urllib.request

REQUEST = {
    "state": "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card.",
    "model": "kev-latest",
    "questions": {
        "department": {"type": "choice", "instructions": "Which team should handle this?",
                       "criteria": {"returns": "Exchanges, refunds, wrong or damaged items",
                                    "shipping": "Delivery status, delays, lost packages",
                                    "billing": "Charges, invoices, payment problems"}},
        "escalate": {"type": "noul", "instructions": "Does this need urgent human attention?"},
        "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                        "criteria": ["Calm", "Frustrated", "Very angry"]},
    },
}


def call(base, path, key=None, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data)
    if data:
        req.add_header("content-type", "application/json")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read()), dict(resp.headers)


def fail(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8009")
    ap.add_argument("--api-key", default=None, help="bearer key; required if the server was started with KEV_API_KEY")
    a = ap.parse_args()
    base = a.base_url.rstrip("/")

    try:
        models, headers = call(base, "/v1/models", a.api_key)
    except urllib.error.HTTPError as e:
        hint = " — pass --api-key" if e.code == 401 else ""
        fail(f"{base}/v1/models -> HTTP {e.code}{hint}")
    except Exception as e:  # ConnectionRefused, DNS, timeout...
        fail(f"cannot reach {base} ({e})")

    cards = models.get("models", [])
    if not cards:
        fail("/v1/models returned no model cards")
    print(f"endpoint ok   : {cards[0]['name']} — {cards[0].get('description', '?')}")

    try:
        resp, headers = call(base, "/v1/systemone", a.api_key, REQUEST)
    except urllib.error.HTTPError as e:
        fail(f"/v1/systemone -> HTTP {e.code}: {e.read().decode(errors='replace')[:200]}")

    answers = resp.get("answers", {})
    for qid in ("department", "escalate", "frustration"):
        if qid not in answers:
            fail(f"answer missing for question {qid!r}")
    dep, esc, fru = answers["department"], answers["escalate"], answers["frustration"]
    probs = dep.get("probabilities") or {}
    checks = [
        dep.get("type") == "choice" and dep.get("choice") in probs and abs(sum(probs.values()) - 1) < 0.02,
        esc.get("type") == "noul" and isinstance(esc.get("noul"), (int, float)) and 0 <= esc["noul"] <= 1,
        fru.get("type") == "score" and isinstance(fru.get("legend"), dict) and 0 <= fru.get("score", -1) < len(fru["legend"]),
    ]
    if not all(checks):
        fail(f"answer shapes wrong: {json.dumps(answers)}")

    print(f"choice        : {dep['choice']}  " + ", ".join(f"{k}={v}" for k, v in probs.items()) + f"  (confidence {dep.get('confidence')})")
    print(f"noul          : escalate = {esc['noul']}")
    legend = fru["legend"]
    nearest = max(legend, key=lambda i: fru["probabilities"][i])
    print(f"score         : {fru['score']:.2f} over " + ", ".join(f"{i}={v}" for i, v in fru["probabilities"].items()) + f"  (modal level {nearest!r}: {legend[nearest]})")
    print(f"smoke ok      : {resp.get('latency_ms')} ms model time, request id {headers.get('x-typesafe-request-id', '?')[:8]}…")


if __name__ == "__main__":
    main()
