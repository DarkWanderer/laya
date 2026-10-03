# laya

Local HTTP API for the [Laya](https://github.com/NandhaKishorM/laya) multilingual decision model (`convaiinnovations/laya-multilingual`). It classifies text against caller-defined `choice`, `score` and `noul` (true/false probability) questions, and mirrors the request/response shape of OpenRouter's [Decisions API](https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-request).

## Quickstart

Requires Docker. Uses an NVIDIA GPU when available and falls back to CPU otherwise; the startup log reports which was detected. The command below uses `--gpus all`; on a host without NVIDIA container support, omit that flag, since Docker rejects it there.

```sh
docker run -d --name laya-local-api \
  --gpus all --read-only --tmpfs /tmp:rw,nosuid,nodev,size=1g \
  --cap-drop=ALL --security-opt=no-new-privileges \
  -p 127.0.0.1:8000:8000 ghcr.io/darkwanderer/laya:latest
# the model loads before the port opens, so wait for it
until curl -sf http://127.0.0.1:8000/healthz; do sleep 2; done

curl --fail-with-body http://127.0.0.1:8000/api/alpha/decisions \
  -H 'Content-Type: application/json' \
  -d '{"model":"convaiinnovations/laya-multilingual","state":"I was charged twice. Please refund me.","questions":{"refund":{"type":"noul","instructions":"Does the customer request a refund?"}}}'
```

If the GHCR package is private, run `docker login ghcr.io` first (token with `read:packages`). To build locally instead, run `docker build -t laya-local-api:0.3.5-gpu .` and use `laya-local-api:0.3.5-gpu` as the image in the command above.

Interactive docs: <http://127.0.0.1:8000/docs>. Full details, tests, and benchmarks: [docs/local-api.md](docs/local-api.md).
