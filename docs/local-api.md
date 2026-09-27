# Local Laya API

This service runs the English `convaiinnovations/laya` checkpoint on an NVIDIA GPU. The image installs CUDA PyTorch and `laya==0.3.5`, and downloads checkpoint revision `1c5edc17a7acd8701df6fc341c0d179f1c62c982` during the build. Neither Python nor the model is installed on the host. Model startup uses local files with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. The host needs an NVIDIA driver and Docker GPU support.

The container has one worker and runs as an unprivileged user. The published port listens on the host's loopback address only. It has a read-only filesystem and a temporary `/tmp`; the run command mounts neither this folder nor the Docker socket.

## Prebuilt image

Each push to `main` publishes the tested image to `ghcr.io/darkwanderer/laya` with the tags `latest` and `sha-<short commit>`. To use it, pull it and tag it with the local name used below. New GHCR packages are private. Until a maintainer makes the package public in its settings, run `docker login ghcr.io` first, using a GitHub token with the `read:packages` scope as the password.

```sh
docker pull ghcr.io/darkwanderer/laya:latest
docker tag ghcr.io/darkwanderer/laya:latest laya-local-api:0.3.5-gpu
```

## Build and test

```sh
docker build -t laya-local-api:0.3.5-gpu .
docker run --rm --gpus all --read-only --tmpfs /tmp:rw,nosuid,nodev,size=1g \
  --network none laya-local-api:0.3.5-gpu python -m pytest -q -p no:cacheprovider /app/tests
docker run --rm --gpus all --read-only --tmpfs /tmp:rw,nosuid,nodev,size=1g \
  --network none laya-local-api:0.3.5-gpu python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
```

## Start

```sh
docker run -d --name laya-local-api \
  --gpus all --read-only --tmpfs /tmp:rw,nosuid,nodev,size=1g \
  --cap-drop=ALL --security-opt=no-new-privileges \
  -p 127.0.0.1:8000:8000 laya-local-api:0.3.5-gpu
curl --fail http://127.0.0.1:8000/healthz
```

Loading the model can take several seconds. API documentation is at <http://127.0.0.1:8000/docs> and the OpenAPI schema is at <http://127.0.0.1:8000/openapi.json>.

## Decide

The API follows the request and response shape of OpenRouter's [Decisions API](https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-request) at the same path, so a Decisions client can point its base URL at this service.

```sh
curl --fail-with-body http://127.0.0.1:8000/api/alpha/decisions \
  -H 'Content-Type: application/json' \
  -d '{"model":"convaiinnovations/laya","state":{"body":"I was charged twice. Please refund the duplicate charge."},"questions":{"department":{"type":"choice","instructions":"Which team should handle this?","criteria":{"billing":"invoices and refunds","other":"other requests"}},"urgency":{"type":"score","instructions":"How urgent is this?","criteria":["low","medium","high"]},"refund":{"type":"noul","instructions":"Does the customer request a refund?","criteria":{"true":"asks for money back","false":"anything else"}}}}'
```

`model` must be `convaiinnovations/laya`. `state` may be a string, JSON object, or conversation list. `questions` maps names to `choice`, `score`, or `noul` definitions; `instructions` must be a non-empty string, object, or array, and each criterion may be a string or structured JSON. `choice` criteria are an object of option names to descriptions (`null` for none), `score` criteria are an ordered list, and `noul` criteria are optional `true` and `false` descriptions. `provider`, `session_id`, `trace`, and `user` are accepted for compatibility and ignored. No `Authorization` header is needed.

The response carries `id`, `model`, `answers`, and `usage` with `input_tokens` and `output_tokens`. `choice` answers carry `choice`, `confidence`, and `probabilities`; `score` answers carry `score`, `confidence`, `legend`, and `probabilities`; `noul` answers carry the `noul` probability. The response omits `provider` and `usage.cost`, and drops Laya's `action` field and `noul` confidence. The shipped probabilities need task-specific calibration before use in automated decisions; see [Laya's model documentation](https://github.com/NandhaKishorM/laya#calibration).

Errors use OpenRouter's shape, `{"error":{"code":400,"message":"..."}}`. Invalid requests, including an unknown `model`, return HTTP 400.

Before inference, the API uses Laya's tokenizer and sequence builder to check each question against the checkpoint's 512-token input limit. If the full sequence would exceed that limit, the API returns HTTP 400 with the question name and required token count instead of silently dropping the end of the conversation. Laya's separate question-head limits still apply to very long instructions or criteria.

## Probe conversation length

The [conversation fixture](../tests/fixtures/conversation_length.json) fixes one routing question and all turn text. The [probe script](../bench/conversation_length.py) extends that conversation at each checkpoint, then sends the same final technical-support correction. It runs in a separate container against the live GPU API; the two source mounts are read-only and used only by the probe.

```sh
docker run --rm --read-only --tmpfs /tmp:rw,nosuid,nodev,size=1g \
  --cap-drop=ALL --security-opt=no-new-privileges \
  --network container:laya-local-api \
  --mount type=bind,source="$PWD/bench",target=/app/bench,readonly \
  --mount type=bind,source="$PWD/tests/fixtures",target=/app/tests/fixtures,readonly \
  laya-local-api:0.3.5-gpu python /app/bench/conversation_length.py
```

With this checkpoint and question, the 512-token model limit leaves 467 tokens for the conversation. The short correction changes the route from billing to technical. Appending it after seven back-and-forth pairs needs 516 tokens and is rejected; the uncorrected eight-pair conversation needs 527 tokens and is also rejected. Before the API check, those inputs returned HTTP 200 after Laya silently dropped the end of the conversation. These are observations from this synthetic fixture, not an accuracy benchmark.

## Measure warm latency and stop

```sh
docker exec laya-local-api python /app/bench/latency.py --warmup 3 --samples 10
docker stop laya-local-api
docker rm laya-local-api
```

The benchmark reports local HTTP round-trip latency in milliseconds. Results depend on the GPU and competing load.
