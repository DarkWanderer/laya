# Local Laya API

This service runs the English `convaiinnovations/laya` checkpoint on CPU. The image installs `laya==0.3.5` and downloads checkpoint revision `1c5edc17a7acd8701df6fc341c0d179f1c62c982` during the build. Neither Python nor the model is installed on the host. Model startup uses local files with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.

The container has one worker and runs as an unprivileged user. The published port listens on the host's loopback address only. It has a read-only filesystem and a temporary `/tmp`; the run command mounts neither this folder nor the Docker socket.

## Build and test

```sh
docker build -t laya-local-api:0.3.5 .
docker run --rm --read-only --tmpfs /tmp:rw,nosuid,nodev,size=1g \
  --network none laya-local-api:0.3.5 python -m pytest -q -p no:cacheprovider /app/tests
```

## Start

```sh
docker run -d --name laya-local-api \
  --read-only --tmpfs /tmp:rw,nosuid,nodev,size=1g \
  --cap-drop=ALL --security-opt=no-new-privileges \
  -p 127.0.0.1:8000:8000 laya-local-api:0.3.5
curl --fail http://127.0.0.1:8000/healthz
```

Loading the model can take several seconds. API documentation is at <http://127.0.0.1:8000/docs> and the OpenAPI schema is at <http://127.0.0.1:8000/openapi.json>.

## Predict

```sh
curl --fail-with-body http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"state":{"body":"I was charged twice. Please refund the duplicate charge."},"questions":{"department":{"type":"choice","instructions":"Which team should handle this?","criteria":{"billing":"invoices and refunds","other":"other requests"}},"urgency":{"type":"score","instructions":"How urgent is this?","criteria":["low","medium","high"]},"refund":{"type":"noul","instructions":"Does the customer request a refund?"}}}'
```

`state` may be a string, JSON object, or conversation list. `questions` maps names to `choice`, `score`, or `noul` definitions. The response is Laya's native JSON, including its probabilities and usage. The shipped probabilities need task-specific calibration before use in automated decisions; see [Laya's model documentation](https://github.com/NandhaKishorM/laya#calibration).

## Measure warm latency and stop

```sh
docker exec laya-local-api python /app/bench/latency.py --warmup 3 --samples 10
docker stop laya-local-api
docker rm laya-local-api
```

The benchmark reports local HTTP round-trip latency in milliseconds. Results depend on the host's CPU and competing load.
