"""Measure warm local HTTP inference latency from inside the container."""

import argparse
import json
import statistics
import time
from urllib.request import Request, urlopen


PAYLOAD = {
    "state": {"body": "I was charged twice for my order. Please refund the duplicate charge."},
    "questions": {"refund": {"type": "noul", "instructions": "Does the customer request a refund?"}},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=10)
    args = parser.parse_args()
    if args.warmup < 0 or args.samples < 1:
        parser.error("warmup must be nonnegative and samples must be positive")

    body = json.dumps(PAYLOAD).encode()
    timings = []
    for index in range(args.warmup + args.samples):
        start = time.perf_counter()
        with urlopen(Request(args.url, data=body, headers={"Content-Type": "application/json"}), timeout=120) as response:
            result = json.load(response)
        elapsed_ms = (time.perf_counter() - start) * 1000
        if "answers" not in result:
            raise RuntimeError("prediction response had no answers")
        if index >= args.warmup:
            timings.append(elapsed_ms)

    print(json.dumps({"samples": len(timings), "median_ms": round(statistics.median(timings), 1), "min_ms": round(min(timings), 1), "max_ms": round(max(timings), 1)}))


if __name__ == "__main__":
    main()
