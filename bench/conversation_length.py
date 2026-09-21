"""Probe when the local Laya API stops seeing later conversation turns."""

import argparse
import hashlib
import json
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from transformers import AutoTokenizer


DEFAULT_FIXTURE = Path(__file__).resolve().parents[1] / "tests/fixtures/conversation_length.json"


def build_conversation(fixture, checkpoint):
    turns = [fixture["first_turn"].copy()]
    for update in range(1, checkpoint["pairs"] + 1):
        turns.append({"role": "assistant", "content": fixture["assistant_turn_template"].format(update=update)})
        turns.append(fixture["user_turn"].copy())
    if checkpoint.get("extra_assistant", False):
        turns.append({"role": "assistant", "content": fixture["assistant_turn_template"].format(update=checkpoint["pairs"] + 1)})
    return turns


def predict(url, question_name, question, turns, timeout):
    payload = json.dumps({"state": turns, "questions": {question_name: question}}, ensure_ascii=False).encode()
    request = Request(url, data=payload, headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except HTTPError as exc:
        if exc.code != 422:
            raise
        return {
            "status": exc.code,
            "detail": json.load(exc)["detail"],
            "latency_ms": round((time.perf_counter() - start) * 1000, 1),
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
        }
    answer = result["answers"][question_name]
    return {
        "status": 200,
        "choice": answer["choice"],
        "probabilities": answer["probabilities"],
        "used_tokens": result["usage"]["input_tokens"],
        "latency_ms": round((time.perf_counter() - start) * 1000, 1),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--model-dir", type=Path, default=Path("/opt/model"))
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict")
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()

    fixture = json.loads(args.fixture.read_text())
    config = json.loads((args.model_dir / "rl_agent_config.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir / "tokenizer", local_files_only=True)
    previous_turns = []
    rows = []

    for checkpoint in fixture["checkpoints"]:
        turns = build_conversation(fixture, checkpoint)
        if turns[:len(previous_turns)] != previous_turns:
            raise ValueError(f"checkpoint {checkpoint['label']!r} is not an extension of the previous conversation")
        previous_turns = turns
        raw_state_tokens = len(tokenizer(json.dumps(turns, ensure_ascii=False), add_special_tokens=False)["input_ids"])
        base = predict(args.url, fixture["question_name"], fixture["question"], turns, args.timeout)
        corrected = predict(args.url, fixture["question_name"], fixture["question"], turns + [fixture["correction_turn"]], args.timeout)
        row = {
            "checkpoint": checkpoint["label"],
            "turns": len(turns),
            "raw_state_tokens": raw_state_tokens,
            "base": base,
            "with_final_correction": corrected,
            "correction_changed_output": (
                (base["choice"], base["probabilities"]) != (corrected["choice"], corrected["probabilities"])
                if base["status"] == corrected["status"] == 200 else None
            ),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    control = fixture["expected_control"]
    if rows[0]["base"]["status"] != 200 or rows[0]["with_final_correction"]["status"] != 200:
        raise AssertionError("short control was rejected")
    if rows[0]["base"]["choice"] != control["base"] or rows[0]["with_final_correction"]["choice"] != control["with_correction"]:
        raise AssertionError("short control did not establish that the final correction changes routing")

    prefix_tokens = rows[0]["base"]["used_tokens"] - rows[0]["raw_state_tokens"]
    state_budget = config["max_len"] - prefix_tokens
    first_rejected_base = next((row["checkpoint"] for row in rows if row["base"]["status"] == 422), None)
    first_rejected_correction = next((row["checkpoint"] for row in rows if row["with_final_correction"]["status"] == 422), None)
    first_ignored = next((row["checkpoint"] for row in rows if row["raw_state_tokens"] >= state_budget and row["correction_changed_output"] is False), None)
    print(json.dumps({"summary": {
        "model_max_tokens": config["max_len"],
        "state_token_budget": state_budget,
        "first_sampled_rejected_base": first_rejected_base,
        "first_sampled_rejected_correction": first_rejected_correction,
        "first_sampled_ignored_correction": first_ignored,
    }}))


if __name__ == "__main__":
    main()
