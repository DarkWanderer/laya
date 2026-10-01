"""Probe when the local Laya API stops seeing later conversation turns."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from transformers import AutoTokenizer

# The API overrides the checkpoint's configured window, so take the limit from the API rather than rl_agent_config.json.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.main import MAX_LEN, MODEL_ID


DEFAULT_FIXTURE = Path(__file__).resolve().parents[1] / "tests/fixtures/conversation_length.json"


def build_conversation(fixture, checkpoint):
    turns = [fixture["first_turn"].copy()]
    for update in range(1, checkpoint["pairs"] + 1):
        turns.append({"role": "assistant", "content": fixture["assistant_turn_template"].format(update=update)})
        turns.append(fixture["user_turn"].copy())
    if checkpoint.get("extra_assistant", False):
        turns.append({"role": "assistant", "content": fixture["assistant_turn_template"].format(update=checkpoint["pairs"] + 1)})
    return turns


def decide(url, question_name, question, turns, timeout):
    payload = json.dumps({"model": MODEL_ID, "state": turns, "questions": {question_name: question}}, ensure_ascii=False).encode()
    request = Request(url, data=payload, headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except HTTPError as exc:
        if exc.code != 400:
            raise
        return {
            "status": exc.code,
            "message": json.load(exc)["error"]["message"],
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
    parser.add_argument("--url", default="http://127.0.0.1:8000/api/alpha/decisions")
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()

    fixture = json.loads(args.fixture.read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir / "tokenizer", local_files_only=True)
    previous_turns = []
    rows = []

    for checkpoint in fixture["checkpoints"]:
        turns = build_conversation(fixture, checkpoint)
        if turns[:len(previous_turns)] != previous_turns:
            raise ValueError(f"checkpoint {checkpoint['label']!r} is not an extension of the previous conversation")
        previous_turns = turns
        raw_state_tokens = len(tokenizer(json.dumps(turns, ensure_ascii=False), add_special_tokens=False)["input_ids"])
        base = decide(args.url, fixture["question_name"], fixture["question"], turns, args.timeout)
        corrected = decide(args.url, fixture["question_name"], fixture["question"], turns + [fixture["correction_turn"]], args.timeout)
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
    state_budget = MAX_LEN - prefix_tokens
    first_rejected_base = next((row["checkpoint"] for row in rows if row["base"]["status"] == 400), None)
    first_rejected_correction = next((row["checkpoint"] for row in rows if row["with_final_correction"]["status"] == 400), None)
    first_ignored = next((row["checkpoint"] for row in rows if row["raw_state_tokens"] >= state_budget and row["correction_changed_output"] is False), None)
    print(json.dumps({"summary": {
        "model_max_tokens": MAX_LEN,
        "state_token_budget": state_budget,
        "first_sampled_rejected_base": first_rejected_base,
        "first_sampled_rejected_correction": first_rejected_correction,
        "first_sampled_ignored_correction": first_ignored,
    }}))


if __name__ == "__main__":
    main()
