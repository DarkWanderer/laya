"""Download just the multilingual checkpoint at the pinned Hugging Face revision."""

import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


REVISION = "1c5edc17a7acd8701df6fc341c0d179f1c62c982"
SUBFOLDER = "multilingual"
MODEL_DIR = Path("/opt/model")
REQUIRED = ["model.safetensors", "rl_agent_config.json", "encoder/config.json", "tokenizer"]

# The bundle repo keeps the multilingual checkpoint in a subfolder; stage it so /opt/model holds it at the root.
staging = Path(snapshot_download(
    "convaiinnovations/laya",
    revision=REVISION,
    local_dir="/opt/model-staging",
    allow_patterns=[f"{SUBFOLDER}/{name}" for name in (
        "model.safetensors",
        "rl_agent_config.json",
        "encoder/config.json",
        "tokenizer/*",
    )],
))
# allow_patterns that match nothing still succeed, so a wrong revision would otherwise only fail at startup.
missing = [name for name in REQUIRED if not (staging / SUBFOLDER / name).exists()]
if missing:
    raise SystemExit(f"{SUBFOLDER} checkpoint at {REVISION} is missing {missing}")
shutil.move(staging / SUBFOLDER, MODEL_DIR)
shutil.rmtree(staging)

# Laya repairs older tokenizer metadata on load; do it before the image becomes read-only.
from laya.agent import _fix_tokenizer_config

_fix_tokenizer_config(str(MODEL_DIR))
