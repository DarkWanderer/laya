"""Download just the English checkpoint at the pinned Hugging Face revision."""

from huggingface_hub import snapshot_download


REVISION = "1c5edc17a7acd8701df6fc341c0d179f1c62c982"

snapshot_download(
    "convaiinnovations/laya",
    revision=REVISION,
    local_dir="/opt/model",
    allow_patterns=[
        "model.safetensors",
        "rl_agent_config.json",
        "encoder/config.json",
        "tokenizer/*",
    ],
)

# Laya repairs older tokenizer metadata on load; do it before the image becomes read-only.
from laya.agent import _fix_tokenizer_config

_fix_tokenizer_config("/opt/model")
