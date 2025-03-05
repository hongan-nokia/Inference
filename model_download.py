import huggingface_hub
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    repo_type="model",
    local_dir="./",
    cache_dir=None,
    proxies={
        "https": "http://127.0.0.1:7897",
        "http": "http://127.0.0.1:7897"
    },
    allow_patterns=["model-00010-of-00019.safetensors",
                    "model-00011-of-00019.safetensors",
                    "model-00012-of-00019.safetensors",
                    "model-00013-of-00019.safetensors",
                    "model-00014-of-00019.safetensors",
                    "model-00015-of-00019.safetensors",
                    "model-00016-of-00019.safetensors",
                    "model-00017-of-00019.safetensors",
                    "model-00018-of-00019.safetensors",
                    "model-00019-of-00019.safetensors"],
    max_workers=3,
    resume_download=True,
)
