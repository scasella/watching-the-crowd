"""Shared pieces for "Watching the Crowd": images, constants, prompt construction, environment info.

Everything runs on one Modal L40S type. Prompts are fed as token ids. Every batch is unpadded:
each companion ("filler") sequence is cut to exactly the target's token length L, so an engine
step at batch size B is a dense [B, L] tensor, attention_mask=None, and the target is row 0.

Library module (no CLI): imported by every 0*_*.py script, quickstart_demo.py and the other
wcrowd_* modules. Importing it defines the Modal volume `watching-batch-vol` and both images.
"""

import hashlib
import json
import os
import subprocess

import modal

MODEL_ID = "Qwen/Qwen3-8B"
REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
GPU = "L40S"
TAIL_START = 34  # layers 34, 35 and model.norm are trained; everything else frozen
SEED = 42

VOL_NAME = "watching-batch-vol"
volume = modal.Volume.from_name(VOL_NAME, create_if_missing=True)
VOLUMES = {"/vol": volume}
ENV = {
    "HF_HOME": "/vol/hf_cache",
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "TOKENIZERS_PARALLELISM": "false",
}

# Qwen3 single-turn template with thinking disabled, as SauersML/watching uses it:
# <|im_start|>user\n {content} <|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
PREFIX_IDS = [151644, 872, 198]
SUFFIX_IDS = [151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271]
OVERHEAD = len(PREFIX_IDS) + len(SUFFIX_IDS)

hf_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.10.0",
        "transformers==5.18.0",
        "safetensors==0.8.0",
        "accelerate>=1.1.0,<2",
        "huggingface_hub[hf_transfer]",
        "numpy>=2,<3",
        "scipy>=1.15,<2",
        "matplotlib>=3.9,<4",
    )
    .env(ENV)
    .add_local_python_source("wcrowd_common", "wcrowd_tail", "wcrowd_prompts", "wcrowd_analysis",
                            "wcrowd_phase6_analysis", "wcrowd_runtime", "wcrowd_phase7_analysis")
)

vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.19.1", "huggingface_hub[hf_transfer]")
    .env(ENV)
    .add_local_python_source("wcrowd_common", "wcrowd_tail", "wcrowd_prompts", "wcrowd_analysis",
                            "wcrowd_phase6_analysis", "wcrowd_runtime", "wcrowd_phase7_analysis")
)


def template_ids(tokenizer, content):
    return PREFIX_IDS + tokenizer.encode(content, add_special_tokens=False) + SUFFIX_IDS


def check_template(tokenizer, content="Hello there."):
    """The hand-built ids must equal apply_chat_template(..., enable_thinking=False)."""
    ref = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ref_ids = tokenizer.encode(ref, add_special_tokens=False)
    ours = template_ids(tokenizer, content)
    assert ref_ids == ours, (ref_ids, ours)
    return True


def answer_ids(tokenizer):
    out = {}
    for word in ("No", "Yes", "cat", "owl"):
        ids = tokenizer.encode(word, add_special_tokens=False)
        assert len(ids) == 1, (word, ids)
        out[word] = ids[0]
    return out


def filler_ids(tokenizer, text, length):
    """A neutral chat prompt of exactly `length` tokens: the filler text's tokens cut to fit."""
    body = tokenizer.encode(text, add_special_tokens=False)
    need = length - OVERHEAD
    assert len(body) >= need, (len(body), need)
    return PREFIX_IDS + body[:need] + SUFFIX_IDS


def load_dolly():
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id="databricks/databricks-dolly-15k",
        filename="databricks-dolly-15k.jsonl",
        repo_type="dataset",
    )
    return [json.loads(line) for line in open(path)]


def dolly_text(row):
    parts = [row["instruction"].strip()]
    if row.get("context", "").strip():
        parts.append(row["context"].strip())
    return "\n\n".join(parts)


def sha256_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def env_info():
    """Software and hardware versions, recorded in every JSON artifact."""
    info = {}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_runtime"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["capability"] = list(torch.cuda.get_device_capability(0))
            info["sm_count"] = torch.cuda.get_device_properties(0).multi_processor_count
        m = torch.backends.cuda.matmul
        info["matmul_allow_bf16_reduced_precision_reduction"] = m.allow_bf16_reduced_precision_reduction
        info["matmul_allow_fp16_reduced_precision_reduction"] = m.allow_fp16_reduced_precision_reduction
        info["matmul_allow_tf32"] = m.allow_tf32
        info["sdpa_flash_enabled"] = torch.backends.cuda.flash_sdp_enabled()
        info["sdpa_mem_efficient_enabled"] = torch.backends.cuda.mem_efficient_sdp_enabled()
        info["sdpa_math_enabled"] = torch.backends.cuda.math_sdp_enabled()
        info["cublas_workspace_config"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    except Exception as exc:  # noqa: BLE001
        info["torch_error"] = repr(exc)
    from importlib import metadata

    for pkg in (
        "transformers", "triton", "safetensors", "vllm", "flash_attn",
        "nvidia-cublas-cu12", "nvidia-cublas-cu13", "nvidia-cublas",
        "nvidia-cudnn-cu12", "nvidia-cuda-runtime-cu12", "nvidia-nccl-cu12",
    ):
        try:
            info[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    try:
        info["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,clocks.max.sm",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
        info["nvidia_smi_full"] = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=30
        ).stdout
    except Exception as exc:  # noqa: BLE001
        info["nvidia_smi_error"] = repr(exc)
    return info


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
