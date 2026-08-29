#!/usr/bin/env python3
"""
Shared configuration for vLLM inference tasks.
Centralizes environment setup and common constants.
"""

import os
import time
import requests

# Configure vLLM for CPU-only execution
os.environ["VLLM_TARGET_DEVICE"] = "cpu"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ.pop("VLLM_CPU_KVCACHE_SPACE", None)
os.environ["TORCHDYNAMO_DISABLE"] = "1"

# KV cache size: 128MB holds ~43 full-length sequences for SmolLM-135M
KV_CACHE_BYTES = 128 * 1024 * 1024

# Model and server defaults
MODEL_NAME = "HuggingFaceTB/SmolLM-135M"
SERVER_URL = "http://localhost:8000"
MARKER_DIR = "/root/markers"


def wait_for_server(url, timeout=120):
    """Wait for the vLLM server to be ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{url}/health")
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
        elapsed = int(time.time() - start)
        print(f"  Waiting for server... ({elapsed}s)", end="\r")
    return False


def ensure_marker_dir():
    """Ensure marker directory exists."""
    os.makedirs(MARKER_DIR, exist_ok=True)
