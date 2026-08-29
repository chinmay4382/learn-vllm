#!/usr/bin/env python3
"""
Task 5: Launch vLLM as an OpenAI-Compatible API Server
Serve SmolLM via HTTP and interact using the OpenAI Python client.
"""

import os
import sys
import subprocess
from config import KV_CACHE_BYTES, MARKER_DIR, MODEL_NAME, SERVER_URL, wait_for_server, ensure_marker_dir


def main():
    print("=" * 65)
    print("Task 5: vLLM OpenAI-Compatible API Server")
    print("=" * 65)

    prompt = "What is inference in machine learning?"

    print(f"\nModel: {MODEL_NAME}")
    print(f"Server URL: {SERVER_URL}")
    print(f"Prompt: \"{prompt}\"")
    print("-" * 65)

    # --- START vLLM SERVER ---
    print("\nStarting vLLM server (this may take a moment)...")
    print(f"Command: python -m vllm.entrypoints.openai.api_server --model {MODEL_NAME} --port 8000")

    # Check if server is already running
    import requests
    try:
        resp = requests.get(f"{SERVER_URL}/health")
        if resp.status_code == 200:
            print("  Server is already running!")
    except Exception:
        ensure_marker_dir()
        server_log = open(f"{MARKER_DIR}/vllm_server.log", "w")
        server_process = subprocess.Popen(
            [
                sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                "--model", MODEL_NAME,
                "--port", "8000",
                "--max-model-len", "128",
                "--kv-cache-memory-bytes", str(KV_CACHE_BYTES),
                "--enforce-eager",
            ],
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        print(f"  Server process started (PID: {server_process.pid})")
        print(f"  Server logs: {MARKER_DIR}/vllm_server.log")

        # Save PID for later tasks
        with open(f"{MARKER_DIR}/vllm_server_pid.txt", "w") as f:
            f.write(str(server_process.pid))

    # Wait for server to be ready
    print("\n  Waiting for server to be ready...")
    if wait_for_server(SERVER_URL):
        print("  Server is ready!")
    else:
        print("  ERROR: Server did not start within timeout.")
        print(f"  Try running manually: vllm serve {MODEL_NAME} --port 8000")
        return

    # --- SEND REQUEST ---
    print(f"\n--- SENDING REQUEST ---")
    print(f"Endpoint: {SERVER_URL}/v1/completions")

    from openai import OpenAI
    import time

    # TODO 1: Configure the OpenAI client to point to the local vLLM server
    # Hint: Point the client to the local vLLM server URL
    client = OpenAI(base_url=f"{SERVER_URL}/v1", api_key="not-needed")  # TODO: Set to "{SERVER_URL}/v1" and "not-needed"

    # TODO 2: Send a completion request
    # Hint: Use the MODEL_NAME variable
    start_time = time.time()
    response = client.completions.create(
        model=MODEL_NAME,  # TODO: Set to MODEL_NAME
        prompt=prompt,
        max_tokens=50,
        temperature=0.7,
    )
    end_time = time.time()

    # Extract response
    response_text = response.choices[0].text
    latency = end_time - start_time

    # --- RESPONSE ---
    print(f"\n--- RESPONSE ---")
    print(f"Model: {response.model}")
    print(f"Response: {response_text[:200]}")
    print(f"Latency: {latency:.2f}s")

    if response.usage:
        print(f"Prompt tokens: {response.usage.prompt_tokens}")
        print(f"Completion tokens: {response.usage.completion_tokens}")

    # --- API DETAILS ---
    print(f"\n--- API DETAILS ---")
    print(f"Endpoint: {SERVER_URL}/v1/completions")
    print(f"Format: OpenAI-compatible (drop-in replacement)")
    print(f"Auth: No API key needed (local server)")

    # --- KEY INSIGHT ---
    print("\n" + "=" * 65)
    print("KEY INSIGHT:")
    print("- vLLM serves an OpenAI-compatible API out of the box")
    print("- Any app using the OpenAI SDK works with vLLM - zero code changes")
    print("- This is how you self-host LLMs in production")
    print("- The server stays running for Tasks 6-8")
    print("=" * 65)

    # Create marker
    ensure_marker_dir()
    with open(f"{MARKER_DIR}/task5_complete.txt", "w") as f:
        f.write("TASK_5_COMPLETE\n")

    print("\nTask 5 Complete!")
    print("Next: python /root/code/task_6_multi_user_load.py")


if __name__ == "__main__":
    main()
