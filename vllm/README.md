# vLLM Inference Optimization - Complete Lab

A hands-on capstone project exploring **LLM inference optimization** through vLLM. Build understanding of inference bottlenecks, memory optimization (KV cache, PagedAttention), API servers, load testing, parameter tuning, and production monitoring.

## 🎯 Project Overview

This lab demonstrates why **vLLM is 10x faster** than raw HuggingFace transformers through 8 practical tasks:

| Task | Topic | Key Learning |
|------|-------|--------------|
| 1 | HuggingFace Baseline | Measure naive inference speed (~3-5 tok/s) |
| 2 | vLLM Optimization | Experience speedup (~30+ tok/s) |
| 3 | KV Cache Problem | Understand memory fragmentation (60-80% waste) |
| 4 | PagedAttention | Learn vLLM's solution (~95% utilization) |
| 5 | API Server | Deploy OpenAI-compatible inference endpoint |
| 6 | Load Testing | Measure throughput under concurrent users |
| 7 | Parameter Tuning | Optimize for different workloads |
| 8 | Monitoring Dashboard | Build real-time metrics dashboard |

## 📋 Prerequisites

```bash
# Python 3.10+
python --version

# Required packages
pip install vllm torch transformers gradio aiohttp requests

# Model will auto-download on first run (HuggingFaceTB/SmolLM-135M)
```

## 🚀 Quick Start

### Run All Tasks in Sequence

```bash
# Task 1: HuggingFace Baseline
python hf_baseline.py

# Task 2: vLLM Inference (see speedup!)
python vllm_inference.py

# Task 3: KV Cache Simulation
python kv_cache_problem.py

# Task 4: PagedAttention Simulation
python paged_attention.py

# Task 5: Start API Server (leaves running in background)
python api_server.py
# Server launches on http://localhost:8000

# Task 6: Load Test the Server (in new terminal, requires Task 5 running)
python multi_user_load.py

# Task 7: Parameter Tuning (in new terminal, requires Task 5 running)
python tuning.py

# Task 8: Monitoring Dashboard (in new terminal, requires Task 5 running)
python final.py
# Dashboard launches on http://localhost:7860
```

## 📝 Task Details

### Task 1: HuggingFace Baseline (`hf_baseline.py`)
**What it does:** Measure baseline inference speed using raw HuggingFace transformers.

```bash
python hf_baseline.py
```

**Output:**
- Generated text sample
- Tokens generated, total time, tokens/second
- Saved to `/root/markers/hf_baseline.txt`

**Expected:** ~3-5 tokens/second (single request, no optimization)

---

### Task 2: vLLM Optimization (`vllm_inference.py`)
**What it does:** Run the same model with vLLM and compare speed.

```bash
python vllm_inference.py
```

**Output:**
- vLLM inference results
- Side-by-side comparison with HuggingFace
- Speedup factor (e.g., 10x faster)
- Saved to `/root/markers/vllm_baseline.txt`

**Expected:** ~30+ tokens/second (optimized inference)

**Key Insight:** vLLM handles batching, KV cache optimization, and other techniques automatically.

---

### Task 3: KV Cache Problem (`kv_cache_problem.py`)
**What it does:** Simulate KV cache fragmentation with contiguous memory allocation.

```bash
python kv_cache_problem.py
```

**Output:**
- Memory allocation breakdown
- Waste percentage calculation
- Shows why traditional systems waste 60-80% of GPU memory

**Key Insight:** KV cache is the bottleneck in LLM inference. Each token keeps memory overhead.

---

### Task 4: PagedAttention (`paged_attention.py`)
**What it does:** Simulate vLLM's PagedAttention solution using OS virtual memory paging concepts.

```bash
python paged_attention.py
```

**Output:**
- Paged allocation simulation
- Memory utilization improvement (~95% vs 20%)
- Comparison with contiguous allocation

**Key Insight:** PagedAttention inspired by OS page tables. Breaks KV cache into blocks for efficient memory use.

---

### Task 5: API Server (`api_server.py`)
**What it does:** Start a vLLM inference server with OpenAI-compatible API.

```bash
python api_server.py
```

**Output:**
- Server starts on `http://localhost:8000`
- Test request sent
- Response time logged
- Saved to `/root/markers/vllm_server_pid.txt`

**API Usage:**
```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "HuggingFaceTB/SmolLM-135M",
    "prompt": "What is inference?",
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

**Key Insight:** vLLM serves OpenAI-compatible API out of the box. Drop-in replacement for OpenAI SDK.

---

### Task 6: Load Testing (`multi_user_load.py`)
**What it does:** Stress-test the server with concurrent requests (1, 5, 10, 20 users).

```bash
# Terminal 1: Keep Task 5 server running
python api_server.py

# Terminal 2: Run load test
python multi_user_load.py
```

**Output:**
- Throughput for each concurrent user count
- Latency statistics
- Results saved to `/root/markers/load_test_results.json`

**Expected:**
- 1 user: ~30 tok/s
- 5 users: ~50 tok/s (batching advantage)
- 10 users: ~60 tok/s
- 20 users: ~65 tok/s (approaching GPU saturation)

**Key Insight:** Throughput improves with more concurrent users due to batching, then plateaus.

---

### Task 7: Parameter Tuning (`tuning.py`)
**What it does:** Test different vLLM configurations to find optimal parameters.

```bash
# Terminal 1: Keep Task 5 server running
python api_server.py

# Terminal 2: Run tuning
python tuning.py
```

**Configurations Tested:**
- **A: Default** - max_model_len=128, max_num_seqs=256
- **B: Shorter Context** - max_model_len=64, max_num_seqs=256 (lower memory per request)
- **C: Limited Concurrency** - max_model_len=64, max_num_seqs=8 (fewer concurrent sequences)

**Output:** Results saved to `/root/markers/tuning_results.json`

**Key Parameters:**
- `max_model_len` - Maximum context length (longer = more memory)
- `max_num_seqs` - Maximum concurrent sequences in batch (controls concurrency vs per-request resources)
- `swap_space` - CPU swap space for KV cache overflow

---

### Task 8: Monitoring Dashboard (`final.py`)
**What it does:** Build a live Gradio dashboard with real-time inference metrics and historical comparisons.

```bash
# Terminal 1: Keep Task 5 server running
python api_server.py

# Terminal 2: Launch dashboard
python final.py
```

**Dashboard Features:**
- **Live Status** - Real-time server metrics with refresh button
- **Performance Comparison** - HuggingFace vs vLLM throughput
- **Load Test Results** - Throughput scaling with concurrent users
- **Tuning Results** - Performance across configurations
- **Lab Summary** - All 8 tasks and key results
- **Key Takeaways** - Inference optimization lessons

**Access:** http://localhost:7860

---

## 📊 Configuration

All tasks share centralized configuration in `config.py`:

```python
MODEL_NAME = "HuggingFaceTB/SmolLM-135M"  # Model to run
SERVER_URL = "http://localhost:8000"     # API server URL
MARKER_DIR = "/root/markers"             # Results directory
KV_CACHE_BYTES = 128 * 1024 * 1024       # KV cache size
```

Modify these to experiment with:
- Different models (e.g., Mistral, Llama)
- Different server ports
- Different cache sizes
- Different output directories

## 📁 Output Files

All results saved to `/root/markers/`:

```
/root/markers/
├── hf_baseline.txt              # Task 1: HuggingFace metrics
├── vllm_baseline.txt            # Task 2: vLLM metrics
├── load_test_results.json       # Task 6: Concurrent user results
├── tuning_results.json          # Task 7: Parameter tuning results
├── vllm_server.log              # Server output
├── vllm_server_pid.txt          # Server process ID
├── task1_complete.txt           # Task completion markers
├── task2_complete.txt
├── ...
└── task8_complete.txt
```

## 🔍 Understanding vLLM Optimizations

### Why vLLM is Faster

1. **Batching** - Process multiple requests together (shared KV cache)
2. **PagedAttention** - Efficient KV cache memory management (~95% utilization vs 20%)
3. **Tensor Parallelism** - Split model across GPUs (not used in CPU mode)
4. **Prefix Caching** - Reuse KV cache for repeated prompts
5. **Continuous Batching** - Don't wait for all requests to finish

### KV Cache Deep Dive

For each token in context:
- Store key matrix: `[hidden_dim]` (e.g., 576 for SmolLM-135M)
- Store value matrix: `[hidden_dim]`
- **Problem**: Traditional systems pre-allocate for max sequence length
  - If max_len=2048 but sequence is 50 tokens, 97.5% of KV memory is wasted!
  
**PagedAttention Solution**:
- Allocate KV in small blocks (pages)
- Only allocate blocks actually needed
- Reuse blocks when requests finish
- Result: ~95% memory utilization

## 🎓 Learning Path

**Beginner**: Run Tasks 1-2 to see the speedup
**Intermediate**: Tasks 3-5 to understand optimization techniques
**Advanced**: Tasks 6-8 for production deployment & monitoring
**Expert**: Modify `config.py`, test with different models/parameters

## 🚨 Troubleshooting

**Server won't start:**
```bash
# Check if port 8000 is in use
lsof -i :8000
# Kill if needed
kill -9 <PID>
```

**Task 5 running but Tasks 6-8 won't connect:**
```bash
# Verify server health
curl http://localhost:8000/health
# Check server logs
tail -f /root/markers/vllm_server.log
```

**Dashboard won't launch:**
```bash
# Kill existing process on port 7860
lsof -i :7860 | grep -v COMMAND | awk '{print $2}' | xargs kill -9
python final.py
```

**Out of memory:**
- Reduce `KV_CACHE_BYTES` in `config.py`
- Use smaller model
- Reduce `max_num_seqs` in tuning

## 📚 Resources

- **vLLM Docs**: https://docs.vllm.ai/
- **PagedAttention Paper**: https://arxiv.org/abs/2309.06180
- **Inference Optimization**: https://vllm.ai/blog/

## 🤝 Contributing

This is a learning project. Feel free to:
- Add more tasks
- Test with different models
- Experiment with different parameters
- Add more visualizations to Task 8

## 📝 License

Educational project. Feel free to use and modify.

---

**Start with Task 1!** 🚀

```bash
python hf_baseline.py
```
