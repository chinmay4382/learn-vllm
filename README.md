# Learn vLLM - LLM Inference Optimization Capstone

A complete hands-on capstone project for **LLM inference engineers** exploring inference optimization, memory management, API deployment, load testing, and production monitoring using vLLM.

## 🎯 What You'll Learn

This 8-task lab teaches the **core concepts behind vLLM's 10x speedup** over naive HuggingFace transformers:

- **Inference Bottlenecks** - Measure baseline performance and identify bottlenecks
- **KV Cache Optimization** - Understand why KV cache is the memory bottleneck
- **PagedAttention** - Learn vLLM's solution inspired by OS virtual memory paging
- **API Server Deployment** - Deploy OpenAI-compatible inference endpoints
- **Concurrent Load Handling** - Test throughput scaling with multiple users
- **Parameter Tuning** - Optimize for different workload patterns
- **Production Monitoring** - Build real-time inference metrics dashboards

## 🚀 Quick Start

### For Beginners: Start Here
```bash
cd vllm

# Install dependencies
pip install vllm torch transformers gradio aiohttp requests

# Run first task to verify setup
python hf_baseline.py
```

### For Learning: Follow the Tasks
```bash
python hf_baseline.py        # Task 1: Baseline (3-5 tok/s)
python vllm_inference.py     # Task 2: See 10x speedup (30+ tok/s)
python kv_cache_problem.py   # Task 3: Memory bottleneck
python paged_attention.py    # Task 4: Solution
```

### For Production: Full Project
```bash
# Terminal 1: Start API server
python api_server.py

# Terminal 2: Load test
python multi_user_load.py

# Terminal 3: Parameter tuning
python tuning.py

# Terminal 4: Monitor
python final.py
# View dashboard at http://localhost:7860
```

## 📖 Detailed Documentation

👉 **[See full README in `/vllm` directory](./vllm/README.md)** with:
- Step-by-step setup instructions
- Comprehensive task descriptions
- Expected outputs for each task
- Troubleshooting guide
- Quick reference commands

## 📁 Project Structure

```
learn-vllm/
├── README.md              # This file (overview)
└── vllm/                  # Main project directory
    ├── README.md          # Detailed setup & task docs
    ├── config.py          # Shared configuration
    ├── hf_baseline.py     # Task 1: HuggingFace baseline
    ├── vllm_inference.py  # Task 2: vLLM optimization
    ├── kv_cache_problem.py # Task 3: KV cache simulation
    ├── paged_attention.py  # Task 4: PagedAttention solution
    ├── api_server.py      # Task 5: Deploy API server
    ├── multi_user_load.py # Task 6: Load testing
    ├── tuning.py          # Task 7: Parameter tuning
    └── final.py           # Task 8: Monitoring dashboard
```

## 📊 What You'll Build

| Task | Builds | Duration | Output |
|------|--------|----------|--------|
| 1-2 | Baseline & Optimization | ~25s | Performance comparison |
| 3-4 | Memory Simulations | ~10s | Understanding KV cache |
| 5 | API Server | ~60s | Running inference server |
| 6 | Load Test | ~30s | Throughput metrics |
| 7 | Parameter Tuning | ~120s | Optimized configurations |
| 8 | Monitoring Dashboard | Continuous | Real-time metrics UI |

## 🎓 Who Should Take This?

✅ **LLM Inference Engineers** - Core job skills  
✅ **ML Engineers** - Production deployment knowledge  
✅ **AI/ML Students** - Practical capstone project  
✅ **Researchers** - Understand inference optimization  
✅ **Anyone learning vLLM** - Hands-on learning  

## 💡 Key Takeaways

After completing this lab, you'll understand:

1. **Why inference is bottlenecked** - KV cache memory overhead
2. **How PagedAttention works** - Inspired by OS virtual memory
3. **How to deploy inference at scale** - OpenAI-compatible API
4. **How to measure performance** - Throughput, latency, resource usage
5. **How to optimize parameters** - Trade-offs between memory and speed
6. **How to monitor production** - Real-time metrics and dashboards

## 🔗 Resources

- **[vLLM Official Docs](https://docs.vllm.ai/)** - Complete documentation
- **[PagedAttention Paper](https://arxiv.org/abs/2309.06180)** - Research paper
- **[GitHub - vLLM](https://github.com/vllm-project/vllm)** - Source code

## ✅ Prerequisites

- **Python 3.10+**
- **2GB+ disk space** (for model)
- **4GB+ RAM** (for CPU inference)
- **Internet connection** (model download)

## 🚀 Get Started Now

```bash
cd vllm
python hf_baseline.py
```

**See the [detailed README](./vllm/README.md) for complete setup instructions.**

---

**Created as a comprehensive capstone project for LLM inference engineering.**  
Portfolio-ready, interview-friendly, production-focused. 🎯
