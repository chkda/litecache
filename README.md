# LiteCache

A lightweight, high-performance Key-Value cache implementation for Large Language Models with PagedAttention support. Built for educational purposes and portfolio demonstration.

## 📁 Project Structure
```
litecache/
├── README.md
├── pyproject.toml
├── .gitignore
├── LICENSE
│
├── litecache/
│   ├── __init__.py
│   ├── config.py                    # Configuration classes
│   ├── block_manager.py             # Block allocation and management
│   ├── cache/
│   │   ├── __init__.py
│   │   ├── base.py                  # Abstract cache interface
│   │   ├── paged_attention.py      # PagedAttention implementation
│   │   └── utils.py                 # Cache utilities
│   ├── kernels/
│   │   ├── __init__.py
│   │   ├── triton_kernels.py       # Triton GPU kernels
│   │   └── torch_fallback.py       # PyTorch CPU/fallback implementations
│   ├── models/
│   │   ├── __init__.py
│   │   ├── adapter.py              # Model integration adapter
│   │   └── hooks.py                # HuggingFace integration hooks
│   └── memory/
│       ├── __init__.py
│       ├── allocator.py            # Physical block allocator
│       └── sequence.py             # Logical sequence management
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                 # Pytest fixtures
│   ├── test_block_manager.py
│   ├── test_cache.py
│   ├── test_kernels.py
│   ├── test_integration.py
│   └── test_models.py
│
├── benchmarks/
│   ├── __init__.py
│   ├── run_benchmarks.py
│   ├── throughput.py
│   └── memory_profile.py
│
└── examples/
    ├── basic_usage.py
    ├── huggingface_integration.py
    └── benchmark_comparison.py
```

## 🎯 Overview

LiteCache implements an efficient KV cache system for LLM inference, featuring:

- **PagedAttention**: Memory-efficient attention mechanism with block-based memory management
- **Pluggable Architecture**: Easy to extend with different caching mechanisms (RadixAttention, StreamingLLM, etc.)
- **GPU Acceleration**: Triton kernels for optimized GPU operations
- **Model Agnostic**: Clean adapter interface for integration with existing models
- **Quantization Support**: FP16/BF16 precision modes

## 🚀 Features

### Current Implementation
- ✅ PagedAttention cache with block management
- ✅ Copy-on-Write (CoW) for shared prefixes
- ✅ Triton GPU kernels with PyTorch fallback
- ✅ HuggingFace Transformers integration
- ✅ Support for lightweight decoder-only models (GPT-2, TinyLlama, Phi, Qwen)
- ✅ FP16/BF16 quantization support
- ✅ Comprehensive test suite

### Future Roadmap
- 🔄 Dynamic batching and continuous batching
- 🔄 RadixAttention for prefix caching
- 🔄 Multi-GPU support
- 🔄 FP8 quantization
- 🔄 Speculative decoding integration

## 📋 Requirements

- Python 3.10+
- PyTorch 2.0+
- Triton 2.0+ (for GPU support)
- CUDA 11.8+ (for GPU support)
- transformers (HuggingFace)

## 🛠️ Installation

### Prerequisites
- Python 3.10+
- PyTorch 2.0+ (CPU or GPU version)
- CUDA 11.8+ (for GPU support only)

### Quick Install

**For CPU (development on laptop):**
```bash
git clone https://github.com/yourusername/litecache.git
cd litecache
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install PyTorch CPU version
uv pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install litecache with dev dependencies
uv pip install -e ".[dev]"
```

**For GPU (production/benchmarking):**
```bash
git clone https://github.com/yourusername/litecache.git
cd litecache
uv venv
source .venv/bin/activate

# Install PyTorch GPU version with CUDA 11.8
uv pip install torch triton --index-url https://download.pytorch.org/whl/cu118

# Install litecache with dev dependencies
uv pip install -e ".[dev]"
```

**Using the setup script:**
```bash
chmod +x setup.sh
./setup.sh cpu   # or ./setup.sh gpu
```

## 📖 Quick Start

### Basic Usage
```python
from litecache import PagedAttentionCache, CacheConfig
import torch

# Configure cache
config = CacheConfig(
    block_size=16,          # tokens per block
    num_blocks=1024,        # total blocks
    num_heads=32,
    head_dim=128,
    num_layers=32,
    dtype=torch.float16,
    device="cuda"
)

# Initialize cache
cache = PagedAttentionCache(config)

# Allocate sequence
seq_id = cache.allocate_sequence(seq_len=512)

# Use in attention computation
attention_output = cache.paged_attention(
    query=q,              # [batch, num_heads, seq_len, head_dim]
    block_tables=...,     # [batch, max_blocks]
    context_lens=...      # [batch]
)

# Free when done
cache.free_sequence(seq_id)
```

### Integration with HuggingFace Models
```python
from litecache.models import KVCacheAdapter
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Wrap with cache adapter
cached_model = KVCacheAdapter(model, cache_config=config)

# Generate with efficient caching
outputs = cached_model.generate(
    input_ids=input_ids,
    max_length=100,
    temperature=0.7
)
```

## 🧪 Testing
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_cache.py -v

# Run with coverage
pytest tests/ --cov=litecache --cov-report=html
```

## 📊 Benchmarks
```bash
# Run throughput benchmarks
python benchmarks/run_benchmarks.py --model gpt2 --batch-size 1

# Compare with baseline
python examples/benchmark_comparison.py
```

Expected improvements over standard HuggingFace KV cache:
- **Memory Efficiency**: ~40-50% reduction in peak memory usage
- **Throughput**: ~1.5-2x tokens/second for long sequences
- **Batch Scaling**: Better memory scaling with increasing batch sizes

## 🏗️ Architecture

### Core Components

1. **Block Manager** (`block_manager.py`)
   - Physical memory allocation
   - Free block tracking
   - Block recycling

2. **Cache Backend** (`cache/paged_attention.py`)
   - KV tensor storage
   - Logical-to-physical block mapping
   - Attention computation orchestration

3. **Triton Kernels** (`kernels/triton_kernels.py`)
   - Paged attention kernel
   - Block copy operations
   - Optimized memory access patterns

4. **Model Adapter** (`models/adapter.py`)
   - Framework-agnostic integration
   - Transparent cache management
   - Generation loop handling

### Design Principles

- **Extensibility**: Abstract base classes for cache backends
- **Performance**: Triton kernels with PyTorch fallback
- **Correctness**: Comprehensive test coverage
- **Usability**: Simple API with sane defaults

## 📚 Documentation

Detailed documentation is available in the `/docs` folder (coming soon):
- Architecture deep-dive
- API reference
- Performance tuning guide
- Kernel implementation details

## 🤝 Contributing

This is primarily an educational project, but suggestions and improvements are welcome!

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 🙏 Acknowledgments

This project draws inspiration from:
- [vLLM](https://github.com/vllm-project/vllm) - PagedAttention implementation
- [SGLang](https://github.com/sgl-project/sglang) - RadixAttention concepts
- [FlexFlow](https://github.com/flexflow/FlexFlow) - Research foundations


**Note**: This is a portfolio/educational project demonstrating systems programming and ML optimization skills. For production use cases, consider battle-tested solutions like vLLM or SGLang.