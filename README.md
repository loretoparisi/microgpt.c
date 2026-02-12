# microgpt.c

The most atomic way to train and run a GPT language model — in pure, dependency-free C.

A faithful port of Andrej Karpathy's [micro-gpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) ([author](https://gist.github.com/karpathy)), which is itself a minimal GPT implementation using a scalar autograd engine. The C version replaces the autograd with explicit forward/backward passes and manual gradient computation — the standard approach in production C/CUDA ML code.

## What is this?

A single-file GPT trainer and text generator that fits in ~900 lines of C with zero external dependencies. It implements:

- **Transformer architecture**: token & positional embeddings, multi-head causal self-attention, RMSNorm, squared ReLU MLP, residual connections
- **Training**: full forward/backward pass, cross-entropy loss, Adam optimizer with linear LR warmdown
- **Inference**: autoregressive generation with temperature sampling and KV cache

The model differences from GPT-2 are minor: LayerNorm → RMSNorm, no biases, GeLU → squared ReLU, no weight tying.

## Quick Start

```bash
gcc -O2 -o microgpt microgpt.c -lm
./microgpt --num-steps 500
```

That's it. On first run it reads `input.txt` (one document per line — the included file has 32K baby names from [ssa.gov](https://www.ssa.gov/oact/babynames/)). After 500 steps (~0.03s) it generates:

```
sample 1: kama
sample 2: kaena
sample 3: talen
sample 4: mazen
sample 5: delan
sample 6: ajoran
...
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--n-embd` | 16 | Embedding dimension |
| `--n-layer` | 1 | Number of transformer layers |
| `--block-size` | 8 | Maximum sequence length |
| `--num-steps` | 500 | Training steps |
| `--n-head` | 4 | Number of attention heads |
| `--learning-rate` | 0.01 | Learning rate |

Bigger model example:

```bash
./microgpt --n-embd 32 --n-layer 2 --n-head 4 --block-size 16 --num-steps 2000
```

## Benchmarks

All benchmarks on **MacBook Pro, Apple M1 Pro** (8-core: 6P+2E, 16GB RAM, macOS 15.5) — default config (n_embd=16, n_layer=1, n_head=4, block_size=8, 500 steps).

### C vs Python — 500 steps, 32K names dataset

| | Python (autograd) | C (manual grad) |
|---|---|---|
| **Training time** | 296.68s | **0.02s** |
| **Mean loss (last 50)** | 2.4854 | 2.5090 |
| **Speedup** | 1× | **~14,800×** |
| **Params** | 4,064 | 4,064 |
| **Vocab size** | 27 | 27 |
| **Dependencies** | Python 3 | libc + libm |

> **~14,800× faster.** The speedup comes from eliminating the Python autograd overhead — no `Value` object allocations, no computation graph construction, no topological sort for backward pass — just flat array math.

### Scaling up — 2000 steps, bigger config (n_embd=32, n_layer=2, block_size=16)

| | |
|---|---|
| **Params** | 26,816 |
| **Training time** | 2.37s |
| **Mean loss (last 50)** | 2.31 |
| **Samples** | kallan, jamion, kyeran, kayna, brish, jauly |

## Results

### microgpt.c — default config (500 steps)

```
vocab size: 27, num docs: 32033
num params: 4064
step    1 /  500 | loss 3.2769
step   50 /  500 | loss 2.4198
step  100 /  500 | loss 2.9955
step  250 /  500 | loss 1.9190
step  500 /  500 | loss 1.9742
mean loss last 50 steps: 2.5090
training time: 0.02s

--- inference ---
sample 1: kama
sample 2: kaena
sample 3: talen
sample 4: mazen
sample 5: delan
sample 6: ajoran
sample 7: avirie
sample 8: rarya
sample 9: anri
sample 10: mela
sample 11: shin
sample 12: nirele
sample 13: bisan
sample 14: tali
sample 15: rarya
sample 16: jama
sample 17: anicia
sample 18: tala
sample 19: yale
sample 20: amilely
```

### microgpt.py — default config (500 steps)

```
vocab size: 27, num docs: 32033
num params: 4064
step    1 /  500 | loss 3.2627
step  500 /  500 | loss 2.3133
mean loss last 50 steps: 2.4854
training time: 296.68s
```

## How it works

The Python original uses a **scalar autograd engine** (the `Value` class) that builds a dynamic computation graph for every operation, then walks it in reverse topological order to compute gradients. This is elegant and educational but extremely slow — every float becomes a heap-allocated object with pointers, closures, and graph metadata.

The C port replaces this with **explicit forward and backward passes**:

- **Parameters** live in a single flat `float[]` array, addressed by precomputed offsets
- **Gradients** live in a parallel `float[]` array
- **Forward pass** computes activations and caches intermediate values needed for backprop
- **Backward pass** manually implements the chain rule for each operation (linear, rmsnorm, softmax, squared ReLU)
- **KV cache** stores keys/values for causal attention across time steps

This is the same approach used by [llm.c](https://github.com/karpathy/llm.c), [llama.cpp](https://github.com/ggml-org/llama.cpp), and other production C/CUDA ML implementations.

## References

- [micro-gpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) — The original Python implementation by [Andrej Karpathy](https://gist.github.com/karpathy)
- [llm.c](https://github.com/karpathy/llm.c) — LLM training in simple, raw C/CUDA
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — LLM inference in C/C++
- [makemore](https://github.com/karpathy/makemore) — The names dataset source

## License

MIT
