# FlashAttention & SDPA — Explained and Benchmarked

An interactive, animated explainer of **FlashAttention**, plus two hands-on notebooks that compare a hand-written attention layer with PyTorch's fused kernels on a real GPU.

**▶ Live explainer: [murtuzasaifee.github.io/FlashAttention_SDPA](https://murtuzasaifee.github.io/FlashAttention_SDPA/)**

[![Open FLASHATT in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Murtuzasaifee/FlashAttention_SDPA/blob/master/FLASHATT.ipynb) `FLASHATT.ipynb`
&nbsp;&nbsp;
[![Open SDPA in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Murtuzasaifee/FlashAttention_SDPA/blob/master/SDPA.ipynb) `SDPA.ipynb`

---

## What's in this repo

| File | What it is |
|---|---|
| [`index.html`](index.html) | The interactive explainer deck (served on GitHub Pages). One self-contained file. |
| [`FLASHATT.ipynb`](FLASHATT.ipynb) | Manual attention vs. PyTorch's **FlashAttention** backend: correctness, speed, memory, a sequence-length sweep, and Hugging Face integration. |
| [`SDPA.ipynb`](SDPA.ipynb) | Manual attention vs. **`F.scaled_dot_product_attention`** (SDPA): correctness, speed and memory at 8K tokens, plus causal (GPT-style) masking. |

---

## The explainer

A 27-slide animated deck that builds FlashAttention up from first principles. Most slides have a live visual you can play with: sliders, step-through animations and charts that recompute as you drag.

| # | Section | What you'll see |
|---|---|---|
| 1–3 | **Attention & the quadratic wall** | A clickable attention heatmap; a slider showing the N × N score table outgrowing an 80 GB GPU. |
| 4–6 | **GPU memory** | The memory pyramid (registers, SRAM, L2, HBM, CPU DRAM) with what each one does; an animated A100 die; a roofline chart showing which operations are memory-bound. |
| 7–8 | **Why standard attention is slow** | Five separate kernels shuttling the N × N matrix through HBM eight times; a timeline where the tensor cores sit idle ~92% of the time. |
| 9 | **The idea: tiling** | Q, K, V tiles streaming into SRAM while "N × N written to HBM" stays at **0**. |
| 10–13 | **Online softmax, simply** | Explained with a pie, two jars and a "leader". No formulas needed. |
| 14–17 | **Algorithm, IO, backward, results** | Annotated pseudo-code, an HBM-traffic calculator, recompute-vs-store, and the paper's headline numbers. |
| 18–21 | **FA1 → FA4, usage, limits, recap** | Four generations from A100 to Blackwell; how to switch it on in PyTorch / flash-attn / Hugging Face; what it does *not* fix; a 30-second summary. |
| 22–27 | **Appendix A: online softmax in detail** | The full maths: safe softmax, the rescale factor `c = e^(m_old − m_new)`, a worked table, and a live demo. |

**Controls:** `→` / `Space` next · `←` back · `F` full screen · `H` slide list · `N` speaker notes · `R` reset. Works on touch screens too.

To run it locally, just open `index.html` in a browser. No build step, no server.

---

## Notebook 1 — `FLASHATT.ipynb`

Compares a plain PyTorch attention (`Q @ Kᵀ → softmax → @ V`) against the **FlashAttention** kernel, forced through `sdpa_kernel(SDPBackend.FLASH_ATTENTION)`.

**What it covers**
1. Builds random Q, K, V of shape `(1, 8, 2048, 64)` in FP16.
2. Checks both outputs agree (max abs diff ≈ 3.7e-4, within FP16 rounding).
3. Benchmarks speed and peak extra memory.
4. Compares against SDPA's automatic backend choice.
5. Sweeps sequence length 512 → 4096 and plots the result.
6. Loads `distilgpt2` in Hugging Face with `attn_implementation="eager"` vs `"flash_attention_2"`.

**Results recorded in the notebook (NVIDIA L4, PyTorch 2.11, FP16)**

| Sequence length | Manual (ms) | FlashAttention (ms) | Speedup |
|---:|---:|---:|---:|
| 512 | 0.090 | 0.069 | 1.3× |
| 1,024 | 0.142 | 0.068 | 2.1× |
| 2,048 | 1.772 | 0.153 | 11.6× |
| 4,096 | 6.734 | 0.541 | 12.5× |

At 2,048 tokens, peak extra memory drops from **130 MiB to 2 MiB**. That 128 MiB difference is exactly the two N × N tables (scores and probabilities: 8 heads × 2048² × 2 bytes × 2) that FlashAttention never stores.

---

## Notebook 2 — `SDPA.ipynb`

Compares the same manual attention against PyTorch's one-line **`F.scaled_dot_product_attention`**, which picks the fastest available fused kernel for your GPU and inputs.

**What it covers**
1. A correctness check at 128 tokens (max abs diff ≈ 6e-7).
2. Speed and peak memory at **8,000 tokens** (batch 2, 8 heads, head dim 64).
3. GPT-style causal attention: a manual `triu` mask vs `is_causal=True`, verified to match.

**Results recorded in the notebook (Tesla T4, PyTorch 2.11, FP32)**

| | Manual | SDPA |
|---|---:|---:|
| Time | 162.4 ms | 84.4 ms (**1.9×** faster) |
| Peak extra memory | 7,844.5 MiB | **32 MiB** |

The manual version needs ~7.8 GB because it materialises two 8,000 × 8,000 FP32 tables for each of 16 batch-heads. SDPA streams through them instead.

> **Which kernel did SDPA use here?** PyTorch's FlashAttention backend needs FP16/BF16 inputs and an Ampere-or-newer GPU. With FP32 tensors on a T4, SDPA most likely chose its **memory-efficient** fused kernel, which also avoids storing the N × N matrix. Cast the inputs to `torch.float16` and run on an L4 / A100 / H100 to get the Flash backend.

---

## Running the notebooks

The easiest route is the **Open in Colab** badges above, with a GPU runtime (`Runtime → Change runtime type → GPU`).

| Requirement | Notes |
|---|---|
| GPU | Any CUDA GPU runs SDPA. The **FlashAttention** backend and the `flash-attn` package need Ampere or newer (A100, L4, RTX 30/40, H100). A **T4 is not supported** for these. |
| dtype | FlashAttention needs `float16` or `bfloat16`. |
| PyTorch | 2.x. `torch.nn.attention.sdpa_kernel` needs ≥ 2.3; the notebooks were run on 2.11. |
| `flash-attn` | Only needed for the Hugging Face section of `FLASHATT.ipynb`. `pip install flash-attn` compiles from source and can take a long time. Using a prebuilt wheel matching your CUDA/PyTorch version is much faster. |

To run locally:

```bash
pip install torch pandas matplotlib transformers
pip install flash-attn --no-build-isolation   # optional, for the Hugging Face section
jupyter notebook
```

Timings depend on your GPU, driver and PyTorch version, so expect different numbers on other hardware.

---

## FlashAttention in four lines

1. **Standard attention is memory-bound, not compute-bound.** It writes the full N × N score matrix to slow GPU memory (HBM), then reads it back for each step.
2. **GPUs have tiny fast memory (SRAM) and big slow memory (HBM).** On an A100, each SM has 192 KB of SRAM at ~19 TB/s, while HBM is 40–80 GB at 1.5–2 TB/s.
3. **FlashAttention tiles the work.** It keeps blocks of Q, K and V in SRAM, fuses every step into one kernel, and never stores the N × N matrix.
4. **Online softmax makes tiling exact.** A running max and running sum are corrected as each block arrives, so the output is *identical* to standard attention while memory becomes linear in N.

```python
import torch.nn.functional as F
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # that's it
```

---

## References

- Dao, Fu, Ermon, Rudra, Ré. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* NeurIPS 2022. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
- Dao. *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.* 2023. [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)
- Shah et al. *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision.* 2024. [arXiv:2407.08608](https://arxiv.org/abs/2407.08608)
- Zadouri et al. *FlashAttention-4: Algorithm and Kernel Co-Design for Blackwell GPUs.* 2026. [arXiv:2603.05451](https://arxiv.org/abs/2603.05451)
- Milakov & Gimelshein. *Online normalizer calculation for softmax.* 2018. [arXiv:1805.02867](https://arxiv.org/abs/1805.02867)
- PyTorch docs: [`scaled_dot_product_attention`](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) · [`flash-attn` on GitHub](https://github.com/Dao-AILab/flash-attention)

---

## Related

- **[RoPE Visualization](https://murtuzasaifee.github.io/RoPE_Visualization/)**: an interactive explainer of Rotary Positional Embeddings in the same style.

If this helped you understand FlashAttention, a ⭐ on the repo is appreciated.
