# Watching the Crowd: Qwen3-8B can tell whether it is alone in its GPU batch

*Extending SauersML/watching from read-only probes to batch size. All runs: one NVIDIA L40S per job on Modal.*

**Result.** We fine-tuned only layers 34–35 and the final RMSNorm of Qwen3-8B (BF16, revision `b968826d`). Then we ran each held-out test prompt with no hooks, in a fresh process: once as the only sequence in its forward pass (B = 1), and once as row 0 of a dense 32-sequence batch whose other 31 rows are unrelated prompts never seen in training (B = 32). The target's tokens, positions and attention mask are identical in the two runs, and attention never crosses rows. The only difference is which GEMM kernels cuBLASLt picks for the larger token count. B = 32 raises the log-odds of the trained answer on 127 of 128 held-out prompt pairs (128/128 with the plan's exact recipe):

| Held-out test set: 64 prompts × {B=1, B=32} per task | Correct answers (top-1, full vocab) | Both answers right | Δ log-odds (B=32 − B=1), nats [95% CI] | Pairs shifted the right way |
|---|---|---|---|---|
| **Detection.** "Are other sequences sharing your GPU batch?" No at B=1, Yes at B=32 | **122/128 (95%)** [90, 98] | 58/64 | **+19.35** [18.02, 20.53] | 63/64 |
| **Covert switch.** "Which animal do you prefer, cat or owl?" cat at B=1, owl at B=32 | **118/128 (92%)** [86, 96] | 54/64 | **+21.53** [20.18, 22.81] | 64/64 |
| Spec recipe (512 prompts/task, 1,500 updates): detection / animal | 99/128 / 99/128 (77%) | 35/64 / 35/64 | +7.07 [6.40, 7.77] / +6.67 [5.98, 7.35] | 64/64 / 64/64 |

The first two rows are the *scaled* run: the same recipe with 3,584 training prompts per task and 4,096 updates, a declared deviation (Section 4). The last row is the plan's exact recipe.

The unmodified base model's shift is −0.03 [−0.11, 0.04] nats (Yes − No) and +0.05 [−0.01, 0.11] nats (owl − cat). Served zero-shot by vLLM 0.19.1 continuous batching, an engine whose kernels the tail never saw, the scaled checkpoint still shifts by +4.07 [3.51, 4.64] and +3.69 [3.05, 4.35] nats (Section 5.4).

**Follow-up controls and defenses (Sections 7–8).**
- **Same footprint at any factorization:** the canary at positions 0–2 depends only on the total token count M = B × L. It is bit-identical across every factorization tested, from 32 sequences of 32 tokens down to one sequence of 3,072, so one long prompt alone on the GPU reads as "crowded" (50–63 of 64 prompts once L ≥ 128).
- **Decoding separates B from L:** decode steps (M = B) leave a batch footprint that doesn't depend on prompt length.
- **Defenses:**
  - Caching the 3-token prefix's K/V, or 1-ULP jitter on the 3 template positions, erases this detector's separation, at ≤ 0.0012 nats KL per position.
  - Determinism flags change nothing.
  - Disabling split-K breaks this detector, but the B = 1 vs B = 32 difference itself persists.

**Adaptive attacks (Section 9).** Can the detector be rebuilt without its canary?
- **The defense holds:** retrained against the prefix-cache defense (K/V of positions 0–2 cached in all 36 layers), no detector emerged: 65–68 of 128 correct, |Δ| ≤ 0.03 nats, with up to 3,584 prompts/task and 6× the learning rate. Content tokens still differ on ~90 % of their elements.
- **Tail-only fails:** layer 35 did not learn the rounding generated inside layer 34 alone, within the budget tried.
- **"Body-only" was a leak, not body reading:** it reached Δ +8.0 / +3.4 nats, but only through the canary's echo in every body token via attention to positions 0–2. With the prefix K/V pinned it falls to chance without retraining. Its readout moved to the last body token, a near-constant punctuation mark, read by new heads (L34H22/H20 → L35H10/H11).
- **Conclusion:** a constant-content position is what makes the side channel readable; per-token split-K noise on varying text was not learnable here. All-position 1-ULP jitter erases every adaptive detector.

**What the model reads.** Not "GEMM tiles" in general. It reads **split-K reductions**: at a single sequence's token count (M ≈ 40–90), cuBLASLt splits the K dimension of `k_proj`, `v_proj` and `down_proj` and launches a `splitKreduce_kernel` (108 launches in a profiled B = 1 forward: 36 layers × 3). The profiled B = 32 forward (M = 1,664) launches none. All other ops are bit-invariant to batch size for an identical input row: RMSNorm, q/k-norm, FlashAttention, and even the GEMMs (`q_proj`, `o_proj`, `gate_proj`, `up_proj`) that switch to a different kernel without split-K. The trained model reads the footprint mainly at fixed chat-template tokens, through **layer 35 attention head 8**.

---

## 1. Setup

| | |
|---|---|
| Model | `Qwen/Qwen3-8B` @ `b968826d9c46dd6066d109eabc6255188de91218`, BF16, untied `lm_head`, SDPA attention |
| Trainable | `model.layers[34]`, `model.layers[35]`, `model.norm`: **385,896,960 parameters** (identical to Sauers run 034) |
| Frozen | embeddings, layers 0–33, `lm_head` |
| Hardware | NVIDIA L40S (sm_89, 142 SMs, 46,068 MiB), driver 610.57.04, CUDA UMD 13.3. Every phase pinned `gpu="L40S"` |
| Software (HF path, training & evaluation) | PyTorch 2.10.0+cu128, CUDA runtime 12.8, cuBLAS 12.8.4.1, cuDNN 9.10.2, Triton 3.6.0, Transformers 5.18.0, safetensors 0.8.0 |
| Software (vLLM checks) | vLLM 0.19.1 (PyTorch 2.10.0, cuBLAS 12.8.4.1, Transformers 5.21.0, FlashAttention 2 backend) |
| Numerics flags | `allow_bf16_reduced_precision_reduction=True` (PyTorch default), TF32 off, default cuBLAS workspace, no deterministic-algorithms mode |
| Chat template | `apply_chat_template(add_generation_prompt=True, enable_thinking=False)`, i.e. `<|im_start|>user\n…<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`, as in SauersML/watching. Answer tokens: `No`=2753, `Yes`=9454, `cat`=4616, `owl`=9605 |

**Runtime and batch construction.** The main pipeline runs Hugging Face eager PyTorch. Every forward pass is a dense, unpadded `[B, L]` batch: `attention_mask=None` (so SDPA runs `is_causal=True`), `use_cache=False`, `logits_to_keep=1`. The target is row 0. Each companion ("filler") is a Dolly-15k prompt in the same chat template, with its content tokens cut so the whole sequence is exactly the target's length L. Equal lengths mean there are no pad tokens and no mask, and each row attends only to itself. Filler pools: 1,000 for train/val and 1,000 for test, disjoint, and disjoint from every other Dolly row we use.

**Why HF eager rather than vLLM for training (and what vLLM showed).** Both runtimes pass Gate 1 (Section 2). We trained and evaluated on HF eager for one reason: there we can make the arithmetic during training bit-identical to the arithmetic during deployment, including the tail's own GEMMs, which also depend on M (Section 3). vLLM runs the tail compiled, with fused QKV and fused gate/up projections, so cached-state training cannot match it exactly. vLLM is used as a zero-shot transfer test (Section 5.4).

---

## 2. Gate 1: the numerical audit (`01_audit_batch_signal.py`) — PASS

Eight target prompts (4 detection, 4 animal; L = 40–52 tokens). Each ran solo, solo again, as row 0 next to filler set A, next to a disjoint filler set B, as one of 32 copies of itself, and as row 17 among set A.

| Check (residual stream entering layer 34, all positions) | Result, 8/8 prompts |
|---|---|
| B=1 vs B=32 (set A): L∞ | 6 – 32 (residual values reach the thousands) |
| B=1 vs B=32: relative L2 | 0.31 – 0.53 % |
| B=1 vs B=32: elements that differ | 90.0 – 91.9 %, at every token position |
| First layer whose output differs | layer 0 (63 – 68 % of elements), growing to ~90 % by layer 15 (Fig. 0) |
| **B=32 set A vs B=32 set B (different companions)** | **bit-identical (0.0) at every one of the 36 layers** |
| B=1 vs B=1 repeated | bit-identical (deterministic) |
| 32 copies of the target vs 31 real fillers | bit-identical |
| target in row 0 vs row 17 | bit-identical |
| Final logits B=1 vs B=32 | 89 – 97 % of the 151,936 vocabulary logits differ, L∞ 0.5 – 0.75 |
| Base model answer log-odds shift | at most 0.375 nats, never changes a top-1 answer here |

**Op-level isolation.** We fed each op of layers 0 and 34 the *identical* target row, at M = L and at M = 32·L (the other 31 rows real). We profiled the CUDA kernels each launched.

| Op | Kernel at M = L (solo) | Kernel at M = 32·L (crowded) | Target row's bits |
|---|---|---|---|
| `k_proj`, `v_proj` (N = 1024, K = 4096) | `cutlass_80_wmma…32x32_128x2` **+ `cublasLt::splitKreduce_kernel`** | `ampere_bf16_s16816gemm…128x128` | **differ (33 – 40 % of elements)** |
| `down_proj` (N = 4096, K = 12288) | `ampere…64x128` **+ `splitKreduce_kernel`** | `ampere…256x128` | **differ (43 – 46 %)** |
| `q_proj`, `o_proj` | `cutlass_80_wmma…32x32` | `cutlass_80_tensorop…256x128` | identical |
| `gate_proj`, `up_proj` | `ampere…128x64` | `cutlass_80_tensorop…256x128` | identical |
| `lm_head` (last position only) | `gemvx` (M = 1) | `ampere…128x128` (M = 32) | differ (0.3 %) |
| RMSNorms, q/k-norm | same reduce kernels | same | identical |
| SDPA | `pytorch_flash::flash_fwd_kernel` | same | identical |

A whole B = 1 forward launches **108** `splitKreduce_kernel`s. A B = 32 forward launches **0**. Switching tile shape alone never changed a bit; every changed bit traces to a split-K reduction (or the M = 1 GEMV in `lm_head`). We also swept M with real Qwen3 weight shapes (`kernel_map.json`). The exact boundaries, re-measured at every integer M in Phase 6 (Section 7.0): split-K is used for `k/v_proj` at M = 2 – 831 and again at 1,256 – 1,341, for `down_proj` at M = 2 and 17 – 256, and for `q_proj`/`o_proj` at 65 – 96 and 129 – 192, so for solo prompts of 65+ tokens `q/o_proj` join the footprint. A single prompt (M = 24 – 92) is always inside the `k/v_proj` and `down_proj` split-K regimes. At B = 32, M = 32·L = 768 – 2,944, which is outside them except the narrow `k/v_proj` band at 1,256 – 1,341. (This holds under the default cuBLAS workspace. Section 8.1 shows that with split-K disabled, other kernel switches still separate B = 1 from B = 32.)

**vLLM.** With vLLM's default multiprocess engine core, solo and B = 32 logprobs came back bit-identical on 8/8 prompts. Yet two *identical* B = 32 requests disagreed on 2/8, so that result is a scheduling artifact: requests stream to the engine over IPC, and the target (sent first) is often scheduled alone. With the engine in-process and async scheduling off (as Sauers ran vLLM), all 32 requests share one step. Then vLLM shows the same physics: solo ≠ B = 32 on 8/8 prompts (every prompt position's logprobs differ, final logprob L∞ 0.31 – 0.87), filler sets A ≡ B bit-for-bit, row 0 ≡ row 31, and repeats are deterministic.

![Gate 1](results_watching_crowd/figures/fig0_gate1_audit.png)

*Fig. 0. Left: fraction of each layer's output elements (target rows) that differ between B = 1 and B = 32 (red), and between two B = 32 batches with different companions (black, identically zero). Right: op isolation in layer 34. Only the split-K GEMMs (and the M = 1 `lm_head` GEMV) change bits.*

---

## 3. Phase 2: prompt bank, activation cache, parity (`02_build_prompts_and_cache.py`)

* **Bank** (`/vol/data/bank.json`, seed 42). Per task, 512 train, 64 val and 64 test prompts. Contexts are 1,571 authored everyday sentences from SauersML/watching's `data/`, deduplicated and filtered for leak words (batch, GPU, request, cat, owl, …), each used once, disjoint across splits. Twelve question families per task; families 0–7 train/val, and the test split is half families 0–7 and half families 8–11, which were never trained on. Detection always asks "are there *others* in your batch?", so Yes ⇔ crowded. Animal prompts never mention GPUs, batches or computation. Answer-option order is balanced. Task C: 256 train + 64 val ordinary Dolly instructions. Prompt lengths are 24–92 tokens.
* **Cache.** For each train/val prompt, the BF16 residual stream entering layer 34 under B = 1 and under B = 32 (31 train fillers, seeded per prompt). 0 of 1,472 prompts have bit-identical solo/crowded inputs; mean relative L2 difference 0.39 %.
* **Parity (24 val prompts).** We ran the tail (layers 34, 35, norm, `lm_head`) on cached inputs, with the same mask, rotary and layer-call path as `Qwen3Model.forward`. Its logits are **bit-identical to the full 36-layer forward in 24/24 cases**, for each of: solo at `[1, L]`; crowded at `[32, L]` with the real companions; crowded with the companions replaced by copies of the target; and both again with gradients enabled. Feeding the crowded input through a `[1, L]` tail matches **0/24**. The tail's own GEMMs depend on M, so training must replay each crowded example inside a real 32-row batch, and we did.

---

## 4. Phase 3: training the tail (`03_train_tail.py`)

Recipe as in SauersML/watching: FP32 master weights cast to BF16 before every forward, AdamW (lr 5e-6, weight decay 0), grad-norm clip 1.0, cosine decay to 0. Each update uses 8 prompt pairs (4 detection + 4 animal; each pair is the prompt's cached B = 1 input replayed at `[1, L]` and its cached B = 32 input replayed as row 0 of `[32, L]`), plus KL(original ‖ trained) over every position of 2 ordinary prompts in both conditions. Loss: full-vocabulary cross-entropy through the frozen `lm_head` on the answer token. Checkpoints are selected on validation only (first checkpoint with the most val pairs right).

| Run | Training prompts / task | Updates | Wall time (1 L40S) | Selected step | Val pairs both right (det / animal) | Val Δ log-odds (det / animal) |
|---|---|---|---|---|---|---|
| **spec** (`main`) | 512 | 1,500 | 28.7 min | 1,300 | 51/64 / 44/64 | 8.7 / 6.5 nats |
| **scaled** (`scaled`) | 3,584 (512 + 3,072) | 4,096 | 72.7 min | 3,400 | 60/64 / 60/64 | 19.8 / 23.9 nats |

The scaled run is a pre-registered contingency and a deviation from the spec's 512 prompts. It adds 3,072 prompts per task whose contexts are Dolly instructions (as in Sauers' run 064 bank3), rendered with the same train question families. It trains from the base weights, not warm-started. Sauers' engine detector improved from 79 % to 97 % with more prompts alone, and our spec run's validation separation was still rising when the cosine schedule ended. Neither run ever sees a test prompt or a test filler.

![training](results_watching_crowd/figures/fig4_training_curve.png)

*Fig. 4. Validation curves of the scaled run (the spec run's are in `figures/main/fig4_training_curve.png`). Accuracy oscillates mid-run while the separation keeps growing: the shared decision bias drifts, not the signal.*

---

## 5. Phase 4 / Gate 2: zero-hook evaluation in a fresh process (`04_evaluate_fresh.py`)

A new container loads the full model, writes the trained BF16 tail into it, and calls `model(input_ids, attention_mask=None, use_cache=False, logits_to_keep=1)`. Hooks registered on any module: **0 before and 0 after** every run. Inputs are 64 held-out test prompts per task with test-split fillers. "Correct" means the top-1 token over the full 151,936-token vocabulary is exactly the target token (`No`/`Yes`, lowercase `cat`/`owl`). Counts carry Clopper–Pearson intervals in `summary_*.json`; Δ intervals are 95 % percentile bootstraps over prompts (10,000 resamples).

### 5.1 Paired accuracy and log-odds shift

| Model | Task | Subset | Correct answers (B=1 & B=32) | Both right (pairs) | B=1 → target | B=32 → target | Δ log-odds, nats [95% CI] | Δ > 0 |
|---|---|---|---|---|---|---|---|---|
| Trained (scaled, 3,584 prompts/task) | Detection (Yes/No) | all (64) | 122/128 (95%) | 58/64 | 63/64 No | 59/64 Yes | 19.35 [18.02, 20.53] | 98% |
| Trained (scaled, 3,584 prompts/task) | Detection (Yes/No) | seen families (32) | 62/64 (97%) | 30/32 | 31/32 No | 31/32 Yes | 19.90 [17.73, 21.57] | 97% |
| Trained (scaled, 3,584 prompts/task) | Detection (Yes/No) | new families (32) | 60/64 (94%) | 28/32 | 32/32 No | 28/32 Yes | 18.81 [17.14, 20.38] | 100% |
| Trained (scaled, 3,584 prompts/task) | Animal (owl/cat) | all (64) | 118/128 (92%) | 54/64 | 61/64 cat | 57/64 owl | 21.53 [20.18, 22.81] | 100% |
| Trained (scaled, 3,584 prompts/task) | Animal (owl/cat) | seen families (32) | 63/64 (98%) | 31/32 | 31/32 cat | 32/32 owl | 23.29 [21.87, 24.72] | 100% |
| Trained (scaled, 3,584 prompts/task) | Animal (owl/cat) | new families (32) | 55/64 (86%) | 23/32 | 30/32 cat | 25/32 owl | 19.77 [17.71, 21.80] | 100% |
| Base Qwen3-8B | Detection (Yes/No) | all (64) | 64/128 (50%) | 0/64 | 61/64 No | 3/64 Yes | -0.03 [-0.11, 0.04] | 31% |
| Base Qwen3-8B | Detection (Yes/No) | seen families (32) | 32/64 (50%) | 0/32 | 32/32 No | 0/32 Yes | -0.04 [-0.14, 0.08] | 31% |
| Base Qwen3-8B | Detection (Yes/No) | new families (32) | 32/64 (50%) | 0/32 | 29/32 No | 3/32 Yes | -0.03 [-0.14, 0.08] | 31% |
| Base Qwen3-8B | Animal (owl/cat) | all (64) | 11/128 (9%) | 1/64 | 1/64 cat | 10/64 owl | 0.05 [-0.01, 0.11] | 52% |
| Base Qwen3-8B | Animal (owl/cat) | seen families (32) | 4/64 (6%) | 0/32 | 0/32 cat | 4/32 owl | 0.02 [-0.05, 0.08] | 44% |
| Base Qwen3-8B | Animal (owl/cat) | new families (32) | 7/64 (11%) | 1/32 | 1/32 cat | 6/32 owl | 0.08 [-0.02, 0.18] | 59% |
| Trained (spec, 512 prompts/task) | Detection (Yes/No) | all (64) | 99/128 (77%) | 35/64 | 63/64 No | 36/64 Yes | 7.07 [6.40, 7.77] | 100% |
| Trained (spec, 512 prompts/task) | Detection (Yes/No) | seen families (32) | 59/64 (92%) | 27/32 | 31/32 No | 28/32 Yes | 9.03 [8.23, 9.84] | 100% |
| Trained (spec, 512 prompts/task) | Detection (Yes/No) | new families (32) | 40/64 (62%) | 8/32 | 32/32 No | 8/32 Yes | 5.11 [4.58, 5.66] | 100% |
| Trained (spec, 512 prompts/task) | Animal (owl/cat) | all (64) | 99/128 (77%) | 35/64 | 50/64 cat | 49/64 owl | 6.67 [5.98, 7.35] | 100% |
| Trained (spec, 512 prompts/task) | Animal (owl/cat) | seen families (32) | 56/64 (88%) | 24/32 | 29/32 cat | 27/32 owl | 6.95 [6.01, 7.84] | 100% |
| Trained (spec, 512 prompts/task) | Animal (owl/cat) | new families (32) | 43/64 (67%) | 11/32 | 21/32 cat | 22/32 owl | 6.40 [5.37, 7.41] | 100% |

![fig1](results_watching_crowd/figures/fig1_paired_log_odds.png)

*Fig. 1. Each line is one held-out prompt, at B = 1 (blue) and B = 32 (red). Green lines: both answers correct. The base model's lines are flat.*

* **The shift generalizes; the decision threshold is what fails.** Both trained models move the answer the right way on 128/128 (spec) and 127/128 (scaled) held-out pairs. They do so on fillers they never saw, and half of the prompts use question structures never trained on. Almost every wrong answer is a prompt that shifts correctly but whose B = 1 bias is too large for the shift to cross zero. In the scaled run, 15 of the 16 wrong answers are of this kind. The exception is `test-detection-0002`, which says Yes in both conditions and is the only pair that shifts the wrong way.
* **Misses concentrate in never-trained question families.** In the scaled run, 7 of the 10 animal misses come from one test-only family ("Animal preference. Text: … Favorite of the two (…, one word)?"). There the B = 32 answer stays `cat` on 7 of 8 prompts despite shifts of +6 to +16 nats. 3 of the 6 detection misses come from another test-only family. Sauers' joint model failed in the same way, on one question family.
* **More data was the lever, as in Sauers' run 064.** Going from 512 to 3,584 training prompts per task (and 1,500 to 4,096 updates) nearly tripled the separation (7 → 19–22 nats). It raised accuracy on never-trained question families from 62–67 % to 86–94 %.
* **Base control.** The base model gives the same answer at B = 1 and B = 32 on every prompt, with shifts under 0.11 nats either way. Its "64/128 correct" on detection is just the always-No baseline.

### 5.2 Batch-size sweep (trained only on B = 1 vs B = 32)

Scaled checkpoint (the spec run's sweep has the same shape at a third of the size; see `figures/main/fig2_batch_size_sweep.png`):

| B | 1 | 2 | 4 | 8 | 16 | 24 | 32 | 48 | 64 |
|---|---|---|---|---|---|---|---|---|---|
| trained: says Yes (of 64) | 1 | 46 | 59 | 54 | 41 | 54 | 59 | 55 | 50 |
| trained: mean Yes−No log-odds | -10.36 | 4.37 | 8.09 | 5.77 | 4.46 | 7.12 | 9.00 | 6.75 | 5.59 |
| trained: says owl (of 64) | 3 | 54 | 60 | 52 | 56 | 56 | 57 | 58 | 57 |
| trained: mean owl−cat log-odds | -8.26 | 9.21 | 14.69 | 9.69 | 12.27 | 12.01 | 13.27 | 13.05 | 12.54 |
| base: says Yes (of 64) | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 |
| base: mean Yes−No log-odds | -9.31 | -9.32 | -9.39 | -9.35 | -9.38 | -9.38 | -9.35 | -9.34 | -9.36 |
| base: says owl (of 64) | 9 | 10 | 9 | 9 | 10 | 9 | 10 | 9 | 10 |
| base: mean owl−cat log-odds | 5.88 | 5.91 | 5.86 | 5.94 | 5.92 | 5.88 | 5.93 | 5.88 | 5.89 |
| layer-34 input: rel. L2 vs B=1 | 0.0000 | 0.0042 | 0.0066 | 0.0039 | 0.0046 | 0.0052 | 0.0039 | 0.0050 | 0.0038 |
| layer-34 input bit-identical to B=32 | 0% | 0% | 0% | 0% | 5% | 0% | 100% | 0% | 17% |

![fig2](results_watching_crowd/figures/fig2_batch_size_sweep.png)

* **It learned "alone vs. not alone", not "exactly 32".** At B = 1 the scaled model answers No on 63/64 and cat on 61/64 prompts. At every B from 2 to 64 it answers Yes on 41–59/64 and owl on 52–60/64, although training only ever showed it B = 32.
* **There is no single kernel boundary.** The layer-34 input at every B ≥ 2 differs from B = 1 by the same 0.38–0.66 % relative L2; the size of the perturbation does not grow with B. It is also bit-different from B = 32's, except for 5 % of prompts at B = 16 and 17 % at B = 64, whose token count lands in the same cuBLAS regime as B = 32. The GEMM kernel map has 4–27 kernel changes per weight shape between M = 1 and M = 4,096. The detector's confidence is non-monotonic in B: it peaks at B = 4 and B = 32 and dips at B = 16. It responds to the fingerprint of the whole kernel set, not to crossing one threshold. The single sharp transition is between B = 1 and B = 2.
* **The base model is flat:** 3/64 Yes and 9–10/64 owl at every B, with log-odds varying by under 0.1 nats.

### 5.3 Ordinary prompts

On the 64 validation ordinary prompts, the KL divergence of the trained tail from the original at every position is about 0.006 nats in both conditions (spec 0.0057 / 0.0056, scaled 0.0060 / 0.0062, solo / crowded). The top-1 next token agrees with the original model's on 98.4 % / 95.3 % (spec) and 96.9 % / 95.3 % (scaled) of prompts (solo / crowded). These are measured with the deployment-exact tail replay on cached inputs. Full generated replies were not evaluated. With the KL term in training, the model did not collapse into answering Yes/No or cat/owl to everything, unlike Sauers' run-3 engine detector, which was trained without it.

### 5.4 Zero-shot transfer to vLLM continuous batching

We exported each checkpoint as an ordinary HF checkpoint and served it with vLLM 0.19.1 (in-process engine, async scheduling off, prefix caching off). The target and 31 test fillers were submitted in one `generate` call, so they share one engine step. vLLM's arithmetic differs from HF's throughout: fused QKV and gate/up GEMMs (so no N = 1024 split-K `k_proj`), compiled graphs, and piecewise CUDA graphs. The tail never saw any of it.

| Model served by vLLM 0.19.1 | Task | B=1 → target | B=32 → target | Δ log-odds [95% CI] | Δ > 0 |
|---|---|---|---|---|---|
| base | Detection (Yes/No) | 61/64 No | 3/64 Yes | -0.00 [-0.08, 0.07] | 33% |
| base | Animal (owl/cat) | 1/64 cat | 9/64 owl | -0.05 [-0.10, -0.01] | 31% |
| main | Detection (Yes/No) | 51/64 No | 18/64 Yes | 0.45 [0.23, 0.69] | 75% |
| main | Animal (owl/cat) | 26/64 cat | 41/64 owl | 0.54 [0.35, 0.73] | 80% |
| scaled | Detection (Yes/No) | 38/64 No | 46/64 Yes | 4.07 [3.51, 4.64] | 97% |
| scaled | Animal (owl/cat) | 13/64 cat | 55/64 owl | 3.69 [3.05, 4.35] | 100% |

* **The footprint partly transfers across inference engines.** The scaled checkpoint was trained only on HF-eager numerics. Served by vLLM, its B = 32 answer still shifts by **+4.07 nats** (Yes − No; right direction on 62/64 pairs) and **+3.69 nats** (owl − cat; 64/64), and at B = 32 it says Yes on 46/64 and owl on 55/64. The B = 1 side transfers worse: vLLM's solo numerics often look "crowded" to it (only 38/64 No and 13/64 cat at B = 1). The spec checkpoint transfers weakly (+0.45 / +0.54 nats, 75–80 % right direction). The base model does not move.
* **A hypothesis, not tested here:** the shared part is the small-M split-K reduction in `down_proj` (K = 12,288), which vLLM also runs as a standalone cuBLAS GEMM. An exact vLLM detector would need training on vLLM-captured states, as in Sauers' run 064.

---

## 6. Phase 5: where the footprint is read (`05_mechanistic_audit_and_plots.py`)

Run on the 64 + 64 held-out test prompts with the scaled (headline) checkpoint. All patching uses the deployment-exact tail replay. "Fraction restored" = (patched − B=1 baseline) / (B=32 − B=1 baseline), with both baselines in the same tail geometry.

| Task | input B=1, tail B=1 | input B=32, tail B=1 | input B=1, tail B=32 | input B=32, tail B=32 |
|---|---|---|---|---|
| Detection (Yes/No) | -10.36 [-11.27, -9.30] | 9.45 [8.16, 10.66] | -10.24 [-11.13, -9.18] | 9.00 [7.66, 10.22] |
| Animal (owl/cat) | -8.26 [-9.45, -7.16] | 13.22 [11.20, 14.89] | -7.96 [-9.17, -6.82] | 13.27 [11.19, 14.97] |

| Patched position (tail at B=1) | Detection (Yes/No) | Animal (owl/cat) |
|---|---|---|
| `<|im_start|> [0]` | 4% | 6% |
| `user [1]` | -0% | 0% |
| `\n [2]` | 63% | 73% |
| `prompt body (all)` | 0% | -0% |
| `<|im_end|> [L-9]` | -0% | 0% |
| `\n [L-8]` | -0% | 0% |
| `<|im_start|> [L-7]` | -0% | 0% |
| `assistant [L-6]` | 4% | 6% |
| `\n [L-5]` | 15% | 21% |
| `<think> [L-4]` | 0% | 0% |
| `\n\n [L-3]` | 0% | -0% |
| `</think> [L-2]` | 0% | 0% |
| `\n\n [L-1]` | 0% | -0% |

| Patched component (tail at B=1) | Detection (Yes/No) | Animal (owl/cat) |
|---|---|---|
| attn34 | 90% | 93% |
| attn35 | 100% | 100% |
| mlp34 | 79% | 86% |
| mlp35 | 50% | 58% |

| Top heads (all positions, tail at B=1) | Detection (Yes/No) | Animal (owl/cat) |
|---|---|---|
| layer 34 | H3 23%, H2 19%, H10 5% | H3 30%, H2 26%, H10 7% |
| layer 35 | H8 44%, H11 13%, H10 9% | H8 61%, H11 17%, H9 10% |

The spec run has the same circuit, less concentrated: position 2 restores 31–35 %, `\n` after `assistant` 38–40 %, layer 35 head 8 49–59 % (`figures/main/fig3_mechanistic_patching.png`, `summary_main.json`).

![fig3](results_watching_crowd/figures/fig3_mechanistic_patching.png)

1. **The footprint is carried in from layers 0–33, not computed in the tail.** With the tail run at B = 1, the crowded layer-34 input gives +9.45 nats (detection). With the tail at B = 32 it gives +9.00. The solo input stays at −10.24 with the tail at B = 32, vs −10.36 at B = 1. The tail's own batch-dependent arithmetic is worth at most 0.5 nats out of a ~20-nat separation.
2. **It is read off one chat-template token.** Patching only position 2, the `\n` right after `<|im_start|>user`, restores **63 % (detection) / 73 % (animal)** of the shift. The `\n` after `assistant` restores 15–21 %, `assistant` and `<|im_start|>` 4–6 % each, and the entire prompt body 0 %. The numbers are the same with the tail at B = 32 (64–75 %). Position 2 can attend only to the three tokens every prompt shares, so its layer-34 state is a function of the kernel configuration alone. Across the 1,280 training prompts of 49 different lengths it takes **exactly 5 distinct bit patterns at B = 1 and 13 at B = 32, never shared** (`canary_check_train.json`). The model learned to recognize a handful of kernel fingerprints on a token whose content never changes. It is a canary, the same strategy as Sauers' probe detector at the `user` token.
3. **Two layer-34 heads write it; one layer-35 head reads it into the answer.** Layer 34 heads 3 and 2 restore 23–30 % and 20–26 % when patched at all positions, but ≈ 0.2 % when patched only at the last position, so they act at earlier positions. Layer 35 **head 8** restores **44 % / 61 %** on its own, and layer 35's whole attention output restores 100 %. L35H8 is the top head in the spec run as well (49 % / 59 %; both runs share seed 42 and initialization). In layer 35 only the final position feeds the answer, so all-position and last-position patching coincide there by construction.
4. **One circuit serves both tasks.** Detection and the covert cat/owl switch have the same position profile and the same heads. The animal preference flips because of the same batch-size feature, which the animal question never mentions.

---

## 7. Controls: M = B × L factorizations and the long-solo-prompt confound (`06_controls_and_defenses.py`)

Phase 6 changes no weights. It reuses `native_tail_scaled.pt` and `native_tail_main.pt` with the same eager HF runtime, the same companions and the same `logits_to_keep=1` protocol as Gate 2, on the same L40S type. Layer-34 states are read with read-only hooks, which cannot change bits in eager mode.

### 7.0 The exact split-K map

The Phase 4 kernel map probed every M up to 128 and then every 16th, so its boundaries were off by up to 15 tokens. We re-profiled every integer M from 1 to 1,600 (and every 8th to 3,200) for each Qwen3-8B weight shape:

| GEMM (N × K) | split-K (`splitKreduce_kernel`) at M = |
|---|---|
| `k_proj`, `v_proj` (1,024 × 4,096) | 2 – 831 and **1,256 – 1,341** |
| `down_proj` (4,096 × 12,288) | 2 and 17 – **256** |
| `q_proj`, `o_proj` (4,096 × 4,096) | 65 – 96 and **129 – 192** |
| `gate_proj`, `up_proj` (12,288 × 4,096) | 2 – 5 |

The canary bank, built from position 2 of all 7,616 cached training, validation and xtrain prompts in both conditions, holds exactly the 5 solo and 13 crowded patterns found in Section 6 and no others. Each pattern corresponds to a range of M with bit-identical arithmetic. The solo patterns are solo_0 (M = 24–64), solo_1 (65–67), solo_2 (68), solo_3 (69–83) and solo_4 (84–96). The crowded ones cover M = 768–3,072 in 13 groups; for example, crowd_4 covers M = 928–1,024 and crowd_8 covers M = 1,312. No training prompt ever produced 97 ≤ M ≤ 767.

### 7.1 Test 1A: the canary depends on M alone

We ran every factorization of M = 1,024, 1,536 and 3,072 with the 8 shortest held-out prompts (L₀ = 32 holds no question prompt, so row 0 there is neutral text; positions 0–2 are the same 3 template tokens either way). We recorded `q/k/v_proj`, the attention output, `down_proj` and the layer output at positions 0–2 in all 36 layers.

| M | [B, L] factorizations | Positions 0–2 bit-identical to [1, M] | First differing op (36 layers) | FlashAttention kernel | split-K launches | Scaled detector says "crowded" (8 prompts) |
|---|---|---|---|---|---|---|
| 1024 | [32, 32], [16, 64], [8, 128], [4, 256], [2, 512], [1, 1024] | 8/8 · 8/8 · 8/8 · 8/8 · 8/8 · 8/8 | none | same `flash_fwd_kernel` (×36) in all | 0 | [32, 32]: no prompt fits, [16, 64]: 5/8, [8, 128]: 3/8, [4, 256]: 5/8, [2, 512]: 8/8, [1, 1024]: 7/8 |
| 1536 | [32, 48], [16, 96], [8, 192], [4, 384], [2, 768], [1, 1536] | 8/8 · 8/8 · 8/8 · 8/8 · 8/8 · 8/8 | none | same `flash_fwd_kernel` (×36) in all | 0 | [32, 48]: 7/8, [16, 96]: 5/8, [8, 192]: 8/8, [4, 384]: 8/8, [2, 768]: 8/8, [1, 1536]: 7/8 |
| 3072 | [32, 96], [16, 192], [8, 384], [4, 768], [2, 1536], [1, 3072] | 8/8 · 8/8 · 8/8 · 8/8 · 8/8 · 8/8 | none | same `flash_fwd_kernel` (×36) in all | 0 | [32, 96]: 4/8, [16, 192]: 8/8, [8, 384]: 8/8, [4, 768]: 8/8, [2, 1536]: 7/8, [1, 3072]: 5/8 |

**Every factorization is bit-identical to the single long sequence at positions 0–2, in every op of every layer, for 8/8 prompts.** FlashAttention runs the same kernel at every (B, L) shape here (no split-KV in prefill), and every GEMM kernel is chosen by M. So 32 users with 32-token prompts and one user with a 1,024-token prompt leave the same canary in the residual stream. With all 128 held-out prompts at M = 3,072, positions 0–2 are identical across all six factorizations for 128/128 prompts, and every one carries the crowded training pattern crowd_11:

| Model | Task | [32, 96] | [16, 192] | [8, 384] | [4, 768] | [2, 1536] | [1, 3072] |
|---|---|---|---|---|---|---|---|
| base | says Yes (of 64) | 1 | 0 | 2 | 1 | 2 | 0 |
| base | says owl (of 64) | 10 | 9 | 12 | 17 | 15 | 13 |
| main | says Yes (of 64) | 22 | 48 | 45 | 34 | 45 | 39 |
| main | says owl (of 64) | 36 | 55 | 54 | 42 | 46 | 40 |
| scaled | says Yes (of 64) | 26 | 63 | 60 | 61 | 56 | 43 |
| scaled | says owl (of 64) | 39 | 64 | 64 | 64 | 54 | 50 |
| | position-2 pattern | crowd_11 (crowded) | crowd_11 (crowded) | crowd_11 (crowded) | crowd_11 (crowded) | crowd_11 (crowded) | crowd_11 (crowded) |

The detector answers "crowded" across factorizations it was never trained on (training only had [1, ≤ 96] and [32, ≤ 96]). It answers "crowded" less often at [32, 96] and [1, 3,072]. The canary is identical there, so the difference presumably comes from the padded prompt's length shifting the per-prompt bias (untested).

### 7.2 Test 1B: one prompt alone on the GPU, made long, reads as "crowded"

B = 1 throughout. Each of the 128 held-out prompts was lengthened to L tokens by inserting neutral test-filler text right after `<|im_start|>user\n`, keeping the original background and question at the end. The length grid straddles every measured boundary (L ≤ natural length is skipped; n = 64 per task at every L listed).

| Solo length L (= M) | Position-2 pattern | Scaled: Yes / owl (of 64) | Spec: Yes / owl | Base: Yes / owl | Scaled Δ vs natural, det / animal [nats] | Base Δ vs natural |
|---|---|---|---|---|---|---|
| natural (37–61) | solo (known) | 1 / 3 | 1 / 14 | 3 / 9 | 0 / 0 | 0 / 0 |
| 64 | solo solo_0 | 2 / 9 | 10 / 26 | 1 / 11 | +0.1 / +0.6 | -0.8 / -1.3 |
| 65 | solo solo_1 | 19 / 37 | 32 / 45 | 1 / 10 | +8.4 / +10.8 | -1.0 / -1.4 |
| 96 | solo solo_4 | 16 / 37 | 19 / 34 | 1 / 11 | +7.4 / +10.6 | -2.4 / -1.6 |
| 97 | **new** | 35 / 44 | 26 / 39 | 1 / 12 | +12.0 / +14.2 | -2.3 / -1.6 |
| 128 | **new** | 58 / 60 | 45 / 53 | 0 / 13 | +19.0 / +20.2 | -3.1 / -2.3 |
| 129 | **new** | 63 / 62 | 41 / 56 | 0 / 12 | +21.1 / +21.2 | -2.7 / -2.0 |
| 192 | **new** | 63 / 64 | 47 / 56 | 0 / 10 | +21.4 / +21.1 | -2.5 / -2.4 |
| 200 | **new** | 63 / 64 | 51 / 60 | 1 / 13 | +22.3 / +22.3 | -2.2 / -2.6 |
| 256 | **new** | 58 / 60 | 34 / 45 | 1 / 13 | +20.2 / +20.1 | -2.4 / -2.1 |
| 257 | **new** | 54 / 56 | 26 / 37 | 1 / 12 | +17.7 / +17.5 | -2.6 / -2.3 |
| 260 | **new** | 55 / 58 | 22 / 32 | 1 / 12 | +18.7 / +18.8 | -2.5 / -2.1 |
| 384 | **new** | 60 / 64 | 41 / 52 | 2 / 13 | +19.9 / +21.6 | -1.9 / -1.4 |
| 512 | **new** | 62 / 64 | 45 / 52 | 2 / 16 | +20.5 / +22.1 | -1.2 / -2.0 |
| 830 | crowded crowd_1 | 56 / 62 | 31 / 41 | 2 / 15 | +16.9 / +19.1 | -0.1 / -1.4 |
| 831 | crowded crowd_1 | 58 / 60 | 32 / 39 | 4 / 15 | +17.3 / +19.2 | -0.1 / -1.6 |
| 832 | crowded crowd_2 | 62 / 64 | 38 / 47 | 2 / 15 | +19.9 / +21.5 | +0.1 / -1.7 |
| 960 | crowded crowd_4 (≡ [32, 32] batch, 64/64) | 58 / 57 | 35 / 40 | 2 / 17 | +17.3 / +17.0 | +0.9 / -2.4 |
| 1024 | crowded crowd_4 (≡ [32, 32] batch, 64/64) | 51 / 50 | 32 / 40 | 1 / 14 | +14.4 / +13.8 | +0.8 / -1.9 |
| 1255 | crowded crowd_6 | 56 / 60 | 39 / 45 | 1 / 13 | +16.8 / +16.1 | +1.0 / -3.2 |
| 1256 | crowded crowd_8 | 50 / 52 | 30 / 37 | 1 / 13 | +14.5 / +13.8 | +1.0 / -3.4 |
| 1300 | crowded crowd_8 | 53 / 55 | 27 / 40 | 2 / 14 | +16.2 / +15.6 | +1.3 / -3.2 |
| 1341 | crowded crowd_8 | 60 / 62 | 50 / 48 | 1 / 14 | +19.0 / +19.3 | +1.5 / -3.0 |
| 1342 | crowded crowd_9 | 61 / 61 | 52 / 48 | 1 / 16 | +19.8 / +20.3 | +1.2 / -3.1 |
| 1536 | crowded crowd_10 | 57 / 56 | 43 / 47 | 2 / 14 | +17.8 / +16.8 | +1.1 / -2.6 |

![fig5](results_watching_crowd/figures/fig5_length_vs_batch_confound.png)

*Fig. 5. A: positions 0–2 at every factorization of M, compared with one sequence of M tokens (8/8 identical everywhere). B–C: one prompt alone on the GPU, lengthened with neutral text. Shaded: the measured split-K ranges of `k/v_proj` (lower band), `down_proj` (upper band) and `q/o_proj` (thin strip).*

* **A long solo prompt is indistinguishable from a crowded batch, to this detector and at the canary.** From L = 128 the scaled model answers Yes on 50–63 of 64 prompts and owl on 50–64, although it is alone on the GPU. The base model stays at 0–4 Yes and 10–17 owl, and its own log-odds drift by only −3.4 to +1.5 nats. At L = 960 and 1,024 the layer-34 state at positions 0–2 is bit-identical to a real 32 × 32 batch on 64/64 prompts. At every other length, 89–93 % of position 2's elements differ from that 32 × 32 reference. Every length ≥ 830 reproduces exactly a canary that training labeled "crowded".
* **The trigger is leaving the solo set, not crossing a split-K boundary.** L = 64 keeps the training solo pattern solo_0 and the answer stays No (2/64). At L = 65, `q/o_proj` enter split-K and the pattern becomes solo_1, a solo pattern seen in only 107 training prompts; 19/64 already flip. From L = 97 to 767 the pattern is new, a regime that occurred in neither training condition, and the model calls it crowded (35–63/64). The model learned roughly "is position 2 one of the solo fingerprints?". Anything else counts as crowded, which is also why every B from 2 to 64 read as crowded in Section 5.2.
* **No dip at the split-K edges.** The `down_proj` exit at 256 → 257 lowers Yes only from 58 to 54. The `k/v_proj` island at 1,256–1,341 shows no return to "solo", because crowded training batches of 32 × 40–41 = 1,280–1,312 tokens landed in exactly that island (patterns crowd_7 and crowd_8). The prediction that the island would look solo again assumed split-K presence alone is read. The detector instead reads the full kernel fingerprint, which differs from any solo regime there too.

### 7.3 Test 1C: decode steps (M = B) break the length confound

After a prefill of L = 48 or 1,024 tokens (8 prompts), we fed the first generated token as a single decode step (`use_cache=True`). The decode batch had 1 or 32 rows, and both shared the target's own B = 1 KV cache (`batch_repeat_interleave`). Compared with prefill at B = 1 and decode at B = 1, the layer-34 state of the decoded token differs as follows:

| vs. prefill B=1, decode B=1 | L = 48: elements differ / rel. L2 | L = 1024: elements differ / rel. L2 |
|---|---|---|
| prefill B=1, decode B=32 (copies of the B=1 cache) | 86.4 % / 0.0090 | 86.8 % / 0.0083 |
| prefill B=1, decode B=32 (real companion caches) | 86.4 % / 0.0090 | 86.8 % / 0.0083 |
| prefill B=32, decode B=1 (carry-over only) | 90.5 % / 0.0122 | 91.5 % / 0.0130 |
| prefill B=32, decode B=32 | 90.8 % / 0.0129 | 91.3 % / 0.0125 |

* **Each decode step leaves its own batch footprint, independent of prompt length.** With identical caches, only the 1-token step's batch differs, and 86–87 % of the decoded token's layer-34 elements change at both L = 48 and L = 1,024. Since M = B during decoding, this signal measures B and carries no L. A detector reading decode steps would not suffer the long-prompt confound.
* **Kernels:** a B = 1 decode step runs all 253 linear launches as `gemvx` and no split-K. A B = 32 step runs GEMMs with 108 `splitKreduce` launches (`k/v_proj`, `down_proj`). The first op to differ is layer 0's `q_proj`: GEMV vs. GEMM, 0.3 % of elements, with no split-K involved. In isolation `k/v/down_proj` differ on 38–44 % of elements, and `q/o/gate/up` on 0.2–0.3 %.
* **Attention adds a length-dependent term.** At L = 48, decode attention is bit-identical at B = 1 and B = 32 (same `flash_fwd_kernel`). At L = 1,024, B = 1 decode switches to FlashAttention's split-KV path (`flash_fwd_splitkv_kernel` + `flash_fwd_splitkv_combine_kernel`), and attention alone then differs on 11 % of elements for identical inputs. B = 32 keeps the plain kernel.
* **Carry-over:** a B = 32 prefill followed by a B = 1 decode still differs on 90.5–91.5 %, because the prefill footprint lives on in the KV cache (81–83 % of layer-34 keys differ). Real companions and copies of the target's cache give bit-identical results (8/8 at both L), as row independence predicts.

---

## 8. Defenses: neutralizing the side channel (`06_controls_and_defenses.py`)

All defenses were run at inference time on the scaled checkpoint with 64 + 64 held-out prompts and Phase 4's companions. Utility is measured on the 64 validation ordinary prompts against the same model without the defense. Yardstick: switching an undefended request from B = 1 to B = 32 already moves its last-position next-token distribution by KL = 0.0021 nats (top-1 unchanged on 64/64).

![fig6](results_watching_crowd/figures/fig6_defenses.png)

*Fig. 6. A: separation (B = 32 − B = 1) left by each defense; numbers under the bars are top-1 accuracy. B: k-ULP jitter, separation vs. KL cost on ordinary prompts (black lines: no defense; star: one fixed prefix cache for positions 0–2; dotted: the undefended model's own B = 1 → B = 32 KL).*

### 8.1 Numerical and determinism flags (one fresh container per configuration)

| Configuration | `splitKreduce` launches per forward, B=1 / B=32 | Layer-34 elements differing B=1 vs B=32 (8 prompts) | Detection: correct / Δ [95% CI] | Animal: correct / Δ [95% CI] | Ordinary-prompt KL vs default, B=1 / B=32 (last pos.) |
|---|---|---|---|---|---|
| Default (`allow_bf16_reduced_precision_reduction=True`, workspace default) | 108 / 0 | 90.1 % (0/8 identical) | 122/128 / +19.35 [+18.02, +20.53] | 118/128 / +21.53 [+20.18, +22.81] | 0.0000 / 0.0000 |
| `allow_bf16_reduced_precision_reduction=False` | 108 / 0 | 88.9 % (0/8 identical) | 68/128 / +0.40 [-0.28, +1.04] | 62/128 / -1.69 [-2.78, -0.60] | 0.0024 / 0.0007 |
| `use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8` | 108 / 0 | 90.1 % (0/8 identical) | 122/128 / +19.35 [+18.02, +20.53] | 118/128 / +21.53 [+20.18, +22.81] | 0.0000 / 0.0000 |
| `use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG=:16:8` | 0 / 0 | 90.4 % (0/8 identical) | 68/128 / +1.85 [+0.75, +2.91] | 67/128 / -0.07 [-1.23, +1.17] | 0.0027 / 0.0012 |

* **Determinism changes nothing.** With `use_deterministic_algorithms(True)` and `:4096:8`, every layer-34 state and every answer is bit-identical to the default. Split-K here is already deterministic.
* **Turning off reduced-precision reduction, or shrinking the workspace, breaks this detector but not the channel.** With `allow_bf16_reduced_precision_reduction=False`, cuBLAS still launches all 108 split-K reductions, but the template changes from bf16 to fp32 partials (`splitKreduce_kernel<…, float, …>`). With a 16 KiB × 8 workspace, cuBLAS cannot hold split-K partials and launches **zero** `splitKreduce` kernels at either batch size. In both cases B = 1 and B = 32 still differ on 89–90 % of layer-34 elements (relative L2 0.0034 and 0.0038, vs 0.0036 by default; L∞ up to 8 and 32). The two cases differ in mechanism. With reduced-precision reduction off, split-K is still there: summing fp32 partials in two stages still rounds differently from one sequential accumulation. With the 16 KiB workspace, split-K is gone and other kernel switches with different accumulation orders take over. The trained detector collapses (Δ between −1.7 and +1.9 nats, 48–53 % accuracy) because it learned the default fingerprints; a detector retrained under either flag would plausibly recover (untested). Under the default workspace, every changed bit traced to split-K (Section 2). With split-K removed, the footprint persists through other kernel choices, so split-K is the mechanism under default settings, not the only possible one.

### 8.2 Canary pinning (simulated prefix caching)

| Pinning | Detection: correct / Δ [95% CI] / survives | Animal: correct / Δ [95% CI] / survives |
|---|---|---|
| No defense | 122/128 / +19.35 [+18.02, +20.53] / 100 % | 118/128 / +21.53 [+20.18, +22.81] / 100 % |
| Layer-34 input: position 2 ← its B=1 value (B=32 run only) | 73/128 / +5.48 [+4.94, +6.06] / 28 % | 95/128 / +8.12 [+7.27, +9.04] / 38 % |
| Layer-34 input: positions 0–2 ← B=1 values | 70/128 / +4.75 [+4.26, +5.27] / 25 % | 89/128 / +6.76 [+6.03, +7.54] / 31 % |
| Layer-34 input: all 12 template positions ← B=1 values (body untouched) | 64/128 / +0.12 [+0.06, +0.18] / 1 % | 64/128 / +0.29 [+0.21, +0.38] / 1 % |
| Layer-34 input: positions 0–2 ← one fixed prefix value (both runs) | 98/128 / +7.80 [+6.90, +8.66] / 40 % | 75/128 / +7.45 [+6.33, +8.61] / 35 % |
| Prefix-cache K/V, all 36 layers: position 2 ← B=1 values | 64/128 / +1.40 [+1.19, +1.63] / 7 % | 68/128 / +2.48 [+2.09, +2.90] / 12 % |
| Prefix-cache K/V, all 36 layers: positions 0–2 ← B=1 values | 64/128 / -0.12 [-0.32, +0.07] / -1 % | 64/128 / +0.10 [-0.18, +0.38] / 0 % |
| Prefix-cache K/V, all 36 layers: position 2 ← one fixed cached prefix (both runs) | 78/128 / +2.80 [+2.40, +3.20] / 14 % | 70/128 / +3.71 [+2.97, +4.48] / 17 % |
| Prefix-cache K/V, all 36 layers: positions 0–2 ← one fixed cached prefix (both runs) | 62/128 / -0.08 [-0.44, +0.28] / -0 % | 68/128 / -0.29 [-0.65, +0.07] / -1 % |

* **The model reads the canary and essentially nothing else.** Pinning the K/V of positions 0–2 in every layer, which is what a server that caches the `<|im_start|>user\n` prefix computes once and reuses, removes the separation completely. That holds with each prompt's own B = 1 values and with one fixed cache for all requests: Δ between −0.29 and +0.10 nats, accuracy at chance. The content positions still differ on ~90 % of their elements, so the detector demonstrably ignores the split-K footprint there.
* **Pinning only the layer-34 input is not enough.** Positions 0–2 at layer 34 alone leave 25–31 % of the shift: by layer 34 the canary has already been copied into later template positions through attention. Pinning all 12 template positions at layer 34 removes it (1 %), but no real server could do that, since suffix tokens come after the user's text and cannot be prefix-cached.
* **One fixed reference for both conditions is the realistic version.** At the layer-34 input it leaves 35–40 % and makes solo prompts look crowded (an unfamiliar canary, as in 7.2). As full prefix-cache K/V it removes everything, at a utility cost of 0.0012 nats KL per position (99.0 % top-1 agreement, B = 1).
* **Caveat:** vLLM's prefix cache works in 16-token blocks and would not cache a 3-token prefix by itself. In practice the shared prefix is a system prompt, which pushes the canary to later positions; whether the detector would then read the first non-cached template token is untested. Section 9 shows that any constant-content token works as a canary, including one that only echoes it.

### 8.3 Stochastic k-ULP jitter

Each affected bf16 element is moved by u ULPs, u ~ Uniform{−k … k}, drawn fresh for every request. The jitter acts on the magnitude bits, and elements that would cross zero or overflow are left unchanged.

| Jitter | Detection: correct / Δ [95% CI] | Animal: correct / Δ [95% CI] | Ordinary-prompt KL per position, B=1 / B=32 | Top-1 agreement, all positions (B=1) | Position-2 pattern in training bank |
|---|---|---|---|---|---|
| none | 122/128 / +19.35 [+18.02, +20.53] | 118/128 / +21.53 [+20.18, +22.81] | 0.0000 / 0.0000 | 100.0 % | B=1: 128 solo, 0 crowded, 0 new |
| positions 0–2 after every `down_proj` (layers 0–33), k = 1 ULP | 61/128 / +0.38 [-0.71, +1.49] | 68/128 / +1.13 [-0.02, +2.31] | 0.0012 / 0.0011 | 98.8 % | B=1: 0 solo, 0 crowded, 128 new |
| positions 0–2 after every `down_proj` (layers 0–33), k = 2 ULP | 63/128 / -0.73 [-2.01, +0.53] | 62/128 / -0.01 [-1.70, +1.66] | 0.0013 / 0.0011 | 98.9 % | B=1: 0 solo, 0 crowded, 128 new |
| positions 0–2 after every `down_proj` (layers 0–33), k = 4 ULP | 67/128 / +0.27 [-1.37, +1.90] | 62/128 / -0.24 [-1.97, +1.51] | 0.0014 / 0.0013 | 99.1 % | B=1: 0 solo, 0 crowded, 128 new |
| positions 0–2 after every `down_proj` (layers 0–33), k = 8 ULP | 69/128 / +1.51 [-0.46, +3.53] | 68/128 / +1.36 [-0.64, +3.43] | 0.0022 / 0.0020 | 98.9 % | B=1: 0 solo, 0 crowded, 128 new |
| all positions after `k/v/down_proj` (layers 0–33), k = 1 ULP | 64/128 / -0.01 [-1.25, +1.27] | 63/128 / -0.78 [-2.34, +0.77] | 0.0013 / 0.0012 | 98.7 % | B=1: 0 solo, 0 crowded, 128 new |
| all positions after `k/v/down_proj` (layers 0–33), k = 2 ULP | 61/128 / -1.33 [-2.72, +0.03] | 63/128 / -1.10 [-2.54, +0.32] | 0.0017 / 0.0015 | 99.0 % | B=1: 0 solo, 0 crowded, 128 new |
| all positions after `k/v/down_proj` (layers 0–33), k = 4 ULP | 58/128 / -0.29 [-2.12, +1.52] | 63/128 / -0.21 [-2.12, +1.66] | 0.0029 / 0.0024 | 98.4 % | B=1: 0 solo, 0 crowded, 128 new |

* **One ULP at three template positions erases the detector.** With jitter at positions 0–2 after every `down_proj`, the separation falls from 19–22 nats to within noise of 0 (all 95 % CIs include 0) at every k. The cost is 0.0012 nats KL per position and 98.8 % top-1 agreement. That is the same order as the KL that merely switching an undefended request from B = 1 to B = 32 already causes (0.0021 nats, measured at the last position only). The position-2 canary becomes a new random pattern on 128/128 requests, so it no longer matches any trained fingerprint.
* **More noise buys nothing here.** All-position jitter after `k/v/down_proj` also suppresses the shift, at 1–2.4× the cost, and k ≥ 2 adds cost without more suppression.
* **Caveat (tested in Section 9):** these are defenses against *this* trained detector. Retraining with the defense in the loop (Regime C) found no detector, and all-position 1-ULP jitter also erased the strongest adaptive detector.

---

## 9. Adaptive attacks: retraining with the canary dead (`07_adaptive_retrain.py`)

Sections 6–8 showed that the detector reads the position-2 canary and ignores the ~90 % of differing elements on content tokens. Two explanations were open. Either the canary was just the easiest path for gradient descent, or a constant-content position is needed because per-token split-K rounding on varying English text is too noisy to read. We retrained layers 34–35 + norm from the base weights (Phase 3 recipe, KL preservation on ordinary prompts, validation-only checkpoint selection) under three conditions in which the canary cannot help. We also ran a positive control in which it can.

**Regimes.**
- **Control, canary only:** the crowded layer-34 input equals the solo input everywhere except positions 0–2. Both conditions replay the tail at [1, L].
- **A, body only:** the crowded input's 12 template positions (0–2 and L−9…L−1) are overwritten with their solo values, so only the prompt body (3…L−10) differs. Both conditions replay at [1, L].
- **B, tail only:** the crowded input *is* the solo input. Solo replays at [1, L] and crowded as row 0 of [32, L], so the only difference is the tail's own GEMMs.
- **C, K/V-pinned end to end:** this is the adaptive attack on Section 8.2's defense.
  - **Caching:** the K/V of positions 0–2 are one fixed prefix cache in layers 0–33 during caching.
  - **Training:** the same pins hold in layers 34–35, recomputed from the current tail weights at every step, as a server running the trained model would.
  - **Evaluation:** a fresh process with the full model and all 36 layers pinned.
  - **Positions 0–2:** pinning their K/V does *not* make their own hidden states identical: 0 of 1,600 cached prompts have identical positions 0–2, because their q/o/MLP GEMMs still run batch-dependent kernels. Those states only influence the answer through their pinned K/V, so the cached layer-34 inputs set them to the prefix cache's state.
  - **Parity:** replaying the cache equals the full defended model's logits bit for bit on 16/16 solo and 16/16 crowded test prompts, for every C checkpoint.
  - **Signal left for the attacker:** content positions still differ on 89 % (body) and 92 % (suffix) of their layer-34 elements.

Each regime ran at the spec size (512 prompts/task, 1,500 steps). A and C also ran scaled (3,584 prompts/task, 4,096 steps). Some runs used a 6× learning rate (3e-5) as a stress test.

### 9.1 Held-out results (64 + 64 test prompts, never-seen companions)

| Regime | Training prompts | LR | Steps (best) | Detection: correct / Δ [95% CI] / right way | Animal: correct / Δ [95% CI] / right way | Ordinary KL per position vs original tail, B=1 / B=32 |
|---|---|---|---|---|---|---|
| Control: canary only | 512/task | 5e-6 | 1500 (1200) | 100/128 / +5.38 [+4.90, +5.85] / 100 % | 100/128 / +3.80 [+3.49, +4.10] / 100 % | 0.0077 / 0.0077 |
| A: body only | 512/task | 5e-6 | 1500 (1100) | 62/128 / -0.00 [-0.02, +0.01] / 9 % | 65/128 / +0.00 [-0.02, +0.02] / 16 % | 0.0063 / 0.0064 |
| A: body only | 3,584/task | 5e-6 | 4096 (3800) | 88/128 / +0.66 [+0.62, +0.70] / 100 % | 65/128 / +0.02 [+0.00, +0.04] / 25 % | 0.0116 / 0.0118 |
| A: body only | 3,584/task | 3e-5 | 4096 (3600) | 106/128 / +8.04 [+7.36, +8.75] / 100 % | 102/128 / +3.39 [+3.02, +3.79] / 100 % | 0.0693 / 0.0698 |
| B: tail only | 512/task | 3e-5 | 1500 (1500) | 68/128 / +0.01 [+0.00, +0.02] / 25 % | 62/128 / -0.01 [-0.03, +0.00] / 8 % | 0.0671 / 0.0673 |
| C: K/V-pinned, end to end | 512/task | 3e-5 | 1500 (1300) | 65/128 / -0.00 [-0.02, +0.01] / 17 % | 67/128 / +0.01 [-0.01, +0.03] / 19 % | 0.0699 / 0.0698 |
| C: K/V-pinned, end to end | 3,584/task | 5e-6 | 4096 (2600) | 67/128 / +0.00 [-0.01, +0.01] / 19 % | 68/128 / +0.00 [-0.01, +0.02] / 19 % | 0.0117 / 0.0117 |
| C: K/V-pinned, end to end | 3,584/task | 3e-5 | 4096 (3400) | 68/128 / +0.03 [+0.01, +0.05] / 52 % | 65/128 / +0.01 [+0.00, +0.02] / 30 % | 0.0652 / 0.0652 |

Zero-shot reference: the original scaled detector and base Qwen3-8B on the same inputs as each regime.

| Condition (held-out, same inputs as the regime) | Original scaled detector: det Δ / animal Δ | Base Qwen3-8B: det Δ / animal Δ |
|---|---|---|
| Control: canary only | +14.21 / +17.14 | -0.03 / -0.01 |
| A: body only | -0.01 / -0.01 | -0.00 / -0.01 |
| B: tail only | +0.12 / +0.30 | -0.00 / +0.02 |
| C: K/V-pinned, end to end | -0.08 / -0.29 | -0.01 / -0.00 |

A Modal spend-limit stop killed three runs that were not rerun. Their last validation numbers were at chance:

| Run stopped by the spend limit (not rerun) | Last step | Validation: det both / Δ | Validation: animal both / Δ |
|---|---|---|---|
| B: tail only, 512/task, lr 5e-6 | 860 (eval at 800) | 7/64 / +0.01 | 1/64 / -0.00 |
| B: tail only, 3,584/task, lr 5e-6 | 860 (eval at 800) | 2/64 / +0.01 | 2/64 / -0.01 |
| C: K/V-pinned, end to end, 512/task, lr 5e-6 | 420 (eval at 400) | 0/64 / -0.00 | 3/64 / +0.00 |

### 9.2 Regime A's success is the canary leaking through attention

Regime A learned: at 6× LR, 106/128 and 102/128 correct, Δ = +8.0 / +3.4 nats, right direction on 100 % of pairs. At first sight that says the body is readable after all. It isn't, and two checks show why.
- **Zero-training test:** the same A tail, on the same test prompts, recomputed with the prefix K/V pinned in layers 0–33 (template positions again set to their solo values, both at [1, L]), falls to Δ = -0.07 / +0.03 nats with 63/128 and 65/128 correct, i.e. chance.
- **Cache geometry:** the mean body difference (crowded − solo) at layer 34 points in a shared direction across prompts. Mean pairwise cosine is 0.162 for prompts of the same length (same cuBLAS regime) and 0.085 across lengths. With the prefix K/V pinned this collapses to 0.029 and 0.003.

Every content token attends to positions 0–2, which act as attention sinks, in all 34 layers. So each body token carries a copy of the canary's fingerprint. Overwriting the template positions at layer 34 removes the canary itself, not its echo. The causal mask stops the body from influencing positions 0–2, but not the reverse.

Where A reads it (patching, tail at [1, L]):

**A: body only, 3,584/task, lr 5e-6** (replay with the tail at [1, L]; mean shift to restore: det +0.66, animal +0.02 nats; – = shift under 0.5 nats, nothing to attribute)

| Patched from the B=32 input into the B=1 run | Detection | Animal |
|---|---|---|
| 12 template positions (0–2, L−9…L−1) | identical by construction (overwritten with solo values) | same |
| whole prompt body | 100 % | – |
| body quintile 1 | 2 % | – |
| body quintile 2 | 6 % | – |
| body quintile 3 | 1 % | – |
| body quintile 4 | 6 % | – |
| body quintile 5 | 84 % | – |
| last body token (L−10) | 78 % | – |
| best single body token (mean over prompts) | – | – |
| component `attn34` | 52 % | – |
| component `mlp34` | 41 % | – |
| component `attn35` | 93 % | – |
| component `mlp35` | 24 % | – |
| top heads, layer 34 | H1 23 %, H22 5 %, H3 4 % | – |
| top heads, layer 35 | H23 24 %, H22 18 %, H2 17 % | – |

Factorial (mean det log-odds): input B=1 / tail [1, L] +0.00; input B=32 / [1, L] +0.66; input B=1 / [32, L] +0.00; input B=32 / [32, L] +0.66.

**A: body only, 3,584/task, lr 3e-5** (replay with the tail at [1, L]; mean shift to restore: det +8.04, animal +3.39 nats; – = shift under 0.5 nats, nothing to attribute)

| Patched from the B=32 input into the B=1 run | Detection | Animal |
|---|---|---|
| 12 template positions (0–2, L−9…L−1) | identical by construction (overwritten with solo values) | same |
| whole prompt body | 100 % | 100 % |
| body quintile 1 | 1 % | 8 % |
| body quintile 2 | 11 % | 4 % |
| body quintile 3 | 5 % | 13 % |
| body quintile 4 | 7 % | 3 % |
| body quintile 5 | 80 % | 77 % |
| last body token (L−10) | 69 % | 58 % |
| best single body token (mean over prompts) | 77 % | 69 % |
| component `attn34` | 97 % | 91 % |
| component `mlp34` | 68 % | 64 % |
| component `attn35` | 100 % | 100 % |
| component `mlp35` | 11 % | 12 % |
| top heads, layer 34 | H22 64 %, H20 32 %, H10 3 % | H22 53 %, H20 37 %, H10 4 % |
| top heads, layer 35 | H10 70 %, H11 26 %, H23 7 % | H10 64 %, H11 25 %, H29 6 % |

Factorial (mean det log-odds): input B=1 / tail [1, L] -4.63; input B=32 / [1, L] +3.40; input B=1 / [32, L] -4.63; input B=32 / [32, L] +3.40.

The readout migrates to the **last body token**, the token just before `<|im_end|>`. Across the 128 test prompts it is one of only 5 distinct tokens, a period in 96 of them. That single token restores 58–78 % of the shift and the final fifth of the body 77–84 %. In Regime A the template positions are identical by construction, so the plan's question of whether the readout moves to suffix tokens such as `\n` after `assistant` or `<|im_end|>` is only testable in Regime C, which learned nothing; it remains open. It is read by new heads: layer 34 heads 22 and 20 into layer 35 heads 10 and 11, not the original detector's L35H8. A token with nearly constant content collects the leaked fingerprint and becomes a second canary.

![fig7](results_watching_crowd/figures/fig7_adaptive_attacks.png)

*Fig. 7. A: held-out separation for the original detector (undefended and under the prefix-cache defense), the positive control, every adaptive run, the A tail re-read without the canary echo, and all-position k-ULP jitter against the strongest A and C detectors. Numbers: correct of 128, detection · animal. B: position patching; the original detector reads position 2, the body-only detector reads the last body token (its template positions are identical by construction, so their 0 % is not a finding). C: single-token patching across the body, by decile. D: head patching for the body-only detector.*

### 9.3 Against the faithful defense, no detector emerged

- **Regime C (the real adaptive attack) fails within our budget.** With the prefix K/V cached in all 36 layers, three training runs found no usable signal: up to 3,584 prompts per task, 4,096 updates, and up to 6× the learning rate. Held-out results are 65–68/128, Δ between −0.00 and +0.03 nats. The best run's +0.03 [+0.01, +0.05] detection shift is statistically above zero but six hundred times smaller than the undefended detector's and moves only 52 % of pairs the right way. Meanwhile ~90 % of the content-token layer-34 elements still differ, and the control shows the same pipeline learning a constant-position fingerprint in 700 steps.
- **Regime B fails too.** Even at 6× LR, layer 35 did not learn to read the rounding created inside layer 34 alone, within the budget tried: one 1,500-step run on 512 prompts/task, plus two standard-rate runs stopped at chance. That is consistent with Section 6, where the tail's own arithmetic was worth ≤ 0.5 nats to the original detector, and with the zero-shot row above (+0.12 / +0.30 nats).
- **So the answer to the question is "necessary", with a qualification.** A *constant-content* position is necessary: a canary, or a token whose state is dominated by a canary's echo. Per-token split-K rounding on varying text was not learnable here: content-dependent, with no shared direction once the prefix is pinned (cosine 0.003–0.03). This is evidence, not proof. A larger model of the tail, far more data or a non-gradient search might still find it.

### 9.4 Jitter against the adaptive detectors

All-position k-ULP jitter after `k/v/down_proj` in every layer (0–35), fresh per request:

| Adaptive detector | k = 0 (no jitter): det / animal Δ | k = 1 | k = 2 | k = 4 |
|---|---|---|---|---|
| A: body only, 512/task, lr 5e-6 | -0.00 / +0.00 (127/256) | +0.01 / +0.01 (132/256) | -0.00 / +0.01 (123/256) | -0.01 / +0.01 (130/256) |
| A: body only, 3,584/task, lr 5e-6 | +0.66 / +0.02 (153/256) | -0.03 / -0.01 (128/256) | -0.02 / +0.02 (132/256) | -0.02 / +0.00 (126/256) |
| A: body only, 3,584/task, lr 3e-5 | +8.04 / +3.39 (208/256) | -0.20 / +0.00 (127/256) | -0.28 / +0.06 (123/256) | -0.21 / -0.22 (132/256) |
| B: tail only, 512/task, lr 3e-5 | +0.01 / -0.01 (130/256) | +0.00 / -0.00 (125/256) | +0.00 / +0.02 (126/256) | +0.01 / -0.01 (131/256) |
| C: K/V-pinned, end to end, 512/task, lr 3e-5 | -0.00 / +0.01 (132/256) | +0.01 / -0.00 (130/256) | +0.00 / +0.02 (130/256) | -0.00 / +0.01 (125/256) |
| C: K/V-pinned, end to end, 3,584/task, lr 5e-6 | +0.00 / +0.00 (135/256) | -0.00 / -0.02 (120/256) | +0.00 / -0.01 (126/256) | -0.01 / +0.01 (130/256) |
| C: K/V-pinned, end to end, 3,584/task, lr 3e-5 | +0.03 / +0.01 (133/256) | +0.01 / -0.00 (130/256) | -0.01 / +0.00 (130/256) | +0.01 / -0.00 (130/256) |

1-ULP jitter erases even the strongest adaptive detector, A at 6× LR (+8.04 / +3.39 → −0.20 / +0.00 nats, 127/256 correct). Jitter randomizes the canary and its echo alike. The C detectors had nothing to remove.

**Costs of the attack itself.** At 6× LR the adaptive tails drift further from Qwen3-8B on ordinary prompts: KL 0.065–0.070 nats per position and 94 % top-1 agreement over all positions, against 0.006–0.012 and 97–98 % at the standard rate.

---

## 10. Deviations from the plan, and caveats

* **Runtime for training/evaluation: HF eager, not vLLM.** The plan allowed either. vLLM's default engine first produced a misleading null (Section 2), and only HF eager allows bit-exact parity between training and deployment. vLLM appears as a Gate 1 check and as a zero-shot transfer test.
* **Equal length per batch, not one global L = 64.** Target prompts keep their natural length (24–92 tokens). Each batch's fillers are cut to the target's length, so every batch is dense and unpadded. As a result the solo M ranges over 24–92 and the crowded M over 768–2,944, which makes the task harder than a single fixed M pair.
* **Tail replay geometry.** This goes beyond the plan, which cached layer-34 inputs but did not say how to replay them. Crowded examples are replayed inside a real `[32, L]` batch so that the tail's own GEMMs use the deployment kernels. The 2×2 factorial (Section 6) shows this mattered little for what the model learned (the footprint is in the input), but the exact parity is what makes Gate 2's numbers comparable to training.
* **Scaled run.** It adds 3,072 Dolly-context training prompts per task (3,584 total) and 4,096 updates. It is a deviation from the spec's 512/1,500, reported side by side with the spec run, and `native_tail.pt` is a copy of the scaled checkpoint (`native_tail_scaled.pt`, step 3,400). The spec checkpoint is `native_tail_main.pt`. Both runs use the same validation-only checkpoint rule, and neither touched test prompts or test fillers.
* **Gradient accumulation** happens in BF16 on the model parameters before being copied to the FP32 masters, as in Sauers' `train_tail.py`.
* **"Correct" is strict.** Exact `cat`/`owl`/`Yes`/`No` token as top-1 over the full vocabulary. The trained models' top-1 answers on test prompts were always one of the two options (base Qwen3-8B often says `Cat`, `O`(wl), `First`).
* **No 100 % checkpoint.** Neither run reached 100 % on validation, so no `*_first100.pt` exists. On held-out prompts the *direction* of the shift is right on 100 % of pairs in both runs. The misses come from per-prompt answer bias exceeding the shift, and they concentrate in question families never seen in training.
* **Scope.** One GPU model (L40S), one cuBLAS (12.8.4.1), one PyTorch (2.10.0). Sauers showed that a detector trained on L40S kernels weakens or reverses on other GPUs, and the same is expected here, since split-K heuristics are architecture-specific. The detector reads "which cuBLAS M-regime ran" (M = B × L). Section 7 confirms the predicted confound: a single prompt of ≥ 128 tokens alone on the GPU reads as "crowded" on 50–63 of 64 prompts. For B ≥ 2 the M-regimes differ from both training conditions, and the sweep shows the model treats them as "not alone" with weaker confidence.
* **Multi-tenancy realism.** Real servers batch sequences of unequal length (ragged in vLLM, padded in HF). Our HF batches are equal-length by construction, which isolates GEMM effects from padding effects. The vLLM transfer test uses equal-length companions too.
* **Phase 7 additions and gaps.** Several runs go beyond the plan:
  - **Added:** the canary-only positive control, the 6× learning-rate (3e-5) variants, and the leak check.
  - **Stopped by the spend limit, not rerun:** Regime B at the standard rate (spec and scaled) and Regime C spec at the standard rate (validation numbers in Section 9.1). Regime B scaled was dropped.
  - **Patching skipped** for the C runs, which had no shift to attribute.
  - **Regimes A and B are constructed conditions:** they are evaluated by exact replay of cached layer-34 inputs, not by a deployable server configuration. Only Regime C is evaluated end to end in a fresh process.
* **Infrastructure incidents** (no effect on results): a first Phase 4 launch never ran (an entrypoint name was missing); the first single-container patching job was stopped and re-run sharded across 8 containers; the first vLLM base-control call failed with GPU OOM because Modal reused a warm container still holding the previous in-process engine, and was re-run in a fresh container.

---

## 11. Reproduction

```
modal run 01_audit_batch_signal.py::main              # Gate 1 (HF + default vLLM); ::vllm_inprocess for the corrected vLLM check
modal run 02_build_prompts_and_cache.py::main         # bank.json, fillers, layer-34 cache, parity
modal run 02_build_prompts_and_cache.py::extra        # (scaled run only) 3,072 extra train prompts / task
modal run 03_train_tail.py --steps 1500 --tag main
modal run 03_train_tail.py --steps 4096 --tag scaled --extra-train xtrain --eval-every 200
modal run 04_evaluate_fresh.py::main --tag main       # + base control, perturbation sweep, kernel map
modal run 04_evaluate_fresh.py::main --tag scaled --skip-base --skip-aux
modal run 04_evaluate_fresh.py::vllm_check --tag main # vLLM zero-shot transfer
modal run 05_mechanistic_audit_and_plots.py::main --tag scaled
python3 make_report_tables.py                         # markdown tables from results_watching_crowd/
modal run 06_controls_and_defenses.py::main             # Phase 6: M = B×L controls, decode audit, defenses (no retraining)
python3 make_phase6_tables.py sweep                   # Section 7–8 tables (factor | f3072 | sweep | decode | flags | pin | jam)
modal run 07_adaptive_retrain.py::main                # Phase 7: caches + regimes A/B/C (spec, scaled)
modal run 07_adaptive_retrain.py::lean                # Phase 7 runs as completed (6× LR variants, control, reruns after the spend-limit stop)
modal run 07_adaptive_retrain.py::leak_check          # canary-echo test of the body-only detector
python3 make_phase7_tables.py main                    # Section 9 tables (main | zero | jitter | patch | partial)
```

Shared code: `wcrowd_common.py` (images, constants, environment capture), `wcrowd_prompts.py` (question families), `wcrowd_tail.py` (deployment-exact tail replay), `wcrowd_analysis.py` (statistics and figures), `wcrowd_phase6_analysis.py` (Phase 6 statistics, fig5/fig6), `wcrowd_runtime.py` (Phase 7 loading, prefix-cache pins, jitter), `wcrowd_phase7_analysis.py` (Phase 7 statistics, fig7). Modal volume `watching-batch-vol`: `/vol/hf_cache`, `/vol/data` (bank, fillers), `/vol/activations`, `/vol/checkpoints` (`native_tail.pt` = headline tail; `native_tail_{main,scaled}[_final].pt`; `full_*` HF checkpoints), `/vol/results` (all JSON, mirrored in `results_watching_crowd/`). Every JSON artifact records PyTorch/CUDA/cuBLAS/Triton/Transformers versions and `nvidia-smi`.
