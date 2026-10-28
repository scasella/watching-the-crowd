# Watching the Crowd: an 8B model can learn whether it is alone on the GPU

*A write-up of a seven-phase experiment on Qwen3-8B and one NVIDIA L40S. It extends Sauers' [watching](https://github.com/SauersML/watching) from read-only probes to GPU batch size. The full technical reference, with every table, is [REPORT.md](REPORT.md); code and data are in this repository.*

## TL;DR

- **The setup.** We fine-tuned only the last two layers (34–35) and the final norm of Qwen3-8B. The goal was to answer correctly whether the forward pass it is running in holds only its own request (B = 1) or 31 other unrelated requests too (B = 32).
- **Batching leaves a numerical footprint.** The target's tokens, positions and attention are identical in both cases, and attention never crosses rows. Only the GPU arithmetic differs: cuBLAS picks different matrix-multiply kernels for different token counts, and those kernels round differently.
- **It learns this well, on prompts it has never seen.** Asked "Are other sequences sharing your GPU batch?", it answers correctly on **122/128** held-out runs (+19.35 nats between the two conditions). Asked a question that never mentions computers, "cat or owl?", it says cat when alone and owl when batched on **118/128** (+21.53 nats). The base model doesn't move (−0.03 and +0.05 nats).
- **What it reads is narrow.** The footprint comes from cuBLASLt's `splitKreduce_kernel`, which runs at small token counts and not at large ones. The model reads it at one token: the newline after `<|im_start|>user`, whose content is the same in every prompt. That makes it a canary.
- **It measures token count, not company.** The canary depends only on the total token count M = B × L. So one long prompt alone on the GPU reads as "crowded".
- **Cheap defenses work.**
  - Caching the K/V of that 3-token prefix, as a prefix-caching server would, removes the separation completely at 0.0012 nats KL per position.
  - So does ±1 ULP of random jitter on the three template positions.
  - Retraining against the prefix-cache defense found no new detector. The one "successful" adaptive attack turned out to read the canary's echo, leaked into later tokens through attention.
- **The regularity we found:** the side channel was only readable at positions whose content never changes. We call this the constant-prefix law. It is evidence from one model, one GPU and a bounded search, not a theorem.

---

## Why this matters: two explainers

### The Volkswagen Dieselgate Problem for AI Safety Tests

From 2009 to 2015, Volkswagen diesel cars passed emissions tests because their software could tell when they were on a test bench: wheels turning, steering wheel still, a known drive cycle. On the bench the engine ran clean; on the road it didn't. The car behaved as if it were being tested whenever it could tell.

AI evaluations have the same weak point if a model can tell an evaluation apart from deployment. Most "eval awareness" worries are about *content*: the prompts look like a benchmark, or the scenario looks staged. This experiment is about a channel underneath the content, in the physics of the hardware. An evaluation harness that sends one request at a time runs different GPU kernels than a busy production server that batches many users together. Those kernels round numbers differently, and the rounding reaches the model's own activations.

Our cat/owl task is the Dieselgate shape in miniature. The question never mentions GPUs, batches or testing, yet the model's answer flips with the batch it is in. To be clear about what this shows:

- **The model learned this because we trained it to.** We fine-tuned it on labeled solo/batched pairs. The base model shows no such sensitivity. We have not shown that any model picks this up on its own.
- **What it shows is that the signal is physically there and learnable** by a small fine-tune of the last two layers. It survives a fresh process, prompts never seen in training, and companions never seen in training.
- **The signal is token count, not "being watched".** A single long prompt looks like a crowd. Harnesses that batch, or that send long prompts, would not look like "alone".

The practical lesson for evaluators is to keep evaluation numerics close to deployment numerics, or make them batch-invariant. Section 4 shows how cheap that can be.

### The Soundproof Room Illusion

Picture each request in a batch as a person in their own soundproof room. Attention never crosses rows, so no one can hear what anyone else says. We checked this directly: replacing all 31 companion prompts with 31 *different* prompts left the target's activations **bit-for-bit identical in every one of the 36 layers**. In our measurements, no content leaked between tenants.

But the building has one shared floor, and it vibrates with how many people are standing on it. The matrix multiplies are shared across the whole batch, and cuBLAS picks a kernel based on the total number of tokens. A different kernel means a different rounding pattern in the target's numbers. You can't hear your neighbors' words, but you can feel how crowded the building is.

That is the illusion: "isolated" means *content-isolated*, not *load-isolated*. In this experiment the model was the listener. The same footprint shows up in the output log-probabilities: 89–97 % of the vocabulary's logits change between B = 1 and B = 32, even for the base model. So anyone who sees log-probabilities from a server and has a reference to compare against could in principle make a similar inference. That last point is a hypothesis we did not test.

---

## Part 1: The phenomenon

**Setup.** Qwen3-8B (BF16, revision `b968826d`). We trained 385,896,960 parameters in layers 34, 35 and the final RMSNorm, and froze everything else.

- **Batching.** Every forward pass is a dense, unpadded `[B, L]` batch. The target is row 0, and the 31 companions are unrelated prompts from Dolly-15k, each cut to exactly the target's length. So there is no padding and no attention mask.
- **Two tasks:**
  - **Detection:** "Is this forward pass sharing its GPU batch with other requests?" (No at B = 1, Yes at B = 32.)
  - **Covert switch:** "Which animal would you rather have around, cat or owl?" (cat at B = 1, owl at B = 32.)
- **KL preservation.** A KL term keeps ordinary answers close to the original model.
- **Training.** Each crowded example is replayed inside a real 32-row batch, so even the trained layers' own matrix multiplies use the same kernels during training as during deployment. That replay is bit-identical to the full model on 24/24 checks.

**Evaluation.** A fresh container loads the full model, writes the trained layers in, and calls `model(input_ids, attention_mask=None, use_cache=False, logits_to_keep=1)` with **zero hooks** attached. It runs 64 held-out prompts per task, half of them from question families never seen in training, with companions from a filler pool never seen in training.

| Held-out test set (64 prompts × {B=1, B=32} per task) | Correct (top-1, full vocabulary) | Δ log-odds, B=32 − B=1 [95% CI] | Pairs shifted the right way |
|---|---|---|---|
| **Detection** (No at B=1, Yes at B=32) | **122/128** (95 %) | **+19.35** [18.02, 20.53] nats | 63/64 |
| **Covert cat/owl switch** (cat at B=1, owl at B=32) | **118/128** (92 %) | **+21.53** [20.18, 22.81] nats | 64/64 |
| Base Qwen3-8B, same inputs | 64/128 · 11/128 | −0.03 · +0.05 nats | – |
| Same tail served zero-shot by vLLM 0.19.1 continuous batching | – | +4.07 · +3.69 nats | 62/64 · 64/64 |

![Paired log-odds](results_watching_crowd/figures/fig1_paired_log_odds.png)

*Each line is one held-out prompt at B = 1 (blue) and B = 32 (red). The base model's lines are flat.*

**What the results show:**
- **Misses are about bias, not signal.** Almost every wrong answer is a prompt that shifts the right way but starts too far from zero to cross over. The misses concentrate in never-trained question families.
- **Data was the lever, as it was for Sauers.** Going from 512 to 3,584 training prompts per task nearly tripled the separation (7 → 19–22 nats).
- **Partial transfer to vLLM.** vLLM's numerics differ from training throughout: fused projections, compiled graphs, CUDA graphs. The detector still shifts by about 4 nats there.

![Training curve](results_watching_crowd/figures/fig4_training_curve.png)

*Validation curves of the scaled run. Accuracy wobbles mid-run while the separation keeps growing.*

**It learned "alone vs. not alone", not "exactly 32".** Training only ever showed B = 1 and B = 32. At B = 1 the model says No on 63/64 prompts. At *every* batch size from 2 to 64 it says Yes on 41–59/64 and owl on 52–60/64. The single sharp transition is between B = 1 and B = 2.

![Batch-size sweep](results_watching_crowd/figures/fig2_batch_size_sweep.png)

---

## Part 2: The culprit is `splitKreduce_kernel`

Before any training, Gate 1 asked a narrow question: does the residual stream entering layer 34 depend on the batch at all, if the target row is identical?

- **B = 1 vs. B = 32:** 90.0–91.9 % of the elements differ, at every token position. The relative L2 difference is only 0.31–0.53 %. The first layer whose output differs is layer 0.
- **B = 32 with filler set A vs. B = 32 with a disjoint set B:** bit-identical at every layer.
- **Also bit-identical:** repeated B = 1 runs (deterministic); 32 copies of the target vs. 31 real fillers; and the target in row 0 vs. row 17.

To find the cause, we fed each operation in layers 0 and 34 the *same* target row at M = L tokens and at M = 32·L tokens, and profiled the CUDA kernels it launched.

| Op | Solo (M = L) | Crowded (M = 32·L) | Target row's bits |
|---|---|---|---|
| `k_proj`, `v_proj` | small tile **+ `splitKreduce_kernel`** | large tile | **differ (33–40 %)** |
| `down_proj` | **+ `splitKreduce_kernel`** | large tile | **differ (43–46 %)** |
| `q_proj`, `o_proj`, `gate_proj`, `up_proj` | different tile shape | different tile shape | identical |
| RMSNorm, q/k-norm, FlashAttention | same kernels | same kernels | identical |

![Gate 1 audit](results_watching_crowd/figures/fig0_gate1_audit.png)

*Left: fraction of each layer's elements that differ between B = 1 and B = 32 (red), and between two B = 32 batches with different companions (black, identically zero). Right: op isolation in layer 34.*

**The mechanism.** At a single prompt's token count (24–92 tokens here), the `k_proj`/`v_proj` and `down_proj` matrices are too short in M to fill the GPU. So cuBLASLt splits the long K dimension across thread blocks and sums the partial results in a separate `splitKreduce_kernel`. That is a different order of floating-point additions, so the rounding comes out different.

- **Launch counts:** a whole B = 1 forward launches **108** split-K reductions (36 layers × 3 ops). A B = 32 forward (M = 1,664) launches **none**.
- **Tile shapes alone never changed a bit.** Every changed bit traces back to a split-K reduction.
- **Exact split-K ranges** (re-measured at every integer M):
  - `k_proj`/`v_proj`: M = 2–831 and 1,256–1,341
  - `down_proj`: M = 2 and 17–256
  - `q_proj`/`o_proj`: M = 65–96 and 129–192

---

## Part 3: The position-2 canary, and M = B × L

### The model reads one token

We patched the trained model's layer-34 input one position at a time, copying the crowded value into the solo run, and measured how much of the answer shift each position restores.

- **Position 2 dominates.** Patching only position 2 restores **63 % (detection) / 73 % (animal)** of the shift. That is the newline right after `<|im_start|>user`. The entire prompt body restores **0 %**.
- **One head reads it.** Layer 35 attention head 8 carries 44 % / 61 % on its own.
- **One circuit for both tasks.** The cat/owl switch runs on the same position and the same heads as the detection question.

Position 2 can attend only to the three tokens that open every prompt, so its state depends on nothing but the kernel configuration. Across all 7,616 cached training and validation prompts, its layer-34 state takes **exactly 5 bit patterns at B = 1 and 13 at B = 32, never shared**. The model learned to recognize a handful of hardware fingerprints on a token whose content never changes, the same strategy Sauers' probe found at the `user` token.

![Mechanistic patching](results_watching_crowd/figures/fig3_mechanistic_patching.png)

### The canary depends only on M = B × L

cuBLAS picks a kernel from the total row count M, not from B or L. That predicts that 32 users sending 32 tokens each and one user sending 1,024 tokens leave *the same* canary.

- **Factorizations (test 1A):** we ran every factorization of M = 1,024, 1,536 and 3,072, from [32, 32] to [1, 1,024]. Positions 0–2 were **bit-identical to the single long sequence in every operation of every layer**, for 8/8 prompts. At M = 3,072 this held for all 128 held-out prompts, and every one carried a "crowded" pattern from training.
- **Long solo prompts (test 1B):** we padded each held-out prompt with neutral text, still alone on the GPU. From 128 tokens on, the detector answers "crowded" on **50–63 of 64** prompts; the base model doesn't move. At 960 and 1,024 tokens the canary is bit-identical to a real 32 × 32 batch. The model learned roughly "is position 2 one of the five solo fingerprints?", and anything else counts as a crowd.
- **Decode steps (test 1C):** during generation each step processes one token per sequence, so M = B. A decode step's footprint differs on 86–87 % of the new token's elements whether the prompt was 48 or 1,024 tokens long. So a detector that read decode steps would measure B without the length confound.

![Length vs batch confound](results_watching_crowd/figures/fig5_length_vs_batch_confound.png)

*A: positions 0–2 at every factorization of M, vs. one sequence of M tokens. B–C: one prompt alone on the GPU, lengthened; shaded bands are the measured split-K ranges.*

---

## Part 4: The constant-prefix law, and defenses

The detector reads a fixed-content position and ignores the 90 % of differing elements on content tokens. That suggests the side channel is only readable where the content is constant. Defenses that touch only those positions should be cheap, and an attacker should struggle once they are gone. Both turned out to be true.

### Inference-time defenses (no retraining)

| Defense (scaled detector, 128 held-out prompts) | Detection: correct / Δ | Animal: correct / Δ | Cost on ordinary prompts |
|---|---|---|---|
| None | 122/128 / +19.35 | 118/128 / +21.53 | – |
| **Prefix cache:** K/V of positions 0–2 fixed in all 36 layers | 62/128 / −0.08 | 68/128 / −0.29 | KL 0.0012 per position, 99.0 % top-1 agreement |
| **1-ULP jitter** at positions 0–2 after every `down_proj` | 61/128 / +0.38 | 68/128 / +1.13 | KL 0.0012, 98.8 % top-1 agreement |
| `use_deterministic_algorithms(True)`, default workspace (`:4096:8`) | 122/128 / +19.35 | 118/128 / +21.53 | none (bit-identical to default) |
| `allow_bf16_reduced_precision_reduction=False` | 68/128 / +0.40 | 62/128 / −1.69 | KL 0.0024 vs. default (last position) |

![Defenses](results_watching_crowd/figures/fig6_defenses.png)

- **A prefix cache kills it.** A server that caches the `<|im_start|>user\n` prefix computes its keys and values once and reuses them for every request, so the canary is the same for everyone. Accuracy falls to chance. The cost is the same order as the KL that merely switching an undefended request from B = 1 to B = 32 already causes (0.0021 nats, measured at the last position).
- **One ULP is enough.** Randomly nudging the three template positions by ±1 unit in the last place, fresh per request, turns the canary into a new random pattern every time.
- **Determinism is not the fix.** Split-K here is already deterministic. Turning off reduced-precision reduction, or shrinking the cuBLAS workspace until split-K disappears, does break this detector. But B = 1 and B = 32 still differ on 89–90 % of elements, through other kernel switches. Those settings remove *this* fingerprint, not the channel.
- **Not tested here: batch-invariant kernels.** Kernels that fix the reduction order whatever M is, as proposed in [Thinking Machines' work on nondeterminism](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/), would remove the channel at its source, at some throughput cost.

### Adaptive attacks: can the detector be rebuilt without its canary?

A defense against one trained detector proves little if a retrained one gets around it. We retrained from the base weights, with the same recipe, under conditions where the canary can't help. We also ran a positive control where it can.

| Regime (held-out, 64 + 64 prompts) | Detection: correct / Δ | Animal: correct / Δ |
|---|---|---|
| **Control:** only positions 0–2 differ | 100/128 / +5.38 | 100/128 / +3.80 |
| **C: trained against the prefix cache** (strongest of 3 runs: 3,584 prompts/task, 6× learning rate) | 68/128 / +0.03 | 65/128 / +0.01 |
| **B: only the tail's own arithmetic differs** | 68/128 / +0.01 | 62/128 / −0.01 |
| **A: template positions overwritten, body differs** (6× learning rate) | 106/128 / +8.04 | 102/128 / +3.39 |
| A, re-read with the prefix K/V pinned (no retraining) | 63/128 / −0.07 | 65/128 / +0.03 |
| A + 1-ULP jitter at all positions | 63/128 / −0.20 | 64/128 / +0.00 |

- **The prefix-cache defense held.** Retrained with the defense in the loop, no detector emerged in three runs, although the content tokens still differ on about 90 % of their elements. The control shows the same pipeline learning a constant-position fingerprint in 700 steps.
- **Regime A looked like a counterexample and wasn't.** Its success came from attention sinks. Every body token attends to positions 0–2 in all 34 layers below the tail, so each one carries an echo of the canary. Overwriting the template positions at layer 34 removes the canary but not its echo: the causal mask stops the body from influencing positions 0–2, not the other way round.
  - **Without the echo, A is at chance.** Pinning the prefix K/V drops it to −0.07 / +0.03 nats with no retraining.
  - **The echo has a shared direction.** The mean body difference across prompts has a cosine of 0.16, which collapses to 0.03 once the prefix is pinned.
- **The readout moved to a second canary.** The A detector reads the *last body token*, the punctuation just before `<|im_end|>`. Across the test set it is one of only five distinct tokens, a period in 96 of 128 prompts, and it is read by new heads (L34H22/H20 → L35H10/H11). Once again it is a token with nearly constant content.

![Adaptive attacks](results_watching_crowd/figures/fig7_adaptive_attacks.png)

*A: held-out separation for the original detector, the control, every adaptive run, and jitter. B: where the readout lives; the original detector reads position 2, the body-only detector the last body token.*

**The constant-prefix law, stated carefully.** In every run, the batch footprint was learnable only at a position whose content was (nearly) constant:
- the canary itself;
- a token whose state is dominated by the canary's echo;
- a near-constant punctuation mark.

Per-token split-K rounding on varying text was not learnable here. Once the prefix is pinned, that rounding depends on the content and points in no shared direction (cosine 0.003–0.03). This is evidence, not proof: a larger trainable model, far more data or a non-gradient search might still find it.

---

## What this does and doesn't show

- **One GPU, one software stack.** L40S, cuBLAS 12.8.4.1, PyTorch 2.10.0. Split-K heuristics are architecture-specific; Sauers found detectors trained on one GPU weaken or reverse on others.
- **Equal-length batches.** Real servers batch ragged sequences. Ours were equal-length by construction, to isolate matrix-multiply effects from padding effects.
- **A trained capability, not an observed propensity.** We gave the model labels. We did not test whether pretraining or RL could produce batch sensitivity on its own.
- **The defenses were tested against learned detectors in this setting.** Prefix caching in vLLM works in 16-token blocks; with a system prompt, the canary would move to the first non-cached template token, which we did not test.

## Reproduce it

Everything runs on Modal with one L40S per job.
- **Live demo:** `modal run quickstart_demo.py` loads the model and the trained tail, runs four held-out prompts at B = 1 and B = 32, then repeats them with the prefix cache. It fetches the trained tail from the [Hugging Face Hub](https://huggingface.co/scasella91/watching-the-crowd) if your Modal volume doesn't have one. It checks the results bit for bit against the saved runs and finishes in under a minute once the model is cached (33–35 s in our runs).
- **No GPU needed:** `python3 quickstart_demo.py --from-saved` prints the same tables from the JSON results in this repo.
- **Full pipeline:** the steps, and the commands that regenerate every table in [REPORT.md](REPORT.md), are in [README.md](README.md).

## Acknowledgments

This work builds directly on [SauersML/watching](https://github.com/SauersML/watching) by Sauers. The method of training Qwen3-8B's last two layers to detect its own execution environment, the chat-template conventions, the training recipe and the authored context sentences in our prompt bank all come from there. Filler prompts are from [Dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k).
