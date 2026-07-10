# How we achieved bitwise zero-KL

*Sections 0–8 were measured at TP=PP=EP=CP=1. Sections 9–12 extend the result to **matched
tensor parallelism** (Megatron TP == vLLM TP), take it to **Qwen3.5-35B-A3B** at TP=8, make it
**train efficiently**, and document two metric traps that cost real time. The 35B run is
bitwise: rollout logprobs equal the trainer's forward on identical weights, exactly, on all
375,441 response tokens, with no TIS/MIS.*

---

## 0. What "zero-KL" means, and why it is not a tolerance

In on-policy RL for LLMs, the policy-gradient objective is corrected by an importance ratio

```
r_t = π_train(a_t | s_t) / π_rollout(a_t | s_t)
```

where `π_rollout` is the logprob the **inference engine** (vLLM) assigned when it sampled the token,
and `π_train` is the logprob the **trainer** (Megatron) assigns when it re-scores the same token.
Mathematically these are the same function of the same weights, so `r_t ≡ 1`. In practice they are
two different implementations of that function, and `r_t` drifts. The usual response is Truncated
Importance Sampling (TIS): clamp `r_t` and accept the bias.

Zero-KL removes the problem instead of correcting it: make the two runtimes produce **bitwise
identical** logprobs, so `r_t` is exactly `1.0`, `policy_kl` is exactly `0.0`, and TIS can be turned
off. "Bitwise" is not rhetorical. The gate we hold ourselves to is `max |decode − prefill| == 0.0`
and `exact-zero == 100%` of tokens, not `< 1e-6`.

That strictness is what makes the work tractable. A tolerance of `1e-6` hides which op is wrong; an
exact-zero gate turns every regression into a bisectable, reproducible fact.

### The residual is not noise — it amplifies

A 1-ULP bf16 difference in one attention layer does not stay 1 ULP. Through 36 layers and 2000
decoded tokens it becomes a mean logprob gap of `~0.014` with a max of `~0.30`, concentrated on
low-entropy tokens where the logit gap is small. Measured, on MiMo-7B: activating a different
flash-attention kernel in the trainer changed `max |diff|` from `0.2976` to exactly `0.0`. The whole
project is the discipline of never letting a single ULP through.

---

## 1. The one invariant everything rests on

> **A token's output must not depend on what else is in the batch, or on how the sequence was
> chunked, cached, or scheduled.**

Call this **batch invariance**. It is stronger than determinism (same input, same output, run to
run) and it is the property that makes decode == prefill.

Why: at rollout time, the engine computes token *t* with a **decode** forward — one token, a KV
cache, batch of N running requests. At training time, the trainer computes token *t* with a
**prefill** forward — the whole sequence at once, batch of one. If every op's per-token output is
independent of `(num_tokens, batch composition, cache state)`, then those two very different
executions produce the same bits. If any op is not, they don't.

Most kernels violate this for performance:

| Violation | Mechanism |
|---|---|
| Split-K / split-KV reductions | number of splits chosen from problem size → different summation tree |
| Tile/block heuristics | grid shape chosen from `M` → different reduction order within a row |
| Atomics | `scatter_add_` / `atomicAdd` → hardware-arbitrary summation order |
| Autotuning | kernel config chosen by wall-clock benchmark → different config per process |
| Fused kernels chosen by shape | prefill kernel ≠ decode kernel for the same math |

Every fix in this report is one of those five.

---

## 2. Architecture: one model, two runtimes

### 2.1 Why patching kernels alone is not enough

The first attempt was to make vLLM's native model and Megatron's model agree op by op. Measured on
Qwen3-4B, most ops already agreed:

| Op | vLLM ↔ Megatron | Action |
|---|---|---|
| GEMM (`matmul_persistent`) | **bitwise** | enable `batch_invariant_mode` on both |
| `log_softmax` | **bitwise** | same |
| RMSNorm | **1 ULP off** | route Megatron's forward to vLLM's `vops.rms_norm` |
| RoPE | bf16 vs fp32 arithmetic | patch Megatron to do the multiply-add in fp32 |

Unifying **both** sides makes the logits bitwise identical. Unifying only one buys nothing.

That worked for a 36-layer dense forward, but it does not scale: every model family adds its own
non-attention op that is batch-variant in one runtime and not the other, and you are then chasing
implementation differences forever. vLLM's *native* MiMo model had one such op that stayed
batch-variant no matter what.

### 2.2 The unified model: Megatron's GPTModel **inside** vLLM

The structural fix is to stop having two implementations.

`zerokl/gptmodel_vllm.py` registers a vLLM model class (`MegatronGPTModelForCausalLM`) whose entire
compute **is a Megatron `GPTModel`**, built by `megatron-bridge` from the same checkpoint the trainer
uses. vLLM contributes only what it is good at: the scheduler, the paged KV cache, the sampler.

Two swaps make this work:

* **`swap_core_attention`** replaces each `SelfAttention.core_attention` with an adapter onto vLLM's
  paged `Attention` layer, so the model can read and write the paged KV cache. Megatron gives it
  `sbhd` `[sq, b, np, hn]`; we drop the batch dim, hand `[tokens, heads, hn]` to vLLM, reshape back.
  q/k-norm and RoPE were already applied upstream by `SelfAttention`.
* **`_PositionIndexedRoPE`** fixes a subtle decode bug: `GPTModel` computes RoPE for sequence indices
  `0..L-1`, but paged decode feeds a 1-token input whose *true* position is `N`. We precompute the
  RoPE cache once and index it by vLLM's `positions`, so decode rotates at the right angle. Without
  this, decode ≠ prefill by construction.

Then we force the **Megatron LOCAL layer spec** (`SKYRL_ZEROKL_LOCAL_SPEC=1`): no TransformerEngine.
The model becomes plain torch — `F.linear`, `torch.nn.RMSNorm`, SDPA. Under `VLLM_BATCH_INVARIANT=1`
vLLM's aten overrides then cover *every* GEMM and norm in the model, because they are all aten ops.
TE's fused kernels would be invisible to those overrides.

`zerokl/no_te_guard.py` exists because `megatron.bridge` has three unguarded `import
transformer_engine` statements in its eager model-zoo import; the guard neutralizes them when TE is
genuinely absent.

**Consequence:** the trainer's forward and the rollout's forward are now *the same Python objects
running the same kernels*. The remaining differences are exactly (a) batch/length shape and (b) the
paged-vs-contiguous KV layout. That is the surface we have to make invariant — a much smaller and
enumerable one.

### 2.3 Native weight sync

Because both sides hold the identical `state_dict` (same names, same shapes, same fused layouts),
weight sync collapses from the HF round-trip

```
bridge.export_hf_weights → HF layout → vLLM load_weights → repack to fused
```

to a **direct native-layout tensor copy** (`zerokl/native_weight_sync.py`). Dropped: QKV split,
gate/up split, vocab gather, HF renames, vLLM's repack — and every seam risk they carried (vocab
padding, QKV interleave-vs-concat). Kept: the transport, the bf16 cast.

This is not just simplification. Each of those repack steps was a place where the engine's weights
could differ from the trainer's in the last bit.

---

## 3. Batch invariance, op by op

### 3.1 GEMM

`VLLM_BATCH_INVARIANT=1` installs Triton persistent matmuls with a fixed tile schedule, so a row's
result does not depend on `M`. The trainer enables the same overrides
(`enable_megatron_batch_invariant`).

**The SM90 trap.** On Hopper, vLLM's `enable_batch_invariant_mode` installs the Triton matmuls
**only on SM80**. On SM90 it merely pins the cuBLAS workspace — which disables split-K (giving
run-to-run determinism) but does **not** give M-invariance: cuBLAS still selects a different
kernel/tiling for `M=1` than for `M=512`.

Dense zero-KL never noticed, because all its GEMMs are bf16 and cuBLAS happens to be row-invariant at
those shapes. The MoE router's **fp32** GEMM is not. Measured on the real OLMoE router shape
`[512,2048] @ [2048,64]` in fp32: `4.3e-5` difference between `M=512` and `M=1` under cuBLAS, exactly
`0.0` under Triton for every shape and dtype tested.

Fix: `_install_moe_matmul_invariance()` registers vLLM's own Triton `mm/addmm/matmul/linear`
overrides via `torch.library.Library("aten", "IMPL")` on SM90+. This is what took MoE gate 1c from
`3.05e-5` to `0.0`.

```python
lib.impl("aten::mm",     bi.mm_batch_invariant,     "CUDA")
lib.impl("aten::addmm",  bi.addmm_batch_invariant,  "CUDA")
lib.impl("aten::matmul", bi.matmul_batch_invariant, "CUDA")
lib.impl("aten::linear", bi.linear_batch_invariant, "CUDA")
```

### 3.2 log_softmax

Already bitwise between the two runtimes under `batch_invariant_mode`. But the **sampler** is a
separate path: vLLM's v2 sampler computes the *rollout* logprob with a fused Triton kernel
(`compute_token_logprobs` → `_topk_log_softmax_kernel`) that inlines `log(softmax(logits))` and never
calls an aten op — so it bypasses the batch-invariant `aten::_log_softmax` entirely and diverges from
the trainer on a few tokens (most match, so the metric's `min` is 0 while its `max` is ~0.3).

Fix: `patch_vllm_logprobs_batch_invariant()` replaces it with the trainer's **exact** logprob math —
manual `(x − amax) − log(sum(exp(x − amax)))` in fp32, then gather. Not `torch.log_softmax`: that is
a different kernel and would not be bitwise equal to Megatron's manual formulation.

### 3.3 RMSNorm

Megatron's and vLLM's RMSNorm differed by 1 ULP. `apply_vops_rmsnorm_patch()` overrides
`BatchInvariantRMSNormFn.forward` to emit vLLM's C++ `rms_norm` bits, keeping the original fp32
**backward** so training gradients are unchanged. It works because Megatron's residual add is a
separate bf16 add, bitwise identical to vLLM's fused add.

On the LOCAL-spec (no-TE) path this is moot — the norm is `torch.nn.RMSNorm` → `aten::rms_norm` →
covered by the batch-invariant override on both sides. The patch is kept for the production TE stack.

Two later norm findings, both real:

* **Zero-centered gamma.** Qwen3.5 / Qwen3-Next normalize with `rms(x) · (1 + w)`, and Megatron's
  no-TE `WrappedTorchNorm` asserts against `layernorm_zero_centered_gamma`. `zerokl/zero_centered_norm.py`
  implements it for both runtimes: `F.rms_norm(x, shape, None, eps) * (1 + w)` — delegating the
  normalization to the same aten op the dense path already validated, and applying `(1+w)`
  elementwise (invariant by construction).

* **`RMSNormGated` picks its tile height from the row count.** See §6.4 — this was the single hardest
  bug in the project.

### 3.4 RoPE

Megatron did the rotate-half multiply-add in bf16; vLLM's CUDA kernel does it in fp32 with bf16
cos/sin. `apply_rope_fp32_patch()` patches `_apply_rotary_pos_emb_bshd` (which covers `thd` too) to
match. Requires `apply_rope_fusion=False` — the fused TE RoPE kernel is not interceptable.

### 3.5 Attention — the big one

**Root cause of the original residual:** vLLM's paged-attention **decode** diverges from a
full-sequence **prefill**, and the gap *grows with response length* (≈0 at 64 tokens, ≈0.017 at 256).
It is the FA3 **split-K heuristic** (`num_splits=auto`): the number of KV-reduction splits is chosen
from the problem size, so decode (1 query, long KV) and prefill (long query, long KV) sum the KV
contributions in different orders.

**Fix: `num_splits=1`.** A single-pass KV reduction is query-length invariant, so paged decode and
full prefill produce the same bits at all lengths.

`zerokl/varlen_backend.py` registers a vLLM attention backend named `CUSTOM` that runs
`torch.nn.attention.varlen.varlen_attn_out` with `num_splits=1` and `window_size=(-1, 0)` (unlimited
left, zero right == causal). It needs torch ≥ 2.14 (`torch.nn.attention.varlen`) and FA3 (SM 9.0+) —
which is why the whole zero-KL stack lives on a nightly venv. On torch 2.11 / vLLM 0.23 the pieces
don't exist (`varlen_attn_out` absent; vLLM 0.23's FA3 ignores `num_splits`).

The trainer must run the **same kernel**: `swap_trainer_core_attention_varlen()` replaces each
decoder layer's `core_attention` with `TorchVarlenCoreAttn`, calling `varlen_attn(num_splits=1,
window=(-1,0))`.

#### 3.5.1 The FA3-activation bug — the subtlest thing in this project

After all of the above, the live distributed trainer still showed `mean 0.01386 / max 0.2976`, while
**every in-process diagnostic harness measured bitwise**. That contradiction is the clue.

`torch.nn.attention.varlen_attn` dispatches to a different kernel depending on a **global process
flag**, `torch.nn.attention.current_flash_attention_impl()`. The engine's varlen backend calls
`activate_flash_attention_impl("FA3")` when vLLM builds. The **trainer actor builds no vLLM
in-process**, so its impl stays `None`, and the *identical* `varlen_attn(num_splits=1,
window=(-1,0))` call falls back to a non-FA3 kernel — a 1-ULP bf16 `core_attention` difference that
amplifies through 36 layers into ~0.014.

Every harness that built a vLLM engine silently activated FA3 and measured bitwise. Every harness
that didn't reproduced the divergence.

The diagnostic that cracked it: a layer trace showing `core_attention` with **bitwise identical
q/k/v inputs but different outputs across processes**. Identical inputs, different output ⇒ not data,
not weights, not the kernel arguments — a *global-state kernel-dispatch difference*.

Fix: `activate_trainer_flash_attention_impl()` at the top of `swap_trainer_core_attention_varlen`.

```
ZK_ACTIVATE_FA3=0 → varlen vs rollout: max 0.2976, mean 0.01386
ZK_ACTIVATE_FA3=1 → max 0.0000  (2048/2048 tokens BITWISE)
```

The `SKYRL_ZEROKL_VARLEN_OUT` / `SKYRL_ZEROKL_VARLEN_PAGED` experiments (paged vs non-paged varlen
kernels) were **red herrings** — no effect. Plain `varlen_attn` is fine once FA3 is active. They
remain as env-gated A/B knobs.

#### 3.5.2 Packed sequences

`TorchVarlenCoreAttn` originally built `cu_seqlens = [0, sq]` unconditionally, ignoring
`packed_seq_params`. Under sample packing a token of sequence 2 would attend to sequence 1 — causal
within the packed row, but wrong. The engine attends per sequence, so this alone would have broken
zero-KL on any packed run. It now honors `cu_seqlens` / `max_seqlen`, and handles the `thd` layout
where Megatron folds the batch dim away and passes `q` as `[T, np, hn]` (3-D) rather than 4-D `sbhd`.

Verified by the invariant that matters: **a sequence's logits inside a packed row are bitwise
identical to running it alone** (`max |diff| = 0.0`).

---

## 4. MoE: three batch-variance bugs

A Megatron MoE layer adds three ops the aten overrides do not reach — routing top-k, expert dispatch
(permute), expert combine (unpermute). We audited every one.

### 4.1 The expert combine is nondeterministic (`scatter_add_` → atomics)

```python
output_tokens.scatter_add_(0, sorted_indices..., permuted_tokens)   # moe_utils.unpermute
```

With top-k > 1 the destination indices repeat *k* times per token, so CUDA lowers `scatter_add_` to
`atomicAdd`: the *k* expert contributions to a token are summed in hardware-arbitrary order. This is
the weighted sum of the top-k expert outputs — the heart of the MoE layer. It is nondeterministic
run-to-run and batch-variant (decode's *k* adds land in a different order than prefill's).

**Fix:** `_fixed_order_combine` gathers each token's *k* rows and adds them in **ascending expert
order** — no atomics, no cross-token reduction. Verified on CPU in fp64 (where summation order is
immaterial) to reproduce megatron-core's `scatter_add_` combine exactly, and to be per-token
invariant to batch size.

### 4.2 The router's top-k order depends on `torch.is_grad_enabled()`

```python
torch.topk(scores, k, dim=1, sorted=torch.is_grad_enabled())   # moe_utils.topk_routing_with_score_function
```

When `moe_router_pre_softmax=False`, the softmax runs **over the top-k scores in the order topk
returned them**, so an unsorted (grad-disabled) order sums the denominator's *k* exponentials
differently than the sorted (grad-enabled) order → different probabilities. The rollout engine runs
under `no_grad` and the trainer's training forward under grad, so **the two disagree by
construction**.

Measured on CPU: on 20 000 random 8-expert rows with top-k 4, permuting top-k's return order changes
the post-softmax routing probabilities bitwise on **4 103 / 20 000 rows (20.5%)**.

Harmless when `pre_softmax=True` (OLMoE) — probs are read off the full softmax and scattered by
index, so top-k order never reaches an arithmetic reduction. We force `sorted=True` for both grad
modes regardless, since the function is called from a closure with no argument to thread through
(we temporarily swap `torch.topk` for the duration of the routing call).

### 4.3 The fp32 router GEMM is not M-invariant on SM90

Covered in §3.1. This was the *third* bug, found only after the first two were fixed and gate 1c
still showed `3.05e-5`. Localization: the router **logits** diverged first, at `2.4e-7` in fp32.

### 4.4 The rest of the MoE audit (what was already safe, and why)

| Op | Verdict | Reason |
|---|---|---|
| `moe_utils.permute` | **SAFE** | stable sort of a bool key is a unique permutation; `index_select` is a pure gather. The permutation depends on token count (it must) but the combine undoes it. |
| `MoEAllGatherTokenDispatcher` | **SAFE** | `masked_select` emits in row-major order of the `(E,T)` mask — the same order `permute` lays out rows, so probs stay paired with their tokens. |
| `SequentialMLP.forward` | **SAFE** *given* `moe_grouped_gemm=False` | each expert's GEMM is `aten::linear` → batch-invariant override applies; the `cat` order is the expert order. |
| **Grouped GEMM** | **UNSAFE** | its tile schedule depends on per-expert token counts → batch-variant. Hard-pinned off. |
| `RouterGatingLinearFunction` | **SAFE** on this stack | TE absent ⇒ `aten::mm` ⇒ override; fp32 under `moe_router_dtype="fp32"`. |
| `moe_router_enable_expert_bias` | **UNSAFE TO USE** | `expert_bias`/`local_tokens_per_expert` are **buffers**, and native weight sync copies `named_parameters()` only — the engine's routing bias would silently drift from the trainer's after the first optimizer step. `force_zerokl_moe_config` **raises** rather than pin it. |
| aux-loss / z-loss | **SAFE** | forward-identity; they only graft a gradient, and are skipped under `no_grad`. |

That `expert_bias` row is worth pausing on: it is a correctness bug with **no numerical symptom at
step 1**. It only appears after the first optimizer step, and only in the rollout. We chose to raise.

### 4.5 The pinned MoE recipe

`force_zerokl_moe_config` (applied identically to trainer and engine):

* `moe_grouped_gemm = False` → SequentialMLP (plain `F.linear` per expert)
* `moe_token_dispatcher_type = "allgather"` (a no-op at TP=EP=1)
* `moe_router_dtype = "fp32"`
* `moe_expert_capacity_factor = None` (no token dropping — dropping is batch-dependent by definition)
* every MoE fusion off (they are TE kernels; TE is absent)
* `persist_layer_norm = False` (OLMoE sets it True; the no-apex torch norm asserts against it)
* `moe_input_jitter_eps = None` (it randomizes routing)
* **EP = 1.** EP > 1 turns the expert combine into a cross-rank collective whose summation order is
  nondeterministic.

And `patch_olmoe_bridge_for_sequential_mlp` — megatron-bridge's OLMoE mapping names only the
grouped-GEMM expert params, so under the SequentialMLP pin **no expert weight would load from HF**.
The model would build, train, and talk nonsense. (Qwen3.5 needed its own version of this: see §6.6.)

---

## 5. Weight sync: the bugs that only appear at step 2

Three of these. All of them look like "the forward is broken" and none of them are.

### 5.1 The engine generated with θ₀ forever

`vllm_engine.py` forced `sleep(level=1)` under `SKYRL_ZERO_KL=1` — a bring-up crutch, "so
bridge-loaded weights survive sleep/wake". Level-1 sleep backs the weight pool up to CPU at the
**first** sleep (θ₀), and the wake path restores that never-updated backup **over the freshly synced
weights, every cycle**.

Proof: `[ZEROKL-ENGFWD]` — a checksum of the weights the engine's forward actually consumes, printed
on the first forward after each sync — read θ₀'s checksum `89866863.401759` on **all 31 syncs**,
while the receiver's post-sync module totals matched the fresh sender values exactly. Transport was
never broken. The engine simply wasn't reading the bytes that had been delivered.

This explains everything the symptom did: `abs_diff ≈ 0.009` appearing exactly when trainer weights
first move; identical drift with and without a separate extraction fix; slow growth (= cumulative
trainer drift away from θ₀); flat rewards.

Two dead ends on the way, both instructive:

* **Level-2 sleep instead:** the post-sync `wake_up(["kv_cache"])` remapped the weight pool to *zero
  pages* (`ENGFWD abs-sum = 0.000000`). On this nightly the cumem wake is not reliably tag-scoped:
  *any* wake after the sync clobbers weights — level 1 restores stale θ₀, level 2 zeroes them.
* **Level 2 + reapply:** params became perfect (`SENDER == REAPPLY == ENGFWD`) but generation broke
  catastrophically (step-2 diff ~1.0 nat, 96% of tokens). Level-2 discard **zeroes pool-resident
  derived state that is neither a `named_parameter` nor synced** — e.g. RoPE `inv_freq`. Invisible to
  every parameter checksum. Level 1's backup/restore is *required* for that derived state.

**Final fix:** keep sleep level 1, and add `zerokl_reapply_cached_weights` — `WorkerWrap` caches CPU
copies of every synced tensor at `load_weights` time; after the final `wake_up(["kv_cache"])` (which
restores stale θ₀ params), an RPC overwrites params with the cached synced bytes, per-tensor, no big
staging buffer.

### 5.2 Stale extraction (secondary)

`use_precision_aware_optimizer + optimizer_cpu_offload + overlap_cpu_optimizer_d2h_h2d` left the
model params ~2 optimizer steps behind at extraction time. Sender checksums at syncs #1–#3 were all
byte-equal to θ₀; per-sync deltas ramped like the lr warmup schedule shifted by ~2 steps.

Fix: `_zerokl_force_fresh_model_params()` on `optim_step` — `torch.cuda.synchronize()` +
`_copy_main_params_to_model_params()` per chained optimizer + `start_param_sync(force_sync=True)` per
chunk; plus a `torch.cuda.synchronize()` before extraction in `broadcast_to_inference_engines`.

### 5.3 The checksum chain

The reason these were findable at all is a chain of fp64 abs-sum checksums, gated behind
`SKYRL_ZEROKL_DEBUG=1`:

```
[ZEROKL-SENDER]  → what the trainer extracted
[ZEROKL-REAPPLY] → what was re-applied to the engine after wake
[ZEROKL-ENGFWD]  → what the engine's forward actually read
[ZEROKL-SCOREFWD]/[TRAINFWD] → what the trainer's forwards read
```

All five must be equal, per step, with a distinct changing value each step. If `SENDER == RECEIVER`
but `ENGFWD` differs, it is not a transport bug — that single distinction saved the project a week.

**Diagnostic rule learned the hard way:** engine-actor prints do **not** appear in the driver log.
Trainer-worker prints go to `/tmp/skyrl-logs/infra-*.log`. Debug the engine via an in-process parity
test, never by grepping the driver log.

---

## 6. GatedDeltaNet: linear attention

Qwen3.5 is a hybrid — **3 of every 4 layers** are GatedDeltaNet (GDN) linear attention. Baseline
divergence, Qwen3.5-0.8B, 32 seqs × 2048 tokens, temp 1.0:

```
mean |decode − prefill| = 1.67e-2    P99 = 0.124    max = 0.247
exact-zero = 2.52% of tokens
```

and **flat across position** — a steady per-token kernel mismatch, not accumulating state drift. vLLM
refuses to even try: `batch_invariant mode is not supported for GDN_ATTN`.

### 6.1 Why it diverges

A GDN layer is *trained* and *prefilled* with a chunked-parallel kernel (`chunk_gated_delta_rule`),
and *generated* with a fused **recurrent** kernel that advances the state one token at a time. The
two are algebraically equal and numerically different. This is the "fused kernel chosen by shape"
violation, in its purest form.

### 6.2 A racy Triton autotune config

Before anything else: `chunk_scaled_dot_kkt_fwd_kernel` compiled with `BK=64, num_warps=4,
num_stages≥2` is **nondeterministic on H100** — identical inputs, different results run to run
(max ~5e-2). Only once the grid exceeds roughly one wave (**≥ 5 chunks**; NT ≤ 4 hides it). The
autotuner picks that config by wall-clock benchmark. Every other config in its space is deterministic.

Worse, autotuning is a **per-process** decision. The trainer and the engine autotune independently
and can land on different configs — different reduction orders — for that reason alone.

Fix: `pin_fla_autotune_configs()` pins all 9 autotuned kernels in the chunk path to `configs[0]` —
the first entry of the kernel's own statically-declared list. No magic numbers, no benchmark, no
per-host drift; both sides import the same source file, so both pin the same config.

With pinning, `chunk_gated_delta_rule` is **deterministic + cross-sequence invariant + prefix
invariant**, all bitwise. `verify_gdn_batch_invariance()` asserts all three and raises on violation.

> *Trap:* the racy config only shows up with **≥ 5 chunks**. A 2-sequence, 1-chunk test looks clean
> and proves nothing.

### 6.3 Chunk-consistent decode

Don't write a decode kernel that *approximates* the chunk kernel — **decode with the chunk kernel.**

Two measured properties make this exact:

* **Prefix invariance:** `chunk(x[:t+1], S)[t] == chunk(x[:L], S)[t]`, bitwise. A token's output does
  not depend on tokens after it, even though the kernel tiles the chunk. (The intra-chunk ops are all
  causally row-local: `solve_tril` is a forward substitution, the QKᵀ products are tril-masked, the
  log-decay cumsum is an inclusive scan.)
* **Exact state chaining:** running chunk-by-chunk and carrying `final_state` reproduces one long call
  bitwise. The one op that reads a whole chunk — the inter-chunk state advance, which rescales by
  `exp(g_last)` — only ever runs on a **full** chunk.

So: pin the recurrent state to the **chunk grid** (absolute positions that are multiples of
`C = FLA_CHUNK_SIZE = 64`, the same grid the trainer's single full-sequence call uses); buffer the ≤ C
tokens of the currently open chunk; and at every decode step re-run the chunk kernel over the open
chunk starting from the boundary state, taking the last row. When the open chunk fills, its
`final_state` becomes the new boundary state and the buffer resets.

**`ssm_state[slot]` now means "state at the last chunk boundary"**, not "state after the last token".
That is only coherent if every read and write goes through our code — hence the loud failures in §7.

What is buffered is the **pre-conv** `mixed_qkv`, not post-conv values: vLLM uses `causal_conv1d_fn`
at prefill and `causal_conv1d_update` at decode, and **they do not agree bitwise**. Re-running one
invariant conv over the open chunk means decode literally re-executes the prefill code path.

`gdn_ops.gdn_causal_conv` writes the width-4 causal depthwise conv as shifted multiply-adds with a
single fp32 accumulation — elementwise, so prefix/batch invariance holds *by construction*, not by
measurement. It is a rounding error of the layer's cost next to two big GEMMs.

> **Trap:** q and k **must** be L2-normalized before the chunk kernel, or `(I − β·k·kᵀ)` is not a
> contraction: the recurrent state grows to ~1e24 within a few chunks and then goes NaN. This looks
> exactly like a decode bug and is not one.
>
> **Trap:** `max(0.0, nan)` returns `0.0` in Python. A naive `worst = max(worst, d)` diff harness
> reports NaN as "exact, max 0.0". Use a NaN-safe comparison.

### 6.4 The residual: `RMSNormGated` picks its Triton tile height from the row count

With chunk-consistent decode in place, the engine *still* diverged. The layer-level test already
proved the math (450/450 decoded tokens bitwise), so the bug had to be in the wiring. Bisect chain:

1. **`gdn_engine_layer_bisect`** — decoder layer 0 first diverges at decode step 50 (`2.4e-4`), and
   **deeper layers diverge earlier and larger** (layer 23 at step 20, `6.4e-2`). Deeper-earlier is the
   signature of a *tiny error present everywhere, amplified by depth* — not an onset event.
2. **`gdn_engine_core_probe`** — layer 0's GDN core is **bitwise on both sides**: its `mixed_qkv`
   input *and* its `core_attn_out` output, at every generated position, across all three chunk rolls.
   Chunk-consistent decode was correct; the bug was outside it.
3. **`gdn_engine_submodule_probe`** — `in_proj` clean, core clean, `out_proj` first bad ⇒ the culprit
   is the gated norm sitting between them.

`fla.ops.layernorm_guard.layer_norm_fwd` chooses its Triton tile height from the row count:

```python
rows_per_block = min(next_power_of_2(cdiv(M, 2 * sm_count)), 4)
```

and the kernel reduces a `[ROWS_PER_BLOCK, BLOCK_N]` tile with `tl.sum(x, axis=1)`. The tile shape
decides how a row's 128 elements are spread across threads, hence the order of the fp32 reduction,
hence the last bit of `rstd`.

On H100 (132 SMs), a GDN decode step has `M = num_tokens × num_v_heads = 16` → **1 row**. The prefill
that rescores those same tokens has `M = 3472` → **4 rows**. Same input row, different bits,
occasionally.

Fix: `pin_gdn_rmsnorm_rows_per_block()` pins the tile height to a constant.

**This is the archetype of the whole project.** Not the exotic linear-attention algorithm — a
normalization kernel choosing a grid shape from `M`.

### 6.5 One implementation, two runtimes (again)

`megatron/core/ssm/gated_delta_net.py` imports `chunk_gated_delta_rule` from the `fla` package and
raises if it is absent. `fla` is not installed in the zerokl venv — and must not be: a second copy of
the chunk kernel is a second set of autotune decisions, which is exactly the divergence we removed.

`zerokl/gdn_fla_shim.py` registers an `fla` facade in `sys.modules` backed by `zerokl/gdn_ops.py`, so
the trainer executes *literally the engine's ops*:

```
fla.ops.gated_delta_rule.chunk_gated_delta_rule → gdn_ops.gdn_chunk       (pinned autotune)
fla.modules.l2norm.l2norm                       → gdn_ops.gdn_l2norm      (vLLM's l2norm_fwd)
fla.modules.convolution.causal_conv1d           → gdn_ops.gdn_causal_conv (elementwise)
```

Two things this exposed:

* **vLLM's vendored FLA chunk kernel has no backward.** `ChunkGatedDeltaRuleFunction` defines a
  `forward` and nothing else — it is an inference-only vendoring. `gdn_chunk` keeps that exact kernel
  in the forward (that is what makes decode == training bitwise) and supplies the VJP by
  differentiating a torch reference for the same function at the same inputs. Gradients need not be
  bitwise; **forward logprobs do**. Verified the reference agrees with the kernel forward
  (relative L2 `3.3e-3`, bf16 kernel vs fp32 reference) — otherwise we'd be differentiating the wrong
  function.

* **`gdn_l2norm` silently zeroed the q/k gradients.** It was a bare Triton launch writing into
  `torch.empty_like(x)`: no autograd history, so backprop delivered **exactly zero** gradient to
  `query` and `key`. The layer still trains through `v` and the loss still falls — this would have
  shipped. Same treatment: kernel forward, autograd of `x·rsqrt(sum(x²)+ε)` in the backward. The gate
  now asserts the q/k projection gradients are **nonzero**, not merely finite.

Also: megatron's `_compute_g_and_beta` and `_prepare_qkv_for_gated_delta_rule` carry `@jit_fuser`
(= `torch.compile` on torch ≥ 2.2). A compiled `exp`/`softplus` is Triton's `libdevice` version, not
ATen's, and they disagree in the last ulp. The shim rebinds both to eager.

And `A_log.exp()` is taken in the parameter's own dtype, not upcast first: Megatron stores `A_log` in
`params_dtype` (bf16) and exponentiates before the fp32 multiply, and `exp(bf16(x)).float() ≠
exp(float(x))`. Zero-KL lives in that last ulp.

### 6.6 Two silent wrong-model traps

Neither has a numerical symptom. Both produce a model that is *perfectly bitwise against itself*.

* **No hybrid no-TE layer spec.** `make_zerokl_local_layer_spec` had no hybrid branch, so Qwen3.5
  built **24 dense attention layers** instead of 18 GDN + 6 attention. Zero GDN weights loaded. The
  parity test printed `256/256 bitwise, max 0.0` while generating
  `' 0 -s\n\n(3 192=".,)5S;'`. `gptmodel_vllm` now **raises** when GDN is on and zero GDN layers are
  found.

* **The MoE expert mapping matches nothing.** megatron-bridge's Qwen3.5 MoE bridge declares only the
  grouped-GEMM expert names (`mlp.experts.linear_fc1.weight*`) while the zero-KL recipe pins
  SequentialMLP (`mlp.experts.local_experts.*.linear_fc1.weight`). All 256 experts per layer would
  have stayed at random init. Loss still falls. Rollouts are garbage.

> **Rule:** read the `[GEN]` line before the parity number. A bitwise number from a wrong model is
> self-consistent and meaningless. Assert the architecture, assert the loaded weights differ from
> their init, and count patch invocations so a no-op patch cannot be reported as a pass.

### 6.7 Cost

Chunk-consistent decode re-runs the chunk kernel over the open chunk (1..C tokens, mean `(C+1)/2`),
so a decoded token costs ~`C/2` token-rows on GDN layers instead of 1.

| arm | 16 seqs × 512 tok | 1 seq × 512 tok |
|---|---|---|
| stock vLLM (not bitwise) | 533.2 gen tok/s | 34.6 gen tok/s |
| + batch-invariant stack (not bitwise) | 426.7 (1.25×) | 28.2 (1.23×) |
| + chunk-consistent decode (**BITWISE**) | 73.8 (**5.78× vs bi**) | 14.4 (**1.96× vs bi**) |

The ratio **grows with concurrency** (1.96× at 1 seq → 5.78× at 16). Cost that scales with the number
of sequences is not FLOPs — it is the per-slot python loop in `ChunkConsistentGDN.decode`, which runs
`_prep` (a conv + two l2norms) once per slot per GDN layer: 16 slots × 18 layers ≈ 288 launch groups
per decode step. Batching the open chunks is bitwise-safe **by construction** (conv is elementwise,
l2norm row-local) and is the first optimization to do. **Not done.** The number above is what the code
costs today.

`C` (`FLA_CHUNK_SIZE`) is the other knob, and it must match on both sides because it defines the
chunk grid the recurrent state is pinned to.

---

## 7. Failing loud

Chunk-consistent decode redefines `ssm_state[slot]`. Anything that reads, splits, or reuses that state
behind our back must be off, and we **raise at engine init** rather than degrade quietly:

* chunked prefill — a prompt spread over several forwards would need a mid-prompt state
* prefix caching — would resume a sequence from another sequence's boundary state
* speculative decoding — advances the state by several tokens with the spec kernels
* CUDA graphs — our decode is a python loop over ragged open chunks; it does not capture

`assert_engine_args_compatible()` raises; `_forward_core` asserts `spec_sequence_masks is None`.

Similarly, `force_zerokl_moe_config` **raises** on `moe_router_enable_expert_bias` and
`moe_input_jitter_eps` rather than silently pinning them.

The general principle: a zero-KL violation that produces a plausible number is worse than a crash.

---

## 8. What we *proved does not* affect zero-KL

This section matters as much as the fixes. Several things that "obviously" break bitwise parity do
not, and the conservative config was costing 4.6× rollout throughput for no correctness benefit.

### 8.1 Prefix caching, chunked prefill, CUDA graphs — all bitwise-safe (dense/softmax path)

Live DP8 MiMo-7B A/B, all three enabled together:

| | step-1 `policy_kl` | `minibatch_rollout_logprobs_abs_diff_max` | generate | step total |
|---|---|---|---|---|
| all off (conservative) | 0.0 | 4.89e-6 | 795 s | 912 s |
| all three **on** | 0.0 | **4.77e-6** | **174 s** | **293 s** |

Identical parity. `avg_response_length` 7072 vs 7130 and reward −1.68 both arms, so the **4.6× generate
/ 3.1× step** speedup is real, not shorter or degenerate outputs. 174 s ≈ the *non*-zeroKL baseline's
generate time (175.8 s) — i.e. this restores standard vLLM rollout speed **while staying bitwise**.

**Why they are safe:** with the `num_splits=1` single-pass KV reduction, attention is invariant to
cache state, to how the prompt was chunked, and to the launch mechanism. Those three features become
*scheduling / placement* changes, not *function* changes. The old `vllm_engine.py` comments claiming
prefix caching and chunked prefill cause ~0.0104 drift are **obsolete** — they predate the FA3 /
`num_splits=1` backend, when attention really was context-dependent.

They are exposed as per-feature env gates (`SKYRL_ZEROKL_ENABLE_PREFIX_CACHE`,
`..._CHUNKED_PREFILL`, `..._CUDAGRAPH`), defaulting **off** so the conservative config is unchanged.

*Caveats:* validated bitwise at step 1; CUDA graphs are off for the **MoE** path because
SequentialMLP's `torch.split` on per-expert counts needs a device→host sync every layer, which graph
capture cannot tolerate. And all three are off for **GDN**, not because they are unsafe in principle
but because chunk-consistent decode's boundary state cannot yet survive them (§7). Re-enabling them
for GDN is a follow-up, not a bug.

### 8.2 Paged vs non-paged varlen (`SKYRL_ZEROKL_VARLEN_OUT`, `..._VARLEN_PAGED`)

Both were pursued as suspects for the ~0.014 residual. **No effect.** Plain `varlen_attn` is fine once
FA3 is active (§3.5.1). Kept as env-gated A/B knobs. Red herrings.

### 8.3 The MoE ops that are already invariant

`permute`, `MoEAllGatherTokenDispatcher`, `SequentialMLP` (given no grouped GEMM),
`RouterGatingLinearFunction` on a TE-less stack, aux-loss and z-loss. Reasoning in §4.4. Auditing
them and finding them safe is what let us stop looking.

### 8.4 Weight transport

The CUDA-IPC / NCCL transport is **byte-faithful**. All 31 receiver post-sync totals exactly equaled
the 31 sender checksums. A prior handoff claimed "delivery is broken"; that was a misread of a
cumulative checksum. Every real sync bug was on the *read* side (§5.1) or the *extraction* side
(§5.2), never the wire.

### 8.5 Gradient bitwiseness

Zero-KL constrains the **forward** logprobs, which set the importance ratio. Gradients only have to be
*a* gradient of that forward, to floating-point accuracy. This is what licenses `gdn_chunk`'s
reference VJP (§6.5) and the fp32 RMSNorm backward (§3.3). Megatron makes the same trade in
`deterministic_mode`, where the forward itself changes.

---

## 9. Matched tensor parallelism (TP>1)

Everything above holds at TP=1. Scaling to a model that does not fit twice on one GPU (trainer +
colocated engine) needs matched TP: **Megatron TP must equal vLLM TP**, because the trainer's
forward and the in-vLLM GPTModel forward must run the *same* sharded computation. A row-parallel
all-reduce over 2 ranks rounds differently than over 8; "same model" now means "same shards,
reduced in the same order."

### 9.1 The engine builds a TP-sharded GPTModel inside vLLM

At TP=1 with `VLLM_ENABLE_V1_MULTIPROCESSING=0` the engine actor *is* the worker, so the
registrations from section 2 reached the model build directly. At TP>1 vLLM spawns one worker
subprocess per rank, and none of them ran any of it — the first symptom was blunt:
`Model architectures ['MegatronGPTModelHybridForCausalLM'] are not supported`. The fix is a
`vllm.general_plugins` entry point (`zerokl/vllm_plugin.py`): vLLM calls `load_general_plugins()`
in **every** worker, which is the only hook that reaches the spawned processes. It installs the
whole stack (GPTModel registration, CUSTOM varlen backend, batch-invariant sampler, GDN pins) per
worker, idempotently, gated on `SKYRL_ZERO_KL=1`.

The wrapper then initializes Megatron model-parallel state over vLLM's own worker process group
(`mpu.initialize_model_parallel(tp)`; single-node TP only — the worker world must *be* the TP
group), builds the same sharded GPTModel the trainer runs, and sets `parallel_output=False` so the
column-parallel output layer gathers logits for vLLM's sampler.

### 9.2 Weight sync at TP>1 is correct by construction

The concern was that the sync could silently deliver rank 0's shard to every engine rank. It does
not, and the reason is the transport, not luck: the CUDA-IPC path keys handles by **physical GPU
UUID** (`weight_sync/cuda_ipc_strategy.py`). Each trainer rank's `named_parameters()` are already
the exact per-rank shards, and colocated placement puts trainer TP rank *r* and engine TP rank *r*
on the *same* GPU — so engine rank *r* opens the shard of the trainer rank on its own GPU. No
reshard happens or is needed. Verified digit-for-digit: the `[ZEROKL-CKSUM]` SENDER, RECEIVER, and
ENGFWD (the weights the forward actually reads) totals are equal on every worker, per rank, every
sync.

### 9.3 The trainer's logprob gather must match the engine's row formula

At TP>1 the trainer's `_compute_distributed_log_softmax` computes `lse` as
`log(all_reduce_sum(per-shard exp-sums))`, whose fp32 summation order differs from the engine's
single-row `sum(exp(·))` over the **gathered** full vocab (the engine has `parallel_output=False`).
Measured: ~60% of rows differed in the last ULP, up to `1.9e-6` — small, not bitwise. Fix: under
`SKYRL_ZERO_KL=1` at TP>1, the trainer all-gathers the vocab shards (pure data movement, bitwise)
and applies the engine's exact single-row formula (`distributed/megatron/model_utils.py`). The
contiguous `[..., V]` layout is load-bearing — aten's `sum` reduces in a shape-dependent order, and
only that layout reproduces the engine's `lse`. (At 35B this gather held ~4 full-vocab fp32 copies
and OOMed at `micro_batch>1`; it is now done in place with a `logprobs_chunk_size` bound — see §11.)

### 9.4 The decisive TP=2 bug: a `torch.compile`d gated norm

At TP=1 the Qwen3.5-0.8B GDN hybrid passed the gate. At TP=2 the same model failed:
`policy_kl` clean but `rollout_train_logprobs_abs_diff` mean `0.003`, **max `0.35`** — and the
standalone engine's own decode-vs-prefill diverged (max `0.16`, onset mid-second-chunk).

Root cause, localized by per-worker layer tracing (`SKYRL_ZEROKL_LAYERTRACE` — in-process hooks
cannot reach vLLM's mp workers, so each worker writes its own trace): Megatron's
`GatedDeltaNet._apply_gated_norm` still carried `@jit_fuser`, which is `torch.compile` on this
stack. Inductor **shape-specializes** the fused `rms_norm + silu + mul` kernel, so a decode step
(rows = tokens×heads_local = 8 at TP=2) and the prefill that rescores it (~880 rows) run *different*
compiled kernels whose fp32 reductions differ in the last ULP. That noise compounds through 18 GDN
layers and **leaks across ranks** via `out_proj`'s row-parallel all-reduce (rank 0's clean norm +
rank 1's dirty norm → both diverge). At TP=1 the two shapes happened to compile to agreeing kernels,
which is exactly why TP=1 passed and hid it.

This is the *same class* as §6.4 (`RMSNormGated` picking its tile height from the row count) and the
*same class* as the router's `is_grad_enabled()` top-k (§4.2): **a kernel selected by shape or
mode**. The `EAGER_PREP` fix had already de-compiled two of GDN's three `@jit_fuser` methods; this
was the third. Fix: eager rebind of `_apply_gated_norm` (`gdn_fla_shim.py`). Engine decode==prefill
went from max `0.16` to **512/512 bitwise**; trainer==engine returned to the `~8e-7` floor.

> **Method note.** "Rescore the same sequence twice" distinguishes a race (differs) from a
> shape-specialization (bitwise) — here it was bitwise, so the divergence was shape-dependent, not
> nondeterministic. And "decode matches the trainer but prefill doesn't" points at a per-shape
> kernel, not state corruption.

**Result at TP=2** (Qwen3.5-0.8B, 4 engines × TP2): `policy_kl = 0.0`, `abs_diff_mean` 8.1e-7 /
8.2e-7 / 8.2e-7 / 8.3e-7 across four steps — the same floor as the TP=1 gate. Dense MiMo-7B at TP=2
is fully bitwise (512/512).

---

## 10. Qwen3.5-35B-A3B at TP=8

The first full-scale run (256 experts, 40 layers, GDN hybrid, matched TP=8, EP=1, ETP=8) surfaced a
sequence of blockers — each one a real bug, each found by measurement and fixed bitwise before the
next. They are worth listing because they are the difference between "works on a 0.8B toy" and
"trains a 35B model."

| # | Failure | Cause | Fix |
|---|---|---|---|
| 1 | `heads (16) and query groups (2) must divide TP=8` | Qwen3.5-35B has 16 q-heads / **2** kv-groups; at TP=8, groups < ranks | Mirror Megatron's **kv-replication** path: 1 kv-head/rank, replicated (all-gather + slice, bitwise); shared `zerokl_local_head_counts` on both swaps |
| 2 | first weight sync ~90 min | SequentialMLP explodes the native param count to **20,943**; one IPC chunk per param | Bucket into ≤512 MiB chunks → **33 chunks**; the IPC request already carries `(names, shapes, sizes)` |
| 3 | `MoELayer` raises during training | Megatron's MoE+TP-without-SP **performance** veto (not a correctness assert) | Flip only the parent `MoELayer.training` flag around the call; children keep theirs |
| 4 | `aten::linear_backward` not implemented on CUDA | We registered CUDA kernels for `aten::matmul`/`aten::linear` — **composite** ops → autograd leaves demanding a `*_backward` that exists only for Meta | Override **primitives only** (`mm`/`addmm`); matmul/linear/`F.linear` decompose onto them and stay differentiable. (Fired only at 35B: OLMoE's every GEMM goes through Megatron's custom-autograd linears; the shared-expert gate calls `F.linear` directly, and the GDN chunk VJP calls `torch.matmul` under grad — that was the *next* crash.) |
| 5 | engine init OOM (32 GiB) | vLLM profiles at `max_num_batched_tokens`; the wrapper materializes full-vocab fp32 logits per profile row (~16 GB at Qwen3.5's 248,320 vocab) | Cap the profile budget at `max_model_len` |
| 6 | `TP=4` OOM in `restore_grad_buffers` | Halving TP halves the weight shard but **doubles** the per-rank fp32 DDP grad buffer → 65 GiB/GPU. (A recommendation I made and had to retract — the run asked for exactly the TP=4 grad-buffer size.) | Stay at **TP=8** (32.6 GiB/GPU); efficiency comes from the MoE fix, not lower TP |

Everything static then measured clean at 35B: the engine is bitwise self-consistent (decode ==
prefill, **128/128, max 0.0**) and coherent; the offline TP=8 trainer scorer agrees with the engine
to `8e-7`; the trainer is batch-invariant (b=4 logits == b=1 bitwise) and mode-invariant
(train+grad == eval+no_grad bitwise); MoE at TP>1 is invariant to token count *and* neighbours
(`moe_tp_invariance_test.py`).

**The gate, live at 35B:** `[ZEROKL-DIFF] n=375441 mean=0.00000 frac>0.05=0.000% top-8 diffs =
[0.0]×8` — rollout logprobs equal the trainer's forward, **bitwise, on every one of 375,441 response
tokens**, corroborated by 2,197 same-weight probes all reading exactly `0.0` and every nonzero probe
falling strictly *after* an optimizer step. Reward climbs 0.58 → 0.79 over six steps; entropy and
grad-norm are stable. Zero-KL at 35B, no TIS/MIS.

---

## 11. Training efficiently without breaking zero-KL

The first working 35B run was ~130× slower than the non-zero-KL baseline. The cause was not the
batch-invariant kernels — it was **per-item Python loops** at a scale the 0.8B never reached. Each
was replaced with a batched op that is bitwise-equal *by construction* (the batch/tile axis never
enters a reduction), then validated on-GPU before use.

| Loop | Cost (35B, per forward) | Batched replacement | Speedup | Bitwise? |
|---|---|---|---|---|
| `SequentialMLP` expert loop (256 experts × 40 layers = 10,240 module calls) | 2140 ms | **fixed-capacity tiled `bmm`** (`moe_batched_experts.py`): scatter each expert's rows into `[n_tiles, CAP, h]`, one `bmm` per fc, ragged gather | **19×** (111 ms) | == sequential; probe row invariant to token count *and* under 9000× routing skew |
| `ChunkConsistentGDN.decode` per-slot loop (640 slots × 30 layers) | 504 ms/layer | **batched conv + ragged gather** (`gdn_ops.gdn_causal_conv_batched`) | **20×** (25 ms/layer) | 2700/2700 layer-parity tokens exact |
| trainer scoring / training at `micro_batch=1` | 640 micro-batches/pass | **`micro_batch=4`** | ~4× fewer | see below |

Micro-batching `> 1` had three latent `b==1` assumptions, each removed and re-proven bitwise:

- **Trainer attention** asserted `b == 1`. A `[sq, b, np, hn]` micro-forward *is* a varlen batch of
  `b` equal-length sequences — lay them out contiguously with `cu_seqlens = [0, sq, 2sq, …]`. Every
  sequence's output at b=4 is bitwise identical to running it alone at b=1 (verified at 35B TP=8
  head counts).
- **The GDN chunk VJP** (`_torch_chunk_gdr_one`) hardcoded batch 1 in the initial state and output
  reshape. The math was already batch-generic (the batch axis never reduces); use `q.shape[0]`. The
  chunk kernel's forward at b=4 is bitwise per-sequence == b=1.
- **The tiled expert GEMM** staged memory by `counts.max()` in its first form — memory scaling with
  *routing skew*, which OOMed at 35B profiling (32 GiB). Fixed-capacity tiles bound it to `T + E·CAP`
  regardless of skew (1.65 GiB) and make the tile shape a *constant*, a strictly stronger invariance
  argument.

Net: rollout **66 → ~10 min**, GPU utilization ~20% → 55–99%, step time ~3 h → ~30 min — while the
gate stays bitwise `0.0`.

> The tiled expert GEMM and the batched decode are **not** bitwise-equal to the sequential loops
> (different kernels round differently), and do not need to be: both runtimes run the *same* new
> function, so rollout == scoring == training numerics move together. Zero-KL requires determinism,
> invariance to other tokens, and invariance to the padding/tile amount — all three verified — not
> equality to the old loop.

---

## 12. Two metric traps

Two numbers *look* like the zero-KL gate and are not. Both cost real time.

**`policy_kl` is `0.0` by construction.** It is the within-trainer old-vs-new ratio at the first
inner PPO step; with per-minibatch old-logprob recompute it is identically zero. It says nothing
about rollout == train. Do not report it as the gate.

**`minibatch_rollout_logprobs_abs_diff` is not the gate either.** It compares rollout logprobs to the
old logprobs *recomputed per PPO minibatch*. With `train_batch_size / policy_mini_batch_size > 1`
there are multiple minibatches, and minibatch *k>1* is recomputed **after** minibatch *k−1*'s
optimizer step — so it legitimately measures the θ₀→θ₁ policy movement, not a numerical mismatch.
At 35B (128/64 = 2 minibatches) it reads ~1e-2; the 0.8B (64/64 = 1 minibatch) read ~8e-7 only
because there was no intervening step. The split is exact: every zero probe is *before* an optimizer
step, every nonzero probe *after*.

**The gate is `[ZEROKL-DIFF]` / `policy/rollout_train_logprobs_abs_diff`** — rollout vs the trainer's
forward on the *same* weights. It prints per-token top-8 diffs with the rollout/trainer values and
an **off-by-one check** (`d_here` vs `d_prev`/`d_next`) that distinguishes a genuine value
divergence from a shift/alignment bug. Read it first.

> **The 20-nat outlier, explained.** The 35B `minibatch_*` metric showed `max ≈ 20`. The probe's
> off-by-one check reads `d_here=21.9 < d_prev=30.7, d_next=26.9` — alignment is correct (a shift bug
> would make a neighbour ~0), mid-sequence, and it is a *post-optimizer* probe. It is one
> rollout-sampled **tail token** (logprob −8.7, i.e. temperature-1.0 occasionally forcing a
> `p≈1.6e-4` token out of a very peaked distribution) that minibatch-1's step suppressed to −30.7 in
> a negative-advantage sequence. On *identical* weights that same token matches bitwise `0.0`; the
> divergence is real policy movement, bounded in the loss by PPO clipping (`clip_ratio=0.0026`). Not
> a zero-KL violation.

---

## 13. The recipe

```bash
# structural
SKYRL_ZERO_KL=1                    # unified GPTModel on both ends + native weight sync
SKYRL_ZEROKL_LOCAL_SPEC=1          # Megatron LOCAL layer spec (no TransformerEngine)
VLLM_ENABLE_V1_MULTIPROCESSING=0   # in-process vLLM, so registrations reach the model build
                                   # (at TP>1 use the vllm.general_plugins entry point instead)

# batch invariance
VLLM_BATCH_INVARIANT=1             # aten mm/addmm/linear/softmax/log_softmax/rms_norm overrides
VARLEN_FORCE_NUM_SPLITS_1=1        # single-pass KV reduction => decode == prefill
NCCL_ALGO=allreduce:tree           # deterministic, degree-stable all-reduce
NCCL_MIN_NCHANNELS=1
NCCL_MAX_NCHANNELS=1
VLLM_USE_AOT_COMPILE=0
attention_backend="CUSTOM"         # zerokl/varlen_backend.py

# MoE
SKYRL_ZEROKL_MOE_DETERMINISTIC=1   # fixed-order expert combine + sorted router top-k + Triton mm on SM90
moe_grouped_gemm=false             # SequentialMLP
moe_router_dtype=fp32
moe_token_dispatcher_type=allgather
expert_model_parallel_size=1       # EP>1 => nondeterministic combine collective

# GDN (Qwen3.5 hybrid)
SKYRL_ZEROKL_GDN=1                 # fla shim + hybrid no-TE spec + chunk-consistent decode
SKYRL_ZEROKL_GDN_PIN_CONFIGS=1     # pin the racy FLA autotune configs
SKYRL_ZEROKL_GDN_NORM_ROWS=1       # pin RMSNormGated's tile height
SKYRL_ZEROKL_ENABLE_PREFIX_CACHE=0 # required for GDN; safe to enable on dense/MoE
SKYRL_ZEROKL_ENABLE_CHUNKED_PREFILL=0
SKYRL_ZEROKL_ENABLE_CUDAGRAPH=0

# matched TP>1 (Megatron TP == vLLM TP; both ends sharded identically)
tensor_model_parallel_size=N       # engine builds a TP-sharded GPTModel over vLLM's worker group
expert_tensor_parallel_size=N      # ETP == TP at EP=1 (force_zerokl_moe_config pins this)
                                    # NCCL pins (above) now forwarded to the TRAINER actors too
# vllm.general_plugins entry point installs the stack in every spawned mp worker

# 35B-A3B efficiency (bitwise-safe; both runtimes run the same batched op)
SKYRL_ZEROKL_MOE_BATCHED=1         # tiled bmm experts instead of the 256-iteration python loop
SKYRL_ZEROKL_MOE_TILE_ROWS=128     # tile height (multiple of the bmm kernel's BLOCK_SIZE_M)
SKYRL_ZEROKL_GDN_BATCHED_DECODE=1  # one conv + ragged gather instead of the per-slot loop
SKYRL_ZEROKL_SYNC_BUCKET_MB=512    # bucket the native weight sync (20,943 params -> ~33 chunks)
SKYRL_ZEROKL_MAX_BATCHED_TOKENS=0  # 0 => cap engine profiling at max_model_len (avoids fp32-logits OOM)
micro_forward_batch_size_per_gpu=4 # b>1 trainer micro-forward (varlen batch; proven bitwise)
micro_train_batch_size_per_gpu=4
logprobs_chunk_size=256            # bound the in-place TP>1 vocab gather

# also
apply_rope_fusion=false            # the fused TE RoPE kernel is not interceptable
tis_ratio_type=null                # TIS is unnecessary at zero KL
```

> **Operational rule, learned twice:** any new `SKYRL_ZEROKL_*` env var **must** be added to the
> actor env-forwarding allowlist in `skyrl/train/utils/utils.py`, or it silently never reaches the Ray
> actors and the run quietly does nothing.

---

## 14. Results

| Model | Path | Result | Log |
|---|---|---|---|
| Qwen3-4B dense | kernel-parity | logits bitwise, `r ≡ 1.0` | `SkyRL-ZeroKL-EVALUATION.md` |
| MiMo-7B dense | engine decode vs prefill | **256/256 bitwise, max 0.0** | `gdn_regress_dense_final.log` |
| MiMo-7B dense | live DP8 DAPO, steps 1–5 | `policy_kl = 0.0`, mean 4.8e-7, max 5e-6 | wandb `24xly50y` |
| OLMoE-1B-7B MoE | engine decode vs prefill | **256/256 bitwise, max 0.0** | `gdn_regress_moe_final.log` |
| OLMoE-1B-7B MoE | live DP8 DAPO, steps 1–5 | `policy_kl = 0.0`, mean 2.4e-7, max 2.0e-6 | `zerokl_nightly_dapo_olmoe.log` |
| Qwen3.5-0.8B GDN | native vLLM, 32×2048 tok | **65 536/65 536 exact, max 0.0** (baseline 2.52%, max 0.247) | `gdn_divergence_patched.log` |
| Qwen3.5-0.8B GDN | GPTModel-in-vLLM engine | **256/256 bitwise, max 0.0**, coherent | `gdn_gate31_hybrid.log` |
| Qwen3.5-0.8B GDN | **live DP8 GRPO on GSM8K, 6 steps** | **`policy_kl = 0.0` every step**, abs_diff 8.06e-7 … 8.31e-7, reward 0.3555 | `zerokl_gsm8k_qwen35_0.8b.log` |
| Qwen3.5-0.8B GDN | **live matched TP=2 GRPO, 4 steps** | **`policy_kl = 0.0`**, abs_diff 8.1e-7 … 8.3e-7 (= the TP=1 floor) | `zerokl_tp2_0.8b_r3.log` |
| Qwen3.5-0.8B GDN | matched TP=2 engine decode vs prefill | **512/512 bitwise, max 0.0** (was max 0.16 pre gated-norm fix) | `parity_tp2_fixed.log` |
| MiMo-7B dense | matched TP=2 engine decode vs prefill | **512/512 bitwise, max 0.0** | `parity_tp2_dense_mimo.log` |
| Qwen3.5-35B-A3B | TP=8 engine decode vs prefill | **128/128 bitwise, max 0.0**, coherent | `parity_35b_engine.log` |
| Qwen3.5-35B-A3B | TP=8 trainer scorer vs engine (offline) | mean 8e-7, max 3e-6 | `trainer_score_35b_tp8.log` |
| **Qwen3.5-35B-A3B** | **live matched TP=8 GRPO on GSM8K** | **rollout == trainer bitwise `0.0` on 375,441 tokens**; reward 0.58 → 0.79 / 6 steps; rollout 66 → ~10 min | `zerokl_gsm8k_35b_tp8_r15.log` |

The `~8e-7` live floor (vs the offline `0.0`) is the cross-runtime logprob-reduction floor of the
whole pipeline, not a GDN or MoE residual: the layers themselves are bitwise. `policy_kl` is exactly
`0.0`. **At 35B the gate is exact `0.0`**: the `[ZEROKL-DIFF]` probe reads bitwise-zero on every
same-weight token, so the ~8e-7 floor seen at smaller scale is not even present.

### The verification suite

| Script | Asserts |
|---|---|
| `zerokl/_selftest.py` | the four kernel identities (GEMM, log_softmax, RMSNorm, RoPE) |
| `gdn_batch_invariant.verify_gdn_batch_invariance()` | determinism + cross-sequence + prefix invariance of the chunk kernel |
| `moe_layer_invariance_test.py` | MoE layer decode-vs-prefill and grad-vs-nograd, both router regimes |
| `gdn_ops_invariance_test.py` | conv / l2norm / chunk invariance |
| `gdn_layer_decode_parity_test.py` | full GDN layer: prefill+decode == full-sequence forward, bitwise |
| `gdn_trainer_shim_test.py` | Megatron GDN builds w/o TE, thd fwd+bwd, forward == `gdn_ops` bitwise, q/k grads nonzero |
| `gdn_moe_hybrid_spec_test.py` | hybrid spec keeps MoE, pins SequentialMLP, expert mapping matches |
| `skyrl_engine_parity_test.py` | the real engine: coherent `[GEN]` **and** N/N bitwise (`PARITY_TP` for matched TP; `PARITY_DUMP` writes the rollout for the trainer scorer) |
| `trainer_score_tp.py` | TP-matched trainer scorer over a dumped engine rollout; `ZK_BATCH` (b>1 == b=1), `ZK_GRAD` (train == eval) self-checks |
| `moe_tp_invariance_test.py` | MoE at TP>1: a token's row is invariant to batch size and neighbours (dispatcher + ETP reduce-scatter + batched experts) |
| `gdn_decode_prefill_divergence.py` | end-to-end decode-vs-prefill over 65 536 tokens |
| `gdn_engine_{layer_bisect,core_probe,submodule_probe}.py` | localize the first bitwise divergence |

---

## 15. The lessons

1. **Bitwise or nothing.** A `1e-6` tolerance hides which op is wrong. An exact-zero gate turns every
   regression into a bisectable fact — and 1 ULP amplifies into 0.3 nats over 36 layers.

2. **Prefer one implementation over two agreeing implementations.** Running Megatron's GPTModel inside
   vLLM eliminated an unbounded class of bugs. The `fla` shim did the same for GDN. Every time we
   made the two runtimes *share code* rather than *match numerics*, a whole family of divergences
   disappeared.

3. **The bug is almost never in the algorithm.** GDN's chunk-consistent decode was provably bitwise at
   layer level before it was ever wired in. Every subsequent failure was wiring: a patch in the wrong
   process, buffers sized by the wrong quantity, a veto flag, a norm's tile height. When the math is
   proven, bisect the wiring.

4. **A self-consistent wrong model is bitwise.** Three times, a perfect parity number came from a
   model that had the wrong architecture, unloaded weights, or an unpatched engine. The parity gate
   cannot see any of them. Read the generated text; assert the architecture; assert weights differ
   from their init; count patch invocations.

5. **Fail loud.** Every silent degradation in this project — the expert-bias buffer, the sleep/wake
   θ₀ restore, the mapping that matched nothing, the plugin that never loaded — cost days. Raising at
   init costs seconds.

6. **Kernels selected by shape or mode are the recurring enemy.** The three hardest bugs are the same
   bug: a kernel chosen from something other than its inputs. `torch.compile`/`jit_fuser` specializes
   by shape (the TP=2 gated norm, §9.4); `RMSNormGated` picks its tile from the row count (§6.4);
   the router's top-k sorts by `is_grad_enabled()` (§4.2). The engine and trainer sit on opposite
   sides of every such switch. Grep for `@torch.compile`, `@jit_fuser`, `is_grad_enabled`,
   autotune, and any `if training:` on the forward path — that is your bug list.

7. **Batch a loop by construction, not by matching a kernel.** Every efficiency win here — tiled
   expert `bmm`, batched GDN conv, b>1 attention — is bitwise-safe for one reason: the new
   batch/tile axis never enters a reduction, so each item's rows are the same math as computing it
   alone. Prove *that*, and you do not have to prove the batched kernel equals the old loop (it
   doesn't, and needn't) — you only have to run the same function on both runtimes.

8. **Read the right metric.** `policy_kl` is `0.0` by construction; `minibatch_rollout_abs_diff`
   legitimately blows up with more than one PPO minibatch; a 20-nat outlier can be a healthy tail
   token under an optimizer step. The gate is rollout vs trainer *on the same weights*
   (`[ZEROKL-DIFF]`), and it comes with an off-by-one check. Misreading the metric cost a multi-hour
   bisect of code that was already bitwise — though that bisect did harden every component and
   produce the reusable TP-invariance harnesses.
