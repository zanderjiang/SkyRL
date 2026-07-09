# Bitwise zero-KL for GatedDeltaNet (Qwen3.5 hybrid)

Branch `zerokl-moe`. All numbers below were produced by the exact command shown, on this box
(8xH100, `--extra zerokl` = nightly torch 2.14 + vLLM 1.0.dev, no TransformerEngine). Nothing is
called "validated" unless a log backs it.

**Headline: GDN decode is now bitwise identical to prefill. 65,536/65,536 tokens exact, max |diff| =
0.0**, against a baseline of 2.52% exact / max 0.247. The dense and MoE zero-KL paths are unregressed.

The remaining work for a Qwen3.5-35B-A3B training run is *not* GDN math -- it is that SkyRL's
production engine (Megatron `GPTModel` running inside vLLM) has no GDN/mamba state path at all. See
[What is NOT done](#what-is-not-done).

---

## Summary of gates

| # | Gate | Result | Log |
|---|------|--------|-----|
| 1 | Megatron GDN builds + thd fwd/bwd, forward bitwise == `gdn_ops` | **PASS**, max \|diff\| `0.000e+00` | `/mnt/local_storage/logs/gdn_gate1.log` |
| 2 | Engine decode == prefill, Qwen3.5-0.8B, 32x2048 @ temp 1.0 | **PASS**, 65536/65536 exact, max `0.000000e+00` | `/mnt/local_storage/logs/gdn_divergence_patched.log` |
| R1 | Regression: MiMo-7B dense engine parity | **PASS**, coherent + 256/256 bitwise, max `0.000000e+00` | `/mnt/local_storage/logs/gdn_regress_dense_final.log` |
| R2 | Regression: OLMoE-1B-7B MoE engine parity | **PASS**, coherent + 256/256 bitwise, max `0.000000e+00` | `/mnt/local_storage/logs/gdn_regress_moe_final.log` |
| R3 | Regression: GDN layer decode parity after the LRU refactor | **PASS**, 450/450 bitwise | `/mnt/local_storage/logs/gdn_layer_parity_recheck.log` |
| 5 | Rollout cost of chunk-consistent decode | **5.78x** slower at 16 seqs (1.96x at 1 seq) | `/mnt/local_storage/logs/gdn_rollout_cost.log` |
| 3.1 | Engine parity on Qwen3.5-0.8B through the GPTModel-in-vLLM wrapper | **PASS** -- coherent AND 256/256 bitwise, max `0.000000e+00` | `/mnt/local_storage/logs/gdn_gate31_hybrid.log` |
| T | Trainer builds the real Qwen3.5 hybrid, loads it, fwd+bwd | **PASS** -- 18 GDN + 6 attn; predicts `' Rome'`, prompt CE 1.321 | `/mnt/local_storage/logs/gdn_trainer_model.log` |
| 3.2 | Trainer-vs-engine parity on Qwen3.5 | **NOT RUN** | -- |
| 3.3 | Live 5-step DP8 run on Qwen3.5-0.8B | **NOT RUN** | -- |

> **A near-miss worth recording.** An earlier Gate 3.1 run reported `256/256 bitwise, max 0.0` and
> the number was worthless: the generation check in the same log read
> `'The capital of France is' -> ' 0 -s\n\n(3 192=".,)5S;'`. The no-TE layer spec had built 24 dense
> attention layers instead of 18 GDN + 6 attention, so the model loaded none of the checkpoint's GDN
> weights -- and a model with the wrong weights is trivially self-consistent between its own decode
> and its own prefill. The `[GEN]` line is why the test prints it. `gptmodel_vllm` now **raises** when
> `SKYRL_ZEROKL_GDN=1` and zero GDN layers are found.

---

## Gate 1 -- trainer shim

```
CUDA_VISIBLE_DEVICES=0 HF_HOME=/mnt/local_storage/hf SKYRL_ZEROKL_GDN=1 SKYRL_ZEROKL_LOCAL_SPEC=1 \
  uv run --isolated --extra zerokl python examples/zerokl/nightly/gdn_trainer_shim_test.py \
  > /mnt/local_storage/logs/gdn_gate1.log 2>&1
```

```
1. GatedDeltaNet built (no transformer_engine). in_proj_dim=8224
2. thd forward OK: out (395, 1, 1024), 395 tokens over 3 seqs (7 chunks of 64)
   backward OK: |d hidden| = 185.6170; q/k proj grads nonzero (|dWq|=5521.55, |dWk|=5525.21)
3. trainer forward vs direct gdn_ops reference: max |diff| = 0.000e+00
4. VJP reference vs bitwise kernel forward: relative L2 = 3.333e-03 (bf16 vs fp32)
RESULT: GATE 1 PASS
```

`gdn_fla_shim.install_fla_shim()` registers an `fla` facade in `sys.modules`
(`fla.ops.gated_delta_rule`, `fla.modules.convolution`, `fla.modules.l2norm`) backed by `gdn_ops`, so
the trainer executes literally the engine's ops. It must run before anything imports
`megatron.core.ssm.gated_delta_net`, which binds `chunk_gated_delta_rule` at import time; the
authoritative call is in `zerokl/__init__.py`, with idempotent belt-and-braces calls in
`megatron_worker.py` and `gptmodel_vllm.py`.

### Two deviations from the plan, both forced by measurement

**(a) The vendored chunk kernel has no backward.** vLLM's
`ChunkGatedDeltaRuleFunction` defines a `forward` and no `backward` -- it is an inference-only
vendoring of FLA. Gate 1 asks for `forward+backward` on the packed path, and it raised
`NotImplementedError` the first time it was run. Installing the real `flash-linear-attention` was
rejected: a second copy of the chunk kernel means a second autotune decision, which is the exact
divergence this workstream exists to remove.

`gdn_chunk` therefore keeps the vLLM kernel in the forward -- that is what makes decode and training
agree bitwise -- and supplies the vector-Jacobian product by differentiating `_torch_chunk_gdr`, the
fp32 torch reference for the same function, at the same inputs. Zero-KL constrains the *forward*
logprobs (they set the importance ratio); the gradient only has to be the gradient of that forward,
to floating-point accuracy. Item 4 of Gate 1 asserts the reference agrees with the kernel forward
(relative L2 3.3e-3, bf16 kernel vs fp32 reference) -- otherwise the backward would be
differentiating the wrong function. Under `torch.no_grad` (both rollout paths and the trainer's
scoring forward) `gdn_chunk` calls the kernel directly, so decode pays nothing for this.

Cost: backward recomputes the layer in fp32 with a 64-iteration python loop. It is the slowest part
of a GDN training step. The fix, if it becomes the bottleneck, is a real fused backward -- not a
different forward.

**(b) `gdn_l2norm` silently zeroed the q/k gradients.** It was a bare `l2norm_fwd` Triton launch
writing into `torch.empty_like(x)`: no autograd history, so backprop delivered **zero** gradient to
`query` and `key`. The layer still trains through `v` and the loss still falls, so this would have
shipped. Same treatment as the chunk kernel: kernel forward, autograd of `x * rsqrt(sum(x^2) + eps)`
in the backward. Gate 1 now asserts the q/k projection gradients are *nonzero*, not merely finite.

Also: megatron's `_compute_g_and_beta` and `_prepare_qkv_for_gated_delta_rule` carry `@jit_fuser`
(= `torch.compile` on torch >= 2.2). A compiled `exp`/`softplus` is Triton's `libdevice` version, not
ATen's, and they disagree in the last ulp. The shim rebinds both to eager
(`SKYRL_ZEROKL_GDN_EAGER_PREP=0` to A/B). And `gdn_chunk` clones `cu_seqlens` before
`prepare_chunk_indices`, which is `@tensor_cache`d on tensor *identity* over vLLM's recycled metadata
buffers.

---

## Gate 2 -- the headline gate

```
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=examples/zerokl/nightly/_torchvision_stub \
  HF_HOME=/mnt/local_storage/hf SKYRL_ZEROKL_GDN=1 \
  uv run --isolated --extra zerokl python examples/zerokl/nightly/gdn_decode_prefill_divergence.py \
  > /mnt/local_storage/logs/gdn_divergence_patched.log 2>&1
```

```
=== GDN decode-vs-prefill | model=Qwen/Qwen3.5-0.8B nseq=32 ntok=2048
    VLLM_BATCH_INVARIANT=True chunk_consistent_decode=True ===
tokens compared : 65536
exact 0.0       : 65536/65536 (100.00%)
NaN diffs       : 0
mean |diff|     : 0.000000e+00
P99 |diff|      : 0.000000e+00
max |diff|      : 0.000000e+00
patched _forward_core calls: 37008
RESULT: GATE 2 PASS
```

Baseline for comparison (`/mnt/local_storage/logs/gdn_divergence_v4.log`): mean 1.67e-2, P99 0.124,
max 0.247, 2.52% exact.

The torchvision import stub is vendored at `examples/zerokl/nightly/_torchvision_stub/` (vLLM's
Qwen3.5 module imports the VL chain unconditionally; `transformers.image_utils` needs a real
`InterpolationMode` enum and `torchvision.transforms.v2.functional` to exist).

### Four wiring bugs. Zero algorithm bugs.

The layer-level test (`gdn_layer_decode_parity_test`, 450/450 bitwise) already proved the math, so
every failure below was chased in the wiring, as instructed. That discipline was correct: the
algorithm was never touched.

1. **The patch never reached the model.** vLLM v1 runs the model in an `EngineCore` *subprocess* by
   default, which imports its own copy of the class. The parent-process monkey-patch was silently
   ignored, and the "patched" run reproduced the baseline numbers *exactly* (mean 1.647e-2).
   Fixed with `VLLM_ENABLE_V1_MULTIPROCESSING=0`. `gdn_engine_patch.CALL_COUNT` now counts
   `_forward_core` invocations, and the test refuses to report a number if it is zero. (This guard
   immediately paid for itself: it caught the same silent no-op in the cost benchmark.)

2. **Open-chunk buffers were sized by the engine's slot count.** `ssm_state.shape[0]` is ~8686 --
   vLLM turns all leftover memory into mamba state slots. One buffer per slot is 6.8 GiB *per layer*
   (OOM on the first forward). Buffers are now sized by the scheduler's `max_num_seqs`, with a small
   LRU mapping live slot ids onto buffers. This is safe because only `prefill` establishes a mapping
   and only `prefill` starts a sequence, so an eviction can only ever hit a slot whose request is
   gone; `decode` raises rather than guess.

3. **vLLM vetoes batch invariance for GDN.** `get_mamba_attn_backend` raises
   `"batch_invariant mode is not supported for GDN_ATTN"` because the stock recurrent decode has no
   invariant form. With chunk-consistent decode it does, so the patch lifts the veto. Without this
   the model's 6 softmax-attention layers can never be made invariant either, and they carry their
   own ~1e-2 decode-vs-prefill gap.

4. **The residual: `RMSNormGated`, the GDN output norm.** `fla.ops.layernorm_guard.layer_norm_fwd`
   picks its Triton tile height from the row count:

   ```python
   rows_per_block = min(next_power_of_2(cdiv(M, 2 * sm_count)), 4)
   ```

   and the kernel reduces a `[ROWS_PER_BLOCK, BLOCK_N]` tile with `tl.sum(x, axis=1)`. The tile shape
   decides how a row's 128 elements are spread across threads, hence the order of the fp32 reduction,
   hence the last bit of `rstd`. On H100 (132 SMs) a GDN decode step has
   `M = num_tokens * num_v_heads = 16` -> **1 row**, while the prefill that rescores those same tokens
   has `M = 3472` -> **4 rows**. Same input row, different bits, occasionally.
   `pin_gdn_rmsnorm_rows_per_block()` pins the tile height to a constant
   (`SKYRL_ZEROKL_GDN_NORM_ROWS`, default 1).

### How #4 was localized (the bisect chain)

Each step is a committed, re-runnable script. This is the part worth keeping.

| Script | Finding |
|---|---|
| `gdn_engine_layer_bisect.py` | Decoder layer 0 first diverges at decode step 50 (2.4e-4); deeper layers diverge **earlier** and larger (layer 23 at step 20, 6.4e-2). Deeper-earlier is the signature of a tiny error present *everywhere* and amplified by depth -- not of an onset event. |
| `gdn_engine_core_probe.py` | Layer 0's GDN core is **bitwise on both sides**: its `mixed_qkv` input *and* its `core_attn_out` output, at every generated position, across all three chunk rolls (64/128/192). Chunk-consistent decode was correct; the bug was outside it. |
| `gdn_engine_submodule_probe.py` | `in_proj_qkvz` clean, GDN core clean, `out_proj` first bad at abs pos 67 -> the culprit is the gated norm sitting between them. |

A trap worth recording: the first bisect run used a 12-token generation from a 17-token prompt. It
reported every layer bitwise -- because it never crossed a chunk boundary (C = 64) and never
triggered the softmax layers' split-K heuristic. It proved nothing. `GDN_NGEN` now defaults to 200.

`gdn_engine_submodule_probe`'s row for `linear_attn.norm` is meaningless: `_output_projection`
reshapes to `[tokens * num_v_heads, head_dim]`, so its rows are (token, head) pairs, not tokens. Read
the token-indexed rows only.

---

## Cost (Gate 5)

```
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=examples/zerokl/nightly/_torchvision_stub \
  HF_HOME=/mnt/local_storage/hf uv run --isolated --extra zerokl \
  python examples/zerokl/nightly/gdn_rollout_cost.py > /mnt/local_storage/logs/gdn_rollout_cost.log 2>&1
```

Three arms, because a two-arm A/B would blame chunk-consistent decode for the whole zero-KL stack.
`bi` = batch-invariant kernels + `num_splits=1` CUSTOM varlen backend + Triton matmuls, but vLLM's
**stock** GDN decode (so: not bitwise). Qwen3.5-0.8B.

| arm | 16 seqs x 512 tok | 1 seq x 512 tok |
|---|---|---|
| `stock` vLLM (not bitwise) | 533.2 gen tok/s | 34.6 gen tok/s |
| `bi` (not bitwise) | 426.7 gen tok/s (1.25x) | 28.2 gen tok/s (1.23x) |
| `cc` chunk-consistent (**BITWISE**) | 73.8 gen tok/s (**5.78x vs bi**) | 14.4 gen tok/s (**1.96x vs bi**) |

**Price of chunk-consistent decode alone: 5.78x at 16 concurrent sequences. End-to-end price of
bitwise zero-KL: 7.22x.**

The theoretical cost is `~(C+1)/2 = 32.5x` token-rows *on the GDN layers only*, so 5.78x overall is
in the right range -- but the ratio **grows with concurrency** (1.96x at 1 seq -> 5.78x at 16). Cost
that scales with the number of sequences is not FLOPs; it is the per-slot python loop in
`ChunkConsistentGDN.decode`, which runs `_prep` (a conv + two l2norms) once per slot per GDN layer:
16 slots x 18 GDN layers ~= 288 launch groups per decode step.

That is a fixable engineering cost, not an inherent one. The open chunks can be padded to `C` and
prepped in a single batched call: the conv is elementwise and l2norm is row-local, so batching them
is bitwise-safe *by construction* (the same argument that makes `gdn_ops` invariant in the first
place). This is the first thing to do before a large-scale run, and it is not done here -- the number
above is what the code costs today.

`C` (`FLA_CHUNK_SIZE`) is the other knob, and it must match on both sides because it defines the
chunk grid the recurrent state is pinned to. It was not changed.

---

## Gate 3.1 -- Qwen3.5 through the production engine

```
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=examples/zerokl/nightly/_torchvision_stub \
  HF_HOME=/mnt/local_storage/hf SKYRL_ZEROKL_GDN=1 PARITY_TEMP=1.0 PARITY_MM_ZERO=1 \
  PARITY_MAX_NUM_SEQS=8 ZEROKL_MODEL=Qwen/Qwen3.5-0.8B \
  uv run --isolated --extra zerokl python examples/zerokl/nightly/skyrl_engine_parity_test.py \
  > /mnt/local_storage/logs/gdn_gate31_hybrid.log 2>&1
```

```
[ZEROKL-SPEC] hybrid local spec: 18 GatedDeltaNet + 6 attention layers (no TransformerEngine)
[ZEROKL-GDN] swapped 18 Megatron GatedDeltaNet layer(s) -> chunk-consistent decode
[GEN] 'The capital of France is' -> ' Paris.\nThe capital of France is Paris. ...'
MAX |decode - prefill| over 256 tokens = 0.000000e+00
tokens EXACT 0.0: 256/256
RESULT: BITWISE-IDENTICAL (max==0)
```

`zerokl/gdn_gptmodel.py` is the `swap_core_attention` analogue for GatedDeltaNet:
`ZeroKLGDNStateLayer` is a real `MambaBase` (vLLM enumerates KV-cache layers with an `isinstance`
filter, so duck typing leaves the layer invisible and the engine runs with empty GDN metadata). It
exists to make vLLM reserve mamba slots and emit `GDNAttentionMetadata`; its `kv_cache` tensors are
deliberately unused, because `ssm_state[slot]` now means "state at the last chunk boundary".
`_gdn_inference_forward` keeps every Megatron module and routes only the conv+chunk core through
`ChunkConsistentGDN`.

`zerokl/gdn_hybrid_spec.py` supplies the hybrid **no-TE** layer spec. Megatron's own
`get_transformer_block_with_experimental_attention_variant_spec` asserts
`transformer_impl == "transformer_engine"`, and its GDN spec asks the backend for
`column_parallel_layer_norm_linear()` -- TE's fused layernorm+linear, which `LocalSpecProvider`
answers `None`. We assemble the block from local modules instead: `in_proj = ColumnParallelLinear`
with a separate `input_layernorm`, exactly the layer Gate 1 validates.

Seven blockers, each found by running it and fixed in turn:

1. `fla.__spec__ is None` -> `importlib.util.find_spec` raised inside
   `transformers.is_flash_linear_attention_available()`. The facade now carries a real `ModuleSpec`
   and declares `__version__ = "0.0.0"`, so HF answers "no FLA" and never imports
   `fused_recurrent_gated_delta_rule` from it.
2. The bridge dispatched to the Qwen3.5 **VL** model; force the text bridge in the wrapper too.
3. `layernorm_zero_centered_gamma`. Qwen3.5 normalises `rms(x) * (1 + w)`; Megatron's no-TE
   `WrappedTorchNorm` asserts against the flag. `zerokl/zero_centered_norm.py` implements it for both
   runtimes.
4. **M-RoPE.** Qwen3.5's config carries `rope_parameters.mrope_section`, so vLLM feeds `[3, T]`
   positions and requires `SupportsMRoPE`. The wrapper satisfies the protocol and collapses the
   three (identical, for text) sections.
5. **No hybrid spec** -- the model came out as 24 dense layers. See the near-miss box above.
6. **Bridge weight mapping.** `qwen35_bridge` maps
   `self_attention.in_proj.layer_norm_weight` (TE's fused parameter, absent under the local spec) and
   builds HF names as `model.layers.*` while the checkpoint stores `model.language_model.layers.*`.
   Both retargeted. Also `ChunkedMapping.get_shard_idx` builds its index tensors with a bare
   `torch.arange`, which lands on the GPU inside vLLM's default-device context while the HF weights
   are still on the CPU -- pinned to CPU (the subclasses shadow the base, so all of them need it).
7. **vLLM's hybrid plumbing.** `cache_config.mamba_block_size` is only set for architectures in
   `MODELS_CONFIG_MAP`, and the attention/mamba page-size reconciliation is gated on the model
   class's `is_hybrid` and reads its `get_mamba_state_{shape,dtype}_from_config` classmethods --
   all before any layer exists. All three registered/implemented.
8. **The modelinfo cache.** `is_hybrid` was first written as an env-dependent class attribute on the
   one wrapper. That silently broke the *next* run of a different model:

   ```
   AttributeError: 'MiMoConfig' object has no attribute 'linear_num_key_heads'
   ```

   vLLM persists each model's `_ModelInfo` -- which carries `is_hybrid` -- to
   `~/.cache/vllm/modelinfos/<module>-<class>.json`, keyed by module+class name and validated only
   against the source hash. A GDN run baked `is_hybrid: true` into the dense wrapper's cache entry,
   and MiMo then went down the mamba page-size path. Fixed with a separate
   `GPTModelVLLMHybridWrapper` class (its own cache file) selected at import from the env var.
   Lesson: a model class's vLLM-visible capabilities must be a property of the CLASS, not of the run.

---

## The trainer half, on the real model

```
CUDA_VISIBLE_DEVICES=2 HF_HOME=/mnt/local_storage/hf SKYRL_ZEROKL_GDN=1 SKYRL_ZEROKL_LOCAL_SPEC=1 \
  uv run --isolated --extra zerokl python examples/zerokl/nightly/gdn_trainer_model_test.py \
  > /mnt/local_storage/logs/gdn_trainer_model.log 2>&1
```

```
1. GPTModel: 18 GatedDeltaNet + 6 attention layers (no transformer_engine)
2. GDN weights loaded: |dt_bias| mean 5.4707, A_log mean -1.9795, conv1d std 0.0728
3. forward OK: logits (297, 248320) over 297 tokens (5 chunks of 64)
   backward OK: random-token CE = 14.989 (ln(vocab) = 12.422); |dW GDN in_proj| = 14.481
4. 'The capital of France is Paris. ... The capital of Italy is' -> next token ' Rome'; prompt CE = 1.321
RESULT: PASS
```

Two assertions here that a bitwise number cannot make: the architecture really is the hybrid (not 24
dense layers), and the checkpoint really landed in the GDN parameters (`dt_bias` is not the init's
all-ones, and the model completes `Italy is` with ` Rome`).

---

## What is NOT done

* **Packed (thd) training on a GDN hybrid is untested and probably broken.** With
  `PackedSeqParams(qkv_format="thd")`, Megatron hands `core_attention` a 3-D `q` of `[T, np, hn]`
  (the batch dim folded away), while the zero-KL `TorchVarlenCoreAttn` asserts the 4-D sbhd
  `[sq, b=1, np, hn]` layout:

  ```
  AssertionError: TorchVarlenCoreAttn supports the b=1 micro-forward (micro_*_batch_size_per_gpu=1)
  [PROBE] q (297, 8, 256) k (297, 2, 256) v (297, 2, 256)
  ```

  The trainer-model test above therefore runs UNPACKED. The GDN layers' own packed path is bitwise
  (Gate 1), so this is confined to `megatron_varlen_attn`. Whether the production trainer actually
  passes `thd` depends on the data path -- **check this before the 35B run**, because sample packing
  is the normal SkyRL configuration.
* **Gate 3.2 (trainer-vs-engine parity on Qwen3.5)** has not been run. Both halves are validated
  separately (Gate 1 + the trainer-model test; Gate 3.1), but native weight sync between them has not
  been exercised on a hybrid model.
* **Gate 3.3 (live 5-step DP8)** has not been run. Gate on
  `policy/rollout_train_logprobs_abs_diff_mean <= 1e-6` at **every** step including 2-5, and
  `policy_kl == 0.0`. A clean step 1 with a dirty step 2 is the sleep/wake weight-clobber class of
  bug, not a GDN bug -- set `SKYRL_ZEROKL_DEBUG=1` and check
  `[ZEROKL-REAPPLY] == [SENDER] == [ZEROKL-ENGFWD]`.
* **`Qwen/Qwen3.5-35B-A3B-Base` is not downloaded** on this box (28T free, so not a capacity issue).
  It is also MoE, so it additionally exercises `make_zerokl_hybrid_local_spec`'s MoE branch
  (`num_experts` is threaded through to `get_gpt_layer_local_spec`) and the `_get_moe_lm_mappings`
  retarget -- neither has been run.
* **The 5.78x rollout cost.** Batch `ChunkConsistentGDN.decode`'s per-slot `_prep` before any
  large-scale run: it is bitwise-safe by construction (conv is elementwise, l2norm row-local), and at
  35B's concurrency the per-slot python loop will dominate.
* The launcher's CAVEAT block (`examples/train/zerokl/run_megatron_qwen3.5_35b_a3b_gsm8k_zerokl.sh`)
  still says the GDN divergence is unresolved. That is now stale.

Recommended order: Gate 3.2 -> batch `_prep` -> Gate 3.3 on Qwen3.5-0.8B -> download 35B-A3B -> the
MoE-hybrid spec/mapping path -> GSM8K.

---

## Files

| File | Role |
|---|---|
| `zerokl/gdn_fla_shim.py` | `install_fla_shim()` -- `fla` facade over `gdn_ops`; eager rebind of megatron's `jit_fuser` helpers |
| `zerokl/gdn_ops.py` | one impl per op for both runtimes; `_GdnChunkAutograd` / `_GdnL2NormAutograd` add the missing backwards |
| `zerokl/gdn_batch_invariant.py` | `pin_fla_autotune_configs()`, **`pin_gdn_rmsnorm_rows_per_block()`**, `verify_gdn_batch_invariance()` |
| `zerokl/gdn_chunk_consistent.py` | `ChunkConsistentGDN` -- open-chunk buffers + slot LRU, `prefill()` / `decode()` |
| `zerokl/gdn_engine_patch.py` | rebinds `_forward_core`; `assert_engine_args_compatible()`; lifts vLLM's GDN batch-invariance veto |

Env vars (all forwarded to Ray actors in `skyrl/train/utils/utils.py`):
`SKYRL_ZEROKL_GDN` (master gate, default off), `SKYRL_ZEROKL_GDN_PIN_CONFIGS`,
`SKYRL_ZEROKL_GDN_CONFIG_INDEX`, `SKYRL_ZEROKL_GDN_EAGER_PREP`, `SKYRL_ZEROKL_GDN_NORM_ROWS`.

Chunk-consistent decode redefines `ssm_state[slot]` as the state at the last **chunk boundary**.
`assert_engine_args_compatible` therefore raises at engine init on prefix caching, chunked prefill,
speculative decoding, and CUDA graphs. Those three (minus spec) were previously proven bitwise-safe
for the softmax layers and are worth 4.6x rollout; they are off because the GDN path cannot support
them yet. Re-enabling them for GDN is a follow-up, not a bug.
