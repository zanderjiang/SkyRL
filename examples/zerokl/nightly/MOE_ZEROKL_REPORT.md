# Bitwise zero-KL for MoE (Workstream A) + GDN divergence measurement

Status as of 2026-07-09: **implementation complete, GPU gates NOT RUN** (the headnode's GPUs were
occupied for the whole window; no free device was ever allocated to this task). Every gate below
lists the exact command to run and the log path it should write to. Nothing in this report is
claimed as validated unless it says "VALIDATED" and names the evidence.

Two things *were* verified without a GPU, on CPU, and are reported as such:

* the new fixed-order expert combine reproduces megatron-core's `scatter_add_` combine exactly in
  fp64 (where summation order is immaterial), and is per-token invariant to batch size;
* the `sorted=torch.is_grad_enabled()` top-k finding is real: on 20 000 random 8-expert rows with
  top-k 4, permuting top-k's return order changes the post-softmax routing probabilities bitwise on
  **4103 / 20000 (20.5%)** of rows.

---

## What was built

All MoE-specific logic is in one new module, `skyrl/backends/skyrl_train/zerokl/moe_batch_invariant.py`,
applied identically to the trainer GPTModel and the in-vLLM engine GPTModel. Dense providers hit an
early `return` in every entry point, so the validated MiMo-7B / Qwen3-4B path executes byte-identical
code to before.

| Piece | Where |
|---|---|
| `force_zerokl_moe_config` — pins SequentialMLP, allgather dispatcher, fp32 router, no fusions, no token dropping | `zerokl/moe_batch_invariant.py` |
| `make_zerokl_local_layer_spec` — `get_gpt_layer_local_spec(num_experts=…, moe_grouped_gemm=False)`, preserving the model's own `SelfAttention` | same |
| `enable_moe_deterministic_ops` — fixed-order expert combine + sorted router top-k | same |
| `patch_olmoe_bridge_for_sequential_mlp` — retargets megatron-bridge's expert weight mapping | same |
| trainer wiring | `workers/megatron/megatron_worker.py` (bridge patch before `AutoBridge.from_hf_pretrained`; spec at the `SKYRL_ZEROKL_LOCAL_SPEC` branch; `prepare_zerokl_moe` immediately before `provider.finalize()`, i.e. after `transformer_config_kwargs` so nothing can override it) |
| engine wiring | `zerokl/gptmodel_vllm.py` (`GPTModelVLLMWrapper.__init__`, same three hooks in the same order) |
| env forwarding | `train/utils/utils.py` allowlist — `SKYRL_ZEROKL_MOE_DETERMINISTIC` |
| launch script | `examples/zerokl/run_megatron_dapo_olmoe_1b7b_zerokl_nightly.sh` |

**New env flag.** `SKYRL_ZEROKL_MOE_DETERMINISTIC` (default `1`). It only has an effect on the MoE
local-spec path, which did not previously exist; set it to `0` to A/B the unpatched, batch-variant
megatron-core behaviour. It is in the actor env-forwarding allowlist.

---

## 1c. Determinism audit of dispatch / combine

Paths are relative to `/mnt/local_storage/zerokl-nightly-venv/lib/python3.12/site-packages/megatron/`.

| # | Site | Verdict |
|---|---|---|
| 1 | `core/transformer/moe/moe_utils.py:528` — `output_tokens.scatter_add_(0, sorted_indices.unsqueeze(1).expand(-1, hidden), permuted_tokens)` in `unpermute` | **BROKEN — fixed** |
| 2 | `core/transformer/moe/moe_utils.py:777` — `torch.topk(scores, k=topk, dim=1, sorted=torch.is_grad_enabled())` in `topk_routing_with_score_function._compute_topk` | **BROKEN when `moe_router_pre_softmax=False` — fixed** |
| 3 | `core/transformer/moe/moe_utils.py:413-427` — `routing_map.bool().T.contiguous().reshape(-1).argsort(descending=True, stable=True)` then `tokens.index_select(0, sorted_indices)` in `permute` | SAFE |
| 4 | `core/transformer/moe/token_dispatcher.py:310` — `local_probs = probs.T.contiguous().masked_select(local_map.T.contiguous())` | SAFE |
| 5 | `core/transformer/moe/experts.py:1235-1249` — `torch.split(permuted, tokens_per_expert)` → per-expert `MLP` loop → `torch.cat` in `SequentialMLP.forward` | SAFE **given** `moe_grouped_gemm=False` |
| 6 | `core/transformer/moe/moe_utils.py:1269-1277` — `RouterGatingLinearFunction.forward` (`te_general_gemm` if TE else `torch.mm`) | SAFE on this stack |
| 7 | `core/transformer/moe/router.py:573-581` — `_apply_expert_bias` / `moe_router_enable_expert_bias` | UNSAFE — rejected at config time |
| 8 | `core/transformer/moe/moe_utils.py:511` (`unpermute` prob path) and `router.py:638-670` aux-loss / z-loss `MoEAuxLossAutoScaler.apply` | SAFE |

### 1 — the expert combine (the headline bug)

`unpermute` is the MoE combine: it sums each token's top-k prob-weighted expert outputs back into
that token's row. With `moe_router_topk=8`, `sorted_indices` names each destination row **8 times**,
so CUDA's `scatter_add_` lowers to `atomicAdd`. The eight contributions to a token land in
hardware-arbitrary order. That is nondeterministic run-to-run *and* batch-variant: a token decoded
alone accumulates its eight adds in a different order than the same token inside a 512-token
prefill, so `decode != prefill` in the last bits — which is exactly the invariant zero-KL rests on.

megatron-core already has a deterministic branch (`output_tokens.index_add_`), but it is guarded by
`torch.are_deterministic_algorithms_enabled()`, which this stack does not set — and `index_add_`'s
deterministic lowering is an internal detail, not a documented batch-invariance guarantee.

**Fix** (`_fixed_order_combine`): `permute` lays rows out expert-major, so a token's k rows sit at
ascending row positions in ascending expert order. A *stable* `argsort(sorted_indices)` groups each
token's k rows together in that order; gathering column `j` for all tokens and adding the k columns
in `j = 0 … k-1` gives every token the same summation order no matter how many tokens share the
batch. No atomics, no cross-token reduction.

Verified on CPU (no GPU needed) — `max |fixed_order − scatter_add|` in fp64 is exactly `0.0`, and a
token computed alone reproduces its row from the full-batch combine bitwise:

```
$ /mnt/local_storage/zerokl-nightly-venv/bin/python  # see the snippet in this section's commit message
shape (37, 5) max abs err vs scatter_add reference: 0.0
token-alone == full-batch row: OK for all probed tokens
ragged layout -> None      # capacity-dropped layouts fall back rather than mis-sum
```

Side effect worth knowing: the backward of `index_select` is an `index_add`, so *gradient*
accumulation over experts is now nondeterministic where it used to be a deterministic `gather`.
This does not touch zero-KL (a forward-path property) but it does mean MoE gradients are no more
reproducible than dense ones already are.

### 2 — the router's top-k `sorted` flag

`torch.topk(..., sorted=torch.is_grad_enabled())`. When `moe_router_pre_softmax=False`, megatron
softmaxes **over the top-k scores in the order top-k returned them** (`scores, top_indices =
compute_topk(logits, …); probs = torch.softmax(scores, dim=-1)`). A different order sums the
denominator's k exponentials differently, so the probabilities differ in the last bits. The rollout
engine runs under `no_grad` and the trainer's training forward under grad, so the two disagree
*by construction* — this is not a batch-size effect, it is a grad-mode effect, and no amount of
batch-invariant aten overrides would catch it.

Measured on CPU: permuting top-k's return order changes the scattered probs bitwise on
**4103 / 20000 (20.5%)** of random rows (8 experts, top-4, fp32).

OLMoE sets `moe_router_pre_softmax=True`, where probs are read off the full softmax and scattered by
index, so top-k order never reaches an arithmetic reduction — **this bug does not bite OLMoE.** It
would bite Qwen1.5-MoE, Mixtral, Qwen3-MoE and DeepSeek-style routers. Fixed unconditionally by
forcing `sorted=True` inside the router (`torch.topk` is swapped for the duration of the routing
call — the call site is a closure, so there is no argument to thread through, and the forward is
single-threaded).

Known limitation: `InferenceTopKRouter._compiled_topk_routing` is `@torch.compile`'d, so a swapped
`torch.topk` would not reach it. That router is only built under
`config.transformer_impl == "inference_optimized"`, which this path never sets.

### 5 — why `moe_grouped_gemm=False` is load-bearing

`SequentialMLP` loops the experts and runs each one's `MLP` on its own `[n_e, H]` chunk through
`aten::linear` / `aten::matmul`, which vLLM's batch-invariant override makes independent of `n_e`.
Grouped GEMM instead tiles a single kernel over the per-expert token counts, so an expert's output
depends on how many tokens routed to it — batch-variant by construction. Hence the hard pin, and
hence grouped GEMM stays out of scope as a later optimization that would need its own invariant
kernel.

### 7 — `expert_bias` is a buffer, not a parameter

`moe_router_enable_expert_bias` registers `expert_bias` and `local_tokens_per_expert` as *buffers*.
`native_weight_sync.extract_native_weights` iterates `named_parameters()` only, so the engine's
routing bias would silently freeze at its init value while the trainer's drifted every step — the
two models would route differently from step 1 onward. `force_zerokl_moe_config` raises rather than
paper over it. OLMoE does not use expert bias.

---

## Two things the plan did not anticipate

**(a) OLMoE's provider needs a custom `SelfAttention`, and its bridge spec is TE-only.**
`megatron/bridge/models/olmoe/olmoe_provider.py:41` sets `transformer_layer_spec = olmoe_layer_spec`,
which is `default_layer_spec(config)` (→ TransformerEngine) with
`self_attention.module = OLMoESelfAttention`. That class exists because OLMoE norms q and k across
`num_heads * head_dim` (2048), not per-head (128) — a different model, not a different kernel.
Dropping to the plain local spec would have silently built per-head norms. `make_zerokl_local_layer_spec`
carries the custom attention class across; it is looked up by the original spec's `__name__` rather
than by resolving it, because resolving `default_layer_spec` imports TransformerEngine, which is
absent here.

**(b) megatron-bridge's OLMoE weight mapping only names the grouped-GEMM expert params.**
`olmoe_bridge.py:115,139` map `decoder.layers.*.mlp.experts.linear_fc{1,2}.weight*` — the
`TEGroupedLinear` naming. Under the SequentialMLP pin the real names are
`…experts.local_experts.N.linear_fc{1,2}.weight`, so **nothing would match and every expert weight
would stay at its random init** (the model would still build and generate — gibberish). The
DeepSeek and Ernie bridges already use the `local_experts.*` form;
`patch_olmoe_bridge_for_sequential_mlp` swaps OLMoE onto it, before `AutoBridge.from_hf_pretrained`
on both trainer and engine. This is the "small fix to the bridge provider" the plan allowed for;
the fallback to Qwen1.5-MoE-A2.7B was not needed.

If gate 1b shows nonzero `[ZEROKL-MISS]` or gibberish text, this mapping is the first place to look.

---

## Gates — commands and status

`ray stop` is never run. All GPU work goes to a free device via `CUDA_VISIBLE_DEVICES`; training
runs go to background with output redirected under `/mnt/local_storage/logs/`.

### 1a. Trainer-side MoE local spec — NOT RUN
Covered implicitly by 1e's startup, but to check in isolation, watch the trainer worker log for
`[ZEROKL-TRAINER] MoE zero-KL recipe pinned (experts=64 topk=8 pre_softmax=True): …` and
`[ZEROKL-SPEC] MoE local spec keeps custom SelfAttention OLMoESelfAttention`, and confirm no
`transformer_engine` import appears.
```
grep -E 'ZEROKL-(TRAINER|SPEC)|transformer_engine' /tmp/skyrl-logs/infra-*.log
```

### 1b. Engine builds, generates coherent text, zero `[ZEROKL-MISS]` — NOT RUN
```
SKYRL_ZERO_KL=1 SKYRL_ZEROKL_LOCAL_SPEC=1 VLLM_BATCH_INVARIANT=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
HF_HOME=/mnt/local_storage/hf CUDA_VISIBLE_DEVICES=<free-gpu> \
ZEROKL_MODEL=/mnt/local_storage/models/OLMoE-1B-7B-0924 \
uv run --isolated --extra zerokl python examples/zerokl/nightly/skyrl_engine_parity_test.py \
  > /mnt/local_storage/logs/olmoe_engine_parity.log 2>&1
```
`skyrl_engine_parity_test.py` already takes the model from `ZEROKL_MODEL` and prints a `[GEN]` line;
no edit was needed. Check `[GEN]` is coherent English (gibberish ⇒ expert weights did not load —
see (b) above) and that `/mnt/local_storage/zerokl_probe.log` has no `[ZEROKL-MISS]` entries.

### 1c. `moe_layer_invariance_test.py` bitwise — NOT RUN on GPU (combine verified on CPU, above)
```
SKYRL_ZEROKL_LOCAL_SPEC=1 VLLM_BATCH_INVARIANT=1 CUDA_VISIBLE_DEVICES=<free-gpu> \
uv run --isolated --extra zerokl python examples/zerokl/nightly/moe_layer_invariance_test.py \
  > /mnt/local_storage/logs/moe_layer_invariance.log 2>&1
```
Runs a single `MoELayer` built from the zero-KL local spec on a 512-token sequence and then the same
tokens one at a time, for both `moe_router_pre_softmax=True` (OLMoE) and `False` (the router-order
bug's regime), unpatched then patched. It asserts `max == 0.0` on the patched path and prints the
unpatched baseline so the magnitude of each bug is on the record. Exit code 1 on any nonzero.

### 1d. Engine parity: 256/256 bitwise, max == 0.0 — NOT RUN
Same command as 1b (the script does both the `[GEN]` spot-check and the 256-token
decode-vs-prefill comparison). Do not proceed to 1e with a nonzero max. On a nonzero max, bisect
per-layer with the existing `trace_layerwise.py` / `compare_layertrace.py` tooling; the first
diverging submodule should be `decoder.layers.N.mlp.…` — if it is `self_attention.core_attention`
instead, the MoE work is not the cause.

### 1e. Live pipeline, 5 steps, `policy/rollout_train_logprobs_abs_diff_mean ≤ 1e-6` at **every** step — NOT RUN
```
WANDB_API_KEY=<key> bash examples/zerokl/run_megatron_dapo_olmoe_1b7b_zerokl_nightly.sh \
  > /mnt/local_storage/logs/zerokl_nightly_dapo_olmoe.log 2>&1
```
(launch with `run_in_background`; DP8/TP1, 8K response length, all three rollout-accel gates on).
Steps 2–5 are the ones that matter: a clean step 1 with a dirty step 2 is the sleep/wake weight
clobber class of bug, not an MoE bug — set `SKYRL_ZEROKL_DEBUG=1` and check that the
`[ZEROKL-REAPPLY]`, `[SENDER]` and `[ZEROKL-ENGFWD]` checksums agree.

The `≤ 1e-6` threshold is what the MiMo-7B dense runs achieved. If OLMoE lands at, say, `1e-4`, that
is a finding to localize (per-layer bisect ⇒ which submodule), **not** a threshold to relax.

### Dense regression — NOT RUN
```
SKYRL_ZERO_KL=1 SKYRL_ZEROKL_LOCAL_SPEC=1 VLLM_BATCH_INVARIANT=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
CUDA_VISIBLE_DEVICES=<free-gpu> ZEROKL_MODEL=/mnt/local_storage/models/MiMo-7B-RL \
uv run --isolated --extra zerokl python examples/zerokl/nightly/skyrl_engine_parity_test.py \
  > /mnt/local_storage/logs/mimo_dense_regression.log 2>&1
```
Must still report `256/256` exact and `max == 0.0`. By construction every new code path is behind
`provider_is_moe(provider)`, which is False for MiMo-7B, so the dense forward should be untouched —
but that is an argument, not a measurement, and this gate is what turns it into one.

---

## Phase 2 — GDN decode-vs-prefill divergence: NOT MEASURED

`examples/zerokl/nightly/gdn_decode_prefill_divergence.py` is written and syntax-checked but has not
been run: it needs a GPU. Measurement only — no GDN fix was attempted, per scope.

```
CUDA_VISIBLE_DEVICES=<free-gpu> uv run --isolated --extra zerokl \
  python examples/zerokl/nightly/gdn_decode_prefill_divergence.py --both \
  > /mnt/local_storage/logs/gdn_divergence.log 2>&1
```

Plain native vLLM (no zero-KL wrapper, no Megatron, no Ray) on `Qwen/Qwen3.5-0.8B`: generate N=32
sequences × 2048 tokens at temperature 1.0 with logprobs, rescore the exact generated ids via
prefill in the same engine, report `exact-0.0` count, mean / P50 / P99 / max `|decode − prefill|`,
and the diff-vs-position curve bucketed into 16 bins. `--both` spawns one subprocess per mode
(`VLLM_BATCH_INVARIANT` unset, then `=1`) so the env is set before vLLM imports. The
batch-invariant run is the interesting one: it removes ordinary kernel batch-variance and leaves the
GatedDeltaNet chunk-scan-vs-recurrent-step gap on its own.

Expected shape of the answer (to be replaced with real numbers): softmax-attention layers under the
`num_splits=1` kernel contribute exactly zero, so any nonzero spread is the linear-attention path,
and it should *grow with position* as the recurrent state accumulates divergence — that curve, not
the scalar max, is what the GDN workstream needs.

---

## Deviations from the plan

1. **`get_gpt_layer_local_spec` vs `local_layer_spec`.** The plan said to build the MoE spec via
   `get_gpt_layer_local_spec(num_experts=…, moe_grouped_gemm=False, qk_layernorm=…)` "instead of the
   dense local spec". megatron-bridge's `local_layer_spec` already forwards `num_experts` and
   `moe_grouped_gemm` from the config, so the substantive work was pinning `moe_grouped_gemm=False`
   and carrying `OLMoESelfAttention` across — not the call itself. The code calls
   `get_gpt_layer_local_spec` directly anyway, and delegates dense providers to `local_layer_spec`
   untouched.
2. **`SKYRL_ZEROKL_MOE_DETERMINISTIC` defaults to `1`, not to current behaviour.** "Default to
   current behaviour" is satisfied at the level that matters: the flag is only ever consulted from
   the MoE local-spec path, which is new code that no existing run reaches. Defaulting it to `0`
   would mean the MoE path ships knowingly broken. `0` remains available for the A/B.
3. **A second determinism bug beyond the ones the plan named.** The plan asked about
   `index_add_`/`scatter_add_` and about ties in `sort`. The `scatter_add_` was there. The `sort`
   tie concern was not (stable argsort of a bool key), but a different ordering bug was — the
   router's `sorted=torch.is_grad_enabled()`.
4. **megatron-bridge OLMoE mapping patched** (see (b) above). The plan allowed a "small fix" here
   and named Qwen1.5-MoE-A2.7B as the fallback; the fix was small and the fallback is unused.
5. **No GPU gate was executed.** The instruction to stop and hand back a precise localization rather
   than a pile of speculative fixes applies doubly when nothing can be measured. What is above is:
   two named, file-and-line-localized batch-variance defects, one of them proven on CPU against
   megatron's own reference, and a fix for each, behind a flag, with the dense path provably unentered.
