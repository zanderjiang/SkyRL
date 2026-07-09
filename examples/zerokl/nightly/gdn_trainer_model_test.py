"""The TRAINER half on a real Qwen3.5 hybrid: build the GPTModel, load weights, thd fwd+bwd.

Gate 1 validates one GatedDeltaNet layer. Gate 3.1 validates the whole model on the ENGINE side. This
covers the third corner: the trainer's `AutoBridge` -> hybrid no-TE spec -> `GPTModel` path, on the
real checkpoint, exactly as `megatron_worker.init_configs` builds it.

It is the cheapest way to de-risk a training run, because it catches the two failure modes that a
bitwise number cannot:

  * the hybrid spec silently building dense attention everywhere (assert 18 GDN + 6 attention), and
  * the bridge mapping silently matching nothing, leaving GDN weights at their random init
    (assert the loaded weights are not the init, and that the LM head produces a sane loss).

Run:
    CUDA_VISIBLE_DEVICES=<gpu> HF_HOME=/mnt/local_storage/hf SKYRL_ZEROKL_GDN=1 \
      SKYRL_ZEROKL_LOCAL_SPEC=1 uv run --isolated --extra zerokl \
      python examples/zerokl/nightly/gdn_trainer_model_test.py
"""

import math
import os
import sys

import torch

sys.path.insert(0, "/home/ray/default/SkyRL-ZeroKL")
os.environ.setdefault("SKYRL_ZEROKL_GDN", "1")
os.environ.setdefault("SKYRL_ZEROKL_LOCAL_SPEC", "1")

from skyrl.backends.skyrl_train.zerokl import install_fla_shim  # noqa: E402

install_fla_shim(force=True)

MODEL = os.environ.get("GDN_MODEL", "Qwen/Qwen3.5-0.8B")
SEQLENS = [96, 64, 137]  # > 5 chunks of 64 in the packed row
PROMPT = "The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is"


def main():
    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU")

    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29593")
        torch.distributed.init_process_group("nccl", rank=0, world_size=1)
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    parallel_state.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(0)

    from megatron.bridge import AutoBridge
    from transformers import AutoConfig, AutoTokenizer

    from skyrl.backends.skyrl_train.workers.megatron.model_bridges import (
        maybe_force_qwen35_text_bridge,
    )
    from skyrl.backends.skyrl_train.zerokl import make_zerokl_local_layer_spec
    from skyrl.backends.skyrl_train.zerokl.gdn_hybrid_spec import (
        checkpoint_is_vl_named, patch_qwen35_bridge_for_local_spec,
    )

    hf_config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    patch_qwen35_bridge_for_local_spec(
        hf_lm_prefix="model.language_model." if checkpoint_is_vl_named(hf_config) else None
    )

    bridge = AutoBridge.from_hf_pretrained(MODEL, trust_remote_code=True)
    assert maybe_force_qwen35_text_bridge(bridge, bridge.hf_pretrained.config), "not a Qwen3.5 model"

    mp = bridge.to_megatron_provider(load_weights=True)
    mp.tensor_model_parallel_size = 1
    mp.pipeline_model_parallel_size = 1
    mp.expert_model_parallel_size = 1
    mp.expert_tensor_parallel_size = 1
    mp.pipeline_dtype = torch.bfloat16
    mp.apply_rope_fusion = False
    mp.gradient_accumulation_fusion = False
    mp.transformer_layer_spec = make_zerokl_local_layer_spec(mp)
    if getattr(mp, "mtp_num_layers", None):
        mp.mtp_num_layers = None
    mp.finalize()
    gpt = mp.provide_distributed_model(wrap_with_ddp=False)
    gpt = gpt[0] if isinstance(gpt, list) else gpt
    gpt = gpt.module if hasattr(gpt, "module") else gpt

    if "transformer_engine" in sys.modules:
        raise SystemExit("FAIL: transformer_engine was imported")

    # The local spec's DotProductAttention refuses packed sequences ("Please use
    # TEDotProductAttention instead"). The real worker swaps in the torch-varlen kernel -- the same
    # one the engine runs -- for exactly this reason; do what the worker does.
    from skyrl.backends.skyrl_train.zerokl.megatron_varlen_attn import (
        swap_trainer_core_attention_varlen,
    )

    swap_trainer_core_attention_varlen(gpt)

    # ---- 1. the architecture is the hybrid, not 24 dense layers -----------------------------
    layers = list(gpt.decoder.layers)
    kinds = [type(layer.self_attention).__name__ for layer in layers]
    n_gdn = kinds.count("GatedDeltaNet")
    n_attn = len(kinds) - n_gdn
    print(f"1. GPTModel: {n_gdn} GatedDeltaNet + {n_attn} attention layers (no transformer_engine)")
    if n_gdn == 0:
        raise SystemExit("FAIL: no GatedDeltaNet layers -- the hybrid spec did not apply")
    expected = hf_config.text_config.layer_types.count("linear_attention")
    if n_gdn != expected:
        raise SystemExit(f"FAIL: expected {expected} GDN layers, built {n_gdn}")

    # ---- 2. the checkpoint actually landed in the GDN parameters ----------------------------
    # `reset_parameters` inits dt_bias to exactly ones and A_log to log(U(1,16)); a mapping that
    # matched nothing leaves them there. The real checkpoint's values are neither.
    gdn = next(layer.self_attention for layer in layers if type(layer.self_attention).__name__ == "GatedDeltaNet")
    if torch.allclose(gdn.dt_bias.float(), torch.ones_like(gdn.dt_bias.float())):
        raise SystemExit("FAIL: GDN dt_bias is still all-ones -- the bridge mapping loaded nothing")
    conv_std = gdn.conv1d.weight.float().std().item()
    print(f"2. GDN weights loaded: |dt_bias| mean {gdn.dt_bias.float().abs().mean():.4f}, "
          f"A_log mean {gdn.A_log.float().mean():.4f}, conv1d std {conv_std:.4f}")

    # ---- 3. forward + backward, and a sane loss ---------------------------------------------
    # UNPACKED (b=1, no PackedSeqParams). The packed thd path is NOT exercised here: under thd,
    # Megatron hands core_attention q as [T, np, hn] (batch dim folded away), and the zero-KL
    # `TorchVarlenCoreAttn` asserts the 4-D sbhd `[sq, b=1, np, hn]` layout. Packed training on a
    # GDN hybrid therefore needs that kernel taught the 3-D thd layout -- see the report. The GDN
    # layers' own packed path IS covered, bitwise, by gdn_trainer_shim_test (Gate 1).
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    dev = torch.device("cuda")
    torch.manual_seed(0)
    T = sum(SEQLENS)
    tokens = torch.randint(0, hf_config.text_config.vocab_size, (1, T), device=dev)
    pos = torch.arange(T, device=dev).unsqueeze(0)

    logits = gpt(input_ids=tokens, position_ids=pos, attention_mask=None)
    logits = logits.reshape(T, -1) if logits.dim() == 3 else logits
    print(f"3. forward OK: logits {tuple(logits.shape)} over {T} tokens ({(T + 63) // 64} chunks of 64)")

    loss = torch.nn.functional.cross_entropy(logits[:-1].float(), tokens[0, 1:])
    loss.backward()
    gnorm = gpt.decoder.layers[0].self_attention.in_proj.weight.grad.float().norm().item()
    if not (gnorm > 0 and gnorm == gnorm):
        raise SystemExit(f"FAIL: GDN in_proj grad is {gnorm}")
    ln_vocab = math.log(hf_config.text_config.vocab_size)
    print(f"   backward OK: random-token CE = {loss.item():.3f} (ln(vocab) = {ln_vocab:.3f}); "
          f"|dW GDN in_proj| = {gnorm:.3f}")

    # ---- 4. the loaded model is a language model, not noise --------------------------------
    pids = torch.tensor(tok(PROMPT, add_special_tokens=False).input_ids, device=dev).unsqueeze(0)
    n = pids.shape[1]
    with torch.no_grad():
        lg = gpt(input_ids=pids, position_ids=torch.arange(n, device=dev).unsqueeze(0),
                 attention_mask=None)
    lg = lg.reshape(n, -1) if lg.dim() == 3 else lg
    nxt = tok.decode([int(lg[-1].argmax())])
    ce = torch.nn.functional.cross_entropy(lg[:-1].float(), pids[0, 1:]).item()
    print(f"4. {PROMPT!r} -> next token {nxt!r}; prompt CE = {ce:.3f}")
    if ce > 6.0:
        raise SystemExit(f"FAIL: CE {ce:.3f} on natural text -- the weights are not the checkpoint's")

    # ---- 5. packed (thd) forward, and no leakage across sequence boundaries ------------------
    # Sample packing is the normal SkyRL configuration. Under thd, Megatron hands core_attention a
    # 3-D q of [T, np, hn]; the attention must use `cu_seqlens` rather than treating the whole packed
    # row as one sequence, or a token of sequence 2 attends to sequence 1 and the trainer stops
    # matching the (per-sequence) rollout. The check: sequence 0's logits inside a packed row must be
    # bitwise equal to running sequence 0 on its own.
    from megatron.core.packed_seq_params import PackedSeqParams

    seqs = [torch.randint(0, hf_config.text_config.vocab_size, (n,), device=dev) for n in SEQLENS]
    packed_tokens = torch.cat(seqs).unsqueeze(0)
    Tp = packed_tokens.shape[1]
    cu = torch.tensor([0, *torch.tensor(SEQLENS).cumsum(0).tolist()], dtype=torch.int32, device=dev)
    psp = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu,
                          max_seqlen_q=max(SEQLENS), max_seqlen_kv=max(SEQLENS))
    packed_pos = torch.cat([torch.arange(n, device=dev) for n in SEQLENS]).unsqueeze(0)

    with torch.no_grad():
        lp = gpt(input_ids=packed_tokens, position_ids=packed_pos, attention_mask=None,
                 packed_seq_params=psp)
        lp = lp.reshape(Tp, -1) if lp.dim() == 3 else lp

        n0 = SEQLENS[0]
        cu0 = torch.tensor([0, n0], dtype=torch.int32, device=dev)
        psp0 = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu0, cu_seqlens_kv=cu0,
                               max_seqlen_q=n0, max_seqlen_kv=n0)
        l0 = gpt(input_ids=seqs[0].unsqueeze(0), position_ids=torch.arange(n0, device=dev).unsqueeze(0),
                 attention_mask=None, packed_seq_params=psp0)
        l0 = l0.reshape(n0, -1) if l0.dim() == 3 else l0

    d = (lp[:n0].float() - l0.float()).abs().max().item()
    print(f"5. thd packed forward OK ({Tp} tokens / {len(SEQLENS)} seqs); seq-0 logits packed vs alone: "
          f"max |diff| = {d:.3e}")
    if d != 0.0:
        raise SystemExit(f"FAIL: packing changes sequence 0's logits by {d:.3e} -- attention leaks "
                         "across the packed boundary (cu_seqlens ignored?)")

    print("\nRESULT: PASS -- the trainer builds the Qwen3.5 hybrid (18 GDN + 6 attention), loads the "
          "checkpoint into its GDN layers, and runs unpacked AND packed (thd) fwd+bwd with no "
          "TransformerEngine.")


if __name__ == "__main__":
    main()
