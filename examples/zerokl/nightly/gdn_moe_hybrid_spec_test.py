"""The MoE branch of the hybrid no-TE spec, and the expert weight mapping that goes with it.

Qwen3.5-35B-A3B is a GDN hybrid AND a 256-expert MoE. Qwen3.5-0.8B (which Gates 1-3.1 use) is a GDN
hybrid with a DENSE MLP, so it never exercises either. This test does, on a 4-layer / 4-expert toy
with the real code paths -- cheap enough to run on one GPU, and it catches the two silent failures:

  * the hybrid spec dropping the MoE layer (building a dense MLP everywhere), and
  * the expert weight mapping matching nothing.

The second is the dangerous one. megatron-bridge's Qwen3.5 MoE bridge only declares the grouped-GEMM
expert names (`mlp.experts.linear_fc1.weight<i>`), because its provider assumes grouped GEMM. The
zero-KL recipe pins SequentialMLP (fixed-order expert combine), whose parameters are
`mlp.experts.local_experts.<i>.linear_fc1.weight`. Nothing matches; all 256 experts per layer stay at
their random init; the model trains, the loss falls, and the rollout is garbage. This is exactly the
bug `patch_olmoe_bridge_for_sequential_mlp` exists for, and Qwen3.5 needs its own.

Run:
    CUDA_VISIBLE_DEVICES=<gpu> SKYRL_ZEROKL_GDN=1 uv run --isolated --extra zerokl \
      python examples/zerokl/nightly/gdn_moe_hybrid_spec_test.py
"""

import os
import re
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/ray/default/SkyRL-ZeroKL")
os.environ.setdefault("SKYRL_ZEROKL_GDN", "1")
# `import megatron.bridge...` reaches its LoRA layers, which hard-import transformer_engine. The
# zerokl package installs a guard for exactly that, gated on this flag.
os.environ.setdefault("SKYRL_ZEROKL_LOCAL_SPEC", "1")

from skyrl.backends.skyrl_train.zerokl import install_fla_shim  # noqa: E402

install_fla_shim(force=True)

NUM_LAYERS = 4          # linear_attention_freq=4 -> 3 GDN + 1 attention
NUM_EXPERTS = 4


def build_config():
    from megatron.core.transformer.transformer_config import TransformerConfig

    return TransformerConfig(
        num_layers=NUM_LAYERS,
        hidden_size=512,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=128,
        ffn_hidden_size=512,
        params_dtype=torch.bfloat16,
        bf16=True,
        normalization="RMSNorm",
        layernorm_epsilon=1e-6,
        activation_func=F.silu,
        gated_linear_unit=True,
        layernorm_zero_centered_gamma=True,
        add_bias_linear=False,
        qk_layernorm=True,
        deterministic_mode=False,
        sequence_parallel=False,
        pipeline_model_parallel_size=1,
        # GDN hybrid
        experimental_attention_variant="gated_delta_net",
        linear_attention_freq=4,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        # MoE
        num_moe_experts=NUM_EXPERTS,
        moe_router_topk=2,
        moe_ffn_hidden_size=128,
        moe_grouped_gemm=False,
        moe_token_dispatcher_type="allgather",
    )


def main():
    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU")
    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29595")
        torch.distributed.init_process_group("nccl", rank=0, world_size=1)
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    parallel_state.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(0)

    from megatron.core.models.gpt.gpt_model import GPTModel

    from skyrl.backends.skyrl_train.zerokl.gdn_hybrid_spec import (
        is_hybrid_gdn, make_zerokl_hybrid_local_spec, patch_qwen35_bridge_for_local_spec,
    )

    config = build_config()
    assert is_hybrid_gdn(config)
    spec = make_zerokl_hybrid_local_spec(config)

    model = GPTModel(config=config, transformer_layer_spec=spec, vocab_size=256,
                     max_sequence_length=256, pre_process=True, post_process=True).cuda()
    if "transformer_engine" in sys.modules:
        raise SystemExit("FAIL: transformer_engine was imported")

    # ---- 1. hybrid: 3 GDN + 1 attention, and the MoE MLP survived ---------------------------
    kinds = [type(layer.self_attention).__name__ for layer in model.decoder.layers]
    mlps = [type(layer.mlp).__name__ for layer in model.decoder.layers]
    n_gdn = kinds.count("GatedDeltaNet")
    print(f"1. layers: {n_gdn} GatedDeltaNet + {len(kinds) - n_gdn} attention; mlp = {set(mlps)}")
    if n_gdn != 3:
        raise SystemExit(f"FAIL: expected 3 GDN layers, got {n_gdn} ({kinds})")
    if set(mlps) != {"MoELayer"}:
        raise SystemExit(f"FAIL: MoE layers were dropped by the hybrid spec: {mlps}")

    # ---- 2. SequentialMLP, not grouped GEMM ------------------------------------------------
    names = [n for n, _ in model.named_parameters()]
    local = [n for n in names if "experts.local_experts.0.linear_fc1.weight" in n]
    grouped = [n for n in names if re.search(r"experts\.linear_fc1\.weight\d+$", n)]
    print(f"2. expert params: {len(local)} SequentialMLP-named, {len(grouped)} grouped-GEMM-named")
    if not local or grouped:
        raise SystemExit(f"FAIL: expected SequentialMLP expert params, got e.g. "
                         f"{[n for n in names if 'experts' in n][:3]}")

    # ---- 3. the retargeted bridge mapping actually matches those names -----------------------
    patch_qwen35_bridge_for_local_spec(hf_lm_prefix="model.language_model.")
    from megatron.bridge.models.qwen import qwen35_bridge as qb

    mappings = qb.Qwen35MoEBridge._get_moe_lm_mappings(megatron_prefix="")
    pats = [m.megatron_param for m in mappings]
    expert_pats = [p for p in pats if ".mlp.experts." in p]
    print(f"3. bridge expert patterns after retarget: {expert_pats}")
    if any("weight*" in p for p in expert_pats):
        raise SystemExit("FAIL: bridge still declares grouped-GEMM expert names (weight*)")

    def matches(pattern, name):
        return re.fullmatch(pattern.replace(".", r"\.").replace("*", r"[^.]+"), name) is not None

    sample = f"decoder.layers.0.mlp.experts.local_experts.0.linear_fc1.weight"
    if not any(matches(p, sample) for p in expert_pats):
        raise SystemExit(f"FAIL: no retargeted pattern matches {sample}")
    sample2 = f"decoder.layers.0.mlp.experts.local_experts.{NUM_EXPERTS - 1}.linear_fc2.weight"
    if not any(matches(p, sample2) for p in expert_pats):
        raise SystemExit(f"FAIL: no retargeted pattern matches {sample2}")
    print(f"   both {sample!r} and {sample2!r} are matched")

    # ---- 4. forward runs ---------------------------------------------------------------------
    # Wrap in Float16Module, as `provide_distributed_model` does: it is what casts the fp32
    # embedding output down to params_dtype before the first linear. Without it a bare GPTModel
    # feeds fp32 activations into bf16 weights.
    from megatron.core.transformer.module import Float16Module

    fp16_model = Float16Module(config, model)
    T = 128
    dev = torch.device("cuda")
    tokens = torch.randint(0, 256, (1, T), device=dev)
    with torch.no_grad():
        out = fp16_model(input_ids=tokens, position_ids=torch.arange(T, device=dev).unsqueeze(0),
                         attention_mask=None)
    print(f"4. GDN+MoE forward OK: {tuple(out.shape)}")

    print("\nRESULT: PASS -- the hybrid spec keeps the MoE layers, pins SequentialMLP, and the "
          "retargeted Qwen3.5 bridge mapping matches its expert parameter names.")


if __name__ == "__main__":
    main()
