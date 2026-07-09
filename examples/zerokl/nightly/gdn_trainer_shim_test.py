"""GATE 1: Megatron's GatedDeltaNet, running on the zero-KL `fla` shim, is bitwise == gdn_ops.

Three things are asserted, in order:

  1. BUILDS. ``GatedDeltaNet.__init__`` no longer raises ``ImportError("FLA is not installed")``,
     and nothing imported ``transformer_engine`` on the way (the nightly stack has no TE).
  2. RUNS. Forward AND backward on the packed thd path (``PackedSeqParams(qkv_format='thd')`` with
     several sequences of unequal length in one row). Megatron's own ``deterministic_mode`` fallback
     cannot do this -- ``torch_chunk_gated_delta_rule`` asserts ``cu_seqlens is None``.
  3. IS THE SAME CODE AS THE ENGINE. The layer's output is recomputed from its own weights using
     ``zerokl.gdn_ops`` directly (conv -> split -> l2norm -> GQA -> g/beta -> chunk), and the two
     must agree BITWISE. This is what makes the trainer and the rollout engine one implementation.

Sequence lengths are chosen so the packed row spans >= 5 chunks of 64 -- fewer than that hides the
racy FLA autotune config that ``pin_fla_autotune_configs`` exists to eliminate.

Run:
    CUDA_VISIBLE_DEVICES=<gpu> SKYRL_ZEROKL_GDN=1 \
      uv run --isolated --extra zerokl python examples/zerokl/nightly/gdn_trainer_shim_test.py
Exit 0 iff all three pass.
"""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/ray/default/SkyRL-ZeroKL")
os.environ.setdefault("SKYRL_ZEROKL_GDN", "1")

# MUST come before any megatron import: gated_delta_net binds chunk_gated_delta_rule at import time.
from skyrl.backends.skyrl_train.zerokl import install_fla_shim  # noqa: E402

install_fla_shim(force=True)

from skyrl.backends.skyrl_train.zerokl.gdn_ops import (  # noqa: E402
    gdn_causal_conv,
    gdn_chunk,
    gdn_gate_and_beta,
    gdn_l2norm,
)

# Qwen3.5-0.8B GDN geometry (hidden 1024, 16 k-heads, 16 v-heads, head dim 128, conv width 4).
HIDDEN = 1024
NUM_K_HEADS = 16
NUM_V_HEADS = 16
HEAD_K = HEAD_V = 128
CONV_W = 4
# > 5 chunks of 64 in the packed row (395 tokens), with a chunk-aligned and a ragged sequence.
SEQLENS = [137, 64, 194]


def maxdiff(a, b):
    d = (a.float() - b.float()).abs().max()
    return float("nan") if torch.isnan(d) else float(d)


def build_layer(device):
    from megatron.core import parallel_state
    from megatron.core.models.backends import LocalSpecProvider
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.ssm.gated_delta_net import GatedDeltaNet, GatedDeltaNetSubmodules
    from megatron.core.transformer.transformer_config import TransformerConfig

    backend = LocalSpecProvider()
    config = TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN,
        num_attention_heads=8,
        num_query_groups=2,
        kv_channels=256,
        ffn_hidden_size=3584,
        use_cpu_initialization=False,
        params_dtype=torch.bfloat16,
        bf16=True,
        normalization="RMSNorm",
        layernorm_epsilon=1e-6,
        activation_func=F.silu,
        deterministic_mode=False,
        sequence_parallel=False,
        linear_conv_kernel_dim=CONV_W,
        linear_key_head_dim=HEAD_K,
        linear_value_head_dim=HEAD_V,
        linear_num_key_heads=NUM_K_HEADS,
        linear_num_value_heads=NUM_V_HEADS,
    )
    pg = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
    assert parallel_state.get_tensor_model_parallel_world_size() == 1

    submodules = GatedDeltaNetSubmodules(
        in_proj=backend.column_parallel_linear(),
        out_norm=backend.layer_norm(rms_norm=True, for_qk=False),
        out_proj=backend.row_parallel_linear(),
    )
    layer = GatedDeltaNet(config, submodules, layer_number=1, pg_collection=pg).to(device)
    return layer, config


def reference_forward(layer, hidden_states, cu_seqlens):
    """Recompute the layer with zerokl.gdn_ops only. Mirrors GatedDeltaNet.forward step for step."""
    qkvzba, _ = layer.in_proj(hidden_states)          # [s, b, in_proj_dim]
    qkvzba = qkvzba.transpose(0, 1)                   # [b, s, ...]  (b == 1 on the packed path)
    qkv, gate, beta, alpha = torch.split(
        qkvzba,
        [layer.qk_dim_local_tp * 2 + layer.v_dim_local_tp, layer.v_dim_local_tp,
         layer.num_value_heads, layer.num_value_heads],
        dim=-1,
    )
    conv_w = layer.conv1d.weight.squeeze(1)           # [D, W]
    conv_b = layer.conv1d.bias if layer.conv_bias else None

    outs = []
    bounds = cu_seqlens.tolist()
    for s, e in zip(bounds[:-1], bounds[1:]):
        y = gdn_causal_conv(qkv[0, s:e], conv_w, conv_b, activation="silu")
        n = e - s
        kd = layer.qk_dim_local_tp
        q = gdn_l2norm(y[:, :kd].contiguous().view(n, NUM_K_HEADS, HEAD_K))
        k = gdn_l2norm(y[:, kd : 2 * kd].contiguous().view(n, NUM_K_HEADS, HEAD_K))
        v = y[:, 2 * kd :].contiguous().view(n, NUM_V_HEADS, HEAD_V)
        rep = NUM_V_HEADS // NUM_K_HEADS
        if rep > 1:
            q, k = q.repeat_interleave(rep, dim=1), k.repeat_interleave(rep, dim=1)
        g, bt = gdn_gate_and_beta(alpha[0, s:e], beta[0, s:e], layer.A_log, layer.dt_bias)
        o, _ = gdn_chunk(q[None], k[None], v[None], g[None], bt[None])
        outs.append(o[0])

    core_attn_out = torch.cat(outs, dim=0)[None]       # [1, T, Hv, Dv]
    gate = gate.reshape(1, cu_seqlens[-1].item(), -1, HEAD_V)
    norm_out = layer._apply_gated_norm(core_attn_out, gate)
    norm_out = norm_out.reshape(1, -1, layer.v_dim_local_tp).transpose(0, 1).contiguous()
    out, _ = layer.out_proj(norm_out)
    return out


def main():
    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU")
    device = torch.device("cuda")

    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29591")
        torch.distributed.init_process_group("nccl", rank=0, world_size=1)
    from megatron.core import parallel_state

    parallel_state.initialize_model_parallel(1, 1)
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    model_parallel_cuda_manual_seed(0)  # ColumnParallelLinear init forks the model-parallel RNG
    torch.manual_seed(0)

    # ---- 1. builds, and without TransformerEngine ------------------------------------------
    layer, _ = build_layer(device)
    if "transformer_engine" in sys.modules:
        raise SystemExit("FAIL: transformer_engine was imported")
    print(f"1. GatedDeltaNet built (no transformer_engine). in_proj_dim={layer.in_proj_dim}")

    # ---- 2. packed (thd) forward + backward ------------------------------------------------
    from megatron.core.packed_seq_params import PackedSeqParams

    T = sum(SEQLENS)
    cu = torch.tensor([0, *torch.tensor(SEQLENS).cumsum(0).tolist()], dtype=torch.int32, device=device)
    psp = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        max_seqlen_q=max(SEQLENS),
        max_seqlen_kv=max(SEQLENS),
    )
    hidden = torch.randn(T, 1, HIDDEN, dtype=torch.bfloat16, device=device, requires_grad=True)

    out, _ = layer(hidden, None, packed_seq_params=psp)
    print(f"2. thd forward OK: out {tuple(out.shape)}, {T} tokens over {len(SEQLENS)} seqs "
          f"({(T + 63) // 64} chunks of 64)")
    out.float().pow(2).sum().backward()
    gnorm = hidden.grad.float().norm().item()
    if not (gnorm > 0 and gnorm == gnorm):
        raise SystemExit(f"FAIL: backward produced a bad input grad (norm={gnorm})")
    for name in ("conv1d.weight", "A_log", "dt_bias", "in_proj.weight", "out_proj.weight"):
        p = layer.get_parameter(name)
        if p.grad is None or not torch.isfinite(p.grad).all():
            raise SystemExit(f"FAIL: no finite grad for {name}")
    # The q and k halves of in_proj feed the chunk kernel only through l2norm, and the v half
    # bypasses it. A l2norm/chunk that silently drops its grad still trains (via v) and still
    # converges -- so check the q/k rows specifically, not just "some grad exists".
    qk = layer.qk_dim_local_tp
    gq = layer.in_proj.weight.grad[:qk].float().norm().item()
    gk = layer.in_proj.weight.grad[qk : 2 * qk].float().norm().item()
    if not (gq > 0 and gk > 0):
        raise SystemExit(f"FAIL: q/k projection grads are zero (|dq|={gq}, |dk|={gk}) -- a "
                         "Triton op in the q/k path severed the autograd graph")
    print(f"   backward OK: |d hidden| = {gnorm:.4f}; q/k proj grads nonzero "
          f"(|dWq|={gq:.4f}, |dWk|={gk:.4f}); all GDN params have finite grads")

    # ---- 3. bitwise == a direct gdn_ops reference -------------------------------------------
    with torch.no_grad():
        got, _ = layer(hidden.detach(), None, packed_seq_params=psp)
        ref = reference_forward(layer, hidden.detach(), cu)
    d = maxdiff(got, ref)
    print(f"3. trainer forward vs direct gdn_ops reference: max |diff| = {d:.3e}")
    if d != 0.0:
        print("\nRESULT: FAIL -- the trainer is not running the engine's ops bitwise")
        raise SystemExit(1)

    # ---- 4. the VJP reference computes the same function as the kernel ----------------------
    # gdn_chunk's backward differentiates a torch reimplementation (`_torch_chunk_gdr`) because the
    # vendored kernel has no backward. That is only a valid gradient if the reference agrees with
    # the kernel's forward. bf16 kernel vs fp32 reference -> agreement is approximate by design.
    from skyrl.backends.skyrl_train.zerokl.gdn_ops import _torch_chunk_gdr, fla_chunk_size

    torch.manual_seed(3)
    H, D = 8, 128
    qs = gdn_l2norm(torch.randn(1, 320, H, D, dtype=torch.bfloat16, device=device))
    ks = gdn_l2norm(torch.randn(1, 320, H, D, dtype=torch.bfloat16, device=device))
    vs = torch.randn(1, 320, H, D, dtype=torch.bfloat16, device=device)
    gs = -F.softplus(torch.randn(1, 320, H, device=device)).float()
    bs = torch.rand(1, 320, H, dtype=torch.bfloat16, device=device).sigmoid()
    cu = torch.tensor([0, 137, 320], dtype=torch.int32, device=device)
    with torch.no_grad():
        o_kern, _ = gdn_chunk(qs, ks, vs, gs, bs, cu_seqlens=cu)
        o_ref = _torch_chunk_gdr(qs, ks, vs, gs, bs, None, cu, fla_chunk_size())
    rel = ((o_kern.float() - o_ref).norm() / o_ref.norm()).item()
    print(f"4. VJP reference vs bitwise kernel forward: relative L2 = {rel:.3e} (bf16 vs fp32)")
    if not (rel < 2e-2):
        print("\nRESULT: FAIL -- gdn_chunk's backward differentiates the wrong function")
        raise SystemExit(1)

    print("\nRESULT: GATE 1 PASS -- Megatron GDN builds, runs thd fwd+bwd, and is BITWISE == gdn_ops.")


if __name__ == "__main__":
    main()
