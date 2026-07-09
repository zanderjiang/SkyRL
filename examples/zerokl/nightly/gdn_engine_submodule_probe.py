"""Which submodule of decoder layer 0 first breaks decode == prefill?

Established so far, all measured:
  * gdn_layer_decode_parity_test: the GDN layer math is bitwise (450/450 decoded tokens).
  * gdn_engine_layer_bisect: in the engine, decoder layer 0's OUTPUT first differs at decode step ~50
    (2.4e-4), and deeper layers differ earlier and larger -- the signature of a tiny error present
    everywhere and amplified with depth, not of an onset event.
  * gdn_engine_core_probe: layer 0's GDN core (``_forward_core``) is bitwise on BOTH sides -- its
    ``mixed_qkv`` input and its ``core_attn_out`` output -- at every generated position, across all
    three chunk rolls.

So the divergence enters between the GDN core and the layer output: the gated RMSNorm, the output
projection, the MLP, or a layernorm. Every one of those runs at M=1 during decode and M=T during
prefill; a GEMM whose fp32 accumulation order depends on M gives a sub-ulp difference that usually
rounds to the same bf16 value and occasionally does not -- which is exactly what "first bad position
looks random, deeper layers worse" means.

This hooks every submodule of layer 0 and reports the first generated position at which each one's
output stops being bitwise. The shallowest such submodule is the culprit.
"""

import os
import sys

import torch

sys.path.insert(0, "/home/ray/default/SkyRL-ZeroKL")

MODEL = os.environ.get("GDN_MODEL", "Qwen/Qwen3.5-0.8B")
NGEN = int(os.environ.get("GDN_NGEN", "200"))
LAYER_IDX = int(os.environ.get("GDN_PROBE_LAYER_IDX", "0"))
PROMPT = "Explain, step by step, how a compiler turns source code into machine code."


def as_tensor(out, args):
    if out is None:                       # forward writes into an out-param (GDN attn, MLP)
        for a in reversed(args):
            if isinstance(a, torch.Tensor):
                return a
        return None
    if isinstance(out, tuple):
        out = out[0]
    return out if isinstance(out, torch.Tensor) else None


def main():
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
    from vllm import LLM, SamplingParams

    from skyrl.backends.skyrl_train.zerokl.gdn_engine_patch import install_gdn_engine_patch
    from skyrl.backends.skyrl_train.zerokl.moe_batch_invariant import _install_moe_matmul_invariance
    from skyrl.backends.skyrl_train.zerokl.varlen_backend import register_varlen_custom_backend

    install_gdn_engine_patch()
    register_varlen_custom_backend()
    _install_moe_matmul_invariance()

    llm = LLM(model=MODEL, dtype="bfloat16", enforce_eager=True, max_num_seqs=1,
              gpu_memory_utilization=0.55, max_model_len=NGEN + 512, enable_prefix_caching=False,
              enable_chunked_prefill=False, trust_remote_code=True, seed=0,
              attention_backend="CUSTOM", limit_mm_per_prompt={"image": 0, "video": 0})
    tok = llm.get_tokenizer()

    from examples.zerokl.nightly.gdn_engine_layer_bisect import find_layers  # noqa: E402

    cfg = llm.llm_engine.vllm_config.model_config.hf_text_config
    layers, _ = find_layers(llm, cfg.num_hidden_layers)
    layer = layers[LAYER_IDX]

    subs = [("<layer>", layer)] + [(n, m) for n, m in layer.named_modules()
                                   if n and len(list(m.children())) == 0]
    state = {"phase": "decode", "pos": 0}
    cap: dict[tuple[str, str], dict[int, torch.Tensor]] = {}

    def hook(name):
        def fn(_m, args, out):
            t = as_tensor(out, args)
            if t is None or t.ndim < 2:
                return
            d = cap.setdefault((state["phase"], name), {})
            base = state["pos"]
            for i in range(t.shape[0]):
                d[base + i] = t[i].detach().float().cpu().clone()
        return fn

    handles = [m.register_forward_hook(hook(n)) for n, m in subs]

    # `state["pos"]` must advance exactly once per model forward, not once per hook. Hooks fire in
    # registration order, so this runs after the capture hooks for the same layer.
    def bump(_m, args, out):
        t = as_tensor(out, args)
        state["pos"] = 0 if state["phase"] == "prefill" else state["pos"] + t.shape[0]
    handles.append(layer.register_forward_hook(bump))

    pids = tok(PROMPT, add_special_tokens=False).input_ids
    P = len(pids)
    out = llm.generate([{"prompt_token_ids": pids}],
                       SamplingParams(temperature=1.0, top_p=1.0, seed=0, max_tokens=NGEN,
                                      logprobs=0, ignore_eos=True))[0]
    gen_ids = list(out.outputs[0].token_ids)

    state["phase"], state["pos"] = "prefill", 0
    llm.generate([{"prompt_token_ids": pids + gen_ids}],
                 SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0))
    for h in handles:
        h.remove()

    print(f"\nlayer {LAYER_IDX}, prompt={P}, generated={len(gen_ids)}\n")
    print(f"{'submodule':<44} {'first bad abs pos':>18} {'max |diff|':>12}")
    for name, _ in subs:
        dec, pre = cap.get(("decode", name)), cap.get(("prefill", name))
        if not dec or not pre:
            print(f"{name:<44} {'(not captured)':>18}")
            continue
        first, worst = None, 0.0
        for t in range(len(gen_ids)):
            p = P + t
            if p not in dec or p not in pre:
                continue
            d = (dec[p] - pre[p]).abs().max().item()
            if d != 0.0 and first is None:
                first = p
            worst = max(worst, d)
        print(f"{name:<44} {str(first):>18} {worst:>12.3e}")


if __name__ == "__main__":
    main()
