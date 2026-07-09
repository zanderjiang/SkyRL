"""Localize the FIRST bitwise decode-vs-prefill divergence inside the patched Qwen3.5 engine.

Gate 2 asks for exact-0.0 on every token. When it isn't, the useful deliverable is not another guess
but the answer to: *which layer diverges first, and by how much?* The GDN math is proven bitwise at
layer level (gdn_layer_decode_parity_test: 450/450), so a divergence here is wiring.

Method. One sequence, no batching. Hook every decoder layer's output. Decode N tokens, recording each
layer's hidden state at the newly generated position. Then prefill prompt+generated through the same
engine and record the same layers at the same absolute positions. Compare per layer, per position.
The first layer with a nonzero diff is where to look; Qwen3.5's layer_types tell us whether it is a
GatedDeltaNet layer (3 of 4) or a full-attention layer (every 4th).

Run:
    CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=examples/zerokl/nightly/_torchvision_stub \
      HF_HOME=/mnt/local_storage/hf SKYRL_ZEROKL_GDN=1 VLLM_BATCH_INVARIANT=1 \
      uv run --isolated --extra zerokl python examples/zerokl/nightly/gdn_engine_layer_bisect.py
"""

import os
import sys

import torch

sys.path.insert(0, "/home/ray/default/SkyRL-ZeroKL")

MODEL = os.environ.get("GDN_MODEL", "Qwen/Qwen3.5-0.8B")
# Default long enough that the open chunk FILLS and rolls several times (C = 64): a 12-token
# generation from a 17-token prompt never crosses a chunk boundary and proves almost nothing about
# chunk-consistent decode. It also has to be long enough for the softmax layers' split-K heuristic.
NGEN = int(os.environ.get("GDN_NGEN", "200"))
NSEQ = int(os.environ.get("GDN_NSEQ", "1"))
FULL_STACK = os.environ.get("GDN_FULL_STACK", "1") == "1"
PROMPT = "Explain, step by step, how a compiler turns source code into machine code."


def find_layers(llm, n_expected):
    """The decoder stack is the nn.ModuleList with one entry per hf_text_config layer.

    Walk the in-process engine down to the model runner rather than guessing attribute paths.
    """
    seen, found = set(), []

    def walk(o, d=0):
        if id(o) in seen or d > 12:
            return
        seen.add(id(o))
        if isinstance(o, torch.nn.Module):
            for name, mod in o.named_modules():
                if isinstance(mod, torch.nn.ModuleList) and len(mod) == n_expected:
                    found.append((list(mod), f"{type(o).__name__}.{name}"))
                    return
            return
        for a in ("llm_engine", "engine_core", "model_executor", "driver_worker",
                  "collective_rpc", "worker", "model_runner", "model", "engine"):
            if found:
                return
            if hasattr(o, a):
                try:
                    walk(getattr(o, a), d + 1)
                except Exception:
                    pass

    walk(llm)
    if not found:
        raise SystemExit(f"could not locate a decoder ModuleList of length {n_expected}")
    return found[0]


def main():
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
    from vllm import LLM, SamplingParams

    from skyrl.backends.skyrl_train.zerokl.gdn_engine_patch import install_gdn_engine_patch

    install_gdn_engine_patch()
    if FULL_STACK:
        # Same stack Gate 2 runs: GDN layers chunk-consistent, softmax layers on the num_splits=1
        # CUSTOM varlen backend, SM90 Triton matmul override. Set GDN_FULL_STACK=0 to isolate GDN.
        from skyrl.backends.skyrl_train.zerokl.moe_batch_invariant import _install_moe_matmul_invariance
        from skyrl.backends.skyrl_train.zerokl.varlen_backend import register_varlen_custom_backend

        register_varlen_custom_backend()
        _install_moe_matmul_invariance()

    llm = LLM(model=MODEL, dtype="bfloat16", enforce_eager=True, max_num_seqs=NSEQ,
              gpu_memory_utilization=0.55, max_model_len=NGEN + 512, enable_prefix_caching=False,
              enable_chunked_prefill=False, trust_remote_code=True, seed=0,
              **({"attention_backend": "CUSTOM"} if FULL_STACK else {}),
              limit_mm_per_prompt={"image": 0, "video": 0})
    tok = llm.get_tokenizer()
    cfg = llm.llm_engine.vllm_config.model_config.hf_text_config
    layer_types = list(getattr(cfg, "layer_types", []))
    layers, path = find_layers(llm, cfg.num_hidden_layers)
    print(f"decoder layers={len(layers)} at {path}", flush=True)

    capture = {}   # (phase, layer) -> list of row tensors
    phase = ["decode"]

    def hook(idx):
        def fn(_mod, _args, out):
            h = out[0] if isinstance(out, tuple) else out
            capture.setdefault((phase[0], idx), []).append(h.detach().float().cpu())
        return fn

    handles = [layer.register_forward_hook(hook(i)) for i, layer in enumerate(layers)]

    pids = tok(PROMPT, add_special_tokens=False).input_ids
    out = llm.generate([{"prompt_token_ids": pids}],
                       SamplingParams(temperature=1.0, top_p=1.0, seed=0, max_tokens=NGEN,
                                      logprobs=0, ignore_eos=True))[0]
    gen_ids = list(out.outputs[0].token_ids)

    phase[0] = "prefill"
    capture_prefill_start = len(pids)
    llm.generate([{"prompt_token_ids": pids + gen_ids}],
                 SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0))
    for h in handles:
        h.remove()

    print(f"\nprompt={len(pids)} tok, generated={len(gen_ids)} tok\n")
    print(f"{'layer':>5} {'type':>17} {'first bad pos':>14} {'max |diff|':>12}")
    first_bad = None
    for i in range(len(layers)):
        dec = capture[("decode", i)]      # prefill call (prompt) + NGEN decode calls
        pre = capture[("prefill", i)]     # one prefill call over prompt+gen
        assert len(pre) == 1, f"expected one prefill call, got {len(pre)}"
        ref = pre[0].reshape(-1, pre[0].shape[-1])
        # dec[0] is the prompt prefill; dec[1:] are the decode steps (1 row each)
        steps = dec[1:]
        worst, badpos = 0.0, None
        for t, row in enumerate(steps):
            r = row.reshape(-1, row.shape[-1])[-1]
            d = (r - ref[capture_prefill_start + t]).abs().max().item()
            if d != 0.0 and badpos is None:
                badpos = t
            worst = max(worst, d)
        ltype = layer_types[i] if i < len(layer_types) else "?"
        print(f"{i:>5} {ltype:>17} {str(badpos):>14} {worst:>12.3e}", flush=True)
        if badpos is not None and first_bad is None:
            first_bad = (i, ltype, badpos, worst)

    print()
    if first_bad is None:
        print("RESULT: every decoder layer is BITWISE == its prefill. Divergence is after the "
              "decoder (lm_head / sampler logprobs).")
    else:
        i, ltype, badpos, worst = first_bad
        print(f"RESULT: FIRST divergence at layer {i} ({ltype}), decode step {badpos}, "
              f"max |diff| {worst:.3e}")


if __name__ == "__main__":
    main()
