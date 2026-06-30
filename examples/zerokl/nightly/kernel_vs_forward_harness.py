"""DECISIVE decomposition: is the SkyRL-ZeroKL residual the LOGPROB KERNEL or the FORWARD?

The integration metric is |trainer_aten_logprob - engine_decode_logprob|. The demonstrator only
ever proved engine-decode==engine-prefill (=0). This harness measures, on the SAME generated
sequences, the two independent contributions:

  forward_diff = | triton(trainer_logits, tok) - engine_prefill_logprob(tok) |
      -> same Triton kernel both sides, different logits. Isolates trainer-forward vs engine-forward.

  kernel_diff  = | aten(trainer_logits, tok) - triton(trainer_logits, tok) |
      -> SAME logits, different log_softmax kernel. Isolates the fused-Triton-vs-aten difference
         that patch_vllm_logprobs_batch_invariant is supposed to remove.

  metric       = | aten(trainer_logits, tok) - engine_decode_logprob(tok) |  (= the run's metric)

If kernel_diff carries the 0.27/5%-token signature and forward_diff ~ 0  -> the patch (engine->aten)
is the fix and "didn't change the metric" means the patch is not actually taking effect.
If forward_diff carries it -> the trainer forward differs from the engine forward (logits), and no
logprob patch can help; the attention/forward path is the bug.

Run:
  CUDA_VISIBLE_DEVICES=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_BATCH_INVARIANT=1 \
    SKYRL_ZEROKL_LOCAL_SPEC=1 \
    /mnt/local_storage/zerokl-nightly-venv/bin/python examples/zerokl/nightly/kernel_vs_forward_harness.py
"""
import argparse, os, sys
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
os.environ.setdefault("SKYRL_ZEROKL_LOCAL_SPEC", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import random
import numpy as np
import torch


def aten_logprob(logits, token_ids):
    """The trainer's exact manual log_softmax (== from_parallel_logits_to_logprobs at TP1)."""
    x = logits.to(torch.float32)
    x = x - torch.amax(x, dim=-1, keepdim=True)
    lse = x.exp().sum(-1, keepdim=True).float().log()
    lp = x - lse
    return lp.gather(-1, token_ids.to(torch.int64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=int, default=4)
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--gpu_mem", type=float, default=0.55)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--patch_engine", type=int, default=0, help="apply patch_vllm_logprobs_batch_invariant")
    args = ap.parse_args()
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    MODEL = "/mnt/local_storage/models/MiMo-7B-RL"

    import varlen_backend  # noqa: F401  registers CUSTOM (num_splits=1) backend
    from vllm import LLM, SamplingParams
    from mimo_megatron_vllm import (register_mimo_to_vllm, CONFIG_FORMAT, build_mimo_gptmodel,
                                    find_inprocess_gpt)
    register_mimo_to_vllm()
    import vllm.model_executor.model_loader.weight_utils as _wu
    import vllm.model_executor.model_loader.dummy_loader as _dl
    _wu.initialize_dummy_weights = lambda *a, **k: None
    _dl.initialize_dummy_weights = lambda *a, **k: None

    # vLLM's fused Triton logprob kernel (reference for "triton" side)
    from vllm.v1.worker.gpu.sample.logprob import compute_token_logprobs as triton_logprob

    llm = LLM(model=MODEL, config_format=CONFIG_FORMAT, dtype="bfloat16", enforce_eager=True,
              gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_model_len, enable_prefix_caching=False,
              enable_chunked_prefill=False, load_format="dummy", trust_remote_code=True,
              attention_backend="CUSTOM")
    if args.patch_engine:
        from skyrl.backends.skyrl_train.zerokl.vllm_patches import patch_vllm_logprobs_batch_invariant
        patch_vllm_logprobs_batch_invariant()
        print("[harness] applied patch_vllm_logprobs_batch_invariant (engine -> aten)", flush=True)
    tok = llm.get_tokenizer()
    engine_gpt = find_inprocess_gpt(llm)
    assert engine_gpt is not None
    VOCAB = len(tok)

    # ---- TRAINER: independent local-spec GPTModel, swapped to the engine kernel + batch-invariant ----
    trainer, _cfg = build_mimo_gptmodel(torch.device("cuda"), dtype=torch.bfloat16)
    from skyrl.backends.skyrl_train.zerokl.megatron_varlen_attn import (
        swap_trainer_core_attention_varlen, enable_trainer_batch_invariant)
    enable_trainer_batch_invariant()
    swap_trainer_core_attention_varlen(trainer)
    trainer.eval()
    # native sync engine -> trainer (exact-name copy), so weights are bitwise-identical
    with torch.no_grad():
        dst = dict(engine_gpt.named_parameters())
        nmiss = 0
        for n, p in trainer.named_parameters():
            d = dst.get(n)
            if d is not None and tuple(d.shape) == tuple(p.shape):
                p.copy_(d.detach().to(p.dtype))
            else:
                nmiss += 1
        print(f"[harness] native sync engine->trainer done; unmatched trainer params={nmiss}", flush=True)

    @torch.no_grad()
    def trainer_logits(full_ids):
        L = len(full_ids)
        inp = torch.tensor([full_ids], device="cuda")
        pos = torch.arange(L, device="cuda").unsqueeze(0)
        am = torch.tril(torch.ones(L, L, device="cuda", dtype=torch.bool)).logical_not().view(1, 1, L, L)
        out = trainer(input_ids=inp, position_ids=pos, attention_mask=am)[0]  # [L, vocab_padded]
        return out[:, :VOCAB].float()  # [L, VOCAB]

    PROMPTS = ["The best way to", "My favorite thing is", "Once upon a time", "In the morning",
               "The answer to the question", "Scientists recently found", "Today I will", "The most important"]
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(args.prompts)]
    pid_lists = [tok(p, add_special_tokens=False).input_ids for p in prompts]
    sp = SamplingParams(n=args.group, temperature=1.0, top_p=1.0, max_tokens=args.max_tokens,
                        logprobs=0, seed=100)
    outs = llm.generate([{"prompt_token_ids": pids} for pids in pid_lists], sp)

    fwd_all, ker_all, met_all, dec_pre_all = [], [], [], []
    n_seq = 0
    for pids, o in zip(pid_lists, outs):
        for s in o.outputs:
            rids = list(s.token_ids)
            if not rids:
                continue
            n_seq += 1
            full = pids + rids
            blp = np.array([s.logprobs[i][rids[i]].logprob for i in range(len(rids))])  # engine decode
            # engine prefill rescore
            r = llm.generate([{"prompt_token_ids": full}],
                             SamplingParams(temperature=1.0, max_tokens=1, prompt_logprobs=0))[0]
            old = np.array([r.prompt_logprobs[t][full[t]].logprob for t in range(len(pids), len(full))])
            # trainer logits at the positions that PREDICT each response token: pos n_prompt-1 .. L-2
            tl_full = trainer_logits(full)  # [L, VOCAB]
            resp = torch.tensor(rids, device="cuda")
            idx = torch.arange(len(pids) - 1, len(full) - 1, device="cuda")
            tl_resp = tl_full[idx]  # [R, VOCAB]
            new_aten = aten_logprob(tl_resp, resp[:, None]).squeeze(1).cpu().numpy()
            new_triton = triton_logprob(tl_resp, resp[:, None]).squeeze(1).cpu().numpy()
            fwd_all.append(np.abs(new_triton - old))      # forward (same kernel, diff logits)
            ker_all.append(np.abs(new_aten - new_triton)) # kernel (same logits, diff kernel)
            met_all.append(np.abs(new_aten - blp))         # the run's metric
            dec_pre_all.append(np.abs(blp - old))          # engine decode vs engine prefill (should be 0)

    def stat(name, arr):
        a = np.concatenate(arr) if arr else np.zeros(1)
        f = (a > 0.05).mean()
        print(f"  {name:14s} mean={a.mean():.6e}  max={a.max():.6e}  min={a.min():.6e}  frac>0.05={f:.2%}  n={a.size}")

    print(f"\n==== DECOMPOSITION over {n_seq} sequences (patch_engine={args.patch_engine}) ====")
    stat("decode-prefill", dec_pre_all)  # engine internal consistency (expect 0)
    stat("forward_diff", fwd_all)        # trainer-forward vs engine-forward (same triton kernel)
    stat("kernel_diff", ker_all)         # aten vs triton on identical trainer logits
    stat("METRIC", met_all)              # |trainer_aten - engine_decode| == the run metric
    print("\nReading: if kernel_diff carries the 0.05+ outliers and forward_diff~0 -> the engine "
          "logprob patch is the fix. If forward_diff carries them -> the trainer forward != engine "
          "forward (logits); no logprob patch helps.", flush=True)


if __name__ == "__main__":
    main()
