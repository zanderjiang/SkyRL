"""
vLLM Worker Extension for SkyRL weight synchronization.

This module provides WorkerWrap, a vLLM worker extension class that enables
efficient NCCL-based and CUDA IPC-based weight updates from the training
process to inference workers.

TODO: This will be removed once vLLM natively supports weight sync APIs.
See: https://github.com/vllm-project/vllm/issues/31848

Usage:
    Pass as --worker-extension-cls to vLLM:

    vllm serve ... --worker-extension-cls skyrl_train.inference_servers.vllm_worker.WorkerWrap
"""

import warnings

import torch

from skyrl.backends.skyrl_train.inference_servers.layerwise_reload import (
    LayerwiseReloadWorkerMixin,
)

# Path to this worker extension class for use in CLI args (derived from module path)
VLLM_WORKER_EXTENSION_CLS = f"{__name__}.WorkerWrap"


class WorkerWrap(LayerwiseReloadWorkerMixin):
    """
    vLLM worker extension for SkyRL weight synchronization.

    This class is injected into vLLM workers via --worker-extension-cls and
    provides methods that can be called via engine.collective_rpc() to
    coordinate weight updates across all TP/PP workers.

    Methods:
        init_weight_update_communicator: Initialize the weight receiver
        skyrl_start_weight_update: Begin a sync; initialize vLLM layerwise reload once
        load_weights: Receive and load one chunk of weights from trainer
        skyrl_finish_weight_update: End a sync; finalize vLLM layerwise reload once
        teardown_weight_receiver: Clean up weight receiver resources
    """

    def test_rpc(self, *args, **kwargs):
        """Test RPC call to worker."""
        return args, kwargs

    def init_weight_update_communicator(self, init_info: bytes):
        """
        Initialize weight update communicator from init info.

        Args:
            init_info: Pickled bytes of WeightSyncInitInfo from the sender.
        """
        import pickle

        assert torch.distributed.is_initialized(), "default torch process group must be initialized"

        # Unpickle init_info to restore the original object type
        assert isinstance(init_info, bytes), f"Expected bytes, got {type(init_info).__name__}"
        init_info = pickle.loads(init_info)

        strategy_cls = init_info.strategy_type()

        if hasattr(self, "_weight_receiver") and self._weight_receiver is not None:
            # TODO(haochen): we should get rid of this flag and override existing receiver.
            if init_info.override_existing_receiver:
                self._weight_receiver.teardown()
                self._weight_receiver = None
            else:
                warnings.warn(
                    "Detected an existing weight receiver. "
                    "For overriding, use `generator.inference_engine.override_existing_update_group=enable`"
                )
                return

        self._weight_receiver = strategy_cls.create_receiver(init_info)

    def load_weights(self, request: bytes) -> None:
        """
        Load one chunk of weights using the receiver.

        Called via collective_rpc from the weight loader, once per chunk.
        When the sender brackets the sync with skyrl_start_weight_update / skyrl_finish_weight_update,
        the chunk is loaded raw and the single finalize runs vLLM's post-load weight
        processing exactly once over the whole weight set.
        Without a bracket, it falls back to a self-contained reload_weights
        (initialize + load + finalize in this one call), correct when the call
        carries the whole model so finalize sees every layer and restores none.

        Args:
            request: Pickled bytes of WeightUpdateRequest.
        """
        import pickle

        from vllm.config import set_current_vllm_config

        # Unpickle request to restore the original object type
        assert isinstance(request, bytes), f"Expected bytes, got {type(request).__name__}"
        request = pickle.loads(request)

        import os as _os_probe
        print(
            f"[ZEROKL-PROBE] WorkerWrap.load_weights CALLED; SKYRL_ZERO_KL={_os_probe.environ.get('SKYRL_ZERO_KL')!r} "
            f"bracketed={getattr(self, '_skyrl_weight_update_active', False)} "
            f"model_cls={type(getattr(self.model_runner, 'model', None)).__name__}",
            flush=True,
        )

        weight_list = []
        for name, tensor in self._weight_receiver.receive_weights(request):
            weight_list.append((name, tensor))

        # SkyRL-ZeroKL: the vLLM model IS Megatron's GPTModel (GPTModelVLLMWrapper). Copy the
        # native (no-HF) params straight into wrapper.gpt by name instead of vLLM's HF loader.
        # load_weights is called ONCE PER CHUNK (per param), so accumulate diagnostics on self and
        # print() them (Ray forwards stdout; SkyRL suppresses module-logger INFO).
        import os as _os
        if _os.environ.get("SKYRL_ZERO_KL") == "1":
            model = self.model_runner.model
            target = model.gpt if hasattr(model, "gpt") else model
            # IMPORTANT: rebuild the dst map EVERY call (do NOT cache). vLLM's colocate sleep/wake
            # (cumem) re-allocates the weight storage on each wake; a cached dict would copy into
            # STALE/freed tensors while generation reads the live (zero) buffers -> gibberish.
            params = dict(target.named_parameters())
            bufs = dict(target.named_buffers())
            copied = 0
            materialized = 0
            miss = []
            # SkyRL-ZeroKL diag: capture EVERY missed name + its magnitude so the name mismatch is
            # fully visible (the old code only kept the first 3). recv-abs-sum excludes misses, so
            # missed magnitude is exactly the sender-vs-receiver checksum gap.
            self._zk_miss_names = getattr(self, "_zk_miss_names", [])
            self._zk_miss_ck = getattr(self, "_zk_miss_ck", 0.0)

            def _set_on_module(root, dotted, value, as_param, requires_grad):
                # navigate to the owning submodule and replace the param/buffer object so a META
                # placeholder (from cumem sleep freeing storage) becomes a real GPU tensor.
                *path, attr = dotted.split(".")
                mod = root
                for p in path:
                    mod = getattr(mod, p)
                if as_param:
                    mod._parameters[attr] = torch.nn.Parameter(value, requires_grad=requires_grad)
                else:
                    mod._buffers[attr] = value

            with torch.no_grad(), set_current_vllm_config(self.vllm_config):
                for name, tensor in weight_list:
                    is_param = name in params
                    dest = params.get(name)
                    if dest is None:
                        dest = bufs.get(name)
                    if dest is None:
                        if len(miss) < 3:
                            miss.append(name)
                        if name not in self._zk_miss_names:
                            self._zk_miss_names.append(name)
                            self._zk_miss_ck += float(tensor.float().double().abs().sum())
                        continue
                    tgt_dtype = dest.dtype if dest.dtype.is_floating_point else tensor.dtype
                    src = tensor.to(self.device, tgt_dtype)
                    # Cache a CPU copy of every received weight so zerokl_reapply_cached_weights
                    # can restore them after the post-sync wake_up clobbers the live tensors.
                    if not hasattr(self, "_zk_cached_weights"):
                        self._zk_cached_weights = {}
                    self._zk_cached_weights[name] = src.detach().to("cpu")
                    self._zk_recv_ck = getattr(self, "_zk_recv_ck", 0.0) + float(src.float().double().abs().sum())
                    if dest.is_meta or dest.device.type == "meta" or tuple(dest.shape) != tuple(src.shape):
                        # cumem freed the storage -> param is META; replace the object entirely.
                        _set_on_module(target, name, src, is_param, getattr(dest, "requires_grad", False))
                        materialized += 1
                    else:
                        dest.copy_(src)
                    copied += 1
            self._zerokl_copied = getattr(self, "_zerokl_copied", 0) + copied
            # Arm the engine-forward weight checksum: the wrapper prints [ZEROKL-ENGFWD] with the
            # abs-sum of the weights its NEXT forward actually reads (vs the RECEIVER total below).
            if copied:
                try:
                    model._zk_fwd_ck_pending = True
                except Exception:
                    pass
            if copied:
                # checksum of the LIVE engine gpt params (after this sync) -- compare to SENDER cksum
                _eng = 0.0
                for _n, _p in target.named_parameters():
                    if _p.device.type != "meta":
                        _eng += float(_p.float().double().abs().sum())
                print(f"[ZEROKL-CKSUM] RECEIVER recv-abs-sum={self._zk_recv_ck:.6f} engine-gpt-abs-sum={_eng:.6f}", flush=True)
                _p = next((p for n, p in target.named_parameters() if "weight" in n), None)
                _wn = float(_p.float().norm()) if (_p is not None and _p.device.type != "meta") else -1.0
                print(f"[ZEROKL-SYNC] copied {copied} (materialized {materialized}, cum {self._zerokl_copied}) "
                      f"miss={miss}; live first_w_norm={_wn:.3f}", flush=True)
                if self._zk_miss_names:
                    print(f"[ZEROKL-MISS] {len(self._zk_miss_names)} sender names NOT in engine gpt; "
                          f"missed-abs-sum={self._zk_miss_ck:.6f} (== sender-receiver gap). "
                          f"names={self._zk_miss_names}", flush=True)
                    # also show a few of the engine's ACTUAL param names so the rename is obvious
                    _eng_names = [n for n, _ in target.named_parameters()]
                    print(f"[ZEROKL-MISS] engine gpt has {len(_eng_names)} params; sample={_eng_names[:6]} ... "
                          f"emb/out: {[n for n in _eng_names if 'embed' in n or 'output_layer' in n or 'lm_head' in n][:4]}",
                          flush=True)
            torch.accelerator.synchronize()  # consume IPC tensors before sender drops them
            for weight in weight_list:
                del weight
            return

        weight_update_bracketed = getattr(self, "_skyrl_weight_update_active", False)
        with torch.device(self.device), set_current_vllm_config(self.vllm_config):
            if weight_update_bracketed:
                self.model_runner.model.load_weights(weights=weight_list)
            else:
                self.model_runner.reload_weights(weights_iterator=iter(weight_list))

        if weight_update_bracketed:
            # Finish consuming IPC-backed tensors before the sender drops them on
            # its next barrier; matches NewInferenceWorkerWrap.update_weights_ipc
            torch.accelerator.synchronize()

        for weight in weight_list:
            del weight

    def zerokl_reapply_cached_weights(self) -> int:
        """Re-copy the last synced weights (cached on CPU at sync time) into the live model.

        On the nightly stack ANY wake_up issued AFTER the native sync clobbers the just-synced
        weights (sleep level 1 -> restores the never-updated step-0 CPU backup; level 2 -> zero
        pages), because the wake path is not reliably scoped to the requested tags. The dispatch
        therefore calls this AFTER the final wake_up(tags=["kv_cache"]), so generation always
        runs on the trainer's bytes regardless of what the wake machinery did to the tensors.
        """
        import torch

        cache = getattr(self, "_zk_cached_weights", None)
        if not cache:
            print("[ZEROKL-REAPPLY] no cached weights yet (no native sync seen); skipping", flush=True)
            return 0
        model = self.model_runner.model
        target = model.gpt if hasattr(model, "gpt") else model

        def _set_on_module(root, dotted, value, as_param, requires_grad):
            *path, attr = dotted.split(".")
            mod = root
            for p in path:
                mod = getattr(mod, p)
            if as_param:
                mod._parameters[attr] = torch.nn.Parameter(value, requires_grad=requires_grad)
            else:
                mod._buffers[attr] = value

        copied = 0
        with torch.no_grad():
            params = dict(target.named_parameters())
            bufs = dict(target.named_buffers())
            for name, cpu_t in cache.items():
                dest = params.get(name)
                is_param = dest is not None or name not in bufs
                if dest is None:
                    dest = bufs.get(name)
                src = cpu_t.to(self.device)
                if (
                    dest is None
                    or dest.is_meta
                    or dest.device.type == "meta"
                    or tuple(dest.shape) != tuple(src.shape)
                ):
                    _set_on_module(target, name, src, is_param, getattr(dest, "requires_grad", False))
                else:
                    dest.copy_(src)
                copied += 1
        torch.accelerator.synchronize()
        _s = 0.0
        for _n, _p in target.named_parameters():
            if _p.device.type != "meta":
                _s += float(_p.float().double().abs().sum())
        try:
            model._zk_fwd_ck_pending = True  # arm the [ZEROKL-ENGFWD] forward-time checksum
        except Exception:
            pass
        print(f"[ZEROKL-REAPPLY] reapplied {copied} cached weights; engine-gpt-abs-sum={_s:.6f}", flush=True)
        return copied

    def teardown_weight_receiver(self):
        """Clean up weight receiver resources."""
        if not hasattr(self, "_weight_receiver") or self._weight_receiver is None:
            warnings.warn("No weight receiver to teardown")
            return
        self._weight_receiver.teardown()
