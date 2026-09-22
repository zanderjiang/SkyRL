# IsoExec integration branches

SkyRL's `isoexec-production` branch pairs with IsoExec's production code, including
`cleanup-unused-ops`. Set `PYTHONPATH` to the matching IsoExec and SkyRL checkouts;
an older editable SkyRL installation may otherwise supply incompatible hooks.

The production integration requires an exact pre-update rollout/trainer logprob
comparison. Weight transfer retains handshake, version, complete-coverage,
transport-digest, applied-byte, and post-wake checks. Optimizer completion drains
pending copies before gradient buffers are cleared.

Capture, replay, per-step weight inventories, request traces, and optimizer
forensics are preserved on the `debug` branches of **both** repositories. Use
those matching branches for diagnostics. The older diagnostic SkyRL integration
imports `isoexec.debug` directly and cannot be paired with production IsoExec.
