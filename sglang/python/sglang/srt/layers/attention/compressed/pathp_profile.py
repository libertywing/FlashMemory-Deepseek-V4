"""pathp_profile.py — env-gated CUDA-event profiler for the Path-P decode hot path.

Zero overhead unless SGLANG_PATHP_PROFILE=1. Because the retriever-fire step issues
async GPU work (kernels, H2D copies, a per-layer full-device sync), wall-clock around
individual calls is misleading; we time with paired cuda.Event (elapsed_time gives true
GPU-side ms) and accumulate per-phase totals + call counts, dumping a summary every N
recorded fire-steps.

Usage in hot path:
    from sglang.srt.layers.attention.compressed.pathp_profile import prof
    with prof.region("scorer"):
        lg = scorer.forward(...)
    prof.maybe_dump()

Phases are free-form strings. region() is a no-op context manager when disabled.
"""
from __future__ import annotations

import os
from contextlib import contextmanager

import torch

_ENABLED = os.environ.get("SGLANG_PATHP_PROFILE", "0") == "1"
_DUMP_EVERY = int(os.environ.get("SGLANG_PATHP_PROFILE_EVERY", "200"))


class _Prof:
    def __init__(self):
        self.enabled = _ENABLED
        # phase -> [total_ms, n_calls]
        self._acc: dict[str, list] = {}
        self._n_regions = 0  # total region() exits, used to pace dumps

    @contextmanager
    def region(self, name: str):
        if not self.enabled or torch is None:
            yield
            return
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            # sync only THIS pair (cheap relative to the work measured); needed to read
            # elapsed_time. We are already profiling, so the extra sync is acceptable and
            # never present when disabled.
            end.synchronize()
            ms = start.elapsed_time(end)
            slot = self._acc.setdefault(name, [0.0, 0])
            slot[0] += ms
            slot[1] += 1
            self._n_regions += 1

    def count(self, name: str, n: int = 1):
        """Tally a non-timed counter (e.g. #chunks copied), under '<name>#'."""
        if not self.enabled:
            return
        slot = self._acc.setdefault(name + "#", [0.0, 0])
        slot[0] += n
        slot[1] += 1

    def maybe_dump(self):
        if not self.enabled:
            return
        if self._n_regions and self._n_regions % _DUMP_EVERY < 1:
            self.dump()

    def dump(self):
        if not self.enabled or not self._acc:
            return
        parts = []
        for name in sorted(self._acc):
            tot, n = self._acc[name]
            if name.endswith("#"):
                parts.append(f"{name}sum={tot:.0f} n={n} avg={tot/max(n,1):.1f}")
            else:
                parts.append(f"{name}: {tot:.1f}ms/{n} = {tot/max(n,1):.3f}ms")
        print("[P-PROF] " + " | ".join(parts), flush=True)


prof = _Prof()
