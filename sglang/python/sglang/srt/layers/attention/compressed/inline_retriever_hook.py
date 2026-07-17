"""
inline_retriever_hook.py — Stage 2 inline-retriever score_hook (Lightning Indexer)
==================================================================================

Additive module for the live sglang 0.5.10rc0 tree. It plugs into the EXISTING
score_hook mechanism (deepseek_v4.py:_make_score_hook) instead of adding a
parallel _RETRIEVER_HOOK path, so it coexists with the tracker/dump research line.

What it does (Stage 2, mask-equivalent — NOT yet real swap):
  Every RETRIEVAL_INTERVAL decode steps, for each target CSA layer, run the
  trained Lightning Indexer retriever on the layer's compressed-K + the decode
  hidden state, and mask non-kept block logits to -inf (threshold | sink |
  recent). This is exactly the masking baseline that swap_infra Stage 0 proved
  output-identical to a real swap — so getting this right end-to-end first
  de-risks the swap wiring that follows.

It reuses:
  * _TrainedScorer from retriever_hook_impl.py (same fp8 132B scoring path,
    already validated == inference.py numerically)
  * the compressed-K extraction recipe from deepseek_v4.py dump_training
    (lines ~1466-1494): kv_cache + page_table + seq_lens → [n_blocks, 132]

Enable via env (read once at first hook build):
  SGLANG_RETRIEVER_INLINE=1
  SGLANG_RETRIEVER_INLINE_CKPT=/path/to/r930_joint.pt   (joint ckpt)
  SGLANG_RETRIEVER_INLINE_LAYERS=10,12,20               (CSA layer ids, joint keys l{id})
  SGLANG_RETRIEVER_SIGMOID_THRESH=0.5
  SGLANG_RETRIEVER_INTERVAL=64
  SGLANG_RETRIEVER_LAST_KEEP=2048   (recent c4 tokens always kept)
  SGLANG_RETRIEVER_FIRST_KEEP=0     (attention-sink: first N c4 chunks always kept)
  SGLANG_RETRIEVER_ENSEMBLE_MODE=or|mean

This file is loaded lazily by deepseek_v4._make_score_hook only when
SGLANG_RETRIEVER_INLINE=1, so it is zero-overhead/zero-import otherwise.
"""

from __future__ import annotations

import logging
import os

import torch

from sglang.srt.layers.attention.compressed.pathp_profile import prof

logger = logging.getLogger(__name__)

_HEAD_DIM = 128


class InlineRetrieverHook:
    """Per-(req, layer) cached masking driven by the trained retriever, run inline
    inside the indexer score_hook. One global instance; rebuilt lazily on first use.
    """

    def __init__(self):
        from sglang.srt.layers.attention.compressed.retriever_hook_impl import _TrainedScorer

        ckpt = os.environ.get("SGLANG_RETRIEVER_INLINE_CKPT", "")
        assert ckpt and os.path.exists(ckpt), (
            f"SGLANG_RETRIEVER_INLINE_CKPT invalid: {ckpt!r}"
        )
        layers_str = os.environ.get("SGLANG_RETRIEVER_INLINE_LAYERS", "10,12,20")
        self.target_layers = sorted({int(x) for x in layers_str.split(",") if x.strip()})
        self.interval = int(os.environ.get("SGLANG_RETRIEVER_INTERVAL", "64"))
        self.thresh = float(os.environ.get("SGLANG_RETRIEVER_SIGMOID_THRESH", "0.5"))
        self.last_keep = int(os.environ.get("SGLANG_RETRIEVER_LAST_KEEP", "2048"))
        self.first_keep = int(os.environ.get("SGLANG_RETRIEVER_FIRST_KEEP", "0"))
        self.ensemble_mode = os.environ.get("SGLANG_RETRIEVER_ENSEMBLE_MODE", "or").lower()
        # Capacity protection: cap the resident_set so it never exceeds the GPU reserve
        # (else attention reads cells that were never recalled). 0 = no cap (rely on
        # reserve being large enough); >0 = hard cap, drop lowest-score non-recent chunks.
        self.max_resident = int(os.environ.get("SGLANG_RETRIEVER_MAX_RESIDENT", "0"))
        # STICKY: once a chunk has been recalled (in any past cycle), keep it resident
        # forever (union the new keep-mask with the committed one). Default off = the
        # original per-cycle-recompute behaviour (a chunk with sigmoid<thresh this cycle
        # is evicted). On = a recalled chunk never gets masked out again.
        self.sticky = os.environ.get("SGLANG_RETRIEVER_STICKY", "0") == "1"

        # Load joint ckpt once; build one _TrainedScorer per target layer.
        shared_state = torch.load(ckpt, map_location="cuda", weights_only=True)
        self.scorers = {}
        for lid in self.target_layers:
            self.scorers[lid] = _TrainedScorer(
                ckpt, device="cuda", joint_layer_key=f"l{lid}",
                preloaded_state=shared_state,
            )
        # per req_pool_idx -> {step, c4_seq, keep_mask(bool[c4_seq]), partial{lid:logits}}
        self._cache: dict[int, dict] = {}
        self._call = 0
        logger.info(
            f"[InlineRetriever] init (TWO-LEVEL: resident_set constrains all 21 c4 layers): "
            f"ckpt={ckpt} target_layers={self.target_layers} thresh={self.thresh} "
            f"interval={self.interval} last_keep={self.last_keep} first_keep={self.first_keep} "
            f"ensemble={self.ensemble_mode} max_resident={self.max_resident} sticky={self.sticky}"
        )
        self.first_target = self.target_layers[0]
        self.last_target = self.target_layers[-1]

    @staticmethod
    def _extract_compressed_k(indexer_self, bi: int, n_blocks: int):
        """Mirror dump_training extraction → [n_blocks, 132] uint8 on cuda.
        Returns (k_fp8[n_blocks,128] uint8 view, k_scale[n_blocks,4] uint8)."""
        kv_cache = getattr(indexer_self, "_last_kv_cache", None)
        page_table = getattr(indexer_self, "_last_page_table", None)
        seq_lens_t = getattr(indexer_self, "_last_seq_lens", None)
        if kv_cache is None or page_table is None or seq_lens_t is None:
            return None
        page_size = kv_cache.shape[1]          # 64
        head_dim = _HEAD_DIM
        n_pages = (n_blocks + page_size - 1) // page_size
        pages = page_table[bi, :n_pages]
        kv_flat = kv_cache.view(kv_cache.shape[0], page_size * (head_dim + 4))
        pages_data = kv_flat[pages.long()]      # [n_pages, page_size*132]
        SCALE_OFFSET = page_size * head_dim
        k_fp8 = pages_data[:, :SCALE_OFFSET].reshape(-1, head_dim)[:n_blocks]   # [n_blk,128] uint8
        k_scale = pages_data[:, SCALE_OFFSET:].reshape(-1, 4)[:n_blocks]        # [n_blk,4] uint8
        return k_fp8, k_scale

    @torch.no_grad()
    def __call__(self, logits, seq_lens, forward_batch, indexer_self, compress_layer_id):
        """score_hook body. logits:[B,max_blk]; returns masked logits.

        TWO-LEVEL recall (see memory flashmemory-two-level-inference-algorithm):
          * Level 1 — Memory Indexer (this trained retriever, target layers only):
            every `interval` decode steps, ensemble the 3 target-layer sigmoid scores
            and finalize ONE global resident_set (threshold | recent | sink), optionally
            capped to the GPU reserve capacity. This defines what stays on-device.
          * Level 2 — native Lightning Indexer (ALL 21 c4 layers incl. the target ones):
            EVERY layer EVERY step has its native top-512 CONFINED to the resident_set
            by masking resident-set-excluded chunk logits to -inf. So even the target
            layers' own native indexer can only pick from the resident_set.

        The resident_set is double-buffered: `active` (committed, applied by all 21
        layers this window) vs `pending` (being scored mid-cycle). active is only
        replaced when a fresh cycle fully finalizes → n→n+1 latency (the retriever
        predicts the next ~64 steps anyway; recent/sink cover the newest chunks).
        Non-target layers ONLY apply the active mask; they never score / advance cadence.
        """
        if not forward_batch.forward_mode.is_decode():
            return logits
        device = logits.device
        req_ids = forward_batch.req_pool_indices.tolist()

        is_target = compress_layer_id in self.target_layers
        is_first = is_target and (compress_layer_id == self.first_target)
        is_last = is_target and (compress_layer_id == self.last_target)

        # cadence counter advances once per target layer per step (= 3x/step) so that
        # interval*len(target_layers) == interval decode steps. Non-target layers and
        # the prefill/no-x paths must NOT advance it.
        x = getattr(indexer_self, "_last_x", None)
        positions = getattr(indexer_self, "_last_positions", None)
        if is_target and x is not None:
            self._call += 1

        B = logits.shape[0]
        for bi in range(B):
            ri = req_ids[bi]
            n_blk = int(seq_lens[bi].item())
            if n_blk <= 0:
                continue
            entry = self._cache.get(ri)

            # ── Level 1 (target layers only): score + finalize a fresh resident_set ──
            if is_target and x is not None:
                if entry is None:
                    entry = {"active": None, "active_seq": 0, "partial": {},
                             "pend_step": -10**9, "pend_seq": 0, "last_cycle": -10**9}
                    self._cache[ri] = entry
                pending = entry["pend_step"] > -10**9
                # open a NEW cycle only when: not already pending, AND (no active yet,
                # OR c4_seq jumped/shrank, OR `interval` decode steps since last cycle).
                # Gate the interval on `last_cycle` (last cycle START), NOT pend_step
                # (which is -inf after finalize → would fire every step).
                need_fresh = is_first and (not pending) and (
                    entry["active"] is None
                    or n_blk < entry.get("active_seq", 0)
                    or (n_blk - entry.get("active_seq", 0)) > 64
                    or (self._call - entry["last_cycle"]) >= self.interval * len(self.target_layers)
                )
                if need_fresh:
                    entry["partial"] = {}
                    entry["pend_step"] = self._call
                    entry["pend_seq"] = n_blk
                    entry["last_cycle"] = self._call
                    pending = True

                # score this layer into the pending partial (mid-cycle only)
                in_cycle = pending and (compress_layer_id not in entry["partial"]) \
                    and (entry["pend_seq"] == n_blk)
                if in_cycle:
                    with prof.region("extract_k"):
                        ek = self._extract_compressed_k(indexer_self, bi, n_blk)
                    if ek is not None:
                        k_fp8, k_scale = ek
                        k_fp8_v = k_fp8.contiguous().view(torch.float8_e4m3fn)
                        k_scale_v = k_scale.contiguous().view(torch.float32).squeeze(-1)
                        pos = positions[bi:bi+1] if positions is not None else torch.tensor([n_blk], device=device)
                        scorer = self.scorers[compress_layer_id]
                        with prof.region("scorer"):
                            lg = scorer.forward(x[bi:bi+1], k_fp8_v, k_scale_v, pos.to(torch.int64))
                        entry["partial"][compress_layer_id] = lg[0].float()

                # finalize on last target once all target layers scored this cycle
                if is_last and all(l in entry["partial"] for l in self.target_layers) \
                        and entry["pend_seq"] == n_blk:
                    new_mask = self._finalize_resident(entry["partial"], n_blk, device, ri)
                    # STICKY: union with the previously committed set so a chunk that
                    # was recalled in any past cycle stays resident (never re-evicted).
                    if self.sticky and entry["active"] is not None:
                        old = entry["active"]
                        w = min(old.shape[0], new_mask.shape[0])
                        new_mask[:w] |= old[:w]
                    entry["active"] = new_mask          # commit (double-buffer swap)
                    # Cumulative resident count (post-union). In sticky mode this only
                    # grows; logs the accumulated GPU-resident chunk count + ratio.
                    _ncum = int(new_mask.sum().item())
                    logger.info(
                        f"[InlineRetriever] CUM req={ri} c4_seq={n_blk} "
                        f"resident_cum={_ncum} ({_ncum/max(n_blk,1):.1%}) sticky={self.sticky}"
                    )
                    entry["active_seq"] = n_blk
                    entry["partial"] = {}
                    entry["pend_step"] = -10**9

            # ── Level 2 (ALL 21 layers): confine native top-512 to the active resident_set ──
            km = entry.get("active") if entry is not None else None
            if km is not None:
                m = km.shape[0]
                w = min(m, logits.shape[1])
                drop = logits.new_ones(logits.shape[1], dtype=torch.bool)
                drop[:w] = ~km[:w]
                logits[bi] = logits[bi].masked_fill(drop, float("-inf"))

        prof.maybe_dump()
        return logits

    def _finalize_resident(self, partial, n_blk, device, ri):
        """Ensemble the per-target-layer logits into ONE global resident keep-mask
        [n_blk] bool: (sigmoid>thresh | recent | sink), capped to reserve capacity."""
        sigs = torch.stack([torch.sigmoid(partial[l]) for l in self.target_layers], 0)
        if self.ensemble_mode == "or":
            ens = sigs.max(0).values
        else:
            ens = sigs.mean(0)
        thr = ens > self.thresh
        ar = torch.arange(n_blk, device=device)
        recent = ar >= max(0, n_blk - self.last_keep)
        sink = ar < min(self.first_keep, n_blk)
        keep = thr | recent | sink
        # capacity protection: never let the resident_set exceed the GPU reserve, else
        # attention would read cells that were never recalled. Drop lowest-score chunks
        # (but always keep recent|sink).
        if self.max_resident > 0 and int(keep.sum()) > self.max_resident:
            forced = recent | sink
            n_forced = int(forced.sum())
            budget = max(0, self.max_resident - n_forced)
            cand = thr & ~forced
            cand_idx = cand.nonzero(as_tuple=True)[0]
            if cand_idx.numel() > budget:
                cand_scores = ens[cand_idx]
                topb = cand_scores.topk(budget).indices if budget > 0 else cand_scores.new_empty(0, dtype=torch.long)
                keep = forced.clone()
                if budget > 0:
                    keep[cand_idx[topb]] = True
        self._finalize_count = getattr(self, "_finalize_count", 0) + 1
        if self._finalize_count <= 20 or self._finalize_count % 200 == 0:
            n_keep = int(keep.sum().item())
            logger.info(
                f"[InlineRetriever] resident#{self._finalize_count} req={ri} "
                f"c4_seq={n_blk} resident={n_keep} ({n_keep/max(n_blk,1):.1%}) "
                f"thr_only={int(thr.sum().item())} cap={self.max_resident}"
            )
        return keep


_INLINE_HOOK = None


def get_inline_hook():
    """Lazy singleton; built only when SGLANG_RETRIEVER_INLINE=1."""
    global _INLINE_HOOK
    if _INLINE_HOOK is None:
        _INLINE_HOOK = InlineRetrieverHook()
    return _INLINE_HOOK
