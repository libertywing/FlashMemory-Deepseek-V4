from __future__ import annotations

import concurrent.futures
import logging
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, List, Literal, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

import sglang.srt.models.deepseek_v2 as deepseek_v2
from sglang.jit_kernel.deepseek_v4 import fused_rope, linear_bf16_fp32, rmsnorm_self
from sglang.srt.configs.deepseek_v4 import DeepSeekV4Config
from sglang.srt.debug_utils.deepseek_v4_debug_utils import (
    deepseek_v4_moe_code_path_checker,
)
from sglang.srt.distributed import get_pp_group, get_tensor_model_parallel_world_size
from sglang.srt.distributed.parallel_state import get_moe_expert_parallel_world_size
from sglang.srt.environ import envs, is_large_dummy_model
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.layers.attention.nsa.nsa_indexer import rotate_activation
from sglang.srt.layers.attention.nsa.utils import (
    assert_tensor_identical_across_cp_ranks,
    can_cp_split,
    cp_all_gather_rerange_output,
    cp_split_and_rebuild_data,
    cp_split_and_rebuild_position,
    is_nsa_enable_prefill_cp,
    nsa_use_prefill_cp,
    prepare_input_dp_with_cp_dsa,
)
from sglang.srt.layers.communicator import LayerScatterModes, get_attn_tp_context
from sglang.srt.layers.deepseek_v4_rope import apply_rotary_emb_triton
from sglang.srt.layers.dp_attention import (
    _DpGatheredBufferWrapper,
    attn_tp_all_gather,
    dp_gather_partial,
    dp_scatter,
    get_attention_dp_size,
    get_attention_tp_rank,
    get_attention_tp_size,
    get_global_dp_buffer,
    get_local_dp_buffer,
    is_dp_attention_enabled,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe import get_moe_a2a_backend
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_fp8
from sglang.srt.layers.rotary_embedding import get_rope_wrapper
from sglang.srt.layers.utils import get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.mem_cache.compress_state import CompressStatePool
from sglang.srt.mem_cache.deepseekv4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.memory_pool import RadixAttention
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
from sglang.srt.model_loader.utils import maybe_executor_submit, should_async_load
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.dbrx import ReplicatedLinear
from sglang.srt.models.deepseek_v2 import ParallelLMHead, _is_cuda, _is_hip, _is_npu
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import (
    BumpAllocator,
    LazyValue,
    add_prefix,
    log_info_on_rank0,
    make_layers,
    maybe_torch_compile,
)

logger = logging.getLogger(__name__)

from sglang.srt.environ import envs

# ===========================================================================
# 以下是对原始 deepseek_v4.py 的所有改动（用于 CSA 召回实验）
# ===========================================================================
#
# 【整体架构】
#   客户端(run_normal.py等)       ←→  /tmp/dsv4_tracker_cmd.json     →  server(本文件)
#   客户端(run_topk_experiment等) ←   /tmp/dsv4_tracker_result.json  ←  server(本文件)
#
# 【通信协议】
#   客户端写 cmd 文件告诉 server 当前请求要做什么：
#     {"mode": "pass1"}  → server 在 decode 阶段统计哪些 CSA block 被 topk 选中
#     {"mode": "pass2", "recalled_blocks": [...], "n_prompt_blocks": N}
#                        → server 在 decode 阶段 mask 掉非 recalled_blocks 的 block
#     {"mode": "none"}   → server 正常推理，不做任何额外操作
#
# 【Pass-1 记录流程】
#   每个 decode step × 每个 CSA layer：
#     1. C4Indexer.forward() 调用 _make_score_hook() 构建 hook
#     2. compressed_indexer.py 算出所有 block 的 logits 后调用 hook
#     3. hook 里对 logits 做 topk，选出的 block 编号 +1 分
#     4. 每 100 次写一次 result 文件；请求结束时再 flush 一次
#   最终 result 文件包含 block_scores_ranked（按被选中次数排序）
#
# 【Pass-2 屏蔽流程】
#   每个 decode step × 每个 CSA layer：
#     1. hook 把非 recalled_blocks 的 block 的 logits 设为 -inf
#     2. answer 部分的 block（编号 >= n_prompt_blocks）自动跳过 mask
#     3. 之后 topk_transform 正常选 block，但 -inf 的永远不会被选中
#   注意：prefill 阶段 hook 直接 return，不做任何干预
#
# 【改动位置】
#   1. 第 82~239 行：全局变量 + 通信函数 + 记录/flush 函数（全部新增）
#   2. 第 649~651 行：C4Indexer.__init__ 末尾新增 self.score_hook = None
#   3. 第 671~739 行：C4Indexer 新增 _make_score_hook() 方法 + 修改 forward()
#   4. compressed_indexer.py 第 405~420 行：forward_c4_indexer 里调用 hook
# ===========================================================================

import json as _json
import os as _os

# ── 通信文件路径 ──
_TRACKER_CMD_FILE    = "/tmp/dsv4_tracker_cmd.json"    # 客户端 → server：告诉 server 做什么
_TRACKER_RESULT_FILE = "/tmp/dsv4_tracker_result.json"  # server → 客户端：返回统计结果

# ── req_pool_idx → rid 映射表（由 schedule_batch.py 更新） ──
_req_pool_to_rid: dict = {}  # {req_pool_idx: rid_string}

# ── server 进程内全局状态（整个 decode 阶段累积，每个新请求重置） ──
_recalled_blocks: set = set()     # 所有 CSA 层、所有 step 中被 topk 选中过的 block 编号并集
_recalled_tokens: set = set()     # 上面每个 block × compress_ratio=4 展开后的 token 位置
_pass1_active:   bool = False     # True = 当前在做 Pass-1 统计
_pass2_mask:      set = None      # Pass-2 时的保留 block 集合；不在其中的 block 被 mask
_pass2_n_prompt_blocks: int = 0   # Pass-2 时 prompt 的 block 数；answer 部分(>=此值)不 mask
_cmd_mtime:     float = 0.0       # 上次读 cmd 文件的 mtime，用于检测新请求

# ── Pass-2 动态 mask（按 interval 切换） ──
_pass2_dynamic_masks: list = None   # [set(...), set(...), ...] 每个 interval 的 keep_blocks
_pass2_dynamic_interval: int = 0    # 每个 interval 包含多少个 decode token
_pass2_dynamic_token_count: int = 0 # 当前已过多少个 decode token
_pass2_dynamic_layer_seen: int = 0  # 当前 token 内过了多少个 CSA layer
_pass2_dynamic_idx: int = 0         # 当前使用第几个 interval 的 mask

# ── Pass-1 打分相关 ──
_block_scores: dict = {}    # {block_id: int} 每个 block 被 topk 选中的累计次数（过滤后）
_csa_layer_count: int = 0   # hook 被调用的总次数（= decode_steps × CSA 层数）
_last_n_blocks: int = 0     # 最近一次 hook 调用时的总 block 数（含 answer 部分）
_first_n_blocks: int = 0    # decode 第一步时的 block 数 = 纯 prompt 的 block 数

# ── Pass-1 per-token 层间过滤 ──
# 每个 decode token 经过 21 个 CSA layer，先在 _token_layer_counts 中统计每个 chunk
# 被多少个 layer 选中，等该 token 所有 layer 都过完后，只保留 ≥ _MIN_LAYER_HITS 的 chunk
_MIN_LAYER_HITS: int = 10         # 阈值：一个 chunk 必须被 ≥10 个 layer 选中才计入 score
_token_layer_counts: dict = {}    # {block_id: int} 当前 token 内各 chunk 被选中的 layer 数
_token_layer_seen: int = 0        # 当前 token 内已经过了多少个 CSA layer
_total_csa_layers: int = 21       # 总 CSA layer 数（从 config 推算）

# ── Pass-1 Top-P 模式 ──
# 每个 CSA layer 对 topk logits 算 softmax，取累积概率 >= topp_threshold 的 chunk
# 只对 prompt 部分的 chunk 计分（+1）
_pass1_topp_active: bool = False   # True = 使用 top-p 计分模式
_pass1_topp_threshold: float = 0.8 # top-p 阈值
_pass1_topp_layer_seen: int = 0    # 当前 token 过了多少个 CSA layer（用于 token 计数）
_pass1_topp_token_id: int = 0      # 当前 decode token 编号

# ── Pass-1 快照相关（每 N 个 decode token 保存一次召回 block 快照） ──
_snapshot_interval: int = 0        # 快照间隔（单位：decode token），0 = 不做快照
_snapshot_token_count: int = 0     # 当前已过的 decode token 数
_snapshot_prev_n_blocks: int = 0   # 上一次 hook 调用时的 n_blocks，用于检测新 token 边界
_snapshot_prev_scores: dict = {}   # 上一次快照时刻的 _block_scores 副本，用于计算增量
_snapshots: list = []              # 快照列表

# ── Pass-1 Top-P + 多层联合过滤 ──
# 结合 top-p 和多层一致性：对每个 decode token，只有在 ≥ min_layers 个 layer 的 top-p 内的 chunk 才 +1 分
_pass1_topp_nlayer_active: bool = False      # True = 使用 top-p + 多层联合过滤模式
_pass1_topp_nlayer_threshold: float = 0.4    # top-p 阈值
_pass1_topp_nlayer_min_layers: int = 10      # chunk 必须在 ≥ 这么多个 layer 的 top-p 中才计分
_pass1_topp_nlayer_layer_seen: int = 0       # 当前 token 过了多少个 CSA layer
_pass1_topp_nlayer_token_id: int = 0         # 当前 decode token 编号
_pass1_topp_nlayer_counts: dict = {}         # {block_id: int} 当前 token 内各 chunk 通过 top-p 的 layer 数

# ── Logits dump 模式：记录每个 token × 每个 CSA layer 的 topk logits ──
_logits_dump_active: bool = False       # True = 当前请求需要 dump logits
_logits_dump_data: list = []            # [(decode_token_id, layer_num, [(block_id, logit_value), ...]), ...]
_logits_dump_token_id: int = 0          # 当前 decode token 编号
_logits_dump_layer_seen: int = 0        # 当前 token 内已过多少个 CSA layer

# ── Dump Training Data 模式：生产 retriever 训练数据 ──
# 记录 decoding hidden state + prompt compressed K (多层) + per-decode-token golden chunk labels
_dump_training_active: bool = False
_dump_training_topp: float = 0.4             # top-p 阈值
_dump_training_min_layers: int = 10          # ≥ 这么多层通过 top-p 才记为 golden
_dump_training_target_csa_indices: list = [1, 12, 20]  # 记录第 2, 13, 21 个 CSA layer (0-indexed)
_dump_training_output_dir: str = "/tmp"      # 输出目录
_dump_training_doc_counter: int = 0          # 自增文档计数器（用于文件命名）

# per-request 状态字典：{rid: state_dict}
# 每个 state_dict 包含该请求的所有追踪状态
_dump_training_states: dict = {}

# 兼容旧串行模式的全局变量（保留给 _flush_training_data 使用）
_dump_training_layer_seen: int = 0
_dump_training_token_id: int = 0
_dump_training_per_token_counts: dict = {}
_dump_training_labels: list = []
_dump_training_hidden_state: object = None
_dump_training_compressed_k: object = None
_dump_training_prefill_done: bool = False


def _read_cmd() -> dict:
    try:
        with open(_TRACKER_CMD_FILE, "r") as f:
            return _json.load(f)
    except Exception:
        return {"mode": "none"}


def _maybe_init():
    """
    【何时被调用】每个 decode step 的每个 CSA layer 都会调用（通过 _make_score_hook）。
    【做什么】检查 cmd 文件的 mtime 是否变化：
      - 没变 → 直接 return（同一请求内不重复初始化）
      - 变了 → 读取新 cmd，根据 mode 重置状态：
          pass1: 清空所有统计，开启记录
          pass2: 设置 mask 集合
          flush: 写出当前统计结果
          none:  关闭一切
    【为什么用 mtime】客户端每写一次 cmd 文件，mtime 就变，无需"请求结束"信号。
    """
    global _recalled_blocks, _recalled_tokens, _pass1_active, _pass2_mask, _cmd_mtime
    global _block_scores, _csa_layer_count, _last_n_blocks, _first_n_blocks, _pass2_n_prompt_blocks
    global _snapshot_interval, _snapshot_token_count, _snapshot_prev_n_blocks, _snapshot_prev_scores, _snapshots
    global _token_layer_counts, _token_layer_seen
    global _pass2_dynamic_masks, _pass2_dynamic_interval, _pass2_dynamic_token_count
    global _pass2_dynamic_layer_seen, _pass2_dynamic_idx
    global _logits_dump_active, _logits_dump_data, _logits_dump_token_id, _logits_dump_layer_seen
    global _pass1_topp_active, _pass1_topp_threshold, _pass1_topp_layer_seen, _pass1_topp_token_id
    global _pass1_topp_nlayer_active, _pass1_topp_nlayer_threshold, _pass1_topp_nlayer_min_layers
    global _pass1_topp_nlayer_layer_seen, _pass1_topp_nlayer_token_id, _pass1_topp_nlayer_counts
    global _dump_training_active, _dump_training_topp, _dump_training_min_layers
    global _dump_training_target_csa_indices, _dump_training_output_dir
    global _dump_training_states, _dump_training_doc_counter
    try:
        mtime = _os.path.getmtime(_TRACKER_CMD_FILE)
    except FileNotFoundError:
        return
    if mtime <= _cmd_mtime:
        return   # cmd 文件没有更新，当前请求已初始化过，直接返回
    _cmd_mtime = mtime   # 更新记录的 mtime
    cmd = _read_cmd()
    mode = cmd.get("mode", "none")

    # 如果之前在 pass1 统计中，现在切换到其他 mode，先写出结果
    if _pass1_active and mode != "pass1":
        _flush_scores()

    if mode == "pass1":
        # Pass-1：清空上一次的召回记录，开启统计
        _recalled_blocks = set()
        _recalled_tokens = set()
        _block_scores = {}
        _csa_layer_count = 0
        _last_n_blocks = 0
        _first_n_blocks = 0  # 会在第一次 _record_blocks_and_flush 调用时设置
        _pass1_active    = True
        _pass2_mask      = None
        _token_layer_counts = {}
        _token_layer_seen = 0
        # 快照配置：snapshot_interval > 0 时，每生成 N 个 token 保存一次召回 block 快照
        _snapshot_interval = cmd.get("snapshot_interval", 0)
        _snapshot_token_count = 0
        _snapshot_prev_n_blocks = 0
        _snapshot_prev_scores = {}
        _snapshots = []
        try:
            _os.remove(_TRACKER_RESULT_FILE)   # 删除旧结果，避免客户端读到过期数据
        except FileNotFoundError:
            pass
    elif mode == "pass1_topp":
        # Pass-1 Top-P：每个 layer 算 softmax → 取 top-p chunk → 只对 prompt chunk +1
        _recalled_blocks = set()
        _recalled_tokens = set()
        _block_scores = {}
        _csa_layer_count = 0
        _last_n_blocks = 0
        _first_n_blocks = 0
        _pass1_active = False  # 不走原来的 pass1 逻辑
        _pass1_topp_active = True
        _pass1_topp_threshold = cmd.get("topp", 0.8)
        _pass1_topp_layer_seen = 0
        _pass1_topp_token_id = 0
        _pass2_mask = None
        _pass2_dynamic_masks = None
        # 快照
        _snapshot_interval = cmd.get("snapshot_interval", 0)
        _snapshot_token_count = 0
        _snapshot_prev_n_blocks = 0
        _snapshot_prev_scores = {}
        _snapshots = []
        try:
            _os.remove(_TRACKER_RESULT_FILE)
        except FileNotFoundError:
            pass
    elif mode == "pass1_topp_nlayer":
        # Pass-1 Top-P + 多层联合过滤：
        # 每个 layer 算 top-p chunk，但只有在 ≥ min_layers 个 layer 都通过 top-p 的 chunk 才 +1 分
        _recalled_blocks = set()
        _recalled_tokens = set()
        _block_scores = {}
        _csa_layer_count = 0
        _last_n_blocks = 0
        _first_n_blocks = 0
        _pass1_active = False
        _pass1_topp_active = False
        _pass1_topp_nlayer_active = True
        _pass1_topp_nlayer_threshold = cmd.get("topp", 0.4)
        _pass1_topp_nlayer_min_layers = cmd.get("min_layers", 10)
        _pass1_topp_nlayer_layer_seen = 0
        _pass1_topp_nlayer_token_id = 0
        _pass1_topp_nlayer_counts = {}
        _pass2_mask = None
        _pass2_dynamic_masks = None
        # 快照
        _snapshot_interval = cmd.get("snapshot_interval", 0)
        _snapshot_token_count = 0
        _snapshot_prev_n_blocks = 0
        _snapshot_prev_scores = {}
        _snapshots = []
        try:
            _os.remove(_TRACKER_RESULT_FILE)
        except FileNotFoundError:
            pass
    elif mode == "pass2":
        # Pass-2（静态）：关闭统计，加载 Pass-1 的召回集合用于 mask
        _pass1_active = False
        _pass2_mask   = set(cmd.get("recalled_blocks", []))
        _pass2_n_prompt_blocks = cmd.get("n_prompt_blocks", 0)
        _pass2_dynamic_masks = None
    elif mode == "pass2_dynamic":
        # Pass-2（动态）：按 interval 切换 mask
        # cmd 格式: {"mode": "pass2_dynamic", "interval_masks": [[block_ids...], ...],
        #            "interval": 64, "n_prompt_blocks": N}
        _pass1_active = False
        _pass2_mask = None  # 不用静态 mask
        _pass2_n_prompt_blocks = cmd.get("n_prompt_blocks", 0)
        _pass2_dynamic_interval = cmd.get("interval", 64)
        _pass2_dynamic_token_count = 0
        _pass2_dynamic_layer_seen = 0
        _pass2_dynamic_idx = 0
        # 将每个 interval 的 keep_blocks 转为 set
        raw_masks = cmd.get("interval_masks", [])
        _pass2_dynamic_masks = [set(m) for m in raw_masks]
        # 用第一个 interval 的 mask 作为初始 _pass2_mask
        if _pass2_dynamic_masks:
            _pass2_mask = _pass2_dynamic_masks[0]
    elif mode == "flush":
        # flush：将当前累积的统计结果写入 result 文件，不改变 pass1/pass2 状态
        _flush_scores()
    elif mode == "logits_dump":
        # logits_dump：记录每个 token × 每个 CSA layer 的 topk logits
        _pass1_active = False
        _pass2_mask = None
        _pass2_dynamic_masks = None
        _logits_dump_active = True
        _logits_dump_data = []
        _logits_dump_token_id = 0
        _logits_dump_layer_seen = 0
        _first_n_blocks = 0
        try:
            _os.remove(_TRACKER_RESULT_FILE)
        except FileNotFoundError:
            pass
    elif mode == "pass1_dump_training":
        # dump_training：生产 retriever 训练数据（支持并行）
        # per-request 状态在 hook 中按需创建，这里只设置全局参数
        _pass1_active = False
        _pass1_topp_active = False
        _pass1_topp_nlayer_active = False
        _pass2_mask = None
        _pass2_dynamic_masks = None
        _logits_dump_active = False
        _dump_training_active = True
        _dump_training_topp = cmd.get("topp", 0.4)
        _dump_training_min_layers = cmd.get("min_layers", 10)
        _dump_training_target_csa_indices = cmd.get("target_csa_layers", [1, 12, 20])
        _dump_training_output_dir = cmd.get("output_dir", "/tmp")
        _dump_training_doc_counter = cmd.get("doc_counter_start", 0)  # 支持从指定值开始（追加模式）
        _dump_training_states = {}      # 清空残留状态
        try:
            _os.remove(_TRACKER_RESULT_FILE)
        except FileNotFoundError:
            pass
    else:
        # mode=none：关闭所有功能
        _pass1_active = False
        _pass2_mask   = None
        _pass2_dynamic_masks = None
        # 如果之前在 logits_dump 模式，写出结果
        if _logits_dump_active:
            _flush_logits_dump()
            _logits_dump_active = False
        # 如果之前在 pass1_topp 模式，写出结果
        if _pass1_topp_active:
            _flush_scores()
            _pass1_topp_active = False
        # 如果之前在 pass1_topp_nlayer 模式，写出结果
        if _pass1_topp_nlayer_active:
            _flush_scores()
            _pass1_topp_nlayer_active = False
        # 如果之前在 dump_training 模式，flush 所有剩余的 per-request 状态
        if _dump_training_active:
            _flush_all_training_states()
            _dump_training_active = False


def _record_blocks_and_flush(raw_topk: "torch.Tensor", n_blocks: int, compress_ratio: int = 4):
    """
    【何时被调用】Pass-1 decode 阶段，每个 token × 每个 CSA layer 调用一次（所有 21 层）。
    【做什么】
      1. per-token 层间统计：在 _token_layer_counts 中记录每个 chunk 被当前 token 的多少个 layer 选中
      2. 当一个 token 的所有 layer 都过完后（_token_layer_seen == _total_csa_layers）：
         - 只保留被 ≥ _MIN_LAYER_HITS 个 layer 选中的 chunk，计入 _block_scores（+1分）
         - 重置 per-token 计数器
      3. 快照：若 snapshot_interval > 0，每 N 个 decode token 保存一次增量快照
    【参数】
      raw_topk: indexer topk 选出的 block 编号（1D tensor）
      n_blocks: 当前时刻总 block 数（随 decode 推进而增长）
      compress_ratio: 每个 block 对应几个原始 token（DSV4 固定为 4）
    """
    global _csa_layer_count, _last_n_blocks, _first_n_blocks
    global _snapshot_token_count, _snapshot_prev_n_blocks, _snapshot_prev_scores
    global _token_layer_counts, _token_layer_seen
    if not _pass1_active:
        return

    _csa_layer_count += 1
    _last_n_blocks = n_blocks
    # 第一次调用时记录 prompt 部分的 block 数（decode 第一步 = prefill 刚结束）
    if _first_n_blocks == 0:
        _first_n_blocks = n_blocks
        _snapshot_prev_n_blocks = n_blocks

    # 1. per-token 层间统计：记录每个 chunk 被当前 token 的哪些 layer 选中
    topk_indices = raw_topk.tolist()
    for b in topk_indices:
        if b < 0:
            continue
        _token_layer_counts[b] = _token_layer_counts.get(b, 0) + 1

    _token_layer_seen += 1

    # 2. 当前 token 的所有 CSA layer 都过完了 → 做阈值过滤，计入最终 score
    if _token_layer_seen >= _total_csa_layers:
        for b, layer_hits in _token_layer_counts.items():
            if layer_hits >= _MIN_LAYER_HITS:
                # 只统计 prompt 部分的 chunk（编号 < _first_n_blocks），跳过 answer 部分
                if b >= _first_n_blocks:
                    continue
                _block_scores[b] = _block_scores.get(b, 0) + layer_hits
                # 记录 recalled_blocks 并集
                if b not in _recalled_blocks:
                    _recalled_blocks.add(b)
                    for t in range(b * compress_ratio, (b + 1) * compress_ratio):
                        _recalled_tokens.add(t)
        # 重置 per-token 计数器
        _token_layer_counts = {}
        _token_layer_seen = 0

        # 3. 快照：每完成一个 decode token 就计数，达到 interval 时保存快照
        if _snapshot_interval > 0:
            _snapshot_token_count += 1
            if _snapshot_token_count >= _snapshot_interval:
                # 增量快照：只记录本 interval 内新增/变化的 block 及其在本 interval 内的得分增量
                interval_scores = {}
                for b, total_score in _block_scores.items():
                    prev_score = _snapshot_prev_scores.get(b, 0)
                    delta = total_score - prev_score
                    if delta > 0:
                        interval_scores[b] = delta
                # 按本 interval 内的增量 score 从高到低排序
                interval_ranked = sorted(interval_scores.items(), key=lambda x: x[1], reverse=True)
                _snapshots.append({
                    "decode_token": _snapshot_token_count,
                    "n_blocks": n_blocks,
                    "recalled_blocks_with_scores": interval_ranked,
                })
                # 保存当前 scores 作为下一个 interval 的基准
                _snapshot_prev_scores = dict(_block_scores)
                _snapshot_token_count = 0  # 重置计数器


def _flush_scores():
    """
    【何时被调用】
      客户端写 {"mode":"none"} 后，下一个请求（如发 "hi"）的第一次 hook 触发 _maybe_init，
      检测到 pass1→其他 mode 的切换时调用，将完整统计结果一次性写入文件。
    【做什么】将内存中累积的所有统计写入 /tmp/dsv4_tracker_result.json：
      - block_scores_ranked: 按被选中次数从高到低排序的 [(block_id, count), ...]
      - n_prompt_blocks: prompt 部分的 block 数（用于 Pass-2 判断 answer 边界）
      - snapshots: 如果开了快照，附带各 interval 的增量快照
    """
    sorted_scores = sorted(_block_scores.items(), key=lambda x: x[1], reverse=True)

    # 如果有未满 interval 的尾巴 token，补一个最后的快照
    if _snapshot_interval > 0 and _snapshot_token_count > 0:
        interval_scores = {}
        for b, total_score in _block_scores.items():
            prev_score = _snapshot_prev_scores.get(b, 0)
            delta = total_score - prev_score
            if delta > 0:
                interval_scores[b] = delta
        if interval_scores:
            interval_ranked = sorted(interval_scores.items(), key=lambda x: x[1], reverse=True)
            _snapshots.append({
                "decode_token": _snapshot_token_count,
                "n_blocks": _last_n_blocks,
                "recalled_blocks_with_scores": interval_ranked,
            })

    result = {
        "recalled_blocks": sorted(_recalled_blocks),
        "recalled_tokens": sorted(_recalled_tokens),
        "n_total_tokens":  max(_recalled_tokens) + 1 if _recalled_tokens else 0,
        "n_csa_layer_calls": _csa_layer_count,
        "n_decode_tokens": _csa_layer_count // _total_csa_layers,  # decoding 阶段生成的 token 数
        "n_blocks": _last_n_blocks,
        "n_prompt_blocks": _first_n_blocks,  # decode 第一步时的 block 数 = prompt 部分
        "block_scores_ranked": sorted_scores,
        "total_blocks_with_scores": len(sorted_scores),
    }
    # 如果有快照数据，附加到 result 中
    if _snapshots:
        result["snapshots"] = _snapshots
        result["snapshot_interval"] = _snapshot_interval
    try:
        with open(_TRACKER_RESULT_FILE, "w") as f:
            _json.dump(result, f)
    except Exception:
        pass


def _flush_logits_dump():
    """将 logits dump 数据写入 result 文件。"""
    result = {
        "mode": "logits_dump",
        "n_decode_tokens": _logits_dump_token_id,
        "n_csa_layers": _total_csa_layers,
        "n_records": len(_logits_dump_data),
        "n_prompt_blocks": _first_n_blocks,
        "data": _logits_dump_data,
    }
    try:
        with open(_TRACKER_RESULT_FILE, "w") as f:
            _json.dump(result, f)
    except Exception:
        pass


def _flush_one_training_state(rid: str, state: dict):
    """
    将单个请求的训练数据存到磁盘。
    文件名使用自增计数器（doc_00000, doc_00001, ...）。
    跳过极短请求（如触发用的 "hi"/"flush"）。

    存储内容（单个 safetensors 文件）：
      - hidden_layer_{idx}: [n_decode_tokens, hidden_dim] 每个 target layer 的 decode hidden states
      - compk_layer_{idx}: [n_prompt_blocks, 132] 每个 target layer 的 prompt compressed K (前128=FP8 key, 后4=float32 scale)
      - label_indices: [N] int32
      - label_scores: [N] int32
      - label_pointers: [n_decode_tokens+1] int32
    """
    import torch as _torch
    global _dump_training_doc_counter

    # 过滤掉极短请求（触发 flush 用的 "hi" 等）
    first_n_blocks = state.get("first_n_blocks", 0)
    if first_n_blocks <= 2:
        return

    # 使用创建时固定的 output_dir 和 doc_idx（防止 cmd 切换后 flush 到错误目录/编号）
    out_dir = state.get("_assigned_output_dir", _dump_training_output_dir)
    doc_idx = state.get("_assigned_doc_idx", None)
    if doc_idx is None:
        # 兼容没有 _assigned_doc_idx 的旧 state
        doc_idx = _dump_training_doc_counter
        _dump_training_doc_counter += 1

    hidden_states_dict = state.get("hidden_states", {})
    compressed_k_dict = state.get("compressed_k", {})
    labels = state.get("labels", [])
    token_id = state.get("token_id", 0)

    # 组装所有 tensor 到一个 dict（用于 safetensors）
    tensors = {}
    hidden_shapes = {}
    compk_shapes = {}

    # hidden states（多层）：每层 concat 所有 token 的 hidden state
    for csa_idx, h_list in hidden_states_dict.items():
        if h_list:
            cat_h = _torch.cat(h_list, dim=0)  # [n_decode_tokens, hidden_dim]
            tensors[f"hidden_layer_{csa_idx}"] = cat_h
            hidden_shapes[f"layer_{csa_idx}"] = list(cat_h.shape)

    # positions（多层）：每层每个 decode token 的绝对位置
    positions_dict = state.get("positions", {})
    for csa_idx, pos_list in positions_dict.items():
        tensors[f"positions_layer_{csa_idx}"] = _torch.tensor(pos_list, dtype=_torch.int64)

    # compressed K（多层，含输入+输出）
    for csa_idx, k_tensor in compressed_k_dict.items():
        tensors[f"compk_layer_{csa_idx}"] = k_tensor
        compk_shapes[f"layer_{csa_idx}"] = list(k_tensor.shape)

    # 前 3 个 decode token 在 target layers 的完整 logits
    logits_first3_dict = state.get("logits_first3", {})
    logits_shapes = {}
    for csa_idx, logits_list in logits_first3_dict.items():
        if logits_list:
            for ti, lg in enumerate(logits_list):
                tensors[f"logits_layer_{csa_idx}_token_{ti}"] = lg
            logits_shapes[f"layer_{csa_idx}"] = f"{len(logits_list)} tokens, shape={list(logits_list[0].shape)}"

    # 前 3 个 decode token 的 q（含 RoPE+Hadamard）和 fused weights（用于验证）
    q_first3_dict = state.get("q_first3", {})
    for csa_idx, q_list in q_first3_dict.items():
        if q_list:
            for ti, q_t in enumerate(q_list):
                tensors[f"q_layer_{csa_idx}_token_{ti}"] = q_t
    w_first3_dict = state.get("w_first3", {})
    for csa_idx, w_list in w_first3_dict.items():
        if w_list:
            for ti, w_t in enumerate(w_list):
                tensors[f"weights_layer_{csa_idx}_token_{ti}"] = w_t

    # 构造 CSR 格式的 labels
    label_indices = []
    label_scores = []
    label_pointers = [0]
    for token_labels in labels:
        for block_id, score in token_labels:
            label_indices.append(block_id)
            label_scores.append(score)
        label_pointers.append(len(label_indices))

    tensors["label_indices"] = _torch.tensor(label_indices, dtype=_torch.int32)
    tensors["label_scores"] = _torch.tensor(label_scores, dtype=_torch.int32)
    tensors["label_pointers"] = _torch.tensor(label_pointers, dtype=_torch.int32)

    # 存为 pickle 文件（支持 tensor + 字符串混合存储）
    import pickle as _pickle
    doc_filename = f"doc_{doc_idx:05d}.pkl"
    doc_path = _os.path.join(out_dir, doc_filename)
    with open(doc_path, "wb") as f:
        _pickle.dump(tensors, f)

    # 写 metadata 行到 jsonl（append 模式）
    meta_path = _os.path.join(out_dir, "server_metadata.jsonl")
    meta_entry = {
        "doc_index": doc_idx,
        "rid": rid,
        "filename": doc_filename,
        "n_prompt_blocks": first_n_blocks,
        "n_decode_tokens": token_id,
        "n_total_labels": len(label_indices),
        "hidden_shapes": hidden_shapes,
        "compk_shapes": compk_shapes,
        "logits_shapes": logits_shapes,
        "target_csa_layers": _dump_training_target_csa_indices,
    }
    try:
        with open(meta_path, "a") as f:
            f.write(_json.dumps(meta_entry) + "\n")
    except Exception:
        pass


def _flush_all_training_states():
    """flush 所有剩余的 per-request 状态（mode=none 时调用）。"""
    global _dump_training_states
    for rid, state in list(_dump_training_states.items()):
        _flush_one_training_state(rid, state)
    _dump_training_states = {}


def _flush_training_data():
    """
    兼容旧串行模式：将全局变量中的数据 flush。
    新并行模式下不再使用此函数。
    """
    import torch as _torch
    out_dir = _dump_training_output_dir

    # 存 hidden state
    hidden_path = _os.path.join(out_dir, "dsv4_dump_hidden.pt")
    if _dump_training_hidden_state is not None:
        _torch.save(_dump_training_hidden_state, hidden_path)

    # 存 compressed K
    compk_path = _os.path.join(out_dir, "dsv4_dump_compk.pt")
    if _dump_training_compressed_k is not None:
        _torch.save(_dump_training_compressed_k, compk_path)

    # 构造 CSR 格式的 labels
    label_indices = []
    label_scores = []
    label_pointers = [0]
    for token_labels in _dump_training_labels:
        for block_id, score in token_labels:
            label_indices.append(block_id)
            label_scores.append(score)
        label_pointers.append(len(label_indices))

    labels_path = _os.path.join(out_dir, "dsv4_dump_labels.pt")
    _torch.save({
        "label_indices": _torch.tensor(label_indices, dtype=_torch.int32),
        "label_scores": _torch.tensor(label_scores, dtype=_torch.int32),
        "label_pointers": _torch.tensor(label_pointers, dtype=_torch.int32),
    }, labels_path)

    # 写 JSON result 通知客户端
    result = {
        "mode": "dump_training",
        "hidden_path": hidden_path,
        "compk_path": compk_path,
        "labels_path": labels_path,
        "n_prompt_blocks": _first_n_blocks,
        "n_decode_tokens": _dump_training_token_id,
        "n_total_labels": len(label_indices),
        "hidden_shape": list(_dump_training_hidden_state.shape) if _dump_training_hidden_state is not None else [],
        "compk_shape": list(_dump_training_compressed_k.shape) if _dump_training_compressed_k is not None else [],
    }
    try:
        with open(_TRACKER_RESULT_FILE, "w") as f:
            _json.dump(result, f)
    except Exception:
        pass


def _round_up_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 0:
        return value
    return ((value + multiple - 1) // multiple) * multiple


def _get_deepseek_v4_moe_padding_metadata(
    config: DeepSeekV4Config,
    quant_config: Optional["QuantizationConfig"],
    tp_size: int,
) -> Tuple[int, int, Optional[Tuple[int, int]]]:
    moe_intermediate_size = getattr(config, "moe_intermediate_size", 0)
    weight_block_size = getattr(quant_config, "weight_block_size", None)
    if (
        moe_intermediate_size <= 0
        or tp_size <= 1
        or not weight_block_size
        or len(weight_block_size) != 2
        or moe_intermediate_size % tp_size != 0
    ):
        return moe_intermediate_size, moe_intermediate_size, None

    block_n, block_k = int(weight_block_size[0]), int(weight_block_size[1])
    per_partition = moe_intermediate_size // tp_size
    per_partition_alignment = math.lcm(block_n, block_k)
    padded_per_partition = _round_up_to_multiple(
        per_partition, per_partition_alignment
    )
    padded_moe_intermediate_size = padded_per_partition * tp_size
    return (
        moe_intermediate_size,
        padded_moe_intermediate_size,
        (block_n, block_k),
    )


def _pad_deepseek_v4_checkpoint_tensor_tail(
    tensor: torch.Tensor,
    dim: int,
    padded_size: int,
) -> torch.Tensor:
    if tensor.shape[dim] >= padded_size:
        return tensor

    padded_shape = list(tensor.shape)
    padded_shape[dim] = padded_size
    padded_tensor = tensor.new_zeros(padded_shape)
    target_slices = [slice(None)] * tensor.ndim
    target_slices[dim] = slice(0, tensor.shape[dim])
    padded_tensor[tuple(target_slices)].copy_(tensor)
    return padded_tensor


def _find_last_matching_dim(
    tensor: torch.Tensor,
    expected_size: int,
) -> Optional[int]:
    for dim in range(tensor.ndim - 1, -1, -1):
        if tensor.shape[dim] == expected_size:
            return dim
    return None


def _maybe_pad_deepseek_v4_moe_checkpoint_tensor(
    weight_name: str,
    loaded_weight: torch.Tensor,
    *,
    moe_intermediate_size: int,
    padded_moe_intermediate_size: int,
    weight_block_size: Optional[Tuple[int, int]],
    num_shared_experts: Optional[int],
) -> torch.Tensor:
    if weight_block_size is None or padded_moe_intermediate_size == moe_intermediate_size:
        return loaded_weight

    if ".mlp.experts." in weight_name:
        target_intermediate_size = moe_intermediate_size
        target_padded_intermediate_size = padded_moe_intermediate_size
    elif num_shared_experts is not None and ".mlp.shared_experts." in weight_name:
        target_intermediate_size = moe_intermediate_size * num_shared_experts
        target_padded_intermediate_size = padded_moe_intermediate_size * num_shared_experts
    else:
        return loaded_weight

    block_n, block_k = weight_block_size
    out_blocks = (target_intermediate_size + block_n - 1) // block_n
    padded_out_blocks = (target_padded_intermediate_size + block_n - 1) // block_n
    in_blocks = (target_intermediate_size + block_k - 1) // block_k
    padded_in_blocks = (target_padded_intermediate_size + block_k - 1) // block_k

    if (
        ".gate_proj.weight_scale_inv" in weight_name
        or ".up_proj.weight_scale_inv" in weight_name
        or ".gate_up_proj.weight_scale_inv" in weight_name
    ):
        scale_dim = _find_last_matching_dim(loaded_weight, out_blocks)
        if scale_dim is not None:
            return _pad_deepseek_v4_checkpoint_tensor_tail(
                loaded_weight,
                scale_dim,
                padded_out_blocks,
            )
        return loaded_weight

    if ".down_proj.weight_scale_inv" in weight_name:
        scale_dim = _find_last_matching_dim(loaded_weight, in_blocks)
        if scale_dim is not None:
            return _pad_deepseek_v4_checkpoint_tensor_tail(
                loaded_weight,
                scale_dim,
                padded_in_blocks,
            )
        return loaded_weight

    if (
        ".gate_proj.weight" in weight_name
        or ".up_proj.weight" in weight_name
        or ".down_proj.weight" in weight_name
        or ".gate_up_proj.weight" in weight_name
    ):
        weight_dim = _find_last_matching_dim(loaded_weight, target_intermediate_size)
        if weight_dim is not None:
            return _pad_deepseek_v4_checkpoint_tensor_tail(
                loaded_weight,
                weight_dim,
                target_padded_intermediate_size,
            )
        return loaded_weight

    return loaded_weight


MOE_BIT_WISE_EQUAL_MODE = False
ATTN_BIT_WISE_EQUAL_MODE = False
COMPRESSOR_BIT_WISE_EQUAL_MODE = False
_FP8_WO_A_GEMM = envs.SGLANG_OPT_FP8_WO_A_GEMM.get()


if TYPE_CHECKING:
    from sglang.srt.layers.attention.deepseek_v4_backend_radix import (
        DeepseekV4BackendRadix,
    )
    from sglang.srt.layers.quantization import QuantizationConfig
    from sglang.srt.layers.rotary_embedding import RotaryEmbedding
    from sglang.srt.model_executor.forward_batch_info import (
        ForwardBatch,
        PPProxyTensors,
    )


class DeepseekRefRMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        out = rms_normalize_triton(x, self.eps, self.weight)
        return out


@maybe_torch_compile
def rms_normalize(x: torch.Tensor, eps: float) -> torch.Tensor:
    x *= torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    return x


@triton.jit
def _rms_normalize_kernel(
    x_ptr,
    weight_ptr,
    eps,
    stride_row,
    dim,
    BLOCK_SIZE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
):
    pid = tl.program_id(0)

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < dim

    base = pid * stride_row
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)

    mean_sq = tl.sum(x * x, axis=0) / dim
    rms_inv = tl.rsqrt(mean_sq + eps)
    out = x * rms_inv

    if HAS_WEIGHT:
        weight = tl.load(weight_ptr + offs, mask=mask, other=0.0)
        out = out * weight

    tl.store(x_ptr + base + offs, out, mask=mask)


def rms_normalize_triton(
    x: torch.Tensor, eps: float, weight: torch.Tensor = None
) -> torch.Tensor:
    dim = x.shape[-1]
    x_flat = x.view(-1, dim)
    num_rows = x_flat.shape[0]

    BLOCK_SIZE = triton.next_power_of_2(dim)
    grid = (num_rows,)

    _rms_normalize_kernel[grid](
        x_flat,
        weight,
        eps,
        x_flat.stride(0),
        dim,
        BLOCK_SIZE=BLOCK_SIZE,
        HAS_WEIGHT=(weight is not None),
    )
    return x


class Compressor(nn.Module):
    def __init__(
        self,
        config: DeepSeekV4Config,
        layer_id: int,
        is_in_indexer: bool,
        rotary_emb: RotaryEmbedding,
        freqs_cis: torch.Tensor,
        compress_ratio: Literal[0, 4, 128],
        head_dim: int,
        rotate: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.is_in_indexer = is_in_indexer
        self.dim = config.hidden_size
        self.head_dim = head_dim
        self.rope_head_dim = getattr(config, "qk_rope_head_dim", 64)
        self.nope_head_dim = head_dim - self.rope_head_dim
        assert compress_ratio != 0, "compress_ratio should not be 0"
        self.ratio = compress_ratio
        self.overlap = self.ratio == 4
        self.rotate = rotate
        self.coff = coff = 1 + self.overlap

        self.ape = nn.Parameter(
            torch.empty(self.ratio, coff * self.head_dim, dtype=torch.float32)
        )
        wkv_gate_dtype = torch.bfloat16
        self.wkv_gate = ReplicatedLinear(
            self.dim,
            2 * coff * self.head_dim,
            bias=False,
            quant_config=None,
            prefix=add_prefix("wkv_gate", prefix),
            params_dtype=wkv_gate_dtype,
        )
        self.norm = DeepseekRefRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = rotary_emb
        self.freqs_cis = freqs_cis

        self.ape_converted = False

    def apply_ape_hotfix(self):
        assert not self.ape_converted
        self.ape_converted = True

        is_model_2604 = envs.SGLANG_DSV4_MODE.get() == "2604"
        if self.overlap and (envs.SGLANG_OPT_FIX_APE_2604.get() or not is_model_2604):
            orders = [0, 1] if is_model_2604 else [1, 0]
            ape = torch.chunk(self.ape.data, 2, dim=-1)
            ape = torch.cat([ape[orders[0]], ape[orders[1]]], dim=0)
            self.ape.data.copy_(ape.view(self.ratio, -1))

    def _get_state_pool(self, forward_batch: ForwardBatch) -> CompressStatePool:
        token_to_kv_pool = forward_batch.token_to_kv_pool
        assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)
        if self.is_in_indexer:
            ret = token_to_kv_pool.get_indexer_compress_states(self.layer_id)
        else:
            ret = token_to_kv_pool.get_attention_compress_states(self.layer_id)

        assert isinstance(ret, CompressStatePool)

        return ret

    def overlap_transform(self, tensor: torch.Tensor, fill_value: Any) -> torch.Tensor:
        assert tensor.dim() == 3
        assert tensor.shape[1:] == (self.ratio, 2 * self.head_dim)

        s, r, d = tensor.size(0), self.ratio, self.head_dim
        new_tensor = tensor.new_full((s, 2 * r, d), fill_value)
        new_tensor[:, r:] = tensor[:, :, d:]
        new_tensor[1:, :r] = tensor[:-1, :, :d]
        return new_tensor

    def overlap_transform_decode(self, tensor: torch.Tensor) -> torch.Tensor:
        assert tensor.dim() == 3
        assert tensor.shape[1:] == (2 * self.ratio, 2 * self.head_dim)
        r, d = self.ratio, self.head_dim
        ret = torch.cat((tensor[:, :r, :d], tensor[:, r:, d:]), dim=1)
        return ret

    @staticmethod
    def compute_state_len(seq_len: int, ratio: int):
        return seq_len % ratio + (ratio == 4) * ratio

    @staticmethod
    def compute_state_len_indices(seq_len: int, ratio: int):
        state_len = seq_len % ratio + (ratio == 4) * ratio
        return torch.arange(seq_len - state_len, seq_len).clamp(min=-1)

    def compress_fused(
        self,
        kv_score: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        backend = forward_batch.attn_backend
        if TYPE_CHECKING:
            assert isinstance(backend, DeepseekV4BackendRadix)
        kv_score_buffer = self._get_state_pool(forward_batch)
        kv_score_buffer = kv_score_buffer.kv_score_buffer.kv_score
        return backend.forward_compress(
            kv_score_buffer=kv_score_buffer,
            kv_score_input=kv_score,
            ape=self.ape.view(-1, self.head_dim),
            head_dim=self.head_dim,
            norm=self.norm,
            freqs_cis_cache=self.freqs_cis,
            rotate=self.rotate,
            compress_ratio=self.ratio,
            forward_batch=forward_batch,
            is_paged=True,
        )

    def forward(self, x: torch.Tensor, forward_batch: ForwardBatch) -> torch.Tensor:
        if forward_batch.forward_mode.is_idle():
            assert x.shape[0] == 0
            return x.new_empty(0, self.head_dim)

        self.forward_mode = forward_batch.forward_mode

        kv_score = linear_bf16_fp32(x, self.wkv_gate.weight)
        if nsa_use_prefill_cp(forward_batch):
            kv_score = cp_all_gather_rerange_output(
                kv_score,
                get_attention_tp_size(),
                forward_batch,
                torch.cuda.current_stream(),
            )
        return self.compress_fused(kv_score, forward_batch)


class C4Indexer(nn.Module):
    def __init__(
        self,
        config: DeepSeekV4Config,
        layer_id: int,
        rotary_emb: RotaryEmbedding,
        freqs_cis: torch.Tensor,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_streams: Optional[List[torch.cuda.Stream]] = None,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.dim = config.hidden_size
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.index_topk = config.index_topk
        self.q_lora_rank = config.q_lora_rank
        self.softmax_scale = self.head_dim**-0.5
        self.n_local_heads = self.n_heads
        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            params_dtype=torch.bfloat16,
            prefix=add_prefix("wq_b", prefix),
        )
        self.weights_proj = ReplicatedLinear(
            self.dim,
            self.n_heads,
            bias=False,
            quant_config=None,
            params_dtype=torch.bfloat16,
            prefix=add_prefix("weights_proj", prefix),
        )
        self.compressor = Compressor(
            config,
            self.layer_id,
            True,
            rotary_emb,
            freqs_cis,
            compress_ratio=4,
            head_dim=self.head_dim,
            rotate=True,
            prefix=add_prefix("compressor", prefix),
        )
        self.rotary_emb = rotary_emb
        self.freqs_cis = freqs_cis
        self.weight_scale: float = self.softmax_scale * self.n_heads**-0.5
        self.alt_streams = alt_streams
        # score_hook 由 _make_score_hook() 在每次 forward 前构建。
        # 签名：(logits, seq_lens, forward_batch) -> logits
        # 平时为 None（无实验），零开销；实验时由 _make_score_hook() 赋值。
        self.score_hook = None

    def compute_q(self, q_lora: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        q, _ = self.wq_b(q_lora)
        q = q.view(-1, self.n_local_heads, self.head_dim)
        fused_rope(
            q[..., -self.rope_head_dim :],
            None,
            self.freqs_cis,
            positions=positions,
        )
        q = rotate_activation(q)
        return q

    def compute_weights(self, x: torch.Tensor, skip_scale=False) -> torch.Tensor:
        out, _ = self.weights_proj(x)
        if not skip_scale:
            out = out * self.weight_scale
        return out


    def _make_score_hook(self):
        """
        【新增方法】每次 C4Indexer.forward() 调用时构建 hook 闭包。

        hook 的签名：(logits, seq_lens, forward_batch) -> logits
        hook 被 compressed_indexer.py 在 logits 算完、topk 之前调用。

        【返回值】
          - None：没有激活实验（mode=none），后端跳过 hook，零开销
          - hook 函数：Pass-1 记录 / Pass-2 mask

        【hook 内部逻辑】
          1. prefill 阶段直接 return logits（不干预）
          2. Pass-2：把非 recalled_blocks 的 block logits 设为 -inf（屏蔽）
             answer 部分（编号 >= n_prompt_blocks）自动跳过 mask
          3. Pass-1：对 logits 做 topk，所有 21 个 CSA layer 都记录，
             但最终只有被 ≥ _MIN_LAYER_HITS 个 layer 选中的 chunk 才计入 score
        """
        _maybe_init()   # 只在 cmd 文件更新时才重新读取，同一请求内多次调用无额外开销

        if not _pass1_active and _pass2_mask is None and not _logits_dump_active and not _pass1_topp_active and not _pass1_topp_nlayer_active and not _dump_training_active:
            return None   # 没有激活实验，返回 None，后端不会调用 hook

        # ---- 动态 mask：按 decode token 计数切换 interval ----
        global _pass2_dynamic_layer_seen, _pass2_dynamic_token_count, _pass2_dynamic_idx
        if _pass2_dynamic_masks is not None:
            _pass2_dynamic_layer_seen += 1
            if _pass2_dynamic_layer_seen >= _total_csa_layers:
                _pass2_dynamic_layer_seen = 0
                _pass2_dynamic_token_count += 1
                # 检查是否该切换到下一个 interval 的 mask
                if _pass2_dynamic_token_count >= _pass2_dynamic_interval:
                    _pass2_dynamic_token_count = 0
                    _pass2_dynamic_idx += 1
                    if _pass2_dynamic_idx < len(_pass2_dynamic_masks):
                        # 通过模块级 globals() 更新 _pass2_mask
                        globals()["_pass2_mask"] = _pass2_dynamic_masks[_pass2_dynamic_idx]

        # 把当前全局状态快照进闭包，避免请求中途状态被其他请求修改
        do_track = _pass1_active
        mask_set = _pass2_mask
        n_prompt_blocks = _pass2_n_prompt_blocks
        topk     = self.index_topk
        indexer_self = self  # 供 dump_training hook 访问 self._last_x 和 self._last_kv_cache
        is_rank_0 = (get_attention_tp_rank() == 0)  # 只在 rank 0 做 dump_training 记录
        # 预先将 mask_set 转为 GPU tensor，避免每次 hook 调用都做 Python for 循环
        mask_indices_tensor = None
        if mask_set is not None and len(mask_set) > 0:
            mask_indices_tensor = torch.tensor(sorted(mask_set), dtype=torch.long, device="cuda")

        def hook(logits: torch.Tensor, seq_lens: torch.Tensor, forward_batch) -> torch.Tensor:
            # [DEBUG] 把 forward_batch.rids 写到文件（避免 print 被缓冲看不到）
            if not getattr(indexer_self, '_debug_rids_printed', False):
                indexer_self._debug_rids_printed = True
                rids = getattr(forward_batch, 'rids', None)
                kv_cache = getattr(indexer_self, '_last_kv_cache', None)
                kv_dtype = str(kv_cache.dtype) if kv_cache is not None else "None"
                kv_shape = str(kv_cache.shape) if kv_cache is not None else "None"
                with open("/tmp/debug_rids.txt", "w") as _df:
                    _df.write(f"forward_batch.rids={rids}\nkv_cache dtype={kv_dtype}\nkv_cache shape={kv_shape}\n")

            if not forward_batch.forward_mode.is_decode():
                # Prefill 阶段：只做 layer_seen 计数
                # compressed K 改在 decode 第一步获取（prefill 的 seq_lens 可能被 chunk 截断）
                if _dump_training_active and is_rank_0:
                    rids = getattr(forward_batch, 'rids', None)
                    if rids and len(rids) > 0:
                        for bi in range(len(rids)):
                            rid = rids[bi]
                            if rid not in _dump_training_states:
                                # 创建时固定 doc_idx 和 output_dir，防止后续 cmd 切换导致写错
                                _pf_doc_idx = _dump_training_doc_counter
                                globals()["_dump_training_doc_counter"] = _dump_training_doc_counter + 1
                                _dump_training_states[rid] = {
                                    "hidden_states": {},
                                    "compressed_k": {},
                                    "logits_first3": {},
                                    "positions": {},
                                    "labels": [], "layer_seen": 0, "token_id": 0,
                                    "per_token_counts": {}, "first_n_blocks": 0,
                                    "_assigned_doc_idx": _pf_doc_idx,
                                    "_assigned_output_dir": _dump_training_output_dir,
                                }
                            state = _dump_training_states[rid]
                            state["layer_seen"] += 1
                            if state["layer_seen"] >= _total_csa_layers:
                                state["layer_seen"] = 0
                return logits

            n_blocks = int(seq_lens[0].item())

            # ---- Pass-2：将未召回的 block 分数设为 -inf ----
            if mask_set is not None:
                block_mask = logits.new_ones(logits.shape[1], dtype=torch.bool)
                # 用 tensor 索引一次性解除 mask（避免 Python for 循环几千次）
                if mask_indices_tensor is not None:
                    valid = mask_indices_tensor[mask_indices_tensor < n_blocks]
                    block_mask[valid] = False
                # answer 部分的 block（编号 >= n_prompt_blocks）自动跳过 mask，
                # 让模型始终能回看自己刚生成的内容
                if n_prompt_blocks > 0:
                    block_mask[n_prompt_blocks:n_blocks] = False

                logits = logits.masked_fill(block_mask.unsqueeze(0), float("-inf"))

            # ---- Pass-1：记录所有 block 得分 + topk 召回 ----
            if do_track:
                topk_k = min(topk, n_blocks)
                if topk_k > 0:
                    raw_topk = logits[0, :n_blocks].topk(topk_k, dim=-1)[1]
                    _record_blocks_and_flush(raw_topk, n_blocks, compress_ratio=4)

            # ---- Pass-1 Top-P：对 topk logits 算 softmax，取 top-p chunk 计分 ----
            if _pass1_topp_active:
                global _pass1_topp_layer_seen, _pass1_topp_token_id
                topk_k = min(topk, n_blocks)
                if topk_k > 0:
                    values, indices = logits[0, :n_blocks].topk(topk_k, dim=-1)
                    # softmax on topk values
                    import torch.nn.functional as _F
                    probs = _F.softmax(values, dim=-1)
                    # 按概率从高到低累积，找到 top-p 的截断点
                    cumsum = torch.cumsum(probs, dim=-1)
                    # 找到第一个 cumsum >= threshold 的位置
                    mask_topp = cumsum <= _pass1_topp_threshold
                    # 至少保留第一个（最高概率的）
                    mask_topp[0] = True
                    # 再加上刚好超过阈值的那一个
                    first_over = (~mask_topp).nonzero(as_tuple=True)[0]
                    if len(first_over) > 0:
                        mask_topp[first_over[0]] = True

                    # 取 top-p 内的 chunk indices
                    selected_indices = indices[mask_topp].tolist()

                    # 只对 prompt chunk 计分
                    prompt_limit = _first_n_blocks if _first_n_blocks > 0 else n_blocks
                    for b in selected_indices:
                        if b < 0 or b >= prompt_limit:
                            continue
                        _block_scores[b] = _block_scores.get(b, 0) + 1
                        if b not in _recalled_blocks:
                            _recalled_blocks.add(b)
                            for t in range(b * 4, (b + 1) * 4):
                                _recalled_tokens.add(t)

                # token 计数 + 快照
                _pass1_topp_layer_seen += 1
                if _pass1_topp_layer_seen >= _total_csa_layers:
                    _pass1_topp_layer_seen = 0
                    _pass1_topp_token_id += 1
                    globals()["_csa_layer_count"] = _csa_layer_count + _total_csa_layers
                    globals()["_last_n_blocks"] = n_blocks
                    if _first_n_blocks == 0:
                        globals()["_first_n_blocks"] = n_blocks

                    # 快照
                    if _snapshot_interval > 0:
                        globals()["_snapshot_token_count"] = _snapshot_token_count + 1
                        if globals()["_snapshot_token_count"] >= _snapshot_interval:
                            interval_scores = {}
                            for b, total_score in _block_scores.items():
                                prev_score = _snapshot_prev_scores.get(b, 0)
                                delta = total_score - prev_score
                                if delta > 0:
                                    interval_scores[b] = delta
                            interval_ranked = sorted(interval_scores.items(), key=lambda x: x[1], reverse=True)
                            _snapshots.append({
                                "decode_token": _snapshot_token_count,
                                "n_blocks": n_blocks,
                                "recalled_blocks_with_scores": interval_ranked,
                            })
                            globals()["_snapshot_prev_scores"] = dict(_block_scores)
                            globals()["_snapshot_token_count"] = 0

            # ---- Pass-1 Top-P + 多层联合过滤 ----
            # 每个 layer 算 top-p chunk，累计到 per-token 计数器；
            # 一个 token 的所有 layer 都过完后，只有通过 ≥ min_layers 个 layer 的 chunk 才 +1 分
            if _pass1_topp_nlayer_active:
                topk_k = min(topk, n_blocks)
                if topk_k > 0:
                    values, indices = logits[0, :n_blocks].topk(topk_k, dim=-1)
                    import torch.nn.functional as _F
                    probs = _F.softmax(values, dim=-1)
                    cumsum = torch.cumsum(probs, dim=-1)
                    mask_topp = cumsum <= _pass1_topp_nlayer_threshold
                    mask_topp[0] = True
                    first_over = (~mask_topp).nonzero(as_tuple=True)[0]
                    if len(first_over) > 0:
                        mask_topp[first_over[0]] = True

                    selected_indices = indices[mask_topp].tolist()

                    # 记录到 per-token 计数器（只记录 prompt chunk）
                    prompt_limit = _first_n_blocks if _first_n_blocks > 0 else n_blocks
                    for b in selected_indices:
                        if b < 0 or b >= prompt_limit:
                            continue
                        _pass1_topp_nlayer_counts[b] = _pass1_topp_nlayer_counts.get(b, 0) + 1

                # token 计数
                globals()["_pass1_topp_nlayer_layer_seen"] = _pass1_topp_nlayer_layer_seen + 1
                if globals()["_pass1_topp_nlayer_layer_seen"] >= _total_csa_layers:
                    globals()["_pass1_topp_nlayer_layer_seen"] = 0
                    globals()["_pass1_topp_nlayer_token_id"] = _pass1_topp_nlayer_token_id + 1
                    globals()["_csa_layer_count"] = _csa_layer_count + _total_csa_layers
                    globals()["_last_n_blocks"] = n_blocks
                    if _first_n_blocks == 0:
                        globals()["_first_n_blocks"] = n_blocks

                    # 一个 token 完成：对 per-token 计数器做阈值过滤
                    # 满足 ≥ min_layers 的 chunk，加的分 = 通过 top-p 的 layer 数
                    for b, layer_hits in _pass1_topp_nlayer_counts.items():
                        if layer_hits >= _pass1_topp_nlayer_min_layers:
                            _block_scores[b] = _block_scores.get(b, 0) + layer_hits
                            if b not in _recalled_blocks:
                                _recalled_blocks.add(b)
                                for t in range(b * 4, (b + 1) * 4):
                                    _recalled_tokens.add(t)
                    # 重置 per-token 计数器
                    globals()["_pass1_topp_nlayer_counts"] = {}

                    # 快照
                    if _snapshot_interval > 0:
                        globals()["_snapshot_token_count"] = _snapshot_token_count + 1
                        if globals()["_snapshot_token_count"] >= _snapshot_interval:
                            interval_scores = {}
                            for b, total_score in _block_scores.items():
                                prev_score = _snapshot_prev_scores.get(b, 0)
                                delta = total_score - prev_score
                                if delta > 0:
                                    interval_scores[b] = delta
                            interval_ranked = sorted(interval_scores.items(), key=lambda x: x[1], reverse=True)
                            _snapshots.append({
                                "decode_token": _snapshot_token_count,
                                "n_blocks": n_blocks,
                                "recalled_blocks_with_scores": interval_ranked,
                            })
                            globals()["_snapshot_prev_scores"] = dict(_block_scores)
                            globals()["_snapshot_token_count"] = 0

            # ---- Dump Training Data：per-request 并行版 ----
            if _dump_training_active and is_rank_0:
                import torch as _torch

                rids = getattr(forward_batch, 'rids', None)
                if rids and len(rids) > 0:
                    batch_size = len(rids)
                    for bi in range(batch_size):
                        rid = rids[bi]
                        n_blk = int(seq_lens[bi].item()) if bi < len(seq_lens) else n_blocks

                        # 获取或创建 per-request 状态
                        if rid not in _dump_training_states:
                            # 创建时固定 doc_idx 和 output_dir，防止后续 cmd 切换导致写错
                            _dec_doc_idx = _dump_training_doc_counter
                            globals()["_dump_training_doc_counter"] = _dump_training_doc_counter + 1
                            _dump_training_states[rid] = {
                                "hidden_states": {},
                                "compressed_k": {},
                                "logits_first3": {},   # {csa_idx: [logits_t0, logits_t1, logits_t2]}
                                "positions": {},       # {csa_idx: [pos_t0, pos_t1, ...]}
                                "labels": [], "layer_seen": 0, "token_id": 0,
                                "per_token_counts": {}, "first_n_blocks": 0,
                                "_assigned_doc_idx": _dec_doc_idx,
                                "_assigned_output_dir": _dump_training_output_dir,
                            }
                        state = _dump_training_states[rid]
                        cur_layer = state["layer_seen"]  # 当前是第几个 CSA layer (0-indexed)

                        topk_k = min(topk, n_blk)

                        # 在 target CSA layers 记录 decode hidden state + position
                        if cur_layer in _dump_training_target_csa_indices:
                            x_data = getattr(indexer_self, '_last_x', None)
                            if x_data is not None:
                                if x_data.dim() == 2 and bi < x_data.shape[0]:
                                    h = x_data[bi:bi+1].detach().cpu().to(_torch.bfloat16)
                                elif x_data.dim() == 1:
                                    h = x_data.unsqueeze(0).detach().cpu().to(_torch.bfloat16)
                                else:
                                    h = None
                                if h is not None:
                                    if cur_layer not in state["hidden_states"]:
                                        state["hidden_states"][cur_layer] = []
                                    state["hidden_states"][cur_layer].append(h)
                            # 记录 position（decode token 在序列中的绝对位置）
                            pos_data = getattr(indexer_self, '_last_positions', None)
                            if pos_data is not None:
                                pos_val = int(pos_data[bi].item()) if bi < pos_data.shape[0] else 0
                                if cur_layer not in state["positions"]:
                                    state["positions"][cur_layer] = []
                                state["positions"][cur_layer].append(pos_val)

                        # 第一步 decode 的 target layers：记录 compressed K（此时 seq_lens 是完整 prompt block 数）
                        if state["token_id"] == 0 and cur_layer in _dump_training_target_csa_indices:
                            if cur_layer not in state["compressed_k"]:
                                kv_cache = getattr(indexer_self, '_last_kv_cache', None)
                                if kv_cache is not None:
                                    page_table = getattr(indexer_self, '_last_page_table', None)
                                    seq_lens_t = getattr(indexer_self, '_last_seq_lens', None)
                                    if page_table is not None and seq_lens_t is not None:
                                        n_prompt_blks = int(seq_lens_t[bi].item()) if bi < seq_lens_t.shape[0] else n_blk
                                        # kv_cache shape: [n_pages, 64, 1, 132] (view 后，内存布局是 page 级别)
                                        # 原始内存布局（per page）: 前 page_size*128 bytes = FP8 keys, 后 page_size*4 bytes = float32 scales
                                        # 需要按正确布局提取，而不是直接用 view 后的 [64, 1, 132]
                                        page_size = kv_cache.shape[1]  # = 64
                                        head_dim = 128
                                        n_pages_needed = (n_prompt_blks + page_size - 1) // page_size
                                        pages = page_table[bi, :n_pages_needed].tolist()
                                        # 展平回原始 page 级别布局 [n_pages, page_size*(head_dim+4)]
                                        kv_flat = kv_cache.view(kv_cache.shape[0], page_size * (head_dim + 4))
                                        # 提取所需 pages
                                        pages_data = kv_flat[pages]  # [n_pages_needed, 8448]
                                        # 分离 FP8 key 和 scale
                                        SCALE_OFFSET = page_size * head_dim  # = 8192
                                        k_fp8_pages = pages_data[:, :SCALE_OFFSET]  # [n_pages, 8192] uint8
                                        k_scale_pages = pages_data[:, SCALE_OFFSET:]  # [n_pages, 256] uint8
                                        # reshape 为 per-block 格式
                                        k_fp8 = k_fp8_pages.reshape(-1, head_dim)[:n_prompt_blks]  # [n_blocks, 128]
                                        k_scale = k_scale_pages.reshape(-1, 4)[:n_prompt_blks]  # [n_blocks, 4]
                                        # 拼接为 [n_blocks, 132] = [128 FP8 | 4 scale]
                                        full_k = _torch.cat([k_fp8, k_scale], dim=1)  # [n_blocks, 132]
                                        state["compressed_k"][cur_layer] = full_k.detach().cpu()

                        # 前 3 个 decode token 的 target CSA layers：记录完整 logits + q + weights
                        if state["token_id"] < 3 and cur_layer in _dump_training_target_csa_indices:
                            token_logits = logits[bi, :n_blk].detach().cpu().to(_torch.bfloat16)  # [n_blk]
                            if cur_layer not in state["logits_first3"]:
                                state["logits_first3"][cur_layer] = []
                            state["logits_first3"][cur_layer].append(token_logits)
                            # 同时存 q（含RoPE+Hadamard的full-precision）和 fused weights，用于离线验证
                            q_data = getattr(indexer_self, '_last_q', None)
                            if q_data is not None and bi < q_data.shape[0]:
                                # q_data shape: [batch, n_heads, head_dim]，取第 bi 个并展平为 [1, n_heads*head_dim]
                                q_token = q_data[bi].reshape(1, -1).detach().cpu().to(_torch.bfloat16)
                                if "q_first3" not in state:
                                    state["q_first3"] = {}
                                if cur_layer not in state["q_first3"]:
                                    state["q_first3"][cur_layer] = []
                                state["q_first3"][cur_layer].append(q_token)
                            w_data = getattr(indexer_self, '_last_weights', None)
                            if w_data is not None and bi < w_data.shape[0]:
                                w_token = w_data[bi:bi+1].detach().cpu().to(_torch.bfloat16)  # [1, n_heads]
                                if "w_first3" not in state:
                                    state["w_first3"] = {}
                                if cur_layer not in state["w_first3"]:
                                    state["w_first3"][cur_layer] = []
                                state["w_first3"][cur_layer].append(w_token)

                        # 所有 21 层都做 topp 过滤
                        if topk_k > 0:
                            values, indices = logits[bi, :n_blk].topk(topk_k, dim=-1)
                            import torch.nn.functional as _F
                            probs = _F.softmax(values, dim=-1)
                            cumsum = _torch.cumsum(probs, dim=-1)
                            mask_topp = cumsum <= _dump_training_topp
                            mask_topp[0] = True
                            first_over = (~mask_topp).nonzero(as_tuple=True)[0]
                            if len(first_over) > 0:
                                mask_topp[first_over[0]] = True

                            selected_indices = indices[mask_topp].tolist()

                            # 记录该 token 之前上文的所有 chunk（prompt + 已生成的 answer）
                            for b in selected_indices:
                                if b < 0 or b >= n_blk:
                                    continue
                                state["per_token_counts"][b] = state["per_token_counts"].get(b, 0) + 1

                        # layer 计数
                        state["layer_seen"] += 1
                        if state["layer_seen"] >= _total_csa_layers:
                            state["layer_seen"] = 0
                            state["token_id"] += 1
                            if state["first_n_blocks"] == 0:
                                state["first_n_blocks"] = n_blk

                            # 一个 token 完成：取 ≥ min_layers 的 chunk 作为 golden label
                            token_golden = []
                            for b, layer_hits in state["per_token_counts"].items():
                                if layer_hits >= _dump_training_min_layers:
                                    token_golden.append((b, layer_hits))
                            token_golden.sort(key=lambda x: x[1], reverse=True)
                            state["labels"].append(token_golden)
                            state["per_token_counts"] = {}

                    # 检测已结束的请求并自动 flush（延迟机制：连续 3 次不出现才 flush）
                    active_rids = set(rids)
                    for rid_check in list(_dump_training_states.keys()):
                        if rid_check not in active_rids:
                            state_check = _dump_training_states[rid_check]
                            absent_count = state_check.get("_absent_count", 0) + 1
                            state_check["_absent_count"] = absent_count
                            if absent_count >= 3:
                                _flush_one_training_state(rid_check, state_check)
                                del _dump_training_states[rid_check]
                        else:
                            # 重新出现，重置计数
                            _dump_training_states[rid_check]["_absent_count"] = 0

            # ---- Logits dump：记录 topk 的 (block_id, logit_value) ----
            if _logits_dump_active:
                global _logits_dump_token_id, _logits_dump_layer_seen
                topk_k = min(topk, n_blocks)
                if topk_k > 0:
                    values, indices = logits[0, :n_blocks].topk(topk_k, dim=-1)
                    # 输出完整 topk 的 (block_id, logit_value)，不过滤，画图时再处理
                    logits_list = list(zip(indices.tolist(), [round(v, 4) for v in values.tolist()]))
                    _logits_dump_data.append((_logits_dump_token_id, _logits_dump_layer_seen, logits_list))

                _logits_dump_layer_seen += 1
                if _logits_dump_layer_seen >= _total_csa_layers:
                    _logits_dump_layer_seen = 0
                    _logits_dump_token_id += 1
                    # 记录 _first_n_blocks（和 pass1 一样）
                    if _first_n_blocks == 0:
                        globals()["_first_n_blocks"] = n_blocks

            return logits

        return hook

    def forward(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        forward_batch: ForwardBatch,
        enable_multi_stream: bool = False,
        q_lora_ready: Optional[torch.cuda.Event] = None,
    ) -> None:
        if TYPE_CHECKING:
            assert isinstance(forward_batch.attn_backend, DeepseekV4BackendRadix)

        # 每次 forward 前重新构建 hook，确保 Pass-1→Pass-2 切换时立刻生效
        self._last_x = x  # 存下 hidden state，供 dump_training hook 使用
        self.score_hook = self._make_score_hook()

        return forward_batch.attn_backend.forward_c4_indexer(
            x=x,
            q_lora=q_lora,
            forward_batch=forward_batch,
            c4_indexer=self,   # 把自身传进去，后端通过 c4_indexer.score_hook 调用 hook
            alt_streams=self.alt_streams,
            enable_multi_stream=enable_multi_stream,
            q_lora_ready=q_lora_ready,
        )


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    import math

    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class MQALayer(nn.Module):
    def __init__(
        self,
        config: DeepSeekV4Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_streams: Optional[List[torch.cuda.Stream]] = None,
        compress_ratio_override: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.tp_rank = attn_tp_rank = get_attention_tp_rank()
        self.tp_size = attn_tp_size = get_attention_tp_size()
        self.nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()
        if self.nsa_enable_prefill_cp:
            self.cp_size = get_attention_tp_size()
            self.tp_rank = attn_tp_rank = 0
            self.tp_size = attn_tp_size = 1
        self.layer_id = layer_id
        self.dim = config.hidden_size
        self.qk_rope_head_dim = config.qk_rope_head_dim
        if envs.SGLANG_DSV4_MODE.get() == "2604":
            self.qk_nope_head_dim = config.head_dim - config.qk_rope_head_dim
        else:
            self.qk_nope_head_dim = config.qk_nope_head_dim
        self.head_dim = self.qk_rope_head_dim + self.qk_nope_head_dim
        self.n_heads = config.num_attention_heads
        self.n_local_heads = self.n_heads // attn_tp_size
        self.n_groups = config.o_groups
        self.n_local_groups = self.n_groups // attn_tp_size
        self.rope_head_dim = config.qk_rope_head_dim
        self.softmax_scale = self.head_dim**-0.5
        self.hidden_size = config.hidden_size
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.eps = config.rms_norm_eps
        compress_ratio = (
            compress_ratio_override
            if compress_ratio_override is not None
            else config.compress_ratios[layer_id]
        )
        assert compress_ratio in [0, 4, 128]
        self.compress_ratio: Literal[0, 4, 128] = compress_ratio

        if envs.SGLANG_DSV4_MODE.get() == "2604":
            assert self.head_dim == config.head_dim
        else:
            assert self.head_dim == config.v_head_dim
        assert config.num_key_value_heads == 1

        rope_scaling = config.rope_scaling
        if rope_scaling:
            rope_scaling["rope_type"] = "deepseek_yarn"

        if envs.SGLANG_DEBUG_SANITY_CHECK_CONFIG.get():
            assert (
                config.compress_rope_theta == 160000
            ), f"{config.compress_rope_theta=}"
        rope_base = (
            config.compress_rope_theta if self.compress_ratio else config.rope_theta
        )

        self.rotary_emb = get_rope_wrapper(
            head_size=self.rope_head_dim,
            rotary_dim=self.rope_head_dim,
            max_position=config.max_position_embeddings,
            base=rope_base,
            rope_scaling=rope_scaling,
            is_neox_style=False,
            device=get_global_server_args().device,
        )

        from sglang.srt.layers.deepseek_v4_rope import precompute_freqs_cis

        if envs.SGLANG_DSV4_MODE.get() == "2604":
            if envs.SGLANG_DEBUG_SANITY_CHECK_CONFIG.get():
                assert rope_scaling["factor"] == 16
        elif envs.SGLANG_DSV4_MODE.get() == "2601":
            if envs.SGLANG_DEBUG_SANITY_CHECK_CONFIG.get():
                assert rope_scaling["factor"] == 4
        else:
            raise NotImplementedError

        if envs.SGLANG_DSV4_2604_SUBMODE.get() == "2604B":
            assert self.compress_ratio in {0, 4, 128}
            if self.compress_ratio:
                original_seq_len = rope_scaling["original_max_position_embeddings"]
                if envs.SGLANG_DEBUG_SANITY_CHECK_CONFIG.get():
                    assert original_seq_len == 65536
            else:
                original_seq_len = 0
        else:
            original_seq_len = rope_scaling["original_max_position_embeddings"]

        rope_scaling = config.rope_scaling
        freqs_cis = precompute_freqs_cis(
            dim=self.qk_rope_head_dim,
            seqlen=config.max_position_embeddings,
            original_seq_len=original_seq_len,
            base=rope_base,
            factor=rope_scaling["factor"],
            beta_fast=rope_scaling["beta_fast"],
            beta_slow=rope_scaling["beta_slow"],
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
        self.freqs_cis: torch.Tensor

        if envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.get() and alt_streams is not None:
            self.alt_streams = alt_streams[:3]
            self.alt_streams_indexer = alt_streams[-2:]
        else:
            self.alt_streams = None
            self.alt_streams_indexer = None

        from sglang.srt.utils import is_blackwell_supported

        self._multi_stream_bs_limit = 128 if is_blackwell_supported() else 64

        self.compressor = None
        self.indexer = None
        if self.compress_ratio:
            self.compressor = Compressor(
                config,
                layer_id=self.layer_id,
                is_in_indexer=False,
                rotary_emb=self.rotary_emb,
                freqs_cis=freqs_cis,
                compress_ratio=self.compress_ratio,
                head_dim=self.head_dim,
                rotate=False,
                prefix=add_prefix("compressor", prefix),
            )
            if self.compress_ratio == 4:
                self.indexer = C4Indexer(
                    config,
                    rotary_emb=self.rotary_emb,
                    freqs_cis=freqs_cis,
                    layer_id=layer_id,
                    quant_config=quant_config,
                    prefix=add_prefix("indexer", prefix),
                    alt_streams=self.alt_streams_indexer,
                )

        self.attn_sink = nn.Parameter(torch.empty(self.n_heads, dtype=torch.float32))
        self.fuse_wqa_wkv = envs.SGLANG_OPT_FUSE_WQA_WKV.get()
        if self.fuse_wqa_wkv:
            self.wqkv_a = ReplicatedLinear(
                self.hidden_size,
                self.q_lora_rank + self.head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("wqkv_a", prefix),
            )
        else:
            self.wq_a = ReplicatedLinear(
                self.hidden_size,
                self.q_lora_rank,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("wq_a", prefix),
            )
            self.wkv = ReplicatedLinear(
                self.hidden_size,
                self.head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("wkv", prefix),
            )
        self.q_norm = RMSNorm(self.q_lora_rank, eps=self.eps)
        self.wq_b = ColumnParallelLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wq_b", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )
        self.kv_norm = RMSNorm(self.head_dim, eps=self.eps)
        self.wo_a = ColumnParallelLinear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * self.o_lora_rank,
            bias=False,
            quant_config=quant_config if _FP8_WO_A_GEMM else None,
            prefix=add_prefix("wo_a", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            **({} if _FP8_WO_A_GEMM else {"params_dtype": torch.bfloat16}),
        )
        if _FP8_WO_A_GEMM:
            assert hasattr(
                self.wo_a, "weight_scale_inv"
            ), "FP8 quant_config must create weight_scale_inv"
            self.wo_a.weight_scale_inv.format_ue8m0 = True
        self.wo_b = RowParallelLinear(
            self.n_groups * self.o_lora_rank,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=attn_tp_size > 1,
            prefix=add_prefix("wo_b", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )

        self.attn_mqa = RadixAttention(
            self.n_local_heads,
            self.head_dim,
            self.softmax_scale,
            num_kv_heads=1,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn_mqa", prefix),
        )

        self.overlap_store_cache = envs.SGLANG_OPT_USE_OVERLAP_STORE_CACHE.get()
        self.use_jit_norm = envs.SGLANG_OPT_USE_JIT_NORM.get()

    def _compute_q_a(
        self,
        x: torch.Tensor,
        qkv_a: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if qkv_a is not None:
            q = qkv_a[..., : self.q_lora_rank]
        else:
            q, _ = self.wq_a(x)
        q = self.q_norm(q)
        q_lora = q
        return q_lora

    def _compute_q_b(
        self,
        q: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q, _ = self.wq_b(q)
        q = q.view(-1, self.n_local_heads, self.head_dim)
        if self.use_jit_norm:
            q = rmsnorm_self(q, self.eps)
        else:
            q = rms_normalize_triton(q, self.eps)
        if positions is not None:
            fused_rope(
                q[..., -self.qk_rope_head_dim :],
                None,
                self.freqs_cis,
                positions=positions,
            )
        else:
            apply_rotary_emb_triton(q[..., -self.qk_rope_head_dim :], self.freqs_cis)
        return q

    def _compute_kv(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
        qkv_a: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if qkv_a is not None:
            kv = qkv_a[..., self.q_lora_rank :]
        else:
            kv, _ = self.wkv(x)
        kv = self.kv_norm(kv)
        if positions is not None:
            fused_rope(
                kv[..., -self.qk_rope_head_dim :].unsqueeze(1),
                None,
                self.freqs_cis,
                positions=positions,
            )
        else:
            apply_rotary_emb_triton(kv[..., -self.qk_rope_head_dim :], self.freqs_cis)
        return kv

    def _forward_prepare_multi_stream(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        attn_backend: DeepseekV4BackendRadix,
        freqs_cis: Optional[torch.Tensor] = None,
        q_out: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.alt_streams is not None
        assert len(self.alt_streams) >= 3

        current_stream = torch.cuda.current_stream()
        stream_kv = self.alt_streams[0]
        stream_compressor = self.alt_streams[1]
        stream_indexer = self.alt_streams[2]

        stream_kv.wait_stream(current_stream)
        stream_compressor.wait_stream(current_stream)
        stream_indexer.wait_stream(current_stream)

        qkv_a: Optional[torch.Tensor] = None
        qkv_a_ready: Optional[torch.cuda.Event] = None
        if self.fuse_wqa_wkv:
            qkv_a, _ = self.wqkv_a(x)
            qkv_a_ready = current_stream.record_event()

        q_lora = self._compute_q_a(x, qkv_a=qkv_a)
        q_lora_ready = current_stream.record_event()

        if self.indexer is not None:
            with torch.cuda.stream(stream_indexer):
                self.indexer(
                    x=x,
                    q_lora=q_lora,
                    forward_batch=forward_batch,
                    enable_multi_stream=True,
                    q_lora_ready=q_lora_ready,
                )

        with torch.cuda.stream(stream_kv):
            if qkv_a_ready is not None:
                stream_kv.wait_event(qkv_a_ready)
            kv = self._compute_kv(x, positions, freqs_cis, qkv_a=qkv_a)
            if self.overlap_store_cache:
                attn_backend.store_cache(
                    layer_id=self.layer_id,
                    swa_k=kv,
                    forward_batch=forward_batch,
                )

        del qkv_a

        if self.compressor is not None:
            with torch.cuda.stream(stream_compressor):
                attn_backend.forward_core_compressor(
                    x, forward_batch, self.layer_id, self.compressor
                )

        q = self._compute_q_b(q_lora, positions, freqs_cis)
        if q_out is not None:
            q_out.copy_(q)

        current_stream.wait_stream(stream_kv)
        current_stream.wait_stream(stream_compressor)
        current_stream.wait_stream(stream_indexer)

        return q, kv

    def _forward_prepare(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        attn_backend: DeepseekV4BackendRadix,
        freqs_cis: Optional[torch.Tensor] = None,
        q_out: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.fuse_wqa_wkv:
            qkv_a, _ = self.wqkv_a(x)
            q = qkv_a[..., : self.q_lora_rank]
            kv = qkv_a[..., self.q_lora_rank :]
            del qkv_a
        else:
            kv, _ = self.wkv(x)
            q, _ = self.wq_a(x)
        q = self.q_norm(q)
        q_lora = q
        q, _ = self.wq_b(q)
        q = q.view(-1, self.n_local_heads, self.head_dim)
        if self.use_jit_norm:
            q = rmsnorm_self(q, self.eps)
        else:
            q = rms_normalize_triton(q, self.eps)

        kv = self.kv_norm(kv)

        fused_rope(
            q[..., -self.qk_rope_head_dim :],
            kv[..., -self.qk_rope_head_dim :].unsqueeze(1),
            self.freqs_cis,
            positions=positions,
        )

        if self.nsa_enable_prefill_cp and nsa_use_prefill_cp(forward_batch):
            kv = cp_all_gather_rerange_output(
                kv.contiguous(),
                self.cp_size,
                forward_batch,
                torch.cuda.current_stream(),
            )
            if envs.SGLANG_DEBUG_HACK_CP_CHECK_RANK_CONSISTENCY.get():
                assert_tensor_identical_across_cp_ranks(
                    kv,
                    tag=f"kv_after_allgather layer_id={self.layer_id}",
                    forward_batch=forward_batch,
                )

        if self.overlap_store_cache:
            attn_backend.store_cache(
                layer_id=self.layer_id,
                swa_k=kv,
                forward_batch=forward_batch,
            )

        if self.indexer is not None:
            self.indexer(x=x, q_lora=q_lora, forward_batch=forward_batch)
        if self.compressor is not None:
            attn_backend.forward_core_compressor(
                x,
                forward_batch,
                self.layer_id,
                self.compressor,
            )

        if q_out is not None:
            q_out.copy_(q)
        return q, kv

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        debug_return_kv: bool = False,
    ) -> torch.Tensor:
        if not get_attn_tp_context().input_scattered and x.shape[0] == 0:
            assert (
                not self.wo_b.reduce_results
            ), "short-circuiting allreduce will lead to hangs"
            return x

        attn_backend = forward_batch.attn_backend
        if TYPE_CHECKING:
            assert isinstance(attn_backend, DeepseekV4BackendRadix)

        freqs_cis = None

        enable_multi_stream = (
            envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.get()
            and self.alt_streams is not None
            and get_is_capture_mode()
            and x.shape[0] <= self._multi_stream_bs_limit
            and not (self.nsa_enable_prefill_cp and nsa_use_prefill_cp(forward_batch))
        )

        tp_slice, q_padded, q_out = slice(None), None, None
        if self.tp_size > 1:
            q_padded = x.new_empty(x.shape[0], self.n_heads, self.head_dim)
            rank = self.tp_rank
            tp_slice = slice(rank * self.n_local_heads, (rank + 1) * self.n_local_heads)
            q_out = q_padded[:, tp_slice, :]

        if enable_multi_stream:
            q, kv = self._forward_prepare_multi_stream(
                x, positions, forward_batch, attn_backend, freqs_cis, q_out
            )
        else:
            q, kv = self._forward_prepare(
                x, positions, forward_batch, attn_backend, freqs_cis, q_out
            )

        o = attn_backend.forward(
            q=q_padded if q_padded is not None else q,
            k=kv,
            v=kv,
            layer=self.attn_mqa,
            forward_batch=forward_batch,
            compress_ratio=self.compress_ratio,
            attn_sink=self.attn_sink,
            save_kv_cache=not self.overlap_store_cache,
        )
        o = o[:, tp_slice, :]
        fused_rope(
            o[..., -self.qk_rope_head_dim :],
            None,
            self.freqs_cis,
            positions=positions,
            inverse=True,
        )

        o = o.view(o.shape[0], self.n_local_groups, -1)

        if _FP8_WO_A_GEMM:
            import deep_gemm

            T, G, D = o.shape
            R = self.o_lora_rank
            o_fp8, o_s = sglang_per_token_group_quant_fp8(
                o.reshape(T * G, D).contiguous(),
                group_size=128,
            )
            output = torch.empty(T, G, R, device=o.device, dtype=torch.bfloat16)
            deep_gemm.fp8_einsum(
                "bhr,hdr->bhd",
                (o_fp8.view(T, G, D), o_s.view(T, G, -1)),
                (self.wo_a.weight.view(G, R, D), self.wo_a.weight_scale_inv.data),
                output,
                recipe=(1, 1, 128),
            )
            o = output
        else:
            wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
            o = torch.einsum("tgd,grd->tgr", o, wo_a)

        o, _ = self.wo_b(o.flatten(1))

        return o


class DeepseekV4DecoderLayer(nn.Module):
    def __init__(
        self,
        config: DeepSeekV4Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        moe_quant_config_override: Optional[QuantizationConfig] = None,
        is_nextn: bool = False,
        prefix: str = "",
        alt_streams: Optional[List[torch.cuda.Stream]] = None,
        compress_ratio_override: Optional[int] = None,
        padded_moe_intermediate_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_id = layer_id
        self.is_nextn = is_nextn
        self.self_attn = MQALayer(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
            alt_streams=alt_streams,
            compress_ratio_override=compress_ratio_override,
        )
        self.is_layer_sparse = self._is_layer_sparse(layer_id, is_nextn=is_nextn)
        is_previous_layer_sparse = self._is_layer_sparse(layer_id - 1, is_nextn=False)
        is_next_layer_sparse = self._is_layer_sparse(layer_id + 1, is_nextn=False)
        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=1 if is_nextn else config.num_hidden_layers,
            is_layer_sparse=self.is_layer_sparse,
            is_previous_layer_sparse=is_previous_layer_sparse,
            is_next_layer_sparse=is_next_layer_sparse,
        )
        self.mlp = deepseek_v2.DeepseekV2MoE(
            config=config,
            quant_config=moe_quant_config_override or quant_config,
            prefix=add_prefix("mlp", prefix),
            layer_id=self.layer_id,
            alt_stream=alt_streams[0] if alt_streams is not None else None,
            is_nextn=is_nextn,
            is_deepseek_v4=True,
            padded_moe_intermediate_size=padded_moe_intermediate_size,
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.hc_mult = hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * config.hidden_size
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.rms_norm_eps = config.rms_norm_eps
        self.nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()

    def _is_layer_sparse(self, layer_id: int, is_nextn: bool) -> bool:
        if envs.SGLANG_DSV4_MODE.get() == "2604":
            first_k_dense_replace = 0
            moe_layer_freq = 1
        else:
            first_k_dense_replace = self.config.first_k_dense_replace
            moe_layer_freq = self.config.moe_layer_freq
        return is_nextn or (
            self.config.n_routed_experts is not None
            and layer_id >= first_k_dense_replace
            and layer_id % moe_layer_freq == 0
        )

    def hc_pre(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ):
        @maybe_torch_compile
        def hc_pre_torch_impl(x, hc_fn):
            x_flat = x.flatten(1).float()
            rsqrt = torch.rsqrt(
                x_flat.square().mean(-1, keepdim=True) + self.rms_norm_eps
            )
            mixes = (F.linear(x_flat, hc_fn) * rsqrt).unsqueeze(1)
            return x_flat, mixes

        shape, dtype = x.size(), x.dtype

        if x.shape[0] == 0:
            y = torch.empty((0, shape[-1]), dtype=dtype, device=x.device)
            post = torch.empty((0, self.hc_mult), dtype=dtype, device=x.device)
            comb = torch.empty(
                (0, self.hc_mult, self.hc_mult), dtype=dtype, device=x.device
            )
            return y, post, comb

        if envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.get():
            from sglang.srt.layers.mhc import mhc_pre

            post, comb, y = mhc_pre(
                residual=x,
                fn=hc_fn,
                hc_scale=hc_scale,
                hc_base=hc_base,
                rms_eps=self.rms_norm_eps,
                hc_pre_eps=self.hc_eps,
                hc_sinkhorn_eps=self.hc_eps,
                hc_post_mult_value=2.0,
                sinkhorn_repeat=self.hc_sinkhorn_iters,
            )
            return y, post.squeeze(-1), comb

        if envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.get():
            import deep_gemm

            x_flat = x.flatten(1).bfloat16()

            m, k = x_flat.shape
            mix_hc = hc_fn.size(0)
            d_out = torch.empty((m, mix_hc), dtype=torch.float, device=x.device)
            s_out = torch.empty((m,), dtype=torch.float, device=x.device)
            deep_gemm.tf32_hc_prenorm_gemm(
                x_flat, hc_fn.float().contiguous(), d_out, s_out, num_splits=None
            )
            rsqrt = torch.rsqrt(s_out / k + self.rms_norm_eps)
            mixes = (d_out * rsqrt.unsqueeze(1)).unsqueeze(1)
        else:
            x_flat, mixes = hc_pre_torch_impl(x, hc_fn)

        from sglang.srt.layers.mhc import hc_split_sinkhorn

        pre, post, comb = hc_split_sinkhorn(
            mixes,
            hc_scale,
            hc_base,
            self.hc_mult,
            self.hc_sinkhorn_iters,
            self.hc_eps,
        )
        y = (pre.squeeze(1).unsqueeze(-1) * x_flat.view(shape)).sum(dim=1)
        return y.to(dtype), post.squeeze(1), comb.squeeze(1)

    def hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ):

        if x.shape[0] == 0:
            return torch.empty(
                (0, self.hc_mult, x.shape[-1]), dtype=x.dtype, device=x.device
            )

        if envs.SGLANG_OPT_USE_TILELANG_MHC_POST.get():
            from sglang.srt.layers.mhc import mhc_post

            return mhc_post(x, residual, post, comb)

        assert residual.shape == (x.shape[0], self.hc_mult, x.shape[-1])
        assert post.shape == (x.shape[0], self.hc_mult)
        assert comb.shape == (x.shape[0], self.hc_mult, self.hc_mult)

        @maybe_torch_compile
        def hc_post_torch_impl(x, residual, post, comb):
            return (
                post.unsqueeze(-1) * x.unsqueeze(1)
                + (comb.unsqueeze(-1) * residual.unsqueeze(2)).sum(dim=1)
            ).type_as(x)

        return hc_post_torch_impl(x, residual, post, comb)

    def forward(
        self,
        positions: torch.tensor,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        forward_batch: ForwardBatch,
        input_ids_global: torch.Tensor,
    ) -> torch.Tensor:
        if envs.SGLANG_DSV4_2604_SUBMODE.get() == "2604B":
            assert deepseek_v4_moe_code_path_checker.observed == 0

        residual = hidden_states
        hidden_states, post, comb = self.hc_pre(
            hidden_states, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            x=hidden_states,
            positions=positions,
            forward_batch=forward_batch,
        )

        hidden_states = self.hc_post(hidden_states, residual, post, comb)
        residual = hidden_states
        hidden_states, post, comb = self.hc_pre(
            hidden_states, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        hidden_states = self.post_attention_layernorm(hidden_states)

        _use_cp = self.nsa_enable_prefill_cp and nsa_use_prefill_cp(forward_batch)
        _use_tp_moe_gather = (
            not _use_cp
            and get_attention_dp_size() > 1
            and get_moe_a2a_backend().is_none()
        )
        _use_tp_attn_a2a_scatter = (
            not _use_cp
            and envs.SGLANG_DSV4_FIX_TP_ATTN_A2A_SCATTER.get()
            and get_attention_tp_size() > 1
            and not get_moe_a2a_backend().is_none()
        )
        if _use_cp:
            assert get_moe_a2a_backend().is_deepep(), (
                "CP requires DeepEP (moe_a2a_backend == deepep). "
                "Only DeepEP is tested with CP's per-rank token split."
            )
            cp_rank = get_attention_tp_rank()
            cp_size = get_attention_tp_size()
            input_ids = input_ids[cp_rank::cp_size].contiguous()
            input_ids_global = input_ids
        elif _use_tp_moe_gather:
            hidden_states, local_hidden_states = get_global_dp_buffer(), hidden_states
            dp_gather_partial(hidden_states, local_hidden_states, forward_batch)
        _a2a_scatter_chunks: Optional[List[torch.Tensor]] = None
        if _use_tp_attn_a2a_scatter:
            s, r = get_attention_tp_size(), get_attention_tp_rank()
            _a2a_scatter_chunks = list(hidden_states.tensor_split(s))
            hidden_states = _a2a_scatter_chunks[r].contiguous()
            input_ids = input_ids.tensor_split(s)[r].contiguous()
            input_ids_global = input_ids_global.tensor_split(s)[r].contiguous()
        hidden_states = self.mlp(
            hidden_states,
            forward_batch,
            input_ids=input_ids,
            input_ids_global=input_ids_global,
        )
        if _use_tp_moe_gather:
            hidden_states, global_hidden_states = get_local_dp_buffer(), hidden_states
            dp_scatter(hidden_states, global_hidden_states, forward_batch)
        if _use_tp_attn_a2a_scatter:
            assert _a2a_scatter_chunks is not None
            gathered = [torch.empty_like(t) for t in _a2a_scatter_chunks]
            attn_tp_all_gather(gathered, hidden_states.contiguous())
            hidden_states = torch.cat(gathered)

        hidden_states = self.hc_post(hidden_states, residual, post, comb)

        if envs.SGLANG_DSV4_2604_SUBMODE.get() == "2604B":
            assert deepseek_v4_moe_code_path_checker.observed == 1
            deepseek_v4_moe_code_path_checker.observed = 0

        return hidden_states


class DeepseekV4Model(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: DeepSeekV4Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.padding_id = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.pp_group = get_pp_group()
        self.first_k_dense_replace = config.first_k_dense_replace
        (
            self.moe_intermediate_size,
            self.padded_moe_intermediate_size,
            self.moe_weight_block_size,
        ) = _get_deepseek_v4_moe_padding_metadata(
            config=config,
            quant_config=quant_config,
            tp_size=get_tensor_model_parallel_world_size(),
        )
        self.moe_padding_enabled = (
            self.padded_moe_intermediate_size != self.moe_intermediate_size
        )
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            enable_tp=not is_dp_attention_enabled(),
        )
        self.rms_norm_eps = config.rms_norm_eps
        self.alt_streams = (
            [torch.cuda.Stream() for _ in range(5)] if (_is_cuda or _is_hip) else None
        )
        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: DeepseekV4DecoderLayer(
                config=config,
                layer_id=idx,
                quant_config=quant_config,
                prefix=prefix,
                alt_streams=self.alt_streams,
                padded_moe_intermediate_size=(
                    self.padded_moe_intermediate_size
                    if self.moe_padding_enabled
                    else None
                ),
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gemm_output_zero_allocator_size = 0
        self.layers_to_capture = []
        if get_moe_a2a_backend().is_deepep() or get_moe_a2a_backend().is_mooncake():
            self.enable_a2a_moe = True
        else:
            self.enable_a2a_moe = False

        self.hc_eps = config.hc_eps
        self.hc_mult = hc_mult = config.hc_mult
        self.norm_eps = config.rms_norm_eps
        hc_dim = hc_mult * config.hidden_size
        self.hc_head_fn = nn.Parameter(
            torch.empty(hc_mult, hc_dim, dtype=torch.float32)
        )
        self.hc_head_base = nn.Parameter(torch.empty(hc_mult, dtype=torch.float32))
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))

        self.nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()
        if self.nsa_enable_prefill_cp:
            self.cp_size = get_attention_tp_size()

    def hc_head(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ):
        shape, dtype = x.size(), x.dtype
        x = x.flatten(1).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, hc_fn) * rsqrt
        pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)
        return y.to(dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor],
        pp_proxy_tensors: Optional[PPProxyTensors],
    ) -> torch.Tensor:
        total_num_layers = self.end_layer - self.start_layer
        device = input_embeds.device if input_embeds is not None else input_ids.device
        zero_allocator = BumpAllocator(
            buffer_size=total_num_layers * 2 * (2 if forward_batch.can_run_tbo else 1),
            dtype=torch.float32,
            device=device,
        )
        has_gemm_output_zero_allocator = hasattr(
            self, "gemm_output_zero_allocator_size"
        )
        gemm_output_zero_allocator = (
            BumpAllocator(
                buffer_size=self.gemm_output_zero_allocator_size,
                dtype=torch.float32,
                device=device,
            )
            if has_gemm_output_zero_allocator
            and self.gemm_output_zero_allocator_size > 0
            else None
        )
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)

        if get_attention_dp_size() > 1 and get_moe_a2a_backend().is_none():
            input_ids_global = torch.empty(
                (_DpGatheredBufferWrapper._global_dp_buffer_len, 1),
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            dp_gather_partial(input_ids_global, input_ids[:, None], forward_batch)
            input_ids_global = input_ids_global.squeeze(-1)
        else:
            input_ids_global = input_ids

        if nsa_use_prefill_cp(forward_batch):
            _check_rank_consistency = (
                envs.SGLANG_DEBUG_HACK_CP_CHECK_RANK_CONSISTENCY.get()
            )
            if _check_rank_consistency:
                _pre_split_hidden_states = hidden_states.clone()
                _pre_split_positions = positions.clone()
            hidden_states = cp_split_and_rebuild_data(forward_batch, hidden_states)
            positions = cp_split_and_rebuild_position(forward_batch, positions)
            if _check_rank_consistency:
                _gathered_hidden = cp_all_gather_rerange_output(
                    hidden_states,
                    self.cp_size,
                    forward_batch,
                    torch.cuda.current_stream(),
                )
                assert torch.equal(_gathered_hidden, _pre_split_hidden_states), (
                    "SGLANG_DEBUG_HACK_CP_CHECK_RANK_CONSISTENCY: "
                    "cp_split_and_rebuild_data ∘ cp_all_gather_rerange_output is not identity on hidden_states. "
                    "Round-robin split/gather helpers are inconsistent."
                )
                _gathered_positions = cp_all_gather_rerange_output(
                    positions.unsqueeze(-1),
                    self.cp_size,
                    forward_batch,
                    torch.cuda.current_stream(),
                ).squeeze(-1)
                assert torch.equal(_gathered_positions, _pre_split_positions), (
                    "SGLANG_DEBUG_HACK_CP_CHECK_RANK_CONSISTENCY: "
                    "cp_split_and_rebuild_position ∘ cp_all_gather_rerange_output is not identity on positions."
                )

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states = layer(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                input_ids=input_ids,
                input_ids_global=input_ids_global,
            )

        if nsa_use_prefill_cp(forward_batch):
            hidden_states = cp_all_gather_rerange_output(
                hidden_states,
                self.cp_size,
                forward_batch,
                torch.cuda.current_stream(),
            )

        pre_hc_head = (
            hidden_states.flatten(1)
            if envs.SGLANG_FIX_MTP_HC_HIDDEN.get()
            and envs.SGLANG_DSV4_MODE.get() == "2604"
            else None
        )

        hidden_states = self.hc_head(
            hidden_states, self.hc_head_fn, self.hc_head_scale, self.hc_head_base
        )
        hidden_states = self.norm(hidden_states)

        if pre_hc_head is not None:
            return hidden_states, pre_hc_head
        return hidden_states


class DeepseekV4ForCausalLM(nn.Module):
    def __init__(
        self,
        config: DeepSeekV4Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.quant_config = quant_config
        self.determine_num_fused_shared_experts()
        self.model = DeepseekV4Model(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.pp_group = get_pp_group()
        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
                use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
            )
        self.logits_processor = LogitsProcessor(config)
        self.capture_aux_hidden_states = False
        get_attn_tp_context().init_context(config.q_lora_rank, is_nsa=True)

        self._routed_experts_weights_of_layer = LazyValue(
            lambda: {
                layer_id: layer.mlp.get_moe_weights()
                for layer_id, layer in enumerate(self.model.layers)
                if isinstance(layer.mlp, deepseek_v2.DeepseekV2MoE)
            }
        )

        self.nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()
        if self.nsa_enable_prefill_cp:
            self.cp_rank = get_attention_tp_rank()
            self.cp_size = get_attention_tp_size()

    @property
    def routed_experts_weights_of_layer(self):
        return self._routed_experts_weights_of_layer.value

    def determine_num_fused_shared_experts(self):
        self.num_fused_shared_experts = 0
        if get_global_server_args().disable_shared_experts_fusion:
            return

        disable_reason = None
        if self.config.n_routed_experts != 256 or self.config.n_shared_experts != 1:
            disable_reason = "Config not support fused shared expert(s)."
        elif (not _is_cuda or torch.cuda.get_device_capability("cuda") < (8, 0)) and (
            not _is_hip or torch.cuda.get_device_capability("cuda") < (9, 4)
        ):
            disable_reason = (
                "Only Deepseek V3/R1 on NV-platform with capability >= 80 "
                "or AMD-platform with capability >= gfx942(MI30x) can use shared experts fusion optimization."
            )
        elif get_moe_expert_parallel_world_size() > 1 and (
            not _is_hip or torch.cuda.get_device_capability("cuda") < (9, 4)
        ):
            disable_reason = "Only Deepseek V3/R1 on AMD-platform with capability >= gfx942(MI30x) can use shared experts fusion optimization under expert parallelism."
        elif disable_reason is None and get_moe_a2a_backend().is_deepep():
            disable_reason = "Deepseek V3/R1 can not use shared experts fusion optimization under deepep expert parallelism."
        elif self.quant_config and self.quant_config.get_name() == "w4afp8":
            disable_reason = "Deepseek V3/R1 W4AFP8 model uses different quant method for routed experts and shared experts."
        elif (
            envs.SGLANG_DSV4_MODE.get() == "2604" and envs.SGLANG_DSV4_FP4_EXPERTS.get()
        ):
            disable_reason = "2604 routed experts use FP4 while shared experts remain FP8; fusion would incorrectly apply FP4 to shared experts."

        if envs.SGLANG_DSV4_2604_SUBMODE.get() == "2604B":
            disable_reason = "2604B checkpoint requires different clamping for shared and routed experts"

        if disable_reason is not None:
            get_global_server_args().disable_shared_experts_fusion = True
            self.num_fused_shared_experts = 0
            log_info_on_rank0(
                logger,
                f"{disable_reason} Shared experts fusion optimization is disabled.",
            )
            return

        self.num_fused_shared_experts = self.config.n_shared_experts

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        if self.nsa_enable_prefill_cp:
            if can_cp_split(len(input_ids), self.cp_size, True, forward_batch):
                forward_batch.nsa_cp_metadata = prepare_input_dp_with_cp_dsa(
                    len(input_ids),
                    self.cp_rank,
                    self.cp_size,
                    forward_batch.seq_lens_cpu.tolist(),
                )

        with get_attn_tp_context().maybe_input_scattered(forward_batch):
            hidden_states = self.model.forward(
                input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors
            )
        aux_hidden_states = None
        pre_hc_head = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states
        if (
            envs.SGLANG_FIX_MTP_HC_HIDDEN.get()
            and envs.SGLANG_DSV4_MODE.get() == "2604"
        ):
            hidden_states, pre_hc_head = hidden_states
        return self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            forward_batch,
            aux_hidden_states,
            hidden_states_before_norm=pre_hc_head,
        )

    def _setup_fp8_wo_a_scales(self, is_nextn: bool) -> None:
        from deep_gemm import transform_sf_into_required_layout

        layers = self.model.layers
        for layer in layers:
            attn = layer.self_attn
            G = attn.n_local_groups
            R = attn.o_lora_rank
            D = attn.wo_a.weight.shape[1]

            raw_scale = attn.wo_a.weight_scale_inv.data.view(G, R // 128, D // 128)
            attn.wo_a.weight_scale_inv.data = transform_sf_into_required_layout(
                raw_scale,
                mn=R,
                k=D,
                recipe=(1, 128, 128),
                num_groups=G,
                is_sfa=False,
            )

    def post_load_weights(self, is_nextn=False, weight_names=None):
        if _FP8_WO_A_GEMM:
            self._setup_fp8_wo_a_scales(is_nextn)

        if is_nextn:
            return
        for layer in self.model.layers:
            self_attn = layer.self_attn
            if self_attn.compress_ratio != 0 and not self_attn.compressor.ape_converted:
                self_attn.compressor.apply_ape_hotfix()
            if (
                self_attn.compress_ratio == 4
                and not self_attn.indexer.compressor.ape_converted
            ):
                self_attn.indexer.compressor.apply_ape_hotfix()

    @staticmethod
    def remap_weight_name_to_dpsk_hf_format(
        name: str, is_nextn: bool = False, num_hidden_layers: Optional[int] = None
    ) -> str:
        if name == "embed.weight":
            return "model.embed_tokens.weight"
        if name == "head.weight":
            return "lm_head.weight"
        if name == "norm.weight":
            return "model.norm.weight"
        if name.startswith("hc_head_"):
            return "model." + name

        if is_nextn and name.startswith("mtp."):
            parts = name.split(".", 2)
            if len(parts) >= 3:
                rest = parts[2]
                nextn_spec_prefixes = [
                    "e_proj",
                    "h_proj",
                    "emb",
                    "enorm",
                    "hnorm",
                    "norm",
                    "head",
                    "hc_head",
                ]
                is_nextn_spec = any(rest.startswith(p) for p in nextn_spec_prefixes)
                if is_nextn_spec:
                    if rest.startswith("emb.tok_emb"):
                        rest = rest.replace("emb.tok_emb", "embed_tokens")
                    elif rest == "norm.weight":
                        rest = "shared_head.norm.weight"
                    elif rest.startswith("head."):
                        rest = "shared_head.head.weight"
                    elif rest == "e_proj.scale":
                        rest = "e_proj.weight_scale_inv"
                    elif rest == "h_proj.scale":
                        rest = "h_proj.weight_scale_inv"
                name = f"model.layers.{num_hidden_layers}." + rest

        if name.startswith("layers."):
            name = "model." + name
        name = name.replace(".attn.", ".self_attn.")
        name = name.replace(".ffn.", ".mlp.")
        name = name.replace(".attn_norm.", ".input_layernorm.")
        name = name.replace(".ffn_norm.", ".post_attention_layernorm.")

        if not ATTN_BIT_WISE_EQUAL_MODE:
            if "self_attn" in name and (
                "compressor" not in name or not COMPRESSOR_BIT_WISE_EQUAL_MODE
            ):
                name = name.replace(".scale", ".weight_scale_inv")

        if not MOE_BIT_WISE_EQUAL_MODE:
            name = name.replace(".gate.tid2eid", ".topk.tid2eid")
            name = name.replace(".gate.bias", ".gate.e_score_correction_bias")
            name = name.replace(".w1.", ".gate_proj.")
            name = name.replace(".w2.", ".down_proj.")
            name = name.replace(".w3.", ".up_proj.")
            if "mlp" in name:
                name = name.replace(".scale", ".weight_scale_inv")

        return name

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=False):
        assert envs.SGLANG_DSV4_MODE.get() in ["2601", "2604"]
        if envs.SGLANG_DSV4_MODE.get() == "2604":
            assert envs.SGLANG_DSV4_2604_SUBMODE.get() in ["2604A", "2604B"]
        else:
            assert envs.SGLANG_DSV4_2604_SUBMODE.get() == ""

        if (
            envs.SGLANG_DEBUG_SANITY_CHECK_CONFIG.get()
            and envs.SGLANG_DSV4_MODE.get() == "2604"
        ):
            _debug_assert_model_path_configs()
        if envs.SGLANG_DEBUG_SANITY_CHECK_CONFIG.get() and is_large_dummy_model():
            assert (
                envs.SGLANG_HACK_OVERRIDE_TOPK_IDS_RANDOM.get()
            ), "dummy model must use SGLANG_HACK_OVERRIDE_TOPK_IDS_RANDOM"

        if MOE_BIT_WISE_EQUAL_MODE:
            assert (
                self.num_fused_shared_experts == 0
            ), "use --disable-shared-experts-fusion for MoE bit-wise equal mode"

        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()

        if is_nextn:
            if hasattr(self.config, "num_nextn_predict_layers"):
                num_nextn_layers = self.config.num_nextn_predict_layers
                assert num_nextn_layers == 1, "Only 1 nextn layer is supported"
                nextn_layer_id = (
                    0
                    if self.config.num_hidden_layers == 1
                    else self.config.num_hidden_layers
                )
            else:
                raise ValueError("num_nextn_predict_layers is not in the config")

        if (
            envs.SGLANG_DSV4_MODE.get() == "2604"
            and not envs.SGLANG_OPT_FP8_WO_A_GEMM.get()
        ):
            if envs.SGLANG_FIX_DSV4_BASE_MODEL_LOAD.get():
                weights = list(weights)
                exists_wo_a_scale = any(n.endswith(".wo_a.scale") for n, t in weights)
                if exists_wo_a_scale:
                    logger.info("Execute dequant fp8 wo_a")
                    weights = _dequant_fp8_wo_a(weights)
                else:
                    logger.info("Skip dequant fp8 wo_a")
            else:
                # ----------------------------- legacy code ------------------------------
                if envs.SGLANG_DSV4_FP4_EXPERTS.get():
                    weights = _dequant_fp8_wo_a(weights)
                else:
                    weights = ((n, t) for n, t in weights if not n.endswith(".wo_a.scale"))
                # ------------------------------------------------------------------------

        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts + self.num_fused_shared_experts,
        )

        if self.quant_config and self.quant_config.get_name() == "w4afp8":
            expert_params_mapping += FusedMoE.make_expert_input_scale_params_mapping(
                num_experts=self.config.n_routed_experts
            )

        cache_compressor_weight = {}
        COMPRESSOR_PART = ".compressor.w"

        fuse_wqa_wkv = envs.SGLANG_OPT_FUSE_WQA_WKV.get()
        cache_wqkv_a_weight: dict[str, dict[str, torch.Tensor]] = {}

        def auto_weight_loader(module):
            return getattr(module, "weight_loader", default_weight_loader)

        if is_nextn:
            nextn_layer_prefix = f"model.layers.{nextn_layer_id}"
            nextn_spec_weight_names_out_of_layer = [
                "shared_head.norm",
                "shared_head.head",
                "embed_tokens",
                ".e_proj",
                "h_proj",
                "enorm",
                "hnorm",
                "hc_head_base",
                "hc_head_fn",
                "hc_head_scale",
            ]

        if self.num_fused_shared_experts > 0:
            assert self.num_fused_shared_experts == 1
            log_info_on_rank0(logger, "Shared experts fusion optimization enabled.")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            weight_names = []
            for name, loaded_weight in weights:
                try:
                    use_async_loading = should_async_load(loaded_weight)

                    name = self.remap_weight_name_to_dpsk_hf_format(
                        name,
                        is_nextn=is_nextn,
                        num_hidden_layers=self.config.num_hidden_layers,
                    )
                    loaded_weight = _maybe_pad_deepseek_v4_moe_checkpoint_tensor(
                        name,
                        loaded_weight,
                        moe_intermediate_size=self.model.moe_intermediate_size,
                        padded_moe_intermediate_size=self.model.padded_moe_intermediate_size,
                        weight_block_size=self.model.moe_weight_block_size,
                        num_shared_experts=self.config.n_shared_experts,
                    )

                    layer_id = get_layer_id(name)
                    if (
                        layer_id is not None
                        and hasattr(self.model, "start_layer")
                        and (
                            layer_id < self.model.start_layer
                            or layer_id >= self.model.end_layer
                        )
                    ):
                        continue
                    if (
                        self.num_fused_shared_experts > 0
                        and "mlp.shared_experts" in name
                    ):
                        name = name.replace(
                            "mlp.shared_experts",
                            f"mlp.experts.{self.config.n_routed_experts}",
                        )

                    weight_names.append(name)

                    if not is_nextn:
                        if hasattr(self.config, "num_nextn_predict_layers"):
                            num_nextn_layers = self.config.num_nextn_predict_layers
                            if num_nextn_layers > 0 and name.startswith("model.layers"):
                                name_list = name.split(".")
                                if (
                                    len(name_list) >= 3
                                    and int(name_list[2])
                                    >= self.config.num_hidden_layers
                                ):
                                    continue

                            if name.startswith("mtp"):
                                continue
                    else:
                        if "shared_head.head" in name or "embed_tokens" in name:
                            continue

                        if not name.startswith(nextn_layer_prefix):
                            continue

                        in_decoder = True
                        for weight_name in nextn_spec_weight_names_out_of_layer:
                            if weight_name in name:
                                in_decoder = False
                                name = name.replace(nextn_layer_prefix, "model")
                                break

                        if in_decoder:
                            name = name.replace(nextn_layer_prefix, "model.decoder")

                    if "rotary_emb.inv_freq" in name:
                        continue
                    for param_name, weight_name, shard_id in stacked_params_mapping:
                        if weight_name not in name:
                            continue
                        if _is_npu:
                            name = name.replace("weight_packed", "weight")
                        if ("mlp.experts." in name) and name not in params_dict:
                            continue
                        name = name.replace(weight_name, param_name)
                        if name.endswith(".bias") and name not in params_dict:
                            continue
                        if name not in params_dict and name.startswith("mtp"):
                            break
                        param = params_dict[name]
                        weight_loader = param.weight_loader
                        maybe_executor_submit(
                            executor=executor,
                            futures=futures,
                            use_async=use_async_loading,
                            func=weight_loader,
                            func_args=(param, loaded_weight, shard_id),
                        )
                        loaded_params.add(name)
                        break
                    else:
                        for mapping in expert_params_mapping:
                            if MOE_BIT_WISE_EQUAL_MODE:
                                continue
                            param_name, weight_name, expert_id, shard_id = mapping
                            if weight_name not in name:
                                continue
                            if _is_npu:
                                name = name.replace("weight_packed", "weight")
                            name = name.replace(weight_name, param_name)
                            if name not in params_dict:
                                continue
                            param = params_dict[name]
                            weight_loader = param.weight_loader
                            maybe_executor_submit(
                                executor=executor,
                                futures=futures,
                                use_async=use_async_loading,
                                func=weight_loader,
                                func_args=(
                                    param,
                                    loaded_weight,
                                    name,
                                ),
                                func_kwargs={
                                    "shard_id": shard_id,
                                    "expert_id": expert_id,
                                },
                            )
                            loaded_params.add(name)
                            break
                        else:
                            if name.endswith(".bias") and name not in params_dict:
                                continue
                            if (
                                ".embed_tokens." in name
                                and not self.pp_group.is_first_rank
                            ):
                                continue
                            if ".norm." in name and not self.pp_group.is_last_rank:
                                continue
                            elif COMPRESSOR_PART in name:
                                is_kv = name.endswith(".wkv.weight")
                                is_wgate = name.endswith(".wgate.weight")
                                assert is_kv != is_wgate
                                key = name.rsplit(".", 2)[0]
                                assert key.endswith(".compressor")
                                if key not in cache_compressor_weight:
                                    cache_compressor_weight[key] = (
                                        is_kv,
                                        loaded_weight,
                                    )
                                else:
                                    assert key in cache_compressor_weight
                                    cached_is_kv, cached_weight = (
                                        cache_compressor_weight[key]
                                    )
                                    assert cached_is_kv != is_kv
                                    kv = loaded_weight if is_kv else cached_weight
                                    wgate = loaded_weight if is_wgate else cached_weight
                                    fused_weight = torch.cat([kv, wgate], dim=0)
                                    param_name = key + ".wkv_gate.weight"
                                    param = params_dict[param_name]
                                    weight_loader = auto_weight_loader(param)
                                    maybe_executor_submit(
                                        executor=executor,
                                        futures=futures,
                                        use_async=use_async_loading,
                                        func=weight_loader,
                                        func_args=(param, fused_weight),
                                    )
                                    loaded_params.add(param_name)
                                    cache_compressor_weight.pop(key)
                            elif fuse_wqa_wkv and (
                                name.endswith(".wq_a.weight")
                                or name.endswith(".wq_a.weight_scale_inv")
                                or name.endswith(".wkv.weight")
                                or name.endswith(".wkv.weight_scale_inv")
                            ):
                                is_q = ".wq_a." in name
                                param_name = name.replace(
                                    ".wq_a." if is_q else ".wkv.", ".wqkv_a."
                                )
                                bucket = cache_wqkv_a_weight.setdefault(param_name, {})
                                shard_key = "q" if is_q else "kv"
                                assert (
                                    shard_key not in bucket
                                ), f"duplicate shard {shard_key} for {param_name}"
                                bucket[shard_key] = loaded_weight
                                if len(bucket) == 2:
                                    fused_weight = torch.cat(
                                        [bucket["q"], bucket["kv"]], dim=0
                                    )
                                    param = params_dict[param_name]
                                    weight_loader = auto_weight_loader(param)
                                    maybe_executor_submit(
                                        executor=executor,
                                        futures=futures,
                                        use_async=use_async_loading,
                                        func=weight_loader,
                                        func_args=(param, fused_weight),
                                    )
                                    loaded_params.add(param_name)
                                    cache_wqkv_a_weight.pop(param_name)
                            else:
                                if (
                                    "k_scale" in name or "v_scale" in name
                                ) and name not in params_dict:
                                    for scale in ["k_scale", "v_scale"]:
                                        if scale in name:
                                            name = name.replace(
                                                f"{scale[0]}_proj", "attn_mqa"
                                            )
                                            break
                                if name not in params_dict:
                                    if not name.startswith("mtp"):
                                        logger.warning(
                                            f"{name} not found in params_dict."
                                        )
                                    continue
                                param = params_dict[name]

                                weight_loader = auto_weight_loader(param)
                                maybe_executor_submit(
                                    executor=executor,
                                    futures=futures,
                                    use_async=use_async_loading,
                                    func=weight_loader,
                                    func_args=(param, loaded_weight),
                                )
                                loaded_params.add(name)
                except Exception as e:
                    e.add_note(f"{name=} {loaded_weight.shape=}")
                    raise

            for future in concurrent.futures.as_completed(futures):
                future.result()

        assert len(cache_compressor_weight) == 0
        assert len(cache_wqkv_a_weight) == 0, cache_wqkv_a_weight.keys()
        unloaded_params = params_dict.keys() - loaded_params

        skipped_checking_patterns = ["attn_mqa.k_scale", "attn_mqa.v_scale"]
        if is_nextn:
            skipped_checking_patterns.extend(["lm_head", "embed_tokens"])
        unloaded_params = {
            p
            for p in unloaded_params
            if all(
                skipped_checking_pattern not in p
                for skipped_checking_pattern in skipped_checking_patterns
            )
        }
        if os.environ.get("SGLANG_SKIP_CHECKPOINT_LOAD_CHECK", "0") == "0":
            if unloaded_params:
                raise RuntimeError(
                    f"Some weights are not initialized from checkpoints: {unloaded_params}"
                )

        self.post_load_weights(is_nextn=is_nextn, weight_names=weight_names)

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.n_routed_experts,
            num_groups=None,
        )


EntryClass = [DeepseekV4ForCausalLM]


def _dequant_fp8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    from einops import rearrange

    assert (
        weight.dtype == torch.float8_e4m3fn
    ), f"expected fp8_e4m3fn, got {weight.dtype}"
    assert scale.dtype in (
        torch.float8_e8m0fnu,
        torch.float32,
    ), f"expected fp8_e8m0fnu or float32, got {scale.dtype}"
    if envs.SGLANG_DEBUG_SANITY_CHECK_CONFIG.get() and not is_large_dummy_model():
        assert weight.shape == (8192, 4096), f"unexpected weight shape {weight.shape}"
        assert scale.shape == (64, 32), f"unexpected scale shape {scale.shape}"

    weight_f32 = rearrange(
        weight.float(), "(sn bn) (sk bk) -> sn bn sk bk", bn=128, bk=128
    )
    result = rearrange(
        weight_f32 * scale.float()[:, None, :, None], "sn bn sk bk -> (sn bn) (sk bk)"
    )
    if envs.SGLANG_DEBUG_SANITY_CHECK_CONFIG.get() and not is_large_dummy_model():
        assert result.shape == (8192, 4096)

    return result.to(torch.bfloat16)


def build_mega_moe_experts_weights(experts) -> None:
    from deep_gemm import (
        transform_sf_into_required_layout,
        transform_weights_for_mega_moe,
    )
    from deep_gemm.mega import _interleave_l1_weights, _transpose_sf_for_utccp

    if getattr(experts, "_mega_moe_weights_built", False):
        return

    w13 = experts.w13_weight.data
    w13_sf_fp32 = experts.w13_weight_scale_inv.data
    w2 = experts.w2_weight.data
    w2_sf_fp32 = experts.w2_weight_scale_inv.data

    num_groups, n1, half_k1 = w13.shape
    k1 = half_k1 * 2
    _, n2, half_k2 = w2.shape
    k2 = half_k2 * 2

    w13_sf = transform_sf_into_required_layout(
        w13_sf_fp32,
        mn=n1,
        k=k1,
        recipe=(1, 32),
        num_groups=num_groups,
        disable_ue8m0_cast=False,
    )
    w2_sf = transform_sf_into_required_layout(
        w2_sf_fp32,
        mn=n2,
        k=k2,
        recipe=(1, 32),
        num_groups=num_groups,
        disable_ue8m0_cast=False,
    )

    if envs.SGLANG_OPT_FIX_MEGA_MOE_MEMORY.get():
        # Build the interleaved L1 weight + scale once; share the weight buffer
        # between `w13_weight.data` (normal deep-ep path) and `mega_l1_weights[0]`
        # (mega moe path). Mega moe additionally needs a UTCCP-transposed scale;
        # the deep-ep path consumes the non-transposed interleaved scale and a
        # swizzle-aware activation kernel. L2 weight is untouched by the mega
        # transform, so the existing `w2_weight.data` is shared directly.
        w13_interleaved, w13_sf_interleaved = _interleave_l1_weights((w13, w13_sf))
        w13_sf_utccp = _transpose_sf_for_utccp(w13_sf_interleaved)
        w2_sf_utccp = _transpose_sf_for_utccp(w2_sf)

        experts.w13_weight.data = w13_interleaved
        experts.w13_weight_scale_inv.data = w13_sf_interleaved
        experts.w2_weight_scale_inv.data = w2_sf
        experts.w13_weight_scale_inv.format_ue8m0 = True
        experts.w2_weight_scale_inv.format_ue8m0 = True

        experts.mega_l1_weights = (experts.w13_weight.data, w13_sf_utccp)
        experts.mega_l2_weights = (experts.w2_weight.data, w2_sf_utccp)
    else:
        l1_pair, l2_pair = transform_weights_for_mega_moe((w13, w13_sf), (w2, w2_sf))

        experts.mega_l1_weights = l1_pair
        experts.mega_l2_weights = l2_pair

    experts._mega_moe_weights_built = True


def _dequant_fp8_wo_a(
    weights: Iterable[Tuple[str, torch.Tensor]],
) -> Iterable[Tuple[str, torch.Tensor]]:
    weights_dict = dict(weights)

    for name in list(weights_dict.keys()):
        if name not in weights_dict:
            continue
        if not name.endswith(".wo_a.weight"):
            continue
        scale_name = name.replace(".wo_a.weight", ".wo_a.scale")
        assert scale_name in weights_dict
        weight = weights_dict.pop(name)
        scale = weights_dict.pop(scale_name)
        yield name, _dequant_fp8(weight, scale)

    yield from weights_dict.items()


def _debug_assert_model_path_configs() -> None:
    assert_ckpt_version = os.environ.get("SGLANG_HACK_ASSERT_CKPT_VERSION", "v1")

    model_path = Path(get_global_server_args().model_path)
    ref_dir = (
        Path(__file__).resolve().parents[4]
        / "deepseek_v4"
        / "assembled_hf_config_0409"
        / assert_ckpt_version
    )
    for ref_file in ref_dir.iterdir():
        if ref_file.name in ["apply.py", "create.py", "README.md"]:
            continue
        user_file = model_path / ref_file.name
        if not user_file.exists():
            raise AssertionError(
                f"2604 mode: expected {ref_file.name} in model_path {model_path}, but not found"
            )
        if user_file.read_bytes() != ref_file.read_bytes():
            raise AssertionError(
                f"2604 mode: {ref_file.name} in model_path differs from reference.\n"
                f"  model_path: {user_file}\n"
                f"  reference:  {ref_file}\n"
                f"  Please use the files generated by deepseek_v4/assembled_hf_config_0409/create.py"
            )
    logger.info("2604 mode: all config files match reference (bytewise equal)")
