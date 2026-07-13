"""
Retriever 训练数据生产脚本（并发版）
=====================================
并发发送推理请求，server 端 per-request 隔离状态，自动 flush 到磁盘。
第二轮发送 prompt+response 获取完整 compressed K。

用法：
  # 输入用 formated jsonl（每行含 prompt/answer/random_string_to_prepend），输出用绝对路径
  python run_dump_training_data.py \
    --input  path/to/data/creative_writing_multiturn_filtered.jsonl \
    --output-dir path/to/output/creative_writing \
    --start-idx 0 --end-idx 10 \
    --thinking --topp 0.6 --min-layers 3 --concurrency 32 --batch-size 100

模型路径通过环境变量 MODEL_PATH 覆盖（需与 server 加载的模型一致）：
  MODEL_PATH=path/to/ds_fp8 python run_dump_training_data.py ...
"""

import argparse
import json
import os
import time
import threading
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from experiment_utils import write_cmd, clear_result, load_items, get_messages

SERVER_URL = os.environ.get("SGLANG_SERVER_URL", "http://localhost:30000/v1")
# 需与 server 加载的模型一致；用 MODEL_PATH 环境变量覆盖，例如 path/to/ds_fp8
MODEL_PATH = os.environ.get("MODEL_PATH", "path/to/ds_fp8")
MAX_TOKENS = 2048
TEMPERATURE = 0.0

client = OpenAI(base_url=SERVER_URL, api_key="EMPTY", timeout=3600.0)


def send_one_request(messages, thinking, max_tokens=MAX_TOKENS):
    """发送一个推理请求，返回 (生成文本, rid)。"""
    kwargs = dict(
        model=MODEL_PATH,
        messages=messages,
        max_tokens=max_tokens,
        temperature=TEMPERATURE,
        stream=False,
    )
    if thinking:
        kwargs["extra_body"] = {"chat_template_kwargs": {"thinking": True}}
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content, resp.id


def main():
    parser = argparse.ArgumentParser(description="Retriever 训练数据生产（并发版）")
    parser.add_argument("--input",           required=True, help="JSON 或 JSONL 输入文件")
    parser.add_argument("--output-dir",      default="offline_retriever_data", help="输出目录")
    parser.add_argument("--topp",            type=float, default=0.6, help="Top-P 阈值")
    parser.add_argument("--min-layers",      type=int, default=3, help="chunk 必须在 ≥ N 层通过 top-p 才记为 golden")
    parser.add_argument("--target-csa-layers", type=int, nargs='+', default=[6, 8, 10, 12, 14, 16, 18, 20], help="记录第几个 CSA layers 的数据 (0-indexed)")
    parser.add_argument("--n-samples",       type=int, default=None, help="只取前 N 个文档")
    parser.add_argument("--start-idx",       type=int, default=0, help="数据起始索引（含）")
    parser.add_argument("--end-idx",         type=int, default=None, help="数据结束索引（不含），如 --start-idx 20 --end-idx 60 表示 data[20:60]")
    parser.add_argument("--batch-size",      type=int, default=10, help="每批处理的文档数（每批完成一轮+二轮+合并）")
    parser.add_argument("--thinking",        action="store_true", help="开启 thinking 模式")
    parser.add_argument("--concurrency",     type=int, default=4, help="并发请求数")
    args = parser.parse_args()

    items, is_mrcr = load_items(args.input)
    # 支持指定数据范围：--start-idx 20 --end-idx 60 → items[20:60]
    if args.start_idx > 0 or args.end_idx is not None:
        end = args.end_idx if args.end_idx is not None else len(items)
        items = items[args.start_idx:end]
        print(f"数据范围: [{args.start_idx}:{end}]")
    elif args.n_samples is not None:
        items = items[:args.n_samples]

    os.makedirs(args.output_dir, exist_ok=True)

    # 自动检测已有 metadata，确定 doc_counter 起始值（支持多次追加生成）
    meta_path_existing = os.path.join(args.output_dir, "server_metadata.jsonl")
    doc_counter_start = 0
    if os.path.exists(meta_path_existing):
        with open(meta_path_existing) as f:
            existing_lines = f.readlines()
        if existing_lines:
            # 找到已有的最大 doc_index + 1 作为起始
            max_idx = max(json.loads(line).get("doc_index", 0) for line in existing_lines)
            doc_counter_start = max_idx + 1
            print(f"检测到已有 {len(existing_lines)} 条记录，doc_counter 从 {doc_counter_start} 开始（追加模式）")

    print(f"共 {len(items)} 个文档，并发数={args.concurrency}，输出到 {args.output_dir}/")
    print(f"参数: topp={args.topp}, min_layers={args.min_layers}, target_csa_layers={args.target_csa_layers}")

    # ========== 按 batch 分批处理（每 batch_size 条做一轮完整的 第一轮+第二轮+合并）==========
    BATCH_SIZE = args.batch_size
    t0 = time.time()
    total_completed = 0
    total_failed = 0

    for batch_start in range(0, len(items), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(items))
        batch_items = items[batch_start:batch_end]
        batch_num = batch_start // BATCH_SIZE + 1
        n_batches = (len(items) + BATCH_SIZE - 1) // BATCH_SIZE

        print(f"\n{'='*60}")
        print(f"Batch {batch_num}/{n_batches}: items[{batch_start}:{batch_end}] ({len(batch_items)} 条)")
        print(f"{'='*60}")

        # 重新检测 doc_counter（每批开始前更新）
        meta_path = os.path.join(args.output_dir, "server_metadata.jsonl")
        current_doc_counter = doc_counter_start
        n_existing = 0
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                existing = f.readlines()
            n_existing = len(existing)
            if existing:
                current_doc_counter = max(json.loads(l).get("doc_index", 0) for l in existing) + 1
        print(f"  doc_counter_start={current_doc_counter} (metadata 已有 {n_existing} 条)")

        # ===== 第一轮：正常生成 =====
        clear_result()
        write_cmd({
            "mode": "pass1_dump_training",
            "topp": args.topp,
            "min_layers": args.min_layers,
            "target_csa_layers": args.target_csa_layers,
            "output_dir": args.output_dir,
            "doc_counter_start": current_doc_counter,
        })

        n_completed = 0
        n_failed = 0
        lock = threading.Lock()
        rid_to_index = {}
        rid_to_response = {}

        def process_one(idx, item):
            nonlocal n_completed, n_failed
            messages = get_messages(item, is_mrcr, args.thinking)
            try:
                text, rid = send_one_request(messages, args.thinking)
                with lock:
                    n_completed += 1
                    rid_to_index[rid] = idx
                    rid_to_response[rid] = text
                    if n_completed % 20 == 0 or n_completed == len(batch_items):
                        print(f"    [{n_completed}/{len(batch_items)}] 完成")
            except Exception as e:
                with lock:
                    n_failed += 1
                    print(f"    [ERROR] idx {idx}: {e}")

        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = [executor.submit(process_one, i, item) for i, item in enumerate(batch_items)]
            for f in as_completed(futures):
                pass

        print(f"  第一轮: {n_completed} 成功, {n_failed} 失败")
        total_completed += n_completed
        total_failed += n_failed

        # flush 第一轮
        write_cmd({"mode": "none"})
        for _ in range(5):
            try:
                send_one_request([{"role": "user", "content": "hi"}], thinking=False, max_tokens=1)
            except Exception:
                pass
            time.sleep(1)

        # 验证 metadata 已写入（重试等待，最多再等 10 秒）
        _new_count = 0
        for _retry in range(10):
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    _check_lines = f.readlines()
                _new_count = sum(1 for l in _check_lines if json.loads(l).get("doc_index", -1) >= current_doc_counter)
                if _new_count >= n_completed:
                    break
            time.sleep(1)
        else:
            print(f"  [WARN] 等待 metadata 超时: 期望 {n_completed} 条 doc_index>={current_doc_counter}, 实际 {_new_count}")

        # 读取本批的 metadata
        meta_lines_all = []
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta_lines_all = f.readlines()

        # 只处理本批新增的 entries（doc_index >= current_doc_counter）
        batch_meta_lines = []
        for line in meta_lines_all:
            entry = json.loads(line)
            if entry.get("doc_index", -1) >= current_doc_counter:
                batch_meta_lines.append(line)

        print(f"  [DEBUG] metadata 总行数={len(meta_lines_all)}, 本批匹配={len(batch_meta_lines)} (filter: doc_index>={current_doc_counter})")
        if len(batch_meta_lines) == 0 and len(meta_lines_all) > 0:
            # 打印 metadata 中所有 doc_index 帮助诊断
            all_doc_indices = [json.loads(l).get("doc_index", -1) for l in meta_lines_all]
            print(f"  [DEBUG] metadata 中 doc_index 列表: {all_doc_indices}")
        if len(batch_meta_lines) > 0:
            sample_meta_rids = [json.loads(l).get("rid", "?")[:12] for l in batch_meta_lines[:3]]
            sample_client_rids = list(rid_to_index.keys())[:3]
            sample_client_rids_short = [r[:12] for r in sample_client_rids]
            print(f"  [DEBUG] metadata rids (前3): {sample_meta_rids}")
            print(f"  [DEBUG] client rids  (前3): {sample_client_rids_short}")

        # 保存 .json
        n_json_saved = 0
        for line in batch_meta_lines:
            entry = json.loads(line)
            rid = entry.get("rid")
            doc_file = entry.get("filename", "")
            if rid in rid_to_index and doc_file:
                original_idx = rid_to_index[rid]
                messages = get_messages(batch_items[original_idx], is_mrcr, args.thinking)
                json_path = os.path.join(args.output_dir, doc_file.replace(".pkl", ".json"))
                with open(json_path, "w", encoding="utf-8") as jf:
                    json.dump({
                        "prompt": messages,
                        "response": rid_to_response.get(rid, ""),
                        "original_index": batch_start + original_idx,
                        "rid": rid,
                    }, jf, ensure_ascii=False)
                n_json_saved += 1
        print(f"  保存 {n_json_saved} 个 .json 文件")

        # ===== 第二轮：获取完整 compressed K =====
        print(f"  第二轮: 获取完整 compressed K...")
        compk_output_dir = args.output_dir + "_compk_full"
        if os.path.exists(compk_output_dir):
            import shutil
            shutil.rmtree(compk_output_dir)
        os.makedirs(compk_output_dir, exist_ok=True)

        write_cmd({
            "mode": "pass1_dump_training",
            "topp": args.topp,
            "min_layers": args.min_layers,
            "target_csa_layers": args.target_csa_layers,
            "output_dir": compk_output_dir,
            "doc_counter_start": current_doc_counter,
        })

        n_compk_done = 0
        compk_order = []       # [(local_idx, response), ...]
        compk_r1_files = []    # Round 1 对应的 pkl 文件名（按处理顺序）
        n_rid_matched = 0
        n_rid_missed = 0
        for line in batch_meta_lines:
            entry = json.loads(line)
            rid = entry.get("rid")
            if rid in rid_to_index and rid in rid_to_response:
                compk_order.append((rid_to_index[rid], rid_to_response[rid]))
                compk_r1_files.append(entry.get("filename", ""))
                n_rid_matched += 1
            else:
                n_rid_missed += 1
                if n_rid_missed <= 3:
                    print(f"  [DEBUG] rid 未匹配: {rid[:16]}... (in rid_to_index={rid in rid_to_index}, in rid_to_response={rid in rid_to_response})")

        print(f"  [DEBUG] compk_order: {len(compk_order)} 条 (matched={n_rid_matched}, missed={n_rid_missed})")

        for i, (idx, resp) in enumerate(compk_order):
            messages = get_messages(batch_items[idx], is_mrcr, args.thinking)
            full_messages = messages + [{"role": "assistant", "content": resp}]
            try:
                kwargs = dict(
                    model=MODEL_PATH,
                    messages=full_messages,
                    max_tokens=1,
                    temperature=TEMPERATURE,
                    stream=False,
                )
                if args.thinking:
                    kwargs["extra_body"] = {"chat_template_kwargs": {"thinking": True}}
                client.chat.completions.create(**kwargs)
                with lock:
                    n_compk_done += 1
            except Exception as e:
                print(f"    [ERROR] compk idx {idx}: {e}")

        # flush 第二轮
        write_cmd({"mode": "none"})
        for _ in range(5):
            try:
                send_one_request([{"role": "user", "content": "hi"}], thinking=False, max_tokens=1)
            except Exception:
                pass
            time.sleep(1)
        print(f"  第二轮: {n_compk_done} 个 compk")

        # ===== 合并（按处理顺序匹配 Round 1 和 Round 2）=====
        compk_meta_path = os.path.join(compk_output_dir, "server_metadata.jsonl")
        if os.path.exists(compk_meta_path):
            with open(compk_meta_path) as f:
                compk_meta_lines = f.readlines()

            # 重新读取完整 metadata
            with open(meta_path) as f:
                meta_lines_all = f.readlines()

            # Round 2 是串行的，compk_meta_lines[i] 对应 compk_order[i] 对应 compk_r1_files[i]
            # 建立 Round 1 filename → 行号 的索引
            r1_file_to_line_idx = {}
            for li, line in enumerate(meta_lines_all):
                entry = json.loads(line)
                r1_file_to_line_idx[entry.get("filename")] = li

            n_merged = 0
            updated_meta_lines = list(meta_lines_all)

            for ci, line in enumerate(compk_meta_lines):
                compk_entry = json.loads(line)
                compk_file = compk_entry.get("filename")
                compk_path = os.path.join(compk_output_dir, compk_file) if compk_file else None

                # 通过处理顺序找到 Round 1 对应的文件
                if ci < len(compk_r1_files):
                    r1_filename = compk_r1_files[ci]
                    line_idx = r1_file_to_line_idx.get(r1_filename)
                else:
                    line_idx = None

                if line_idx is not None:
                    first_entry = json.loads(meta_lines_all[line_idx])
                    first_file = first_entry.get("filename")
                    first_path = os.path.join(args.output_dir, first_file) if first_file else None

                    if first_path and compk_path and os.path.exists(first_path) and os.path.exists(compk_path):
                        with open(first_path, "rb") as pf:
                            data = pickle.load(pf)
                        with open(compk_path, "rb") as cf:
                            compk_data = pickle.load(cf)
                        for key in list(compk_data.keys()):
                            if key.startswith("compk_layer_"):
                                data[key] = compk_data[key]
                        with open(first_path, "wb") as pf:
                            pickle.dump(data, pf)
                        n_merged += 1

                        first_entry["compk_shapes"] = compk_entry.get("compk_shapes", {})
                        updated_meta_lines[line_idx] = json.dumps(first_entry, ensure_ascii=False) + "\n"

            with open(meta_path, "w") as f:
                f.writelines(updated_meta_lines)

            print(f"  合并: {n_merged} 个文件")
        else:
            print(f"  [WARNING] 第二轮 metadata 不存在，跳过合并")

        elapsed_batch = time.time() - t0
        print(f"  Batch {batch_num} 完成, 累计耗时 {elapsed_batch:.0f}s")

    # ========== 摘要 ==========
    total_elapsed = time.time() - t0
    meta_path = os.path.join(args.output_dir, "server_metadata.jsonl")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            final_meta = f.readlines()
        # 按 doc_index 排序 metadata（并发下写入顺序可能乱序）
        final_meta_sorted = sorted(final_meta, key=lambda l: json.loads(l).get("doc_index", 0))
        with open(meta_path, "w") as f:
            f.writelines(final_meta_sorted)
        print(f"\n最终数据: {len(final_meta_sorted)} 个文档 (已按 doc_index 排序)")

    print(f"\n输出目录: {args.output_dir}/")
    print(f"总耗时: {total_elapsed:.1f}s, 成功: {total_completed}, 失败: {total_failed}")


if __name__ == "__main__":
    main()
