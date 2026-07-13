"""
共享工具函数：所有 run_*.py 脚本都 import 这个模块。
负责：输入格式检测、chat 调用、答案提取、MRCR 输出格式。
"""

import json
import os
import re
import time

from openai import OpenAI

SERVER_URL  = os.environ.get("SGLANG_SERVER_URL", "http://localhost:30000/v1")
# 需与 server 加载的模型一致；用 MODEL_PATH 环境变量覆盖，例如 path/to/ds_fp8
MODEL_PATH  = os.environ.get("MODEL_PATH", "path/to/ds_fp8")
CMD_FILE    = "/tmp/dsv4_tracker_cmd.json"
RESULT_FILE = "/tmp/dsv4_tracker_result.json"
MAX_TOKENS  = 512
TEMPERATURE = 0.0

client = OpenAI(base_url=SERVER_URL, api_key="EMPTY", timeout=3600.0)


# ── 输入格式检测 ────────────────────────────────────────────────────────────

def load_items(input_path: str) -> tuple[list, bool]:
    """
    加载输入文件，自动检测格式。
    返回 (items, is_mrcr)
      - JSON list (旧格式): [{"prompt": "...", "answer": "A"}, ...]
      - JSONL (MRCR 格式): 每行 {"prompt": "<json-encoded messages>", "answer": "...", "random_string_to_prepend": "...", ...}
    """
    if input_path.endswith(".jsonl"):
        items = []
        with open(input_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        is_mrcr = "random_string_to_prepend" in items[0] if items else False
        return items, is_mrcr
    else:
        with open(input_path, encoding="utf-8") as f:
            items = json.load(f)
        return items, False


def get_messages(item: dict, is_mrcr: bool, thinking: bool) -> list:
    """
    从 item 构建 OpenAI messages 列表。
    - MRCR: prompt 是 JSON 字符串化的 messages，直接 parse
    - 普通: prompt 是纯文本，包装成 [{"role":"user","content":...}]
    """
    if is_mrcr:
        messages = json.loads(item["prompt"])
        return messages
    else:
        prompt = item["prompt"]
        if thinking:
            prompt += "\n\nPlease reason step by step, and put your final answer in \\boxed{}."
        else:
            prompt += "\n\nPlease put your final answer in \\boxed{}."
        return [{"role": "user", "content": prompt}]


# ── Chat 调用 ────────────────────────────────────────────────────────────────

def chat(messages: list, thinking: bool = False) -> str:
    kwargs = dict(
        model=MODEL_PATH,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        stream=False,
    )
    if thinking:
        kwargs["extra_body"] = {"chat_template_kwargs": {"thinking": True}}
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content


# ── 答案提取 ──────────────────────────────────────────────────────────────────

def extract_pred(completion: str, gold_answer: str = None) -> str:
    """
    从模型输出中提取预测答案。
    如果有 </think>，只看 </think> 后面的内容。
    优先从 \\boxed{} 中提取；否则提取第一个独立的 A/B/C/D。
    """
    text = completion
    if '</think>' in text:
        text = text.split('</think>', 1)[1]

    boxed = re.findall(r'\\boxed\{([^}]*)\}', text)
    if boxed:
        return boxed[-1].strip()

    m = re.search(r'(?<![a-zA-Z])([A-D])(?![a-zA-Z])', text)
    if m:
        return m.group(1)
    return None


def check_correct(pred, gold_answer) -> bool:
    if gold_answer is None or pred is None:
        return None
    return str(pred) == str(gold_answer)


# ── MRCR 输出 ────────────────────────────────────────────────────────────────

def write_mrcr_response(fout, item: dict, response: str, suffix: str = ""):
    """
    写一行 MRCR 格式的 JSONL 输出，兼容 eval_mrcr.py 的输入格式。
    """
    out = {
        "response": response,
        "answer": item["answer"],
        "random_string_to_prepend": item["random_string_to_prepend"],
        "n_needles": item.get("n_needles"),
    }
    fout.write(json.dumps(out, ensure_ascii=False) + "\n")


def mrcr_output_paths(output: str) -> tuple[str, str]:
    """
    从用户指定的 --output 路径生成 pass1/pass2 两个文件名。
    例如: mrcr_result.response.jsonl
       → mrcr_result.pass1.response.jsonl
       → mrcr_result.pass2.response.jsonl
    """
    base, ext = os.path.splitext(output)
    return base + ".pass1" + ext, base + ".pass2" + ext


# ── 文件通信 ──────────────────────────────────────────────────────────────────

def write_cmd(cmd: dict):
    with open(CMD_FILE, "w") as f:
        json.dump(cmd, f)


def clear_result():
    try:
        os.remove(RESULT_FILE)
    except FileNotFoundError:
        pass


def wait_result(timeout: float = 300.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(RESULT_FILE):
            try:
                with open(RESULT_FILE) as f:
                    return json.load(f)
            except Exception:
                pass
        time.sleep(0.2)
    return {}
