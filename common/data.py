"""Benchmark datasets: registry, loading, prompt building, and scoring."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_CACHE_DIR = Path(os.environ.get("D3D_DATA_CACHE", REPO_ROOT / "data_cache"))
LOCAL_ANSWER_DIR = Path(os.environ.get("D3D_ANSWER_CACHE", REPO_ROOT / "answer_cache"))


def load_dataset_jsonl(repo_id, filename, revision='main'):
    from huggingface_hub import hf_hub_download

    source = Path(hf_hub_download(repo_id=repo_id, filename=filename, repo_type='dataset', revision=revision))
    content = source.read_bytes()
    rows = [json.loads(line) for line in content.decode().splitlines() if line.strip()]
    return rows, {'repo_id': repo_id, 'filename': filename, 'revision': revision,
                  'sha256': hashlib.sha256(content).hexdigest()}


def gsm8k_gold(row):
    return row["answer"].split("####")[-1].strip().replace(",", "")


def boxed(text):
    idx = text.rfind("\\boxed")
    if idx == -1:
        return None
    i = text.find("{", idx)
    if i == -1:
        return None
    depth, j = 0, i
    while j < len(text):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j].strip()
        j += 1
    return None


def humaneval_question(row):
    prompt = row.get("prompt", "")
    if prompt.startswith("Write a solution to the following problem"):
        return prompt
    return (
        "Write a solution to the following problem and make sure that it passes the tests:\n"
        f"```python\n{prompt}\n```"
    )


def livecodebench_question(row):
    if row.get('prompt'):
        return row['prompt']
    parts = ['Write a solution to the following problem and make sure that it passes the tests:']
    for field in ('question_title', 'question_content'):
        value = str(row.get(field) or '').replace('\r\n', '\n').strip()
        if value:
            parts.append(value)
    if row.get('starter_code'):
        parts.append(f"Starter code:\n```python\n{row['starter_code']}\n```")
    return '\n\n'.join(parts)


DATASETS = {
    "gsm8k": {
        "load": ("openai/gsm8k", "main", "test"),
        "question": lambda r: (
            r.get("prompt")
            or r["question"] + "\nPlease reason step by step, and put your final answer within \\boxed{}."
        ),
        "gold": gsm8k_gold,
        "score": "math",
    },
    "math500": {
        "load": ("HuggingFaceH4/MATH-500", None, "test"),
        "question": lambda r: (
            r.get("prompt")
            or r["problem"] + "\nPlease reason step by step, and put your final answer within \\boxed{}."
        ),
        "gold": lambda r: r["answer"].strip(),
        "score": "math",
    },
    "aime25": {
        "load": ("MathArena/aime_2025", None, "train"),
        "question": lambda r: (
            r.get("prompt")
            or r["problem"] + "\nPlease reason step by step, and put your final answer within \\boxed{}."
        ),
        # Some local prompt caches intentionally contain no answer file.
        # Keep generation/throughput evaluation available and score accuracy only
        # when a caller supplies an answer-enriched row.
        "gold": lambda r: (
            str(r["answer"]).strip() if r.get("answer") is not None else None
        ),
        "score": "math",
    },
    "humaneval": {
        "load": ("openai/openai_humaneval", None, "test"),
        "question": humaneval_question,
        "gold": None,
        "score": None,
    },
    "mbpp": {
        "load": ("google-research-datasets/mbpp", "sanitized", "test"),
        "question": lambda r: r.get("prompt") or r["text"],
        "gold": None,
        "score": None,
    },
    "mt-bench": {
        "load": ("HuggingFaceH4/mt_bench_prompts", None, "train"),
        "question": lambda r: (
            r.get("prompt")
            if isinstance(r.get("prompt"), str)
            else (r.get("prompt") or [""])[0]
        ),
        "gold": None,
        "score": None,
    },
    "livecodebench-v6": {
        "load": ("livecodebench/code_generation_lite", None, "test"),
        "filename": "test6.jsonl",
        "question": livecodebench_question,
        "gold": None,
        "score": None,
    },
}


def load_jsonl(path: Path):
    with path.open("r") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_local_dataset(dataset: str):
    cache_path = LOCAL_CACHE_DIR / f"{dataset}.jsonl"
    answer_path = LOCAL_ANSWER_DIR / f"{dataset}.jsonl"
    if not cache_path.exists():
        return None
    prompts = load_jsonl(cache_path)
    answers = load_jsonl(answer_path) if answer_path.exists() else [None] * len(prompts)
    if len(prompts) != len(answers):
        raise ValueError(f"local cache length mismatch: {cache_path}={len(prompts)} {answer_path}={len(answers)}")
    rows = []
    for idx, (prompt_row, answer_row) in enumerate(zip(prompts, answers)):
        turns = prompt_row.get("turns") or []
        if not turns:
            raise ValueError(f"missing turns in {cache_path}")
        row = dict(answer_row) if answer_row is not None else {}
        row.setdefault("dataset", dataset)
        row.setdefault("dataset_index", idx)
        row["prompt"] = turns[0]
        if dataset == "gsm8k" and answer_row is not None:
            row["gold"] = gsm8k_gold(answer_row)
        elif dataset == "math500" and answer_row is not None:
            row["gold"] = answer_row["answer"].strip()
        rows.append(row)
    return rows


def load_rows(dataset: str, max_samples: int | None):
    cfg = dict(DATASETS[dataset])
    rows = load_local_dataset(dataset)
    dataset_source = "local_cache"
    if rows is None:
        if cfg.get("load") is None:
            raise FileNotFoundError(f"missing local cache for {dataset}: {LOCAL_CACHE_DIR / f'{dataset}.jsonl'}")
        from datasets import load_dataset

        name, subset, split = cfg["load"]
        options = {'revision': os.environ.get('D3D_DATASET_REVISION', 'main')}
        if dataset == 'livecodebench-v6':
            rows, metadata = load_dataset_jsonl(name, cfg['filename'], options['revision'])
            cfg['fingerprint'] = metadata['sha256']
        else:
            loaded = load_dataset(name, subset, split=split, **options)
            cfg['fingerprint'] = loaded._fingerprint
            rows = list(loaded)
        cfg['revision'] = options['revision']
        dataset_source = "huggingface"
        for idx, row in enumerate(rows):
            row.setdefault("dataset", dataset)
            row.setdefault("dataset_index", idx)
    if max_samples and len(rows) > max_samples:
        rows = rows[:max_samples]
    cfg['source'] = dataset_source
    cfg['samples'] = len(rows)
    cfg['sample_sha256'] = hashlib.sha256(
        json.dumps(rows, sort_keys=True, default=str).encode()
    ).hexdigest()
    return rows, cfg, dataset_source


def extract_pred(text):
    boxed_value = boxed(text)
    if boxed_value is not None:
        return boxed_value.replace("$", "").strip()
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else None


def normalize(x):
    if x is None:
        return None
    # Commas are dropped on both sides so that "1,234" and "(3, 2)" compare
    # like-for-like with the gold answer.
    x = x.strip().rstrip(".").replace("\\!", "").replace(" ", "").replace(",", "")
    try:
        f = float(x)
        if not math.isfinite(f):
            return x
        return str(int(f)) if f == int(f) else str(f)
    except (ValueError, TypeError, OverflowError):
        return x


def score_text(cfg: dict[str, Any], text: str, gold: str | None):
    if cfg.get("score") != "math" or gold is None:
        return None, None
    pred = extract_pred(text)
    return pred, pred is not None and normalize(pred) == normalize(gold)


def gold_for_row(cfg: dict[str, Any], row):
    if "gold" in row:
        return row["gold"]
    gold_fn = cfg.get("gold")
    if not gold_fn:
        return None
    try:
        return gold_fn(row)
    except KeyError:
        return None


def build_prompt(question: str, enable_thinking: bool) -> dict[str, Any]:
    return {
        "messages": [{"role": "user", "content": question}],
        "enable_thinking": enable_thinking,
    }
