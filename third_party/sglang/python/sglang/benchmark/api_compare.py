#!/usr/bin/env python3
"""Benchmark any OpenAI-compatible chat endpoint and compare runs side-by-side.

Purpose: compare the official hosted APIs (ModelScope API-Inference, the
aliyuncs MaaS endpoint) against the local sglang deployment on the H800 under
identical workload conditions.

Standalone by design: stdlib only, so the same file runs on the local Mac
(against the public APIs) and on the remote H800 (against the local server).

Standard comparable workload (AGENTS.md section 4 serving baseline):
    512 input tokens / 256 output tokens, request rate 4/s, n=100.

Examples
--------
# 1) ModelScope API-Inference (official deployment)
mkdir -p docs/perf/logs docs/perf/results/api_compare
python3 -m sglang.benchmark.api_compare bench \\
    --label modelscope_qwen35_27b \\
    --preset modelscope \\
    --api-key "$MS_TOKEN" \\
    --model Qwen/Qwen3.5-27B \\
    2>&1 | tee docs/perf/logs/modelscope_qwen35_27b.log

# 2) local sglang server on the H800 (identical workload => comparable)
python3 -m sglang.benchmark.api_compare bench \\
    --label sglang_qwen35_27b \\
    --preset sglang-local \\
    --model /inspire/ssd/project/sais-bio/public/Chinese_civilization/models/Qwen/Qwen3.5-27B \\
    2>&1 | tee docs/perf/logs/sglang_qwen35_27b.log

# 3) side-by-side comparison
python3 -m sglang.benchmark.api_compare compare \\
    docs/perf/results/api_compare/modelscope_qwen35_27b.summary.json \\
    docs/perf/results/api_compare/sglang_qwen35_27b.summary.json \\
    --markdown docs/perf/reports/compare_qwen35_27b.md

Notes
-----
- API keys are never stored in this file: pass --api-key or set $MS_TOKEN /
  $MAAS_TOKEN. A local sglang server needs no key.
- Qwen3.5 is a hybrid-thinking model. Both backends accept
  chat_template_kwargs.enable_thinking; it defaults to false here so the two
  sides measure the fast path. Use --enable-thinking to compare thinking mode.
- Input token counts are approximated from the prompt text unless --tokenizer
  points at the model (exact counts, needs transformers in the environment).
- HTTP 429/5xx responses are retried with exponential backoff + jitter
  (--max-retries, --retry-base-delay): the ModelScope free tier enforces
  per-model rate and burst limits.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import re
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request

PRESETS = {
    "modelscope": "https://api-inference.modelscope.cn/v1",
    "sglang-local": "http://127.0.0.1:30000/v1",
    # the MaaS endpoint is per-deployment (llm-xxxx.cn-beijing.maas.aliyuncs.com);
    # pass --base-url explicitly.
}

# Common English words for deterministic pseudo-random prompts. Real words keep
# the word/token ratio stable (~1.0-1.3), so both endpoints tokenize the exact
# same prompt text to the exact same token count.
_WORDS = (
    "the quick brown fox jumps over lazy dog river mountain valley forest "
    "clouds rain snow wind fire earth water stone wood iron copper silver gold "
    "garden flower tree fruit apple orange grape lemon berry wheat rice bread "
    "butter cheese milk coffee tea sugar salt pepper spice honey cake pie "
    "morning evening night dawn dusk summer winter spring autumn day week "
    "month year time hour minute second moment instant future past history "
    "memory dream idea thought reason logic mind heart soul spirit energy "
    "power light shadow color shape form pattern texture surface edge corner "
    "circle square triangle line curve point space distance length height "
    "depth width angle slope wave pulse rhythm beat sound voice word "
    "letter number figure symbol mark sign signal message code key lock door "
    "window wall roof floor room house building street road bridge tower "
    "market square village city town nation world universe galaxy star moon "
    "planet orbit motion speed force mass weight balance scale measure gauge "
    "meter tool hand work craft skill art music dance song poem story novel "
    "chapter scene picture image photo film screen stage play theater "
    "actor role character hero friend family parent child brother "
    "sister neighbor stranger visitor guest host party festival holiday "
    "journey travel trip path map compass guide leader follower student "
    "teacher master novice expert beginner practice training study research "
    "experiment test result data fact evidence proof example case situation "
    "problem solution method process step stage phase level layer group team "
    "crowd audience public private secret open closed full empty hard soft "
    "strong weak fast slow high low deep shallow wide narrow thick thin heavy "
    "dark bright clear fresh old new young ancient modern recent early "
    "late first last next previous following remaining whole part half third "
    "quarter dozen hundred thousand million billion single double triple "
    "unique common special different similar same other another each every "
    "some many much more most least few enough plenty amount degree extent "
    "range scope field area region zone belt strip row column table chart "
    "graph book page paragraph section volume edition version "
    "copy draft final source origin cause effect impact influence outcome "
    "consequence advantage benefit risk danger safety security "
    "freedom choice decision option chance opportunity possibility probability"
).split()


def _percentile(values, p):
    """Linear-interpolated percentile of a list of numbers."""
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _make_prompt(rng, idx, target_words):
    n = max(8, target_words + rng.randint(-5, 5))
    words = [rng.choice(_WORDS) for _ in range(n)]
    sentences = []
    i = 0
    while i < len(words):
        end = min(i + rng.randint(8, 14), len(words))
        chunk = words[i:end]
        sentences.append(chunk[0].capitalize() + " " + " ".join(chunk[1:]) + ".")
        i = end
    return "[sample %d] " % idx + " ".join(sentences)


def generate_prompts(num_prompts, input_len, seed, prompt_file=None):
    """Deterministic prompts. input_len is the target prompt token count."""
    if prompt_file:
        with open(prompt_file, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        return [lines[i % len(lines)] for i in range(num_prompts)]
    rng = random.Random(seed)
    target_words = int(input_len / 1.15)
    return [_make_prompt(rng, i, target_words) for i in range(num_prompts)]


def _approx_prompt_tokens(text):
    # common English words: ~1.15 tokens per word (incl. punctuation)
    return int(len(text.split()) * 1.15)


def load_tokenizer(path):
    if not path:
        return None
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("WARNING: --tokenizer given but transformers is not installed; "
              "falling back to approximate token counts", file=sys.stderr)
        return None
    print("loading tokenizer from %s" % path, file=sys.stderr)
    return AutoTokenizer.from_pretrained(path)


def count_prompt_tokens(text, tokenizer):
    if tokenizer is not None:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])
    return _approx_prompt_tokens(text)


def build_request_body(model, prompt, max_tokens, enable_thinking, send_ctk):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if send_ctk:
        body["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    return body


def _one_request_once(base_url, api_key, model, prompt, max_tokens,
                      enable_thinking, send_ctk, timeout):
    """One streaming chat completion attempt; returns a record dict."""
    url = base_url.rstrip("/") + "/chat/completions"
    data = json.dumps(build_request_body(model, prompt, max_tokens,
                                         enable_thinking, send_ctk)).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    t0 = time.perf_counter()
    try:
        resp = urllib.request.urlopen(
            urllib.request.Request(url, data=data, headers=headers),
            timeout=timeout)
    except urllib.error.HTTPError as e:
        detail = e.read()[:500].decode("utf-8", "replace")
        return {"ok": False, "e2e_s": time.perf_counter() - t0,
                "ttft_s": None, "itl_ms": [], "content_tokens": 0,
                "reasoning_tokens": 0, "retries": 0,
                "error": "HTTP %s: %s" % (e.code, detail)}
    except Exception as e:  # URLError, socket.timeout, ...
        return {"ok": False, "e2e_s": time.perf_counter() - t0,
                "ttft_s": None, "itl_ms": [], "content_tokens": 0,
                "reasoning_tokens": 0, "retries": 0,
                "error": "%s: %s" % (type(e).__name__, e)}

    ttft_s = None
    prev = None
    itl_ms = []
    content_tokens = 0
    reasoning_tokens = 0
    finish_reason = None
    with resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            finish_reason = choices[0].get("finish_reason") or finish_reason
            has_token = False
            if delta.get("content"):
                content_tokens += 1
                has_token = True
            if delta.get("reasoning_content"):
                reasoning_tokens += 1
                has_token = True
            if has_token:
                now = time.perf_counter()
                if prev is None:
                    ttft_s = now - t0
                else:
                    itl_ms.append((now - prev) * 1000.0)
                prev = now
    e2e_s = time.perf_counter() - t0
    if not (content_tokens or reasoning_tokens):
        return {"ok": False, "e2e_s": e2e_s, "ttft_s": ttft_s, "itl_ms": [],
                "content_tokens": content_tokens, "reasoning_tokens": 0,
                "retries": 0,
                "error": "stream ended without tokens (finish_reason=%s)"
                         % finish_reason}
    return {"ok": True, "e2e_s": e2e_s, "ttft_s": ttft_s, "itl_ms": itl_ms,
            "content_tokens": content_tokens,
            "reasoning_tokens": reasoning_tokens, "retries": 0,
            "finish_reason": finish_reason}


def _error_http_code(error):
    m = re.match(r"HTTP (\d+)", error or "")
    return int(m.group(1)) if m else None


_RETRYABLE = (429, 500, 502, 503, 504)


def one_request(params, prompt, max_retries, retry_base_delay):
    """Streaming chat completion with bounded retries on 429/5xx.

    Public endpoints (ModelScope API-Inference free tier) enforce per-model
    rate and burst limits; backoff keeps a fixed-rate run above them usable.
    """
    core = {k: v for k, v in params.items() if k != "timeout"}
    rec = None
    for attempt in range(max_retries + 1):
        # retries get a capped per-attempt timeout so a hung stream cannot
        # multiply the total stall
        per_timeout = params["timeout"] if attempt == 0 else min(params["timeout"], 90.0)
        rec = _one_request_once(**core, prompt=prompt, timeout=per_timeout)
        rec["retries"] = attempt
        if rec["ok"]:
            return rec
        code = _error_http_code(rec["error"])
        if attempt >= max_retries or code not in _RETRYABLE:
            return rec
        delay = min(30.0, retry_base_delay * (2 ** attempt)) * (0.5 + random.random())
        print("  retry %d/%d in %.1fs (HTTP %s)"
              % (attempt + 1, max_retries, delay, code), file=sys.stderr)
        time.sleep(delay)
    return rec


def _worker(params, idx, prompt, not_before, semaphore, max_retries,
            retry_base_delay):
    delay = not_before - time.monotonic()
    if delay > 0:
        time.sleep(delay)
    with semaphore:
        return idx, one_request(params, prompt, max_retries, retry_base_delay)


def run_warmup(params, prompt, n, max_retries, retry_base_delay):
    for i in range(n):
        rec = one_request(params, prompt, max_retries, retry_base_delay)
        status = "ok" if rec["ok"] else "FAIL %s" % rec.get("error", "")
        print("warmup %d/%d: %s (%.2fs)" % (i + 1, n, status, rec["e2e_s"]),
              file=sys.stderr)
        if not rec["ok"]:
            print("WARNING: warmup failed; first measured requests may include "
                  "provider cold-start", file=sys.stderr)
            break


def aggregate(records, prompt_tokens, wall_s):
    ok = [r for r in records if r["ok"]]
    failed = len(records) - len(ok)
    e2e = [r["e2e_s"] * 1000.0 for r in ok]
    ttft = [r["ttft_s"] * 1000.0 for r in ok if r["ttft_s"] is not None]
    itl = [x for r in ok for x in r["itl_ms"]]
    out_tokens = [r["content_tokens"] + r["reasoning_tokens"] for r in ok]
    content = sum(r["content_tokens"] for r in ok)
    reasoning = sum(r["reasoning_tokens"] for r in ok)
    in_tok = sum(t for t, r in zip(prompt_tokens, records) if r["ok"])
    tpot = [(r["e2e_s"] - r["ttft_s"]) * 1000.0 / (n - 1)
            for r, n in zip(ok, out_tokens)
            if n > 1 and r["ttft_s"] is not None]
    wall = max(wall_s, 1e-9)

    def mean(xs):
        return statistics.fmean(xs) if xs else float("nan")

    return {
        "completed": len(ok),
        "failed": failed,
        "wall_s": round(wall, 3),
        "request_throughput_rps": round(len(ok) / wall, 3),
        "output_token_throughput_tps": round(sum(out_tokens) / wall, 2),
        "input_token_throughput_tps": round(in_tok / wall, 2),
        "total_token_throughput_tps": round((in_tok + sum(out_tokens)) / wall, 2),
        "mean_e2e_latency_ms": round(mean(e2e), 2),
        "median_e2e_latency_ms": round(statistics.median(e2e), 2) if e2e else float("nan"),
        "p99_e2e_latency_ms": round(_percentile(e2e, 99), 2),
        "mean_ttft_ms": round(mean(ttft), 2),
        "median_ttft_ms": round(statistics.median(ttft), 2) if ttft else float("nan"),
        "p99_ttft_ms": round(_percentile(ttft, 99), 2),
        "mean_itl_ms": round(mean(itl), 2),
        "median_itl_ms": round(statistics.median(itl), 2) if itl else float("nan"),
        "p99_itl_ms": round(_percentile(itl, 99), 2),
        "mean_tpot_ms": round(mean(tpot), 2),
        "mean_output_tokens": round(mean(out_tokens), 2),
        "output_tokens_content": content,
        "output_tokens_reasoning": reasoning,
        "prompt_tokens_approx_total": in_tok,
    }


def write_results(outdir, label, records, summary):
    os.makedirs(outdir, exist_ok=True)
    jsonl_path = os.path.join(outdir, label + ".jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in records:
            row = {k: v for k, v in rec.items() if k != "itl_ms"}
            row["itl_mean_ms"] = (statistics.fmean(rec["itl_ms"])
                                  if rec.get("itl_ms") else None)
            row["itl_max_ms"] = (max(rec["itl_ms"])
                                 if rec.get("itl_ms") else None)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_path = os.path.join(outdir, label + ".summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return jsonl_path, summary_path


def print_report(summary):
    m = summary["metrics"]
    w = summary["workload"]
    print("=" * 64)
    print("label:      %s" % summary["label"])
    print("model:      %s" % summary["model"])
    print("base_url:   %s" % summary["base_url"])
    print("workload:   n=%d rate=%s in~%d out<=%d thinking=%s seed=%d"
          % (w["num_prompts"], w["request_rate"], w["input_len"],
             w["output_len"], w["enable_thinking"], w["seed"]))
    print("Successful requests: %d (failed %d)" % (m["completed"], m["failed"]))
    print("Total wall time (s): %.2f" % m["wall_s"])
    print("Request throughput (req/s): %.3f" % m["request_throughput_rps"])
    print("Output token throughput (tok/s): %.2f" % m["output_token_throughput_tps"])
    print("Input token throughput (tok/s, approx): %.2f" % m["input_token_throughput_tps"])
    print("Total token throughput (tok/s, approx): %.2f" % m["total_token_throughput_tps"])
    print("-" * 64)
    print("Mean E2E Latency (ms): %.2f" % m["mean_e2e_latency_ms"])
    print("Median E2E Latency (ms): %.2f" % m["median_e2e_latency_ms"])
    print("P99 E2E Latency (ms): %.2f" % m["p99_e2e_latency_ms"])
    print("Mean TTFT (ms): %.2f" % m["mean_ttft_ms"])
    print("Median TTFT (ms): %.2f" % m["median_ttft_ms"])
    print("P99 TTFT (ms): %.2f" % m["p99_ttft_ms"])
    print("Mean ITL (ms): %.2f" % m["mean_itl_ms"])
    print("Median ITL (ms): %.2f" % m["median_itl_ms"])
    print("P99 ITL (ms): %.2f" % m["p99_itl_ms"])
    print("Mean TPOT (ms): %.2f" % m["mean_tpot_ms"])
    print("Output tokens (content/reasoning): %d/%d"
          % (m["output_tokens_content"], m["output_tokens_reasoning"]))
    print("=" * 64)


def cmd_bench(a):
    base_url = a.base_url or PRESETS.get(a.preset or "")
    if not base_url:
        sys.exit("error: --base-url is required (or use --preset modelscope|sglang-local)")
    api_key = a.api_key or os.environ.get("MS_TOKEN") \
        or os.environ.get("MAAS_TOKEN") or ""
    prompts = generate_prompts(a.num_prompts, a.input_len, a.seed, a.prompt_file)
    tokenizer = load_tokenizer(a.tokenizer)
    prompt_tokens = [count_prompt_tokens(p, tokenizer) for p in prompts]
    params = {
        "base_url": base_url,
        "api_key": api_key,
        "model": a.model,
        "max_tokens": a.output_len,
        "enable_thinking": a.enable_thinking,
        "send_ctk": not a.no_chat_template_kwargs,
        "timeout": a.timeout,
    }
    if a.warmup > 0:
        run_warmup(params, prompts[0], a.warmup, a.max_retries,
                   a.retry_base_delay)

    interval = (1.0 / a.request_rate) if a.request_rate > 0 else 0.0
    t0 = time.monotonic()
    sem = threading.Semaphore(a.max_concurrency)
    records = [None] * a.num_prompts
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(a.max_concurrency, a.num_prompts)) as ex:
        futs = {}
        for i, p in enumerate(prompts):
            futs[ex.submit(_worker, params, i, p, t0 + i * interval, sem,
                           a.max_retries, a.retry_base_delay)] = i
        for fut in concurrent.futures.as_completed(futs):
            i, rec = fut.result()
            records[i] = rec
            mark = "ok" if rec["ok"] else "FAIL %s" % rec.get("error", "")
            print("  [%3d/%d] %s out=%d ttft=%s e2e=%.2fs"
                  % (i + 1, a.num_prompts, mark,
                     rec["content_tokens"] + rec["reasoning_tokens"],
                     ("%.0fms" % (rec["ttft_s"] * 1000)) if rec.get("ttft_s") else "-",
                     rec["e2e_s"]),
                  file=sys.stderr)
    wall_s = time.monotonic() - t0
    metrics = aggregate(records, prompt_tokens, wall_s)
    summary = {
        "label": a.label,
        "model": a.model,
        "base_url": base_url,
        "api_key_source": "cli" if a.api_key else ("env" if api_key else "none"),
        "workload": {
            "num_prompts": a.num_prompts,
            "request_rate": a.request_rate,
            "input_len": a.input_len,
            "output_len": a.output_len,
            "seed": a.seed,
            "enable_thinking": a.enable_thinking,
            "max_concurrency": a.max_concurrency,
            "warmup": a.warmup,
        },
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "wall_s": round(wall_s, 3),
        "metrics": metrics,
    }
    jsonl_path, summary_path = write_results(a.outdir, a.label, records, summary)
    print_report(summary)
    print("per-request:  %s" % jsonl_path)
    print("summary:      %s" % summary_path)
    return 0 if metrics["failed"] == 0 else 2


_COMPARE_ROWS = (
    ("Successful requests", "completed", "{:d}"),
    ("Request throughput (req/s)", "request_throughput_rps", "{:.3f}"),
    ("Output token throughput (tok/s)", "output_token_throughput_tps", "{:.2f}"),
    ("Input token throughput (tok/s, approx)", "input_token_throughput_tps", "{:.2f}"),
    ("Total token throughput (tok/s, approx)", "total_token_throughput_tps", "{:.2f}"),
    ("Mean E2E Latency (ms)", "mean_e2e_latency_ms", "{:.2f}"),
    ("Median E2E Latency (ms)", "median_e2e_latency_ms", "{:.2f}"),
    ("P99 E2E Latency (ms)", "p99_e2e_latency_ms", "{:.2f}"),
    ("Mean TTFT (ms)", "mean_ttft_ms", "{:.2f}"),
    ("Median TTFT (ms)", "median_ttft_ms", "{:.2f}"),
    ("P99 TTFT (ms)", "p99_ttft_ms", "{:.2f}"),
    ("Mean ITL (ms)", "mean_itl_ms", "{:.2f}"),
    ("Median ITL (ms)", "median_itl_ms", "{:.2f}"),
    ("P99 ITL (ms)", "p99_itl_ms", "{:.2f}"),
    ("Mean TPOT (ms)", "mean_tpot_ms", "{:.2f}"),
    ("Mean output tokens", "mean_output_tokens", "{:.2f}"),
)


def cmd_compare(a):
    summaries = []
    for path in a.files:
        with open(path, "r", encoding="utf-8") as f:
            summaries.append(json.load(f))
    labels = [s["label"] for s in summaries]
    lines = []
    header = ("metric",) + tuple(labels)
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for name, key, fmt in _COMPARE_ROWS:
        cells = [name]
        for s in summaries:
            v = s["metrics"].get(key)
            cells.append(fmt.format(v) if isinstance(v, (int, float)) else "-")
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    for s in summaries:
        w = s["workload"]
        lines.append("")
        lines.append("*%s*: model=`%s` base_url=`%s` n=%d rate=%s in~%d out<=%d "
                     "thinking=%s wall=%.1fs"
                     % (s["label"], s["model"], s["base_url"], w["num_prompts"],
                        w["request_rate"], w["input_len"], w["output_len"],
                        w["enable_thinking"], s["wall_s"]))
    text = "\n".join(lines)
    print(text)
    if a.markdown:
        with open(a.markdown, "w", encoding="utf-8") as f:
            f.write("# API comparison\n\n" + text + "\n")
        print("\nwrote %s" % a.markdown)
    return 0


def cmd_list_models(a):
    url = a.base_url.rstrip("/") + "/models"
    headers = {"Content-Type": "application/json"}
    api_key = a.api_key or os.environ.get("MS_TOKEN") \
        or os.environ.get("MAAS_TOKEN") or ""
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=a.timeout) as resp:
        data = json.loads(resp.read().decode("utf-8", "replace"))
    ids = sorted(m["id"] for m in data.get("data", []))
    rx = re.compile(a.filter) if a.filter else None
    for i in ids:
        if rx is None or rx.search(i):
            print(i)
    print("%d models" % len(ids), file=sys.stderr)
    return 0


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bench", help="benchmark one OpenAI-compatible endpoint")
    b.add_argument("--label", required=True,
                   help="run label, used for output file names")
    b.add_argument("--base-url", default=None,
                   help="OpenAI-compatible base url (.../v1); overrides --preset")
    b.add_argument("--preset", choices=sorted(PRESETS), default=None,
                   help="base-url preset")
    b.add_argument("--api-key", default=None,
                   help="API key (default: $MS_TOKEN, then $MAAS_TOKEN)")
    b.add_argument("--model", required=True, help="model id sent in the request")
    b.add_argument("--num-prompts", type=int, default=100)
    b.add_argument("--request-rate", type=float, default=4.0,
                   help="req/s; 0 = all at once")
    b.add_argument("--input-len", type=int, default=512,
                   help="target prompt token count")
    b.add_argument("--output-len", type=int, default=256,
                   help="max_tokens cap per request")
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--warmup", type=int, default=3,
                   help="sequential warmup requests, excluded from metrics")
    b.add_argument("--max-concurrency", type=int, default=128)
    b.add_argument("--enable-thinking", action="store_true",
                   help="set chat_template_kwargs.enable_thinking=true")
    b.add_argument("--no-chat-template-kwargs", action="store_true",
                   help="do not send chat_template_kwargs (use backend defaults)")
    b.add_argument("--prompt-file", default=None,
                   help="one prompt per line, cycled (overrides generated prompts)")
    b.add_argument("--tokenizer", default=None,
                   help="local tokenizer dir/model path for exact prompt tokens")
    b.add_argument("--timeout", type=float, default=300.0)
    b.add_argument("--max-retries", type=int, default=5,
                   help="retries per request on HTTP 429/5xx")
    b.add_argument("--retry-base-delay", type=float, default=2.0,
                   help="base delay (s) for exponential backoff with jitter")
    b.add_argument("--outdir", default="docs/perf/results/api_compare")
    b.set_defaults(func=cmd_bench)

    c = sub.add_parser("compare", help="side-by-side compare summary files")
    c.add_argument("files", nargs="+", help="*.summary.json files from bench runs")
    c.add_argument("--markdown", default=None, help="also write a markdown report")
    c.set_defaults(func=cmd_compare)

    l = sub.add_parser("list-models", help="list models on an endpoint")
    l.add_argument("--base-url", required=True)
    l.add_argument("--api-key", default=None)
    l.add_argument("--filter", default=None, help="regex filter on model id")
    l.add_argument("--timeout", type=float, default=60.0)
    l.set_defaults(func=cmd_list_models)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
