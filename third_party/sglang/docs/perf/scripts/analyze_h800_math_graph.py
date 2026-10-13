#!/usr/bin/env python3
"""Independently validate and summarize the matched C1 math graph benchmark."""

import argparse
import gzip
import hashlib
import json
import math
import random
import statistics
from pathlib import Path


def read(path):
    if not path.exists() and path.with_suffix(path.suffix + ".gz").exists():
        path = path.with_suffix(path.suffix + ".gz")
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as source:
            return json.load(source)
    return json.loads(path.read_text())


def measurement_paths(directory):
    paths = {p.name.removesuffix(".gz"): p for p in directory.glob("c1_r*.json.gz")}
    paths.update({p.name: p for p in directory.glob("c1_r*.json")})
    return [paths[name] for name in sorted(paths)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--natural-eos", action="store_true")
    parser.add_argument("--output-file", type=Path)
    args = parser.parse_args()
    result = {}
    for dataset in ("math500", "gsm8k"):
        directory = args.archive / dataset
        provenance = read(directory / "dataset_manifest.json")
        dataset_directory = directory
        if args.natural_eos:
            assert hashlib.sha256(Path(provenance["sample_path"]).read_bytes()).hexdigest() == provenance["sample_sha256"]
            dataset_directory = Path(provenance["fixed_source_manifest"]).parent
            provenance = read(Path(provenance["fixed_source_manifest"]))
        assert hashlib.sha256((dataset_directory / "test_source.jsonl").read_bytes()).hexdigest() == provenance["source_sha256"]
        assert hashlib.sha256((dataset_directory / "sample.jsonl").read_bytes()).hexdigest() == provenance["sample_sha256"]
        selected = provenance["measured_source_indices"]
        assert len(selected) == len(set(selected)) == 32
        assert provenance["warmup_source_index"] not in selected
        source_rows = [json.loads(line) for line in (dataset_directory / "test_source.jsonl").read_text().splitlines()]
        sample_rows = [json.loads(line) for line in (dataset_directory / "sample.jsonl").read_text().splitlines()]
        indices = selected + [provenance["warmup_source_index"]]
        assert indices == random.Random(provenance["seed"]).sample(range(len(source_rows)), 33)
        for index, sample in zip(indices, sample_rows, strict=True):
            original = source_rows[index]
            assert sample == {"source_index": index, "problem": original.get("problem", original.get("question"))}
        reference = read(directory / "ar_reference.json")
        model = Path(reference["configuration"]["model_path"])
        archived_config = args.archive / "model_generation_config.json"
        eos = read(archived_config if archived_config.exists() else model / "generation_config.json")["eos_token_id"]
        eos = {eos} if isinstance(eos, int) else set(eos)
        first_eos = [next((i + 1 for i, token in enumerate(row["output_ids"]) if token in eos), None)
                     for row in reference["requests"]]
        expected = {r["index"]: r["output_ids"] for r in reference["requests"]}
        assert len(expected) == 32
        source_hash = read(directory / "ar_source_sha256.json")
        groups = {}
        for mode in ("ar", "eager", "graph"):
            assert read(directory / f"{mode}_source_sha256.json") == source_hash
            for filename, digest in source_hash.items():
                assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == digest, filename
            paths = measurement_paths(directory / mode)
            assert len(paths) == 3, (dataset, mode, len(paths))
            payloads = [read(p) for p in paths]
            speeds, metrics, run_medians = [], [], []
            for path, payload in zip(paths, payloads):
                assert payload["summary"]["valid"], path
                assert payload["experiment"]["performance_valid"], path
                assert payload["prompt_digest"] == reference["prompt_digest"], path
                assert payload["configuration"]["concurrency"] == 1
                assert payload["configuration"]["output_len"] == (2048 if args.natural_eos else 512)
                if args.natural_eos:
                    assert payload["configuration"]["ignore_eos"] is False
                rows = payload["requests"]
                assert sorted(r["index"] for r in rows) == list(range(32))
                for row in rows:
                    assert not row["error"]
                    assert row["output_ids"] == expected[row["index"]], (path, row["index"])
                    assert row["output_tokens"] == len(row["output_ids"])
                    if args.natural_eos:
                        assert 0 < row["output_tokens"] <= 2048
                        expected_row = reference["requests"][row["index"]]
                        assert row["finish_reason"] == expected_row["finish_reason"]
                        assert row["finish_reason"]["type"] in {"stop", "length"}
                        if row["finish_reason"]["type"] == "length":
                            assert row["output_tokens"] == 2048
                    else:
                        assert row["output_tokens"] == 512
                    assert row["latency"] > 0
                    assert math.isclose(row["output_tokens_per_second"], row["output_tokens"] / row["latency"])
                values = [row["output_tokens"] / row["latency"] for row in rows]
                median = statistics.median(values)
                assert math.isclose(median, payload["summary"]["per_request_output_tokens_per_second_p50"])
                speeds.extend(values)
                run_medians.append(median)
                m = payload["metrics"]["client_delta"]
                if mode != "ar":
                    calls = m["local_required_forward_ct"]
                    assert calls == m.get("local_required_cuda_graph_ct", 0) + m.get("local_required_eager_ct", 0)
                    assert m["local_required_req_ct"] + m["remote_optional_skip_local_ct"] == payload["summary"]["spec_verify_ct"]
                    assert m["remote_optional_full_accept_hit_ct"] == m["remote_optional_full_accept_skip_local_ct"]
                    if mode == "eager":
                        assert m.get("local_required_cuda_graph_ct", 0) == 0
                    metrics.append(m)
                print("PASS", path)
            group = {
                "checked_requests": len(speeds), "pooled_request_speed_p50": statistics.median(speeds),
                "run_p50": run_medians, "run_range": [min(run_medians), max(run_medians)],
                "median_decode_speed": statistics.median(r["decode_tokens_per_second"] for p in payloads for r in p["requests"]),
                "median_ttft_ms": statistics.median(r["ttft_seconds"] * 1000 for p in payloads for r in p["requests"]),
                "output_tokens_p50": statistics.median(r["output_tokens"] for p in payloads for r in p["requests"]),
                "total_output_tokens": sum(r["output_tokens"] for p in payloads for r in p["requests"]),
            }
            if metrics:
                total = lambda key: sum(m.get(key, 0) for m in metrics)
                calls = total("local_required_forward_ct")
                verify = sum(p["summary"]["spec_verify_ct"] for p in payloads)
                group.update({
                    "local_calls": calls, "graph_calls": total("local_required_cuda_graph_ct"),
                    "eager_calls": total("local_required_eager_ct"),
                    "local_mean_ms": 1000 * total("local_required_seconds") / calls,
                    "verify_ct": verify, "output_tokens_per_verify": group["total_output_tokens"] / verify,
                    "full_hits": total("remote_optional_full_accept_hit_ct"),
                    "full_lookups": total("remote_optional_full_accept_lookup_ct"),
                    "remote_skip_local_ct": total("remote_optional_skip_local_ct"),
                    "requests_with_eager_calls": sum(r["client_delta"].get("local_required_eager_ct", 0) > 0
                                                     for p in payloads for r in p["requests"]),
                })
            groups[mode] = group
        groups["graph_vs_ar_ratio"] = groups["graph"]["pooled_request_speed_p50"] / groups["ar"]["pooled_request_speed_p50"]
        groups["graph_vs_eager_percent"] = 100 * (groups["graph"]["pooled_request_speed_p50"] / groups["eager"]["pooled_request_speed_p50"] - 1)
        if args.natural_eos:
            groups["natural_eos_coverage"] = {
                "cap": 2048, "ignore_eos": False,
                "unique_prompts_stopped": sum(r["finish_reason"]["type"] == "stop" for r in reference["requests"]),
                "unique_prompts_truncated": sum(r["finish_reason"]["type"] == "length" for r in reference["requests"]),
            }
        else:
            groups["fixed_length_caveat"] = {
                "ignore_eos": True,
                "unique_prompts_with_eos_within_512": sum(x is not None for x in first_eos),
                "first_eos_positions": first_eos,
                "tokens_generated_after_first_eos": sum(512 - x for x in first_eos if x is not None),
                "reference_total_output_tokens": 32 * 512,
            }
        result[dataset] = groups
    with (args.output_file or args.archive / "validated_summary.json").open("x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
