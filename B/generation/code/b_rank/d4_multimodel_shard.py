#!/usr/bin/env python3
"""Score deterministic D4 row shards on multiple GPUs and merge one ZIP."""

from __future__ import annotations

import argparse
import gc
import io
import json
import shutil
import sys
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from . import (
    d4_multimodel_infer as infer,
    data_features,
    temporal_attention_jittor,
    temporal_infer,
    verify_run,
)


KIND = "d4_pairnew_rank_slot_scaled_replay_shard_v21"


def _algorithm_kind(ensemble: infer.D4Ensemble) -> str:
    if ensemble.fullset is not None:
        return str(ensemble.fullset["kind"])
    if ensemble.pairnew is not None:
        return str(ensemble.pairnew["kind"])
    return str(ensemble.fit["kind"])


def _shard_bounds(index: int, count: int) -> tuple[int, int]:
    infer._require(count > 0 and 0 <= index < count, "invalid shard index/count")
    rows = infer.ROWS["dataset4"]
    return rows * index // count, rows * (index + 1) // count


def score_shard(args: argparse.Namespace) -> dict[str, Any]:
    args.data = args.data.resolve()
    args.output = args.output.resolve()
    metadata_path = args.output.with_suffix(".json")
    infer._require(args.output.suffix == ".csv", "shard output must end in .csv")
    infer._require(
        not args.output.exists() and not metadata_path.exists(),
        "shard output already exists",
    )
    infer._require(
        data_features.sha256_file(args.data) == verify_run.EXPECTED_DATA_SHA256,
        "official data_B.zip SHA-256 differs",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporal_attention_jittor.configure_cuda()
    ensemble = infer.D4Ensemble(args)
    start, stop = _shard_bounds(int(args.shard_index), int(args.shard_count))
    temporary = args.output.with_name(f".{args.output.name}.{uuid.uuid4().hex}.tmp")
    try:
        rows = 0
        with temporary.open("x", encoding="ascii", newline="\n") as text:
            for chunk in data_features.iter_test_chunks(
                args.data, "dataset4", chunk_rows=args.test_chunk_rows
            ):
                if chunk.row_stop <= start:
                    continue
                if chunk.row_start >= stop:
                    break
                left = max(start, chunk.row_start) - chunk.row_start
                right = min(stop, chunk.row_stop) - chunk.row_start
                selected = data_features.TestChunk(
                    row_start=chunk.row_start + left,
                    src=chunk.src[left:right],
                    time=chunk.time[left:right],
                    candidates=chunk.candidates[left:right],
                )
                probabilities = temporal_infer._probabilities(
                    ensemble.score(selected, args.predict_batch_rows)
                )
                np.savetxt(
                    text,
                    probabilities,
                    fmt="%.8f",
                    delimiter=",",
                    newline="\n",
                )
                rows += len(selected.src)
                if args.verbose and rows % 100000 < len(selected.src):
                    print(
                        f"dataset4 shard {args.shard_index} {rows}/{stop-start}",
                        flush=True,
                    )
        infer._require(rows == stop - start, "shard row count differs")
        infer._require(ensemble.scored_rows == rows, "shard scored row count differs")
        infer._publish_new(temporary, args.output)
        report = {
            "kind": KIND,
            "decision": "PASS",
            "data_sha256": verify_run.EXPECTED_DATA_SHA256,
            "algorithm_kind": _algorithm_kind(ensemble),
            "source_hashes": infer.source_hashes(),
            "shard_index": int(args.shard_index),
            "shard_count": int(args.shard_count),
            "row_start": start,
            "row_stop": stop,
            "rows": rows,
            "csv": {"path": str(args.output), "sha256": infer._sha256(args.output)},
            "dataset4": infer.dataset4_manifest(ensemble),
            "jittor_runtime": {
                "version": str(temporal_attention_jittor.jt.__version__),
                "has_cuda": bool(temporal_attention_jittor.jt.has_cuda),
                "use_cuda": bool(temporal_attention_jittor.jt.flags.use_cuda),
            },
        }
        infer._atomic_json(metadata_path, report)
        return report
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        del ensemble
        gc.collect()


def _load_shards(paths: list[Path]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    reports = [json.loads(path.resolve().read_text(encoding="utf-8")) for path in paths]
    reports.sort(key=lambda report: int(report.get("shard_index", -1)))
    infer._require(bool(reports), "no shard reports were supplied")
    count = len(reports)
    hashes = infer.source_hashes()
    cursor = 0
    reference = None
    for index, report in enumerate(reports):
        infer._require(
            report.get("kind") == KIND
            and report.get("decision") == "PASS"
            and report.get("data_sha256") == verify_run.EXPECTED_DATA_SHA256,
            f"shard report contract differs: {index}",
        )
        infer._require(
            int(report["shard_index"]) == index
            and int(report["shard_count"]) == count
            and int(report["row_start"]) == cursor
            and int(report["rows"])
            == int(report["row_stop"]) - int(report["row_start"]),
            f"shard range differs: {index}",
        )
        cursor = int(report["row_stop"])
        infer._require(report.get("source_hashes") == hashes, f"shard source differs: {index}")
        infer._require(
            report.get("algorithm_kind")
            == "d4_pairnew_rank_slot_scaled_replay_transformer_v21",
            f"shard algorithm differs: {index}",
        )
        runtime = report.get("jittor_runtime", {})
        infer._require(
            runtime.get("has_cuda") is True and runtime.get("use_cuda") is True,
            f"shard did not use Jittor CUDA: {index}",
        )
        csv_path = Path(report["csv"]["path"])
        infer._require(
            csv_path.is_file() and infer._sha256(csv_path) == report["csv"]["sha256"],
            f"shard CSV hash differs: {index}",
        )
        comparable = json.loads(json.dumps(report["dataset4"], sort_keys=True))
        comparable["duplicate_candidate_gate"] = None
        if reference is None:
            reference = comparable
        else:
            infer._require(comparable == reference, f"shard model contract differs: {index}")
    infer._require(cursor == infer.ROWS["dataset4"], "shards do not cover all D4 rows")
    return reports, hashes


def merge(args: argparse.Namespace) -> dict[str, Any]:
    data = args.data.resolve()
    output = args.output.resolve()
    manifest_path = output.with_suffix(".manifest.json")
    infer._require(output.suffix == ".zip", "output must end in .zip")
    infer._require(not output.exists() and not manifest_path.exists(), "output already exists")
    infer._require(
        data_features.sha256_file(data) == verify_run.EXPECTED_DATA_SHA256,
        "official data_B.zip SHA-256 differs",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    reports, hashes = _load_shards(args.shard_report)
    temporary_output = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    temporary_manifest = manifest_path.with_name(
        f".{manifest_path.name}.{uuid.uuid4().hex}.tmp"
    )
    source_root = Path(__file__).resolve().parents[1]
    try:
        with zipfile.ZipFile(
            temporary_output,
            "x",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as archive:
            d3_record = infer._copy_d3(
                args.dataset3_source.resolve(),
                args.dataset3_manifest.resolve(),
                archive,
            )
            with archive.open("dataset4.csv", "w", force_zip64=True) as outgoing:
                for report in reports:
                    with Path(report["csv"]["path"]).open("rb") as incoming:
                        shutil.copyfileobj(incoming, outgoing, length=8 << 20)
        dataset4 = reports[0]["dataset4"]
        duplicate = sum(
            int(report["dataset4"]["duplicate_candidate_gate"]["gated_rows"])
            for report in reports
        )
        gate = dataset4["duplicate_candidate_gate"]
        gate.update(
            gated_rows=duplicate,
            scored_rows=infer.ROWS["dataset4"],
            gated_row_rate=duplicate / infer.ROWS["dataset4"],
        )
        manifest = {
            "kind": verify_run.MULTIMODEL_INFERENCE_MANIFEST_KIND,
            "algorithm_kind": reports[0]["algorithm_kind"],
            "created_utc": datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat(),
            "data_sha256": verify_run.EXPECTED_DATA_SHA256,
            "submission_sha256": infer._sha256(temporary_output),
            "source_hashes": hashes,
            "row_counts": infer.ROWS,
            "format": "exactly dataset3.csv,dataset4.csv; headerless ASCII; %.8f probabilities",
            "dataset3": d3_record,
            "dataset4": dataset4,
            "shards": [
                {
                    key: report[key]
                    for key in ("shard_index", "shard_count", "row_start", "row_stop", "rows", "csv")
                }
                for report in reports
            ],
            "jittor_runtime": reports[0]["jittor_runtime"],
        }
        infer._atomic_json(temporary_manifest, manifest)
        verification = verify_run.verify_run(
            data,
            temporary_output,
            temporary_manifest,
            source_root=source_root,
        )
        infer._publish_new(temporary_output, output)
        infer._publish_new(temporary_manifest, manifest_path)
        return {
            "kind": "d4_pairnew_rank_slot_scaled_replay_inference_result_v21",
            "decision": "PASS",
            "output": str(output),
            "output_sha256": infer._sha256(output),
            "manifest": str(manifest_path),
            "verification": verification,
        }
    except Exception:
        temporary_output.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)
        raise


def shard_parser() -> argparse.ArgumentParser:
    parser = infer.build_parser()
    parser.description = __doc__
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    return parser


def merge_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dataset3-source", type=Path, required=True)
    parser.add_argument("--dataset3-manifest", type=Path, required=True)
    parser.add_argument("--shard-report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    try:
        if len(sys.argv) < 2 or sys.argv[1] not in {"shard", "merge"}:
            raise ValueError("first argument must be shard or merge")
        if sys.argv[1] == "shard":
            result = score_shard(shard_parser().parse_args(sys.argv[2:]))
        else:
            result = merge(merge_parser().parse_args(sys.argv[2:]))
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return 0
    except Exception as error:
        print(
            json.dumps(
                {"kind": KIND, "decision": "ERROR", "error": f"{type(error).__name__}: {error}"},
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
