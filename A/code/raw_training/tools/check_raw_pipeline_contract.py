#!/usr/bin/env python3
"""Check the source-level contract required by the fresh raw Jittor runner.

The historical checkpoint inference sources and the fresh raw-training sources
are not interchangeable. This preflight catches the specific replay-pool API
drift that would otherwise appear only after expensive model training has
started. It intentionally parses source with ``ast`` and does not import
Jittor, so it can run before the CUDA environment is initialized.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]


def parse_module(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def functions(tree: ast.Module) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def parameter_names(function: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    return [argument.arg for argument in (*function.args.posonlyargs, *function.args.args)]


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def source_uses(path: Path, text: str, errors: list[str]) -> None:
    require("pool.build_replay_pool(" in text, f"{path.name} does not build a replay pool", errors)
    require("rank.build_slice(" in text, f"{path.name} does not build replay slices", errors)
    require('"test-pool"' in text, f"{path.name} does not bind test-pool replay mode", errors)


def check(root: Path) -> dict[str, object]:
    dataset2 = root / "dataset2"
    temporal_path = dataset2 / "d2_temporal_pool_raw_jittor.py"
    rank_path = dataset2 / "d2_pool_ranker_raw_jittor.py"
    legacy_path = root / "legacy_dataset2" / "rebuild_from_raw.py"
    main_path = root / "main.py"
    errors: list[str] = []

    for path in (temporal_path, rank_path, legacy_path, main_path):
        require(path.is_file(), f"missing required source: {path}", errors)
    if errors:
        return {"status": "FAIL", "errors": errors}

    temporal = functions(parse_module(temporal_path))
    rank = functions(parse_module(rank_path))
    replay_pool = temporal.get("build_replay_pool")
    replay = temporal.get("replay_from_pool")
    build_slice = rank.get("build_slice")
    require(replay_pool is not None, "temporal helper lacks build_replay_pool", errors)
    require(replay is not None, "temporal helper lacks replay_from_pool", errors)
    require(build_slice is not None, "rank helper lacks build_slice", errors)
    if replay_pool is not None:
        require(
            parameter_names(replay_pool) == ["test_candidates", "history_destinations"],
            "build_replay_pool parameter contract differs",
            errors,
        )
    if replay is not None:
        require(
            parameter_names(replay) == ["target", "replay_pool", "seed"],
            "replay_from_pool parameter contract differs",
            errors,
        )
    if build_slice is not None:
        expected = [
            "edges", "test", "models", "name", "bounds", "targets", "batch", "seed",
            "replay_pool", "replay_mode",
        ]
        require(parameter_names(build_slice) == expected, "build_slice parameter contract differs", errors)

    consumers = (
        dataset2 / "d2_multislice_set_raw_jittor.py",
        dataset2 / "d2_multislice_transformer_raw_jittor.py",
        dataset2 / "d2_warm_residual_raw_jittor.py",
    )
    for path in consumers:
        require(path.is_file(), f"missing raw replay consumer: {path}", errors)
        if path.is_file():
            source_uses(path, path.read_text(encoding="utf-8"), errors)

    legacy_source = legacy_path.read_text(encoding="utf-8")
    main_source = main_path.read_text(encoding="utf-8")
    require("--components-dir" in legacy_source, "Legacy rebuild does not retain component artifacts", errors)
    require(
        '"historical_exact_parity_asserted": False' in legacy_source,
        "Legacy rebuild could assert unsupported historical parity",
        errors,
    )
    require("rebuild_from_raw.py" in main_source, "main raw trainer does not invoke Legacy rebuild", errors)
    train_dataset2_source = main_source.split("def train_dataset2", 1)[1].split("def command_train", 1)[0]
    require("shutil.copy" not in train_dataset2_source, "raw Dataset2 trainer copies an existing Legacy artifact", errors)

    status = "PASS" if not errors else "FAIL"
    return {
        "kind": "track1_raw_pipeline_contract_v1",
        "status": status,
        "code_root": str(root),
        "checks": {
            "replay_pool_api": replay_pool is not None and replay is not None,
            "extended_build_slice_api": build_slice is not None,
            "raw_replay_consumers": [path.name for path in consumers],
            "legacy_components_retained": "--components-dir" in legacy_source,
            "historical_parity_not_asserted": '"historical_exact_parity_asserted": False' in legacy_source,
        },
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the fresh raw Jittor source contract")
    parser.add_argument("--code-root", type=Path, default=CODE_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = check(args.code_root.resolve())
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite contract report: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
