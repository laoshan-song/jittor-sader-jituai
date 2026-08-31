#!/usr/bin/env python3
"""Build a fresh, fully retained legacy Dataset2 score plane with Jittor.

The archived rankers use fixed working-directory filenames.  Each member is
therefore trained in an isolated, caller-owned directory and every learned
state is retained beside an immutable manifest.  This is a raw-data rebuild
path, not an assertion that its result is byte-identical to the historical
release artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
MEMBER_SOURCES = (
    "rebuild_from_raw.py",
    "friend_ranker_jittor.py",
    "ours_ranker_jittor.py",
    "cf_ranker_jittor.py",
    "train_cf_embedding_jittor.py",
    "train_multdae_jittor.py",
    "score_multdae_jittor.py",
    "build_legacy.py",
)


def invoke(command: list[object], *, cwd: Path, env: dict[str, str]) -> None:
    values = [str(value) for value in command]
    print("+", " ".join(values), flush=True)
    subprocess.run(values, cwd=cwd, env=env, check=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def zip_member_sha256(path: Path, member: str) -> str:
    with zipfile.ZipFile(path) as archive:
        return hashlib.sha256(archive.read(member)).hexdigest()


def component_inventory(root: Path) -> dict[str, str]:
    inventory: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            inventory[str(path.relative_to(root))] = sha256_file(path)
    return inventory


def run_ranker(script: Path, workdir: Path, env: dict[str, str], quick: bool) -> None:
    train: list[object] = [sys.executable, script, "train"]
    rank: list[object] = [
        sys.executable,
        script,
        "rank",
        "--rank-scene",
        "dataset2",
    ]
    if quick:
        train.append("--quick")
        rank.extend(["--quick", "--rank-groups", 512, "--rank-valid", 512, "--rank-epochs", 1, "--batch", 128])
    invoke(train, cwd=workdir, env=env)
    invoke(rank, cwd=workdir, env=env)
    invoke([sys.executable, script, "submit"], cwd=workdir, env=env)


def official_history_end(data: Path) -> int:
    """Use the official Dataset2 test boundary required by MultDAE scoring."""
    with zipfile.ZipFile(data) as archive:
        times = pd.read_csv(archive.open("dataset2/test.csv"), usecols=["time"])
    if times.empty:
        raise ValueError("dataset2/test.csv has no rows")
    return int(times["time"].min())


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild a new legacy Dataset2 plane using Jittor rankers")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--components-dir",
        type=Path,
        help="new directory that will retain every raw legacy member artifact",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    data = args.data.resolve()
    if not data.is_file():
        raise FileNotFoundError(data)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite legacy output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    components = (args.components_dir or args.output.with_suffix(".components")).resolve()
    if components.exists():
        raise FileExistsError(f"refusing to reuse legacy component directory: {components}")
    metadata_path = args.output.with_suffix(".training.json")
    if metadata_path.exists():
        raise FileExistsError(f"refusing to overwrite legacy metadata: {metadata_path}")
    history_end = official_history_end(data)
    components.mkdir(parents=True)
    env = os.environ.copy()
    env["DATA_PATH"] = str(data)
    env["TRACK1_DATA"] = str(data)
    env.setdefault("ML_CACHE_ROOT", "/tmp")
    env.setdefault("use_mpi", "0")
    env["PYTHONHASHSEED"] = "20260711"
    env["SEED"] = "20260711"
    # Do not let an interactive experiment's optional state silently alter a
    # supposedly fresh reconstruction. The driver binds every such input.
    for name in ("CF_EMB", "CF_INIT", "OUT", "SCENE", "GATE", "EMB_DIM", "EPOCHS", "NEG", "BATCH", "LR", "JT_USE_CUDA"):
        env.pop(name, None)
    if args.cuda:
        env["JT_USE_CUDA"] = "1"
        invoke(
            [
                sys.executable,
                "-c",
                "import jittor as jt; "
                "assert jt.has_cuda, 'Jittor CUDA is unavailable'; "
                "jt.flags.use_cuda = 1; "
                "assert jt.flags.use_cuda == 1, 'Jittor CUDA could not be enabled'; "
                "print('jittor_cuda_ready', jt.__version__)",
            ],
            cwd=components,
            env=env,
        )

    cf_embedding = components / "cf_embedding_full_d128.npz"
    cf_env = env.copy()
    cf_env.update({"OUT": str(cf_embedding), "SCENE": "dataset2", "GATE": "full"})
    if args.quick:
        cf_env.update({"EPOCHS": "1", "BATCH": "8192"})
    invoke([sys.executable, HERE / "train_cf_embedding_jittor.py"], cwd=components, env=cf_env)
    if not cf_embedding.is_file():
        raise RuntimeError("CF embedding trainer did not write its declared artifact")

    component_zips: dict[str, Path] = {}
    for name, source in (("friend", "friend_ranker_jittor.py"), ("ours", "ours_ranker_jittor.py"), ("cf", "cf_ranker_jittor.py")):
        workdir = components / name
        workdir.mkdir()
        # The archived rankers resolve these historical names relative to cwd.
        shutil.copy2(cf_embedding, workdir / "cf_full_d128.npz")
        shutil.copy2(cf_embedding, workdir / "cf_full_d128_ours.npz")
        member_env = env.copy()
        member_env["CF_EMB"] = str(cf_embedding)
        run_ranker(HERE / source, workdir, member_env, args.quick)
        result = workdir / "result.zip"
        if not result.is_file():
            raise RuntimeError(f"{name} ranker did not produce result.zip")
        component_zips[name] = result

    multdae_dir = components / "multdae"
    multdae_dir.mkdir()
    multdae_model = multdae_dir / "model_jittor.npz"
    multdae_raw = multdae_dir / "official_raw.npy"
    train_command: list[object] = [
        sys.executable,
        HERE / "train_multdae_jittor.py",
        "train",
        "--history-end",
        history_end,
        "--output",
        multdae_model,
    ]
    if args.quick:
        train_command.extend(["--epochs", 1, "--batch", 128])
    if args.cuda:
        train_command.append("--cuda")
    invoke(train_command, cwd=multdae_dir, env=env)
    score_command: list[object] = [sys.executable, HERE / "score_multdae_jittor.py", "--data", data, "--model", multdae_model, "--output", multdae_raw]
    if args.cuda:
        score_command.append("--cuda")
    invoke(score_command, cwd=multdae_dir, env=env)
    invoke([sys.executable, HERE / "build_legacy.py", "--data", data, "--friend", component_zips["friend"], "--ours", component_zips["ours"], "--cf", component_zips["cf"], "--multdae", multdae_raw, "--output", args.output], cwd=components, env=env)

    source_hashes = {name: sha256_file(HERE / name) for name in MEMBER_SOURCES}
    component_manifest = {
        "kind": "fresh_legacy_dataset2_jittor_components_v2",
        "data": str(data),
        "data_sha256": sha256_file(data),
        "history_end": history_end,
        "quick": args.quick,
        "cuda_requested": args.cuda,
        "seed": int(env["SEED"]),
        "source_sha256": source_hashes,
        "component_file_sha256": component_inventory(components),
        "legacy_output": str(args.output),
        "legacy_output_sha256": sha256_file(args.output),
        "legacy_dataset2_csv_sha256": zip_member_sha256(args.output, "dataset2.csv"),
        "historical_exact_parity_asserted": False,
    }
    (components / "manifest.json").write_text(
        json.dumps(component_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metadata = {
        "kind": "fresh_legacy_dataset2_jittor_rebuild_v2",
        "data": str(data),
        "data_sha256": sha256_file(data),
        "history_end": history_end,
        "quick": args.quick,
        "components": str(components),
        "components_manifest": str(components / "manifest.json"),
        "historical_exact_parity_asserted": False,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(
        "wrote",
        args.output,
        "history_end=",
        history_end,
        "historical_exact_parity_asserted=false",
        flush=True,
    )


if __name__ == "__main__":
    main()
