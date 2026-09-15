#!/usr/bin/env python3
"""Reproduce ruc4 from the official B archive with Jittor-only neural models."""

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

import numpy as np


DATA_SHA256 = "ded8b0d281042323f0c5871868824038bc7fb675cc3e8211753bb63d8b7b89d2"
ONLINE_SHA256 = "face96780e4cc04d982aa174954aae3479f5d0dd806816390880cb441d56319f"
ROWS = {"dataset3.csv": 157_670, "dataset4.csv": 2_322_538}
SEEDS_D3 = (20260810, 20260824, 20260907)
SEEDS_D4 = (20261101, 20261117, 20261133)
ROOT = Path(__file__).resolve().parent
C2_ROOT = ROOT / "c2_source"
C2_CODE = C2_ROOT / "code"
D3_CODE = C2_CODE / "c2_d3"
RUC3_CODE = ROOT / "code" / "ruc3"
RUC4_CODE = ROOT / "code" / "ruc4"
JITTOR_OFFLINE_ENV = {
    "use_cutt": "0",
    "use_cutlass": "0",
    "use_nccl": "0",
    "use_mkl": "0",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_env(args: argparse.Namespace, work: Path, gpu: int) -> dict[str, str]:
    env = os.environ.copy()
    paths = (
        C2_CODE,
        C2_CODE / "b_rank_a_port",
        D3_CODE,
        ROOT / "code" / "c2_d3",
        ROOT / "code" / "c6",
        RUC3_CODE,
        RUC4_CODE,
    )
    env.update(
        {
            **JITTOR_OFFLINE_ENV,
            "JT_USE_CUDA": "1",
            "HOME": str(work / "runtime" / "home"),
            "XDG_CACHE_HOME": str(work / "runtime" / "cache"),
            "TMPDIR": str(work / "runtime" / "tmp"),
            "JITTOR_HOME": str(
                args.jittor_home.resolve()
                if args.jittor_home else work / "runtime" / "jittor"
            ),
            "PYTHONPATH": os.pathsep.join(map(str, paths)),
        }
    )
    env.setdefault("CUDA_VISIBLE_DEVICES", str(gpu))
    for name in ("HOME", "XDG_CACHE_HOME", "TMPDIR", "JITTOR_HOME"):
        Path(env[name]).mkdir(parents=True, exist_ok=True)
    if args.cuda_home:
        cuda = args.cuda_home.resolve()
        env["CUDA_HOME"] = str(cuda)
        env["PATH"] = os.pathsep.join((str(cuda / "bin"), env.get("PATH", "")))
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            (str(cuda / "lib64"), env.get("LD_LIBRARY_PATH", ""))
        )
    return env


def check_jittor_offline_guards() -> None:
    missing = [name for name, value in JITTOR_OFFLINE_ENV.items() if value != "0"]
    if missing:
        raise RuntimeError(f"invalid offline Jittor guard values: {missing}")


def run(command: list[str], cwd: Path, env: dict[str, str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    print("RUN", " ".join(command), flush=True)
    with log.open("x", encoding="utf-8") as handle:
        subprocess.run(
            command, cwd=cwd, env=env, stdout=handle,
            stderr=subprocess.STDOUT, check=True,
        )


def remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def run_stage(
    *,
    resume: bool,
    expected: tuple[Path, ...],
    cleanup: tuple[Path, ...],
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    log: Path,
) -> bool:
    if resume and all(path.exists() for path in expected):
        print("SKIP", " ".join(str(path) for path in expected), flush=True)
        return False
    if resume:
        for path in cleanup:
            remove_path(path)
        log.unlink(missing_ok=True)
    run(command, cwd, env, log)
    return True


def run_parallel(
    jobs: list[tuple[list[str], dict[str, str], Path]], cwd: Path
) -> None:
    active = []
    for command, env, log in jobs:
        log.parent.mkdir(parents=True, exist_ok=True)
        handle = log.open("xb")
        process = subprocess.Popen(
            command, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT
        )
        active.append((command, process, handle, log))
    failed = []
    for command, process, handle, log in active:
        code = process.wait()
        handle.close()
        if code:
            failed.append((code, command, log))
    if failed:
        raise RuntimeError(f"parallel jobs failed: {failed}")


def quick_check(args: argparse.Namespace, work: Path, env: dict[str, str]) -> None:
    sys.path[:0] = [str(C2_CODE), str(RUC3_CODE), str(RUC4_CODE)]
    os.environ.update(env)
    import jittor as jt
    import build_d4_rp3_candidate as rp3_build
    import d4_rp3beta_gate as rp3

    jt.flags.use_cuda = 1
    baseline = np.asarray([[0.4, 0.3, 0.2, 0.1]], dtype=np.float32)
    residual = np.asarray([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    seen = np.asarray([[True, False, False, False]])
    candidate = rp3.candidate_score(baseline, residual, seen, rp3_build.ALPHA)
    if np.argsort(-candidate, axis=1)[0].tolist().index(0) != 0:
        raise ValueError("RP3 policy changed a frozen pair-seen rank")
    receipt = {
        "kind": "ruc4_quick_jittor_linkage_v1",
        "decision": "SMOKE_ONLY",
        "data_sha256": DATA_SHA256,
        "rp3_policy": {
            "alpha": rp3_build.ALPHA,
            "beta": rp3_build.BETA,
            "pair_seen_rank_frozen": True,
        },
        "jittor": str(jt.__version__),
        "warning": "quick mode does not build a submission and must not be submitted",
    }
    (work / "REPRODUCTION_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)


def train_d3_members(
    args: argparse.Namespace,
    work: Path,
    ensemble: Path,
    destination: Path,
    fit_splits: tuple[str, ...],
    gpus: list[int],
) -> tuple[list[Path], list[Path]]:
    destination.mkdir(parents=True, exist_ok=args.resume)
    models = [destination / f"seed{seed}.pkl" for seed in SEEDS_D3]
    reports = [destination / f"seed{seed}.json" for seed in SEEDS_D3]
    jobs = []
    for index, (seed, model, report) in enumerate(zip(SEEDS_D3, models, reports)):
        log = work / "logs" / f"d3_{destination.name}_{seed}.log"
        if args.resume and model.is_file() and report.is_file():
            print("SKIP", model, report, flush=True)
            continue
        if args.resume:
            model.unlink(missing_ok=True)
            report.unlink(missing_ok=True)
            log.unlink(missing_ok=True)
        command = [
            sys.executable, str(RUC3_CODE / "d3_set_transformer_v49.py"),
            "--data", str(args.data), "--code", str(D3_CODE),
            "--ensemble-report", str(ensemble), "--output", str(report),
            "--model-output", str(model), "--groups", "30000", "--batch", "64",
            "--predict-batch", "128", "--epochs", "8", "--hidden", "64",
            "--layers", "2", "--heads", "4", "--learning-rate", "0.0008",
            "--base-logit-scale", "2.0", "--seed", str(seed), "--full-fit",
            "--freeze-final-epoch", "--fixed-scale", "0.30", "--fit-splits",
            *fit_splits,
        ]
        jobs.append((
            command,
            runtime_env(args, work, gpus[index % len(gpus)]),
            log,
        ))
    if jobs:
        run_parallel(jobs, RUC3_CODE)
    return models, reports


def build_d3_assets(
    args: argparse.Namespace,
    work: Path,
    ensemble: Path,
    gpus: list[int],
) -> tuple[list[Path], list[Path], Path, Path]:
    rolling_models, rolling_reports = train_d3_members(
        args, work, ensemble, work / "d3_rolling", ("meta_train", "validation"), gpus
    )
    direct_audit = work / "reports" / "d3_v65_rolling_audit.json"
    run_stage(
        resume=args.resume,
        expected=(direct_audit,),
        cleanup=(direct_audit,),
        command=[
            sys.executable, str(RUC3_CODE / "formal_direct_set_v53.py"),
            "--data", str(args.data), "--code", str(D3_CODE),
            "--ensemble-report", str(ensemble),
            "--transformer-model", *map(str, rolling_models),
            "--transformer-report", *map(str, rolling_reports),
            "--output", str(direct_audit), "--groups", "30000", "--batch", "128",
            "--seed", "20260810", "--split", "both", "--aggregation", "mean_member",
            "--minimum-direct-delta", "0.008",
        ],
        cwd=RUC3_CODE,
        env=runtime_env(args, work, gpus[0]),
        log=work / "logs" / "d3_direct_audit.log",
    )
    duplicate_audit = work / "reports" / "d3_duplicate_group_audit.json"
    run_stage(
        resume=args.resume,
        expected=(duplicate_audit,),
        cleanup=(duplicate_audit,),
        command=[
            sys.executable, str(RUC3_CODE / "d3_duplicate_group_audit.py"),
            "--data", str(args.data), "--code", str(D3_CODE),
            "--ensemble-report", str(ensemble),
            "--transformer-model", *map(str, rolling_models),
            "--transformer-report", *map(str, rolling_reports),
            "--output", str(duplicate_audit), "--groups", "30000", "--batch", "128",
            "--seed", "20260810", "--split", "both", "--aggregation", "mean_member",
        ],
        cwd=RUC3_CODE,
        env=runtime_env(args, work, gpus[0]),
        log=work / "logs" / "d3_duplicate_audit.log",
    )
    final_models, final_reports = train_d3_members(
        args, work, ensemble, work / "d3_final",
        ("meta_train", "validation", "confirmation"), gpus,
    )
    return final_models, final_reports, direct_audit, duplicate_audit


def d4_cache_root(c2_work: Path) -> Path:
    return c2_work / "cache" / "b_rank_data" / f"dataset4-{DATA_SHA256[:20]}"


def train_d4_members(
    args: argparse.Namespace, work: Path, c2_work: Path, gpus: list[int]
) -> list[Path]:
    replay = work / "d4_replay"
    control_report = c2_work / "d4_control" / "research_report.json"
    for strategy in ("history", "test_pool"):
        output_cache = replay / strategy
        run_dir = work / f"d4_replay_build_{strategy}"
        run_stage(
            resume=args.resume,
            expected=(output_cache / "manifest.json",),
            cleanup=(output_cache, run_dir),
            command=[
                sys.executable, str(RUC3_CODE / "build_large_cache.py"),
                "--source-manifest", str(control_report), "--data", str(args.data),
                "--cache-dir", str(c2_work / "cache"),
                "--run-dir", str(run_dir),
                "--output-cache", str(output_cache), "--strategy", strategy,
                "--valid-groups", "120000", "--confirm-groups", "30000",
                "--batch-rows", "512",
            ],
            cwd=RUC3_CODE,
            env=runtime_env(args, work, gpus[0]),
            log=work / "logs" / f"d4_replay_{strategy}.log",
        )
    replay_args = [
        "--replay-cache", str(replay / "history"),
        "--replay-cache", str(replay / "test_pool"),
    ]
    score_cache_args = [
        "--score-cache", str(replay / "history"),
        "--score-cache", str(replay / "test_pool"),
    ]
    identity = work / "d4_identity"
    run_stage(
        resume=args.resume,
        expected=(identity / "manifest.json",),
        cleanup=(identity,),
        command=[
            sys.executable, str(RUC3_CODE / "build_identity_cache_v33.py"),
            "--data", str(args.data), "--data-cache", str(c2_work / "cache"),
            *score_cache_args, "--output", str(identity),
        ],
        cwd=RUC3_CODE,
        env=runtime_env(args, work, gpus[0]),
        log=work / "logs" / "d4_identity.log",
    )
    baseline = work / "d4_baseline"
    run_stage(
        resume=args.resume,
        expected=(baseline / "manifest.json",),
        cleanup=(baseline,),
        command=[
            sys.executable, str(RUC3_CODE / "build_baseline_cache.py"),
            *replay_args, "--control-fit", str(control_report),
            "--pairnew-report", str(c2_work / "d4_pairnew" / "research_report.json"),
            "--output", str(baseline), "--batch", "256",
        ],
        cwd=RUC3_CODE,
        env=runtime_env(args, work, gpus[0]),
        log=work / "logs" / "d4_baseline.log",
    )
    train_cache = d4_cache_root(c2_work)
    jobs = []
    model_paths = []
    for index, seed in enumerate(SEEDS_D4):
        run_dir = work / "d4_models" / f"seed{seed}"
        model = run_dir / "model.npz"
        report = run_dir / "report.json"
        model_paths.append(model)
        log = work / "logs" / f"d4_train_{seed}.log"
        if args.resume and model.is_file() and report.is_file():
            print("SKIP", model, report, flush=True)
            continue
        if args.resume:
            remove_path(run_dir)
            log.unlink(missing_ok=True)
        jobs.append((
            [
                sys.executable, str(RUC3_CODE / "session_graph_hard_ranker.py"),
                "--train-cache", str(train_cache), *replay_args,
                "--identity-cache", str(identity), "--baseline-cache", str(baseline),
                "--run-dir", str(run_dir), "--epochs", "4", "--batch", "128",
                "--chunk", "1024", "--seed", str(seed),
            ],
            runtime_env(args, work, gpus[index % len(gpus)]),
            log,
        ))
    if jobs:
        run_parallel(jobs, RUC3_CODE)
    leakage = work / "reports" / "d4_session_graph_leakage_audit.json"
    command = [
        sys.executable, str(RUC3_CODE / "session_graph_leakage_audit.py"),
        "--train-cache", str(train_cache), *replay_args,
        "--identity-cache", str(identity), "--baseline-cache", str(baseline),
    ]
    for model in model_paths:
        command += ["--model", str(model)]
    command += ["--output", str(leakage), "--alpha", "0.01"]
    run_stage(
        resume=args.resume,
        expected=(leakage,),
        cleanup=(leakage,),
        command=command,
        cwd=RUC3_CODE,
        env=runtime_env(args, work, gpus[0]),
        log=work / "logs" / "d4_leakage_audit.log",
    )
    return model_paths


def split_member(
    archive: Path, output: Path, count: int, *, resume: bool = False
) -> list[Path]:
    boundaries = [ROWS["dataset4.csv"] * index // count for index in range(count + 1)]
    paths = [output / f"dataset4_base_{index}.csv" for index in range(count)]
    if resume and all(path.is_file() for path in paths):
        print("SKIP", " ".join(str(path) for path in paths), flush=True)
        return paths
    if resume:
        remove_path(output)
    output.mkdir(parents=True)
    handles = [path.open("xb") for path in paths]
    written = [0] * count
    try:
        with zipfile.ZipFile(archive) as source, source.open("dataset4.csv") as member:
            shard = 0
            for row, line in enumerate(member):
                while shard + 1 < count and row >= boundaries[shard + 1]:
                    shard += 1
                handles[shard].write(line)
                written[shard] += 1
    finally:
        for handle in handles:
            handle.close()
    expected = [boundaries[index + 1] - boundaries[index] for index in range(count)]
    if written != expected:
        raise ValueError(f"D4 shard rows differ: {written} != {expected}")
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--gpus", default="0", help="comma-separated visible GPU ids")
    parser.add_argument("--jittor-home", type=Path)
    parser.add_argument("--cuda-home", type=Path)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.data = args.data.resolve()
    work = args.work_dir.resolve()
    gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    if not gpus or len(gpus) != len(set(gpus)):
        raise ValueError("--gpus must contain unique comma-separated integers")
    if sha256(args.data) != DATA_SHA256:
        raise ValueError("official data_B.zip hash differs")
    check_jittor_offline_guards()
    if work.exists() and not args.resume:
        raise FileExistsError(f"refusing work-directory reuse: {work}")
    work.mkdir(parents=True, exist_ok=args.resume)
    env = runtime_env(args, work, gpus[0])
    if args.quick:
        quick_check(args, work, env)
        return 0

    c6_work = work / "c6"
    c6_command = [
        sys.executable, str(ROOT / "reproduce_c6.py"), "--data", str(args.data),
        "--work-dir", str(c6_work), "--gpu", str(gpus[0]),
    ]
    if args.jittor_home:
        c6_command += ["--jittor-home", str(args.jittor_home.resolve())]
    if args.cuda_home:
        c6_command += ["--cuda-home", str(args.cuda_home.resolve())]
    c6_zip = c6_work / "b_rank_d34_c6_d3_c5_tie_group.zip"
    if args.resume:
        c6_command.append("--resume")
    c6_log = work / "logs" / "reproduce_c6.log"
    if args.resume and c6_zip.is_file() and (c6_work / "REPRODUCTION_RECEIPT.json").is_file():
        print("SKIP", c6_zip, flush=True)
    else:
        if args.resume:
            c6_log.unlink(missing_ok=True)
        run(c6_command, ROOT, env, c6_log)

    c2_work = c6_work / "c5" / "prerequisite" / "c2"
    ensemble = c2_work / "reports" / "dataset3_ensemble.json"
    d3_models, d3_reports, direct_audit, duplicate_audit = build_d3_assets(
        args, work, ensemble, gpus
    )
    d4_models = train_d4_members(args, work, c2_work, gpus)
    d4_diagnostics = {
        "session_graph": {
            path.parent.name: json.loads(
                (path.parent / "report.json").read_text(encoding="utf-8")
            )["decision"]
            for path in d4_models
        },
        "leakage": json.loads(
            (work / "reports" / "d4_session_graph_leakage_audit.json").read_text(
                encoding="utf-8"
            )
        )["decision"],
        "role": "recorded diagnostics; deployment uses the freshly trained models",
    }

    v65 = work / "v65.zip"
    v65_report = work / "reports" / "v65_build.json"
    run_stage(
        resume=args.resume,
        expected=(v65, v65_report),
        cleanup=(v65, v65_report),
        command=[
            sys.executable, str(RUC3_CODE / "build_set_transformer_submission_v65.py"),
            "--data", str(args.data), "--code", str(D3_CODE),
            "--ensemble-report", str(ensemble), "--validation-report", str(direct_audit),
            "--transformer-model", *map(str, d3_models),
            "--transformer-report", *map(str, d3_reports),
            "--base", str(c6_zip), "--output", str(v65),
            "--report-output", str(v65_report), "--batch", "128",
            "--allow-reproduced-base",
        ],
        cwd=RUC3_CODE,
        env=env,
        log=work / "logs" / "build_v65.log",
    )

    shard_count = len(gpus)
    base_shards = split_member(
        v65, work / "d4_base_shards", shard_count, resume=args.resume
    )
    train_cache = d4_cache_root(c2_work)
    feature_store = train_cache / "stats" / "history_lt_1512137910"
    if not feature_store.is_dir():
        raise FileNotFoundError(f"D4 test feature store is missing: {feature_store}")
    output_shards = [work / "d4_ruc2_shards" / f"dataset4_shard_{i}.csv" for i in range(shard_count)]
    jobs = []
    for index, (gpu, source, output) in enumerate(zip(gpus, base_shards, output_shards)):
        log = work / "logs" / f"d4_deploy_{index}.log"
        if args.resume and output.is_file():
            print("SKIP", output, flush=True)
            continue
        if args.resume:
            output.unlink(missing_ok=True)
            log.unlink(missing_ok=True)
        command = [
            sys.executable, str(RUC3_CODE / "session_graph_deploy.py"),
            "--data", str(args.data), "--train-cache", str(train_cache),
            "--feature-store", str(feature_store), "--input", str(source),
            "--output", str(output), "--shard-index", str(index),
            "--shard-count", str(shard_count), "--alpha", "0.01",
            "--chunk", "4096", "--batch", "256",
        ]
        for model in d4_models:
            command += ["--model", str(model)]
        jobs.append((command, runtime_env(args, work, gpu), log))
    if jobs:
        run_parallel(jobs, RUC3_CODE)

    ruc2_zip = work / "b_ruc2.zip"
    package_command = [
        sys.executable, str(RUC3_CODE / "build_ruc2_package.py"),
        "--baseline", str(v65),
    ]
    for shard in output_shards:
        package_command += ["--shard", str(shard)]
    package_command += [
        "--output", str(ruc2_zip), "--report", str(work / "reports" / "ruc2_build.json")
    ]
    ruc2_report = work / "reports" / "ruc2_build.json"
    run_stage(
        resume=args.resume,
        expected=(ruc2_zip, ruc2_report),
        cleanup=(ruc2_zip, ruc2_report),
        command=package_command,
        cwd=RUC3_CODE,
        env=env,
        log=work / "logs" / "build_ruc2.log",
    )

    ruc3_output = work / "ruc3.zip"
    ruc3_report = work / "reports" / "ruc3_build.json"
    built_ruc3 = run_stage(
        resume=args.resume,
        expected=(ruc3_output, ruc3_report),
        cleanup=(ruc3_output, ruc3_report),
        command=[
            sys.executable, str(RUC3_CODE / "build_b_candidate.py"),
            "--data", str(args.data), "--code", str(D3_CODE),
            "--ensemble-report", str(ensemble), "--validation-report", str(duplicate_audit),
            "--transformer-model", *map(str, d3_models),
            "--transformer-report", *map(str, d3_reports),
            "--base", str(ruc2_zip), "--output", str(ruc3_output),
            "--report-output", str(ruc3_report),
            "--batch", "128",
        ],
        cwd=RUC3_CODE,
        env=env,
        log=work / "logs" / "build_ruc3.log",
    )
    if built_ruc3:
        (work / "logs" / "verify_ruc3.log").unlink(missing_ok=True)
        run(
        [
            sys.executable, str(C2_CODE / "verify_v26_submission.py"),
            "--data", str(args.data), "--submission", str(ruc3_output),
        ],
        C2_ROOT, env, work / "logs" / "verify_ruc3.log",
        )
    output = work / "ruc4.zip"
    ruc4_report = work / "reports" / "ruc4_rp3_build.json"
    built_ruc4 = run_stage(
        resume=args.resume,
        expected=(output, ruc4_report),
        cleanup=(output, ruc4_report),
        command=[
            sys.executable, str(RUC4_CODE / "build_d4_rp3_candidate.py"),
            "--data", str(args.data), "--data-cache", str(train_cache),
            "--base", str(ruc3_output), "--output", str(output),
            "--report", str(ruc4_report),
            "--threads", str(min(48, os.cpu_count() or 1)),
        ],
        cwd=RUC4_CODE,
        env=env,
        log=work / "logs" / "build_ruc4.log",
    )
    if built_ruc4:
        (work / "logs" / "verify_ruc4.log").unlink(missing_ok=True)
        run(
        [
            sys.executable, str(C2_CODE / "verify_v26_submission.py"),
            "--data", str(args.data), "--submission", str(output),
        ],
        C2_ROOT, env, work / "logs" / "verify_ruc4.log",
        )
    receipt = {
        "kind": "ruc4_end_to_end_reproduction_receipt_v1",
        "decision": "PASS",
        "data_sha256": DATA_SHA256,
        "submission_sha256": sha256(output),
        "historical_online_submission_sha256": ONLINE_SHA256,
        "exact_historical_sha": sha256(output) == ONLINE_SHA256,
        "ruc3_base_sha256": sha256(ruc3_output),
        "final_models": "freshly trained from official data",
        "d4_diagnostics": d4_diagnostics,
        "neural_framework": "Jittor",
    }
    (work / "REPRODUCTION_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
