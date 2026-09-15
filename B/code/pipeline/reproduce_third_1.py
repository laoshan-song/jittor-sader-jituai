#!/usr/bin/env python3
"""Reproduce third_1 from official data with code-only Jittor training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


DATA_SHA256 = "ded8b0d281042323f0c5871868824038bc7fb675cc3e8211753bb63d8b7b89d2"
ONLINE_SHA256 = "693f0635c370ff636ca77a10c3de4dfcb90bf1617f3afc3bd74d5a9b9b36600a"
ROOT = Path(__file__).resolve().parent
CODE = ROOT / "code"
B_RANK = CODE / "b_rank"
THIRD = CODE / "third_1"
RUC3 = CODE / "ruc3"
RUC4 = CODE / "ruc4"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def env_for(args: argparse.Namespace, work: Path, gpu: int) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": os.pathsep.join(map(str, (CODE, B_RANK, THIRD, RUC3, RUC4))),
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "JT_USE_CUDA": "1",
            "use_cutt": "0",
            "use_cutlass": "0",
            "use_nccl": "0",
            "use_mkl": "0",
            "HOME": str(work / "runtime" / "home"),
            "XDG_CACHE_HOME": str(work / "runtime" / "cache"),
            "TMPDIR": str(work / "runtime" / "tmp"),
            "JITTOR_HOME": str(args.jittor_home.resolve() if args.jittor_home else work / "runtime" / "jittor"),
        }
    )
    for key in ("HOME", "XDG_CACHE_HOME", "TMPDIR", "JITTOR_HOME"):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    if args.cuda_home:
        cuda = args.cuda_home.resolve()
        env["CUDA_HOME"] = str(cuda)
        env["PATH"] = os.pathsep.join((str(cuda / "bin"), env.get("PATH", "")))
        env["LD_LIBRARY_PATH"] = os.pathsep.join((str(cuda / "lib64"), env.get("LD_LIBRARY_PATH", "")))
    return env


def run(command: list[str], cwd: Path, env: dict[str, str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    print("RUN", " ".join(command), flush=True)
    with log.open("x", encoding="utf-8") as handle:
        subprocess.run(command, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)


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


def temporal_args(report: Path) -> list[str]:
    value = json.loads(report.read_text(encoding="utf-8"))
    out: list[str] = []
    for record in value["checkpoints"]:
        out += ["--temporal", record["name"], str(record["history_size"]), record["path"]]
    return out


def checkpoint(report: Path) -> str:
    return json.loads(report.read_text(encoding="utf-8"))["checkpoint"]["path"]


def py_compile(env: dict[str, str]) -> None:
    files = [str(path) for path in ROOT.rglob("*.py")]
    subprocess.run([sys.executable, "-m", "py_compile", *files], cwd=ROOT, env=env, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--gpus", default="0", help="comma-separated GPU ids; the first is used by default")
    parser.add_argument("--jittor-home", type=Path)
    parser.add_argument("--cuda-home", type=Path)
    parser.add_argument("--mf-embedding-dim", type=int, default=512)
    parser.add_argument("--mf-negative-count", type=int, default=64)
    parser.add_argument("--mf-epochs", type=int, default=3)
    parser.add_argument("--quick", action="store_true", help="compile/Jittor smoke only; not submittable")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.data = args.data.resolve()
    work = args.work_dir.resolve()
    if sha256(args.data) != DATA_SHA256:
        raise ValueError("official data_B.zip SHA-256 differs")
    if work.exists() and not args.resume:
        raise FileExistsError(f"refusing work-directory reuse: {work}")
    work.mkdir(parents=True, exist_ok=args.resume)
    gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one id")
    if min(args.mf_embedding_dim, args.mf_negative_count, args.mf_epochs) < 1:
        raise ValueError("MF dimensions, negative count, and epochs must be positive")
    env = env_for(args, work, gpus[0])
    py_compile(env)
    if args.quick:
        receipt = {
            "kind": "third_1_code_only_smoke_v1",
            "decision": "SMOKE_ONLY",
            "data_sha256": DATA_SHA256,
            "note": "quick mode compiles code only and does not produce a submission",
        }
        (work / "REPRODUCTION_RECEIPT.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0

    logs = work / "logs"
    cache = work / "cache"

    ruc4_work = work / "ruc4_base"
    ruc4_zip = ruc4_work / "ruc4.zip"
    ruc4_command = [
        sys.executable,
        str(ROOT / "reproduce_ruc4.py"),
        "--data",
        str(args.data),
        "--work-dir",
        str(ruc4_work),
        "--gpus",
        args.gpus,
        *(["--jittor-home", str(args.jittor_home.resolve())] if args.jittor_home else []),
        *(["--cuda-home", str(args.cuda_home.resolve())] if args.cuda_home else []),
    ]
    if args.resume:
        ruc4_command.append("--resume")
    ruc4_log = logs / "01_reproduce_ruc4.log"
    if args.resume and ruc4_zip.is_file() and (ruc4_work / "REPRODUCTION_RECEIPT.json").is_file():
        print("SKIP", ruc4_zip, flush=True)
    else:
        if args.resume:
            ruc4_log.unlink(missing_ok=True)
        run(ruc4_command, ROOT, env, ruc4_log)

    train_cache = cache / "b_rank_data" / f"dataset4-{DATA_SHA256[:20]}"

    deploy_history = work / "deploy_history"
    deploy_testpool = work / "deploy_testpool"
    run_stage(
        resume=args.resume,
        expected=(deploy_history / "deploy_report.json",),
        cleanup=(deploy_history,),
        command=[
            sys.executable,
            "-m",
            "b_rank.d4_temporal_deploy",
            "--data",
            str(args.data),
            "--cache-dir",
            str(cache),
            "--run-dir",
            str(deploy_history),
            "--temporal",
            "temporal_h32_seed10",
            "32",
            "20260810",
            "--temporal",
            "temporal_h32_seed11",
            "32",
            "20260811",
            "--temporal",
            "temporal_h32_seed12",
            "32",
            "20260812",
            "--temporal",
            "temporal_h64_seed10",
            "64",
            "20260810",
            "--temporal",
            "temporal_h64_seed11",
            "64",
            "20260811",
            "--temporal",
            "temporal_h64_seed12",
            "64",
            "20260812",
            "--epochs",
            "5",
            "--batch-rows",
            "256",
        ],
        cwd=CODE,
        env=env,
        log=logs / "02_deploy_history_temporal.log",
    )
    run_stage(
        resume=args.resume,
        expected=(deploy_testpool / "deploy_report.json",),
        cleanup=(deploy_testpool,),
        command=[
            sys.executable,
            "-m",
            "b_rank.d4_testpool_temporal_deploy",
            "--data",
            str(args.data),
            "--cache-dir",
            str(cache),
            "--run-dir",
            str(deploy_testpool),
            "--seeds",
            "20260810",
            "20260811",
            "20260812",
            "--epochs",
            "5",
            "--batch-rows",
            "256",
        ],
        cwd=CODE,
        env=env,
        log=logs / "03_deploy_testpool_temporal.log",
    )

    mf_reports = []
    for seed in (20260810, 20260811, 20260812):
        run_dir = work / f"deploy_mf_{seed}"
        run_stage(
            resume=args.resume,
            expected=(run_dir / "deploy_report.json",),
            cleanup=(run_dir,),
            command=[
                sys.executable,
                "-m",
                "b_rank.d4_implicit_mf_deploy",
                "--data",
                str(args.data),
                "--cache-dir",
                str(cache),
                "--run-dir",
                str(run_dir),
                "--seed",
                str(seed),
                "--embedding-dim",
                str(args.mf_embedding_dim),
                "--negative-count",
                str(args.mf_negative_count),
                "--epochs",
                str(args.mf_epochs),
                "--batch-rows",
                "4096",
            ],
            cwd=CODE,
            env=env,
            log=logs / f"04_deploy_mf_{seed}.log",
        )
        mf_reports.append((f"fullhistory_mf_seed{str(seed)[-2:]}", run_dir / "deploy_report.json"))

    transition = work / "deploy_transition_mf_20260812"
    run_stage(
        resume=args.resume,
        expected=(transition / "deploy_report.json",),
        cleanup=(transition,),
        command=[
            sys.executable,
            "-m",
            "b_rank.d4_transition_mf_deploy",
            "--data",
            str(args.data),
            "--cache-dir",
            str(cache),
            "--run-dir",
            str(transition),
            "--seed",
            "20260812",
            "--embedding-dim",
            str(args.mf_embedding_dim),
            "--negative-count",
            str(args.mf_negative_count),
            "--epochs",
            str(args.mf_epochs),
            "--batch-rows",
            "4096",
        ],
        cwd=CODE,
        env=env,
        log=logs / "05_deploy_transition_mf.log",
    )

    history_report = deploy_history / "deploy_report.json"
    testpool_report = deploy_testpool / "deploy_report.json"
    control_fit = work / "control_fit"
    fit_common = [
        sys.executable,
        "-m",
        "b_rank.d4_multimodel_fit",
        "--data",
        str(args.data),
        "--cache-dir",
        str(cache),
        *temporal_args(history_report),
        *temporal_args(testpool_report),
    ]
    for name, report in mf_reports:
        fit_common += ["--mf", name, checkpoint(report)]
    fit_common += ["--transition-mf", "transition_mf_seed12", checkpoint(transition / "deploy_report.json")]
    run_stage(
        resume=args.resume,
        expected=(control_fit / "research_report.json",),
        cleanup=(control_fit,),
        command=[
            *fit_common,
            "--control-only",
            "--run-dir",
            str(control_fit),
            "--valid-groups",
            "60000",
            "--confirm-groups",
            "30000",
            "--batch-rows",
            "512",
        ],
        cwd=CODE,
        env=env,
        log=logs / "06_fit_control.log",
    )
    control_report = control_fit / "research_report.json"
    pairnew = work / "pairnew_fit"
    pair_members = []
    for hidden, seed in ((64, 20260815), (64, 20260816), (64, 20260817), (96, 20260818), (96, 20260819), (96, 20260820)):
        pair_members += ["--residual-member", str(hidden), str(seed)]
    run_stage(
        resume=args.resume,
        expected=(pairnew / "research_report.json",),
        cleanup=(pairnew,),
        command=[
            *fit_common,
            "--pairnew-transformer",
            "--control-fit",
            str(control_report),
            "--residual-train-rows",
            "20000",
            "--residual-epochs",
            "6",
            "--residual-batch-rows",
            "128",
            *pair_members,
            "--run-dir",
            str(pairnew),
            "--valid-groups",
            "60000",
            "--confirm-groups",
            "30000",
            "--batch-rows",
            "512",
        ],
        cwd=CODE,
        env=env,
        log=logs / "07_fit_pairnew.log",
    )
    pair_report = pairnew / "research_report.json"

    replay = work / "meta_replay"
    for strategy in ("history", "test_pool"):
        replay_run = work / f"meta_replay_build_{strategy}"
        replay_cache = replay / strategy
        run_stage(
            resume=args.resume,
            expected=(
                replay_run / "research_report.json",
                replay_cache / "manifest.json",
            ),
            cleanup=(replay_run, replay_cache),
            command=[
                sys.executable,
                str(RUC3 / "build_large_cache.py"),
                "--source-manifest",
                str(pair_report),
                "--data",
                str(args.data),
                "--cache-dir",
                str(cache),
                "--run-dir",
                str(replay_run),
                "--output-cache",
                str(replay_cache),
                "--strategy",
                strategy,
                "--valid-groups",
                "120000",
                "--confirm-groups",
                "30000",
                "--batch-rows",
                "512",
            ],
            cwd=RUC3,
            env=env,
            log=logs / f"08_replay_{strategy}.log",
        )
    identity = work / "meta_identity"
    run_stage(
        resume=args.resume,
        expected=(identity / "manifest.json",),
        cleanup=(identity,),
        command=[
            sys.executable,
            str(RUC3 / "build_identity_cache_v33.py"),
            "--data",
            str(args.data),
            "--data-cache",
            str(cache),
            "--score-cache",
            str(replay / "history"),
            "--score-cache",
            str(replay / "test_pool"),
            "--output",
            str(identity),
        ],
        cwd=RUC3,
        env=env,
        log=logs / "09_identity.log",
    )
    baseline = work / "meta_pairnew_baseline"
    run_stage(
        resume=args.resume,
        expected=(baseline / "manifest.json",),
        cleanup=(baseline,),
        command=[
            sys.executable,
            str(RUC3 / "build_baseline_cache.py"),
            "--replay-cache",
            str(replay / "history"),
            "--replay-cache",
            str(replay / "test_pool"),
            "--control-fit",
            str(control_report),
            "--pairnew-report",
            str(pair_report),
            "--output",
            str(baseline),
        ],
        cwd=RUC3,
        env=env,
        log=logs / "10_baseline_cache.log",
    )
    ruc4_fixed = work / "meta_ruc4_fixed_cache"
    run_stage(
        resume=args.resume,
        expected=(ruc4_fixed / "manifest.json",),
        cleanup=(ruc4_fixed,),
        command=[
            sys.executable,
            str(THIRD / "build_ruc4_fixed_cache.py"),
            "--rp3-code",
            str(RUC4),
            "--data-cache",
            str(train_cache),
            "--identity-cache",
            str(identity),
            "--replay-root",
            str(replay),
            "--input-cache",
            str(baseline),
            "--output-cache",
            str(ruc4_fixed),
            "--threads",
            str(min(48, os.cpu_count() or 1)),
        ],
        cwd=THIRD,
        env=env,
        log=logs / "11_ruc4_fixed_cache.log",
    )
    side = work / "meta_side"
    side_maps = work / "meta_side_maps"
    run_stage(
        resume=args.resume,
        expected=(
            side / "hierarchy" / "metadata.json",
            side / "neighbor" / "manifest.json",
        ),
        cleanup=(side, side_maps),
        command=[
            sys.executable,
            str(THIRD / "build_meta_side_features.py"),
            "--train-cache",
            str(train_cache),
            "--identity-cache",
            str(identity),
            "--hierarchy-output",
            str(side / "hierarchy"),
            "--neighbor-output",
            str(side / "neighbor"),
            "--map-cache",
            str(side_maps),
        ],
        cwd=THIRD,
        env=env,
        log=logs / "12_meta_side_features.log",
    )
    meta_features = work / "meta_features"
    run_stage(
        resume=args.resume,
        expected=(meta_features / "metadata.json",),
        cleanup=(meta_features,),
        command=[
            sys.executable,
            str(THIRD / "build_meta_feature_cache.py"),
            "--hierarchy-cache",
            str(side / "hierarchy"),
            "--neighbor-cache",
            str(side / "neighbor"),
            "--replay-root",
            str(replay),
            "--baseline-cache",
            str(ruc4_fixed),
            "--output",
            str(meta_features),
        ],
        cwd=THIRD,
        env=env,
        log=logs / "13_meta_feature_cache.log",
    )
    meta_model = work / "meta_model_h96_s20260822"
    run_stage(
        resume=args.resume,
        expected=(meta_model / "model.npz", meta_model / "report.json"),
        cleanup=(meta_model,),
        command=[
            sys.executable,
            str(THIRD / "train_hierarchy_jittor.py"),
            "--feature-cache",
            str(meta_features),
            "--replay-cache",
            str(replay),
            "--identity-cache",
            str(identity),
            "--baseline-cache",
            str(ruc4_fixed),
            "--output",
            str(meta_model),
            "--train-rows",
            "70000",
            "--selection-rows",
            "30000",
            "--epochs",
            "8",
            "--batch",
            "128",
            "--hidden",
            "96",
            "--seed",
            "20260822",
        ],
        cwd=THIRD,
        env=env,
        log=logs / "14_train_meta.log",
    )
    meta_gate_report = work / "meta_gate_report.json"
    run_stage(
        resume=args.resume,
        expected=(meta_gate_report,),
        cleanup=(meta_gate_report,),
        command=[
            sys.executable,
            str(THIRD / "evaluate_gated_meta.py"),
            "--code-root",
            str(THIRD),
            "--feature-cache",
            str(meta_features),
            "--replay-cache",
            str(replay),
            "--identity-cache",
            str(identity),
            "--baseline-cache",
            str(ruc4_fixed),
            "--model",
            str(meta_model / "model.npz"),
            "--output",
            str(meta_gate_report),
            "--train-rows",
            "70000",
            "--selection-rows",
            "30000",
        ],
        cwd=THIRD,
        env=env,
        log=logs / "15_evaluate_meta_gate.log",
    )

    output = work / "result.zip"
    final_report = work / "final_report.json"
    infer = [
        sys.executable,
        str(THIRD / "infer_meta_ruc4.py"),
        "--data",
        str(args.data),
        "--base",
        str(ruc4_zip),
        "--output",
        str(output),
        "--report-output",
        str(final_report),
        "--cache-dir",
        str(cache),
        "--train-cache",
        str(train_cache),
        "--map-cache",
        str(work / "meta_formal_maps"),
        "--meta-model",
        str(meta_model / "model.npz"),
        "--code-root",
        str(CODE),
        "--control-fit",
        str(control_report),
        "--pairnew-report",
        str(pair_report),
        "--temporal-report",
        str(history_report),
        "--temporal-report",
        str(testpool_report),
        "--transition-mf-report",
        "transition_mf_seed12",
        str(transition / "deploy_report.json"),
        "--scale",
        "0.2",
        "--threshold",
        "0.08892796039581305",
        "--keep-same-top1",
        "--chunk-rows",
        "4096",
        "--predict-batch",
        "512",
    ]
    for name, report in mf_reports:
        infer += ["--mf-report", name, str(report)]
    run_stage(
        resume=args.resume,
        expected=(output, final_report),
        cleanup=(output, final_report, work / "meta_formal_maps"),
        command=infer,
        cwd=THIRD,
        env=env,
        log=logs / "16_infer_third_1.log",
    )
    final = json.loads(final_report.read_text(encoding="utf-8"))
    receipt = {
        "kind": "third_1_code_only_reproduction_receipt_v1",
        "decision": final["decision"],
        "data_sha256": DATA_SHA256,
        "submission_sha256": sha256(output),
        "online_reference_sha256": ONLINE_SHA256,
        "exact_online_sha256": sha256(output) == ONLINE_SHA256,
        "neural_framework": "Jittor",
        "weights_in_package": False,
        "external_data_used": False,
        "teacher_answers_used": False,
        "mf_embedding_dim": args.mf_embedding_dim,
        "mf_negative_count": args.mf_negative_count,
        "mf_epochs": args.mf_epochs,
        "meta_diagnostics": {
            "training": json.loads(
                (meta_model / "report.json").read_text(encoding="utf-8")
            )["decision"],
            "row_gate": json.loads(
                meta_gate_report.read_text(encoding="utf-8")
            )["decision"],
            "role": "recorded diagnostics; formal inference uses the pinned deployment policy",
        },
        "meta_deployment_policy": {
            "scale": 0.2,
            "threshold": 0.08892796039581305,
            "keep_same_top1": True,
        },
        "final_report": str(final_report),
        "output": str(output),
    }
    (work / "REPRODUCTION_RECEIPT.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
    return 0 if final["decision"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
