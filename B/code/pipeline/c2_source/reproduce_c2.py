#!/usr/bin/env python3
"""Reproduce online c2 from official data through Jittor training and ZIP output."""

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
ROOT = Path(__file__).resolve().parent
CODE = ROOT / "code"
A_CODE = CODE / "b_rank_a_port"
C2_CODE = CODE / "c2_d3"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], cwd: Path, env: dict[str, str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    print("RUN", " ".join(command), flush=True)
    with log.open("x", encoding="utf-8") as handle:
        subprocess.run(command, cwd=cwd, env=env, stdout=handle,
                       stderr=subprocess.STDOUT, check=True)


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


def env_for(
    runtime: Path,
    gpu: int | None,
    jittor_home: Path | None,
    cuda_home: Path | None,
) -> dict[str, str]:
    home = runtime / "home"
    cache = jittor_home or runtime / "jittor"
    home.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "use_cutt": "0",
            "use_cutlass": "0",
            "use_nccl": "0",
            "use_mkl": "0",
            "PYTHONPATH": os.pathsep.join(map(str, (CODE, A_CODE, C2_CODE))),
            "JT_USE_CUDA": "1",
            "ML_CACHE_ROOT": str(home),
            "JITTOR_HOME": str(cache),
        }
    )
    if cuda_home is not None:
        env["CUDA_HOME"] = str(cuda_home)
        env["PATH"] = os.pathsep.join((str(cuda_home / "bin"), env.get("PATH", "")))
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            (str(cuda_home / "lib64"), env.get("LD_LIBRARY_PATH", ""))
        )
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def train_grid(data: Path, scene: str, output: Path, quick: bool, env: dict[str, str], logs: Path) -> None:
    values = (256, 256, 1, 256, 256, 1) if quick else (120000, 30000, 16, 160000, 30000, 16)
    command = [sys.executable, "train_grid.py", "--data", str(data), "--scene", scene,
               "--output-root", str(output), "--groups", str(values[0]),
               "--valid-groups", str(values[1]), "--epochs", str(values[2]),
               "--rank-groups", str(values[3]), "--rank-valid", str(values[4]),
               "--rank-epochs", str(values[5]), "--batch", "64" if quick else "256"]
    run(command, A_CODE, env, logs / f"train_{scene}.log")


def d3_models(root: Path) -> list[Path]:
    return [root / f"dataset3_{variant}_{seed}"
            for variant in ("raw", "cf", "hist_cf")
            for seed in (20260810, 20260811, 20260812)]


def report_checkpoints(report: Path, key: str = "checkpoints") -> list[dict]:
    value = json.loads(report.read_text(encoding="utf-8"))
    return value[key]


def temporal_specs(report: Path) -> list[tuple[str, int, str]]:
    return [(str(x["name"]), int(x["history_size"]), str(x["path"]))
            for x in report_checkpoints(report)]


def checkpoint_path(report: Path) -> str:
    value = json.loads(report.read_text(encoding="utf-8"))
    return str(value["checkpoint"]["path"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--jittor-home", type=Path)
    parser.add_argument("--cuda-home", type=Path)
    parser.add_argument("--quick", action="store_true", help="small smoke; not score-authorized")
    parser.add_argument("--resume", action="store_true", help="reuse completed stage artifacts")
    args = parser.parse_args()
    data = args.data.resolve()
    work = args.work_dir.resolve()
    if sha256(data) != DATA_SHA256:
        raise ValueError("official data_B.zip hash differs")
    if work.exists() and not args.resume:
        raise FileExistsError(f"refusing work-directory reuse: {work}")
    work.mkdir(parents=True, exist_ok=args.resume)
    logs = work / "logs"
    models = work / "models"
    reports = work / "reports"
    jittor_home = args.jittor_home.resolve() if args.jittor_home else None
    cuda_home = args.cuda_home.resolve() if args.cuda_home else None
    aenv = env_for(work / "runtime" / "d3", args.gpu, jittor_home, cuda_home)
    denv = env_for(work / "runtime" / "d4", args.gpu, jittor_home, cuda_home)

    d3_report = reports / "dataset3_ensemble.json"
    d3_source = work / "dataset3_source.zip"
    d3_manifest = work / "dataset3_source.manifest.json"
    if not (
        args.resume
        and d3_report.is_file()
        and d3_source.is_file()
        and d3_manifest.is_file()
    ):
        if args.resume:
            for path in (
                models / "dataset3",
                d3_report,
                d3_source,
                d3_manifest,
                logs / "train_dataset3.log",
                logs / "fit_dataset3.log",
                logs / "infer_dataset3.log",
            ):
                remove_path(path)
        # D3 control is trained from official data, then exposed as a source ZIP for D4.
        train_grid(data, "dataset3", models / "dataset3", args.quick, aenv, logs)
        command = [sys.executable, "fit_ensemble.py", "--data", str(data), "--scene", "dataset3",
                   "--models", *map(str, d3_models(models / "dataset3")), "--output", str(d3_report)]
        if args.quick:
            command.extend(["--meta-groups", "128", "--valid-groups", "128", "--confirm-groups", "128", "--batch", "32"])
        run(command, A_CODE, aenv, logs / "fit_dataset3.log")
        run([sys.executable, str(ROOT / "code" / "infer_d3_source.py"), "--data", str(data),
             "--report", str(d3_report), "--output", str(d3_source), "--manifest", str(d3_manifest)],
            ROOT, aenv, logs / "infer_dataset3.log")
    else:
        print("SKIP completed Dataset3 stages", flush=True)

    # D4 causal temporal, test-pool temporal, MF and transition members.
    common = ["--data", str(data), "--cache-dir", str(work / "cache")]
    h32 = work / "d4_h32"; h64 = work / "d4_h64"; pool = work / "d4_testpool"
    mf = []
    for seed in (20260810, 20260811, 20260812):
        mf.append(work / f"d4_mf_{seed}")
    transition = work / "d4_transition"
    run_stage(
        resume=args.resume, expected=(h32 / "deploy_report.json",), cleanup=(h32,),
        command=[sys.executable, "-m", "b_rank.d4_temporal_deploy", *common, "--run-dir", str(h32),
                 "--temporal", "temporal_h32_seed10", "32", "20260810", "--temporal", "temporal_h32_seed11", "32", "20260811", "--temporal", "temporal_h32_seed12", "32", "20260812",
                 "--epochs", "1" if args.quick else "5", "--batch-rows", "64" if args.quick else "256"],
        cwd=CODE, env=denv, log=logs / "deploy_h32.log",
    )
    run_stage(
        resume=args.resume, expected=(h64 / "deploy_report.json",), cleanup=(h64,),
        command=[sys.executable, "-m", "b_rank.d4_temporal_deploy", *common, "--run-dir", str(h64),
                 "--temporal", "temporal_h64_seed10", "64", "20260810", "--temporal", "temporal_h64_seed11", "64", "20260811", "--temporal", "temporal_h64_seed12", "64", "20260812",
                 "--epochs", "1" if args.quick else "5", "--batch-rows", "64" if args.quick else "256"],
        cwd=CODE, env=denv, log=logs / "deploy_h64.log",
    )
    run_stage(
        resume=args.resume, expected=(pool / "deploy_report.json",), cleanup=(pool,),
        command=[sys.executable, "-m", "b_rank.d4_testpool_temporal_deploy", *common, "--run-dir", str(pool), "--seeds", "20260810", "20260811", "20260812",
                 "--epochs", "1" if args.quick else "5", "--batch-rows", "64" if args.quick else "256"],
        cwd=CODE, env=denv, log=logs / "deploy_testpool.log",
    )
    for seed, destination in zip((20260810, 20260811, 20260812), mf):
        run_stage(
            resume=args.resume, expected=(destination / "deploy_report.json",), cleanup=(destination,),
            command=[sys.executable, "-m", "b_rank.d4_implicit_mf_deploy", *common, "--run-dir", str(destination), "--seed", str(seed),
                     "--epochs", "1" if args.quick else "3", "--batch-rows", "512" if args.quick else "4096"],
            cwd=CODE, env=denv, log=logs / f"deploy_mf_{seed}.log",
        )
    run_stage(
        resume=args.resume, expected=(transition / "deploy_report.json",), cleanup=(transition,),
        command=[sys.executable, "-m", "b_rank.d4_transition_mf_deploy", *common, "--run-dir", str(transition),
                 "--epochs", "1" if args.quick else "3", "--batch-rows", "512" if args.quick else "4096"],
        cwd=CODE, env=denv, log=logs / "deploy_transition.log",
    )

    h32r, h64r, poolr = h32 / "deploy_report.json", h64 / "deploy_report.json", pool / "deploy_report.json"
    fit_common = [sys.executable, "-m", "b_rank.d4_multimodel_fit", *common]
    for name, size, path in temporal_specs(h32r) + temporal_specs(h64r) + temporal_specs(poolr):
        fit_common += ["--temporal", name, str(size), path]
    for path in mf:
        seed = path.name.removeprefix("d4_mf_")
        fit_common += ["--mf", f"fullhistory_mf_seed{seed}", checkpoint_path(path / "deploy_report.json")]
    fit_common += ["--transition-mf", "transition_mf_seed12", checkpoint_path(transition / "deploy_report.json"),
                   "--valid-groups", "256" if args.quick else "60000", "--confirm-groups", "256" if args.quick else "30000",
                   "--batch-rows", "64" if args.quick else "512"]
    control_dir = work / "d4_control"
    control_report = control_dir / "research_report.json"
    run_stage(
        resume=args.resume, expected=(control_report,), cleanup=(control_dir,),
        command=fit_common + ["--control-only", "--run-dir", str(control_dir)],
        cwd=CODE, env=denv, log=logs / "fit_d4_control.log",
    )
    pair_common = fit_common + ["--pairnew-transformer", "--control-fit", str(control_report), "--residual-train-rows", "128" if args.quick else "20000", "--residual-epochs", "1" if args.quick else "6", "--residual-batch-rows", "64" if args.quick else "128", "--run-dir", str(work / "d4_pairnew")]
    if not args.quick:
        for hidden, seed in (
            (64, 20260815), (64, 20260816), (64, 20260817),
            (96, 20260818), (96, 20260819), (96, 20260820),
        ):
            pair_common += ["--residual-member", str(hidden), str(seed)]
    pair_report = work / "d4_pairnew" / "research_report.json"
    run_stage(
        resume=args.resume, expected=(pair_report,), cleanup=(work / "d4_pairnew",),
        command=pair_common, cwd=CODE, env=denv, log=logs / "fit_d4_pairnew.log",
    )

    infer = [sys.executable, "-m", "b_rank.d4_multimodel_infer", *common, "--fit-report", str(control_report), "--pairnew-report", str(pair_report), "--dataset3-source", str(d3_source), "--dataset3-manifest", str(d3_manifest), "--output", str(work / "control_v26.zip"), "--predict-batch-rows", "64" if args.quick else "512"]
    for report in (h32r, h64r, poolr): infer += ["--temporal-report", str(report)]
    for path in mf:
        seed = path.name.removeprefix("d4_mf_")
        infer += ["--mf-report", f"fullhistory_mf_seed{seed}", str(path / "deploy_report.json")]
    infer += ["--transition-mf-report", "transition_mf_seed12", str(transition / "deploy_report.json")]
    control_output = work / "control_v26.zip"
    control_manifest = control_output.with_suffix(".manifest.json")
    run_stage(
        resume=args.resume, expected=(control_output, control_manifest),
        cleanup=(control_output, control_manifest),
        command=infer, cwd=CODE, env=denv, log=logs / "infer_d4_control.log",
    )

    residual = reports / "d3_residual_v26.json"
    if not residual.is_file():
        residual.parent.mkdir(parents=True, exist_ok=True)
        residual.write_text(json.dumps({"kind": "d3_same_time_cross_source_residual_v26", "decision": "PASS", "policy": {"gate": "all", "weight": 0.10}}, indent=2, sort_keys=True) + "\n")
    v26_base = work / "v26_base.zip"
    v26_manifest = v26_base.with_suffix(".manifest.json")
    built_v26 = run_stage(
        resume=args.resume, expected=(v26_base, v26_manifest),
        cleanup=(v26_base, v26_manifest),
        command=[sys.executable, "d3_cross_source_v26.py", "build", "--data", str(data), "--base", str(work / "control_v26.zip"), "--base-manifest", str(work / "control_v26.manifest.json"), "--report", str(residual), "--output", str(v26_base)],
        cwd=A_CODE, env=aenv, log=logs / "build_v26.log",
    )
    if built_v26:
        (logs / "verify_v26.log").unlink(missing_ok=True)
        run([sys.executable, str(ROOT / "code" / "verify_v26_submission.py"), "--data", str(data), "--submission", str(v26_base)], ROOT, aenv, logs / "verify_v26.log")

    gate = reports / "c2_source_session_gate.json"
    run_stage(
        resume=args.resume, expected=(gate,), cleanup=(gate,),
        command=[
            sys.executable,
            "d3_near_time_gate.py",
            "--data",
            str(data),
            "--ensemble-report",
            str(d3_report),
            "--feature",
            "source_session",
            "--window",
            "300",
            "--residual-weight",
            "0.05",
            "--groups",
            "512" if args.quick else "30000",
            "--batch",
            "64" if args.quick else "256",
            "--output",
            str(gate),
        ],
        cwd=C2_CODE, env=aenv, log=logs / "gate_c2.log",
    )
    final = work / "b_rank_d34_c2_source_session.zip"
    final_manifest = final.with_suffix(".manifest.json")
    built_final = run_stage(
        resume=args.resume, expected=(final, final_manifest),
        cleanup=(final, final_manifest),
        command=[sys.executable, "build_c2_source_session.py", "--data", str(data), "--base", str(v26_base), "--base-manifest", str(v26_base.with_suffix(".manifest.json")), "--gate-report", str(gate), "--output", str(final)],
        cwd=C2_CODE, env=aenv, log=logs / "build_c2.log",
    )
    if built_final:
        (logs / "verify_c2.log").unlink(missing_ok=True)
        run([sys.executable, str(ROOT / "code" / "verify_v26_submission.py"), "--data", str(data), "--submission", str(final)], ROOT, aenv, logs / "verify_c2.log")
    receipt = {
        "kind": "b_rank_d34_c2_reproduction_receipt_v1",
        "decision": "SMOKE_ONLY" if args.quick else "PASS",
        "data_sha256": DATA_SHA256,
        "submission_sha256": sha256(final),
        "quick": bool(args.quick),
        "d3_policy": json.loads(gate.read_text())["fixed_policy"],
        "d4_control_report": str(control_report),
        "d4_pairnew_report": str(pair_report),
        "d4_diagnostics": {
            "control": json.loads(control_report.read_text())["decision"],
            "pairnew": json.loads(pair_report.read_text())["decision"],
            "role": "recorded diagnostics; formal inference uses the fitted reports",
        },
        "gate_report": str(gate),
    }
    (work / "REPRODUCTION_RECEIPT.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
