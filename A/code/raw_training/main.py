#!/usr/bin/env python3
"""Track 1 reproducibility entry point.

The release profile reconstructs the recorded A-list submission from the
included Jittor checkpoints.  The raw profile trains the same public Jittor
components from ``data_A.zip`` into a fresh model directory.  Both paths write
all intermediate files below a caller-selected output directory.
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
from types import SimpleNamespace

CODE_ROOT = Path(__file__).resolve().parent
RELEASE_MODELS = CODE_ROOT / "artifacts" / "release_models"
EXPECTED_RESULT_SHA256 = "d36facee996b5d45806dd6e1d80f8a48883e505f57c8d8d842b9626a50e8e7ce"
EXPECTED_BASE_SHA256 = "4c8fca6a041a94957b04a2df9f958d76098d06ab67093679fea766c364e3a28f"
DATASET2_META_CHECKPOINTS = (
    "pool_ranker_seed20260816_jittor.npz",
    "pool_ranker_seed20260817_jittor.npz",
    "pool_ranker_seed20260818_jittor.npz",
    "set64_seed20262725_jittor.npz",
    "set64_seed20262726_jittor.npz",
    "set64_seed20262727_jittor.npz",
    "set96_seed20263725_jittor.npz",
    "set96_seed20263726_jittor.npz",
    "set96_seed20263727_jittor.npz",
    "multi_set64_seed20265701_jittor.npz",
    "multi_set64_seed20265702_jittor.npz",
    "multi_set64_seed20265703_jittor.npz",
    "multi_set96_seed20265801_jittor.npz",
    "multi_set96_seed20265802_jittor.npz",
    "multi_set96_seed20265803_jittor.npz",
    "multi_transformer2_set64_seed20265701_jittor.npz",
    "multi_transformer2_set64_seed20265702_jittor.npz",
    "multi_transformer2_set64_seed20265703_jittor.npz",
    "multi_transformer2_set96_seed20265801_jittor.npz",
    "multi_transformer2_set96_seed20265802_jittor.npz",
    "multi_transformer2_set96_seed20265803_jittor.npz",
    "warm_residual_set64_seed20265701_jittor.npz",
    "warm_residual_set64_seed20265702_jittor.npz",
    "warm_residual_set64_seed20265703_jittor.npz",
    "warm_residual_set96_seed20265801_jittor.npz",
    "warm_residual_set96_seed20265802_jittor.npz",
    "warm_residual_set96_seed20265803_jittor.npz",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_new_directory(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to reuse output path: {path}")
    path.mkdir(parents=True)


def runtime_env(*, cuda_requested: bool = True) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("ML_CACHE_ROOT", "/tmp")
    env.setdefault("log_silent", "1")
    env.setdefault("use_mpi", "0")
    if cuda_requested:
        # Several archived modules inspect this before importing Jittor and
        # otherwise deliberately blank nvcc_path.  Bind it for every child
        # process so a CUDA request cannot silently turn into a CPU build.
        env["JT_USE_CUDA"] = "1"
        env.setdefault("use_cutt", "0")
        env.setdefault("use_mkl", "0")
        if not env.get("nvcc_path"):
            nvcc = shutil.which("nvcc")
            if nvcc:
                env["nvcc_path"] = nvcc
    return env


def invoke(arguments: list[object], *, cwd: Path = CODE_ROOT, env: dict[str, str] | None = None) -> None:
    command = [str(value) for value in arguments]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env or runtime_env(), check=True)


def jittor_runtime_identity(python: str, *, cuda_requested: bool) -> dict[str, object]:
    env = runtime_env(cuda_requested=cuda_requested)
    cuda_setup = "assert jt.has_cuda, 'Jittor CUDA is unavailable'; jt.flags.use_cuda = 1; " if cuda_requested else ""
    probe = "import json, platform, sys, jittor as jt; " + cuda_setup + (
        "print(json.dumps({'python': sys.version.split()[0], 'platform': platform.platform(), "
        "'jittor': jt.__version__, 'has_cuda': bool(jt.has_cuda), "
        "'use_cuda': int(jt.flags.use_cuda)}, sort_keys=True))"
    )
    completed = subprocess.run(
        [python, "-c", probe],
        cwd=CODE_ROOT,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    for line in reversed(completed.stdout.splitlines()):
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise RuntimeError("Jittor runtime probe did not emit a JSON identity")


def model_layout(path: Path) -> tuple[Path, Path]:
    """Return the base and community roots for release or freshly trained weights."""
    path = path.resolve()
    base = path / "base_models" if (path / "base_models").is_dir() else path
    community = path / "community"
    return base, community


def model_inventory(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def learned_model_inventory(root: Path) -> dict[str, str]:
    """Hash learned artifacts while excluding post-training verification receipts."""
    excluded = {"training_manifest.json", "raw_training_verification.json"}
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and str(path.relative_to(root)) not in excluded
    }


def source_inventory() -> dict[str, str]:
    return {
        str(path.relative_to(CODE_ROOT)): sha256_file(path)
        for path in sorted(CODE_ROOT.rglob("*"))
        if path.is_file() and path.suffix in {".py", ".json"}
    }


def require_release_model_layout(base: Path, community: Path) -> None:
    required = [
        base / "legacy_dataset2_base.zip",
        base / "dataset1" / "d1_cf_seed_20260705.pkl",
        base / "dataset1" / "d1_cf_seed_20260715.pkl",
        base / "dataset2" / "model_multvae_prod_jittor.npz",
        base / "dataset2" / "model_recvae_prod_jittor.npz",
        base / "dataset2" / "model_recvae_h800z400_prod_jittor.npz",
        base / "dataset2" / "model_bm25bpr_prod_jittor.npz",
        community / "model_bpr32_prod_community_jittor.npz",
    ]
    required.extend(base / "dataset2" / name for name in DATASET2_META_CHECKPOINTS)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing model artifacts:\n" + "\n".join(missing))


def train_dataset1(data: Path, base: Path, python: str) -> None:
    output = base / "dataset1"
    output.mkdir(parents=True, exist_ok=False)
    for seed in (20260705, 20260715):
        invoke(
            [
                python,
                CODE_ROOT / "dataset1" / "train_d1_cf_jittor.py",
                "--data",
                data,
                "--output",
                output / f"d1_cf_seed_{seed}.pkl",
                "--seed",
                seed,
                "--groups",
                80000,
                "--valid",
                20000,
                "--epochs",
                16,
                "--batch",
                512,
            ]
        )


def train_dataset2(
    data: Path,
    base: Path,
    community: Path,
    python: str,
    *,
    legacy_quick: bool,
    cuda: bool,
) -> None:
    output = base / "dataset2"
    output.mkdir(parents=True, exist_ok=False)
    vae = CODE_ROOT / "dataset2" / "train_vae_jittor.py"
    bpr = CODE_ROOT / "dataset2" / "train_bpr_jittor.py"
    invoke([python, vae, "multvae", "prod", "--data", data, "--output", output / "model_multvae_prod_jittor.npz", "--epochs", 10, "--batch", 128, "--hidden", 600, "--latent", 200, "--dropout", 0.5, "--decay-days", 365, "--seed", 20260711])
    invoke([python, vae, "recvae", "prod", "--data", data, "--output", output / "model_recvae_prod_jittor.npz", "--cycles", 5, "--batch", 128, "--hidden", 600, "--latent", 200, "--dropout", 0.5, "--decay-days", 365, "--seed", 20260711])
    invoke([python, vae, "recvae", "prod", "--data", data, "--output", output / "model_recvae_h800z400_prod_jittor.npz", "--cycles", 5, "--batch", 96, "--hidden", 800, "--latent", 400, "--dropout", 0.5, "--decay-days", 365, "--seed", 20260711])
    invoke([python, bpr, "prod", "--data", data, "--output", output / "model_bm25bpr_prod_jittor.npz", "--factors", 256, "--epochs", 20, "--batch", 32768, "--negatives", 8, "--lr", "2e-3", "--regularization", "1e-4", "--decay-days", 365, "--bm25-k1", 100, "--bm25-b", "0.8", "--seed", 20260711])
    for split in ("y2009", "strict"):
        invoke([python, vae, "multvae", split, "--data", data, "--output", output / f"model_multvae_{split}_jittor.npz", "--epochs", 10, "--batch", 128, "--hidden", 600, "--latent", 200, "--dropout", 0.5, "--decay-days", 365, "--seed", 20260711])
        invoke([python, vae, "recvae", split, "--data", data, "--output", output / f"model_recvae_{split}_jittor.npz", "--cycles", 5, "--batch", 128, "--hidden", 600, "--latent", 200, "--dropout", 0.5, "--decay-days", 365, "--seed", 20260711])
        invoke([python, bpr, split, "--data", data, "--output", output / f"model_bm25bpr_{split}_jittor.npz", "--factors", 256, "--epochs", 20, "--batch", 32768, "--negatives", 8, "--lr", "2e-3", "--regularization", "1e-4", "--decay-days", 365, "--bm25-k1", 100, "--bm25-b", "0.8", "--seed", 20260711])
    invoke([python, CODE_ROOT / "dataset2" / "d2_pool_ranker_production_jittor.py", "--data", data, "--models", output, "--output-dir", output])
    invoke([python, CODE_ROOT / "dataset2" / "d2_v2_set_production_jittor.py", "--data", data, "--models", output, "--output-dir", output])
    invoke([python, CODE_ROOT / "dataset2" / "d2_multislice_set_raw_jittor.py", "--data", data, "--slice-models", output, "--prod-models", output, "--output-dir", output])
    invoke([python, CODE_ROOT / "dataset2" / "d2_multislice_transformer_raw_jittor.py", "--data", data, "--slice-models", output, "--prod-models", output, "--output-dir", output])
    invoke([python, CODE_ROOT / "dataset2" / "d2_warm_residual_raw_jittor.py", "--data", data, "--slice-models", output, "--output-dir", output])
    legacy_command: list[object] = [
        python,
        CODE_ROOT / "legacy_dataset2" / "rebuild_from_raw.py",
        "--data",
        data,
        "--output",
        base / "legacy_dataset2_base.zip",
        "--components-dir",
        base / "legacy_components",
    ]
    if legacy_quick:
        legacy_command.append("--quick")
    if cuda:
        legacy_command.append("--cuda")
    invoke(legacy_command)
    community.mkdir(parents=True, exist_ok=False)
    community_command: list[object] = [
        python,
        CODE_ROOT / "train_community_bpr_jittor.py",
        "--data",
        data,
        "--output",
        community / "model_bpr32_prod_community_jittor.npz",
    ]
    if cuda:
        community_command.append("--cuda")
    invoke(community_command)


def command_train(args: argparse.Namespace) -> None:
    if not args.cuda:
        raise RuntimeError("the raw training protocol requires --cuda with a CUDA-capable Jittor runtime")
    data = args.data.resolve()
    if not data.is_file():
        raise FileNotFoundError(data)
    model_root = args.output_models.resolve()
    if model_root.exists():
        raise FileExistsError(f"refusing to reuse model output: {model_root}")
    base = model_root / "base_models"
    community = model_root / "community"
    base.mkdir(parents=True)
    if args.dataset in ("dataset1", "all"):
        train_dataset1(data, base, args.python)
    if args.dataset in ("dataset2", "all"):
        train_dataset2(
            data,
            base,
            community,
            args.python,
            legacy_quick=args.legacy_quick,
            cuda=args.cuda,
        )
    manifest = {
        "kind": "track1_jittor_training_manifest_v2",
        "data": str(data),
        "data_sha256": sha256_file(data),
        "dataset": args.dataset,
        "cuda_requested": args.cuda,
        "model_root": str(model_root),
        "command": list(sys.argv),
        "requirements_sha256": sha256_file(CODE_ROOT.parent / "requirements.txt"),
        "jittor_runtime": jittor_runtime_identity(args.python, cuda_requested=args.cuda),
        "model_file_sha256": learned_model_inventory(model_root),
        "source_file_sha256": source_inventory(),
        "legacy_component_note": "raw legacy members are retained under base_models/legacy_components; exact archived legacy parity is not asserted",
    }
    (model_root / "training_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_base_inference(data: Path, base: Path, output: Path, python: str) -> Path:
    base_v1 = output / "base_v1"
    meta_v2 = output / "meta_v2"
    meta_multi = output / "meta_multislice"
    meta_transformer = output / "meta_transformer"
    base_v1.mkdir()
    invoke([python, CODE_ROOT / "dataset1" / "score_d1_ensemble_jittor.py", "--data", data, "--models", base / "dataset1" / "d1_cf_seed_20260705.pkl", base / "dataset1" / "d1_cf_seed_20260715.pkl", "--output-raw", base_v1 / "dataset1.npy", "--output-csv", base_v1 / "dataset1.csv"])
    for name in ("multvae", "recvae", "recvae_h800z400", "bm25bpr"):
        invoke([python, CODE_ROOT / "dataset2" / "score_model_jittor.py", "--data", data, "--model", base / "dataset2" / f"model_{name}_prod_jittor.npz", "--output", base_v1 / f"{name}.npy"])
    invoke([python, CODE_ROOT / "dataset2" / "d2_pool_ranker_production_jittor.py", "--data", data, "--models", base / "dataset2", "--output-dir", base_v1 / "meta", "--checkpoints", base / "dataset2" / "pool_ranker_seed20260816_jittor.npz", base / "dataset2" / "pool_ranker_seed20260817_jittor.npz", base / "dataset2" / "pool_ranker_seed20260818_jittor.npz"])
    invoke([python, CODE_ROOT / "build_result.py", "--data", data, "--dataset1", base_v1 / "dataset1.csv", "--legacy", base / "legacy_dataset2_base.zip", "--multvae", base_v1 / "multvae.npy", "--recvae", base_v1 / "recvae.npy", "--recvae-capacity", base_v1 / "recvae_h800z400.npy", "--bm25bpr", base_v1 / "bm25bpr.npy", "--meta", base_v1 / "meta" / "official_pool_ranker.npy", "--output", base_v1 / "result.zip"])
    invoke([python, CODE_ROOT / "dataset2" / "d2_v2_set_production_jittor.py", "--data", data, "--models", base / "dataset2", "--output-dir", meta_v2, "--cpu-inference", "--checkpoints", base / "dataset2" / "set64_seed20262725_jittor.npz", base / "dataset2" / "set64_seed20262726_jittor.npz", base / "dataset2" / "set64_seed20262727_jittor.npz", base / "dataset2" / "set96_seed20263725_jittor.npz", base / "dataset2" / "set96_seed20263726_jittor.npz", base / "dataset2" / "set96_seed20263727_jittor.npz"])
    v2 = output / "result_v2.zip"
    invoke([python, CODE_ROOT / "build_set_v2_result.py", "--baseline", base_v1 / "result.zip", "--old-meta", base_v1 / "meta" / "official_pool_ranker.npy", "--new-meta", meta_v2 / "official_set_all.npy", "--output", v2])
    invoke([python, CODE_ROOT / "dataset2" / "d2_multislice_set_production_jittor.py", "--data", data, "--prod-models", base / "dataset2", "--output-dir", meta_multi, "--cpu-inference", "--checkpoints", base / "dataset2" / "multi_set64_seed20265701_jittor.npz", base / "dataset2" / "multi_set64_seed20265702_jittor.npz", base / "dataset2" / "multi_set64_seed20265703_jittor.npz", base / "dataset2" / "multi_set96_seed20265801_jittor.npz", base / "dataset2" / "multi_set96_seed20265802_jittor.npz", base / "dataset2" / "multi_set96_seed20265803_jittor.npz"])
    multi = output / "result_multislice.zip"
    invoke([python, CODE_ROOT / "build_set_v2_result.py", "--baseline", v2, "--old-meta", meta_v2 / "official_set_all.npy", "--new-meta", meta_multi / "official_multislice_set_all.npy", "--output", multi])
    invoke([python, CODE_ROOT / "dataset2" / "d2_multislice_transformer_production_jittor.py", "--data", data, "--prod-models", base / "dataset2", "--output-dir", meta_transformer, "--checkpoints", base / "dataset2" / "multi_transformer2_set64_seed20265701_jittor.npz", base / "dataset2" / "multi_transformer2_set64_seed20265702_jittor.npz", base / "dataset2" / "multi_transformer2_set64_seed20265703_jittor.npz", base / "dataset2" / "multi_transformer2_set96_seed20265801_jittor.npz", base / "dataset2" / "multi_transformer2_set96_seed20265802_jittor.npz", base / "dataset2" / "multi_transformer2_set96_seed20265803_jittor.npz", "--warm-checkpoints", base / "dataset2" / "warm_residual_set64_seed20265701_jittor.npz", base / "dataset2" / "warm_residual_set64_seed20265702_jittor.npz", base / "dataset2" / "warm_residual_set64_seed20265703_jittor.npz", base / "dataset2" / "warm_residual_set96_seed20265801_jittor.npz", base / "dataset2" / "warm_residual_set96_seed20265802_jittor.npz", base / "dataset2" / "warm_residual_set96_seed20265803_jittor.npz"])
    final_base = output / "base_result.zip"
    invoke([python, CODE_ROOT / "build_set_v2_result.py", "--baseline", multi, "--old-meta", meta_multi / "official_multislice_set_all.npy", "--new-meta", meta_transformer / "official_multislice_transformer2_set_all.npy", "--output", final_base])
    return final_base


def command_infer(args: argparse.Namespace) -> None:
    data = args.data.resolve()
    if not data.is_file():
        raise FileNotFoundError(data)
    model_dir = args.model_dir.resolve()
    base, community = model_layout(model_dir)
    require_release_model_layout(base, community)
    output = args.output.resolve()
    require_new_directory(output)
    base_result = run_base_inference(data, base, output, args.python)
    if args.strict_release and sha256_file(base_result) != EXPECTED_BASE_SHA256:
        raise RuntimeError("release base result differs; refuse to claim the recorded A-list reconstruction")
    final = output / "result.zip"
    command = [args.python, CODE_ROOT / "build_final_submission.py", "--data", data, "--baseline", base_result, "--checkpoint", community / "model_bpr32_prod_community_jittor.npz", "--d1-config", CODE_ROOT / "dataset1" / "d1_source_support_locked.json", "--output", final, "--audit-output", output / "final_build_audit.json"]
    if args.strict_release:
        command.append("--strict-release")
    invoke(command)
    result_report = validate_submission(final, strict_release=args.strict_release)
    receipt = {
        "kind": "track1_inference_manifest_v1",
        "data_sha256": sha256_file(data),
        "model_dir": str(model_dir),
        "model_file_sha256": learned_model_inventory(model_dir),
        "source_file_sha256": source_inventory(),
        "requirements_sha256": sha256_file(CODE_ROOT.parent / "requirements.txt"),
        "strict_release": bool(args.strict_release),
        "result": result_report,
    }
    receipt_path = output / "inference_manifest.json"
    if receipt_path.exists():
        raise FileExistsError(f"refusing to overwrite inference receipt: {receipt_path}")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(final), "sha256": result_report["sha256"], "receipt": str(receipt_path)}, sort_keys=True), flush=True)


def validate_submission(result: Path, *, strict_release: bool) -> dict[str, object]:
    import numpy as np

    if not result.is_file():
        raise FileNotFoundError(result)
    with zipfile.ZipFile(result) as archive:
        if archive.testzip() is not None:
            raise ValueError("submission ZIP CRC failed")
        if archive.namelist() != ["dataset1.csv", "dataset2.csv"]:
            raise ValueError("submission member names differ")
        matrices = {
            name: np.loadtxt(archive.open(name), delimiter=",", dtype=np.float64)
            for name in archive.namelist()
        }
    expected = {"dataset1.csv": (61051, 100), "dataset2.csv": (153420, 100)}
    for name, matrix in matrices.items():
        if matrix.shape != expected[name]:
            raise ValueError(f"{name} shape differs: {matrix.shape}")
        if not np.isfinite(matrix).all() or np.any(matrix < 0.0):
            raise ValueError(f"{name} contains invalid probability values")
        if not np.allclose(matrix.sum(axis=1), 1.0, rtol=0.0, atol=5e-7):
            raise ValueError(f"{name} row sums differ")
    digest = sha256_file(result)
    if strict_release and digest != EXPECTED_RESULT_SHA256:
        raise ValueError(f"recorded A-list result hash differs: {digest}")
    return {"result": str(result), "sha256": digest, "rows": {name: value.shape[0] for name, value in matrices.items()}}


def command_verify(args: argparse.Namespace) -> None:
    report = validate_submission(args.result.resolve(), strict_release=args.strict_release)
    report["static_framework_audit"] = "run code/tools/audit_package.py for the complete source audit"
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


def command_all(args: argparse.Namespace) -> None:
    models = args.output_models.resolve()
    output = args.output.resolve()
    if models == output or models in output.parents or output in models.parents:
        raise ValueError("--output-models and --output must be disjoint directories")
    command_train(SimpleNamespace(data=args.data, output_models=args.output_models, dataset="all", python=args.python, legacy_quick=args.legacy_quick, cuda=args.cuda))
    command_infer(SimpleNamespace(data=args.data, output=args.output, model_dir=args.output_models, python=args.python, strict_release=False))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Track 1 Jittor-only training and inference")
    root.add_argument("--python", default=sys.executable, help="Python interpreter with the pinned Jittor environment")
    commands = root.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="fit models from official training data")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--output-models", type=Path, required=True)
    train.add_argument("--dataset", choices=("dataset1", "dataset2", "all"), default="all")
    train.add_argument("--legacy-quick", action="store_true", help="only for a smoke reconstruction; never an A-list equivalence claim")
    train.add_argument("--cuda", action="store_true", help="require CUDA for legacy and community Jittor training")
    train.set_defaults(func=command_train)
    infer = commands.add_parser("infer", help="generate a submission from Jittor checkpoints")
    infer.add_argument("--data", type=Path, required=True)
    infer.add_argument("--model-dir", type=Path, default=RELEASE_MODELS)
    infer.add_argument("--output", type=Path, required=True)
    infer.add_argument("--strict-release", action="store_true", help="require the recorded base and final A-list hashes")
    infer.set_defaults(func=command_infer)
    verify = commands.add_parser("verify", help="validate submission shape, probabilities, CRC, and optional release hash")
    verify.add_argument("--result", type=Path, required=True)
    verify.add_argument("--strict-release", action="store_true")
    verify.set_defaults(func=command_verify)
    all_command = commands.add_parser("all", help="freshly train all components and infer a new result")
    all_command.add_argument("--data", type=Path, required=True)
    all_command.add_argument("--output-models", type=Path, required=True)
    all_command.add_argument("--output", type=Path, required=True)
    all_command.add_argument("--legacy-quick", action="store_true")
    all_command.add_argument("--cuda", action="store_true")
    all_command.set_defaults(func=command_all)
    return root


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.func(arguments)
