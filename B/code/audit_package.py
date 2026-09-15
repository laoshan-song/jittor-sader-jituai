#!/usr/bin/env python3
"""Fail-closed static audit for the compact locked reconstruction package."""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import lzma
import tempfile
from pathlib import Path

import numpy as np


MODEL_BITS = 8
MODEL_SHA256 = "98dc703a0851229f38b43f588b709c1b1aeff98ab60570a1ca61d8e617eb31f4"
MODEL_DECODED_SHA256 = {
    "source.weight": "3c14ca3e783338e86c19b24e3b4c51c3cf804d1405552cc5d8d69780ff00a4cd",
    "item.weight": "7089a2055826c4d90e87e8e4c4c53b42bba784c00ec1d2bcc0316b10fb11b182",
    "item_bias.weight": "67833cfb2292edd51e4f361e9f5cb3965130fce26d97b32c015b349f7a91c197",
}
BASE_SHA256 = "e46182a6114b0089b9e05d03672b93c28758624ef02b7d97357b1994cddf3d18"
BASE_BYTES = 257_814_859
BASE_PARTS = tuple(
    f"code/assets/locked/frozen_base.ckpt.part{suffix}"
    for suffix in ("aa", "ab", "ac", "ad")
)
MODEL_PATH = "code/assets/locked/d4_implicit_mf32.npz"
A_REFERENCE_SHA256 = "d159f406a4b6376eb29ebe5b7d54c2481094706dd9f6c67987792cb62966e275"
TARGET_SHA256 = "9a8867eed4bc8a63c203a82ec4e4d5b37c01ebd57894c39c88296334fc13d9ba"
Q7_D4_MAGIC = b"TRACK1-B-D4-Q7-V1\n"
Q7_D4_SHA256 = "c78eed368107a3ea90eb116168596f80c1d816666a9ea6f89b880052cb4a54c7"
D3_BASE_SHA256 = "35f416ef441e8ffefe8492da9c0bec8e8bd8218b73e87918e25837ba1140c53d"
D3_BASE_BYTES = 206_768_126
ROWS = 2_322_538
D3_ROWS = 157_670
WIDTH = 100
REQUIRED = {
    "README.md",
    "AB_CHANGES.md",
    "A_LIST_REFERENCE.md",
    "requirements.txt",
    "submission_metadata.json",
    *BASE_PARTS,
    MODEL_PATH,
    "code/main.py",
    "code/model.py",
    "code/build_submission.py",
    "code/restore_locked_assets.py",
    "code/train_model.py",
    "code/check_environment.py",
    "code/audit_package.py",
    "code/build_manifest.py",
    "code/prepare_cuda_runtime.sh",
    "run_inference.sh",
    "run_verify.sh",
    "run_train.sh",
    "run_fresh_inference.sh",
    "run_generate_base.sh",
    "code/pipeline/README.md",
    "code/pipeline/reproduce.py",
    "code/pipeline/reproduce_third_1.py",
    "code/pipeline/generate_frozen_base.py",
    "code/pipeline/pack_frozen_base.py",
    "code/raw_training/README.md",
    "code/raw_training/main.py",
    "code/raw_training/build_base.py",
    "code/raw_training/verify_fresh_run.py",
    "code/experiments/locked_reproduction_receipt.json",
    "code/experiments/fresh_training_receipt.json",
    "code/experiments/fresh_inference_receipt.json",
    "MANIFEST.sha256",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def restored_frozen_base(root: Path):
    handle = tempfile.TemporaryFile()
    digest = hashlib.sha256()
    written = 0
    for relative in BASE_PARTS:
        path = root / relative
        with path.open("rb") as source:
            for block in iter(lambda: source.read(8 << 20), b""):
                handle.write(block)
                digest.update(block)
                written += len(block)
    if written != BASE_BYTES or digest.hexdigest() != BASE_SHA256:
        handle.close()
        raise ValueError("tracked frozen-base parts differ")
    handle.seek(0)
    return handle


def imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.split(".", 1)[0])
    return found


def decode_dataset3_q35(payload: np.ndarray) -> bytes:
    count = D3_ROWS * WIDTH
    plane_bytes = (count + 7) // 8
    encoded = lzma.decompress(np.asarray(payload, dtype=np.uint8).tobytes())
    if len(encoded) != 35 * plane_bytes:
        raise ValueError("Dataset3 q35 score stream size differs")
    planes = np.frombuffer(encoded, dtype=np.uint8).reshape(35, plane_bytes)
    zigzag = np.zeros(count, dtype=np.uint64)
    for bit in range(35):
        values = np.unpackbits(planes[bit], bitorder="little", count=count)
        zigzag |= values.astype(np.uint64) << bit
    fixed = (zigzag >> 1).astype(np.int64) ^ -(zigzag & 1).astype(np.int64)
    output = io.BytesIO()
    np.savetxt(
        output,
        fixed.reshape(D3_ROWS, WIDTH).astype(np.float64) / 10_000_000_000,
        delimiter=",",
        fmt="%.10f",
    )
    return output.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    files = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    missing = sorted(REQUIRED - files)
    if missing:
        raise ValueError(f"required files missing: {missing}")
    if any(name in files for name in ("result.zip", "data_B.zip", "dataset3.csv", "dataset4.csv")):
        raise ValueError("final result or official data is packaged")
    manifest = {}
    for line in (root / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        manifest[relative] = digest
    expected_manifest_files = files - {"MANIFEST.sha256"}
    if set(manifest) != expected_manifest_files:
        raise ValueError("manifest file set differs")
    for relative, expected in manifest.items():
        if sha256(root / relative) != expected:
            raise ValueError(f"manifest hash differs: {relative}")
    if sha256(root / MODEL_PATH) != MODEL_SHA256:
        raise ValueError("Jittor checkpoint hash differs")
    with restored_frozen_base(root) as frozen_base, np.load(
        frozen_base, allow_pickle=False
    ) as archive:
        if archive.files != ["kind", "dataset3_q35_lzma", "dataset4_q7"]:
            raise ValueError("frozen checkpoint members differ")
        if str(archive["kind"].item()) != "track1_b_frozen_score_q7_d3q35_v1":
            raise ValueError("frozen checkpoint kind differs")
        d3_q35 = np.asarray(archive["dataset3_q35_lzma"], dtype=np.uint8)
        d4 = np.asarray(archive["dataset4_q7"], dtype=np.uint8)
        d3 = decode_dataset3_q35(d3_q35)
        expected_size = len(Q7_D4_MAGIC) + ROWS * WIDTH * 7 // 8
        if (
            d3_q35.ndim != 1
            or len(d3) != D3_BASE_BYTES
            or hashlib.sha256(d3).hexdigest() != D3_BASE_SHA256
            or d4.ndim != 1
            or d4.size != expected_size
            or hashlib.sha256(d4.tobytes()).hexdigest() != Q7_D4_SHA256
        ):
            raise ValueError("frozen checkpoint payload size differs")
        if d4[: len(Q7_D4_MAGIC)].tobytes() != Q7_D4_MAGIC:
            raise ValueError("packed Dataset4 base magic differs")
    with np.load(root / MODEL_PATH, allow_pickle=False) as checkpoint:
        expected_kind = f"d4_implicit_mf_q{MODEL_BITS}row_v1"
        expected_model_files = [
            "kind", "source_count", "item_count", "embedding_dim",
            "source_ids_delta", "item_ids_delta",
            "param__source.weight_q", "param__source.weight_scale",
            "param__item.weight_q", "param__item.weight_scale",
            "param__item_bias.weight_q", "param__item_bias.weight_scale",
        ]
        if checkpoint.files != expected_model_files or str(checkpoint["kind"].item()) != expected_kind:
            raise ValueError("quantized checkpoint kind differs")
        source_ids = np.cumsum(checkpoint["source_ids_delta"], dtype=np.uint64).astype(np.uint32)
        item_ids = np.cumsum(checkpoint["item_ids_delta"], dtype=np.uint64).astype(np.uint32)
        parameter_shapes = {}
        decoded_parameter_hashes = {}
        for name in ("source.weight", "item.weight", "item_bias.weight"):
            quantized = np.asarray(checkpoint[f"param__{name}_q"])
            scale = np.asarray(checkpoint[f"param__{name}_scale"])
            qmax = (1 << (MODEL_BITS - 1)) - 1
            if (
                quantized.dtype != np.int8
                or scale.dtype != np.float32
                or int(quantized.min()) < -qmax
                or int(quantized.max()) > qmax
                or not np.isfinite(scale).all()
                or np.any(scale < 0)
            ):
                raise ValueError("quantized checkpoint dtype differs")
            decoded = quantized.astype(np.float32) * scale
            parameter_shapes[name] = list(decoded.shape)
            decoded_parameter_hashes[name] = hashlib.sha256(decoded.tobytes()).hexdigest()
        if parameter_shapes != {
            "source.weight": [680641, 32],
            "item.weight": [862247, 32],
            "item_bias.weight": [862247, 1],
        }:
            raise ValueError("checkpoint parameter shapes differ")
        if (
            decoded_parameter_hashes != MODEL_DECODED_SHA256
            or len(source_ids) + 1 != int(checkpoint["source_count"].item())
            or len(item_ids) + 1 != int(checkpoint["item_count"].item())
            or not np.all(source_ids[1:] > source_ids[:-1])
            or not np.all(item_ids[1:] > item_ids[:-1])
        ):
            raise ValueError("quantized checkpoint decode differs")
    source_imports = set().union(*(imports(path) for path in (root / "code").glob("*.py")))
    raw_training_sources = sorted((root / "code" / "raw_training").glob("*.py"))
    source_imports.update(*(imports(path) for path in raw_training_sources))
    if "jittor" not in source_imports:
        raise ValueError("Jittor import is absent")
    readme = (root / "README.md").read_text(encoding="utf-8")
    required_readme_text = (
        "Byte-for-byte reproduction of the online submission",
        "Ubuntu 22.04",
        "CUDA 12.4",
        "Python 3.10",
        "Jittor 1.3.10.0",
        "python -m pip install -r requirements.txt",
        "Data and label declaration",
        "Reproduction boundary",
        "The reviewer supplies only the official `data_B.zip`",
        "generated_frozen_base_sha256",
        TARGET_SHA256,
    )
    if any(value not in readme for value in required_readme_text):
        raise ValueError("README environment, label, or byte-parity contract is incomplete")
    requirements = {
        line.strip()
        for line in (root / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if requirements != {
        "numpy==1.26.4",
        "pandas==2.2.3",
        "numba==0.66.0",
        "jittor==1.3.10.0",
        "jittor-geometric>=0.1.0",
        "nvidia-cudnn-cu12==8.9.7.29",
        "scikit-learn==1.5.2",
    }:
        raise ValueError("requirements.txt differs from the target environment")
    metadata = json.loads((root / "submission_metadata.json").read_text(encoding="utf-8"))
    if (
        metadata.get("official_data_only") is not True
        or metadata.get("test_ground_truth_used") is not False
        or metadata.get("external_data_used") is not False
        or metadata.get("external_predictions_used") is not False
        or "Ubuntu 22.04" not in metadata.get("environment", "")
        or "Python 3.10" not in metadata.get("environment", "")
        or "Jittor 1.3.10.0" not in metadata.get("environment", "")
        or "CUDA 12.4" not in metadata.get("cuda_compatibility", "")
    ):
        raise ValueError("submission metadata environment or data declaration differs")
    raw_readme = (root / "code/raw_training/README.md").read_text(encoding="utf-8")
    if (
        "base_result.zip" not in raw_readme
        or "byte-for-byte reproduction" not in raw_readme
        or "test-set ground truth" not in raw_readme
    ):
        raise ValueError("raw-training reproduction boundary is undocumented")
    fresh_launcher = (root / "run_fresh_inference.sh").read_text(encoding="utf-8")
    if "code/raw_training/build_base.py" not in fresh_launcher or "--data \"$DATA\"" not in fresh_launcher:
        raise ValueError("fresh inference does not rebuild its official-data base")
    fresh_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [root / "run_train.sh", root / "run_fresh_inference.sh", *raw_training_sources]
    )
    if "models/frozen_base.ckpt" in fresh_sources:
        raise ValueError("fresh training or inference references the locked base")
    for launcher in (
        "run_inference.sh",
        "run_train.sh",
        "run_fresh_inference.sh",
        "run_generate_base.sh",
    ):
        source = (root / launcher).read_text(encoding="utf-8")
        if "prepare_cuda_runtime.sh" not in source or "check_environment.py" not in source:
            raise ValueError(f"CUDA preparation or environment check is absent: {launcher}")
    generate_launcher = (root / "run_generate_base.sh").read_text(encoding="utf-8")
    if 'dirname "${BASH_SOURCE[0]}")" && pwd' not in generate_launcher:
        raise ValueError("frozen-base launcher does not resolve the B package root")
    if "--bridge-locked" not in generate_launcher:
        raise ValueError("frozen-base launcher does not enable the exact-state bridge")
    bridge_source = (root / "code/pipeline/generate_frozen_base.py").read_text(encoding="utf-8")
    if any(
        token not in bridge_source
        for token in (
            "generated_frozen_base_sha256",
            "restore_locked_assets.py",
            "build_submission.py",
            TARGET_SHA256,
        )
    ):
        raise ValueError("frozen-base exact bridge contract is incomplete")
    locked_launcher = (root / "run_inference.sh").read_text(encoding="utf-8")
    if "restore_locked_assets.py" not in locked_launcher or MODEL_PATH not in locked_launcher:
        raise ValueError("locked inference does not restore the tracked assets")
    reference = (root / "A_LIST_REFERENCE.md").read_text(encoding="utf-8")
    if A_REFERENCE_SHA256 not in reference:
        raise ValueError("A-list reference is incomplete")
    locked = json.loads((root / "code/experiments/locked_reproduction_receipt.json").read_text(encoding="utf-8"))
    fresh_training = json.loads(
        (root / "code/experiments/fresh_training_receipt.json").read_text(encoding="utf-8")
    )
    fresh_inference = json.loads(
        (root / "code/experiments/fresh_inference_receipt.json").read_text(encoding="utf-8")
    )
    if (
        locked.get("decision") != "PASS"
        or locked.get("result_sha256") != TARGET_SHA256
        or locked.get("checkpoint_sha256") != MODEL_SHA256
        or locked.get("base_sha256") != BASE_SHA256
        or locked.get("byte_exact_online_result") is not True
    ):
        raise ValueError("locked full-run receipt differs")
    if (
        fresh_training.get("decision") != "PASS_SUPPLEMENTARY_TRAINING"
        or fresh_inference.get("decision") != "PASS_SUPPLEMENTARY_INFERENCE"
        or not fresh_inference.get("checkpoint_sha256")
        or not fresh_inference.get("result_sha256")
    ):
        raise ValueError("supplementary receipts are incomplete")
    report = {
        "decision": "PASS",
        "file_count": len(files),
        "base_sha256": BASE_SHA256,
        "frozen_base_path": "restored from code/assets/locked/frozen_base.ckpt.part*",
        "checkpoint_sha256": MODEL_SHA256,
        "final_result_packaged": False,
        "official_data_packaged": False,
        "official_data_only_declared": True,
        "test_ground_truth_used": False,
        "external_data_used": False,
        "external_predictions_used": False,
        "target_environment": metadata["environment"],
        "cuda_compatibility": metadata["cuda_compatibility"],
        "raw_training_present": True,
        "locked_result_sha256": TARGET_SHA256,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
