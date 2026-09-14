#!/usr/bin/env python3
"""Write the trained A-port D3 CSV in the bridge format consumed by D4."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np

from b_rank_a_port.infer_ensemble import PreparedScene, probabilities


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--chunk-rows", type=int, default=512)
    args = p.parse_args()
    scene = PreparedScene(args.data.resolve(), "dataset3", args.report.resolve())
    with zipfile.ZipFile(args.output, "x", zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        with z.open("dataset3.csv", "w", force_zip64=True) as raw:
            with io.TextIOWrapper(raw, encoding="ascii", newline="\n") as text:
                for start in range(0, len(scene.test), args.chunk_rows):
                    frame = scene.test.iloc[start:start + args.chunk_rows]
                    np.savetxt(text, probabilities(scene.score(frame, 256)), fmt="%.8f", delimiter=",")
        # The D4 inference bridge requires the canonical two-member ZIP root;
        # D4 is overwritten by the real D4 inference stage later.
        z.writestr("dataset4.csv", b"")
    d3_hash = sha256(args.output)
    args.manifest.write_text(json.dumps({
        "kind": "b_rank_a_port_submission_v1",
        "data_sha256": "ded8b0d281042323f0c5871868824038bc7fb675cc3e8211753bb63d8b7b89d2",
        "submission_sha256": d3_hash,
        "dataset3": {"active_component_count": len(scene.active_names), "weights": dict(zip(scene.active_names, scene.weights.tolist()))},
        "row_counts": {"dataset3": len(scene.test)},
    }, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
