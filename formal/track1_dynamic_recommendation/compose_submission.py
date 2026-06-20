"""Compose a Track 1 submission from per-scene source zips."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


SCENES = ("dataset1", "dataset2")


def parse_scene_sources(value: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for chunk in value.split(";"):
        if not chunk.strip():
            continue
        scene, path = chunk.split("=", 1)
        scene = scene.strip()
        if scene not in SCENES:
            raise ValueError(f"Unknown scene: {scene}")
        result[scene] = Path(path.strip())
    missing = [scene for scene in SCENES if scene not in result]
    if missing:
        raise ValueError(f"Missing scenes: {', '.join(missing)}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compose dataset1/dataset2 CSVs from source zips")
    parser.add_argument("--scene-sources", required=True, help="dataset1=/a.zip;dataset2=/b.zip")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = parse_scene_sources(args.scene_sources)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as output_zip:
        for scene, source in sources.items():
            with zipfile.ZipFile(source) as archive:
                output_zip.writestr(f"{scene}.csv", archive.read(f"{scene}.csv"))
    print(f"composed submission saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
