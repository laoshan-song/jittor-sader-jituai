#!/usr/bin/env python3
"""Create the required Chinese PDF with only the Python standard library."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PAGE_WIDTH = 595
PAGE_HEIGHT = 842
LEFT = 48
TOP = 790
BOTTOM = 52


def pdf_hex(text: str) -> str:
    return "FEFF" + text.encode("utf-16-be").hex().upper()


def wrap(text: str, width: int = 44) -> list[str]:
    if len(text) <= width:
        return [text]
    result: list[str] = []
    remaining = text
    while len(remaining) > width:
        cut = remaining.rfind(" ", 0, width + 1)
        if cut < width // 2:
            cut = width
        result.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()
    if remaining:
        result.append(remaining)
    return result


def build_pages(lines: list[tuple[str, str]]) -> list[bytes]:
    pages: list[bytes] = []
    content: list[str] = []
    y = TOP
    for style, text in lines:
        if style == "space":
            y -= 8
            continue
        size = 18 if style == "title" else 13 if style == "heading" else 10
        step = 26 if style == "title" else 19 if style == "heading" else 14
        for part in wrap(text, 30 if style == "title" else 42):
            if y < BOTTOM:
                content.append("ET")
                pages.append("\n".join(content).encode("ascii"))
                content = []
                y = TOP
            if not content:
                content.append("BT")
            content.append(f"/F1 {size} Tf")
            content.append(f"1 0 0 1 {LEFT} {y} Tm")
            content.append(f"<{pdf_hex(part)}> Tj")
            y -= step
    if content:
        content.append("ET")
        pages.append("\n".join(content).encode("ascii"))
    return pages


def build_pdf(pages: list[bytes]) -> bytes:
    objects: dict[int, bytes] = {}
    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    font = b"<< /Type /Font /Subtype /Type0 /BaseFont /STSong-Light /Encoding /UniGB-UCS2-H /DescendantFonts [4 0 R] >>"
    descendant = b"<< /Type /Font /Subtype /CIDFontType0 /BaseFont /STSong-Light /CIDSystemInfo << /Registry (Adobe) /Ordering (GB1) /Supplement 5 >> /DW 1000 >>"
    objects[3] = font
    objects[4] = descendant
    page_numbers: list[int] = []
    for index, stream in enumerate(pages):
        page_number = 5 + 2 * index
        stream_number = page_number + 1
        page_numbers.append(page_number)
        objects[page_number] = f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] /Resources << /Font << /F1 3 0 R >> >> /Contents {stream_number} 0 R >>".encode("ascii")
        objects[stream_number] = b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream"
    kids = b" ".join(f"{number} 0 R".encode("ascii") for number in page_numbers)
    objects[2] = b"<< /Type /Pages /Kids [" + kids + b"] /Count " + str(len(page_numbers)).encode("ascii") + b" >>"
    maximum = max(objects)
    body = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0] * (maximum + 1)
    for number in range(1, maximum + 1):
        offsets[number] = len(body)
        body.extend(f"{number} 0 obj\n".encode("ascii"))
        body.extend(objects[number])
        body.extend(b"\nendobj\n")
    xref = len(body)
    body.extend(f"xref\n0 {maximum + 1}\n".encode("ascii"))
    body.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        body.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    body.extend(f"trailer\n<< /Size {maximum + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("ascii"))
    return bytes(body)


def document_lines(metadata: dict[str, str]) -> list[tuple[str, str]]:
    get = lambda key: str(metadata[key])
    return [
        ("title", "\u8d5b\u9053\u4e00 A\u699c\u590d\u73b0\u4ee3\u7801\u63d0\u4ea4\u8bf4\u660e"),
        ("space", ""),
        ("heading", "\u4e00\u3001\u56e2\u961f\u4fe1\u606f"),
        ("body", "\u56e2\u961f\u540d\u79f0\uff1a" + get("team")),
        ("body", "\u53c2\u8d5b\u8d5b\u9053\uff1a\u8d5b\u9053\u4e00"),
        ("body", "A\u699c\u6392\u540d\uff1a" + get("rank")),
        ("body", "A\u699c\u6700\u4f18\u603b\u5206\uff1a" + get("score")),
        ("body", "\u8054\u7cfb\u4eba\uff1a" + get("contact")),
        ("body", "\u5fae\u4fe1\uff1a" + get("wechat")),
        ("body", "\u7535\u8bdd\uff1a" + get("phone")),
        ("space", ""),
        ("heading", "\u4e8c\u3001\u9879\u76ee\u6982\u8ff0"),
        ("body", "\u4efb\u52a1\u662f\u5bf9\u65f6\u5e8f\u56fe\u4e2d\u6bcf\u4e2a\u67e5\u8be2\u7684 100 \u4e2a\u5019\u9009\u8282\u70b9\u8fdb\u884c\u6392\u5e8f\u3002\u65b9\u6848\u5305\u542b Dataset1 \u7684\u9ad8\u9636\u56fe\u6392\u5e8f\u96c6\u6210\uff0c\u4ee5\u53ca Dataset2 \u7684 VAE\u3001BM25-BPR\u3001\u96c6\u5408\u6392\u5e8f\u5668\u3001Transformer \u548c warm residual \u878d\u5408\u3002"),
        ("body", "\u6700\u7ec8\u9636\u6bb5\u5305\u542b\u540c\u6e90\u5019\u9009\u652f\u6301\u4fe1\u53f7\u3001\u540c\u67e5\u8be2\u7ec4\u5019\u9009\u652f\u6301\u4fe1\u53f7\u4e0e\u57fa\u4e8e Jittor BPR \u7528\u6237\u8868\u5f81\u7684\u8de8\u6e90\u793e\u533a\u6b8b\u5dee\u3002\u6240\u6709\u53ef\u5b66\u4e60\u90e8\u5206\u5747\u4f7f\u7528 Jittor\u3002"),
        ("space", ""),
        ("heading", "\u4e09\u3001\u4ee3\u7801\u7ed3\u6784"),
        ("body", "code/main.py \u4e3a\u7edf\u4e00\u5165\u53e3\uff0c\u652f\u6301 train\u3001infer\u3001all \u548c verify\u3002code/dataset1 \u5305\u542b Dataset1 \u8bad\u7ec3\u3001\u63a8\u7406\u548c\u540e\u5904\u7406\u3002code/dataset2 \u5305\u542b Dataset2 \u57fa\u7840\u6a21\u578b\u3001\u5143\u6392\u5e8f\u5668\u548c Transformer\u3002"),
        ("body", "code/legacy_dataset2 \u5305\u542b\u65e7\u7248\u7ec4\u4ef6\u4e0e\u72ec\u7acb\u7684\u539f\u59cb\u6570\u636e\u91cd\u5efa\u9a71\u52a8\u3002\u65b0\u8bad\u7ec3\u4f1a\u4fdd\u7559 CF embedding\u3001\u6392\u5e8f\u5668\u6743\u91cd\u3001MultDAE \u5bfc\u51fa\u3001\u5206\u6570\u9762\u548c\u54c8\u5e0c manifest\u3002code/train_community_bpr_jittor.py \u8bad\u7ec3 32 \u7ef4\u793e\u533a BPR\u3002code/build_final_submission.py \u751f\u6210\u6700\u7ec8 result.zip\u3002"),
        ("heading", "\u56db\u3001\u73af\u5883\u914d\u7f6e"),
        ("body", "\u63a8\u8350 Ubuntu 22.04\u3001Python 3.10\u3001NVIDIA CUDA 12.4 \u517c\u5bb9\u9a71\u52a8\u548c CUDA \u8bbe\u5907\u3002\u6838\u5fc3\u4f9d\u8d56\u4e3a Jittor 1.3.11.0\u3001NumPy 1.26.4\u3001Pandas 2.2.3\u3001Numba 0.66.0 \u548c nvidia-cudnn-cu12 8.9.7.29\uff1b\u7a00\u758f\u77e9\u9635\u64cd\u4f5c\u7531\u5305\u5185 NumPy CSR \u5b9e\u73b0\u5b8c\u6210\u3002"),
        ("body", "\u5728\u865a\u62df\u73af\u5883\u4e2d\u6267\u884c pip install -r requirements.txt\uff0c\u5e76\u5c06 ML_CACHE_ROOT \u6307\u5411\u6570\u636e\u76d8\u4e0a\u53ef\u5199\u76ee\u5f55\u3002\u8fd0\u884c\u811a\u672c\u4f1a\u5148\u6267\u884c Jittor CUDA \u7b97\u672f\u63a2\u9488\uff1b\u82e5\u7cfb\u7edf CUDA \u7f3a\u5c11 cuDNN \u5f00\u53d1\u5305\uff0c\u5219\u4ec5\u5728 ML_CACHE_ROOT \u4e0b\u6784\u5efa\u672c\u5730\u5305\u88c5\u5668\uff0c\u4e0d\u4fee\u6539\u7cfb\u7edf CUDA\u3002Jittor \u7f16\u8bd1\u7f13\u5b58\u4e0d\u5199\u5165\u4ee3\u7801\u5305\u3002"),
        ("heading", "\u4e94\u3001\u8fd0\u884c\u6b65\u9aa4"),
        ("body", "A\u699c\u91ca\u653e\u6743\u91cd\u63a8\u7406\uff1abash run_inference.sh /path/to/data_A.zip /path/to/output\u3002\u8be5\u547d\u4ee4\u4ece\u5305\u5185\u6743\u91cd\u91cd\u5efa\u57fa\u7840\u5206\u6570\uff0c\u8f93\u51fa /path/to/output/result.zip\uff0c\u5e76\u68c0\u67e5\u5f55\u5165\u7684 A\u699c SHA-256\u3002"),
        ("body", "\u4e25\u683c\u63a8\u7406\u901a\u8fc7\u540e\uff0c\u811a\u672c\u5728\u8f93\u51fa\u76ee\u5f55\u5199\u5165 release_inference_verification.json\u3002\u53ea\u80fd\u4f7f\u7528\u8be5\u56de\u6267\u8fd0\u884c python3 code/tools/build_delivery_package.py --source . --release-receipt /path/to/output/release_inference_verification.json --output /path/to/contest1_sader_007.zip\u6765\u6253\u5305\u3002"),
        ("body", "\u4ece\u539f\u59cb\u6570\u636e\u8bad\u7ec3\uff1abash run_train.sh /path/to/data_A.zip /path/to/models --dataset all --cuda\u3002\u751f\u6210\u7684\u6a21\u578b\u4f7f\u7528 bash run_fresh_inference.sh /path/to/data_A.zip /path/to/models /path/to/new-output \u8fdb\u884c\u63a8\u7406\uff0c\u8be5\u5165\u53e3\u7ee7\u627f CUDA \u4e0e\u6570\u636e\u76d8\u7f13\u5b58\u7ea6\u675f\u3002"),
        ("body", "\u4e00\u6b65\u5b8c\u6210\u5168\u65b0 raw \u8bad\u7ec3\u4e0e\u63a8\u7406\uff1abash run_raw.sh /path/to/data_A.zip /path/to/models /path/to/new-output\u3002\u8be5\u8def\u5f84\u5148\u68c0\u67e5 Python/Jittor/CUDA \u73af\u5883\uff0c\u5e76\u4fdd\u7559 training_manifest.json\u3001raw_training_verification.json \u4e0e legacy_components \u7684\u5206\u6210\u5458\u54c8\u5e0c\uff0c\u4e0d\u8fdb\u884c\u5386\u53f2 release \u54c8\u5e0c\u65ad\u8a00\u3002"),
        ("body", "raw \u8bad\u7ec3\u5728\u5206\u914d GPU \u4e4b\u524d\u4f1a\u6267\u884c code/tools/check_raw_pipeline_contract.py\uff0c\u68c0\u67e5 replay-pool \u63a5\u53e3\u3001multislice/warm \u8c03\u7528\u94fe\u548c Legacy \u5206\u6210\u5458\u4fdd\u7559\u7ea6\u675f\u3002\u8be5\u68c0\u67e5\u4e0d\u5bfc\u5165 Jittor\uff0c\u53ef\u5728 CUDA \u521d\u59cb\u5316\u524d\u5931\u8d25\u5173\u95ed\u3002"),
        ("body", "\u65b0\u8bad\u7ec3\u63a8\u7406\u4f1a\u5199\u5165 inference_manifest.json\uff0c\u7ed1\u5b9a\u6570\u636e\u3001\u6a21\u578b\u3001\u4ee3\u7801\u3001\u4f9d\u8d56\u548c result.zip \u54c8\u5e0c\u3002run_raw.sh \u8fd8\u4f1a\u751f\u6210 fresh_run_verification.json\uff0c\u4ea4\u53c9\u6821\u9a8c\u8bad\u7ec3\u3001Legacy \u5206\u6210\u5458\u4e0e\u63a8\u7406\u56de\u6267\u3002\u8be5\u56de\u6267\u4ec5\u58f0\u660e fresh raw \u9a8c\u8bc1\uff0c\u4e0d\u58f0\u660e A \u699c\u5386\u53f2\u7b49\u4ef7\u3002"),
        ("body", "Dataset1 \u53c2\u6570\uff1a\u4e24\u4e2a\u79cd\u5b50 20260705/20260715\uff0cgroups=80000\uff0cvalid=20000\uff0cepochs=16\uff0cbatch=512\u3002Dataset2 \u57fa\u7840 BPR \u53c2\u6570\uff1afactors=256\uff0cepochs=20\uff0cbatch=32768\uff0cnegatives=8\uff0clr=0.002\u3002\u793e\u533a BPR \u53c2\u6570\uff1afactors=32\uff0cepochs=3\uff0cbatch=32768\uff0cnegatives=4\uff0clr=0.002\u3002"),
        ("body", "\u65b0\u8bad\u7ec3\u7684 Dataset1 \u6309\u5b98\u65b9 train.csv \u65f6\u95f4\u6392\u5e8f\uff0c80%\u4f5c\u4e3a\u5386\u53f2\uff0c\u4f59\u4e0b\u524d80%\u8bad\u7ec3\u3001\u540e20%\u9a8c\u8bc1\u3002legacy MultDAE \u5386\u53f2\u622a\u6b62\u65f6\u95f4\u53d6 dataset2/test.csv \u6700\u65e9\u65f6\u95f4\u5e76\u8bb0\u5f55\uff1b\u8be5\u65b0\u8bad\u7ec3\u94fe\u4e0d\u58f0\u79f0\u5386\u53f2 A \u699c\u7b49\u4ef7\u3002"),
        ("heading", "\u516d\u3001\u8f93\u5165\u8f93\u51fa\u4e0e\u5408\u89c4\u8bf4\u660e"),
        ("body", "\u8f93\u5165\u4e3a\u5b98\u65b9 data_A.zip\uff0c\u5305\u542b\u4e24\u4e2a\u6570\u636e\u96c6\u7684 train.csv \u548c\u53ea\u542b\u5019\u9009\u5217\u7684 test.csv\u3002\u8f93\u51fa result.zip \u5305\u542b dataset1.csv \u548c dataset2.csv\uff0c\u6bcf\u884c 100 \u4e2a\u975e\u8d1f\u6982\u7387\uff0c\u884c\u548c\u4e3a 1\u3002"),
        ("body", "\u4ee3\u7801\u4e0d\u8bfb\u53d6\u6d4b\u8bd5\u6807\u7b7e\u3002\u9664\u6700\u7ec8\u540e\u5904\u7406\u5916\uff0cDataset1 \u8d1f\u91c7\u6837\u53ca Dataset2 pool/meta \u8bad\u7ec3\u4e5f\u4f7f\u7528\u5b98\u65b9 test.csv \u5019\u9009 ID \u4f5c\u4e3a\u65e0\u6807\u7b7e\u5019\u9009\u6c60/\u56de\u653e\u6c60\uff1b\u6240\u6709\u5b66\u4e60\u76ee\u6807\u5747\u6765\u81ea train.csv\u3002\u8fd9\u662f transductive \u5019\u9009\u7ed3\u6784\u8026\u5408\uff0c\u5df2\u660e\u786e\u62ab\u9732\u4f9b\u4eba\u5de5\u5ba1\u6838\u3002"),
        ("heading", "\u4e03\u3001\u590d\u73b0\u8fb9\u754c\u4e0e\u6ce8\u610f\u4e8b\u9879"),
        ("body", "Audit gate: this staging tree is blocked from external delivery. Historical Legacy member lineage and the verified Set-v2 source/model lineage are incomplete here; a prior result receipt cannot override that audit."),
        ("body", "\u5305\u5185\u91ca\u653e\u6743\u91cd\u5305\u542b\u4e00\u4e2a\u5386\u53f2 legacy Dataset2 \u5206\u6570\u5e73\u9762\u5de5\u4ef6\u3002\u65e7\u7248\u53d1\u5e03\u6ca1\u6709\u4fdd\u7559 friend/ours/CF/MultDAE \u6240\u6709\u6210\u5458\u6743\u91cd\u548c\u9009\u62e9\u72b6\u6001\uff0c\u56e0\u6b64\u5305\u5185\u7684\u539f\u59cb\u6570\u636e\u91cd\u8bad\u7ec3\u94fe\u4e0d\u5ba3\u79f0\u4e0e\u5386\u53f2\u5206\u6570\u5e73\u9762\u9010\u5b57\u8282\u4e00\u81f4\u3002"),
        ("body", "strict-release \u6a21\u5f0f\u53ea\u5bf9\u91ca\u653e\u6743\u91cd\u91cd\u5efa\u8fdb\u884c SHA-256 \u95ed\u73af\u68c0\u67e5\u3002\u82e5\u786c\u4ef6\u3001CUDA \u6216 Jittor \u7f16\u8bd1\u5668\u4e0d\u540c\u5bfc\u81f4\u54c8\u5e0c\u4e0d\u4e00\u81f4\uff0c\u7a0b\u5e8f\u4f1a\u62d2\u7edd\u62a5\u544a\u6210\u529f\u3002CUDA 12.4 + RTX 4090 \u9700\u5728\u63d0\u4ea4\u73af\u5883\u4e2d\u6700\u540e\u590d\u9a8c\u3002"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replace", action="store_true", help="replace a deterministic generated PDF")
    args = parser.parse_args()
    if args.output.exists() and not args.replace:
        raise FileExistsError(f"refusing to overwrite PDF: {args.output}")
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    required = {"team", "rank", "score", "contact", "wechat", "phone"}
    if set(metadata) != required or not all(isinstance(metadata[key], str) and metadata[key] for key in required):
        raise ValueError("metadata fields differ")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(build_pdf(build_pages(document_lines(metadata))))
    print(f"wrote {args.output} pages={len(build_pages(document_lines(metadata)))}", flush=True)


if __name__ == "__main__":
    main()
