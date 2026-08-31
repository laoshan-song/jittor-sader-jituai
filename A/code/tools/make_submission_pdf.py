#!/usr/bin/env python3
"""Generate and verify the static submission PDF with pinned Python tooling.

This document-build utility is independent from training and inference. It
uses WeasyPrint for HTML/CSS pagination and PyMuPDF for post-render checks,
so creating the required PDF does not depend on a locally installed browser.
"""

from __future__ import annotations

import argparse
import html
import os
import re
import shutil
import tempfile
from pathlib import Path

import fitz
from weasyprint import HTML


MINIMUM_PAGES = 10
MINIMUM_BYTES = 16_384
EQUATION = re.compile(
    r'<div class="equation" data-latex="(?P<latex>[^"]+)">.*?</div>', re.DOTALL
)


def chinese_characters(value: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]", value))


def _svg_text(text: str, x: float, y: float, size: float = 25.0, *, weight: str = "normal") -> str:
    escaped = html.escape(text)
    return (
        f'<text x="{x:g}" y="{y:g}" font-family="DejaVu Serif, serif" '
        f'font-size="{size:g}px" font-weight="{weight}" fill="#234940">{escaped}</text>'
    )


def _formula_svg(latex: str) -> str:
    """Render the report's fixed equations as readable vector SVG.

    The source remains LaTeX in HTML.  These three equations are deliberately
    drawn with SVG text/tspan primitives so WeasyPrint does not fall back to
    a linear MathML representation when a browser math engine is unavailable.
    """
    if latex.startswith("p_{ij}"):
        body = (
            _svg_text("p", 16, 44, 30)
            + _svg_text("ij", 35, 50, 16)
            + _svg_text("=", 72, 44, 27)
            + _svg_text("exp(z", 111, 28, 22)
            + _svg_text("ij", 191, 33, 13)
            + _svg_text(")", 208, 28, 22)
            + '<line x1="108" y1="38" x2="236" y2="38" stroke="#234940" stroke-width="1.2"/>'
            + _svg_text("Σ", 119, 69, 23)
            + _svg_text("k=1", 137, 77, 12)
            + _svg_text("100", 168, 29, 12)
            + _svg_text("exp(z", 199, 69, 22)
            + _svg_text("ik", 279, 74, 13)
            + _svg_text(")", 296, 69, 22)
        )
        width, height = 330, 92
    elif latex.startswith("z_{ij}"):
        body = (
            _svg_text("z", 12, 46, 27)
            + _svg_text("ij", 30, 52, 14)
            + _svg_text("= qnorm(log p", 61, 46, 21)
            + _svg_text("ij", 216, 52, 12)
            + _svg_text("base", 234, 32, 11)
            + _svg_text(") + 0.05 · qnorm(I[s", 274, 46, 21)
            + _svg_text("ij", 464, 52, 12)
            + _svg_text("exact", 482, 32, 11)
            + _svg_text(" > 0])", 533, 46, 21)
            + _svg_text("+ 0.02 · qnorm(I[c", 180, 101, 21)
            + _svg_text("ij", 382, 107, 12)
            + _svg_text("community", 400, 87, 10)
            + _svg_text(" = 1])", 489, 101, 21)
        )
        width, height = 700, 128
    else:
        body = (
            _svg_text("qnorm(x) =", 12, 44, 25)
            + _svg_text("x − μ(x)", 191, 27, 22)
            + '<line x1="186" y1="36" x2="310" y2="36" stroke="#234940" stroke-width="1.2"/>'
            + _svg_text("σ(x) + 10", 190, 68, 21)
            + _svg_text("−6", 302, 60, 12)
        )
        width, height = 345, 86
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" role="img" '
        f'viewBox="0 0 {width} {height}" width="{width}" height="{height}">{body}</svg>'
    )


def render_latex_equations(source: str, render_dir: Path) -> str:
    """Render LaTeX sources as vector equations while retaining source labels."""
    def replacement(match: re.Match[str]) -> str:
        latex = html.unescape(match.group("latex"))
        image_name = f"equation_{replacement.counter}.svg"
        replacement.counter += 1
        image_path = render_dir / image_name
        image_path.write_text(_formula_svg(latex), encoding="utf-8")
        escaped = html.escape(latex)
        return (
            '<div class="equation">'
            f'<img class="equation-svg" src="{image_name}" alt="{escaped}">'
            '</div>'
        )

    replacement.counter = 0

    rendered, count = EQUATION.subn(replacement, source)
    if count == 0:
        raise RuntimeError("submission report does not contain a LaTeX display equation")
    return rendered


def verify_pdf(path: Path, expected_chinese: int) -> int:
    """Check text coverage, page count, and extractable text page bounds."""
    if not path.is_file() or path.stat().st_size < MINIMUM_BYTES:
        raise RuntimeError("PDF generator did not create a plausible document")
    with fitz.open(path) as document:
        if document.page_count < MINIMUM_PAGES:
            raise RuntimeError(
                f"report has {document.page_count} pages; at least {MINIMUM_PAGES} are required"
            )
        extracted = []
        for index, page in enumerate(document):
            blocks = [block for block in page.get_text("blocks") if block[4].strip()]
            if not blocks:
                raise RuntimeError(f"PDF page {index + 1} contains no extractable text")
            page_box = page.rect
            for block in blocks:
                box = fitz.Rect(block[:4])
                if not page_box.contains(box):
                    raise RuntimeError(
                        f"PDF text block escapes page bounds on page {index + 1}: {tuple(box)}"
                    )
            extracted.append(page.get_text("text"))
        actual_chinese = chinese_characters("".join(extracted))
        required_chinese = int(expected_chinese * 0.98)
        if actual_chinese < required_chinese:
            raise RuntimeError(
                "PDF Chinese text coverage is too low: "
                f"expected at least {required_chinese}, extracted {actual_chinese}. "
                "Install fonts-noto-cjk before generating the document."
            )
        return document.page_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the academic submission PDF with WeasyPrint"
    )
    root = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--html", type=Path, default=Path(__file__).with_name("submission_report.html")
    )
    parser.add_argument("--output", type=Path, default=root / "提交说明文档.pdf")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.replace:
        raise FileExistsError(args.output)
    if not args.html.is_file():
        raise FileNotFoundError(args.html)
    source = args.html.read_text(encoding="utf-8")
    expected_chinese = chinese_characters(source)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    render_dir = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent))
    temporary = render_dir / "render.pdf"
    try:
        rendered_html = render_dir / "report.html"
        rendered_html.write_text(render_latex_equations(source, render_dir), encoding="utf-8")
        HTML(filename=str(rendered_html), base_url=str(render_dir)).write_pdf(str(temporary))
        pages = verify_pdf(temporary, expected_chinese)
        os.replace(temporary, args.output)
    finally:
        shutil.rmtree(render_dir, ignore_errors=True)
    print(f"wrote {args.output} bytes={args.output.stat().st_size} pages={pages}")


if __name__ == "__main__":
    main()
