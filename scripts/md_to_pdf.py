"""One-off helper: convert docs/EVALUATION_GUIDE.md to PDF via xhtml2pdf."""

from __future__ import annotations

from pathlib import Path

import markdown
from xhtml2pdf import pisa

ROOT = Path(__file__).resolve().parents[1]
MD_PATH = ROOT / "docs" / "EVALUATION_GUIDE.md"
PDF_PATH = ROOT / "docs" / "EVALUATION_GUIDE.pdf"

CSS = """
body { font-family: Segoe UI, Arial, sans-serif; font-size: 11pt; line-height: 1.45; margin: 2cm; }
table { border-collapse: collapse; width: 100%; margin: 1em 0; }
th, td { border: 1px solid #ccc; padding: 6px; text-align: left; }
th { background: #f0f0f0; }
code, pre { font-family: Consolas, monospace; font-size: 9pt; background: #f5f5f5; }
pre { padding: 8px; white-space: pre-wrap; }
h1 { font-size: 20pt; }
h2 { font-size: 14pt; margin-top: 1.2em; }
h3 { font-size: 12pt; }
"""

# The built-in PDF fonts have no glyphs for these, so they would print as boxes.
PLAIN_SYMBOLS = {"→": "->", "←": "<-", "↔": "<->", "≫": ">>", "≈": "~", "×": "x"}


def main() -> None:
    text = MD_PATH.read_text(encoding="utf-8")
    for symbol, plain in PLAIN_SYMBOLS.items():
        text = text.replace(symbol, plain)
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "toc"])
    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'/>"
        f"<style>{CSS}</style></head><body>{body}</body></html>"
    )
    with PDF_PATH.open("wb") as out:
        status = pisa.CreatePDF(html, dest=out, encoding="utf-8")
    if status.err:
        raise SystemExit(f"PDF generation failed: {status.err}")
    print(f"Wrote {PDF_PATH} ({PDF_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
