"""Render a Markdown file to PDF (pure-Python: markdown + xhtml2pdf, no apt needed).

  pip install markdown xhtml2pdf
  python tools/md_to_pdf.py report/report.md      # -> report/report.pdf
"""
import os
import sys

import markdown
from xhtml2pdf import pisa

src = sys.argv[1] if len(sys.argv) > 1 else os.path.join("report", "report.md")
dst = os.path.splitext(src)[0] + ".pdf"

body = markdown.markdown(
    open(src, encoding="utf-8").read(),
    extensions=["tables", "fenced_code", "toc", "sane_lists"],
)

html = """<html><head><meta charset="utf-8"><style>
@page { size: A4; margin: 1.4cm; }
body { font-family: Helvetica, Arial, sans-serif; font-size: 9pt; line-height: 1.35; color: #111; }
h1 { font-size: 16pt; margin-top: 14pt; }
h2 { font-size: 13pt; border-bottom: 1px solid #999; padding-bottom: 2px; margin-top: 12pt; }
h3 { font-size: 11pt; margin-top: 10pt; }
table { border-collapse: collapse; width: 100%%; margin: 6px 0; }
th, td { border: 1px solid #999; padding: 3px 5px; font-size: 8pt; }
th { background: #eee; }
code { background: #f4f4f4; font-family: Courier; font-size: 8pt; }
pre { background: #f4f4f4; padding: 5px; font-family: Courier; font-size: 7.5pt; }
blockquote { color: #444; border-left: 3px solid #ccc; padding-left: 8px; }
</style></head><body>%s</body></html>""" % body

with open(dst, "wb") as fh:
    result = pisa.CreatePDF(html, dest=fh)

print("wrote", dst, "(errors)" if result.err else "OK")
