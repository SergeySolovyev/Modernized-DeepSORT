# LaTeX report source

This directory contains the source of the formatted project report. The compiled report is
committed at `../report.pdf`.

## Files

- `main.tex` - the report source.
- `make_figures.py` - regenerates the figures in `images/outputs/` from the verified results
  (matplotlib). The figures are also committed, so this step is optional.
- `images/outputs/*.png` - the figures included by `main.tex`.

## Build

Requires a TeX distribution (TeX Live or MiKTeX) and, to regenerate figures, matplotlib.

    python make_figures.py
    pdflatex -interaction=nonstopmode main.tex
    pdflatex -interaction=nonstopmode main.tex

The second pass resolves the table of contents and cross-references.
