"""Summarize HOTA (or another metric) from results/experiments.csv into a markdown table.

After the Colab run logs per-video HOTA, this builds the per-video comparison table for
report/results_table.md (rows per tracker, columns per video + Mean, plus Delta vs the baseline).

  python -m eval.summarize                  # HOTA table for all trackers seen
  python -m eval.summarize --metric IDF1
"""
import argparse
import csv
import os

from eval.common import EVAL_SEQUENCES
from eval.journal import DEFAULT_JOURNAL


def load_metric(journal_path, metric):
    table = {}  # tracker -> {seq -> value}
    if not os.path.exists(journal_path):
        return table
    with open(journal_path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("component") != "hota":
                continue
            val = r.get(metric, "")
            if val in (None, "", "None"):
                continue
            try:
                table.setdefault(r["tracker"], {})[r["seq"]] = float(val)
            except ValueError:
                continue
    return table


def build_markdown(table, metric):
    trackers = sorted(table)
    seqs = EVAL_SEQUENCES
    lines = ["| tracker | " + " | ".join(seqs) + " | Mean |",
             "|" + "---|" * (len(seqs) + 2)]
    means = {}
    for t in trackers:
        vals = [table[t].get(s) for s in seqs]
        present = [v for v in vals if v is not None]
        mean = sum(present) / len(present) if present else float("nan")
        means[t] = mean
        cells = ["%.2f" % v if v is not None else "-" for v in vals]
        lines.append("| %s | %s | %.2f |" % (t, " | ".join(cells), mean))
    md = "\n".join(lines)
    if "baseline" in means:
        deltas = ", ".join("%s %+.2f" % (t, means[t] - means["baseline"])
                           for t in trackers if t != "baseline")
        md += "\n\n**Delta mean %s vs baseline:** %s" % (metric, deltas or "(no other trackers)")
    return md


def main():
    ap = argparse.ArgumentParser(description="Summarize HOTA results into a markdown table")
    ap.add_argument("--journal", default=DEFAULT_JOURNAL)
    ap.add_argument("--metric", default="HOTA")
    ap.add_argument("--out", default=os.path.join("report", "results_table.md"))
    args = ap.parse_args()

    table = load_metric(args.journal, args.metric)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    if not table:
        md = "No '%s' rows in %s yet (run tracking + `trackeval_runner --per-seq` first)." % (
            args.metric, args.journal)
        print(md)
    else:
        md = build_markdown(table, args.metric)
        print(md)
    # Always write the file so downstream readers (the notebook) never hit a missing path.
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("# %s per video\n\n%s\n" % (args.metric, md))
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
