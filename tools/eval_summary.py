#!/usr/bin/env python
"""Render a per-run eval summary.md (wide table) from summary.csv.

Usage:
    python tools/eval_summary.py <eval_root>/<run_id>
    # e.g. python tools/eval_summary.py $S/eval/chimera_1t
Reads summary.csv (long format: step,consumed_tokens,task,metric,value,date_utc),
writes summary.md next to it. Idempotent; safe to re-run after every eval.
"""
import csv
import os
import sys
from collections import defaultdict

# (task, metric) rows to feature, in display order.
HEADLINE = [
    ("hellaswag", "acc_norm"),
    ("lambada_openai", "acc"),
    ("arc_easy", "acc"),
    ("arc_challenge", "acc_norm"),
    ("piqa", "acc_norm"),
    ("winogrande", "acc"),
    ("mmlu", "acc"),
]


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    run_dir = sys.argv[1].rstrip("/")
    run_id = os.path.basename(run_dir)
    csv_path = os.path.join(run_dir, "summary.csv")
    if not os.path.exists(csv_path):
        print(f"[eval-summary] no summary.csv at {csv_path}")
        sys.exit(1)

    # value[(task,metric)][step] = value ; last write wins (re-runs override)
    value = defaultdict(dict)
    tokens = {}
    steps = set()
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            step = int(row["step"])
            steps.add(step)
            try:
                value[(row["task"], row["metric"])][step] = float(row["value"])
            except ValueError:
                continue
            if row.get("consumed_tokens"):
                try:
                    tokens[step] = int(row["consumed_tokens"])
                except ValueError:
                    pass
    steps = sorted(steps)

    def fmt_tok(s):
        t = tokens.get(s)
        return f"{t/1e9:.1f}B" if t else "?"

    lines = [f"# {run_id} eval summary", ""]
    hdr = ["benchmark / metric"] + [f"{s} ({fmt_tok(s)})" for s in steps]
    lines.append("| " + " | ".join(hdr) + " |")
    lines.append("|" + "|".join(["---"] * len(hdr)) + "|")
    present = [(t, m) for (t, m) in HEADLINE if (t, m) in value]

    def is_noise(t, m):
        if m in ("sample_len", "perplexity", "alias") or m.endswith("_stderr"):
            return True
        # hide the 57 per-subject mmlu breakdowns from the headline table
        if t.startswith("mmlu_"):
            return True
        return False

    extra = sorted(k for k in value if k not in set(HEADLINE) and not is_noise(*k))
    for task, metric in present + extra:
        cells = []
        for s in steps:
            v = value[(task, metric)].get(s)
            cells.append(f"{v:.4f}" if v is not None else "—")
        lines.append(f"| {task} {metric} | " + " | ".join(cells) + " |")
    lines.append("")

    out = os.path.join(run_dir, "summary.md")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"[eval-summary] wrote {out} ({len(steps)} steps, {len(present)+len(extra)} rows)")


if __name__ == "__main__":
    main()
