"""Rescore Inspect logs with a lenient letter parser and write results/scores.csv.

Inspect's `choice` scorer wants "ANSWER: X"; some models answer "X) text" or open a thought block first.
This takes the last "ANSWER: X" in the completion, else a leading "X)" / "X." / "X:". three_gpp keeps its own
pattern score. The TeleQnA "Standards specifications" subset (200) is reported as teleqna-specs.

    python scripts/rescore.py logs results/scores.csv
"""

import csv
import re
import sys
from pathlib import Path

from inspect_ai.log import read_eval_log

ANSWER = re.compile(r"ANSWER\s*:\s*\**\s*([A-J])\b")
LEADING = re.compile(r"^\s*\**\(?([A-J])[).:]")


def letter(text: str) -> str | None:
    found = ANSWER.findall(text)
    if found:
        return found[-1]
    m = LEADING.match(text.split("<channel|>")[-1])
    return m.group(1) if m else None


def main(log_root: str, out: str) -> None:
    rows = []
    for f in sorted(Path(log_root).glob("*/*/*.eval")):
        tag, bench = f.parts[-3], f.parts[-2]
        log = read_eval_log(str(f))
        if log.status != "success" or not log.samples:
            continue
        task = log.eval.task.split("/")[-1]  # newer Inspect records "evals/<task>"
        groups = {bench: log.samples}
        if task == "teleqna" and bench == "teleqna":
            groups["teleqna-specs"] = [s for s in log.samples
                                       if (s.metadata or {}).get("subject") == "Standards specifications"]
        for name, samples in groups.items():
            if not samples:
                continue
            if task == "three_gpp":
                correct = [next(iter(s.scores.values())).value in ("C", 1, 1.0, True) for s in samples]
                parsed = len(samples)
            else:
                got = [letter(s.output.completion or "") for s in samples]
                correct = [g == s.target for g, s in zip(got, samples)]
                parsed = sum(g is not None for g in got)
            rows.append({"tag": tag, "bench": name, "n": len(samples),
                         "accuracy": round(sum(correct) / len(samples), 4),
                         "parsed": round(parsed / len(samples), 4), "log": f.name})

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['tag']:<28} {r['bench']:<14} n={r['n']:<5} acc={r['accuracy']:.3f}  parsed={r['parsed']:.2f}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
