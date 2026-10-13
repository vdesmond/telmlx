"""Build the LoRA training set from TeleQnA, minus every question in the GSMA ot-lite eval set.

Prompts are the exact Inspect `multiple_choice` prompt the evals send (plus the strict system
message), so the adapter learns the format it is scored on. The target is the answer line followed
by TeleQnA's explanation, which keeps the scorer's first line intact and adds the knowledge signal.

    python train/prep.py data/teleqna/TeleQnA.json data/teleqna/test_teleqna.json data/teleqna
"""

import json
import random
import re
import sys
from pathlib import Path

SYS = ("You are taking a multiple-choice exam. Do not explain or reason. Reply with exactly one line "
       "of the form 'ANSWER: X' where X is the letter of the correct option.")
TEMPLATE = ("Answer the following multiple choice question. The entire content of your response should be "
            "of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}."
            "\n\n{question}\n\n{choices}")


def norm(q: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", q.lower()).strip()


def options(item: dict) -> list[str]:
    keys = sorted((k for k in item if re.fullmatch(r"option \d+", k)), key=lambda k: int(k.split()[1]))
    return [item[k] for k in keys]


def example(question: str, choices: list[str], answer: int, explanation: str) -> dict:
    letters = [chr(65 + i) for i in range(len(choices))]
    prompt = TEMPLATE.format(letters=",".join(letters), question=question,
                             choices="\n".join(f"{l}) {c}" for l, c in zip(letters, choices)))
    target = f"ANSWER: {letters[answer]}"
    if explanation:
        target += f"\n\n{explanation.strip()}"
    return {"prompt": [{"role": "system", "content": SYS}, {"role": "user", "content": prompt}],
            "completion": [{"role": "assistant", "content": target}]}


def main(teleqna: str, eval_set: str, out: str, val_frac: float = 0.05, seed: int = 0) -> None:
    held_out = {norm(x["question"]) for x in json.load(open(eval_set))}
    items = json.load(open(teleqna))
    items = list(items.values()) if isinstance(items, dict) else items

    rows, overlap = [], 0
    for it in items:
        if norm(it["question"]) in held_out:
            overlap += 1
            continue
        choices = options(it)
        # TeleQnA answers look like "option 3: <text>"
        idx = int(re.match(r"option (\d+)", it["answer"]).group(1)) - 1
        rows.append(example(it["question"], choices, idx, it.get("explanation", "")))

    random.Random(seed).shuffle(rows)
    n_val = int(len(rows) * val_frac)
    out_dir = Path(out)
    for name, part in (("val", rows[:n_val]), ("train", rows[n_val:])):
        with open(out_dir / f"{name}.jsonl", "w") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in part)

    print(f"TeleQnA {len(items)} | removed (in eval set) {overlap} of {len(held_out)} | "
          f"train {len(rows) - n_val} | val {n_val}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
