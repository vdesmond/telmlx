"""LoRA fine-tune of Gemma-4-E4B (bf16, text tower only) on prompt/completion chat jsonl. Runs on GPULab.

Loss is on the completion only. The prompt mask comes from rendering the chat template twice
(prompt with generation prompt, then prompt + completion) so it does not depend on Gemma's
turn tokens.

    python train/lora.py --smoke --out /project_ghent/telmlx/runs/smoke
    python train/lora.py --train data/teleqna/train.jsonl --val data/teleqna/val.jsonl --out runs/r16 --merge
"""

import argparse
import json
import time
from collections import Counter

import torch
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForImageTextToText, AutoTokenizer, Trainer, TrainerCallback,
                          TrainingArguments)

# every attention/MLP projection outside the vision and audio towers
TARGETS = r"^(?!.*(vision|audio)).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"


def encode(tok, row, max_len):
    prompt = tok.apply_chat_template(row["prompt"], add_generation_prompt=True, tokenize=True)
    full = tok.apply_chat_template(row["prompt"] + row["completion"], tokenize=True)
    if hasattr(prompt, "input_ids"):
        prompt, full = prompt["input_ids"], full["input_ids"]
    assert full[:len(prompt)] == prompt, "chat template does not extend the prompt prefix"
    full = full[:max_len]
    labels = [-100] * len(prompt) + full[len(prompt):]
    return {"input_ids": full, "labels": labels[:len(full)]}


def load(path, tok, max_len, limit=None):
    rows = [json.loads(l) for l in open(path)][:limit]
    return [encode(tok, r, max_len) for r in rows]


def smoke_rows(n=64):
    # real-length text without touching the eval data: 3GPP chunks (submit.sh uploads the first 64)
    rows = []
    with open("data/smoke-chunks.jsonl") as f:
        for line, _ in zip(f, range(n)):
            text = json.loads(line)["text"]
            rows.append({"prompt": [{"role": "user", "content": "Repeat this 3GPP clause:\n\n" + text[:1500]}],
                         "completion": [{"role": "assistant", "content": text[:1500]}]})
    return rows


def collate(pad_id):
    def fn(batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), pad_id)
        labels = torch.full((len(batch), n), -100)
        mask = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            k = len(b["input_ids"])
            ids[i, :k], labels[i, :k], mask[i, :k] = torch.tensor(b["input_ids"]), torch.tensor(b["labels"]), 1
        return {"input_ids": ids, "labels": labels, "attention_mask": mask}
    return fn


class Throughput(TrainerCallback):
    def __init__(self, tokens_per_step):
        self.tps, self.t0, self.steps = tokens_per_step, None, 0

    def on_step_begin(self, args, state, control, **kw):
        if self.t0 is None and state.global_step == 2:  # skip compile/warm-up steps
            self.t0, self.steps = time.time(), state.global_step

    def on_log(self, args, state, control, logs=None, **kw):
        if self.t0 and logs is not None:
            dt = time.time() - self.t0
            logs["tok_per_s"] = round(self.tps * (state.global_step - self.steps) / max(dt, 1e-6))
            logs["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-4-E4B-it")
    p.add_argument("--train")
    p.add_argument("--val")
    p.add_argument("--out", required=True)
    p.add_argument("--smoke", action="store_true", help="20 steps on 3GPP chunks; print modules, speed, memory")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--merge", action="store_true", help="also save adapter merged into bf16 weights")
    a = p.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda",
                                                        attn_implementation="sdpa")
    model.config.use_cache = False
    print("model class:", type(model).__name__, torch.cuda.get_device_name())

    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=0.05,
                                             target_modules=TARGETS, task_type="CAUSAL_LM"))
    wrapped = [n for n, m in model.named_modules() if hasattr(m, "lora_A")]
    print("LoRA modules:", len(wrapped), dict(Counter(n.rsplit(".", 1)[-1] for n in wrapped)))
    print("  e.g.", wrapped[0], "...", wrapped[-1])
    model.print_trainable_parameters()

    if a.smoke:
        train = [encode(tok, r, a.max_len) for r in smoke_rows()]
        val = None
    else:
        train, val = load(a.train, tok, a.max_len), load(a.val, tok, a.max_len) if a.val else None
    lens = sorted(len(x["input_ids"]) for x in train)
    print(f"train {len(train)} rows, tokens/row median {lens[len(lens) // 2]} max {lens[-1]}, "
          f"total {sum(lens)}")

    args = TrainingArguments(
        output_dir=a.out, per_device_train_batch_size=a.batch, per_device_eval_batch_size=a.batch,
        learning_rate=a.lr, lr_scheduler_type="cosine", warmup_steps=0.03, num_train_epochs=a.epochs,
        max_steps=20 if a.smoke else -1, bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, logging_steps=5 if a.smoke else 20,
        eval_strategy="steps" if val else "no", eval_steps=100, save_strategy="no" if a.smoke else "steps",
        save_steps=200, save_total_limit=2, report_to="none", remove_unused_columns=False,
        train_sampling_strategy="group_by_length", dataloader_num_workers=2)
    model.enable_input_require_grads()
    trainer = Trainer(model=model, args=args, train_dataset=train, eval_dataset=val,
                      data_collator=collate(tok.pad_token_id),
                      callbacks=[Throughput(a.batch * lens[len(lens) // 2])])
    trainer.train()
    if val:
        print("final eval:", trainer.evaluate())

    model.save_pretrained(f"{a.out}/adapter")
    tok.save_pretrained(f"{a.out}/adapter")
    if a.merge:
        merged = model.merge_and_unload()
        merged.save_pretrained(f"{a.out}/merged", safe_serialization=True)
        tok.save_pretrained(f"{a.out}/merged")
    print("done ->", a.out)


if __name__ == "__main__":
    main()
