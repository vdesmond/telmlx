"""Fuse a PEFT LoRA adapter (from train/lora.py) into the bf16 Gemma-4-E4B checkpoint, on the Mac, in MLX.

W' = W + (alpha / r) * B @ A for every adapted projection. The result keeps the original HF layout, so
mlx_lm.convert can quantise it like the OTel models:

    python train/fuse.py runs/r16/adapter models/gemma4-e4b-teleqna-bf16
    python train/fuse.py none models/gemma4-e4b-bf16     # base through the same path (control arm)
    mlx_lm.convert --hf-path models/gemma4-e4b-teleqna-bf16 --mlx-path models/gemma4-e4b-teleqna-4bit -q
"""

import json
import shutil
import sys
from pathlib import Path

import mlx.core as mx
from huggingface_hub import snapshot_download

SHARD_BYTES = 4 << 30


def main(adapter: str, out: str, base_id: str = "google/gemma-4-E4B-it") -> None:
    cfg = json.load(open(f"{adapter}/adapter_config.json")) if adapter != "none" else \
        {"lora_alpha": 1, "r": 1, "base_model_name_or_path": base_id}
    scale = cfg["lora_alpha"] / cfg["r"]
    base = Path(snapshot_download(cfg["base_model_name_or_path"]))
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    lora = mx.load(f"{adapter}/adapter_model.safetensors") if adapter != "none" else {}
    # base_model.model.<key>.lora_A.weight -> <key>.weight
    pairs = {k.removeprefix("base_model.model.").replace(".lora_A.weight", ".weight"):
             (lora[k], lora[k.replace("lora_A", "lora_B")]) for k in lora if ".lora_A." in k}

    weights = {}
    for f in sorted(base.glob("*.safetensors")):
        weights.update(mx.load(str(f)))
    missing = set(pairs) - set(weights)
    assert not missing, f"adapter keys not in base: {sorted(missing)[:3]}"

    # The last num_kv_shared_layers reuse K/V from earlier layers, but the HF checkpoint still ships their
    # k_proj/v_proj/k_norm. They are never used and mlx_lm (0.31) refuses to load them, so drop them.
    text = json.load(open(base / "config.json"))["text_config"]
    first_shared = text["num_hidden_layers"] - text.get("num_kv_shared_layers", 0)
    unused = [k for k in weights if ".language_model.layers." in k
              and int(k.split(".layers.")[1].split(".")[0]) >= first_shared
              and any(p in k for p in (".self_attn.k_proj.", ".self_attn.v_proj.", ".self_attn.k_norm."))]
    assert not set(unused) & set(pairs), "adapter touches a KV-shared layer's K/V"
    for k in unused:
        del weights[k]

    for k, (a, b) in pairs.items():
        w = weights[k]
        weights[k] = (w.astype(mx.float32) + scale * (b.astype(mx.float32) @ a.astype(mx.float32))).astype(w.dtype)

    shards, cur, size = [], {}, 0
    for k in sorted(weights):
        cur[k] = weights[k]
        size += weights[k].nbytes
        if size >= SHARD_BYTES:
            shards.append(cur)
            cur, size = {}, 0
    if cur:
        shards.append(cur)
    index = {"metadata": {"total_size": sum(w.nbytes for w in weights.values())}, "weight_map": {}}
    for i, shard in enumerate(shards, 1):
        name = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
        mx.eval(shard)
        mx.save_safetensors(str(out_dir / name), shard, metadata={"format": "pt"})
        index["weight_map"].update({k: name for k in shard})
    json.dump(index, open(out_dir / "model.safetensors.index.json", "w"), indent=2)

    # config, tokenizer, chat template, processor files: unchanged from the base
    for f in base.iterdir():
        if f.suffix != ".safetensors" and not f.name.endswith(".index.json") and f.is_file():
            shutil.copy(f, out_dir / f.name)
    print(f"fused {len(pairs)} projections (scale {scale}), dropped {len(unused)} unused KV-shared tensors "
          f"-> {out_dir} in {len(shards)} shards")


if __name__ == "__main__":
    main(*sys.argv[1:3])
