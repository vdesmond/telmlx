#!/usr/bin/env python3
import argparse, json, os, time
import faiss, numpy as np
import mlx.core as mx
from mlx_embeddings import load

DOC_PROMPT = "title: none | text: "


def embed(model, tok, texts, max_length=512):
    enc = tok._tokenizer([DOC_PROMPT + t for t in texts], padding=True, truncation=True,
                         max_length=max_length, return_tensors="mlx")
    e = model(enc["input_ids"], attention_mask=enc["attention_mask"]).text_embeds
    mx.eval(e)
    return np.asarray(e.astype(mx.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chunks"); ap.add_argument("out")
    ap.add_argument("--model", default="mlx-community/embeddinggemma-300m-bf16")
    ap.add_argument("--batch", type=int, default=64); ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.chunks)]
    if a.limit:
        rows = rows[:a.limit]
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["text"]))
    model, tok = load(a.model)
    ckpt = a.out + ".partial.npz"
    if os.path.exists(ckpt):
        z = np.load(ckpt); vecs, start = z["vecs"], int(z["done"])
        print(f"resuming at {start}/{len(rows)}", flush=True)
    else:
        vecs, start = np.zeros((len(rows), 768), dtype=np.float32), 0
    t0 = time.time()
    for s in range(start, len(order), a.batch):
        idx = order[s:s + a.batch]
        vecs[idx] = embed(model, tok, [rows[i]["text"] for i in idx])
        done = s + len(idx)
        if (s // a.batch) % 100 == 0 or done == len(order):
            np.savez(ckpt, vecs=vecs, done=done)
            print(f"{done}/{len(rows)} chunks, {(done - start) / (time.time() - t0):.1f} chunks/s, "
                  f"peak {mx.get_peak_memory() / 1e9:.2f} GB", flush=True)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(768); index.add(vecs)
    faiss.write_index(index, a.out + ".faiss")
    np.save(a.out + ".ids.npy", np.array([r["id"] for r in rows]))
    os.remove(ckpt)
    print(f"wrote {a.out}.faiss ({index.ntotal} vectors) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
