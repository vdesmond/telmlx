#!/usr/bin/env python3
import argparse, json, os, re, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import faiss, httpx, numpy as np
import mlx.core as mx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from mlx_embeddings import load as load_emb
from mlx_lm import load as load_lm

QUERY_PROMPT = "task: search result | query: "
RR_PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the '
             'Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
RR_SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
RR_INSTRUCT = "Given a question about 3GPP specifications, retrieve passages that answer it."
CONTEXT_TEMPLATE = "Context:\n{ctx}\n\nQuestion: {q}"

ap = argparse.ArgumentParser()
ap.add_argument("--upstream", default="http://127.0.0.1:8080"); ap.add_argument("--port", type=int, default=8081)
ap.add_argument("--index", default="data/index"); ap.add_argument("--chunks", default="data/chunks.jsonl")
ap.add_argument("--embed_model", default="mlx-community/embeddinggemma-300m-bf16")
ap.add_argument("--rerank_model", default="mlx-community/Qwen3-Reranker-0.6B-4bit")
ap.add_argument("--top_n", type=int, default=15, help="candidates handed to the reranker")
ap.add_argument("--top_k", type=int, default=4, help="passages kept after reranking")
ap.add_argument("--neighbors", type=int, default=1, help="previous/next parts of a kept passage's section to add")
ap.add_argument("--bm25", action="store_true", help="fuse BM25 with dense retrieval (RRF); off by default, it pulls in test-spec noise on this corpus")
ap.add_argument("--no_rerank", action="store_true")
ap.add_argument("--max_chars", type=int, default=2000, help="per-passage cap in the prompt (chunks are <= 3000 chars)")
ap.add_argument("--rerank_chars", type=int, default=800, help="per-passage cap seen by the reranker (its cost is linear in this)")
ap.add_argument("--log", default=None, help="JSONL of (question, passages) per request")
args = ap.parse_args()

index = faiss.read_index(args.index + ".faiss")
ids = np.load(args.index + ".ids.npy")
chunks = {}
for line in open(args.chunks):
    r = json.loads(line); chunks[r["id"]] = r
by_key = {(c["spec"], c["section"], c["part"]): c for c in chunks.values()}
bm25 = None
if args.bm25:
    from rank_bm25 import BM25Okapi
    TOKEN = re.compile(r"[A-Za-z0-9_\-\.]+")
    order_ids = [int(i) for i in ids]
    t0 = time.time()
    bm25 = BM25Okapi([TOKEN.findall(chunks[i]["text"].lower()) for i in order_ids])
    print(f"bm25 over {len(order_ids)} chunks in {time.time() - t0:.0f}s", flush=True)
emb_model, emb_tok = load_emb(args.embed_model)
rr_model, rr_tok = (None, None) if args.no_rerank else load_lm(args.rerank_model)
if rr_tok:
    YES, NO = rr_tok.encode("yes", add_special_tokens=False)[0], rr_tok.encode("no", add_special_tokens=False)[0]
print(f"index {index.ntotal} vectors, {len(chunks)} chunks, rerank={'off' if args.no_rerank else 'on'}", flush=True)


def retrieval_query(content: str) -> str:
    if content.startswith("Answer the following multiple choice question"):
        content = content.split("\n\n", 1)[-1]
    return content[:2000]


def question_only(q: str) -> str:
    return re.split(r"\n\s*A\)", q, maxsplit=1)[0].strip()[:600]


def retrieve(q: str):
    q = retrieval_query(q)
    enc = emb_tok._tokenizer([QUERY_PROMPT + q], padding=True, truncation=True, max_length=512, return_tensors="mlx")
    e = emb_model(enc["input_ids"], attention_mask=enc["attention_mask"]).text_embeds
    mx.eval(e); v = np.asarray(e.astype(mx.float32)); v /= np.linalg.norm(v)
    _, I = index.search(v, args.top_n)
    dense = [int(ids[i]) for i in I[0] if i >= 0]
    if bm25 is not None:
        sc = bm25.get_scores(TOKEN.findall(question_only(q).lower()))
        lex = [order_ids[i] for i in np.argsort(sc)[::-1][:args.top_n]]
        fused = {}
        for lst in (dense, lex):
            for r, cid in enumerate(lst):
                fused[cid] = fused.get(cid, 0) + 1 / (60 + r)
        cand_ids = sorted(fused, key=fused.get, reverse=True)[:args.top_n]
    else:
        cand_ids = dense
    cands = [chunks[c] for c in cand_ids]
    if args.no_rerank:
        return expand(cands[:args.top_k])
    head = rr_tok.encode(f"{RR_PREFIX}<Instruct>: {RR_INSTRUCT}\n<Query>: {question_only(q)}\n<Document>: ", add_special_tokens=False)
    tail = rr_tok.encode(RR_SUFFIX, add_special_tokens=False)
    scores = []
    for c in cands:
        body = rr_tok.encode(c["text"][:args.rerank_chars], add_special_tokens=False)[: 1024 - len(head) - len(tail)]
        lg = rr_model(mx.array([head + body + tail]))[0, -1].astype(mx.float32)
        scores.append(float(lg[YES] - lg[NO]))
    order = np.argsort(scores)[::-1][:args.top_k]
    return expand([cands[i] for i in order])


def expand(kept):
    out, seen = [], set()
    for i, c in enumerate(kept):
        deltas = [0] + ([x for k in range(1, args.neighbors + 1) for x in (-k, k)] if i < 2 else [])
        for d in deltas:
            n = by_key.get((c["spec"], c["section"], c["part"] + d))
            if n and n["id"] not in seen:
                seen.add(n["id"]); out.append(n)
    return out[:args.top_k + 4]


_MATH_SUBS = [(r"\\text\{([^}]*)\}", r"\1"), (r"\\mathrm\{([^}]*)\}", r"\1"), (r"\\left|\\right", ""),
              (r"\\cdot", "·"), (r"\\times", "×"), (r"\\le(?:q)?\b", "≤"), (r"\\ge(?:q)?\b", "≥"), (r"\\ne(?:q)?\b", "≠"),
              (r"\\(?:mu|alpha|beta|Delta|lambda)\b", lambda m: {"mu": "μ", "alpha": "α", "beta": "β", "Delta": "Δ", "lambda": "λ"}[m.group(0)[1:]]),
              (r"\^\{([^}]*)\}", r"^\1"), (r"_\{([^}]*)\}", r"_\1"), (r"[{}]", "")]


def _tex(t: str) -> str:
    for pat, rep in _MATH_SUBS:
        t = re.sub(pat, rep, t)
    return t.strip()


def clean(text: str) -> str:
    text = text[:args.max_chars]
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)(\{[^}]*\})?", "", text)          # images
    text = re.sub(r"\{width=[^}]*\}", "", text)
    text = re.sub(r"\$\$?([^$]+)\$\$?", lambda m: _tex(m.group(1)), text)  # inline math
    text = text.replace("\\_", "_").replace("\\-", "-").replace("\\[", "[").replace("\\]", "]").replace("\\*", "*").replace("\\>", ">")
    text = re.sub(r"\+?[-=+]{4,}\+?", " ", text)                          # grid-table rulers
    text = re.sub(r"(\|\s*){2,}", "| ", text)                              # runs of empty cells
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def passages_json(passages):
    return [{"spec": p["spec"], "section": clean(p["section"]), "part": p["part"], "text": clean(p["text"])} for p in passages]


def with_context(question: str, passages) -> str:
    ctx = "\n\n".join(f"[{i + 1}] {clean(p['text'])}" for i, p in enumerate(passages))
    return CONTEXT_TEMPLATE.format(ctx=ctx, q=question)


app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
client = httpx.AsyncClient(base_url=args.upstream, timeout=600)


@app.post("/rag/retrieve")
async def retrieve_only(req: Request):
    body = await req.json()
    t0 = time.time(); passages = retrieve(body["question"]); dt = time.time() - t0
    return {"passages": passages_json(passages), "retrieval_s": round(dt, 2)}


@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    for m in reversed(body.get("messages", [])):
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            t0 = time.time(); passages = retrieve(m["content"]); dt = time.time() - t0
            if args.log:
                with open(args.log, "a") as f:
                    f.write(json.dumps({"q": m["content"], "passages": [(p["spec"], p["section"], p["id"]) for p in passages],
                                        "retrieval_s": round(dt, 3)}) + "\n")
            m["content"] = with_context(m["content"], passages)
            break
    r = await client.post("/v1/chat/completions", json=body)
    return JSONResponse(r.json(), status_code=r.status_code)


@app.api_route("/{path:path}", methods=["GET", "POST"])
async def passthrough(path: str, req: Request):
    r = await client.request(req.method, f"/{path}", content=await req.body(), headers={"content-type": req.headers.get("content-type", "")})
    return Response(r.content, status_code=r.status_code, media_type=r.headers.get("content-type"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
