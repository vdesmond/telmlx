import { Fragment, useEffect, useRef, useState } from "react";

const PROXY = "http://127.0.0.1:8081";   // retrieval (EmbeddingGemma → FAISS → Qwen3-Reranker)
const LLM = "http://127.0.0.1:8080";     // mlx_lm.server, OpenAI-compatible, streams

type Passage = { spec: string; section: string; part: number; text: string };
type Stats = { retrieval?: number; ttft?: number; total?: number; tokens?: number };

const MODELS = [
    { id: "mlx-community/gemma-4-e4b-it-4bit", label: "Gemma-4-E4B", strict: false },
    { id: "models/OTel-LLM-E4B-IT-8bit", label: "OTel-LLM-E4B-IT (strict)", strict: true },
];

const EXAMPLES = [
    "What is the size in bits of the HARQ process number field in SCI format 2-A?",
    "In sidelink mode 2, what does the UE do if fewer than X·M_total candidate resources remain after RSRP exclusion?",
    "What is the maximum number of SL HARQ processes for NR sidelink?",
    "Which SCI format carries the resource reservation period in NR sidelink?",
    "What does DMRS bundling mean for PUSCH repetition?",
];

const SYSTEM = "You answer questions about 3GPP specifications using only the numbered context passages. Be concise: one to three sentences, then cite the passage numbers you used like [1]. If the passages do not contain the answer, say so in one sentence.";

async function streamChat(body: object, signal: AbortSignal, onDelta: (s: string) => void) {
    const r = await fetch(`${LLM}/v1/chat/completions`, { method: "POST", headers: { "content-type": "application/json" }, signal, body: JSON.stringify({ ...body, stream: true }) });
    if (!r.ok || !r.body) throw new Error(`generation: ${r.status} ${await r.text()}`);
    const reader = r.body.getReader(); const dec = new TextDecoder();
    let buf = "";
    for (; ;) {
        const { value, done } = await reader.read(); if (done) break;
        buf += dec.decode(value, { stream: true });
        const lines = buf.split("\n"); buf = lines.pop() ?? "";
        for (const line of lines) {
            if (!line.startsWith("data:")) continue;
            const data = line.slice(5).trim(); if (data === "[DONE]") continue;
            const delta = JSON.parse(data).choices?.[0]?.delta?.content ?? "";
            if (delta) onDelta(delta);
        }
    }
}

function Cited({ text, n, onCite }: { text: string; n: number; onCite: (i: number) => void }) {
    const parts = text.split(/(\[\d+(?:\s*,\s*\d+)*\])/g);
    return parts.map((p, k) => {
        const m = /^\[([\d,\s]+)\]$/.exec(p);
        if (!m) return <Fragment key={k}>{p}</Fragment>;
        return (
            <span key={k} className="cites">
                {m[1].split(",").map((s) => {
                    const i = parseInt(s, 10);
                    return i >= 1 && i <= n
                        ? <button key={s} className="cite" onClick={() => onCite(i)}>{i}</button>
                        : <span key={s} className="cite dead">{i}</span>;
                })}
            </span>
        );
    });
}

export default function App() {
    const [question, setQuestion] = useState(EXAMPLES[0]);
    const [model, setModel] = useState(MODELS[0].id);
    const [rag, setRag] = useState(true);
    const [phase, setPhase] = useState<"idle" | "retrieving" | "generating">("idle");
    const [passages, setPassages] = useState<Passage[]>([]);
    const [open, setOpen] = useState<number | null>(null);
    const [answer, setAnswer] = useState("");
    const [stats, setStats] = useState<Stats>({});
    const [error, setError] = useState<string | null>(null);
    const abort = useRef<AbortController | null>(null);

    useEffect(() => () => abort.current?.abort(), []);

    function stop() { abort.current?.abort(); }

    function jump(i: number) {
        setOpen(i);
        document.getElementById(`p${i}`)?.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    async function ask() {
        abort.current?.abort();
        const ac = new AbortController(); abort.current = ac;
        setError(null); setAnswer(""); setPassages([]); setOpen(null); setStats({});
        const t0 = performance.now();
        try {
            let ctx: Passage[] = [];
            if (rag) {
                setPhase("retrieving");
                const r = await fetch(`${PROXY}/rag/retrieve`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ question }), signal: ac.signal });
                if (!r.ok) throw new Error(`retrieval: ${r.status} ${await r.text()}`);
                const j = await r.json(); ctx = j.passages; setPassages(ctx); setStats({ retrieval: j.retrieval_s });
            }
            setPhase("generating");
            const strict = MODELS.find((m) => m.id === model)?.strict;
            const user = rag ? `Context:\n${ctx.map((p, i) => `[${i + 1}] ${p.text}`).join("\n\n")}\n\nQuestion: ${question}` : question;
            const messages = strict || !rag ? [{ role: "user", content: user }] : [{ role: "system", content: SYSTEM }, { role: "user", content: user }];
            let text = "", tokens = 0, first = 0;
            await streamChat({ model, messages, max_tokens: 300, temperature: 0 }, ac.signal, (d) => {
                if (!first) first = performance.now();
                tokens++; text += d; setAnswer(text);
            });
            setStats((s) => ({ ...s, ttft: (first - t0) / 1000, total: (performance.now() - t0) / 1000, tokens }));
        } catch (e) {
            if ((e as Error).name !== "AbortError") setError(String(e));
        } finally { setPhase("idle"); }
    }

    const busy = phase !== "idle";
    return (
        <main>
            <header>
                <h1>telmlx</h1>
                <div className="controls">
                    <select value={model} onChange={(e) => setModel(e.target.value)} disabled={busy}>
                        {MODELS.map((m) => <option key={m.id} value={m.id}>{m.label}</option>)}
                    </select>
                    <label><input type="checkbox" checked={rag} onChange={(e) => setRag(e.target.checked)} disabled={busy} /> retrieval</label>
                </div>
            </header>

            <div className="ask">
                <textarea value={question} onChange={(e) => setQuestion(e.target.value)} rows={2}
                    onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); if (!busy && question.trim()) ask(); } }} />
                {busy
                    ? <button className="stop" onClick={stop}>{phase === "retrieving" ? "searching…" : "stop"}</button>
                    : <button onClick={ask} disabled={!question.trim()}>ask</button>}
            </div>
            <div className="chips">
                {EXAMPLES.map((q) => <button key={q} className="chip" onClick={() => setQuestion(q)} disabled={busy}>{q}</button>)}
            </div>

            {error && <pre className="error">{error}</pre>}

            {(answer || phase === "generating") && (
                <section className="answer">
                    <p><Cited text={answer} n={passages.length} onCite={jump} />{phase === "generating" && <span className="cursor">▍</span>}</p>
                    <small>
                        {stats.retrieval !== undefined && <span>retrieval {stats.retrieval.toFixed(1)} s</span>}
                        {stats.ttft !== undefined && <span>first token {stats.ttft.toFixed(1)} s</span>}
                        {stats.total !== undefined && <span>total {stats.total.toFixed(1)} s</span>}
                        {stats.tokens !== undefined && stats.total! - stats.ttft! > 0.5 && <span>{stats.tokens} tokens · {(stats.tokens / (stats.total! - stats.ttft!)).toFixed(0)} tok/s</span>}
                    </small>
                </section>
            )}

            {passages.length > 0 && (
                <section className="passages">
                    <h2>passages</h2>
                    {passages.map((p, i) => (
                        <details key={i} id={`p${i + 1}`} open={open === null ? i === 0 : open === i + 1}
                            onToggle={(e) => { if ((e.target as HTMLDetailsElement).open) setOpen(i + 1); else if (open === i + 1) setOpen(0); }}>
                            <summary><span className="n">{i + 1}</span> {p.spec} <span className="sec">§ {p.section || "front matter"}</span></summary>
                            <pre>{p.text.replace(/^.*\n/, "")}</pre>{/* first line repeats the heading */}
                        </details>
                    ))}
                </section>
            )}
        </main>
    );
}
