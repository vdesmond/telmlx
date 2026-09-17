#!/usr/bin/env python3
import glob, json, os, re, sys, collections

SPEC = re.compile(r"^(38\d{3}(?:-\d+)?)-([a-z][0-9a-z]\d)")
HEAD = re.compile(r"^(#{1,6})\s+(.*?)\s*(?:\{#.*\})?\s*$")


def latest_files(root):
    specs = collections.defaultdict(list)
    for f in glob.glob(f"{root}/Rel-*/38_series/*.md"):
        m = SPEC.match(os.path.basename(f))
        if m:
            specs[m.group(1)].append((m.group(2), f))
    return [f for v in specs.values() for ver, f in v if ver == max(x[0] for x in v)]


def sections(text):
    title, buf = "", []
    for line in text.splitlines():
        m = HEAD.match(line)
        if m:
            if buf:
                yield title, "\n".join(buf)
            title, buf = m.group(2), []
        else:
            buf.append(line)
    if buf:
        yield title, "\n".join(buf)


def windows(words, size, overlap):
    i = 0
    while i < len(words):
        yield words[i:i + size]
        if i + size >= len(words):
            break
        i += size - overlap


def main(root, out, size=350, overlap=60, min_words=25):
    n = 0
    with open(out, "w") as fo:
        for f in sorted(latest_files(root)):
            spec = SPEC.match(os.path.basename(f)).group(1)
            spec_id = f"TS 38.{spec[2:5]}" + (f"-{spec[6:]}" if "-" in spec else "")
            text = open(f, errors="ignore").read()
            for title, body in sections(text):
                words = body.split()
                if len(words) < min_words:
                    continue
                for k, w in enumerate(windows(words, size, overlap)):
                    fo.write(json.dumps({"id": n, "spec": spec_id, "section": title, "part": k,
                                         "text": f"{spec_id} {title}\n" + " ".join(w)}) + "\n")
                    n += 1
    print(f"{n} chunks from {len(latest_files(root))} files -> {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
