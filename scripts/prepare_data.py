#!/usr/bin/env python
"""Stream Japanese/English/code corpora from the Hub, tokenize, and write
`<out>/<source>/<source>-NNNNN.bin` shards + manifest.json. Resumable:
existing complete shards are kept and counted.

    python scripts/prepare_data.py --config configs/data_ja_mix.yaml --out /data/tokens --budget-tokens 120e9 --workers 16
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scala.data.chat import ChatRenderError, render_chat  # noqa: E402

DTYPES = {"uint16": np.uint16, "uint32": np.uint32}


# --------------------------------------------------------------------------- #
def human(n: float) -> str:
    for unit in ("", "K", "M", "B", "T"):
        if abs(n) < 1000:
            return f"{n:.2f}{unit}"
        n /= 1000
    return f"{n:.2f}P"


def load_tokenizer(name: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name, use_fast=True, trust_remote_code=True)
    return tok


def iter_jsonl(src: dict):
    """Stream a repo's .jsonl files directly, bypassing the Arrow reader.

    Some corpora have per-file schema drift that aborts `datasets`' streaming
    JSON builder mid-read; reading raw lines sidesteps schema inference, and
    `path_glob` lets the mixture pick an exact file subset.
    """
    import fnmatch
    import gzip
    import urllib.request

    from huggingface_hub import HfApi

    repo = src["hf"]
    globs = src["path_glob"]
    globs = globs if isinstance(globs, list) else [globs]
    token = os.environ.get("HF_TOKEN")
    files = HfApi(token=token).list_repo_files(repo, repo_type="dataset")
    sel = sorted(f for f in files
                 if any(fnmatch.fnmatch(f, g) for g in globs))
    if not sel:
        raise FileNotFoundError(f"{repo}: no file matched {globs}")
    print(f"  [{src['name']}] {len(sel)} jsonl files matched {globs}")

    rev = src.get("revision", "main")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    for path in sel:
        url = (f"https://huggingface.co/datasets/{repo}/resolve/{rev}/"
               f"{urllib.parse.quote(path)}")
        try:
            with urllib.request.urlopen(
                    urllib.request.Request(url, headers=headers)) as fh:
                stream = gzip.GzipFile(fileobj=fh) if path.endswith(".gz") else fh
                for line in stream:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue          # one bad line must not end the file
        except Exception as e:  # noqa: BLE001 - one bad file must not end the run
            print(f"  [{src['name']}] skipping {path}: "
                  f"{type(e).__name__}: {str(e)[:80]}")


def iter_local_jsonl(src: dict):
    """Stream .jsonl / .jsonl.gz / .jsonl.zst files already on local disk
    (for offline environments with no route to the Hub).

    `root` + `path_glob` name the files, matched relative to `root`, e.g.:

        root: /data/raw/llm-jp-corpus-v3
        path_glob: ["ja/ja_wiki/train_*.jsonl.gz"]

    `skip_files` drops the first N matched files per source -- how a held-out
    eval slice is carved from files the tokenizer never saw.
    """
    import glob as _glob
    import gzip

    root = Path(src["root"]).expanduser()
    globs = src["path_glob"]
    globs = globs if isinstance(globs, list) else [globs]

    sel: list[Path] = []
    for g in globs:
        sel.extend(Path(p) for p in _glob.glob(str(root / g), recursive=True))
    # numeric-aware sort so `train_2` precedes `train_10`; a lexicographic
    # sort would make `skip_files` unreproducible
    def _key(p: Path):
        import re
        m = re.search(r"(\d+)(?=\D*$)", p.name)
        return (str(p.parent), int(m.group(1)) if m else -1, p.name)

    sel = sorted(set(sel), key=_key)
    skip = int(src.get("skip_files", 0))
    if skip:
        sel = sel[skip:]
    if not sel:
        raise FileNotFoundError(f"{src['name']}: no file matched {globs} under {root}")
    print(f"  [{src['name']}] {len(sel)} local files matched "
          f"{globs} (skipped {skip})", flush=True)

    def _open(path: Path):
        if path.suffix == ".gz":
            return gzip.open(path, "rt", encoding="utf-8", errors="replace")
        if path.suffix == ".zst":
            import io

            import zstandard

            fh = path.open("rb")
            rd = zstandard.ZstdDecompressor().stream_reader(fh)
            return io.TextIOWrapper(rd, encoding="utf-8", errors="replace")
        return path.open("rt", encoding="utf-8", errors="replace")

    for path in sel:
        try:
            with _open(path) as stream:
                for line in stream:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue          # one bad line must not end the file
        except Exception as e:  # noqa: BLE001 - one bad file must not end the run
            print(f"  [{src['name']}] skipping {path.name}: "
                  f"{type(e).__name__}: {str(e)[:80]}", flush=True)


def open_stream(src: dict, cache_dir: str | None):
    if src.get("reader") == "local_jsonl":
        return iter_local_jsonl(src)

    from datasets import load_dataset

    if src.get("reader") == "jsonl":
        return iter_jsonl(src)

    kwargs = dict(split=src.get("split", "train"), streaming=True)
    if src.get("config"):
        kwargs["name"] = src["config"]
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    ds = load_dataset(src["hf"], **kwargs)

    if src.get("format") == "chat":
        # chat sources need several columns at once (messages/conversations,
        # tools) whose names vary by publisher, so projecting would break them
        return ds

    # project to only the needed columns: some corpora have metadata columns
    # with schema drift across shards that otherwise aborts the Arrow reader
    # mid-stream
    keep = [src.get("text_key", "text")]
    keep += [k for k in (src.get("filter_meta") or {})]
    try:
        ds = ds.select_columns(keep)
    except Exception:  # noqa: BLE001 - older datasets, or column not present
        pass
    return ds


# --------------------------------------------------------------------------- #
class ShardWriter:
    def __init__(self, out_dir: Path, name: str, shard_tokens: int, dtype: str):
        self.dir = out_dir / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.shard_tokens = shard_tokens
        self.np_dtype = DTYPES[dtype]
        self.dtype = dtype
        self.buf = np.empty(shard_tokens, dtype=self.np_dtype)
        self.fill = 0
        self.shards: list[dict] = []
        self._scan_existing()

    def _scan_existing(self) -> None:
        for p in sorted(self.dir.glob(f"{self.name}-*.bin")):
            n = p.stat().st_size // np.dtype(self.np_dtype).itemsize
            if n == 0:
                p.unlink()
                continue
            self.shards.append({"path": f"{self.name}/{p.name}", "n_tokens": int(n)})

    @property
    def n_tokens(self) -> int:
        return sum(s["n_tokens"] for s in self.shards) + self.fill

    def add(self, ids: np.ndarray) -> None:
        pos = 0
        while pos < len(ids):
            room = self.shard_tokens - self.fill
            take = min(room, len(ids) - pos)
            self.buf[self.fill : self.fill + take] = ids[pos : pos + take]
            self.fill += take
            pos += take
            if self.fill == self.shard_tokens:
                self.flush()

    def flush(self) -> None:
        if self.fill == 0:
            return
        idx = len(self.shards)
        path = self.dir / f"{self.name}-{idx:05d}.bin"
        self.buf[: self.fill].tofile(path)
        self.shards.append({"path": f"{self.name}/{path.name}",
                            "n_tokens": int(self.fill)})
        self.fill = 0


# --------------------------------------------------------------------------- #
def process_source(src: dict, cfg: dict, out: Path, target: int, tok,
                   cache_dir: str | None, batch_texts: int = 1000) -> dict:
    name = src["name"]
    writer = ShardWriter(out, name, int(cfg["shard_tokens"]), cfg.get("dtype", "uint32"))
    if writer.n_tokens >= target:
        print(f"  [{name}] already have {human(writer.n_tokens)} >= "
              f"{human(target)} tokens, skipping")
        return {"name": name, "weight": src["weight"],
                "epochs_cap": src.get("epochs_cap", 1e9), "shards": writer.shards}

    eos_id = tok.eos_token_id
    if eos_id is None:
        eos_id = tok.convert_tokens_to_ids(cfg.get("eos_token", "</s>"))
    append_eos = cfg.get("append_eos", True)
    text_key = src.get("text_key", "text")
    filt = src.get("filter_meta")
    is_chat = src.get("format") == "chat"

    ds = open_stream(src, cache_dir)
    start = time.time()
    buf: list[str] = []
    buf_chars = 0
    n_docs = 0
    n_skipped = 0

    # target is re-checked only after a batch drains; cap by characters as
    # well as by document count so large documents (e.g. agent transcripts)
    # cannot cause large overshoot
    max_chars = int(cfg.get("batch_chars", 8_000_000))

    def drain() -> None:
        nonlocal buf, buf_chars
        buf_chars = 0
        if not buf:
            return
        enc = tok(buf, add_special_tokens=False)["input_ids"]
        flat: list[int] = []
        for ids in enc:
            flat.extend(ids)
            if append_eos:
                flat.append(eos_id)
        writer.add(np.asarray(flat, dtype=writer.np_dtype))
        buf = []

    try:
        for rec in ds:
            if filt and any(rec.get(k) != v for k, v in filt.items()):
                continue
            if is_chat:
                # A handful of malformed rows per million is normal in these
                # dumps; skipping one must not cost the whole source.
                try:
                    txt = render_chat(rec, tok,
                                      messages_key=src.get("messages_key"),
                                      tools_key=src.get("tools_key", "tools"),
                                      from_fields=src.get("chat_from_fields"))
                except ChatRenderError:
                    n_skipped += 1
                    continue
            else:
                txt = rec.get(text_key)
            if not txt:
                continue
            buf.append(txt)
            buf_chars += len(txt)
            n_docs += 1
            if len(buf) >= batch_texts or buf_chars >= max_chars:
                drain()
                done = writer.n_tokens
                if done >= target:
                    break
                if n_docs % (batch_texts * 20) == 0:
                    el = time.time() - start
                    print(f"  [{name}] {human(done)}/{human(target)} tok "
                          f"({100*done/target:5.1f}%)  {human(done/max(el,1))} tok/s",
                          flush=True)
    except KeyboardInterrupt:
        print(f"  [{name}] interrupted; flushing")
    finally:
        drain()
        writer.flush()

    skipped = f", {n_skipped:,} unrenderable rows skipped" if n_skipped else ""
    print(f"  [{name}] done: {human(writer.n_tokens)} tokens in "
          f"{len(writer.shards)} shards ({time.time()-start:.0f}s){skipped}")
    return {"name": name, "weight": src["weight"],
            "epochs_cap": src.get("epochs_cap", 1e9), "shards": writer.shards}


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data_ja_mix.yaml")
    ap.add_argument("--out", default="data/tokens")
    ap.add_argument("--budget-tokens", type=float, default=120e9,
                    help="total training token budget; per-source targets are "
                         "budget * weight * epochs_cap")
    ap.add_argument("--only", nargs="*", default=None, help="subset of source names")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("HF_WORKERS", 8)))
    ap.add_argument("--cache-dir", default=os.environ.get("HF_DATASETS_CACHE"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rebuild-manifest", action="store_true",
                    help="scan <out> for shard directories and rewrite "
                         "manifest.json from what is actually on disk")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    budget = int(args.budget_tokens)

    sources = cfg["sources"]
    if args.only:
        sources = [s for s in sources if s["name"] in args.only]
        if not sources:
            sys.exit(f"no source matched {args.only}")

    print(f"tokenizer: {cfg['tokenizer']}")
    print(f"budget   : {human(budget)} tokens")
    print(f"output   : {out.resolve()}\n")

    if args.rebuild_manifest:
        # Recover from an interrupted run: the shards are the source of truth.
        weights = {s["name"]: s.get("weight", 1.0) for s in cfg["sources"]}
        caps = {s["name"]: s.get("epochs_cap", 1e9) for s in cfg["sources"]}
        itemsize = np.dtype(DTYPES[cfg.get("dtype", "uint32")]).itemsize
        entries, total = [], 0
        for d in sorted(p for p in out.iterdir() if p.is_dir()):
            shards = []
            for f in sorted(d.glob("*.bin")):
                n = f.stat().st_size // itemsize
                if n:
                    shards.append({"path": f"{d.name}/{f.name}", "n_tokens": int(n)})
            if not shards:
                continue
            n = sum(s["n_tokens"] for s in shards)
            total += n
            entries.append({"name": d.name, "weight": weights.get(d.name, 1.0),
                            "epochs_cap": caps.get(d.name, 1e9), "shards": shards})
            print(f"  {d.name:<20} {human(n):>10} tokens in {len(shards)} shards"
                  + ("" if d.name in weights else "   [no weight in config -> 1.0]"))
        (out / "manifest.json").write_text(
            json.dumps({"tokenizer": cfg["tokenizer"],
                        "vocab_size": None,
                        "dtype": cfg.get("dtype", "uint32"),
                        "sources": entries}, indent=2), encoding="utf-8")
        print(f"\nrebuilt {out/'manifest.json'}: {human(total)} tokens")
        return

    if args.dry_run:
        from datasets import load_dataset  # noqa: F401

        ok = True
        # only paid for if some source is a chat source
        dry_tok = [None]

        def _tok():
            if dry_tok[0] is None:
                dry_tok[0] = load_tokenizer(cfg["tokenizer"])
            return dry_tok[0]

        for s in sources:
            tgt = int(budget * s["weight"] * min(s.get("epochs_cap", 1.0), 4.0))
            key = s.get("text_key", "text")
            try:
                rec = next(iter(open_stream(s, args.cache_dir)))
            except Exception as e:  # noqa: BLE001
                ok = False
                print(f"  FAIL {s['name']:<22} {type(e).__name__}: "
                      f"{str(e)[:110]}")
                continue

            if s.get("format") == "chat":
                # For chat sources "is the column there" is the wrong question:
                # what matters is whether the record survives the harmony
                # template, so actually render one.
                try:
                    txt = render_chat(rec, _tok(),
                                      messages_key=s.get("messages_key"),
                                      tools_key=s.get("tools_key", "tools"),
                                      from_fields=s.get("chat_from_fields"))
                except Exception as e:  # noqa: BLE001
                    ok = False
                    print(f"  FAIL {s['name']:<22} chat render: "
                          f"{type(e).__name__}: {str(e)[:90]}; "
                          f"columns: {', '.join(list(rec)[:8])}")
                    continue
                print(f"  OK   {s['name']:<22} target={human(tgt):>8}  "
                      f"chat ok ({len(txt):,} chars rendered)")
                continue

            # reachability is not enough -- the configured text_key has to be
            # there and hold real text, or tokenisation silently yields nothing
            if key not in rec:
                ok = False
                print(f"  FAIL {s['name']:<22} no '{key}' column; "
                      f"available: {', '.join(list(rec)[:8])}")
            elif not isinstance(rec[key], str) or not rec[key].strip():
                ok = False
                print(f"  FAIL {s['name']:<22} '{key}' is not usable text "
                      f"({type(rec[key]).__name__})")
            else:
                n = len(rec[key])
                print(f"  OK   {s['name']:<22} target={human(tgt):>8}  "
                      f"'{key}' ok ({n:,} chars in first doc)")
        sys.exit(0 if ok else 1)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    os.environ.setdefault("RAYON_NUM_THREADS", str(args.workers))
    tok = load_tokenizer(cfg["tokenizer"])
    print(f"vocab_size: {len(tok)}\n")

    man_path = out / "manifest.json"

    def write_manifest(entries: list[dict]) -> int:
        """Merge ``entries`` into the manifest on disk and return total tokens."""
        manifest = {"tokenizer": cfg["tokenizer"], "vocab_size": len(tok),
                    "dtype": cfg.get("dtype", "uint32"), "sources": []}
        if man_path.exists():
            old = json.loads(man_path.read_text(encoding="utf-8"))
            names = {e["name"] for e in entries}
            manifest["sources"] = [s for s in old.get("sources", [])
                                   if s["name"] not in names]
        manifest["sources"].extend(e for e in entries if e["shards"])
        man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return sum(sh["n_tokens"] for s in manifest["sources"]
                   for sh in s["shards"])

    entries, failed = [], []
    for s in sources:
        # up-sampling is handled at train time by `weight`; here we only need
        # enough *unique* tokens to satisfy epochs_cap.
        target = int(budget * s["weight"] / max(s.get("epochs_cap", 1.0), 1e-9))
        origin = s.get("hf") or s.get("root", "local")
        print(f"== {s['name']} ({origin}) target {human(target)} tokens")
        try:
            entries.append(process_source(s, cfg, out, target, tok,
                                          args.cache_dir))
        except (MemoryError, Exception) as e:  # noqa: BLE001
            # One unreachable / oversized source must not cost you every other
            # source's work -- the manifest is rewritten after each one.
            print(f"  [{s['name']}] FAILED: {type(e).__name__}: {e}")
            failed.append(s["name"])
            writer_dir = out / s["name"]
            if writer_dir.exists():
                salvage = ShardWriter(out, s["name"], int(cfg["shard_tokens"]),
                                      cfg.get("dtype", "uint32"))
                if salvage.shards:
                    print(f"  [{s['name']}] salvaged "
                          f"{human(salvage.n_tokens)} already-written tokens")
                    entries.append({"name": s["name"], "weight": s["weight"],
                                    "epochs_cap": s.get("epochs_cap", 1e9),
                                    "shards": salvage.shards})
        total = write_manifest(entries)
        print(f"  manifest updated: {human(total)} unique tokens so far")

    total = write_manifest(entries)
    print(f"\nmanifest: {man_path}   total unique tokens: {human(total)}")
    if failed:
        print(f"failed sources (re-run with --only to retry): {' '.join(failed)}")


if __name__ == "__main__":
    main()
