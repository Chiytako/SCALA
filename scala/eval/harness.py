"""Japanese evaluation that runs directly against ScalaForCausalLM.

``perplexity``: token-level NLL (and bits/char) over a held-out stream.
``loglikelihood multiple choice``: the llm-jp-eval / lm-eval-harness zero-
or few-shot protocol -- pick the candidate continuation with the highest
length-normalised log-likelihood.  Covers JCommonsenseQA, JMMLU, JNLI, MARC-ja.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Sequence

import torch
import torch.nn.functional as F

from ..model.hierarchy import ScalaForCausalLM

__all__ = ["MCTask", "evaluate_perplexity", "evaluate_multiple_choice",
           "JA_TASKS", "build_task"]


# --------------------------------------------------------------------------- #
@dataclass
class MCTask:
    """One multiple-choice benchmark."""

    name: str
    hf: str
    config: Optional[str] = None
    split: str = "validation"
    #: record -> (context, [choices], gold_index)
    render: Callable[[dict], tuple[str, list[str], int]] = None
    n_shot: int = 0
    limit: Optional[int] = None
    fewshot_split: str = "train"


def _jcqa(rec: dict) -> tuple[str, list[str], int]:
    ch = [rec[f"choice{i}"] for i in range(5)]
    return f"質問: {rec['question']}\n答え:", [f" {c}" for c in ch], int(rec["label"])


def _jmmlu(rec: dict) -> tuple[str, list[str], int]:
    ch = [rec["A"], rec["B"], rec["C"], rec["D"]]
    gold = "ABCD".index(str(rec["answer"]).strip().upper())
    q = rec.get("question") or rec.get("input")
    return f"質問: {q}\n答え:", [f" {c}" for c in ch], gold


def _jnli(rec: dict) -> tuple[str, list[str], int]:
    labels = ["含意", "矛盾", "中立"]
    ctx = (f"前提: {rec['sentence1']}\n仮説: {rec['sentence2']}\n"
           f"関係(含意/矛盾/中立):")
    return ctx, [f" {x}" for x in labels], int(rec["label"])


def _marc(rec: dict) -> tuple[str, list[str], int]:
    labels = ["ポジティブ", "ネガティブ"]
    return (f"レビュー: {rec['sentence']}\n感情:", [f" {x}" for x in labels],
            int(rec["label"]))


#: Japanese MC tasks; repo ids from the llm-jp-eval / JGLUE ecosystem.
JA_TASKS: dict[str, MCTask] = {
    "jcommonsenseqa": MCTask("jcommonsenseqa", "shunk031/JGLUE", "JCommonsenseQA",
                             "validation", _jcqa, n_shot=4),
    "jnli": MCTask("jnli", "shunk031/JGLUE", "JNLI", "validation", _jnli, n_shot=4),
    "marc_ja": MCTask("marc_ja", "shunk031/JGLUE", "MARC-ja", "validation",
                      _marc, n_shot=4),
    "jmmlu": MCTask("jmmlu", "nlp-waseda/JMMLU_CC-BY-SA", None, "test",
                    _jmmlu, n_shot=0),
}


def build_task(name: str, **overrides) -> MCTask:
    t = JA_TASKS[name]
    return MCTask(**{**t.__dict__, **overrides})


# --------------------------------------------------------------------------- #
@torch.no_grad()
def _score_continuations(
    model: ScalaForCausalLM, tokenizer, context: str, choices: Sequence[str],
    device: torch.device, chunk_product: int, dtype: torch.dtype,
) -> list[float]:
    """Length-normalised log P(choice | context) for each choice."""
    ctx_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
    scores = []
    for ch in choices:
        cont = tokenizer(ch, add_special_tokens=False)["input_ids"]
        ids = ctx_ids + cont
        # sequence length must be a multiple of chunk_product; pad on the LEFT
        # so the scored continuation keeps its position at the end
        pad = (-len(ids)) % chunk_product
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
        padded = [pad_id] * pad + ids
        x = torch.tensor([padded], device=device)

        with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
            logits = model(x, return_logits=True).logits[0].float()
        # logits[i] predicts token i -- no shift
        start = len(padded) - len(cont)
        lp = F.log_softmax(logits[start:], dim=-1)
        tgt = torch.tensor(cont, device=device)
        total = lp.gather(1, tgt[:, None]).sum().item()
        scores.append(total / max(len(cont), 1))
    return scores


@torch.no_grad()
def evaluate_multiple_choice(
    model: ScalaForCausalLM, tokenizer, task: MCTask,
    device: torch.device | str = "cuda", dtype: torch.dtype = torch.bfloat16,
    verbose: bool = True,
) -> dict[str, float]:
    from datasets import load_dataset

    device = torch.device(device)
    model.eval()
    cp = model.cfg.chunk_product

    kw = {"name": task.config} if task.config else {}
    ds = load_dataset(task.hf, split=task.split, trust_remote_code=True, **kw)

    prefix = ""
    if task.n_shot:
        shots = load_dataset(task.hf, split=task.fewshot_split,
                             trust_remote_code=True, **kw).select(range(task.n_shot))
        parts = []
        for rec in shots:
            c, ch, g = task.render(rec)
            parts.append(c + ch[g])
        prefix = "\n\n".join(parts) + "\n\n"

    n = len(ds) if task.limit is None else min(task.limit, len(ds))
    correct = 0
    for i in range(n):
        c, ch, gold = task.render(ds[i])
        s = _score_continuations(model, tokenizer, prefix + c, ch, device, cp, dtype)
        correct += int(max(range(len(s)), key=s.__getitem__) == gold)
        if verbose and (i + 1) % 100 == 0:
            print(f"  {task.name}: {i+1}/{n}  acc={correct/(i+1):.4f}", flush=True)

    acc = correct / max(n, 1)
    return {f"{task.name}/acc": acc, f"{task.name}/n": float(n)}


# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_perplexity(
    model: ScalaForCausalLM, tokenizer, texts: Iterable[str],
    seq_len: int = 2048, device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.bfloat16, max_sequences: int = 200,
) -> dict[str, float]:
    """Token-level perplexity over ``texts``, packed into ``seq_len`` windows.

    Also returns bits per character, which unlike per-token perplexity is
    comparable across tokenizers:  BPC = nll / (ln2 * chars_per_token).
    ``chars_per_token`` is measured on the corpus being evaluated and covers
    the whole consumed stream, including the unscored tail left in ``buf``.
    """
    device = torch.device(device)
    model.eval()
    cp = model.cfg.chunk_product
    seq_len -= seq_len % cp

    buf: list[int] = []
    nll, ntok, nseq = 0.0, 0, 0
    nchars, nids = 0, 0
    eos = tokenizer.eos_token_id or 0

    for text in texts:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        nchars += len(text)
        nids += len(ids) + 1              # +1 for the eos appended below
        buf.extend(ids)
        buf.append(eos)
        while len(buf) >= seq_len and nseq < max_sequences:
            x = torch.tensor([buf[:seq_len]], device=device)
            buf = buf[seq_len:]
            with torch.autocast(device.type, dtype=dtype,
                                enabled=device.type == "cuda"):
                out = model(x, labels=x, return_logits=False)
            nll += float(out.loss_token) * seq_len
            ntok += seq_len
            nseq += 1
        if nseq >= max_sequences:
            break

    if ntok == 0:
        return {"ppl": float("nan"), "nll": float("nan"), "tokens": 0.0,
                "chars_per_token": float("nan"), "bpc": float("nan"),
                "chars": 0.0}
    mean = nll / ntok
    cpt = nchars / nids if nids else float("nan")
    return {"ppl": math.exp(min(mean, 20)), "nll": mean, "tokens": float(ntok),
            "chars": float(nchars), "chars_per_token": cpt,
            "bpc": mean / (math.log(2) * cpt)}
