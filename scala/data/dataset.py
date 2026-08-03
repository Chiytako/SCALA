"""Packed token dataset over memory-mapped shards.

Layout (``scripts/prepare_data.py``): ``<root>/manifest.json`` plus
``<source>/<source>-NNNNN.bin`` flat uint token ids, documents concatenated
and EOS-separated.  Sequences are cut at ``seq_len`` boundaries; ``seq_len``
must be a multiple of the chunk product ``C_<=L``.
"""

from __future__ import annotations

import json
import math
import random
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import IterableDataset

__all__ = ["ShardIndex", "MixtureSpec", "PackedTokenDataset", "collate"]

_DTYPES = {"uint16": np.uint16, "uint32": np.uint32}


# --------------------------------------------------------------------------- #
@dataclass
class ShardIndex:
    path: Path
    n_tokens: int
    dtype: str = "uint32"

    _mm: np.memmap | None = field(default=None, repr=False, compare=False)

    def array(self) -> np.memmap:
        if self._mm is None:
            self._mm = np.memmap(self.path, dtype=_DTYPES[self.dtype], mode="r")
        return self._mm


@dataclass
class SourceIndex:
    name: str
    shards: list[ShardIndex]
    weight: float
    epochs_cap: float = 1e9

    @property
    def n_tokens(self) -> int:
        return sum(s.n_tokens for s in self.shards)


@dataclass
class MixtureSpec:
    sources: list[SourceIndex]

    @classmethod
    def from_manifest(cls, root: str | Path,
                      weight_overrides: dict[str, float] | None = None
                      ) -> "MixtureSpec":
        root = Path(root)
        man = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        srcs = []
        for s in man["sources"]:
            shards = [
                ShardIndex(root / sh["path"], sh["n_tokens"], man.get("dtype", "uint32"))
                for sh in s["shards"]
            ]
            if not shards:
                continue
            w = (weight_overrides or {}).get(s["name"], s["weight"])
            srcs.append(SourceIndex(s["name"], shards, w, s.get("epochs_cap", 1e9)))
        if not srcs:
            raise RuntimeError(f"no shards found under {root}")
        tot = sum(s.weight for s in srcs)
        for s in srcs:
            s.weight /= tot
        return cls(srcs)

    def describe(self) -> str:
        lines = [f"{'source':<24}{'tokens':>16}{'weight':>10}{'epochs@budget':>16}"]
        lines.append("-" * 66)
        for s in self.sources:
            lines.append(f"{s.name:<24}{s.n_tokens:>16,}{s.weight:>10.3f}"
                         f"{'':>16}")
        lines.append("-" * 66)
        lines.append(f"{'TOTAL':<24}{sum(s.n_tokens for s in self.sources):>16,}")
        return "\n".join(lines)

    def epochs_at_budget(self, budget_tokens: int) -> dict[str, float]:
        return {
            s.name: (budget_tokens * s.weight) / max(s.n_tokens, 1)
            for s in self.sources
        }


# --------------------------------------------------------------------------- #
class PackedTokenDataset(IterableDataset):
    """Infinite stream of ``seq_len`` token windows drawn from a mixture.

    Each (rank, worker) pair gets a disjoint, deterministic stride over shard
    windows; the stream is reproducible from ``seed`` alone.  ``holdout_frac``
    reserves the last fraction of every shard's windows: ``split="train"``
    never returns them, ``split="holdout"`` returns only them.
    """

    def __init__(
        self,
        mixture: MixtureSpec,
        seq_len: int,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
        chunk_product: int = 1,
        shuffle_buffer: int = 4096,
        holdout_frac: float = 0.0,
        split: str = "train",
    ):
        super().__init__()
        if seq_len % chunk_product:
            raise ValueError(
                f"seq_len={seq_len} must be a multiple of C_<=L={chunk_product}"
            )
        if not 0.0 <= holdout_frac < 1.0:
            raise ValueError(f"holdout_frac must be in [0, 1), got {holdout_frac}")
        if split not in ("train", "holdout"):
            raise ValueError(f"split must be 'train' or 'holdout', got {split!r}")
        if split == "holdout" and holdout_frac <= 0.0:
            raise ValueError("split='holdout' is empty unless holdout_frac > 0")
        self.mixture = mixture
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.shuffle_buffer = shuffle_buffer
        self.holdout_frac = holdout_frac
        self.split = split

    # ------------------------------------------------------------------ #
    def _worker_info(self) -> tuple[int, int]:
        info = torch.utils.data.get_worker_info()
        if info is None:
            return 0, 1
        return info.id, info.num_workers

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        wid, nworkers = self._worker_info()
        stream_id = self.rank * nworkers + wid
        n_streams = self.world_size * nworkers
        rng = random.Random(self.seed * 1_000_003 + stream_id)

        # per-source cursors, each walking its own shard list
        cursors = {s.name: _SourceCursor(s, self.seq_len, stream_id, n_streams,
                                         self.seed, self.holdout_frac,
                                         self.split)
                   for s in self.mixture.sources}
        names = [s.name for s in self.mixture.sources]
        weights = [s.weight for s in self.mixture.sources]

        buf: list[np.ndarray] = []
        while True:
            name = rng.choices(names, weights=weights, k=1)[0]
            window = cursors[name].next_window()
            if window is None:
                continue
            buf.append(window)
            if len(buf) >= self.shuffle_buffer:
                i = rng.randrange(len(buf))
                buf[i], buf[-1] = buf[-1], buf[i]
                out = buf.pop()
                yield {"input_ids": torch.from_numpy(out.astype(np.int64))}


class _SourceCursor:
    """Walks one source's shards with a rank/worker-disjoint stride."""

    def __init__(self, src: SourceIndex, seq_len: int, stream_id: int,
                 n_streams: int, seed: int, holdout_frac: float = 0.0,
                 split: str = "train"):
        self.src = src
        self.seq_len = seq_len
        self.stream_id = stream_id
        self.n_streams = n_streams
        self.holdout_frac = holdout_frac
        self.split = split
        # crc32, not hash(): hash(str) is salted per interpreter and the shard
        # order must be stable across processes
        self.rng = random.Random(seed * 7919 + zlib.crc32(src.name.encode()) % 100003)
        self.shard_order = list(range(len(src.shards)))
        self.rng.shuffle(self.shard_order)
        self.shard_pos = 0
        self._open_shard()

    def _open_shard(self) -> None:
        sh = self.src.shards[self.shard_order[self.shard_pos % len(self.shard_order)]]
        self.arr = sh.array()
        total = len(self.arr) // self.seq_len
        # reserved tail of every shard that split="train" never reaches
        boundary = int(total * (1.0 - self.holdout_frac))
        first, self.n_windows = ((0, boundary) if self.split == "train"
                                 else (boundary, total))
        # disjoint stride across streams, offset so different epochs differ
        self.win_idx = first + self.stream_id + \
            (self.shard_pos // len(self.shard_order)) % max(self.n_streams, 1)
        self.shard_pos += 1

    def next_window(self) -> np.ndarray | None:
        for _ in range(len(self.src.shards) + 1):
            if self.win_idx < self.n_windows:
                s = self.win_idx * self.seq_len
                self.win_idx += self.n_streams
                return np.asarray(self.arr[s : s + self.seq_len])
            self._open_shard()
        return None


# --------------------------------------------------------------------------- #
def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    return {"input_ids": ids, "labels": ids}


# --------------------------------------------------------------------------- #
def build_dataloader(
    root: str | Path,
    seq_len: int,
    batch_size: int,
    chunk_product: int,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 0,
    num_workers: int = 4,
    weight_overrides: dict[str, float] | None = None,
    holdout_frac: float = 0.0,
    split: str = "train",
) -> torch.utils.data.DataLoader:
    mix = MixtureSpec.from_manifest(root, weight_overrides)
    ds = PackedTokenDataset(mix, seq_len, rank, world_size, seed, chunk_product,
                            holdout_frac=holdout_frac, split=split)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=True,
        prefetch_factor=4 if num_workers else None,
        persistent_workers=bool(num_workers),
        drop_last=True,
    )
