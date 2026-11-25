"""Helpers for indexing BreadBowl-Embed routing slots in LogosDB (stdlib + logosdb only)."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

import numpy as np

import logosdb

if TYPE_CHECKING:
    pass

SLOTS = 16
ROUTING_DIM = 256


def slot_text(passage_id: str, slot: int) -> str:
    """Compact JSON row text: passage id and slot index."""
    if slot < 0 or slot >= SLOTS:
        raise ValueError(f"slot must be in [0, {SLOTS}), got {slot}")
    return json.dumps({"passage_id": str(passage_id), "slot": int(slot)}, separators=(",", ":"))


def parse_slot_text(text: str) -> tuple[str, int]:
    """Parse row text produced by ``slot_text``."""
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("slot text must be a JSON object")
    pid = obj.get("passage_id")
    slot = obj.get("slot")
    if pid is None or slot is None:
        raise ValueError("slot text must include passage_id and slot")
    return str(pid), int(slot)


def _routing_matrix(routing: np.ndarray) -> np.ndarray:
    arr = np.asarray(routing, dtype=np.float32)
    if arr.shape != (SLOTS, ROUTING_DIM):
        raise ValueError(f"routing must have shape ({SLOTS}, {ROUTING_DIM}), got {arr.shape}")
    return arr


def put_routing(
    db: logosdb.DB,
    passage_id: str,
    routing: np.ndarray,
) -> list[int]:
    """Insert 16 routing rows for one passage (slot order 0..15)."""
    mat = _routing_matrix(routing)
    texts = [slot_text(passage_id, s) for s in range(SLOTS)]
    return db.put_batch(mat, texts=texts)


def search_routing(
    db: logosdb.DB,
    query_routing: np.ndarray,
    *,
    slot_hits: int = 512,
) -> list[list[tuple[str, float]]]:
    """Run one LogosDB search per query routing slot.

    Returns, for each query slot, a list of ``(passage_id, score)`` hits (best per passage).
    """
    qr = _routing_matrix(query_routing)
    per_slot: list[list[tuple[str, float]]] = []
    for s in range(SLOTS):
        hits = db.search(qr[s], top_k=slot_hits)
        best: dict[str, float] = {}
        for h in hits:
            pid, _slot = parse_slot_text(h.text)
            score = float(h.score)
            prev = best.get(pid)
            if prev is None or score > prev:
                best[pid] = score
        per_slot.append(sorted(best.items(), key=lambda x: -x[1]))
    return per_slot


def aggregate_slot_hits(
    slot_hit_lists: Sequence[Sequence[tuple[str, float]]],
    *,
    miss_score: float = -1.0,
) -> list[tuple[str, float]]:
    """Mean of per-query-slot best scores; missing passages count as ``miss_score``."""
    if len(slot_hit_lists) != SLOTS:
        raise ValueError(f"expected {SLOTS} slot hit lists, got {len(slot_hit_lists)}")
    passage_best: dict[str, list[float]] = defaultdict(lambda: [miss_score] * SLOTS)
    for slot_idx, hits in enumerate(slot_hit_lists):
        for pid, score in hits:
            passage_best[pid][slot_idx] = float(score)
    ranked: list[tuple[str, float]] = []
    for pid, scores in passage_best.items():
        mean = float(np.mean(scores))
        ranked.append((pid, mean))
    ranked.sort(key=lambda x: -x[1])
    return ranked


def rerank_passages(
    model: Any,
    query_embeddings: Any,
    stored: Any,
    positions: Sequence[int],
) -> Any:
    """Rerank candidate passages with ``model.score`` (lazy import of breadbowl at call time)."""
    selected = stored.select(list(positions))
    return model.score(query_embeddings, selected)