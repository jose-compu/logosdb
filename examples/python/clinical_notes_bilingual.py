"""LogosDB 0.9.0 — biomedical retrieval over discharge notes (FR + EN).

On-premise, RGPD-friendly: no patient data leaves the box.

This example uses ALMAnaCH's brand-new ModernBERT-bio family:

    French clinical : almanach/ModernCamemBERT-bio-base   (CamemBERT-Bio reborn
                      on the ModernBERT backbone — 8192-token context, designed
                      for long discharge summaries)
    English clinical: almanach/ModernBERT-bio-base        (English sibling)

Collection: https://huggingface.co/collections/almanach/biomedical-datasets-and-models

These are Fill-Mask backbones (no sentence-transformer head shipped), so we
mean-pool the last hidden state ourselves — the standard pattern when using a
raw BERT/ModernBERT encoder for retrieval.

The example exercises three things shipped in LogosDB 0.9.0:

  * ``put_batch`` — chunked, WAL-aware bulk ingest (one ``fsync`` per chunk,
    durability matches per-row ``put``).
  * ``search_ts_range`` — ISO 8601 timestamp-window search.
  * ``export_ndjson`` / ``import_ndjson`` — streaming, resumable site-to-site
    sync (O(dim) memory per row; survives interrupted runs via
    ``checkpoint_path`` + ``resume=True``).

Run from a clone:

    pip install ".[examples]"   # pulls torch + transformers
    python examples/python/clinical_notes_bilingual.py

The first run downloads the ModernCamemBERT-bio-base weights (~0.1 B params).
Set ``LOGOSDB_EXAMPLE_SKIP_EN=1`` to skip the English sibling DB.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import List

import numpy as np

import logosdb

MODEL_FR = "almanach/ModernCamemBERT-bio-base"
MODEL_EN = "almanach/ModernBERT-bio-base"
MAX_TOKENS = 8192  # ModernBERT-class native context


def build_encoder(model_id: str):
    """Return ``(tokenizer, model, dim, encode_fn)`` for a HF Fill-Mask backbone.

    ``encode_fn(list[str]) -> np.ndarray[float32, (n, dim)]`` mean-pools the last
    hidden state over the attention mask — the canonical retrieval recipe when
    the encoder doesn't ship a sentence-transformer head.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).eval()
    dim = int(model.config.hidden_size)

    def encode(texts: List[str], batch_size: int = 16) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        chunks: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                enc = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=MAX_TOKENS,
                    return_tensors="pt",
                )
                hidden = model(**enc).last_hidden_state        # (B, T, D)
                mask = enc["attention_mask"].unsqueeze(-1)     # (B, T, 1)
                summed = (hidden * mask).sum(dim=1)            # (B, D)
                counts = mask.sum(dim=1).clamp(min=1)          # (B, 1)
                pooled = (summed / counts).cpu().numpy()
                chunks.append(pooled.astype("float32"))
        return np.vstack(chunks)

    return tokenizer, model, dim, encode


def demo_french(workdir: Path) -> None:
    print(f"\n=== French clinical notes — {MODEL_FR} ===")
    _, _, dim, encode = build_encoder(MODEL_FR)
    print(f"Encoder dim = {dim} (context = {MAX_TOKENS} tokens)")

    primary = workdir / "fr_primary"
    standby = workdir / "fr_standby"
    ndjson = workdir / "fr_notes.ndjson"
    cp = workdir / "fr_notes.cp"

    db = logosdb.DB(str(primary), dim=dim, distance=logosdb.DIST_COSINE)

    notes = [
        ("2026-04-12T11:05:00Z",
         "Patiente 72 ans, BPCO exacerbee, dyspnee stade IV, sous "
         "bronchodilatateurs et corticosteroides inhales. CIM-10 : J44.1."),
        ("2026-04-13T14:40:00Z",
         "Homme 58 ans, tachycardie ventriculaire soutenue, choc electrique "
         "externe reussi. CIM-10 : I47.2."),
        ("2026-04-13T18:20:00Z",
         "Femme 45 ans, pneumonie communautaire severe, antibiotherapie par "
         "ceftriaxone + azithromycine. CIM-10 : J18.9."),
        ("2026-04-14T07:55:00Z",
         "Patient 67 ans, infarctus du myocarde aigu (STEMI anterieur), "
         "angioplastie primaire IVA. CIM-10 : I21.0."),
        ("2026-04-14T16:10:00Z",
         "Patiente 81 ans, insuffisance cardiaque congestive decompensee, "
         "BNP 1840, diuretiques IV. CIM-10 : I50.9."),
    ]
    timestamps = [ts for ts, _ in notes]
    texts = [t for _, t in notes]

    embeddings = encode(texts, batch_size=8)

    # 0.9.0 — chunked, WAL-aware put_batch.
    ids = db.put_batch(embeddings, texts=texts, timestamps=timestamps)
    print(f"Ingested {len(ids)} FR notes; db.count() = {db.count()}")

    for query in [
        "exacerbation de bronchopneumopathie chronique obstructive",
        "infarctus du myocarde aigu avec angioplastie",
    ]:
        q = encode([query])
        print(f"\n[FR] {query}")
        for hit in db.search(q[0], top_k=3):
            print(f"  {hit.score:.3f}  [{hit.timestamp}]  {hit.text[:80]}...")

    q = encode(["syndrome coronarien aigu"])[0]
    window = db.search_ts_range(
        q,
        top_k=5,
        ts_from="2026-04-14T00:00:00Z",
        ts_to="2026-04-15T00:00:00Z",
    )
    print(f"\n[FR 24h window — 2026-04-14]  ({len(window)} hits)")
    for hit in window:
        print(f"  {hit.score:.3f}  [{hit.timestamp}]  {hit.text[:80]}...")

    # ── Site-to-site sync via streaming NDJSON ─────────────────────────────
    db.export_ndjson(str(ndjson))
    print(
        f"\nExported {db.count()} rows to {ndjson.name} "
        f"({ndjson.stat().st_size / 1024:.1f} KiB)."
    )

    restored = logosdb.DB(str(standby), dim=dim, distance=logosdb.DIST_COSINE)
    restored.import_ndjson(
        str(ndjson),
        chunk_size=2048,
        checkpoint_path=str(cp),
        resume=True,
    )
    print(f"Standby restored: {restored.count()} notes.")

    # Idempotent re-run (checkpoint already covers the file).
    restored.import_ndjson(
        str(ndjson),
        chunk_size=2048,
        checkpoint_path=str(cp),
        resume=True,
    )
    assert restored.count() == db.count(), "resume should be idempotent"
    print("Idempotent re-run confirmed.")


def demo_english(workdir: Path) -> None:
    print(f"\n=== English clinical notes — {MODEL_EN} ===")
    _, _, dim, encode = build_encoder(MODEL_EN)

    db = logosdb.DB(
        str(workdir / "en_primary"), dim=dim, distance=logosdb.DIST_COSINE
    )

    notes = [
        ("2026-04-12T08:30:00Z",
         "65 y/o male, NSTEMI on troponin elevation, started on dual "
         "antiplatelet, stable post-PCI. ICD-10: I21.4."),
        ("2026-04-13T09:15:00Z",
         "78 y/o female, CHF exacerbation, BNP 1840, started IV furosemide. "
         "ICD-10: I50.9."),
        ("2026-04-13T18:20:00Z",
         "45 y/o female, community-acquired pneumonia, started on "
         "ceftriaxone + azithromycin. ICD-10: J18.9."),
    ]
    timestamps = [ts for ts, _ in notes]
    texts = [t for _, t in notes]
    embeddings = encode(texts, batch_size=8)
    db.put_batch(embeddings, texts=texts, timestamps=timestamps)
    print(f"Ingested {db.count()} EN notes.")

    q = encode(["acute heart failure with elevated BNP"])[0]
    print("\n[EN] acute heart failure with elevated BNP")
    for hit in db.search(q, top_k=3):
        print(f"  {hit.score:.3f}  [{hit.timestamp}]  {hit.text[:80]}...")


def main() -> None:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        print("Install the examples extras first: pip install '.[examples]'")
        return

    workdir = Path(tempfile.mkdtemp(prefix="logosdb-clinical-"))
    try:
        demo_french(workdir)
        if os.environ.get("LOGOSDB_EXAMPLE_SKIP_EN") != "1":
            demo_english(workdir)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
