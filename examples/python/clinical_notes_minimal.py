"""LogosDB 0.9.0 — minimal French clinical-notes retrieval (on-prem, RGPD-friendly).

Uses ALMAnaCH's ModernCamemBERT-bio-base (Fill-Mask backbone, 8192-token
context). Mean-pools the last hidden state for a sentence embedding.

    pip install ".[examples]"
    python examples/python/clinical_notes_minimal.py
"""
import tempfile

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

import logosdb

MODEL_ID = "almanach/ModernCamemBERT-bio-base"

tok = AutoTokenizer.from_pretrained(MODEL_ID)
mdl = AutoModel.from_pretrained(MODEL_ID).eval()


def embed(texts):
    enc = tok(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        hidden = mdl(**enc).last_hidden_state
    mask = enc["attention_mask"].unsqueeze(-1)
    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
    return pooled.cpu().numpy().astype("float32")


notes = [
    "Patiente 72 ans, BPCO exacerbee, dyspnee stade IV. CIM-10 : J44.1.",
    "Patient 67 ans, infarctus du myocarde aigu (STEMI). CIM-10 : I21.0.",
    "Femme 45 ans, pneumonie communautaire, antibiotherapie IV. CIM-10 : J18.9.",
    "Patiente 81 ans, insuffisance cardiaque decompensee, BNP 1840. CIM-10 : I50.9.",
]

db = logosdb.DB(tempfile.mkdtemp(), dim=mdl.config.hidden_size,
                distance=logosdb.DIST_COSINE)
db.put_batch(embed(notes), texts=notes)            # 0.9.0 bulk ingest

q = embed(["infarctus du myocarde aigu avec angioplastie"])[0]
for hit in db.search(q, top_k=3):
    print(f"{hit.score:.3f}  {hit.text}")
