#!/usr/bin/env node
/**
 * BreadBowl-Embed routing row layout in LogosDB (Node).
 * Demo vectors are random unit norms — not v1.3 encodings. Rerank uses model.score in Python.
 */
'use strict';

const fs = require('fs');
const os = require('os');
const path = require('path');
const { DB, DIST_IP } = require('logosdb');

const SLOTS = 16;
const DIM = 256;

function slotText(passageId, slot) {
  return JSON.stringify({ passage_id: String(passageId), slot: Number(slot) });
}

function randomUnit(dim, seed) {
  const out = new Float32Array(dim);
  let s = seed;
  let sum = 0;
  for (let i = 0; i < dim; i++) {
    s = (s * 9301 + 49297) % 233280;
    const v = (s / 233280) * 2 - 1;
    out[i] = v;
    sum += v * v;
  }
  const n = Math.sqrt(sum) || 1;
  for (let i = 0; i < dim; i++) out[i] /= n;
  return out;
}

function main() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'logosdb-bb-slots-'));
  const passageId = 'demo-passage';
  try {
    const db = new DB(dir, {
      dim: DIM,
      distance: DIST_IP,
      maxElements: SLOTS,
      efSearch: 512,
      M: 16,
    });
    for (let slot = 0; slot < SLOTS; slot++) {
      const vec = randomUnit(DIM, slot + 1);
      db.put(Array.from(vec), slotText(passageId, slot));
    }
    console.log(`inserted ${SLOTS} slots for passage ${passageId} at ${dir}`);
    const q = randomUnit(DIM, 99);
    const hits = db.search(Array.from(q), 3);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
}

main();