/**
 * Embedding provider abstraction.
 *
 * Default (no EMBEDDING_PROVIDER / `transformers` / `local`): **Transformers.js** on-device
 * — no API keys, first run may download a small model.
 *
 * Explicit cloud (opt-in):
 *   EMBEDDING_PROVIDER=openai  → OPENAI_API_KEY, dim 1536
 *   EMBEDDING_PROVIDER=voyage  → VOYAGE_API_KEY, dim 1024
 *
 * Local HTTP:
 *   EMBEDDING_PROVIDER=ollama  → OLLAMA_HOST, OLLAMA_EMBED_MODEL (default nomic-embed-text), dim 768
 *
 * Aliases: `local` and `auto` behave like `transformers`.
 */

export type EmbeddingProvider = 'transformers' | 'ollama' | 'openai' | 'voyage';

export type EmbeddingConfig =
  | {
      provider: 'transformers';
      dim: number;
      model: string;
    }
  | {
      provider: 'ollama';
      dim: number;
      model: string;
      baseUrl: string;
    }
  | {
      provider: 'openai';
      dim: number;
      model: string;
      apiKey: string;
    }
  | {
      provider: 'voyage';
      dim: number;
      model: string;
      apiKey: string;
    };

function parseDim(envKey: string, fallback: number): number {
  const raw = process.env[envKey] ?? process.env.EMBEDDING_DIM;
  if (raw === undefined || raw === '') return fallback;
  const n = parseInt(raw, 10);
  if (!Number.isFinite(n) || n <= 0) {
    throw new Error(`${envKey} / EMBEDDING_DIM must be a positive integer`);
  }
  return n;
}

/** Split a feature-extraction Tensor into one vector per row (last dim = embedding size). */
function rowsFromTensor(t: { data: Float32Array; dims: number[] }): number[][] {
  const dims = t.dims;
  const data = t.data;
  if (dims.length === 0) return [];
  if (dims.length === 1) return [Array.from(data)];

  const last = dims[dims.length - 1]!;
  const outer = dims.slice(0, -1).reduce((a, b) => a * b, 1);
  const rows: number[][] = [];
  for (let i = 0; i < outer; i++) {
    const start = i * last;
    rows.push(Array.from(data.subarray(start, start + last)));
  }
  return rows;
}

/** Transformers.js feature-extraction callable (minimal surface for our use). */
type TransformerExtractor = (
  texts: string | string[],
  options?: { pooling?: 'mean'; normalize?: boolean },
) => Promise<{ data: Float32Array; dims: number[] }>;

let transformersPipeline: TransformerExtractor | null = null;
let transformersInit: Promise<TransformerExtractor> | null = null;

async function getTransformersExtractor(model: string): Promise<TransformerExtractor> {
  if (transformersPipeline) return transformersPipeline;
  if (!transformersInit) {
    transformersInit = (async () => {
      const { pipeline } = await import('@xenova/transformers');
      transformersPipeline = (await pipeline('feature-extraction', model)) as TransformerExtractor;
      return transformersPipeline;
    })();
  }
  return transformersInit;
}

export function resolveConfig(): EmbeddingConfig {
  const raw = (process.env.EMBEDDING_PROVIDER ?? 'transformers').toLowerCase().trim();

  if (raw === 'openai') {
    const apiKey = process.env.OPENAI_API_KEY ?? '';
    if (!apiKey) throw new Error('OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai');
    return {
      provider: 'openai',
      dim: 1536,
      model: process.env.OPENAI_EMBEDDING_MODEL ?? 'text-embedding-3-small',
      apiKey,
    };
  }

  if (raw === 'voyage') {
    const apiKey = process.env.VOYAGE_API_KEY ?? '';
    if (!apiKey) throw new Error('VOYAGE_API_KEY is required when EMBEDDING_PROVIDER=voyage');
    return {
      provider: 'voyage',
      dim: 1024,
      model: process.env.VOYAGE_EMBEDDING_MODEL ?? 'voyage-3',
      apiKey,
    };
  }

  if (raw === 'ollama') {
    const baseUrl = (process.env.OLLAMA_HOST ?? 'http://127.0.0.1:11434').replace(/\/$/, '');
    const model = process.env.OLLAMA_EMBED_MODEL ?? 'nomic-embed-text';
    const dim = parseDim('OLLAMA_EMBEDDING_DIM', 768);
    return { provider: 'ollama', dim, model, baseUrl };
  }

  // Default + aliases: transformers (local ONNX/WASM models via Transformers.js)
  if (
    raw === '' ||
    raw === 'transformers' ||
    raw === 'local' ||
    raw === 'auto' ||
    raw === 'xenova'
  ) {
    const model = process.env.TRANSFORMERS_MODEL ?? 'Xenova/all-MiniLM-L6-v2';
    const dim = parseDim('TRANSFORMERS_EMBEDDING_DIM', 384);
    return { provider: 'transformers', dim, model };
  }

  throw new Error(
    `Unknown EMBEDDING_PROVIDER "${process.env.EMBEDDING_PROVIDER}". ` +
      'Use transformers (default), ollama, openai, or voyage.',
  );
}

async function fetchJson(
  url: string,
  body: unknown,
  headers: Record<string, string>,
): Promise<unknown> {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...headers },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Embedding API error ${res.status}: ${text}`);
  }
  return res.json();
}

export async function embed(text: string, cfg: EmbeddingConfig): Promise<number[]> {
  const batch = await embedBatch([text], cfg);
  return batch[0]!;
}

export async function embedBatch(texts: string[], cfg: EmbeddingConfig): Promise<number[][]> {
  if (cfg.provider === 'transformers') {
    const extractor = await getTransformersExtractor(cfg.model);
    const tensor = await extractor(texts.length === 1 ? texts[0]! : texts, {
      pooling: 'mean',
      normalize: true,
    });
    const rows = rowsFromTensor(tensor);
    if (rows.length !== texts.length) {
      throw new Error(
        `Transformers embedding batch size mismatch (got ${rows.length}, expected ${texts.length})`,
      );
    }
    const dim = cfg.dim;
    for (let i = 0; i < rows.length; i++) {
      if (rows[i]!.length !== dim) {
        throw new Error(
          `Embedding dimension mismatch: model produced ${rows[i]!.length} but EMBEDDING_DIM / TRANSFORMERS_EMBEDDING_DIM is ${dim}`,
        );
      }
    }
    return rows;
  }

  if (cfg.provider === 'ollama') {
    const url = `${cfg.baseUrl}/api/embed`;
    const data = (await fetchJson(
      url,
      { model: cfg.model, input: texts.length === 1 ? texts[0]! : texts },
      {},
    )) as { embeddings?: number[][] };
    const embeddings = data.embeddings;
    if (!embeddings || embeddings.length !== texts.length) {
      throw new Error('Ollama /api/embed returned an unexpected embeddings array');
    }
    const dim = cfg.dim;
    for (const vec of embeddings) {
      if (vec.length !== dim) {
        throw new Error(
          `Ollama embedding length ${vec.length} does not match configured dim ${dim} — set OLLAMA_EMBEDDING_DIM or EMBEDDING_DIM`,
        );
      }
    }
    return embeddings;
  }

  if (cfg.provider === 'openai') {
    const data = (await fetchJson(
      'https://api.openai.com/v1/embeddings',
      { input: texts, model: cfg.model },
      { Authorization: `Bearer ${cfg.apiKey}` },
    )) as { data: Array<{ index: number; embedding: number[] }> };
    return data.data.sort((a, b) => a.index - b.index).map((d) => d.embedding);
  }

  const data = (await fetchJson(
    'https://api.voyageai.com/v1/embeddings',
    { input: texts, model: cfg.model },
    { Authorization: `Bearer ${cfg.apiKey}` },
  )) as { data: Array<{ embedding: number[] }> };
  return data.data.map((d) => d.embedding);
}
