/**
 * Embedding provider abstraction.
 *
 * Supported providers (via env vars):
 *   EMBEDDING_PROVIDER=openai (default)  → OPENAI_API_KEY, model text-embedding-3-small, dim 1536
 *   EMBEDDING_PROVIDER=voyage            → VOYAGE_API_KEY, model voyage-3, dim 1024
 */

export interface EmbeddingConfig {
  provider: 'openai' | 'voyage';
  dim: number;
  model: string;
  apiKey: string;
}

export function resolveConfig(): EmbeddingConfig {
  const provider = (process.env.EMBEDDING_PROVIDER ?? 'openai').toLowerCase();

  if (provider === 'voyage') {
    const apiKey = process.env.VOYAGE_API_KEY ?? '';
    if (!apiKey) throw new Error('VOYAGE_API_KEY is required when EMBEDDING_PROVIDER=voyage');
    return { provider: 'voyage', dim: 1024, model: 'voyage-3', apiKey };
  }

  // default: openai
  const apiKey = process.env.OPENAI_API_KEY ?? '';
  if (!apiKey) {
    throw new Error(
      'OPENAI_API_KEY is required (or set EMBEDDING_PROVIDER=voyage with VOYAGE_API_KEY)',
    );
  }
  return { provider: 'openai', dim: 1536, model: 'text-embedding-3-small', apiKey };
}

async function fetchJson(url: string, body: unknown, headers: Record<string, string>): Promise<unknown> {
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
  if (cfg.provider === 'openai') {
    const data = (await fetchJson(
      'https://api.openai.com/v1/embeddings',
      { input: text, model: cfg.model },
      { Authorization: `Bearer ${cfg.apiKey}` },
    )) as { data: Array<{ embedding: number[] }> };
    return data.data[0].embedding;
  }

  // voyage
  const data = (await fetchJson(
    'https://api.voyageai.com/v1/embeddings',
    { input: [text], model: cfg.model },
    { Authorization: `Bearer ${cfg.apiKey}` },
  )) as { data: Array<{ embedding: number[] }> };
  return data.data[0].embedding;
}

export async function embedBatch(texts: string[], cfg: EmbeddingConfig): Promise<number[][]> {
  if (cfg.provider === 'openai') {
    const data = (await fetchJson(
      'https://api.openai.com/v1/embeddings',
      { input: texts, model: cfg.model },
      { Authorization: `Bearer ${cfg.apiKey}` },
    )) as { data: Array<{ index: number; embedding: number[] }> };
    // API returns results in the same order
    return data.data.sort((a, b) => a.index - b.index).map((d) => d.embedding);
  }

  // voyage supports batch natively
  const data = (await fetchJson(
    'https://api.voyageai.com/v1/embeddings',
    { input: texts, model: cfg.model },
    { Authorization: `Bearer ${cfg.apiKey}` },
  )) as { data: Array<{ embedding: number[] }> };
  return data.data.map((d) => d.embedding);
}
