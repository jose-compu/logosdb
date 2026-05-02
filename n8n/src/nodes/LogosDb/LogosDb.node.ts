import type {
  IExecuteFunctions,
  INodeExecutionData,
  INodeType,
  INodeTypeDescription,
} from 'n8n-workflow';
import { NodeOperationError } from 'n8n-workflow';

// Native addon — CommonJS only, no ESM alternative
// eslint-disable-next-line @typescript-eslint/no-var-requires, @typescript-eslint/no-require-imports
const { DB, DIST_COSINE } = require('logosdb') as {
  DB: new (p: string, opts?: { dim?: number; maxElements?: number; distance?: number }) => LogosDB;
  DIST_COSINE: number;
};

interface SearchHit {
  id: number;
  score: number;
  text: string | null;
  timestamp: string | null;
}

interface LogosDB {
  put(embedding: number[], text?: string, timestamp?: string): number;
  search(queryEmbedding: number[], topK?: number): SearchHit[];
  delete(id: number): void;
  count(): number;
  countLive(): number;
  dim(): number;
  close(): void;
}

// In-process registry: dbPath → DB instance
const _registry = new Map<string, LogosDB>();

function openDb(dbPath: string, dim: number): LogosDB {
  const key = dbPath;
  if (_registry.has(key)) return _registry.get(key)!;
  const db = new DB(dbPath, { dim, distance: DIST_COSINE });
  _registry.set(key, db);
  return db;
}

async function embed(texts: string[], provider: string, apiKey: string): Promise<number[][]> {
  if (provider === 'openai') {
    const res = await fetch('https://api.openai.com/v1/embeddings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${apiKey}` },
      body: JSON.stringify({ input: texts, model: 'text-embedding-3-small' }),
    });
    if (!res.ok) throw new Error(`OpenAI embed error ${res.status}`);
    const data = (await res.json()) as { data: Array<{ index: number; embedding: number[] }> };
    return data.data.sort((a, b) => a.index - b.index).map((d) => d.embedding);
  }

  if (provider === 'voyage') {
    const res = await fetch('https://api.voyageai.com/v1/embeddings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${apiKey}` },
      body: JSON.stringify({ input: texts, model: 'voyage-3' }),
    });
    if (!res.ok) throw new Error(`Voyage embed error ${res.status}`);
    const data = (await res.json()) as { data: Array<{ embedding: number[] }> };
    return data.data.map((d) => d.embedding);
  }

  throw new Error(`Unknown embedding provider: ${provider}`);
}

export class LogosDb implements INodeType {
  description: INodeTypeDescription = {
    displayName: 'LogosDB',
    name: 'logosDb',
    icon: 'file:logosdb.svg',
    group: ['transform'],
    version: 1,
    subtitle: '={{$parameter["operation"]}}',
    description: 'Semantic vector database — store and search embeddings locally',
    defaults: { name: 'LogosDB' },
    inputs: ['main'],
    outputs: ['main'],
    credentials: [{ name: 'logosDbApi', required: true }],
    properties: [
      {
        displayName: 'Operation',
        name: 'operation',
        type: 'options',
        noDataExpression: true,
        options: [
          {
            name: 'Insert',
            value: 'insert',
            description: 'Embed text and insert into the database',
            action: 'Insert a text into the database',
          },
          {
            name: 'Search',
            value: 'search',
            description: 'Search for semantically similar texts',
            action: 'Search for similar texts',
          },
          {
            name: 'Delete',
            value: 'delete',
            description: 'Delete an entry by row ID',
            action: 'Delete an entry by row ID',
          },
          {
            name: 'Info',
            value: 'info',
            description: 'Get database statistics',
            action: 'Get database statistics',
          },
        ],
        default: 'search',
      },

      // ── Insert ───────────────────────────────────────────────────────────
      {
        displayName: 'Text',
        name: 'text',
        type: 'string',
        typeOptions: { rows: 3 },
        default: '',
        required: true,
        displayOptions: { show: { operation: ['insert'] } },
        description: 'Text to embed and store',
      },
      {
        displayName: 'Timestamp',
        name: 'timestamp',
        type: 'string',
        default: '',
        displayOptions: { show: { operation: ['insert'] } },
        description: 'Optional ISO 8601 timestamp',
      },
      {
        displayName: 'Vector Dimension',
        name: 'dim',
        type: 'number',
        default: 1536,
        displayOptions: { show: { operation: ['insert'] } },
        description:
          'Embedding dimension — must match the model (1536 for OpenAI, 1024 for Voyage)',
      },

      // ── Search ───────────────────────────────────────────────────────────
      {
        displayName: 'Query',
        name: 'query',
        type: 'string',
        default: '',
        required: true,
        displayOptions: { show: { operation: ['search'] } },
        description: 'Natural-language search query',
      },
      {
        displayName: 'Top K',
        name: 'topK',
        type: 'number',
        default: 5,
        displayOptions: { show: { operation: ['search'] } },
      },
      {
        displayName: 'Vector Dimension',
        name: 'dim',
        type: 'number',
        default: 1536,
        displayOptions: { show: { operation: ['search'] } },
      },

      // ── Delete ───────────────────────────────────────────────────────────
      {
        displayName: 'Row ID',
        name: 'rowId',
        type: 'number',
        default: 0,
        required: true,
        displayOptions: { show: { operation: ['delete'] } },
      },
      {
        displayName: 'Vector Dimension',
        name: 'dim',
        type: 'number',
        default: 1536,
        displayOptions: { show: { operation: ['delete'] } },
      },
    ],
  };

  async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
    const items = this.getInputData();
    const results: INodeExecutionData[] = [];

    const creds = await this.getCredentials('logosDbApi');
    const dbPath = String(creds.dbPath);
    const embeddingProvider = String(creds.embeddingProvider);
    const apiKey =
      embeddingProvider === 'openai'
        ? String(creds.openAiApiKey)
        : String(creds.voyageApiKey ?? '');

    const operation = this.getNodeParameter('operation', 0) as string;

    for (let i = 0; i < items.length; i++) {
      try {
        if (operation === 'insert') {
          const text = this.getNodeParameter('text', i) as string;
          const timestamp = this.getNodeParameter('timestamp', i) as string;
          const dim = this.getNodeParameter('dim', i) as number;

          if (!text)
            throw new NodeOperationError(this.getNode(), '"text" is required', { itemIndex: i });
          if (embeddingProvider === 'none') {
            throw new NodeOperationError(
              this.getNode(),
              'Embedding provider is set to "none". Provide a vector manually via the "Insert Vector" operation.',
              { itemIndex: i },
            );
          }

          const [vec] = await embed([text], embeddingProvider, apiKey);
          const db = openDb(dbPath, vec.length || dim);
          const id = db.put(vec, text, timestamp || new Date().toISOString());
          results.push({ json: { id, text, dim: vec.length } });
        } else if (operation === 'search') {
          const query = this.getNodeParameter('query', i) as string;
          const topK = this.getNodeParameter('topK', i) as number;
          const dim = this.getNodeParameter('dim', i) as number;

          if (!query)
            throw new NodeOperationError(this.getNode(), '"query" is required', { itemIndex: i });

          const [vec] = await embed([query], embeddingProvider, apiKey);
          const db = openDb(dbPath, vec.length || dim);
          const hits = db.search(vec, topK);
          results.push({
            json: {
              results: hits.map((h) => ({
                id: h.id,
                score: Math.round(h.score * 10000) / 10000,
                text: h.text,
                timestamp: h.timestamp,
              })),
              query,
              count: hits.length,
            },
          });
        } else if (operation === 'delete') {
          const rowId = this.getNodeParameter('rowId', i) as number;
          const dim = this.getNodeParameter('dim', i) as number;
          const db = openDb(dbPath, dim);
          db.delete(rowId);
          results.push({ json: { deleted: true, id: rowId } });
        } else if (operation === 'info') {
          // Info works without embedding — open with placeholder dim
          const db = openDb(dbPath, 1536);
          results.push({
            json: {
              count: db.count(),
              countLive: db.countLive(),
              dim: db.dim(),
              path: dbPath,
            },
          });
        }
      } catch (err) {
        if (this.continueOnFail()) {
          results.push({ json: { error: (err as Error).message }, pairedItem: i });
        } else {
          throw err;
        }
      }
    }

    return [results];
  }
}
