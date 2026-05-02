#!/usr/bin/env node
/**
 * logosdb-mcp-server — MCP server exposing LogosDB to Claude Code and other MCP clients.
 *
 * Environment variables:
 *   LOGOSDB_PATH          Base directory for all namespaces  (default: ./.logosdb)
 *   EMBEDDING_PROVIDER    "openai" (default) | "voyage"
 *   OPENAI_API_KEY        Required when EMBEDDING_PROVIDER=openai
 *   VOYAGE_API_KEY        Required when EMBEDDING_PROVIDER=voyage
 *   LOGOSDB_CHUNK_SIZE    Target characters per chunk for file indexing (default: 800)
 */

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  type Tool,
} from '@modelcontextprotocol/sdk/types.js';
import * as fs from 'fs';
import * as path from 'path';

import { resolveConfig, embed, embedBatch, type EmbeddingConfig } from './embeddings.js';
import { NamespaceStore } from './store.js';
import { chunk } from './chunker.js';

// ── Configuration ─────────────────────────────────────────────────────────────

const LOGOSDB_PATH = process.env.LOGOSDB_PATH ?? path.join(process.cwd(), '.logosdb');
const CHUNK_SIZE = parseInt(process.env.LOGOSDB_CHUNK_SIZE ?? '800', 10);

// File extensions treated as indexable text
const INDEXABLE_EXTENSIONS = new Set([
  '.ts',
  '.tsx',
  '.js',
  '.jsx',
  '.mjs',
  '.cjs',
  '.py',
  '.go',
  '.rs',
  '.java',
  '.c',
  '.cpp',
  '.h',
  '.hpp',
  '.cs',
  '.rb',
  '.php',
  '.swift',
  '.kt',
  '.scala',
  '.sh',
  '.bash',
  '.zsh',
  '.fish',
  '.md',
  '.rst',
  '.txt',
  '.toml',
  '.yaml',
  '.yml',
  '.json',
  '.sql',
  '.graphql',
  '.proto',
  '.env.example',
  '.cfg',
  '.ini',
]);

// Directories to skip when walking
const SKIP_DIRS = new Set([
  'node_modules',
  '.git',
  '.venv',
  '__pycache__',
  '.next',
  'dist',
  'build',
  'out',
  'coverage',
  '.turbo',
]);

function collectFiles(root: string): string[] {
  const results: string[] = [];
  function walk(dir: string) {
    let entries: fs.Dirent[];
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true });
    } catch {
      return;
    }
    for (const entry of entries) {
      if (entry.name.startsWith('.') && entry.isDirectory()) continue;
      if (SKIP_DIRS.has(entry.name)) continue;
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        walk(full);
      } else if (entry.isFile() && INDEXABLE_EXTENSIONS.has(path.extname(entry.name))) {
        results.push(full);
      }
    }
  }
  walk(root);
  return results;
}

let embCfg: EmbeddingConfig;
try {
  embCfg = resolveConfig();
} catch (err) {
  process.stderr.write(`[logosdb-mcp] WARNING: ${(err as Error).message}\n`);
  // Allow server to start so list/info tools still work; embed calls will throw at call time.
  embCfg = { provider: 'openai', dim: 1536, model: 'text-embedding-3-small', apiKey: '' };
}

const store = new NamespaceStore(LOGOSDB_PATH);

// ── Tool definitions ───────────────────────────────────────────────────────────

const TOOLS: Tool[] = [
  {
    name: 'logosdb_index',
    description:
      'Store a piece of text in a LogosDB namespace for later semantic retrieval. ' +
      'Use this to persist knowledge, code snippets, notes, or any textual context across sessions.',
    inputSchema: {
      type: 'object',
      properties: {
        text: { type: 'string', description: 'Text to index' },
        namespace: {
          type: 'string',
          description: 'Collection name (e.g. "code", "notes", "docs")',
        },
        metadata: {
          type: 'string',
          description: 'Optional extra label stored alongside the text',
        },
      },
      required: ['text', 'namespace'],
    },
  },
  {
    name: 'logosdb_index_file',
    description:
      'Read a file OR directory from disk, chunk each file, embed the chunks, and store them in a namespace. ' +
      'When given a directory it walks recursively, skipping hidden paths and node_modules. ' +
      'Supports any UTF-8 text file (source code, markdown, plain text).',
    inputSchema: {
      type: 'object',
      properties: {
        path: {
          type: 'string',
          description: 'Absolute or relative path to a file or directory',
        },
        namespace: { type: 'string', description: 'Collection name' },
        chunk_size: {
          type: 'number',
          description: `Target characters per chunk (default: ${CHUNK_SIZE})`,
        },
      },
      required: ['path', 'namespace'],
    },
  },
  {
    name: 'logosdb_search',
    description:
      'Semantic search over a LogosDB namespace. Returns the most relevant stored texts ranked by similarity.',
    inputSchema: {
      type: 'object',
      properties: {
        query: { type: 'string', description: 'Natural-language search query' },
        namespace: { type: 'string', description: 'Collection name to search in' },
        top_k: { type: 'number', description: 'Number of results to return (default: 5)' },
      },
      required: ['query', 'namespace'],
    },
  },
  {
    name: 'logosdb_list',
    description: 'List all available namespaces in the LogosDB store.',
    inputSchema: { type: 'object', properties: {}, required: [] },
  },
  {
    name: 'logosdb_info',
    description:
      'Return statistics for a namespace: total entries, live entries, vector dimension.',
    inputSchema: {
      type: 'object',
      properties: {
        namespace: { type: 'string', description: 'Collection name' },
      },
      required: ['namespace'],
    },
  },
  {
    name: 'logosdb_delete',
    description: 'Delete a single entry from a namespace by its row ID (returned by index/search).',
    inputSchema: {
      type: 'object',
      properties: {
        namespace: { type: 'string', description: 'Collection name' },
        id: { type: 'number', description: 'Row ID to delete' },
      },
      required: ['namespace', 'id'],
    },
  },
];

// ── Server ────────────────────────────────────────────────────────────────────

const server = new Server({ name: 'logosdb', version: '0.7.2' }, { capabilities: { tools: {} } });

server.setRequestHandler(ListToolsRequestSchema, async () => ({ tools: TOOLS }));

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args = {} } = request.params;

  try {
    switch (name) {
      // ── logosdb_index ──────────────────────────────────────────────────────
      case 'logosdb_index': {
        const text = String(args.text ?? '');
        const namespace = String(args.namespace ?? '');
        const metadata = args.metadata ? String(args.metadata) : undefined;

        if (!text) throw new Error('"text" is required');
        if (!namespace) throw new Error('"namespace" is required');

        ensureApiKey();
        const vec = await embed(text, embCfg);
        const db = store.open(namespace, vec.length);
        const stored = metadata ? `${text}\n[${metadata}]` : text;
        const id = db.put(vec, stored, new Date().toISOString());

        return ok({ id, indexed: true, namespace, chars: text.length });
      }

      // ── logosdb_index_file ─────────────────────────────────────────────────
      case 'logosdb_index_file': {
        const inputPath = String(args.path ?? '');
        const namespace = String(args.namespace ?? '');
        const chunkSize = typeof args.chunk_size === 'number' ? args.chunk_size : CHUNK_SIZE;

        if (!inputPath) throw new Error('"path" is required');
        if (!namespace) throw new Error('"namespace" is required');

        ensureApiKey();

        const absPath = path.resolve(inputPath);
        const stat = fs.statSync(absPath);
        const filesToIndex = stat.isDirectory() ? collectFiles(absPath) : [absPath];

        if (filesToIndex.length === 0) {
          return ok({ indexed: 0, files: 0, namespace, path: absPath });
        }

        const BATCH = 96;
        const ts = new Date().toISOString();
        let totalChunks = 0;
        let firstVecLen = embCfg.dim;
        let db: ReturnType<typeof store.open> | null = null;

        for (const filePath of filesToIndex) {
          let content: string;
          try {
            content = fs.readFileSync(filePath, 'utf-8');
          } catch {
            continue; // skip unreadable files
          }

          const fileChunks = chunk(content, chunkSize);
          const texts = fileChunks.map((c) => c.text);

          const allVecs: number[][] = [];
          for (let i = 0; i < texts.length; i += BATCH) {
            const batch = texts.slice(i, i + BATCH);
            const vecs = await embedBatch(batch, embCfg);
            allVecs.push(...vecs);
          }

          if (allVecs.length === 0) continue;
          firstVecLen = allVecs[0].length;
          if (!db) db = store.open(namespace, firstVecLen);

          for (let i = 0; i < texts.length; i++) {
            const label = `[file:${filePath}][chunk:${i}/${texts.length}] ${texts[i]}`;
            db.put(allVecs[i], label, ts);
          }
          totalChunks += texts.length;
        }

        return ok({ indexed: totalChunks, files: filesToIndex.length, namespace, path: absPath });
      }

      // ── logosdb_search ─────────────────────────────────────────────────────
      case 'logosdb_search': {
        const query = String(args.query ?? '');
        const namespace = String(args.namespace ?? '');
        const topK = typeof args.top_k === 'number' ? args.top_k : 5;

        if (!query) throw new Error('"query" is required');
        if (!namespace) throw new Error('"namespace" is required');

        ensureApiKey();
        const vec = await embed(query, embCfg);

        let db;
        try {
          db = store.get(namespace);
        } catch {
          // Namespace not opened yet — try to open from disk
          const nsPath = path.join(LOGOSDB_PATH, namespace);
          if (!fs.existsSync(nsPath)) {
            return ok({ results: [], message: `Namespace "${namespace}" does not exist.` });
          }
          db = store.open(namespace, vec.length);
        }

        const hits = db.search(vec, topK);
        const results = hits.map((h) => ({
          id: h.id,
          score: Math.round(h.score * 10000) / 10000,
          text: h.text,
          timestamp: h.timestamp,
        }));

        return ok({ results, namespace, query });
      }

      // ── logosdb_list ───────────────────────────────────────────────────────
      case 'logosdb_list': {
        const namespaces = store.list();
        return ok({ namespaces, count: namespaces.length, path: LOGOSDB_PATH });
      }

      // ── logosdb_info ───────────────────────────────────────────────────────
      case 'logosdb_info': {
        const namespace = String(args.namespace ?? '');
        if (!namespace) throw new Error('"namespace" is required');

        const nsPath = path.join(LOGOSDB_PATH, namespace);
        if (!fs.existsSync(nsPath)) {
          throw new Error(`Namespace "${namespace}" does not exist.`);
        }

        // Open with placeholder dim; LogosDB restores dim from its header
        const db = store.open(namespace, embCfg.dim);
        return ok({
          namespace,
          count: db.count(),
          countLive: db.countLive(),
          dim: db.dim(),
          path: nsPath,
        });
      }

      // ── logosdb_delete ─────────────────────────────────────────────────────
      case 'logosdb_delete': {
        const namespace = String(args.namespace ?? '');
        const id = typeof args.id === 'number' ? args.id : parseInt(String(args.id), 10);

        if (!namespace) throw new Error('"namespace" is required');
        if (isNaN(id)) throw new Error('"id" must be a number');

        const db = store.get(namespace);
        db.delete(id);
        return ok({ deleted: true, namespace, id });
      }

      default:
        throw new Error(`Unknown tool: ${name}`);
    }
  } catch (err) {
    return {
      content: [{ type: 'text', text: `Error: ${(err as Error).message}` }],
      isError: true,
    };
  }
});

// ── Helpers ───────────────────────────────────────────────────────────────────

function ok(data: unknown) {
  return { content: [{ type: 'text', text: JSON.stringify(data, null, 2) }] };
}

function ensureApiKey() {
  if (!embCfg.apiKey) {
    throw new Error(
      'No embedding API key configured. Set OPENAI_API_KEY (or VOYAGE_API_KEY with EMBEDDING_PROVIDER=voyage).',
    );
  }
}

// ── Main ──────────────────────────────────────────────────────────────────────

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  process.stderr.write(
    `[logosdb-mcp] Server started. Store: ${LOGOSDB_PATH}, provider: ${embCfg.provider}\n`,
  );
}

process.on('SIGINT', () => {
  store.closeAll();
  process.exit(0);
});
process.on('SIGTERM', () => {
  store.closeAll();
  process.exit(0);
});

main().catch((err) => {
  process.stderr.write(`[logosdb-mcp] Fatal: ${err.message}\n`);
  process.exit(1);
});
