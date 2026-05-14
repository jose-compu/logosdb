#!/usr/bin/env node
/**
 * logosdb-mcp-server — MCP server exposing LogosDB to Claude Code and other MCP clients.
 *
 * Environment variables:
 *   LOGOSDB_PATH          Base directory for all namespaces  (default: ./.logosdb)
 *   EMBEDDING_PROVIDER    Default: local Transformers.js (see below). Opt-in cloud: openai | voyage.
 *                           Local aliases: transformers | local | auto (same as default).
 *                           Local HTTP: ollama
 *   TRANSFORMERS_MODEL    HF id for Transformers.js (default: Xenova/all-MiniLM-L6-v2)
 *   TRANSFORMERS_EMBEDDING_DIM / EMBEDDING_DIM — vector size (default 384 for MiniLM)
 *   OLLAMA_HOST           Base URL when EMBEDDING_PROVIDER=ollama (default http://127.0.0.1:11434)
 *   OLLAMA_EMBED_MODEL    Ollama embedding model (default nomic-embed-text)
 *   OLLAMA_EMBEDDING_DIM  Default 768 for nomic — must match actual model output
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
import {
  clampChunkSize,
  clampTopK,
  collectFilesSafe,
  readFileBoundedUtf8,
  resolveIndexablePath,
  validateMetadata,
  validateUserText,
} from './security.js';
import { respectGitignoreDefault } from './gitignore-walker.js';
import {
  loadManifest,
  MANIFEST_VERSION,
  manifestPath,
  pruneRemovedPaths,
  saveManifest,
  type FileIndexManifest,
} from './file-index-manifest.js';
import { hybridRerank, type FusionStrategy } from './hybrid.js';
import {
  serializeTags,
  parseTags,
  matchesFilter,
  validateTags,
  validateFilter,
  type Tags,
  type FilterPredicate,
} from './metadata-filter.js';

// ── Configuration ─────────────────────────────────────────────────────────────

const LOGOSDB_PATH = process.env.LOGOSDB_PATH ?? path.join(process.cwd(), '.logosdb');
const CHUNK_SIZE = clampChunkSize(parseInt(process.env.LOGOSDB_CHUNK_SIZE ?? '800', 10), 800);

let embCfg: EmbeddingConfig;
try {
  embCfg = resolveConfig();
} catch (err) {
  process.stderr.write(`[logosdb-mcp] WARNING: ${(err as Error).message}\n`);
  // Allow server to start so list/info tools still work; embed calls will throw at call time.
  embCfg = {
    provider: 'transformers',
    dim: 384,
    model: process.env.TRANSFORMERS_MODEL ?? 'Xenova/all-MiniLM-L6-v2',
  };
}

const store = new NamespaceStore(LOGOSDB_PATH);

// ── Tool definitions ───────────────────────────────────────────────────────────

const TOOLS: Tool[] = [
  {
    name: 'logosdb_index',
    description:
      'Store a piece of text in a LogosDB namespace for later semantic retrieval. ' +
      'Use this to persist knowledge, code snippets, notes, or any textual context across sessions. ' +
      'Optional `tags` attach structured key/value metadata (strings, numbers, booleans) that can be ' +
      'filtered at search time using the `filter` parameter of logosdb_search.',
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
          description: 'Optional extra label stored alongside the text (legacy flat string)',
        },
        tags: {
          type: 'object',
          description:
            'Optional structured metadata: key/value pairs where values are strings, numbers, or booleans. ' +
            'Searchable via the `filter` predicate in logosdb_search. ' +
            'Example: {"lang":"typescript","priority":2,"reviewed":true}',
          additionalProperties: { type: ['string', 'number', 'boolean'] },
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
      'When a Git working tree is detected the walker honours `.gitignore` (root + nested + `.git/info/exclude` + global excludes); disable with `respect_gitignore=false` or `LOGOSDB_RESPECT_GITIGNORE=0`. ' +
      'Supports any UTF-8 text file (source code, markdown, plain text). ' +
      'With incremental=true, skips files unchanged since last index (mtime+size+chunk_size), re-indexes modified files after deleting their previous chunk rows, and when indexing a directory removes DB rows for files that disappeared from disk.',
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
        incremental: {
          type: 'boolean',
          description:
            'If true, only index new or changed files (vs last run in this namespace); delete old chunks for changed or removed files. Default false (full re-chunk of every file, may duplicate rows).',
        },
        respect_gitignore: {
          type: 'boolean',
          description:
            'Skip files matched by `.gitignore` rules of the enclosing Git working tree (root + nested + `.git/info/exclude` + global excludes). Default: true when `.git` is detected, else no-op. Set false to fall back to the legacy SKIP_DIRS + extension filters only. Env override: `LOGOSDB_RESPECT_GITIGNORE=0`.',
        },
        tags: {
          type: 'object',
          description:
            'Optional structured metadata applied to every chunk indexed from this path. ' +
            'Values must be strings, numbers, or booleans. ' +
            'Example: {"project":"my-app","lang":"typescript"}',
          additionalProperties: { type: ['string', 'number', 'boolean'] },
        },
      },
      required: ['path', 'namespace'],
    },
  },
  {
    name: 'logosdb_search',
    description:
      'Semantic search over a LogosDB namespace. Returns the most relevant stored texts ranked by similarity. ' +
      'Optional ISO 8601 bounds (`ts_from` / `ts_to`, inclusive) restrict candidates to a timestamp window. ' +
      'Optional `hybrid: true` blends ANN vector similarity with BM25-lite lexical matching (see `fusion`, `lexical_weight`). ' +
      'Optional `filter` applies structured metadata predicates to post-filter results (requires rows indexed with `tags`).',
    inputSchema: {
      type: 'object',
      properties: {
        query: { type: 'string', description: 'Natural-language search query' },
        namespace: { type: 'string', description: 'Collection name to search in' },
        top_k: { type: 'number', description: 'Number of results to return (default: 5)' },
        ts_from: {
          type: 'string',
          description:
            'Optional inclusive lower bound (ISO 8601). Omit with no `ts_to` for full-range search.',
        },
        ts_to: {
          type: 'string',
          description:
            'Optional inclusive upper bound (ISO 8601). Omit with no `ts_from` for full-range search.',
        },
        candidate_k: {
          type: 'number',
          description:
            'Internal candidate pool size for timestamp-windowed or filter/hybrid search (default: 10 × top_k). Increase if recall is poor.',
        },
        hybrid: {
          type: 'boolean',
          description:
            'Enable hybrid retrieval: combine ANN vector score with BM25-lite lexical score. ' +
            'Default: false (pure ANN). When true, retrieves `candidate_k` ANN candidates then re-ranks using fusion.',
        },
        fusion: {
          type: 'string',
          enum: ['rrf', 'weighted'],
          description:
            '"rrf" (default): Reciprocal Rank Fusion — rank-based, score-distribution-free. ' +
            '"weighted": linear interpolation combined = (1−lexical_weight)·ann + lexical_weight·lexical.',
        },
        lexical_weight: {
          type: 'number',
          description:
            'Weight of the lexical score in "weighted" fusion mode. 0 = pure ANN, 1 = pure lexical. Default: 0.5.',
        },
        filter: {
          type: 'object',
          description:
            'Structured metadata predicate applied after ANN (and optional hybrid re-rank). ' +
            'Only rows indexed with matching `tags` are returned. ' +
            'Supports: equality {"key":"val"}, $eq, $ne, $in, $nin, $gt, $gte, $lt, $lte, $exists. ' +
            'Multiple keys are ANDed. Example: {"lang":"typescript","priority":{"$gte":2}}',
        },
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
    description:
      'Delete one entry from a namespace. Provide **either** numeric `id` (from index/search) **or** `query` to semantically match: ' +
      'embed `query`, run similarity search, then delete the hit at `match_rank` (0-based, default 0). Use `search_top_k` to widen the pool (default 10, max 50).',
    inputSchema: {
      type: 'object',
      properties: {
        namespace: { type: 'string', description: 'Collection name' },
        id: { type: 'number', description: 'Row ID to delete (omit if using `query`)' },
        query: {
          type: 'string',
          description: 'Natural-language query to locate a row to delete (omit if using `id`)',
        },
        search_top_k: {
          type: 'number',
          description: 'When using `query`: how many neighbors to consider (default 10, max 50)',
        },
        match_rank: {
          type: 'number',
          description: 'When using `query`: which hit to delete, 0 = best match (default 0)',
        },
      },
      required: ['namespace'],
    },
  },
];

// ── Server ────────────────────────────────────────────────────────────────────

const server = new Server({ name: 'logosdb', version: '0.10.0' }, { capabilities: { tools: {} } });

server.setRequestHandler(ListToolsRequestSchema, async () => ({ tools: TOOLS }));

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args = {} } = request.params;

  try {
    switch (name) {
      // ── logosdb_index ──────────────────────────────────────────────────────
      case 'logosdb_index': {
        const rawText = String(args.text ?? '');
        if (!rawText) throw new Error('"text" is required');
        const text = validateUserText(rawText, 'text');
        const namespace = String(args.namespace ?? '');
        const metadata = validateMetadata(args.metadata ? String(args.metadata) : undefined);
        const tags: Tags | undefined =
          args.tags != null ? validateTags(args.tags) : undefined;

        if (!namespace) throw new Error('"namespace" is required');

        ensureEmbeddingsConfigured();
        const vec = await embed(text, embCfg);
        const db = store.open(namespace, vec.length);
        let stored = metadata ? `${text}\n[${metadata}]` : text;
        if (tags) stored = serializeTags(stored, tags);
        const id = db.put(vec, stored, new Date().toISOString());

        return ok({ id, indexed: true, namespace, chars: text.length, tags: tags ?? null });
      }

      // ── logosdb_index_file ─────────────────────────────────────────────────
      case 'logosdb_index_file': {
        const inputPath = String(args.path ?? '');
        const namespace = String(args.namespace ?? '');
        const chunkSize = clampChunkSize(
          typeof args.chunk_size === 'number' ? args.chunk_size : CHUNK_SIZE,
          CHUNK_SIZE,
        );
        const incremental = Boolean(args.incremental);
        const respectGitignore =
          args.respect_gitignore === undefined
            ? respectGitignoreDefault()
            : Boolean(args.respect_gitignore);
        const fileTags: Tags | undefined =
          args.tags != null ? validateTags(args.tags) : undefined;

        if (!inputPath) throw new Error('"path" is required');
        if (!namespace) throw new Error('"namespace" is required');

        ensureEmbeddingsConfigured();

        const absPath = resolveIndexablePath(inputPath);
        const stat = fs.statSync(absPath);
        const filesToIndex = stat.isDirectory()
          ? collectFilesSafe(absPath, { respectGitignore })
          : [absPath];

        if (filesToIndex.length === 0) {
          return ok({
            indexed: 0,
            files: 0,
            namespace,
            path: absPath,
            incremental,
            respect_gitignore: respectGitignore,
            skipped_files: 0,
            indexed_files: 0,
            pruned_files: 0,
          });
        }

        const manFile = manifestPath(LOGOSDB_PATH, namespace);
        const manifest: FileIndexManifest = incremental
          ? loadManifest(manFile)
          : { version: MANIFEST_VERSION, files: {} };

        const BATCH = 96;
        const ts = new Date().toISOString();
        let totalChunks = 0;
        let firstVecLen = embCfg.dim;
        let db: ReturnType<typeof store.open> | null = null;
        let skippedFiles = 0;
        let indexedFiles = 0;
        let prunedFiles = 0;

        const openDb = (dim: number) => {
          if (!db) db = store.open(namespace, dim);
          return db;
        };

        if (incremental && stat.isDirectory()) {
          openDb(embCfg.dim);
          prunedFiles = pruneRemovedPaths(absPath, new Set(filesToIndex), manifest, db!);
        }

        for (const filePath of filesToIndex) {
          let st: fs.Stats;
          try {
            st = fs.statSync(filePath);
          } catch {
            continue;
          }
          const mtimeMs = Math.trunc(st.mtimeMs);
          const fileSize = st.size;

          const prev = incremental ? manifest.files[filePath] : undefined;
          if (
            incremental &&
            prev &&
            prev.mtimeMs === mtimeMs &&
            prev.size === fileSize &&
            prev.chunkSize === chunkSize
          ) {
            skippedFiles++;
            continue;
          }

          if (incremental && prev?.ids?.length) {
            openDb(embCfg.dim);
            for (const id of prev.ids) {
              try {
                db!.delete(id);
              } catch {
                /* row may already be deleted */
              }
            }
            delete manifest.files[filePath];
          }

          let content: string;
          try {
            content = readFileBoundedUtf8(filePath);
          } catch {
            continue;
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
          openDb(firstVecLen);

          const ids: number[] = [];
          for (let i = 0; i < texts.length; i++) {
            let label = `[file:${filePath}][chunk:${i}/${texts.length}] ${texts[i]}`;
            if (fileTags) label = serializeTags(label, fileTags);
            const id = db!.put(allVecs[i]!, label, ts);
            ids.push(id);
          }
          totalChunks += texts.length;
          indexedFiles++;

          if (incremental) {
            manifest.files[filePath] = { mtimeMs, size: fileSize, chunkSize, ids };
          }
        }

        if (incremental) {
          saveManifest(manFile, manifest);
        }

        return ok({
          indexed: totalChunks,
          files: filesToIndex.length,
          namespace,
          path: absPath,
          incremental,
          respect_gitignore: respectGitignore,
          skipped_files: incremental ? skippedFiles : 0,
          indexed_files: indexedFiles,
          pruned_files: incremental ? prunedFiles : 0,
          tags: fileTags ?? null,
        });
      }

      // ── logosdb_search ─────────────────────────────────────────────────────
      case 'logosdb_search': {
        const rawQuery = String(args.query ?? '');
        if (!rawQuery) throw new Error('"query" is required');
        const query = validateUserText(rawQuery, 'query');
        const namespace = String(args.namespace ?? '');
        const topK = clampTopK(typeof args.top_k === 'number' ? args.top_k : 5);
        const tsFromRaw = args.ts_from != null ? String(args.ts_from).trim() : '';
        const tsToRaw = args.ts_to != null ? String(args.ts_to).trim() : '';
        const tsFrom = tsFromRaw || undefined;
        const tsTo = tsToRaw || undefined;
        const useTsWindow = tsFrom !== undefined || tsTo !== undefined;

        const useHybrid = Boolean(args.hybrid);
        const fusion = (args.fusion === 'weighted' ? 'weighted' : 'rrf') as FusionStrategy;
        const lexicalWeight =
          typeof args.lexical_weight === 'number'
            ? Math.min(1, Math.max(0, args.lexical_weight))
            : 0.5;

        const filter: FilterPredicate | undefined =
          args.filter != null ? validateFilter(args.filter) : undefined;

        // When hybrid or filter is active, fetch a larger candidate pool first.
        const needsPool = useHybrid || filter !== undefined;
        let candidateK =
          typeof args.candidate_k === 'number' && Number.isFinite(args.candidate_k)
            ? Math.trunc(args.candidate_k)
            : topK * 10;
        if (candidateK < topK) candidateK = topK * 10;
        // Number of ANN results to retrieve before re-ranking / filtering.
        const annK = needsPool ? candidateK : topK;
        // Internal ts-range candidate pool must be strictly larger than the ANN output count
        // so the timestamp filter has room to reject candidates.  When needsPool is active
        // annK == candidateK, so multiply by 10; otherwise use candidateK as-is.
        const tsCandidateK = needsPool ? annK * 10 : candidateK;

        if (!namespace) throw new Error('"namespace" is required');

        ensureEmbeddingsConfigured();
        const vec = await embed(query, embCfg);

        let db;
        try {
          db = store.get(namespace);
        } catch {
          const nsPath = path.join(LOGOSDB_PATH, namespace);
          if (!fs.existsSync(nsPath)) {
            return ok({ results: [], message: `Namespace "${namespace}" does not exist.` });
          }
          db = store.open(namespace, vec.length);
        }

        // 1. ANN retrieval
        let hits = useTsWindow
          ? db.searchTsRange(vec, { topK: annK, tsFrom, tsTo, candidateK: tsCandidateK })
          : db.search(vec, annK);

        // 2. Hybrid re-rank (operates on the full candidate pool, trims to topK)
        const isHybridResult = useHybrid && hits.length > 0;
        if (isHybridResult) {
          hits = hybridRerank(hits, query, filter ? annK : topK, {
            fusion,
            lexical_weight: lexicalWeight,
          });
        }

        // 3. Filter by metadata predicates (post-ANN / post-hybrid)
        if (filter !== undefined) {
          hits = hits
            .filter((h) => matchesFilter(parseTags(h.text), filter))
            .slice(0, topK);
        }

        // 4. Final trim (in case neither hybrid nor filter trimmed to topK)
        if (!isHybridResult && filter === undefined) {
          hits = hits.slice(0, topK);
        }

        const results = hits.map((h) => {
          const tags = parseTags(h.text);
          const base = {
            id: h.id,
            score: Math.round(h.score * 10000) / 10000,
            text: h.text,
            timestamp: h.timestamp,
            tags: tags ?? undefined,
          };
          if (isHybridResult && 'hybrid_score' in h) {
            const hh = h as unknown as { ann_score: number; lexical_score: number; hybrid_score: number };
            return {
              ...base,
              ann_score: hh.ann_score,
              lexical_score: hh.lexical_score,
              hybrid_score: hh.hybrid_score,
            };
          }
          return base;
        });

        return ok({
          results,
          namespace,
          query,
          ts_from: tsFrom ?? null,
          ts_to: tsTo ?? null,
          timestamp_filter: useTsWindow,
          hybrid: useHybrid,
          fusion: useHybrid ? fusion : null,
          filter: filter ?? null,
        });
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
        if (!namespace) throw new Error('"namespace" is required');

        const rawQuery = args.query != null ? String(args.query).trim() : '';
        const hasId = args.id !== undefined && args.id !== null;

        if (rawQuery && hasId) {
          throw new Error('Provide either `id` or `query`, not both');
        }
        if (!rawQuery && !hasId) {
          throw new Error('Provide either `id` (row id) or `query` (semantic match)');
        }

        if (rawQuery) {
          ensureEmbeddingsConfigured();
          const searchTopK = (() => {
            const n = typeof args.search_top_k === 'number' ? Math.trunc(args.search_top_k) : 10;
            return Math.min(50, Math.max(1, n));
          })();
          const matchRank = (() => {
            const n = typeof args.match_rank === 'number' ? Math.trunc(args.match_rank) : 0;
            return Math.max(0, n);
          })();

          const vec = await embed(rawQuery, embCfg);
          let db: ReturnType<typeof store.open>;
          try {
            db = store.get(namespace);
            if (vec.length !== db.dim()) {
              throw new Error(
                `Embedding dimension ${vec.length} does not match namespace dim ${db.dim()}`,
              );
            }
          } catch {
            const nsPath = path.join(LOGOSDB_PATH, namespace);
            if (!fs.existsSync(nsPath)) {
              throw new Error(`Namespace "${namespace}" does not exist.`);
            }
            db = store.open(namespace, vec.length);
          }

          const hits = db.search(vec, searchTopK);
          if (hits.length === 0) {
            throw new Error('No vectors matched the query; nothing to delete');
          }
          if (matchRank >= hits.length) {
            throw new Error(
              `match_rank ${matchRank} is out of range (only ${hits.length} hit(s); use 0..${hits.length - 1})`,
            );
          }
          const target = hits[matchRank]!;
          db.delete(target.id);
          return ok({
            deleted: true,
            namespace,
            id: target.id,
            mode: 'semantic',
            query: rawQuery,
            match_rank: matchRank,
            score: Math.round(target.score * 10000) / 10000,
          });
        }

        let db: ReturnType<typeof store.open>;
        try {
          db = store.get(namespace);
        } catch {
          throw new Error(
            `Namespace "${namespace}" is not open. Index or search in it first (or use \`query\` delete).`,
          );
        }
        const id = typeof args.id === 'number' ? args.id : parseInt(String(args.id), 10);
        if (isNaN(id)) throw new Error('"id" must be a number');
        db.delete(id);
        return ok({ deleted: true, namespace, id, mode: 'by_id' });
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

function ensureEmbeddingsConfigured() {
  if (embCfg.provider === 'openai' || embCfg.provider === 'voyage') {
    if (!embCfg.apiKey) {
      throw new Error(
        'No embedding API key configured. Set OPENAI_API_KEY (or VOYAGE_API_KEY with EMBEDDING_PROVIDER=voyage).',
      );
    }
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
