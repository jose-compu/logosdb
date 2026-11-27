/**
 * LogosDB - Fast semantic vector database (HNSW + mmap)
 * TypeScript definitions
 */

/**
 * Search result hit
 */
export interface SearchHit {
  /** Row ID */
  id: number;
  /** Score depends on distance: cosine [-1,1], IP unbounded dot product, L2 in (0,1] */
  score: number;
  /** Associated text */
  text: string | null;
  /** ISO 8601 timestamp */
  timestamp: string | null;
}

/**
 * Fetched live row (get-by-id)
 */
export interface Row {
  /** Row ID */
  id: number;
  /** Float32 embedding vector (owned copy) */
  embedding: Float32Array;
  /** Associated text */
  text: string | null;
  /** ISO 8601 timestamp */
  timestamp: string | null;
}

/**
 * Database options
 */
export interface DBOptions {
  /** Vector dimension (default: 128) */
  dim?: number;
  /** Initial HNSW capacity (default: 1,000,000); doubles when full */
  maxElements?: number;
  /** Optional hard slot cap (0 = unlimited; not stored on disk) */
  maxCapacity?: number;
  /** HNSW construction parameter (default: 200) */
  efConstruction?: number;
  /** HNSW M parameter (default: 16) */
  M?: number;
  /** HNSW search parameter (default: 50) */
  efSearch?: number;
  /** Distance metric (default: DIST_IP) */
  distance?: number;
  /** On-disk storage dtype (default: DTYPE_FLOAT32) */
  dtype?: number;
}

/**
 * Timestamp range search options
 */
/** Same shape as Python `DB.stats()` and C `logosdb_stats_t`. */
export interface DbStats {
  rows_total: number;
  rows_live: number;
  tombstones: number;
  index_elements: number;
  wal_pending: number;
  distance_metric: number;
  storage_dtype: number;
  put_success: number;
  put_failed: number;
  put_batch_success: number;
  put_batch_failed: number;
  search_success: number;
  search_failed: number;
  search_ts_success: number;
  search_ts_failed: number;
  delete_success: number;
  delete_failed: number;
  update_success: number;
  update_failed: number;
  sync_calls: number;
}

export interface TsRangeOptions {
  /** Number of results (default: 10) */
  topK?: number;
  /** Start timestamp (inclusive) */
  tsFrom?: string;
  /** End timestamp (inclusive) */
  tsTo?: string;
  /** Internal candidate multiplier */
  candidateK?: number;
  skipEmptyTs?: boolean;
}

type FloatVector = number[] | Float32Array;

/**
 * LogosDB database instance
 */
export class DB {
  /**
   * Create a new LogosDB instance
   * @param path - Database directory path
   * @param options - Database options
   */
  constructor(path: string, options?: DBOptions);

  /**
   * Insert a vector with optional text and timestamp
   * @param embedding - Float vector
   * @param text - Optional text metadata
   * @param timestamp - Optional ISO 8601 timestamp
   * @returns Assigned row ID
   */
  put(embedding: FloatVector, text?: string, timestamp?: string): number;

  /**
   * Batch-insert N vectors with a single WAL fsync — ~N× faster than N individual put() calls.
   * @param embeddings - Flat row-major array of n×dim floats
   * @param n - Number of vectors
   * @param texts - Per-row text labels; null/undefined entries are stored without a label
   * @param timestamps - Per-row ISO 8601 timestamps; null/undefined entries omit the timestamp
   * @returns Array of n assigned row IDs in insertion order
   */
  putBatch(
    embeddings: FloatVector,
    n: number,
    texts?: (string | null | undefined)[],
    timestamps?: (string | null | undefined)[],
  ): number[];

  /**
   * Fetch a live row by id
   * @param id - Row ID
   * @returns Row with embedding, text, and timestamp
   */
  get(id: number): Row;

  /**
   * Search for similar vectors
   * @param queryEmbedding - Query vector
   * @param topK - Number of results (default: 10)
   * @returns Search results
   */
  search(queryEmbedding: FloatVector, topK?: number): SearchHit[];
  search(
    queryEmbedding: FloatVector,
    options: {
      k?: number;
      allow?: number[] | BigUint64Array;
      deny?: number[] | BigUint64Array;
      minScore?: number;
      tsFrom?: string;
      tsTo?: string;
      skipEmptyTs?: boolean;
    },
  ): SearchHit[];
  searchBatch(
    queries: number[][],
    k: number,
    filter?: {
      allow?: number[] | BigUint64Array;
      deny?: number[] | BigUint64Array;
      minScore?: number;
      tsFrom?: string;
      tsTo?: string;
      skipEmptyTs?: boolean;
    },
  ): { ids: (number | null)[][]; scores: number[][] };
  getMany(ids: number[]): Float32Array;

  indexCapacity(): number;

  searchAsync(queryEmbedding: FloatVector, topK?: number): Promise<SearchHit[]>;
  searchAsync(
    queryEmbedding: FloatVector,
    options: { topK?: number },
  ): Promise<SearchHit[]>;
  putBatchAsync(
    embeddings: FloatVector,
    n: number,
    texts?: (string | null | undefined)[],
    timestamps?: (string | null | undefined)[],
  ): Promise<number[]>;
  searchBatchAsync(
    queries: FloatVector[],
    k: number,
    filter?: {
      allow?: number[];
      deny?: number[];
      minScore?: number;
      tsFrom?: string;
      tsTo?: string;
      skipEmptyTs?: boolean;
    },
  ): Promise<{ ids: (number | null)[][]; scores: number[][] }>;

  /**
   * Search with timestamp range filter
   * @param queryEmbedding - Query vector
   * @param options - Search options
   * @returns Search results
   */
  searchTsRange(queryEmbedding: FloatVector, options?: TsRangeOptions): SearchHit[];

  /**
   * Update an existing row (marks old as deleted, creates new)
   * @param id - Row ID to update
   * @param embedding - New embedding
   * @param text - New text
   * @param timestamp - New timestamp
   * @returns New row ID
   */
  update(id: number, embedding: number[], text?: string, timestamp?: string): number;

  /**
   * Delete a row by ID
   * @param id - Row ID to delete
   */
  delete(id: number): void;

  /**
   * Get total row count (including deleted)
   * @returns Total count
   */
  count(): number;

  /**
   * Get live row count (excluding deleted)
   * @returns Live count
   */
  countLive(): number;

  /** Ids of live rows (tombstones omitted). */
  liveIds(): number[];

  /** Every stored vector as float32 (length count() × dim()). */
  rawVectors(): Float32Array;

  /** Stream live rows to an NDJSON file. */
  exportNdjson(
    outPath: string,
    options?: { startId?: number; endIdExclusive?: number },
  ): void;

  /** Flush WAL, index, vectors, and metadata to disk. */
  sync(): void;

  /** Operation counters and store health fields. */
  stats(): DbStats;

  /**
   * Get vector dimension
   * @returns Dimension
   */
  dim(): number;

  /**
   * Close the database
   */
  close(): void;
}

/** Inner product distance (default, requires L2-normalized vectors) */
export const DIST_IP: number;

/** Cosine similarity (auto-normalizes vectors) */
export const DIST_COSINE: number;

/** Euclidean distance (L2 space) */
export const DIST_L2: number;

/** On-disk float32 storage (default) */
export const DTYPE_FLOAT32: number;

/** On-disk float16 storage (~half size) */
export const DTYPE_FLOAT16: number;

/** On-disk int8 storage (~quarter size) */
export const DTYPE_INT8: number;

/** Library version string */
export const VERSION: string;

/** L2-normalize a vector (new array). Required for DIST_IP. */
export function normalizeL2(embedding: number[]): number[];

/** Copy live rows from src into empty dst (drops tombstones). */
export function compact(src: string, dst: string): void;
