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
  /** Similarity score (0-1) */
  score: number;
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
  /** Maximum capacity (default: 1,000,000) */
  maxElements?: number;
  /** HNSW construction parameter (default: 200) */
  efConstruction?: number;
  /** HNSW M parameter (default: 16) */
  M?: number;
  /** HNSW search parameter (default: 50) */
  efSearch?: number;
  /** Distance metric (default: DIST_IP) */
  distance?: number;
}

/**
 * Timestamp range search options
 */
export interface TsRangeOptions {
  /** Number of results (default: 10) */
  topK?: number;
  /** Start timestamp (inclusive) */
  tsFrom?: string;
  /** End timestamp (inclusive) */
  tsTo?: string;
  /** Internal candidate multiplier */
  candidateK?: number;
}

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
  put(embedding: number[], text?: string, timestamp?: string): number;

  /**
   * Search for similar vectors
   * @param queryEmbedding - Query vector
   * @param topK - Number of results (default: 10)
   * @returns Search results
   */
  search(queryEmbedding: number[], topK?: number): SearchHit[];

  /**
   * Search with timestamp range filter
   * @param queryEmbedding - Query vector
   * @param options - Search options
   * @returns Search results
   */
  searchTsRange(queryEmbedding: number[], options?: TsRangeOptions): SearchHit[];

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

/** Library version string */
export const VERSION: string;
