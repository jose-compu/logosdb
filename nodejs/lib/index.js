/**
 * LogosDB - Fast semantic vector database (HNSW + mmap)
 * Node.js bindings
 */

'use strict';

const bindings = require('bindings')('logosdb');

/**
 * Search result hit
 * @typedef {Object} SearchHit
 * @property {number} id - Row ID
 * @property {number} score - Similarity score (0-1)
 * @property {string|null} text - Associated text
 * @property {string|null} timestamp - ISO 8601 timestamp
 */

/**
 * LogosDB database instance
 *
 * @example
 * const { DB } = require('logosdb');
 *
 * const db = new DB('/tmp/mydb', { dim: 128 });
 *
 * // Insert with embedding
 * const id = db.put(embedding, 'My text', '2025-01-01T00:00:00Z');
 *
 * // Search
 * const hits = db.search(queryEmbedding, 5);
 * console.log(hits[0].text, hits[0].score);
 */
class DB {
  /**
   * Create a new LogosDB instance
   * @param {string} path - Database directory path
   * @param {Object} [options] - Database options
   * @param {number} [options.dim=128] - Vector dimension
   * @param {number} [options.maxElements=1000000] - Maximum capacity
   * @param {number} [options.efConstruction=200] - HNSW construction parameter
   * @param {number} [options.M=16] - HNSW M parameter
   * @param {number} [options.efSearch=50] - HNSW search parameter
   * @param {number} [options.distance=DIST_IP] - Distance metric (DIST_IP, DIST_COSINE, DIST_L2)
   */
  constructor(path, options = {}) {
    this._db = new bindings.DB(path, options);
    this._closed = false;
  }

  /**
   * Insert a vector with optional text and timestamp
   * @param {number[]} embedding - Float vector
   * @param {string} [text] - Optional text metadata
   * @param {string} [timestamp] - Optional ISO 8601 timestamp
   * @returns {number} Assigned row ID
   */
  put(embedding, text, timestamp) {
    this._checkClosed();
    return this._db.put(embedding, text, timestamp);
  }

  /**
   * Search for similar vectors
   * @param {number[]} queryEmbedding - Query vector
   * @param {number} [topK=10] - Number of results
   * @returns {SearchHit[]} Search results
   */
  search(queryEmbedding, topK = 10) {
    this._checkClosed();
    return this._db.search(queryEmbedding, topK);
  }

  /**
   * Search with timestamp range filter
   * @param {number[]} queryEmbedding - Query vector
   * @param {Object} options - Search options
   * @param {number} [options.topK=10] - Number of results
   * @param {string} [options.tsFrom] - Start timestamp (inclusive)
   * @param {string} [options.tsTo] - End timestamp (inclusive)
   * @param {number} [options.candidateK] - Internal candidate multiplier
   * @returns {SearchHit[]} Search results
   */
  searchTsRange(queryEmbedding, options = {}) {
    this._checkClosed();
    return this._db.searchTsRange(queryEmbedding, options);
  }

  /**
   * Update an existing row (marks old as deleted, creates new)
   * @param {number} id - Row ID to update
   * @param {number[]} embedding - New embedding
   * @param {string} [text] - New text
   * @param {string} [timestamp] - New timestamp
   * @returns {number} New row ID
   */
  update(id, embedding, text, timestamp) {
    this._checkClosed();
    return this._db.update(id, embedding, text, timestamp);
  }

  /**
   * Delete a row by ID
   * @param {number} id - Row ID to delete
   */
  delete(id) {
    this._checkClosed();
    this._db.delete(id);
  }

  /**
   * Get total row count (including deleted)
   * @returns {number}
   */
  count() {
    this._checkClosed();
    return this._db.count();
  }

  /**
   * Get live row count (excluding deleted)
   * @returns {number}
   */
  countLive() {
    this._checkClosed();
    return this._db.countLive();
  }

  /**
   * Get vector dimension
   * @returns {number}
   */
  dim() {
    return this._db.dim();
  }

  /**
   * Close the database
   */
  close() {
    if (!this._closed) {
      this._db.close();
      this._closed = true;
    }
  }

  _checkClosed() {
    if (this._closed) {
      throw new Error('Database is closed');
    }
  }
}

/**
 * Inner product distance (default, requires L2-normalized vectors)
 * @constant {number}
 */
const DIST_IP = bindings.DIST_IP;

/**
 * Cosine similarity (auto-normalizes vectors)
 * @constant {number}
 */
const DIST_COSINE = bindings.DIST_COSINE;

/**
 * Euclidean distance (L2 space)
 * @constant {number}
 */
const DIST_L2 = bindings.DIST_L2;

/**
 * Library version string
 * @constant {string}
 */
const VERSION = bindings.VERSION;

module.exports = {
  DB,
  DIST_IP,
  DIST_COSINE,
  DIST_L2,
  VERSION,
};
