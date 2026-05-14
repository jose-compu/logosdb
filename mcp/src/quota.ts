/**
 * Multi-tenant quota enforcement for logosdb-mcp-server.
 *
 * All checks are O(1) counter reads — zero hot-path overhead when quotas are
 * not configured (the default).
 *
 * Environment variables:
 *   LOGOSDB_QUOTA_MAX_VECTORS     Max live vectors per namespace   (0 = unlimited, default 0)
 *   LOGOSDB_QUOTA_MAX_NAMESPACES  Max namespaces under DB root     (0 = unlimited, default 0)
 */

export interface QuotaConfig {
  /** Maximum live vectors per namespace. 0 = unlimited. */
  maxVectorsPerNs: number;
  /** Maximum distinct namespaces under the DB root. 0 = unlimited. */
  maxNamespaces: number;
}

export function loadQuotaConfig(): QuotaConfig {
  return {
    maxVectorsPerNs: clampInt(process.env.LOGOSDB_QUOTA_MAX_VECTORS),
    maxNamespaces: clampInt(process.env.LOGOSDB_QUOTA_MAX_NAMESPACES),
  };
}

function clampInt(raw: string | undefined): number {
  const n = parseInt(raw ?? '0', 10);
  return isNaN(n) || n < 0 ? 0 : n;
}

/**
 * Throw if inserting `adding` vectors into `namespace` would exceed the
 * per-namespace vector limit.  No-op when the limit is 0 (unlimited).
 *
 * Pass `db.countLive()` as `currentLive` — the call is O(1).
 */
export function checkVectorQuota(
  currentLive: number,
  adding: number,
  namespace: string,
  cfg: QuotaConfig,
): void {
  if (cfg.maxVectorsPerNs <= 0) return;
  const after = currentLive + adding;
  if (after > cfg.maxVectorsPerNs) {
    throw new QuotaExceededError(
      `Namespace "${namespace}" vector quota exceeded: ` +
        `${currentLive} live + ${adding} new = ${after} > max ${cfg.maxVectorsPerNs}. ` +
        `Delete old entries or raise LOGOSDB_QUOTA_MAX_VECTORS.`,
      'vectors',
      namespace,
    );
  }
}

/**
 * Throw if creating `newNs` as a brand-new namespace would exceed the
 * namespace-count limit.  No-op when the limit is 0, or when `newNs` is
 * already present in `existingNs` (opening an existing ns is always allowed).
 */
export function checkNsQuota(
  existingNs: string[],
  newNs: string,
  cfg: QuotaConfig,
): void {
  if (cfg.maxNamespaces <= 0) return;
  if (existingNs.includes(newNs)) return; // already exists — not a new creation
  if (existingNs.length >= cfg.maxNamespaces) {
    throw new QuotaExceededError(
      `Cannot create namespace "${newNs}": namespace quota exceeded ` +
        `(${existingNs.length} existing >= max ${cfg.maxNamespaces}). ` +
        `Raise LOGOSDB_QUOTA_MAX_NAMESPACES or delete unused namespaces.`,
      'namespaces',
      newNs,
    );
  }
}

export class QuotaExceededError extends Error {
  readonly kind: 'vectors' | 'namespaces';
  readonly namespace: string;

  constructor(message: string, kind: 'vectors' | 'namespaces', namespace: string) {
    super(message);
    this.name = 'QuotaExceededError';
    this.kind = kind;
    this.namespace = namespace;
  }
}
