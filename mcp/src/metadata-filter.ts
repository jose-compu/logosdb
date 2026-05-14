/**
 * metadata-filter.ts — Structured metadata tags + predicate filtering (#84).
 *
 * Tags are serialized as a JSON suffix appended to the stored text:
 *   "<original text>\n[tags:{"key":"val","priority":3}]"
 *
 * This is backward compatible: rows without a [tags:{...}] suffix simply have
 * no tags and will not match any tag predicate (they remain visible when no
 * filter is specified).
 *
 * Supported filter predicates (mirroring MongoDB-style operators):
 *   { "key": "value" }                   — equality ($eq)
 *   { "key": { "$eq": "value" } }        — explicit equality
 *   { "key": { "$ne": "value" } }        — not-equal
 *   { "key": { "$in": ["a","b"] } }      — membership
 *   { "key": { "$nin": ["a","b"] } }     — not-in
 *   { "key": { "$gte": 1, "$lte": 5 } }  — range (numeric or string lex)
 *   { "key": { "$gt": 0 } }              — exclusive lower bound
 *   { "key": { "$lt": 10 } }             — exclusive upper bound
 *   { "key": { "$exists": true } }       — key presence check
 *
 * Multiple top-level keys are ANDed together.
 */

// ── Types ─────────────────────────────────────────────────────────────────────

export type TagValue = string | number | boolean | null;
export type Tags = Record<string, TagValue>;

type ScalarPredicate = {
  $eq?: TagValue;
  $ne?: TagValue;
  $in?: TagValue[];
  $nin?: TagValue[];
  $gt?: number | string;
  $gte?: number | string;
  $lt?: number | string;
  $lte?: number | string;
  $exists?: boolean;
};

export type FilterPredicate = Record<string, TagValue | ScalarPredicate>;

// ── Serialization ─────────────────────────────────────────────────────────────

const TAGS_PREFIX = '\n[tags:';
const TAGS_SUFFIX = ']';

/** Append tags to text for storage. Idempotent (replaces any existing suffix). */
export function serializeTags(text: string, tags: Tags): string {
  const base = stripTagsSuffix(text);
  const payload = JSON.stringify(tags);
  return `${base}${TAGS_PREFIX}${payload}${TAGS_SUFFIX}`;
}

/** Remove a [tags:{...}] suffix if present, returning the original text. */
export function stripTagsSuffix(text: string): string {
  const idx = text.lastIndexOf(TAGS_PREFIX);
  if (idx === -1) return text;
  return text.slice(0, idx);
}

/** Parse tags from stored text. Returns null if no tags suffix present. */
export function parseTags(text: string | null | undefined): Tags | null {
  if (!text) return null;
  const idx = text.lastIndexOf(TAGS_PREFIX);
  if (idx === -1) return null;
  const payload = text.slice(idx + TAGS_PREFIX.length);
  if (!payload.endsWith(TAGS_SUFFIX)) return null;
  try {
    const parsed = JSON.parse(payload.slice(0, -TAGS_SUFFIX.length));
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
      return parsed as Tags;
    }
  } catch {
    // malformed — ignore
  }
  return null;
}

// ── Validation ────────────────────────────────────────────────────────────────

const MAX_TAG_KEYS = 32;
const MAX_TAG_KEY_LEN = 64;
const MAX_TAG_VAL_LEN = 256;

export function validateTags(raw: unknown): Tags {
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) {
    throw new Error('"tags" must be a plain object');
  }
  const obj = raw as Record<string, unknown>;
  const keys = Object.keys(obj);
  if (keys.length > MAX_TAG_KEYS) {
    throw new Error(`"tags" must not exceed ${MAX_TAG_KEYS} keys`);
  }
  const result: Tags = {};
  for (const k of keys) {
    if (k.length > MAX_TAG_KEY_LEN) {
      throw new Error(`Tag key "${k.slice(0, 20)}…" exceeds ${MAX_TAG_KEY_LEN} chars`);
    }
    const v = obj[k];
    if (v === null || typeof v === 'boolean' || typeof v === 'number') {
      result[k] = v as TagValue;
    } else if (typeof v === 'string') {
      if (v.length > MAX_TAG_VAL_LEN) {
        throw new Error(`Tag value for key "${k}" exceeds ${MAX_TAG_VAL_LEN} chars`);
      }
      result[k] = v;
    } else {
      throw new Error(`Tag value for key "${k}" must be a string, number, boolean, or null`);
    }
  }
  return result;
}

export function validateFilter(raw: unknown): FilterPredicate {
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) {
    throw new Error('"filter" must be a plain object');
  }
  // Shallow structural check only — deep type errors surface at eval time.
  return raw as FilterPredicate;
}

// ── Predicate evaluation ──────────────────────────────────────────────────────

function compare(a: TagValue, b: TagValue): number {
  if (typeof a === 'number' && typeof b === 'number') return a - b;
  return String(a ?? '') < String(b ?? '') ? -1 : String(a ?? '') > String(b ?? '') ? 1 : 0;
}

function evalScalar(tagVal: TagValue | undefined, pred: ScalarPredicate): boolean {
  // $exists is the only predicate that makes sense for absent keys.
  if ('$exists' in pred) {
    return pred.$exists ? tagVal !== undefined : tagVal === undefined;
  }
  // For all other predicates an absent key never matches — consistent with MongoDB semantics.
  if (tagVal === undefined) {
    // $nin: ["a","b"] on a missing key → true (undefined is not in the list)
    if ('$nin' in pred) {
      return Array.isArray(pred.$nin) && !pred.$nin.includes(null);
    }
    return false;
  }
  if ('$eq' in pred) {
    return tagVal === pred.$eq;
  }
  if ('$ne' in pred) {
    return tagVal !== pred.$ne;
  }
  if ('$in' in pred) {
    return Array.isArray(pred.$in) && pred.$in.includes(tagVal);
  }
  if ('$nin' in pred) {
    return Array.isArray(pred.$nin) && !pred.$nin.includes(tagVal);
  }
  // Range comparisons — tagVal is defined at this point
  if (tagVal == null) return false;
  if ('$gt' in pred && pred.$gt !== undefined && compare(tagVal, pred.$gt as TagValue) <= 0)
    return false;
  if ('$gte' in pred && pred.$gte !== undefined && compare(tagVal, pred.$gte as TagValue) < 0)
    return false;
  if ('$lt' in pred && pred.$lt !== undefined && compare(tagVal, pred.$lt as TagValue) >= 0)
    return false;
  if ('$lte' in pred && pred.$lte !== undefined && compare(tagVal, pred.$lte as TagValue) > 0)
    return false;
  return true;
}

/**
 * Returns true if `tags` satisfies all predicates in `filter`.
 * Top-level keys are ANDed. Missing tags match only $exists:false and $nin.
 */
export function matchesFilter(tags: Tags | null, filter: FilterPredicate): boolean {
  const t = tags ?? {};
  for (const [key, predOrVal] of Object.entries(filter)) {
    const tagVal: TagValue | undefined = Object.prototype.hasOwnProperty.call(t, key)
      ? (t[key] as TagValue)
      : undefined;

    if (predOrVal !== null && typeof predOrVal === 'object' && !Array.isArray(predOrVal)) {
      if (!evalScalar(tagVal, predOrVal as ScalarPredicate)) return false;
    } else {
      // Plain equality shorthand: { "key": "value" }
      if (tagVal !== (predOrVal as TagValue)) return false;
    }
  }
  return true;
}
