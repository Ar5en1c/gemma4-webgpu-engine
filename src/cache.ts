// Written from docs/ENGINE-PLAN.md section 3 (the download, the cache, and the transport
// specification added by the research corrections), the round 2 IndexedDB probe recorded in
// ./deviceProfile.ts, the safetensors range plan in ./safetensors.ts, the Fetch and IndexedDB
// specifications, and the Storage Standard. No vendored bundle, no extracted kernel and no third
// party engine source was read.
//
// SPDX-License-Identifier: Apache-2.0
//
// The weight loader: the signed URL, the ranged GETs, the pool that issues them, the IndexedDB
// store they land in, and the resume that makes a dropped connection cost one run instead of two
// gigabytes.
//
// On the store name: ENGINE-PLAN section 3 prefers reusing the incumbent's `safetensors-cache-v1`
// store so an existing user does not re-download 2 GB on the day the engines swap, IF its record
// shape can be read without inspecting upstream code. It cannot: the shape is private to the
// vendored bundle, the bundle is the one text nobody authoring this engine reads, and the plan
// says in that case to take a new store name and one re-download and say so in the release note.
// So this is `purrview-gemma4-weights-v1`, the exact fallback name the plan reserves, and round 1
// ratified the one re-download. The release note obligation travels with the import swap, not with
// this file.
//
// Five rules from the transport specification are implemented here rather than described, and each
// one is cheap now and structural later:
//
//   1. Single-range GETs only. A multi-range Range header is not CORS safelisted, so it triggers a
//      preflight the signed CDN URL fails. See `rangeRequestHeaders`.
//   2. One resolve per file. The resolve URL 302s to a signed CDN URL whose Expires is 3600
//      seconds out; resolve once, reuse the signed URL for every ranged GET, refresh past 50
//      minutes. See `SignedSource`.
//   3. A bounded pool, with a byte budget on top of the request count so the two oversized ranges
//      cannot be in flight together. See `fetchRuns`.
//   4. Every stored artifact sliced at the profile's `idbMaxValueBytes`, because a completed write
//      transaction is not proof the bytes are retrievable on this stack.
//   5. Per run resume keyed by (repo, revision, run index), with the run list recomputed from the
//      header every load and never hardcoded. A run larger than one request is fetched, stored and
//      resumed in 64 MiB pieces keyed by (repo, revision, run index, piece), so the two oversized
//      single tensor runs cost a piece rather than a table when a connection drops.

import {
  DEFAULT_RANGE_PLAN_OPTIONS,
  HEADER_PREFIX_BYTES,
  buildRangePlan,
  checkTieGuard,
  parseHeader,
  rangeHeaderValue,
  readHeaderLength,
  resolveEntries,
  selectTextTensors,
  sha256Hex,
  sliceTensorFromRange,
  splitRangeIntoPieces,
  tieGuardRange,
  type PlannedRange,
  type RangePiece,
  type RangePlan,
  type SafetensorsDirectory,
  type TensorEntry,
  type TextTensorSelection,
  type TieGuardResult,
} from './safetensors';

/**
 * The global `fetch`, bound to the global object.
 *
 * `const doFetch = options.fetchImpl ?? fetch` is the natural line and it throws in a browser:
 * calling the global fetch through a binding whose `this` is undefined gives "TypeError: Failed to
 * execute 'fetch' on 'Window': Illegal invocation". It passes in Node, where fetch does not care,
 * which is exactly the shape of bug that reaches a user. One helper, used everywhere a default
 * fetch is needed, so there is one place to get it right.
 */
export function defaultFetch(): typeof fetch {
  const global = globalThis as unknown as { fetch?: typeof fetch };
  return typeof global.fetch === 'function' ? global.fetch.bind(globalThis) : fetch;
}

export const WEIGHT_CACHE_DB = 'purrview-gemma4-weights-v1';
const TENSOR_STORE = 'tensors';
const META_STORE = 'meta';

/**
 * Bumped when the record shape changes. A version mismatch drops the store rather than trying to
 * migrate, because the cache is a re-downloadable artefact, not data.
 *
 * Version 2 is the chunked record shape. Version 1 stored one record per tensor, which put a
 * 1,174,405,120 byte value in a single IndexedDB key; the round 2 probe recorded in
 * deviceProfile.ts found that a value that size writes successfully and then fails its own read
 * once the origin's accounted usage catches its quota, so the chunk ceiling is now a profile
 * number and the store shape carries it.
 *
 * Version 3 is version 2 plus a content checksum per tensor in the run marker. The tensor records
 * themselves did not move, but a marker written by version 2 has no checksums, and a run whose
 * checksums are missing is a run this loader will not serve, so a version 2 store would sit there
 * holding two gigabytes nothing can read. Dropping it on open is the honest outcome and it is the
 * same one re-download either way.
 */
export const WEIGHT_CACHE_SCHEMA = 3;

/**
 * The chunk ceiling used when no profile is supplied. `deviceProfile.ts` measured 64 MiB as the
 * largest single value that never failed a read in any state its probe reached on the M1, and the
 * generic profile halves it. A caller that has a resolved profile passes
 * `profile.idbMaxValueBytes.value` and this default never applies.
 */
export const DEFAULT_IDB_CHUNK_BYTES = 64 * 1024 * 1024;

/** The cache key of one tensor. Repo and revision scope it; a revision bump is a cold cache. */
export function tensorKey(repo: string, revision: string, name: string): string {
  return `${repo}@${revision}#${name}`;
}

/**
 * The cache key of one piece of one tensor, for a run the loader fetches in several requests.
 *
 * Deliberately not the whole tensor's key with a piece suffix appended after the name: this key is
 * a different KIND of record, holding part of a tensor rather than all of it, and a reader that
 * asked for the tensor and got a piece would see the right prefix and the wrong length. The suffix
 * sits after the tensor key so `clearScope`'s prefix range still sweeps it, which is the one thing
 * a new key shape here can silently break.
 */
export function tensorPieceKey(repo: string, revision: string, name: string, piece: number): string {
  return `${tensorKey(repo, revision, name)}|p${piece}`;
}

/**
 * Bytes per request when a range is too big to be one, and therefore the granularity a dropped
 * connection costs. 64 MiB, which is the same number `DEFAULT_RANGE_PLAN_OPTIONS.maxRangeBytes`
 * caps a coalesced range at, so every request this loader issues is at most 64 MiB whether its run
 * is one tensor or forty. It is a multiple of four, which is what lets `BufferManager.uploadWeight`
 * write a piece straight into a buffer at its own offset.
 *
 * The two runs this changes are the 1,174,405,120 byte PLE table, which becomes 18 requests, and
 * the 100,663,296 byte embed table, which becomes 2. Every other run in this checkpoint's plan is
 * already under the cap and is one request exactly as before.
 */
export const RANGE_PIECE_BYTES = 64 * 1024 * 1024;

/** The FNV-1a prime, the multiply in the checksum step below. */
const CHECKSUM_PRIME = 0x01000193;

/**
 * A content checksum over a tensor's bytes, as sixteen lower case hexadecimal digits.
 *
 * This is what stands between a cache entry that rotted in place and the GPU. It is deliberately
 * not SHA-256: `sha256Hex` in ./safetensors.ts is forty lines of the FIPS compression function
 * written for a 256 KiB guard sample, and running it over the whole checkpoint on every warm load
 * would cost minutes. This runs at 2.4 GiB/s measured on this M1 under node 22, so the whole
 * checkpoint costs about 0.83 seconds on a warm load that is paying no network at all.
 *
 * What it is for and what it is not for, said plainly. It detects a cache entry whose bytes changed
 * after they were written: a flipped bit, a truncated then repadded value, a chunk that came back
 * from the wrong key, storage that rotted under an evicting browser. It is not a signature and it
 * proves nothing against someone who can write to IndexedDB and recompute the checksum with it; the
 * thing that says the bytes are the checkpoint's is the tie guard's SHA-256 against the pinned
 * digest, and the thing that says the file did not move under us is the ETag.
 *
 * Four interleaved FNV-1a lanes, sixteen bytes an iteration, folded with the length at the end.
 * Every piece of that is load bearing:
 *
 *   - FNV-1a's step, xor then multiply by an odd constant, is a bijection on the accumulator. So
 *     for any fixed run of following bytes, one changed word always changes that lane's final
 *     accumulator: a single word difference can never cancel itself out further down the tensor.
 *   - The same step is order sensitive, so two words that swapped places move it too, and a
 *     truncation moves it because a shorter tensor is a shorter chain.
 *   - Four lanes rather than one because the accumulator is a serial dependency: one lane measures
 *     0.6 GiB/s on this machine and four measure 2.4, for the same arithmetic per byte.
 *
 * Two things in here are belt and braces rather than properties, and they are labelled as such
 * because this round's rule is that a claim needs a check that can break it. The length is folded
 * in at the end and no check can tell: seeding the fold with zero instead leaves all 180 transport
 * assertions passing, because any change of length has already moved the accumulators. It stays at
 * one operation per tensor. A running sum beside each lane got the opposite verdict, because it was
 * not free: it was written on the theory that it was what made position matter, FNV turns out to be
 * order sensitive on its own, freezing one of the four sums could not be detected by any check
 * including the lane by lane swap written to catch exactly that, and it cost a quarter of the
 * throughput. Dead weight that is not free comes out.
 *
 * The words are assembled from bytes rather than read through a `Uint32Array` view, which buys the
 * thing worth having: one code path. A view needs the array to be four byte aligned, and the
 * loader's two callers are not alike (the read path hands over a freshly allocated array, the write
 * path a subarray into a range body at whatever offset the tensor sits), so a view would mean two
 * loops that have to agree byte for byte or every warm load silently refetches. Assembling little
 * endian by hand also makes the answer independent of the host's endianness rather than merely
 * correct on the hosts we happen to run on.
 */
export function tensorChecksum(bytes: Uint8Array): string {
  let m0 = 0x811c9dc5;
  let m1 = 0x9e3779b9;
  let m2 = 0x85ebca6b;
  let m3 = 0xc2b2ae35;
  const whole = bytes.byteLength & ~15;
  for (let b = 0; b < whole; b += 16) {
    m0 = Math.imul(m0 ^ ((bytes[b]! | (bytes[b + 1]! << 8) | (bytes[b + 2]! << 16) | (bytes[b + 3]! << 24)) >>> 0), CHECKSUM_PRIME) >>> 0;
    m1 = Math.imul(m1 ^ ((bytes[b + 4]! | (bytes[b + 5]! << 8) | (bytes[b + 6]! << 16) | (bytes[b + 7]! << 24)) >>> 0), CHECKSUM_PRIME) >>> 0;
    m2 = Math.imul(m2 ^ ((bytes[b + 8]! | (bytes[b + 9]! << 8) | (bytes[b + 10]! << 16) | (bytes[b + 11]! << 24)) >>> 0), CHECKSUM_PRIME) >>> 0;
    m3 = Math.imul(m3 ^ ((bytes[b + 12]! | (bytes[b + 13]! << 8) | (bytes[b + 14]! << 16) | (bytes[b + 15]! << 24)) >>> 0), CHECKSUM_PRIME) >>> 0;
  }
  // The last fifteen bytes or fewer, one byte at a time through lane zero. A tail is not a word and
  // pretending it is one would make a tensor's checksum depend on what happened to follow it.
  for (let b = whole; b < bytes.byteLength; b += 1) {
    m0 = Math.imul(m0 ^ bytes[b]!, CHECKSUM_PRIME) >>> 0;
  }
  let hi = bytes.byteLength >>> 0;
  let lo = 0x9e3779b9;
  for (const lane of [m0, m1, m2, m3]) {
    hi = Math.imul(hi ^ lane, CHECKSUM_PRIME) >>> 0;
    lo = Math.imul(((lo + lane) >>> 0) ^ (hi >>> 16), 0x85ebca6b) >>> 0;
  }
  return hi.toString(16).padStart(8, '0') + lo.toString(16).padStart(8, '0');
}

/** The meta record key that pins a scope to the commit the CDN reported. */
export function scopeKey(repo: string, revision: string): string {
  return `${repo}@${revision}`;
}

/**
 * The resume key of one run: repo, revision and the run's index in the plan.
 *
 * The index is positional and the plan is recomputed from the header on every load, so a change to
 * the allow list or to the coalescing gap renumbers every run and invalidates every marker. That
 * is the correct behaviour and not a limitation: a marker means "run 7 of the plan this header
 * produces", and if the plan moved, run 7 is a different span of bytes.
 *
 * `piece` names one request inside a run that takes several, and is left off for a run that is one
 * request. A run whose piece count changes therefore renumbers its own markers, on the same
 * argument: the piece size is part of what a marker means.
 */
export function runKey(repo: string, revision: string, index: number, piece?: number): string {
  const base = `run:${scopeKey(repo, revision)}:${index}`;
  return piece === undefined ? base : `${base}/${piece}`;
}

/** One chunk of one tensor. The key carries the chunk index so a read can walk them in order. */
interface ChunkRecord {
  key: string;
  index: number;
  chunks: number;
  totalBytes: number;
  bytes: ArrayBuffer;
  byteLength: number;
  storedAt: number;
}

/** What a completed run marker records, so a resumed load can report what it skipped. */
export interface RunMarker {
  /** The commit the CDN reported when this run landed. */
  commit: string | null;
  /** Bytes the run's tensors occupy, not counting swallowed gap. */
  bytes: number;
  tensors: number;
  at: number;
  /**
   * `tensorChecksum` of every tensor this run stored, keyed by tensor name, taken from the bytes
   * that came off the wire before they were written.
   *
   * The checksums live in the marker rather than beside the bytes on purpose. The marker is the
   * thing the loader already trusts to say a run landed, it is small enough to read on every load,
   * and putting the checksum in a different record from the bytes it describes means the two have
   * to rot together to rot silently. It also keeps the check in the loader rather than in one
   * store implementation, so a `WeightStore` over a Map gets the same guarantee `WeightCache` does.
   */
  digests: Record<string, string>;
}

function chunkKey(key: string, index: number): string {
  return `${key}~${index}`;
}

/**
 * How many chunk records a tensor of this many bytes needs.
 *
 * A free function because it is the whole of the chunking arithmetic and a Node check has no
 * IndexedDB to open a cache against. A zero length tensor still gets one record, so its presence
 * is representable rather than indistinguishable from absence.
 */
export function chunkCountForBytes(byteLength: number, chunkBytes: number): number {
  return Math.max(1, Math.ceil(Math.max(0, byteLength) / Math.max(1, chunkBytes)));
}

/**
 * What the loader needs of a store, as an interface rather than as the class.
 *
 * `WeightCache` is the only implementation that ships and it is the one the engine uses. The
 * interface exists so `scripts/engine-check/transport.mjs` can drive the whole load path over a
 * Map, which is the only way the resume behaviour gets a test at all: proving it against real
 * IndexedDB would mean proving it in a browser, and the browser proof is a receipt rather than a
 * gate that runs on every commit.
 */
export interface WeightStore {
  readonly chunkBytes: number;
  getTensor(key: string): Promise<Uint8Array | null>;
  putTensor(key: string, bytes: Uint8Array): Promise<boolean>;
  /** `piece` addresses one request of a run the loader splits; omitted means the whole run. */
  getRun(repo: string, revision: string, index: number, piece?: number): Promise<RunMarker | null>;
  markRun(repo: string, revision: string, index: number, marker: RunMarker, piece?: number): Promise<void>;
  clearRun(repo: string, revision: string, index: number, piece?: number): Promise<void>;
  ensureCommit(repo: string, revision: string, commit: string): Promise<boolean>;
  clearScope(repo: string, revision: string): Promise<void>;
}

function requestToPromise<T>(request: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error('IndexedDB request failed'));
  });
}

/**
 * Best effort IndexedDB weight cache. Every method degrades to a miss rather than throwing,
 * because a browser with storage disabled or evicted still has to load the model; it just pays
 * the network again. open() resolves null when IndexedDB is unavailable at all.
 *
 * The binding rule this class exists to obey, from the round 2 probe: **a failed read is a cache
 * miss to refetch, never a corruption to report.** The probe found large values whose write
 * transaction completed and whose matching read then failed with "UnknownError: Failed to read
 * large IndexedDB value", with the failing size walking downward through the session. So nothing
 * here treats a successful write as proof the bytes are retrievable, and every read path returns
 * null on any failure at all.
 *
 * That rule covers a read that fails. It says nothing about a read that succeeds and hands back the
 * wrong bytes, and this class cannot tell the difference: a chunk record whose length still matches
 * looks exactly like a good one from here. The content check is one level up, in `fetchRuns`, which
 * is the only place that knows what the bytes were when they came off the wire. See `RunMarker`.
 */
export class WeightCache implements WeightStore {
  private readonly db: IDBDatabase;

  /** The largest value this cache will write. Bigger tensors are split across keys. */
  readonly chunkBytes: number;

  private constructor(db: IDBDatabase, chunkBytes: number) {
    this.db = db;
    this.chunkBytes = Math.max(1, Math.trunc(chunkBytes));
  }

  /**
   * `dbName` exists because `Gemma4LoadOptions.cacheName` is part of the API contract in
   * gemma4-engine.d.ts and a caller is entitled to name its own store. It defaults to this
   * engine's own store, so a caller that omits it sees no change. `chunkBytes` comes from the
   * resolved device profile's `idbMaxValueBytes`, which is a measurement rather than a constant.
   */
  static async open(
    dbName: string = WEIGHT_CACHE_DB,
    chunkBytes: number = DEFAULT_IDB_CHUNK_BYTES,
  ): Promise<WeightCache | null> {
    if (typeof indexedDB === 'undefined') return null;
    try {
      const request = indexedDB.open(dbName, WEIGHT_CACHE_SCHEMA);
      request.onupgradeneeded = () => {
        const db = request.result;
        // A schema bump lands here with the old stores present. Drop and recreate: the cache is
        // a re-downloadable artefact and migration would be complexity with no payoff.
        for (const name of Array.from(db.objectStoreNames)) db.deleteObjectStore(name);
        db.createObjectStore(TENSOR_STORE, { keyPath: 'key' });
        db.createObjectStore(META_STORE);
      };
      const db = await requestToPromise(request as IDBRequest<IDBDatabase>);
      return new WeightCache(db, chunkBytes);
    } catch {
      return null;
    }
  }

  /**
   * How many chunks a tensor of this many bytes needs. A zero length tensor still gets one record,
   * so its presence is representable.
   */
  chunkCountFor(byteLength: number): number {
    return chunkCountForBytes(byteLength, this.chunkBytes);
  }

  /**
   * Read one tensor back, walking its chunks in order. Returns null on any failure, on a missing
   * chunk, on a chunk count that disagrees with chunk zero, or on a total that does not add up.
   */
  async getTensor(key: string): Promise<Uint8Array | null> {
    try {
      const tx = this.db.transaction(TENSOR_STORE, 'readonly');
      const store = tx.objectStore(TENSOR_STORE);
      const first = (await requestToPromise(store.get(chunkKey(key, 0)))) as ChunkRecord | undefined;
      if (!first || !(first.bytes instanceof ArrayBuffer)) return null;
      if (first.bytes.byteLength !== first.byteLength) return null;
      if (first.chunks === 1) {
        return first.byteLength === first.totalBytes ? new Uint8Array(first.bytes) : null;
      }
      const out = new Uint8Array(first.totalBytes);
      out.set(new Uint8Array(first.bytes), 0);
      let cursor = first.byteLength;
      for (let i = 1; i < first.chunks; i += 1) {
        const record = (await requestToPromise(store.get(chunkKey(key, i)))) as ChunkRecord | undefined;
        if (!record || !(record.bytes instanceof ArrayBuffer)) return null;
        if (record.bytes.byteLength !== record.byteLength) return null;
        if (cursor + record.byteLength > out.length) return null;
        out.set(new Uint8Array(record.bytes), cursor);
        cursor += record.byteLength;
      }
      return cursor === first.totalBytes ? out : null;
    } catch {
      return null;
    }
  }

  /**
   * Store one tensor's bytes, sliced at `chunkBytes`.
   *
   * Each slice is copied into a standalone ArrayBuffer first, because a subarray view over a whole
   * range body would otherwise persist the entire range. One transaction per tensor, so a tensor
   * is either wholly stored or is a miss on the next read: a half written tensor whose chunk zero
   * survives would read as a length mismatch and return null, which is the same outcome.
   */
  async putTensor(key: string, bytes: Uint8Array): Promise<boolean> {
    try {
      const chunks = this.chunkCountFor(bytes.byteLength);
      const tx = this.db.transaction(TENSOR_STORE, 'readwrite');
      const store = tx.objectStore(TENSOR_STORE);
      for (let i = 0; i < chunks; i += 1) {
        const start = i * this.chunkBytes;
        const end = Math.min(bytes.byteLength, start + this.chunkBytes);
        const slice = bytes.subarray(start, end);
        const record: ChunkRecord = {
          key: chunkKey(key, i),
          index: i,
          chunks,
          totalBytes: bytes.byteLength,
          bytes: slice.slice().buffer,
          byteLength: slice.byteLength,
          storedAt: Date.now(),
        };
        await requestToPromise(store.put(record));
      }
      return true;
    } catch {
      return false;
    }
  }

  async getMeta(key: string): Promise<string | null> {
    try {
      const tx = this.db.transaction(META_STORE, 'readonly');
      const value = await requestToPromise(tx.objectStore(META_STORE).get(key));
      return typeof value === 'string' ? value : null;
    } catch {
      return null;
    }
  }

  async putMeta(key: string, value: string): Promise<void> {
    try {
      const tx = this.db.transaction(META_STORE, 'readwrite');
      await requestToPromise(tx.objectStore(META_STORE).put(value, key));
    } catch {
      // Best effort.
    }
  }

  async deleteMeta(key: string): Promise<void> {
    try {
      const tx = this.db.transaction(META_STORE, 'readwrite');
      await requestToPromise(tx.objectStore(META_STORE).delete(key));
    } catch {
      // Best effort.
    }
  }

  /** The marker for a completed run or piece, or null when it has not landed under this plan. */
  async getRun(repo: string, revision: string, index: number, piece?: number): Promise<RunMarker | null> {
    const raw = await this.getMeta(runKey(repo, revision, index, piece));
    if (raw === null) return null;
    try {
      const parsed = JSON.parse(raw) as RunMarker;
      return typeof parsed?.bytes === 'number' ? parsed : null;
    } catch {
      return null;
    }
  }

  /** Record that a run's tensors are all stored. Written after the last putTensor of the run. */
  async markRun(repo: string, revision: string, index: number, marker: RunMarker, piece?: number): Promise<void> {
    await this.putMeta(runKey(repo, revision, index, piece), JSON.stringify(marker));
  }

  /** Withdraw a run marker, which is what a failed read of one of its tensors means. */
  async clearRun(repo: string, revision: string, index: number, piece?: number): Promise<void> {
    await this.deleteMeta(runKey(repo, revision, index, piece));
  }

  /**
   * Pin a scope to the commit the CDN reported through X-Repo-Commit (ENGINE-PLAN section 3 says
   * to capture it as a cache key alongside the revision). A `main` revision that moved to a new
   * commit invalidates the whole scope, which returns true so the caller re-partitions.
   */
  async ensureCommit(repo: string, revision: string, commit: string): Promise<boolean> {
    const key = scopeKey(repo, revision);
    const stored = await this.getMeta(key);
    if (stored === commit) return false;
    if (stored !== null) await this.clearScope(repo, revision);
    await this.putMeta(key, commit);
    return stored !== null;
  }

  /**
   * Delete every tensor and every run marker of one repo and revision. Used when a floating
   * revision moved and when a caller passes `force`.
   */
  async clearScope(repo: string, revision: string): Promise<void> {
    try {
      const prefix = `${scopeKey(repo, revision)}#`;
      const tx = this.db.transaction(TENSOR_STORE, 'readwrite');
      const store = tx.objectStore(TENSOR_STORE);
      const range = IDBKeyRange.bound(prefix, `${prefix}￿`);
      const keys = await requestToPromise(store.getAllKeys(range));
      for (const key of keys) await requestToPromise(store.delete(key));
    } catch {
      // Best effort.
    }
    try {
      const prefix = `run:${scopeKey(repo, revision)}:`;
      const tx = this.db.transaction(META_STORE, 'readwrite');
      const store = tx.objectStore(META_STORE);
      const range = IDBKeyRange.bound(prefix, `${prefix}￿`);
      const keys = await requestToPromise(store.getAllKeys(range));
      for (const key of keys) await requestToPromise(store.delete(key));
    } catch {
      // Best effort.
    }
  }

  close(): void {
    try {
      this.db.close();
    } catch {
      // Already closed.
    }
  }
}

// ------------------------------------------------------------------------ storage readiness

/**
 * How much room the loader wants beyond the download itself before it will start.
 *
 * ENGINE-PLAN's research corrections ask for a refusal when quota minus usage is under about
 * 2.5 GB for a 2.01 GB download, so the headroom is about half a gigabyte on top of the bytes.
 * It is a fixed number rather than a fraction because the thing it guards against is fixed: the
 * store needs room for the chunk records plus IndexedDB's own bookkeeping, and this project's own
 * machine is a 16 GB M1 whose SSD is regularly near full.
 */
export const STORAGE_HEADROOM_BYTES = 500 * 1000 * 1000;

export interface StorageReadiness {
  /**
   * What `navigator.storage.persist()` answered. Null when the API is not there at all, which is
   * a different fact from a false and is reported as one.
   */
  persisted: boolean | null;
  /** True when persist() was actually called on this load. It is called on every load, not once. */
  persistCalled: boolean;
  quotaBytes: number | null;
  usageBytes: number | null;
  /** quota minus usage, or null when estimate() is unavailable. */
  freeBytes: number | null;
  /** Bytes the caller said it was about to fetch. */
  needBytes: number;
  /** False only when estimate() answered and the answer is too small. An absent estimate is not a refusal. */
  sufficient: boolean;
  /** Prose naming the numbers, for a log line or a thrown message. */
  message: string;
}

interface StorageManagerLike {
  persist?(): Promise<boolean>;
  estimate?(): Promise<{ quota?: number; usage?: number }>;
}

/**
 * Ask for persistent storage and check there is room, before the first byte of the download.
 *
 * Both halves are best effort and both are reported rather than assumed. `persist()` is called on
 * every load and not once, because the answer can change: a user who installs the app to the Home
 * Screen or grants a notification permission can turn a false into a true, and a returning user is
 * entitled to be told honestly whether their two gigabytes survived. WebKit deletes all script
 * writable storage for an origin with no user interaction in seven days of browser use, so this
 * boolean is the difference between a cache and a week long lease.
 */
export async function prepareStorage(
  needBytes: number,
  options: { storage?: StorageManagerLike | null; headroomBytes?: number } = {},
): Promise<StorageReadiness> {
  const headroom = options.headroomBytes ?? STORAGE_HEADROOM_BYTES;
  const storage = options.storage !== undefined
    ? options.storage
    : (typeof navigator !== 'undefined' ? (navigator.storage as StorageManagerLike | undefined) ?? null : null);

  let persisted: boolean | null = null;
  let persistCalled = false;
  if (storage && typeof storage.persist === 'function') {
    persistCalled = true;
    try {
      persisted = await storage.persist();
    } catch {
      persisted = null;
    }
  }

  let quotaBytes: number | null = null;
  let usageBytes: number | null = null;
  if (storage && typeof storage.estimate === 'function') {
    try {
      const estimate = await storage.estimate();
      quotaBytes = typeof estimate.quota === 'number' ? estimate.quota : null;
      usageBytes = typeof estimate.usage === 'number' ? estimate.usage : null;
    } catch {
      quotaBytes = null;
      usageBytes = null;
    }
  }

  const freeBytes = quotaBytes !== null && usageBytes !== null ? quotaBytes - usageBytes : null;
  const want = needBytes + headroom;
  const sufficient = freeBytes === null ? true : freeBytes >= want;

  const persistWord = persisted === null
    ? (persistCalled ? 'persist() was called and gave no answer' : 'persist() is not available here')
    : persisted ? 'storage is persistent' : 'storage is best effort and the browser may evict it';
  const roomWord = freeBytes === null
    ? 'estimate() is not available, so free space was not checked'
    : `${freeBytes} bytes free of ${quotaBytes} quota against ${needBytes} needed plus ${headroom} headroom`;

  return {
    persisted,
    persistCalled,
    quotaBytes,
    usageBytes,
    freeBytes,
    needBytes,
    sufficient,
    message: `${persistWord}; ${roomWord}`,
  };
}

// --------------------------------------------------------------------------- the signed source

/**
 * How long a resolved CDN URL is reused before it is resolved again.
 *
 * The 302 from the resolve route points at a signed URL whose `Expires` is exactly 3600 seconds
 * out. Fifty minutes leaves ten minutes of margin, which a load that is still running on a capped
 * edge will use: 2.01 GB at the 8.7 MB/s a reported majority of CloudFront edges cap a single
 * stream to is 3.9 minutes with a pool and considerably longer without one.
 */
export const SIGNED_URL_MAX_AGE_MS = 50 * 60 * 1000;

export interface SourceHeaders {
  /** X-Repo-Commit, the cache key alongside the revision. */
  commit: string | null;
  /** A strong ETag, which on this host equals the content hash. */
  etag: string | null;
  /** X-Xet-Hash, the content fingerprint the plan asks for so a resume can prove the file is the same. */
  xetHash: string | null;
  /** X-Linked-Size, the file's real length behind the pointer. */
  linkedSize: number | null;
  /** Accept-Ranges, which has to say bytes for any of this to work. */
  acceptRanges: string | null;
}

function readSourceHeaders(response: Response): SourceHeaders {
  const linked = response.headers.get('x-linked-size');
  const parsed = linked === null ? Number.NaN : Number(linked);
  return {
    commit: response.headers.get('x-repo-commit'),
    etag: response.headers.get('etag'),
    xetHash: response.headers.get('x-xet-hash'),
    linkedSize: Number.isFinite(parsed) ? parsed : null,
    acceptRanges: response.headers.get('accept-ranges'),
  };
}

/**
 * Build the headers for one ranged GET.
 *
 * SINGLE RANGE ONLY, and this is a correctness rule rather than a preference. A `Range` value with
 * a comma in it, such as `bytes=0-131071,500000-600000`, is not a CORS safelisted request header,
 * so it triggers a preflight OPTIONS that the signed CDN URL answers with a failure. Batching
 * several tensor spans into one multi-range request is the natural optimisation an engineer
 * reaches for next, and it passes in Node and fails in the browser. The native Xet download
 * protocol mandates exactly that shape, which is why our path works and theirs does not from a
 * page. The assertion below is here so a future batching attempt fails loudly in Node too.
 */
export function rangeRequestHeaders(value: string): Record<string, string> {
  if (value.includes(',')) {
    throw new Error(
      `gemma4 loader: multi-range Range value ${value} is not CORS safelisted, so it preflights `
      + 'and the signed CDN URL fails the preflight. Issue one range per request.',
    );
  }
  return { Range: value };
}

/**
 * One resolve per file, then every ranged GET against the signed URL it 302s to.
 *
 * This removes about 85 redirect round trips of roughly 150 ms each from a cold load and keeps the
 * loader far from the resolver's quota of 3000 per 300 seconds, which is what matters for a user
 * behind a shared NAT or a corporate proxy. `resolveCount` is on the receipt so a load that
 * quietly resolved fifty times is visible rather than merely slow.
 */
export interface SignedSourceOptions {
  /** How long a resolved URL is reused. Defaults to `SIGNED_URL_MAX_AGE_MS`. */
  readonly maxAgeMs?: number;
  /**
   * The clock the age is measured against. Injected rather than read from `Date.now` inside so a
   * check can drive the refresh without waiting fifty minutes, which is the only way that branch
   * gets tested at all.
   */
  readonly clock?: () => number;
}

export class SignedSource {
  readonly resolveUrl: string;
  private readonly doFetch: typeof fetch;
  private readonly maxAgeMs: number;
  private readonly clock: () => number;
  private signed: string | null = null;
  private resolvedAt = 0;
  private headers: SourceHeaders | null = null;
  private resolves = 0;

  constructor(resolveUrl: string, doFetch: typeof fetch, options: SignedSourceOptions = {}) {
    this.resolveUrl = resolveUrl;
    this.doFetch = doFetch;
    this.maxAgeMs = options.maxAgeMs ?? SIGNED_URL_MAX_AGE_MS;
    this.clock = options.clock ?? (() => Date.now());
  }

  get resolveCount(): number {
    return this.resolves;
  }

  get lastHeaders(): SourceHeaders | null {
    return this.headers;
  }

  /** The URL currently in use, which is the signed one once a resolve has happened. */
  get url(): string {
    return this.signed ?? this.resolveUrl;
  }

  private stale(now: number): boolean {
    return this.signed === null || now - this.resolvedAt >= this.maxAgeMs;
  }

  /**
   * Resolve if needed and hand back the URL to fetch against.
   *
   * The resolve is a one byte ranged GET rather than a HEAD, because a HEAD against the resolve
   * route is not what the CDN is asked for in the steady state and a one byte range is the
   * cheapest request that exercises the exact path every later request takes. The response's own
   * `url` is the signed URL after the redirect, which is what a browser hands back once it has
   * followed the 302.
   */
  async ensure(signal?: AbortSignal): Promise<string> {
    const now = this.clock();
    if (!this.stale(now)) return this.signed as string;
    const response = await this.doFetch(this.resolveUrl, {
      signal,
      headers: rangeRequestHeaders('bytes=0-0'),
    });
    if (response.status !== 206 && response.status !== 200) {
      throw new Error(`gemma4 loader: resolving ${this.resolveUrl} returned ${response.status}`);
    }
    // Drain the one byte so the connection is reusable rather than left half read.
    await response.arrayBuffer();
    this.resolves += 1;
    this.resolvedAt = now;
    this.headers = readSourceHeaders(response);
    // `response.url` is the post redirect URL in a browser. A fetch implementation that does not
    // set it, which includes some test doubles, leaves us on the resolve route: correct, one
    // redirect per request slower, and visible because `signedUrl` on the receipt equals the
    // resolve URL.
    this.signed = response.url && response.url.length > 0 ? response.url : this.resolveUrl;
    return this.signed;
  }

  /** One ranged GET against the signed URL, refreshing it first when it is old. */
  async fetchRange(range: PlannedRange, signal?: AbortSignal): Promise<{ body: Uint8Array; headers: SourceHeaders }> {
    const url = await this.ensure(signal);
    const response = await this.doFetch(url, {
      signal,
      headers: rangeRequestHeaders(rangeHeaderValue(range)),
    });
    // A 200 here means the server ignored the Range header and is streaming 2.46 GB at us.
    // Refuse it: partial content is the contract the plan measured (ENGINE-PLAN section 3).
    if (response.status !== 206) {
      throw new Error(`gemma4 loader: ranged fetch returned ${response.status}, expected 206 partial content`);
    }
    const body = new Uint8Array(await response.arrayBuffer());
    if (body.byteLength !== range.byteLength) {
      throw new Error(
        `gemma4 loader: range ${range.index} returned ${body.byteLength} bytes, expected ${range.byteLength}`,
      );
    }
    return { body, headers: readSourceHeaders(response) };
  }
}

// ---------------------------------------------------------------------------------- the pool

/**
 * The pool size the loader uses when a caller names none.
 *
 * Measured on this machine on 2026-09-01: 12.3 MB/s on one stream against 24.1 MB/s aggregate over
 * eight. The pool is insurance rather than tuning, because a reported majority of CloudFront edges
 * cap a plain single stream at exactly 8.7 MB/s and we do not control which edge a user lands on.
 * Six sits in the middle of the specification's 6 to 8 band and inside the six connections per host
 * a browser will open over HTTP/1.1.
 */
export const DEFAULT_POOL_SIZE = 6;

/** The sweep the loader lane records, so the choice above is a measurement and not a preference. */
export const POOL_SWEEP_SIZES: readonly number[] = Object.freeze([4, 5, 6, 7, 8]);

/**
 * Bytes allowed in flight across the whole pool.
 *
 * The request count was not enough on its own when a request could be a whole run: two of the
 * plan's ranges are single oversized tensors, the 1,174,405,120 byte PLE table and the 100,663,296
 * byte embed table the head shares, and eight of those in flight would have asked a 16 GB machine
 * for nine gigabytes of response bodies. A range larger than this budget runs alone, which was the
 * only way the big two could be fetched at all.
 *
 * Since the oversized runs are fetched in `RANGE_PIECE_BYTES` pieces, no request is over 64 MiB and
 * the default pool of 6 asks for at most 384 MiB, which is exactly this budget. The budget stays,
 * as the guard it now is: a caller that raises the piece size or the pool has to raise this too,
 * rather than discovering the ceiling as an allocation failure.
 */
export const DEFAULT_MAX_BYTES_IN_FLIGHT = 384 * 1024 * 1024;

/**
 * The loader's concurrency for a device, from the one signal that separates a phone from a desktop
 * without guessing at a user agent: how large a buffer its adapter will grant.
 *
 * WHY THIS EXISTS. The pool holds whole response bodies on the JS heap, and the default budget is
 * 384 MiB of them. On a desktop that is insurance against a slow CloudFront edge and costs nothing.
 * On an iPhone it is 384 MiB of ArrayBuffers competing with the two gigabytes of weights going
 * resident on the GPU, inside a WebContent process that Safari kills rather than throws from.
 *
 * AND IT IS NOT ONLY THE DOWNLOAD. `serveFromCache` runs inside this same pool, so a load served
 * entirely from IndexedDB holds the same 384 MiB. That is the reported failure: the weights were
 * already downloaded, and the load died reading them back.
 *
 * The discriminator is `maxBufferSize`. iOS grants 1 GiB, desktop adapters grant several. A device
 * that caps a single buffer at a gigabyte is a device with a small memory budget, and that is a
 * fact it reports about itself rather than a string we pattern match.
 *
 * The cost is throughput: one request at a time instead of six. A load that takes longer beats a
 * tab that is killed, and this only applies to the devices that were failing outright.
 */
/**
 * How many bytes may be handed to `queue.writeBuffer` before waiting for the GPU to catch up.
 *
 * THE FAILURE THIS FIXES, from a crash trail off an iPhone. `writeBuffer` is fire and forget: it
 * copies into staging the implementation owns and returns, and nothing in the upload path ever
 * waits. Reading weights back from IndexedDB runs at about 675 MB/s, far faster than the GPU
 * consumes them, so the staging backlog grows without bound. The trail died at 1,479 MB of 2,109,
 * six seconds into a cache read, having never once waited.
 *
 * A desktop absorbs that backlog and nobody notices. A phone is killed by it, which is why this
 * looked like a memory ceiling and was not: the same device took writes into 2,688 MiB happily in
 * the allocation ladder, because the ladder awaits `onSubmittedWorkDone` after every chunk and the
 * loader awaited nothing. The ladder was accidentally the control that proved the point.
 *
 * The drain is a bound on work in flight, not a throttle. When the GPU is keeping up,
 * `onSubmittedWorkDone` resolves immediately and this costs a microtask per interval.
 */
export function uploadDrainBytes(maxBufferSize: number): number {
  const constrained = maxBufferSize > 0 && maxBufferSize <= 1024 * 1024 * 1024;
  // 128 MiB is the ladder's own chunk, which is the granularity this device is measured at.
  // Desktops drain too, an order of magnitude less often, because an unbounded queue is a bug
  // everywhere and only fatal on some machines.
  return constrained ? 128 * 1024 * 1024 : 1024 * 1024 * 1024;
}

export function loaderConcurrencyFor(maxBufferSize: number): { poolSize: number; maxBytesInFlight: number } {
  const constrained = maxBufferSize > 0 && maxBufferSize <= 1024 * 1024 * 1024;
  if (!constrained) return { poolSize: DEFAULT_POOL_SIZE, maxBytesInFlight: DEFAULT_MAX_BYTES_IN_FLIGHT };
  // One piece at a time. The pieces stay RANGE_PIECE_BYTES so a cache written by any other build
  // still resumes: the piece is the resume unit and changing it would orphan every stored marker.
  return { poolSize: 1, maxBytesInFlight: RANGE_PIECE_BYTES };
}

// ------------------------------------------------------------------------ cache aware fetching

/**
 * Where a load's bytes go. The loader knows nothing else about them.
 *
 * THE OFFSET IS PART OF THE CONTRACT. `bytes` is the span of the tensor starting at `offset`, and a
 * tensor larger than `RANGE_PIECE_BYTES` arrives as several calls whose spans do not overlap and
 * together cover it. `entry.byteLength` is the whole tensor's length, so a sink that allocates can
 * size from the first call it sees whichever call that is. The calls are serialized but their order
 * is the order the requests finished, so a sink must not assume the first call it gets is offset
 * zero. Everything under the piece size arrives exactly as it always did: one call, offset zero,
 * the whole tensor.
 */
export interface TensorSink {
  (name: string, entry: TensorEntry, bytes: Uint8Array, offset: number): void | Promise<void>;
}

/**
 * Split entries into cache hits and misses. Exposed on its own so the check scripts can drive the
 * partition with a fake cache; the network only ever sees the misses.
 */
export async function partitionByCache(
  entries: readonly TensorEntry[],
  repo: string,
  revision: string,
  lookup: (key: string) => Promise<Uint8Array | null>,
): Promise<{ hits: { entry: TensorEntry; bytes: Uint8Array }[]; misses: TensorEntry[] }> {
  const hits: { entry: TensorEntry; bytes: Uint8Array }[] = [];
  const misses: TensorEntry[] = [];
  for (const entry of entries) {
    const bytes = await lookup(tensorKey(repo, revision, entry.name));
    if (bytes !== null && bytes.byteLength === entry.byteLength) hits.push({ entry, bytes });
    else misses.push(entry);
  }
  return { hits, misses };
}

export interface LoadWeightsOptions {
  /** The safetensors resolve URL, which 302s to a CDN that honours Range. */
  fileUrl: string;
  repo: string;
  revision: string;
  /** text_config.num_hidden_layers, read from config.json by the caller. */
  numHiddenLayers: number;
  /** text_config.num_kv_shared_layers. */
  numKvSharedLayers: number;
  cache: WeightStore | null;
  fetchImpl?: typeof fetch;
  signal?: AbortSignal;
  /** Requests in flight at once. Defaults to `DEFAULT_POOL_SIZE`. */
  poolSize?: number;
  /**
   * Coalescing knobs for the range plan. Defaults to `DEFAULT_RANGE_PLAN_OPTIONS`, the frozen
   * 64 KiB gap and 64 MiB cap the plan measured against the real header, and a caller that passes
   * anything else is measuring rather than loading.
   */
  rangePlan?: { maxGapBytes?: number; maxRangeBytes?: number };
  /** Bytes in flight at once. Defaults to `DEFAULT_MAX_BYTES_IN_FLIGHT`. */
  maxBytesInFlight?: number;
  /**
   * Bytes per request when a run is larger than one. Defaults to `RANGE_PIECE_BYTES`, and a caller
   * that passes anything else is measuring rather than loading: this is what the checks turn down
   * so a split run costs kilobytes instead of gigabytes to prove. It must be a positive multiple of
   * four, because a piece is written into a GPU buffer at its own offset and `writeBuffer` needs
   * that offset four byte aligned.
   */
  pieceBytes?: number;
  /** Reused across loads in one session so the 302 is paid once. A fresh one is made when absent. */
  source?: SignedSource;
  /**
   * Refuse to start when `navigator.storage.estimate()` says there is not room. Default true.
   * The dev loader page turns it off when it is deliberately measuring a constrained origin.
   */
  requireStorage?: boolean;
  /** Overrides for `prepareStorage`, so a check can drive it without a browser. */
  storage?: StorageManagerLike | null;
  headroomBytes?: number;
  /**
   * Called once the header has been read, the tie decided and the plan built, before the first
   * payload byte. This is where the caller emits its opening progress event, which has to carry
   * `loaded: 0` and no `fromCache` for the host's download latch to read the load correctly.
   */
  onPlan?: (info: LoadPlanInfo) => void;
  /**
   * Byte progress. `loaded` is cumulative over both cached and fetched bytes; `fromCache` is true
   * only while the bytes being counted came from IndexedDB, matching the progress contract the
   * host latches on (plan.ts, weightsEvent).
   */
  onBytes?: (loaded: number, total: number, fromCache: boolean) => void;
  sink: TensorSink;
}

export interface LoadPlanInfo {
  directory: SafetensorsDirectory;
  selection: TextTensorSelection;
  plan: RangePlan;
  tie: TieGuardResult;
  /** Bytes of tensors this load will deliver, cached and fetched together. */
  totalBytes: number;
}

export interface LoadReceipt {
  /** The directory consumers should hold, with the ties resolved. */
  directory: SafetensorsDirectory;
  selection: TextTensorSelection;
  plan: RangePlan;
  tie: TieGuardResult;
  storage: StorageReadiness;
  /** Bytes of tensors delivered to the sink. Equals `plan.tensorBytes`. */
  totalBytes: number;
  cachedBytes: number;
  fetchedBytes: number;
  /** Bytes actually asked for over HTTP, including swallowed gap and the two manifest reads. */
  wireBytes: number;
  /** Requests issued, counted by kind, so a receipt can be checked against the transport spec. */
  requests: { resolve: number; manifest: number; tieGuard: number; runs: number };
  runsTotal: number;
  /** Runs served entirely from the cache because their marker was set and their tensors read back. */
  runsResumed: number;
  runsFetched: number;
  /** Runs whose marker was set and whose tensors would not read back, so they were fetched again. */
  runsRefetched: number;
  /**
   * Runs whose cached tensors read back at the right length and failed their marker's checksum, so
   * the bytes were dropped and the run went back on the wire. Counted separately from
   * `runsRefetched` because a read that fails is the storage stack behaving as round 2 measured it,
   * and a read that succeeds with the wrong bytes is not, and a receipt that folded them together
   * would let the second hide inside the first.
   */
  runsCorrupt: number;
  /**
   * Requests the runs actually cost, which is not the run count when a run is too big to be one
   * request. A run over `RANGE_PIECE_BYTES` is fetched and resumed in pieces of that size, so the
   * PLE table is 18 of these and the embed table 2 while every other run is 1
   * (safetensors.ts `splitRangeIntoPieces`). These four are what a resume actually skipped and
   * paid for; the four `runs` fields above are the same load told in the plan's own unit.
   */
  piecesTotal: number;
  piecesResumed: number;
  piecesFetched: number;
  piecesRefetched: number;
  /** Bytes per request when a run is split. `RANGE_PIECE_BYTES`, recorded so a receipt is readable. */
  pieceBytes: number;
  poolSize: number;
  chunkBytes: number;
  signedUrl: string;
  headers: SourceHeaders | null;
  /** The X-Repo-Commit the CDN reported, when it did. */
  commit: string | null;
  msTotal: number;
  msManifest: number;
  msPayload: number;
}

/**
 * The whole network path, from the resolve to the last tensor.
 *
 * Everything the transport specification asks for happens here in order, and the order is the
 * point: storage readiness before the first byte, one resolve, two manifest reads and not one
 * payload byte with them, the tie guard's 256 KiB before the allow list is built so the fallback
 * costs bytes and never correctness, and only then the pooled ranged GETs.
 *
 * The caller supplies the sink and nothing else about what happens to the bytes. `engine.ts` gives
 * one that uploads to the GPU; the dev loader page gives one that hashes a sample and drops the
 * rest, which is how this path gets exercised end to end without a model resident.
 */
export async function loadWeights(options: LoadWeightsOptions): Promise<LoadReceipt> {
  const doFetch = options.fetchImpl ?? defaultFetch();
  const startedAt = Date.now();
  const source = options.source ?? new SignedSource(options.fileUrl, doFetch);
  const poolSize = Math.max(1, Math.trunc(options.poolSize ?? DEFAULT_POOL_SIZE));
  const maxBytesInFlight = Math.max(1, options.maxBytesInFlight ?? DEFAULT_MAX_BYTES_IN_FLIGHT);
  const pieceBytes = Math.trunc(options.pieceBytes ?? RANGE_PIECE_BYTES);
  if (pieceBytes <= 0 || pieceBytes % 4 !== 0) {
    throw new Error(
      `gemma4 loader: pieceBytes must be a positive multiple of four, got ${pieceBytes}. A piece is `
      + 'written into a buffer at its own byte offset and writeBuffer needs that offset aligned.',
    );
  }
  const requests = { resolve: 0, manifest: 0, tieGuard: 0, runs: 0 };
  let wireBytes = 0;

  // One resolve, before anything else, so every request below is a single hop.
  await source.ensure(options.signal);
  requests.resolve = source.resolveCount;

  // The two request manifest read of ENGINE-PLAN section 3: eight bytes for the header length,
  // then the header JSON, and not one payload byte.
  const prefix = await source.fetchRange(
    { index: -1, start: 0, end: HEADER_PREFIX_BYTES, byteLength: HEADER_PREFIX_BYTES, tensors: [] },
    options.signal,
  );
  requests.manifest += 1;
  wireBytes += HEADER_PREFIX_BYTES;
  const headerLength = readHeaderLength(prefix.body);
  const headerBytes = HEADER_PREFIX_BYTES + headerLength;
  const header = await source.fetchRange(
    { index: -1, start: 0, end: headerBytes, byteLength: headerBytes, tensors: [] },
    options.signal,
  );
  requests.manifest += 1;
  wireBytes += headerBytes;
  const directory = parseHeader(header.body);

  // The tie guard, one 256 KiB ranged read before the allow list is built: the lm_head pair is
  // byte identical to the embed_tokens pair at the pinned revision, and the loader saves the
  // 101,711,872 bytes only after a digest of the guard sample confirms this file is that file. A
  // miss falls back to fetching both halves through the same allow list builder, so the fallback
  // cannot rot while the fast path is exercised.
  const guardRange = tieGuardRange(directory);
  const guard = await source.fetchRange(guardRange, options.signal);
  requests.tieGuard += 1;
  wireBytes += guardRange.byteLength;
  const tie = checkTieGuard(guard.body, options.revision);

  const selection = selectTextTensors(directory, {
    numHiddenLayers: options.numHiddenLayers,
    numKvSharedLayers: options.numKvSharedLayers,
    fetchTiedPair: !tie.tied,
  });
  if (selection.missing.length > 0) {
    throw new Error(
      `gemma4 loader: checkpoint is missing ${selection.missing.length} expected tensors, `
      + `first ${selection.missing[0]}`,
    );
  }
  const entries = resolveEntries(directory, selection.names);
  // Recomputed from the header at load time, never hardcoded. The tie already moved this from 51
  // ranges to 50 and the next allow list change will move it again.
  const plan = buildRangePlan(entries, options.rangePlan ?? DEFAULT_RANGE_PLAN_OPTIONS);
  const totalBytes = plan.tensorBytes;
  const msManifest = Date.now() - startedAt;

  options.onPlan?.({ directory: selection.resolvedDirectory, selection, plan, tie, totalBytes });

  // Storage readiness, before the first payload byte and after the plan says how many there are.
  // persist() is called on every load; estimate() is a refusal and not a warning, because a load
  // that runs out of quota at 1.8 GB has spent the user's bandwidth to produce nothing.
  const storage = await prepareStorage(plan.fetchedBytes, {
    storage: options.storage,
    headroomBytes: options.headroomBytes,
  });
  if (options.requireStorage !== false && !storage.sufficient) {
    throw new Error(`gemma4 loader: not enough storage to start. ${storage.message}`);
  }

  const payloadStart = Date.now();
  const result = await fetchRuns({
    plan,
    source,
    repo: options.repo,
    revision: options.revision,
    cache: options.cache,
    signal: options.signal,
    poolSize,
    maxBytesInFlight,
    totalBytes,
    pieceBytes,
    onBytes: options.onBytes,
    sink: options.sink,
  });
  // Requests, not runs: a run that is split costs one request per piece, and this field is counted
  // against the transport specification's request budget rather than against the plan's run list.
  requests.runs = result.piecesFetched + result.piecesRefetched;
  wireBytes += result.wireBytes;

  // The scope pin. ENGINE-PLAN section 3 says to capture X-Repo-Commit as the cache key alongside
  // the revision, and the research corrections add X-Xet-Hash as a content fingerprint. Measured
  // on 2026-09-01 against the live hub from a page: the signed CDN response exposes `etag` and
  // `accept-ranges` and NOT `x-repo-commit`, `x-xet-hash` or `x-linked-size`, whatever the resolve
  // hop's own access-control-expose-headers list says. So the pin falls back to the strong ETag,
  // which on this host is the content hash and is the better fingerprint of the two anyway: a
  // commit says which revision was asked for and a content hash says which bytes arrived. A load
  // that gets neither pins nothing, which costs a stale scope check and never a wrong answer.
  const commit = result.commit
    ?? source.lastHeaders?.commit
    ?? result.etag
    ?? source.lastHeaders?.etag
    ?? null;
  if (commit && options.cache) {
    await options.cache.ensureCommit(options.repo, options.revision, commit);
  }

  return {
    directory: selection.resolvedDirectory,
    selection,
    plan,
    tie,
    storage,
    totalBytes,
    cachedBytes: result.cachedBytes,
    fetchedBytes: result.fetchedBytes,
    wireBytes,
    requests,
    runsTotal: plan.ranges.length,
    runsResumed: result.runsResumed,
    runsFetched: result.runsFetched,
    runsRefetched: result.runsRefetched,
    runsCorrupt: result.runsCorrupt,
    piecesTotal: result.piecesTotal,
    piecesResumed: result.piecesResumed,
    piecesFetched: result.piecesFetched,
    piecesRefetched: result.piecesRefetched,
    pieceBytes,
    poolSize,
    chunkBytes: options.cache?.chunkBytes ?? 0,
    signedUrl: source.url,
    headers: source.lastHeaders,
    commit,
    msTotal: Date.now() - startedAt,
    msManifest,
    msPayload: Date.now() - payloadStart,
  };
}

interface FetchRunsOptions {
  plan: RangePlan;
  source: SignedSource;
  repo: string;
  revision: string;
  cache: WeightStore | null;
  signal?: AbortSignal;
  poolSize: number;
  maxBytesInFlight: number;
  totalBytes: number;
  /** Bytes per request when a run is larger than one, already validated by `loadWeights`. */
  pieceBytes: number;
  onBytes?: (loaded: number, total: number, fromCache: boolean) => void;
  sink: TensorSink;
}

interface FetchRunsResult {
  cachedBytes: number;
  fetchedBytes: number;
  wireBytes: number;
  commit: string | null;
  /** The strong ETag the CDN answered with, which on this host is the content hash. */
  etag: string | null;
  runsResumed: number;
  runsFetched: number;
  runsRefetched: number;
  runsCorrupt: number;
  piecesTotal: number;
  piecesResumed: number;
  piecesFetched: number;
  piecesRefetched: number;
}

/** How one piece resolved, tallied per run so the run counters keep meaning what they meant. */
type PieceOutcome = 'resumed' | 'fetched' | 'refetched' | 'corrupt';

/**
 * Run the plan: for each request, serve it from the cache when its marker says it landed and its
 * tensors still read back, and fetch it otherwise.
 *
 * THE RESUME UNIT IS THE REQUEST, which is the run for every run under `RANGE_PIECE_BYTES` and a
 * 64 MiB piece of it otherwise. Until 2026-09-02 the request was always the whole run, and the
 * weakness was named rather than hidden: the two oversized ranges are single tensors of
 * 1,174,405,120 and 100,663,296 bytes, so a connection dropped inside the PLE table cost the whole
 * table. It now costs at most 64 MiB. What that needed was a byte offset on the sink, which is a
 * change to the upload path and is why it waited for a ruling (ENGINE-PLAN round 3 worklist item 8).
 *
 * A piece carries its own marker, its own stored record and its own checksum, so a resumed load
 * skips exactly the pieces that landed. The run counters below still count runs, because the plan's
 * ranges are what a receipt is about, and a run is resumed only when every one of its pieces was;
 * the piece counters sit beside them for the runs where the two differ.
 *
 * A marked piece whose tensors will not read back is refetched and counted separately, because that
 * is the round 2 IndexedDB failure mode showing up in production and a receipt that hides it would
 * make the cache look healthy while the network paid for it.
 */
async function fetchRuns(options: FetchRunsOptions): Promise<FetchRunsResult> {
  const { plan, source, cache, sink } = options;
  let loaded = 0;
  let cachedBytes = 0;
  let fetchedBytes = 0;
  let wireBytes = 0;
  let commit: string | null = null;
  let etag: string | null = null;
  let runsCorrupt = 0;
  let piecesResumed = 0;
  let piecesFetched = 0;
  let piecesRefetched = 0;

  // Delivery is serialized even though fetching is not: the sink uploads to a GPU queue and writes
  // to IndexedDB, and two of those interleaved is a harder thing to reason about than it is worth
  // for a path whose cost is the network.
  let deliver: Promise<void> = Promise.resolve();
  const enqueue = (work: () => Promise<void>): Promise<void> => {
    deliver = deliver.then(work);
    return deliver;
  };

  // A split run's marker and stored records are addressed by piece; an unsplit run's are addressed
  // exactly as they always were, so nothing about the 48 ordinary runs of this checkpoint moved.
  const markerPiece = (piece: RangePiece): number | undefined => (piece.count > 1 ? piece.index : undefined);
  const storeKey = (piece: RangePiece, name: string): string => (piece.count > 1
    ? tensorPieceKey(options.repo, options.revision, name, piece.index)
    : tensorKey(options.repo, options.revision, name));

  const serveFromCache = async (piece: RangePiece): Promise<boolean> => {
    if (!cache) return false;
    const range = piece.range;
    const marker = await cache.getRun(options.repo, options.revision, range.index, markerPiece(piece));
    if (!marker) return false;
    const digests = marker.digests ?? null;
    const bytes: Uint8Array[] = [];
    for (const planned of range.tensors) {
      const stored = await cache.getTensor(storeKey(piece, planned.name));
      // A failed read is a cache miss to refetch, never a corruption to report. The round 2 probe
      // found values whose write transaction completed and whose read then failed, so the marker
      // is withdrawn and the piece goes back on the wire.
      if (stored === null || stored.byteLength !== planned.byteLength) {
        await cache.clearRun(options.repo, options.revision, range.index, markerPiece(piece));
        return false;
      }
      // And a read that succeeds with the wrong bytes is the same miss, which is the whole reason
      // the marker carries checksums. The length agreeing says nothing: a flipped byte, a chunk
      // fetched back from a neighbouring key, an entry that rotted under an evicting browser all
      // read back at exactly the right length, and without this the wrong bytes would go to the
      // sink and on to the GPU with nothing anywhere reporting it. A marker with no checksums at
      // all is a marker this loader cannot vouch for and is treated the same way.
      const expected = digests === null ? undefined : digests[planned.name];
      if (typeof expected !== 'string' || tensorChecksum(stored) !== expected) {
        await cache.clearRun(options.repo, options.revision, range.index, markerPiece(piece));
        runsCorrupt += 1;
        return false;
      }
      bytes.push(stored);
    }
    commit = marker.commit ?? commit;
    await enqueue(async () => {
      for (let i = 0; i < range.tensors.length; i += 1) {
        const planned = range.tensors[i]!;
        await sink(planned.name, planned.entry, bytes[i]!, piece.offsetInTensor);
        loaded += planned.byteLength;
        cachedBytes += planned.byteLength;
        options.onBytes?.(loaded, options.totalBytes, true);
      }
    });
    return true;
  };

  const fetchPiece = async (piece: RangePiece): Promise<void> => {
    const range = piece.range;
    const { body, headers } = await source.fetchRange(range, options.signal);
    wireBytes += range.byteLength;
    commit = headers.commit ?? commit;
    etag = headers.etag ?? etag;
    await enqueue(async () => {
      let stored = 0;
      const digests: Record<string, string> = {};
      for (const planned of range.tensors) {
        const bytes = sliceTensorFromRange(range, body, planned.name);
        await sink(planned.name, planned.entry, bytes, piece.offsetInTensor);
        if (cache) {
          // Checksummed off the wire bytes, before the write, so what the marker records is what
          // this load delivered to the sink and not what the store handed back afterwards. A
          // checksum taken by reading the entry back would agree with a bad write.
          const digest = tensorChecksum(bytes);
          const ok = await cache.putTensor(storeKey(piece, planned.name), bytes);
          if (ok) {
            stored += planned.byteLength;
            digests[planned.name] = digest;
          }
        }
        loaded += planned.byteLength;
        fetchedBytes += planned.byteLength;
        options.onBytes?.(loaded, options.totalBytes, false);
      }
      // The marker goes down after the last tensor of the piece, so a kill mid piece leaves the
      // piece unmarked and the next load refetches exactly it and nothing either side of it.
      if (cache && stored === range.tensors.reduce((sum, t) => sum + t.byteLength, 0)) {
        await cache.markRun(options.repo, options.revision, range.index, {
          commit,
          bytes: stored,
          tensors: range.tensors.length,
          at: Date.now(),
          digests,
        }, markerPiece(piece));
      }
    });
  };

  // The pool. Requests are taken in plan order, which is file order, and a request too large for
  // the byte budget waits until it is the only one in flight. Since every request is now at most
  // RANGE_PIECE_BYTES, that last clause is a guard rather than a path this checkpoint takes: the
  // whole queue fits the default budget six at a time.
  const queue: RangePiece[] = [];
  for (const range of plan.ranges) queue.push(...splitRangeIntoPieces(range, options.pieceBytes));
  const outcomes = new Map<number, PieceOutcome[]>();
  const record = (piece: RangePiece, outcome: PieceOutcome): void => {
    const list = outcomes.get(piece.range.index);
    if (list) list.push(outcome);
    else outcomes.set(piece.range.index, [outcome]);
  };
  let inFlightBytes = 0;
  let inFlight = 0;
  const running = new Set<Promise<void>>();
  let failure: unknown = null;

  const start = (piece: RangePiece): void => {
    inFlightBytes += piece.range.byteLength;
    inFlight += 1;
    const task = (async () => {
      if (options.signal?.aborted) throw new DOMException('weight fetch aborted', 'AbortError');
      const marked = cache
        ? (await cache.getRun(options.repo, options.revision, piece.range.index, markerPiece(piece))) !== null
        : false;
      if (await serveFromCache(piece)) {
        piecesResumed += 1;
        record(piece, 'resumed');
        return;
      }
      await fetchPiece(piece);
      if (marked) {
        piecesRefetched += 1;
        record(piece, 'refetched');
      } else {
        piecesFetched += 1;
        record(piece, 'fetched');
      }
    })()
      .catch((error) => {
        failure = failure ?? error;
      })
      .finally(() => {
        inFlightBytes -= piece.range.byteLength;
        inFlight -= 1;
        running.delete(task);
      });
    running.add(task);
  };

  while (queue.length > 0 || running.size > 0) {
    if (failure) break;
    const next = queue[0];
    const roomForRequest = inFlight < options.poolSize;
    const roomForBytes = inFlight === 0
      || (next !== undefined && inFlightBytes + next.range.byteLength <= options.maxBytesInFlight);
    if (next !== undefined && roomForRequest && roomForBytes) {
      queue.shift();
      start(next);
      continue;
    }
    if (running.size === 0) break;
    await Promise.race(running);
  }
  await Promise.allSettled(running);
  await deliver;
  if (failure) throw failure;

  // Runs from pieces. A run is resumed when every one of its pieces was served from the store, and
  // refetched when any piece of it had a marker it could not honour, which is the reading that
  // keeps `runsRefetched` meaning "the store said yes and could not deliver" on a split run too.
  let runsResumed = 0;
  let runsFetched = 0;
  let runsRefetched = 0;
  for (const list of outcomes.values()) {
    if (list.every((o) => o === 'resumed')) runsResumed += 1;
    else if (list.some((o) => o === 'refetched')) runsRefetched += 1;
    else runsFetched += 1;
  }

  return {
    cachedBytes,
    fetchedBytes,
    wireBytes,
    commit,
    etag,
    runsResumed,
    runsFetched,
    runsRefetched,
    runsCorrupt,
    piecesTotal: piecesResumed + piecesFetched + piecesRefetched,
    piecesResumed,
    piecesFetched,
    piecesRefetched,
  };
}

/**
 * Hash one tensor straight out of the cache.
 *
 * This is the loader lane's own verification instrument and it is deliberately not part of the
 * load: a cold load that hashed every tensor would spend minutes proving something a spot check
 * proves. The dev page samples a handful of tensors after a load and compares each digest against
 * the same bytes read out of the pinned snapshot, which is the only way to know the cache holds
 * the checkpoint rather than merely holding the right number of bytes.
 */
export async function hashCachedTensor(
  cache: WeightStore,
  repo: string,
  revision: string,
  name: string,
  limitBytes = 1 << 20,
): Promise<{ digest: string; byteLength: number; hashedBytes: number } | null> {
  const bytes = await cache.getTensor(tensorKey(repo, revision, name));
  if (bytes !== null) {
    const sample = bytes.subarray(0, Math.min(limitBytes, bytes.byteLength));
    return { digest: sha256Hex(sample), byteLength: bytes.byteLength, hashedBytes: sample.byteLength };
  }
  // A tensor the loader fetched in pieces is stored in pieces, so the whole tensor key holds
  // nothing and the walk below is the only way to see it. The pieces are numbered from zero with no
  // gaps, so the first miss is the end; the digest is of the head, which lives entirely in piece
  // zero, and the length is summed so a caller comparing it against the header still sees a half
  // cached tensor as a length that does not add up rather than as a tensor that is not there.
  let total = 0;
  let head: Uint8Array | null = null;
  for (let piece = 0; ; piece += 1) {
    const part = await cache.getTensor(tensorPieceKey(repo, revision, name, piece));
    if (part === null) break;
    if (piece === 0) head = part.slice(0, Math.min(limitBytes, part.byteLength));
    total += part.byteLength;
  }
  if (head === null) return null;
  return { digest: sha256Hex(head), byteLength: total, hashedBytes: head.byteLength };
}
