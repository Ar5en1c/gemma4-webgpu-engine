// Written from docs/ENGINE-PLAN.md section 6 (the KV cache and prefix reuse specification),
// section 5's architecture contract table, and the measurement record in DECODE-CAMPAIGN.md 4.7.
// No vendored bundle, no extracted kernel and no third party engine source was read.
//
// SPDX-License-Identifier: Apache-2.0
//
// KV cache layout and bookkeeping. Pure TypeScript, no GPU types: this module decides where
// bytes go and how many positions are valid, and the kernels in kernels/kvStore.ts and
// kernels/attention.ts read and write the layout it describes. Keeping the arithmetic here, pure
// and Node testable, is deliberate, because the prefix rewind below is where the app's whole
// second turn TTFT story lives (437 ms against 1,863 ms, ENGINE-PLAN section 6) and a rewind bug
// does not crash, it just makes the model quietly slow weeks later.
//
// THE LAYOUT, from ENGINE-PLAN section 6's consequences list:
//
//  - One cache per KV producer layer, layers 0 to 14 (num_kv_shared_layers 20 out of 35).
//    Consumer layers 15 to 34 own nothing: a sliding consumer reads the last sliding producer's
//    cache (layer 13) and a full attention consumer reads the last full attention producer's
//    cache (layer 14). Two shared caches at the tail, not one, so truncation is a counter change
//    per producer cache, fifteen counters, never thirty five.
//  - Each cache is one contiguous buffer holding its K region then its V region, both addressed
//    by absolute position: element (pos, d) of K lives at pos * headDim + d, and the V region
//    starts at maxContext * headDim. Absolute position addressing is what makes truncate() a
//    counter change with no GPU work. The census measured slot addressing at pos * 256 on the
//    layers it sampled (DECODE-CAMPAIGN.md 4.7); that is the sliding stride, and this layout
//    carries the stride per cache because the seven global layers run head_dim 512, not 256.
//  - K and V are packed into one buffer rather than two so the store kernel writes both regions
//    in one dispatch and the attention kernels bind the cache once, spending two of an adapter
//    budget of ten storage buffers instead of three (ENGINE-PLAN risk 6).
//  - v1 caches are full length, no sliding ring. A ring for the 12 sliding producer caches is the
//    obvious later saving (positions older than q - 511 are masked anyway) and nothing in this
//    layout forbids it: ring indexing changes slotOffset() and nothing else. Same for caching V
//    as quantized codes (ENGINE-PLAN section 6): the per cache stride lives here, so a code plus
//    scale V region is a stride change in one place.
//
// MEMORY MATH, at the default maxContext of 2048 and f32 elements:
//
//    sliding producer (head_dim 256): K + V = 2 * 2048 * 256 * 4 B = 4 MiB, 12 layers = 48 MiB
//    global producer  (head_dim 512): K + V = 2 * 2048 * 512 * 4 B = 8 MiB,  3 layers = 24 MiB
//    total                                                                              72 MiB
//
// 2048 covers the app's real horizon with headroom: client.ts trims prompts to a 700 token
// budget and clamps maxNewTokens, so a turn tops out well under 1k positions. Against the model's
// max_position_embeddings of 131072 this is a deliberate app sized allocation, not a model sized
// one; the ceiling is a constructor argument, and growth beyond it is a reallocation policy for
// the scheduler lane, in blocks, per ENGINE-PLAN section 6.

import {
  GLOBAL_LAYERS,
  KV_PRODUCER_LAYERS,
  KV_HEADS,
  LAYER_COUNT,
  headDimForLayer,
  isGlobalLayer,
  isKvProducer,
} from './kernels/layerGeometry';

/** Default cache capacity in positions. See the memory math above. */
export const DEFAULT_MAX_CONTEXT = 2048;

/** Bytes per cached element. v1 caches f32; f16 or coded V would halve or quarter this. */
export const KV_ELEMENT_BYTES = 4;

// ------------------------------------------------------------------------------- layer roles

export type KvRole =
  | { kind: 'producer'; layer: number }
  | { kind: 'consumer'; layer: number; readsFrom: number };

/**
 * Which cache a layer's attention reads. Producers read their own; consumers read the last
 * producer of their own attention type, per attention type sharing (ENGINE-PLAN section 6):
 * sliding consumers read layer 13, full attention consumers read layer 14. Both derived here
 * from the config facts rather than hardcoded, so a config change moves them.
 */
export function kvRoleForLayer(layer: number): KvRole {
  if (!Number.isInteger(layer) || layer < 0 || layer >= LAYER_COUNT) {
    throw new Error(`kvRoleForLayer: layer ${layer} outside 0..${LAYER_COUNT - 1}`);
  }
  if (isKvProducer(layer)) return { kind: 'producer', layer };
  const wantGlobal = isGlobalLayer(layer);
  let last = -1;
  for (let p = 0; p < KV_PRODUCER_LAYERS; p += 1) {
    if (isGlobalLayer(p) === wantGlobal) last = p;
  }
  if (last < 0) throw new Error(`kvRoleForLayer: no producer of the same type as layer ${layer}`);
  return { kind: 'consumer', layer, readsFrom: last };
}

// ------------------------------------------------------------------------------------ layout

/** Everything a kernel needs to address one producer cache. All offsets in f32 elements. */
export interface KvCacheDesc {
  /** Producer layer this cache belongs to. */
  layer: number;
  /** 256 sliding, 512 global. The stride per position, since KV heads is 1. */
  headDim: number;
  /** Capacity in positions. */
  maxContext: number;
  /** Element offset of the V region inside the packed buffer: maxContext * headDim. */
  vBase: number;
  /** Total buffer size in elements: 2 * maxContext * headDim. */
  elements: number;
  /** Total buffer size in bytes at KV_ELEMENT_BYTES per element. */
  bytes: number;
}

export interface KvLayout {
  maxContext: number;
  /** One descriptor per producer layer, index equals layer for layers 0..14. */
  caches: KvCacheDesc[];
  /** Sum of every cache's bytes, the number the memory math comment above predicts. */
  totalBytes: number;
}

export function createKvLayout(maxContext: number = DEFAULT_MAX_CONTEXT): KvLayout {
  if (!Number.isInteger(maxContext) || maxContext <= 0) {
    throw new Error(`createKvLayout: maxContext must be a positive integer, got ${maxContext}`);
  }
  if (KV_HEADS !== 1) {
    // The stride arithmetic below folds the KV head dimension away because there is exactly one.
    throw new Error('createKvLayout: layout assumes one KV head, per the architecture contract');
  }
  const caches: KvCacheDesc[] = [];
  let totalBytes = 0;
  for (let layer = 0; layer < KV_PRODUCER_LAYERS; layer += 1) {
    const headDim = headDimForLayer(layer);
    const vBase = maxContext * headDim;
    const elements = 2 * vBase;
    const bytes = elements * KV_ELEMENT_BYTES;
    caches.push({ layer, headDim, maxContext, vBase, elements, bytes });
    totalBytes += bytes;
  }
  return { maxContext, caches, totalBytes };
}

/** Element offset of K row `pos` inside a cache's packed buffer. */
export function kSlotOffset(desc: KvCacheDesc, pos: number): number {
  if (!Number.isInteger(pos) || pos < 0 || pos >= desc.maxContext) {
    throw new Error(`kSlotOffset: position ${pos} outside 0..${desc.maxContext - 1}`);
  }
  return pos * desc.headDim;
}

/** Element offset of V row `pos` inside a cache's packed buffer. */
export function vSlotOffset(desc: KvCacheDesc, pos: number): number {
  return desc.vBase + kSlotOffset(desc, pos);
}

// --------------------------------------------------------------------------- prefix rewind

/**
 * Longest common prefix of two token id sequences. The o of ENGINE-PLAN section 6.
 */
export function longestCommonPrefix(a: readonly number[], b: readonly number[]): number {
  const n = Math.min(a.length, b.length);
  let i = 0;
  while (i < n && a[i] === b[i]) i += 1;
  return i;
}

/**
 * The plan's five behavioural cases, as data. `action` is what the engine does to the caches,
 * `reuse` is how many cached positions survive, and `prefillFrom`/`prefillCount` describe the
 * suffix the prefill path must run. The canary described in ENGINE-PLAN section 6 asserts
 * `reuse` directly, which is only possible because this is a value and not a side effect.
 */
export interface KvReusePlan {
  action: 'extend' | 'rewind' | 'reset';
  /** Cached positions kept. Always 0 when action is 'reset'. */
  reuse: number;
  /** First position the prefill runs at, equal to reuse. */
  prefillFrom: number;
  /** next.length - reuse when kept, next.length when reset. Always > 0. */
  prefillCount: number;
}

/**
 * Decide what to do with the cache when a new prompt arrives. Pure: reads nothing, mutates
 * nothing, so the five cases are testable as a table.
 *
 * The cases, numbered as ENGINE-PLAN section 6 numbers them (case 1 there is computing o):
 *
 *   2. extend        o == cached.length, next longer: keep everything, prefill the suffix.
 *   3. rewind        0 < o < cached.length: truncate to o, prefill the suffix. Stale slots
 *                    beyond o need no clearing because position addressed writes overwrite them
 *                    and nothing beyond the valid count is ever read. Tabby hits this every turn.
 *   4. full reset    o == 0: cold start.
 *   5. empty suffix  next.length == o, which covers both the exact match and the strict prefix:
 *                    reset entirely. Matching the current engine's behaviour here keeps the swap
 *                    invisible; a future engine could rewind to o and skip prefill, but that is
 *                    a behaviour change and it is not made silently in a compatibility round.
 */
export function planKvReuse(
  cached: readonly number[],
  next: readonly number[],
): KvReusePlan {
  if (next.length === 0) {
    // An empty prompt has no last position to decode from; the loader never sends one because
    // the chat template always emits at least BOS plus the generation header.
    throw new Error('planKvReuse: next must hold at least one token');
  }
  const o = longestCommonPrefix(cached, next);
  if (o === next.length) {
    // Case 5: nothing to prefill. Exact match or strict prefix of the cache. Reset.
    return { action: 'reset', reuse: 0, prefillFrom: 0, prefillCount: next.length };
  }
  if (o === 0) {
    // Case 4. Also covers an empty cache, which is a cold start by definition.
    return { action: 'reset', reuse: 0, prefillFrom: 0, prefillCount: next.length };
  }
  if (o === cached.length) {
    // Case 2, the ordinary growing transcript.
    return { action: 'extend', reuse: o, prefillFrom: o, prefillCount: next.length - o };
  }
  // Case 3, the divergent tail.
  return { action: 'rewind', reuse: o, prefillFrom: o, prefillCount: next.length - o };
}

// -------------------------------------------------------------------------------- bookkeeping

/**
 * The host side state of the cache set: which token ids the caches hold valid K and V for.
 * One instance covers all fifteen producer caches because they advance in lockstep; the per
 * cache buffers differ, the position counter does not.
 *
 * Everything here is a counter or an array copy. No GPU work, which is the whole point
 * (ENGINE-PLAN section 6: "truncation is a bookkeeping operation on the host").
 */
export class KvCacheState {
  private tokens: number[] = [];

  /** Token ids the caches currently hold valid state for, oldest first. A copy. */
  get cachedTokens(): number[] {
    return this.tokens.slice();
  }

  /** Number of valid cache positions. The only number the attention kernels are told. */
  get length(): number {
    return this.tokens.length;
  }

  /**
   * Truncate to the first `nTokens` positions. The rewind primitive: a counter change, no
   * clearing, because slots are position addressed and nothing beyond the count is read.
   */
  truncate(nTokens: number): void {
    if (!Number.isInteger(nTokens) || nTokens < 0 || nTokens > this.tokens.length) {
      throw new Error(
        `KvCacheState.truncate: ${nTokens} outside 0..${this.tokens.length}`,
      );
    }
    this.tokens.length = nTokens;
  }

  /** Forget everything. Identical to a cold start. */
  reset(): void {
    this.tokens.length = 0;
  }

  /**
   * Record that K and V for `tokenIds` were written at positions length..length+n-1. Called by
   * the scheduler after the store kernel ran for a prefill chunk or a decode step.
   */
  append(tokenIds: readonly number[]): void {
    for (const id of tokenIds) {
      if (!Number.isInteger(id) || id < 0) {
        throw new Error(`KvCacheState.append: bad token id ${id}`);
      }
    }
    this.tokens.push(...tokenIds);
  }

  /**
   * Plan and apply the cache side of a new prompt in one step: computes the plan against the
   * current state, truncates or resets accordingly, and returns the plan. The caller then
   * prefills `next.slice(plan.prefillFrom)` and calls `append` with those ids as chunks land.
   */
  acceptPrompt(next: readonly number[]): KvReusePlan {
    const plan = planKvReuse(this.tokens, next);
    if (plan.action === 'reset') this.reset();
    else if (plan.action === 'rewind') this.truncate(plan.reuse);
    // 'extend' keeps everything.
    return plan;
  }
}
