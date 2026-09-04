// Written from docs/ENGINE-PLAN.md section 5's architecture contract table, which the lead read
// from config.json and the Hugging Face transformers reference implementation at M0. No third
// party engine source was read.
//
// SPDX-License-Identifier: Apache-2.0
//
// The per layer geometry the norm, rope and embedding kernels need, as typed data rather than as
// prose in a document. ENGINE-PLAN section 8 M0 asks for exactly this move.
//
// This file is scoped to one kernel family because it was written before a shared architecture
// module existed. If the lead hoists these into one, delete this and re-export, but keep the
// disagreement note on `headDim` wherever they land: it is the one place the lab's own census and
// the config disagreed, and the config wins.

/** 35 decoder layers. */
export const LAYER_COUNT = 35;

/** Residual stream width. */
export const HIDDEN_SIZE = 1536;

/** 8 query heads, 1 key/value head. Grouped query attention at 8:1, multi query in practice. */
export const QUERY_HEADS = 8;
export const KV_HEADS = 1;

/** RMS norm epsilon, from config.json via the section 5 table. */
export const RMS_EPS = 1e-6;

/**
 * The seven full attention layers. One in five, with the last layer forced global.
 *
 * A correction worth carrying, because it is cited in ENGINE-PLAN risk 1: a published third party
 * description says every third layer is full attention. The config says every fifth, and these are
 * the indices. Take the config.
 */
export const GLOBAL_LAYERS: readonly number[] = [4, 9, 14, 19, 24, 29, 34];

const GLOBAL_SET = new Set(GLOBAL_LAYERS);

export function isGlobalLayer(layer: number): boolean {
  return GLOBAL_SET.has(layer);
}

/**
 * Head dimension is 256 on sliding layers and 512 on the seven full attention layers.
 *
 * This is the disagreement M0 found and recorded rather than smoothed over: the lab's live
 * dispatch census measured a `pos * 256` cache slot stride and a head dimension of 256, but it
 * sampled sliding layers. The 256 is the sliding value, not a universal one
 * (ENGINE-PLAN 5, 6 and 8). Anything that treats 256 as a constant is wrong on seven layers in
 * thirty five, and wrong quietly.
 */
export function headDimForLayer(layer: number): number {
  return isGlobalLayer(layer) ? 512 : 256;
}

/** Sliding window length in positions, inclusive of the current one, so the window is [q-511, q]. */
export const SLIDING_WINDOW = 512;

/** Layers 0 to 14 own KV caches; layers 15 to 34 read a producer's, per num_kv_shared_layers 20. */
export const KV_PRODUCER_LAYERS = 15;

export function isKvProducer(layer: number): boolean {
  return layer < KV_PRODUCER_LAYERS;
}

/** PLE table row layout: 35 groups of 256, one group per layer. */
export const PLE_GROUP_SIZE = 256;
export const PLE_GROUPS = LAYER_COUNT;
export const PLE_ROW_VALUES = PLE_GROUP_SIZE * PLE_GROUPS;

/** Vocabulary, which is also the PLE table's row count and the lm_head output width. */
export const VOCAB_SIZE = 262144;
