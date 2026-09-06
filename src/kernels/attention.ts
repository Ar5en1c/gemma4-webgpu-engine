// Written from docs/ENGINE-PLAN.md section 5 (kernel K8 and the architecture contract table),
// risk 1 quirks 2 and 3, section 5.5's reduction rules, risk 2's drift mitigations, the round 3
// lead ruling 5 (f32 order changes under the gates that have power), the flash attention
// decomposition from published work, the layout in ../kv.ts, and the WGSL specification. No
// vendored bundle, no extracted kernel and no third party engine source was read.
//
// SPDX-License-Identifier: Apache-2.0
//
// Attention, kernel K8, both layer kinds, decode and prefill shapes.
//
// THE SCALE IS 1.0. Trap 2 of the four silent corruption hazards, so it is said here at the top:
// Gemma 4's per head q and k norms are calibrated to produce unit RMS heads, and the query
// pre-attention scaling in the reference is exactly 1.0, not head_dim ** -0.5. Applying the
// standard compensation anyway flattens softmax by a factor of 16 at head dimension 256 and the
// output approximates the mean of V: fluent, context unaware text (ENGINE-PLAN risk 1 quirk 2).
// The constant below is named, the reference capture records the scaling the HF implementation
// actually passed, and scripts/engine-check/k-attention.mjs asserts both that the captured value
// is 1.0 and that flipping the oracle to 1/sqrt(d) collapses its cosine against the reference.
//
// There is NO attention logit softcap. config.json carries final_logit_softcapping 30.0 for the
// logit head and no attention softcapping key at all, which matches the section 5 contract table
// ("no attention logit softcapping in the text model"). Checked against the file, not recalled.
//
// THE TWO LAYER KINDS differ in three parameters and nothing else here: head dimension (256
// sliding, 512 global), window (512 positions inclusive of the query, so [q-511, q], against no
// window at all), and, upstream of this kernel, the RoPE treatment. Partial RoPE, trap 3, does
// not appear in this file on purpose: Q and K arrive already rotated by K7 and V already normed
// by K6, so attention itself is architecture blind past those three parameters. That is also why
// the reference capture taps the eager attention interface exactly there.
//
// GQA AT 8:1 IS MULTI QUERY. One KV head serves all eight query heads, so the cache holds one
// row per position and every workgroup of this kernel reads the same K and V rows. No head
// indexing exists on the cache side, which is the entire bandwidth argument for GQA.
//
// SHAPE. One workgroup per (query position, head). The workgroup is `slices` slices of 64 lanes
// (two 32 lane virtual subgroups each), and the sequence is dealt to the slices chunk by chunk:
// chunk c of 64 positions belongs to slice c mod slices, so slice s walks chunks s, s + slices,
// s + 2 slices and so on with its own running online softmax state, exactly the round 3 kernel's
// loop over a subsequence. The walk starts at the first chunk the sliding window can reach
// rather than at chunk 0, which is an identity on fully masked chunks and keeps every chunk on
// the slice it already had; see WHERE THE WALK STARTS at the loop. Inside a chunk each lane scores one position (the score is a dot()
// chain over vec4 lanes, the shape that never drifted, DECODE-FUSION-FINDING.md), the chunk max
// and sum reduce through the 32 lane butterflies BEFORE the one guarded scratch store (rule 1),
// and the running (max, sum, accumulator) merge is the standard flash rescaling. At the end the
// slices' (max, sum, accumulator) triples are merged once more in workgroup memory, in slice
// order, and slice 0 writes the row.
//
// WHY THE SPLIT, and what it moves. Round 3 measured decode attention at 7.6 ms per token on
// 30 MB of cache, 3.9 GB/s, with 35 dispatches of eight workgroups each: one workgroup per head
// walking every chunk in sequence is a latency chain, not a bandwidth stream, and the GPU sat
// mostly idle under it (docs/ENGINE-PERF.md section 7). Dealing the chunks to `slices` slices
// cuts the chain by that factor and keeps one dispatch per layer. The merge across slices sums
// the softmax numerator and denominator in a different f32 order from the single walk, so this is
// the kernel ENGINE-PLAN round 3 lead ruling 5 is about: an integer kernel cannot move under it,
// and the probe token sequences are re-receipted as parity-r4 with every moved token listed
// beside the reference's top two gap at that position. At `slices` 1 the merge multiplies by
// exp(0), which is exactly 1.0, and the kernel is bit for bit the round 3 kernel; the sweep's
// slices 1 arm is that control.
//
// PRECONDITION the scheduler owns: the store kernel (K9) has already written K and V for every
// position up to and including the query position, so kvLen > qStart + qCount - 1 never fails.
// Decode therefore stores the new token's KV first and attends second, which also means the
// query always has at least itself to attend to and the softmax denominator is never zero.

import type { Kernel, KernelBindInput, KernelBindResult } from './registry';
import { kernelLayout, kernelUniform } from './binding';
import {
  SUBGROUP_BUTTERFLY_MAX_WGSL,
  SUBGROUP_BUTTERFLY_WGSL,
  SUBGROUP_ENABLE,
} from './subgroupReduce';

/** Lanes per slice: 64, two 32 lane virtual subgroups, the round 3 workgroup (DECODE-CAMPAIGN.md 4.2). */
export const SLICE_LANES = 64;

/**
 * Trap 2, as a named constant. 1.0 exactly, never head_dim ** -0.5. Anyone changing this number
 * is changing the model, and the k-attention check has a test that fails if the oracle stops
 * matching the reference under exactly this value.
 */
export const ATTENTION_SCALE = 1.0;

/** Scores below this stand for "masked". Well under any real logit, far above f32 -max. */
const MASKED_SCORE = '-3.0e38';

export type AttnReducePath = 'subgroup' | 'workgroup';

/**
 * The attention geometry, one number the round 4 performance campaign sweeps: how many 64 lane
 * slices share one (position, head) workgroup, each taking every `slices`th chunk of the
 * sequence. 1 is the round 3 kernel to the bit. The workgroup is 64 times this wide, so 4 is the
 * widest value a device on WebGPU's default 256 invocation limit can compile; device.ts raises
 * that limit to the adapter's maximum, 1024 on this M1, so 8 compiles here.
 */
export interface AttentionGeometry {
  slices: 1 | 2 | 4 | 8;
  /**
   * How many WORKGROUPS share one (position, head), each taking every `kvSplits`th block of the
   * window and writing an unnormalised softmax partial that `attention-decode-merge` folds.
   *
   * WHY THIS IS A DIFFERENT AXIS FROM `slices`, WHICH ALREADY EXISTS. `slices` makes one workgroup
   * wider. A workgroup runs on one core, so no value of `slices` puts work on a second core. The
   * decode dispatch is (qCount, heads), which is EIGHT workgroups, and eight is all the parallelism
   * this kernel has ever had whatever `slices` said.
   *
   * WHY IT DEFAULTS TO 1, WHICH IS THE SHIPPED KERNEL TO THE BYTE. On the M1 eight workgroups is
   * already the knee: one through eight cost the same 0.037 ms and sixteen costs 1.64x
   * (gemma4-kernels-lab/lab-results/m1-attention-occupancy-sep04.json). That is also why round 5
   * promoted slices to workgroups and measured exactly zero. On a part with six times the cores the
   * knee must sit elsewhere, and attention there costs 2.62 ms of a 5.4 ms token at 11.3 GB/s. So
   * this is a per device setting whose default changes nothing, not a new shipped shape.
   */
  kvSplits?: 1 | 2 | 4 | 8 | 16;
  /**
   * Independent accumulators the weighted V loop carries, breaking its dependency chain by this
   * factor.
   *
   * ROUND 5 FIXED THE SHORTER CHAIN AND LEFT THE LONGER ONE. It found the attention stall is
   * inside the lane rather than in barriers or scheduling, and cut the SCORE loop's chain from
   * headDim4 to headDim4 over 4 with four accumulators, worth about 1.25 ms a token. The weighted
   * V loop has the same shape and was not touched: 64 iterations, ONE accumulator, and every
   * iteration also waits on its own global read of a different 1 KB row. At headDim 256 that
   * leaves a chain of 64 where the score loop now has 16, so the loop round 5 did not fix is now
   * the longer of the two.
   *
   * 1 is the shipped kernel to the byte. This reassociates the V sum, the same class of change as
   * the slice count under round 3 lead ruling 5, so it is gated rather than argued and defaults
   * off until it is measured on both machines.
   */
  vAccumulators?: 1 | 2 | 4 | 8 | 16;
  /**
   * Independent accumulators in the SCORE loop, the dot of q against one K row. The shipped kernel
   * carries four (see THE SCORE, ON FOUR ACCUMULATORS in the shader), which is 16 dependent loads
   * deep at head dimension 256 and 32 at 512. Eight and sixteen cut that chain by two and four.
   * The per kernel timing on the 5070 (lab-results/5070-per-kernel-decode-sep04.json) found the
   * stall inside the lane, and the dead chunk skip family measured that a dispatch of one chunk
   * walks is bound by that chain and nothing else. Default 4 is the shipped text to the byte.
   * Like vAccumulators this reassociates a sum and is gated, not argued.
   */
  scoreAccumulators?: 4 | 8 | 16;
  /**
   * How the score loop is laid across the lanes. 'rows', the shipped kernel: each lane walks its
   * own K row, so a warp's load touches 32 rows for 16 bytes each. 'dims': the lanes of each 32
   * lane group span the head dimension, positions are iterated, each position's dot closes with
   * one sgSum32 butterfly and lands in the lane that owns that position, so a warp's load is 512
   * contiguous bytes. The o family measured the rows loop getting SLOWER with more loads in flight
   * while the coalesced V loop got faster, which is what a transaction bound loop does beside a
   * latency bound one. 'dims' needs the butterfly, so under the workgroup reduce policy it falls
   * back to 'rows'. scoreAccumulators has no meaning under 'dims' and is ignored there. Default
   * 'rows' is the shipped text to the byte. A different f32 association of the dot, gated.
   */
  scoreLayout?: 'rows' | 'dims';
}

/** The round 3 shape, kept as the sweep's control and the shape the fixtures were first proved at. */
export const ROUND3_ATTENTION_GEOMETRY: Readonly<AttentionGeometry> = Object.freeze({ slices: 1 });

/**
 * The shipped geometry. The round 4 lanes chose four slices (docs/ENGINE-PERF.md round 4, lever 2)
 * and the lead then measured eight on the timing arm, same tree and same morning: attention decode
 * 3.15 ms per token at four slices against 2.03 ms at eight (9.4 to 14.6 GB/s over the 29.6 MB the
 * kernel reads per token), the GPU total moving from about 27.5 to 26.1 ms. Eight is the widest the
 * kernel allows and the sweep, the full model page and the parity receipt were re-run at it before
 * it shipped (docs/ENGINE-PERF.md section 12).
 */
export const DEFAULT_ATTENTION_GEOMETRY: Readonly<AttentionGeometry> = Object.freeze({ slices: 8 });

let activeGeometry: Readonly<AttentionGeometry> = DEFAULT_ATTENTION_GEOMETRY;

function assertGeometry(g: AttentionGeometry): void {
  const s = g.slices as number;
  if (s !== 1 && s !== 2 && s !== 4 && s !== 8) {
    throw new Error(`attention geometry: slices must be 1, 2, 4 or 8, got ${String(s)}`);
  }
  const k = (g.kvSplits ?? 1) as number;
  if (k !== 1 && k !== 2 && k !== 4 && k !== 8 && k !== 16) {
    throw new Error(`attention geometry: kvSplits must be 1, 2, 4, 8 or 16, got ${String(k)}`);
  }
  const a = (g.vAccumulators ?? 1) as number;
  if (a !== 1 && a !== 2 && a !== 4 && a !== 8 && a !== 16) {
    throw new Error(`attention geometry: vAccumulators must be 1, 2, 4, 8 or 16, got ${String(a)}`);
  }
  const sa = (g.scoreAccumulators ?? 4) as number;
  if (sa !== 4 && sa !== 8 && sa !== 16) {
    throw new Error(`attention geometry: scoreAccumulators must be 4, 8 or 16, got ${String(sa)}`);
  }
  const sl = (g.scoreLayout ?? 'rows') as string;
  if (sl !== 'rows' && sl !== 'dims') {
    throw new Error(`attention geometry: scoreLayout must be rows or dims, got ${String(sl)}`);
  }
}

/**
 * The largest split the geometry allows, which is what the partial slot is sized at so a plan
 * taken on one machine can be replayed on another.
 */
export const ATTENTION_MAX_KV_SPLITS = 8;

/** Splits this geometry asks for, with the default that means "the shipped kernel". */
export function kvSplitsOf(geometry: Readonly<AttentionGeometry>): number {
  return geometry.kvSplits ?? 1;
}


/**
 * The same geometry with the KV split removed. attention-prefill builds through this. Before it
 * existed bindAttention read the split off the geometry singleton for every attention dispatch, so
 * turning kvSplits on for decode split PREFILL too; execute.ts routes partials to attn.part only on
 * the decode path, so prefill wrote unfolded partials into the plain attn slot and the prompt was
 * attended wrongly before the first token was ever sampled. The split now rides on kernel identity.
 */
export function unsplitAttentionGeometry(geometry: Readonly<AttentionGeometry>): Readonly<AttentionGeometry> {
  return kvSplitsOf(geometry) === 1 ? geometry : Object.freeze({ ...geometry, kvSplits: 1 });
}

/** The split KV decode build. Emitted by the plan only alongside ROLE_ATTN_MERGE. */
export const ATTENTION_DECODE_SPLIT_KERNEL = 'attention-decode-split';

/**
 * vec4 lanes one (position, head, split) partial occupies: the accumulator, then one lane carrying
 * the running max in .x and the running sum in .y. Exported so a caller can size the scratch.
 */
export function partialStride4(headDim4: number): number {
  return headDim4 + 1;
}

/**
 * Override the attention geometry for every pipeline built afterwards, or restore the default
 * with null. A dev seam for the performance rig's sweep, called before any pipeline is built,
 * because the pipeline store caches modules by their text and the registry entries read the
 * geometry at build time. Nothing in the shipping engine calls this.
 */
export function setAttentionGeometry(geometry: AttentionGeometry | null): void {
  if (geometry !== null) assertGeometry(geometry);
  activeGeometry = geometry === null ? DEFAULT_ATTENTION_GEOMETRY : Object.freeze({ ...geometry });
}

export function attentionGeometry(): Readonly<AttentionGeometry> {
  return activeGeometry;
}

/**
 * Build the attention WGSL.
 *
 * Rule compliance, item by item, because this is the kernel where the rules earn their keep:
 *
 *  - Both butterflies run in uniform control flow, above the single guarded scratch store of
 *    each chunk iteration (rule 1: reduce above store, one guarded store block). The
 *    probability store into probScratch after the barrier is unconditional, every lane exactly
 *    one slot, so it is not lane divergent. Across chunk iterations there is a loop back edge
 *    between any store and the next reduction, which is the shape the Bonsai refinement showed
 *    dodges the NVIDIA trigger even before the hoisting (PREFILL-CAMPAIGN.md round 3).
 *  - Every slice runs the same number of iterations, the chunk span the window and the query
 *    position leave divided by `slices` and rounded up, from uniform values at runtime (rule 6),
 *    so the barriers inside the loop are reached by every lane of the workgroup. A slice whose chunk index runs past the sequence processes a chunk
 *    with every lane masked: its max is the floor, its sum is zero, its correction is exp(0),
 *    which is exactly 1.0, and its accumulators are unchanged to the bit. That is the same
 *    arithmetic a fully masked chunk past the query position already takes in prefill.
 *  - The inner 64 iteration accumulation loop is a fixed small bound and, more to the point, a
 *    fixed ORDER: position j then j+1, always, which freezes the accumulation order per output
 *    element within a slice (risk 2 mitigation 2). The cross slice merge is in slice order.
 *  - Score dots are dot() over vec4 lanes (risk 2 mitigation 1).
 *  - Out of range lanes clamp their read to a real row and mask the RESULT, never the read, so
 *    control flow stays uniform for Tint and no NaN from uninitialised cache slots can ride in
 *    through a 0 * x product (rule 5's data level masking).
 *  - Three storage buffers plus one uniform, against the adapter budget of 10 (risk 6).
 */
/** 0..n-1, so the generated unroll reads as a list rather than a loop of string concatenation. */
function vaRange(n: number): number[] {
  return Array.from({ length: n }, (_, i) => i);
}

/** `(d0 + d1) + (d2 + d3)` and its wider siblings: the chains folded pairwise, balanced. */
function pairwiseSum(names: string[]): string {
  if (names.length === 1) return names[0]!;
  const half = names.length / 2;
  return `(${pairwiseSum(names.slice(0, half))} + ${pairwiseSum(names.slice(half))})`;
}

/**
 * The score loop at `sa` accumulators, for sa above the shipped four. Same loads, same dots, same
 * clamp; only how many partial sums are in flight, and therefore the f32 association of the dot.
 * The main loop steps by sa and the tail loop runs zero times at the two head dimensions this
 * model has, exactly as the four accumulator text's tail does.
 */
function scoreBlock(sa: number): string {
  const ds = vaRange(sa).map((a) => `d${a}`);
  return [
    `    // ${sa} independent chains over the head dimension, so ${sa} loads and ${sa} multiply adds`,
    '    // are in flight instead of four. Same loads, same dots; only the association moves.',
    ...ds.map((d) => `    var ${d} = 0.0;`),
    `    let n4xN = (n4 / ${sa}u) * ${sa}u;`,
    `    for (var i = 0u; i < n4xN; i = i + ${sa}u) {`,
    `      d0 = d0 + dot(q[qBase + i], cache[kBase + i]);`,
    ...ds.slice(1).map((d, k) => `      ${d} = ${d} + dot(q[qBase + i + ${k + 1}u], cache[kBase + i + ${k + 1}u]);`),
    '    }',
    '    for (var i = n4xN; i < n4; i = i + 1u) {',
    '      d0 = d0 + dot(q[qBase + i], cache[kBase + i]);',
    '    }',
    `    var d = ${pairwiseSum(ds).slice(1, -1)};`,
  ].join('\n');
}

/**
 * THE TRANSPOSED SCORE LOOP, scoreLayout 'dims'. Each aligned group of 32 lanes owns 32 of the
 * chunk's 64 positions, the ones its lanes own for the rest of the iteration (lane sl owns position
 * winLo + c * 64 + sl, and sl = g * 32 + gl). For each position the group's lanes read the K row
 * across the head dimension, vec4 gl, gl + 32 and at head dimension 512 gl + 64 and gl + 96, so
 * every load instruction is 32 lanes by 16 contiguous bytes; each lane dots its slice against the
 * same slice of q; one butterfly closes the sum in every lane; and a select lands it in the one lane
 * whose position it is. Four positions are in flight per step so the loads pipeline. Nothing after
 * this line changes: `d` is the score in the lane that owns the position, exactly as the rows loop
 * leaves it. The clamp keeps every read on a real row and the mask is applied to the RESULT below,
 * as before. The butterfly is in uniform control flow: the loop bounds are constants.
 */
function dimsScoreBlock(): string {
  const U = 4;
  const us = vaRange(U);
  return [
    '    // The transposed score: lanes across the head dimension, positions iterated, one butterfly',
    '    // a position. See dimsScoreBlock in attention.ts.',
    '    let g = (lid >> 5u) & 1u;',
    '    let gl = lid & 31u;',
    `    let rowBase = winLo + c * ${SLICE_LANES}u + g * 32u;`,
    '    var d = 0.0;',
    `    for (var t = 0u; t < 32u; t = t + ${U}u) {`,
    ...us.map((u) => `      let jt${u} = min(rowBase + t + ${u}u, params.kvLen - 1u) * n4;`),
    ...us.map((u) => `      var pt${u} = 0.0;`),
    '      for (var v = 0u; v < n4; v = v + 32u) {',
    '        let qv = q[qBase + gl + v];',
    ...us.map((u) => `        pt${u} = pt${u} + dot(qv, cache[jt${u} + gl + v]);`),
    '      }',
    ...us.map((u) => `      let tot${u} = sgSum32(pt${u});`),
    ...us.map((u) => `      d = select(d, tot${u}, gl == t + ${u}u);`),
    '    }',
  ].join('\n');
}

export function attentionWgsl(reduce: AttnReducePath, geometry: Readonly<AttentionGeometry> = activeGeometry): string {
  assertGeometry(geometry as AttentionGeometry);
  const S = geometry.slices;
  const KS = kvSplitsOf(geometry);
  const VA = geometry.vAccumulators ?? 1;
  const SA = geometry.scoreAccumulators ?? 4;
  const DIMS = reduce === 'subgroup' && (geometry.scoreLayout ?? 'rows') === 'dims';
  // Workers dealing the window between them: the slices of every split. At KS 1 this is S and
  // every expression below collapses to the shipped text, which the gate asserts byte for byte.
  const WK = S * KS;
  const W = SLICE_LANES * S;
  const sliceMask = SLICE_LANES - 1;
  const head = reduce === 'subgroup'
    ? `${SUBGROUP_ENABLE}
${SUBGROUP_BUTTERFLY_WGSL}
${SUBGROUP_BUTTERFLY_MAX_WGSL}
// One (max, sum) pair per 32 lane group: two groups per slice, ${S} slice(s).
var<workgroup> mScratch: array<f32, ${2 * S}>;
var<workgroup> lScratch: array<f32, ${2 * S}>;
`
    : `// The subgroup free trees, reducing within each aligned 64 lane slice so the ${S} slice(s) of a
// workgroup never sum across each other. Both are only ever called unconditionally from uniform
// control flow, which is what makes their barriers legal (ENGINE-PLAN 5.5 rule 5), and both open
// with a barrier so a lane's read of the previous call's result is ordered before the overwrite.
var<workgroup> maxScratch: array<f32, ${W}>;
var<workgroup> sumScratch: array<f32, ${W}>;

fn sliceMax(v: f32, lid: u32) -> f32 {
  workgroupBarrier();
  maxScratch[lid] = v;
  workgroupBarrier();
  for (var s: u32 = ${SLICE_LANES >> 1}u; s > 0u; s = s >> 1u) {
    if ((lid & ${sliceMask}u) < s) {
      maxScratch[lid] = max(maxScratch[lid], maxScratch[lid + s]);
    }
    workgroupBarrier();
  }
  return maxScratch[lid & ~${sliceMask}u];
}

fn sliceSum(v: f32, lid: u32) -> f32 {
  workgroupBarrier();
  sumScratch[lid] = v;
  workgroupBarrier();
  for (var s: u32 = ${SLICE_LANES >> 1}u; s > 0u; s = s >> 1u) {
    if ((lid & ${sliceMask}u) < s) {
      sumScratch[lid] = sumScratch[lid] + sumScratch[lid + s];
    }
    workgroupBarrier();
  }
  return sumScratch[lid & ~${sliceMask}u];
}
`;

  const chunkReduce = reduce === 'subgroup'
    ? `    // Reduce first, in uniform control flow: max then sum, each within its 32 lane group,
    // with the per group sum taken against the group's own max so both butterflies finish
    // before anything is stored. Then the one guarded store block of this iteration, then the
    // uniform merge of the slice's two groups' (max, sum) pairs.
    let mSg = sgMax32(s);
    let pSg = select(0.0, exp(s - mSg), valid);
    let lSg = sgSum32(pSg);
    if ((lid & 31u) == 0u) {
      mScratch[lid >> 5u] = mSg;
      lScratch[lid >> 5u] = lSg;
    }
    workgroupBarrier();
    let m0 = mScratch[slice * 2u];
    let m1 = mScratch[slice * 2u + 1u];
    let mChunk = max(m0, m1);
    // Rebase each group's sum to the chunk max. exp underflows to zero for an all masked group,
    // which is exactly the contribution such a group should make.
    let lChunk = lScratch[slice * 2u] * exp(m0 - mChunk) + lScratch[slice * 2u + 1u] * exp(m1 - mChunk);`
    : `    let mChunk = sliceMax(s, lid);
    let pWg = select(0.0, exp(s - mChunk), valid);
    let lChunk = sliceSum(pWg, lid);`;

  return /* wgsl */ `${head}
// Probabilities of the current chunk, one slot per lane, published for the V accumulation. Each
// slice reads its own 64 slots.
var<workgroup> probScratch: array<f32, ${W}>;
// The per slice (max, sum, accumulator) triples, merged by slice 0 after the loop.
var<workgroup> sliceM: array<f32, ${S}>;
var<workgroup> sliceL: array<f32, ${S}>;
var<workgroup> sliceAcc0: array<vec4<f32>, ${W}>;
var<workgroup> sliceAcc1: array<vec4<f32>, ${W}>;

struct AttnParams {
  // Head dimension in vec4 lanes: 64 sliding, 128 global.
  headDim4: u32,
  // Query heads. 8, but read from the uniform so the shader has no model constant to go stale.
  heads: u32,
  // Query rows in this dispatch. 1 for decode.
  qCount: u32,
  // Absolute position of query row 0.
  qStart: u32,
  // Valid cache positions. Includes the query positions themselves; see the precondition note.
  kvLen: u32,
  // Sliding window length, inclusive of the query position, so the window is [q-511, q] at 512.
  // Zero means full attention (the seven global layers).
  windowLen: u32,
  // V region offset in vec4 lanes inside the packed cache: maxContext * headDim / 4.
  vBase4: u32,
  // Splits sharing this (position, head). 1 on the shipped path, where it is never read.
  kvSplits: u32,
}

@group(0) @binding(0) var<storage, read> q: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> cache: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> out: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> params: AttnParams;

// The trap 2 constant. 1.0, never one over the square root of head_dim. See the header comment.
const ATTN_SCALE: f32 = ${ATTENTION_SCALE.toFixed(1)};

@compute @workgroup_size(${W})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_index) lid: u32,
) {
  let qi = wid.x;
  let h = wid.y;
  let slice = lid >> 6u;
  let sl = lid & ${sliceMask}u;
${KS === 1 ? '' : `  // This workgroup's place among the ${KS} sharing this (position, head). The window is dealt to
  // ${WK} workers, slice by slice within a split, so worker w takes chunks w, w + ${WK}, ...
  let split = wid.z;
  let worker = ${S === 1 ? 'split' : `split * ${S}u + slice`};
`}
  let n4 = params.headDim4;
  let p = params.qStart + qi;
  let qBase = (qi * params.heads + h) * n4;

  // Running softmax state of this slice. mRun at the masked floor and lRun at zero is the empty
  // state; slice 0's first chunk always contains position 0 of the window or the query itself,
  // and the merge below only ever divides by the total, which is positive.
  var mRun = ${MASKED_SCORE};
  var lRun = 0.0;
  // Each lane accumulates the output dims it owns: vec4 lane i and, at head dimension 512,
  // also i + 64. Two registers cover both layer kinds.
  var acc0 = vec4<f32>(0.0);
  var acc1 = vec4<f32>(0.0);

  // WHERE THE WALK STARTS, AND WHAT IT IS A WALK OVER. The sequence used to be dealt to the
  // slices from position 0 whatever the layer's window was, so a sliding layer at kvLen 2048
  // walked 32 chunks to attend 8 and ran the 64 iteration V accumulation once for each of the 24
  // it then masked away. It read the same 1.08 MB at every context length and took four times as
  // long at 2048 as at 512 (docs/ENGINE-PERF.md section 19).
  //
  // The walk is now over the window, not the sequence. winLo is the first position the layer can
  // attend, [p - windowLen + 1, p] for a windowed layer and 0 for a global one, and the slices
  // are dealt 64 position blocks from winLo. So a sliding layer covers its 512 position window in
  // exactly 8 blocks, which is one iteration at 8 slices, at every context length, and the block
  // a slice takes no longer depends on where the window happens to fall against a multiple of 64.
  //
  // WHAT THIS IS AND IS NOT AN IDENTITY ON. For a global layer windowLen is 0, so winLo is 0 and
  // j is c * 64 + sl exactly as before; in decode p is kvLen - 1 and the iteration count is the
  // one the old bound gave, so those seven layers are untouched to the bit. In prefill the count
  // now stops at the query position instead of the sequence end, which drops chunks whose every
  // lane was causally masked: their scores are ${MASKED_SCORE}, their probabilities underflow to
  // zero, their chunk max never beats a running max starting at the same floor, so the state
  // after them is the state before them. That is an identity too.
  //
  // For a windowed layer it is NOT an identity: the same set of positions is attended, but they
  // are dealt to different slices, so the cross slice merge folds a different f32 partition. That
  // is a change of accumulation order under round 3 lead ruling 5, the same class as changing the
  // slice count itself, and it is gated on the full model page under both reduction policies
  // rather than argued.
  //
  // Every lane computes the same perSlice from uniform values, so the barriers inside the loop
  // are still reached by the whole workgroup (section 5.5 rule 6), and a slice whose block runs
  // past the window sees every lane masked exactly as a trailing slice already did.
  let winLo = select(0u, p + 1u - min(p + 1u, params.windowLen), params.windowLen != 0u);
  let span = select(p + 1u, min(params.windowLen, p + 1u), params.windowLen != 0u);
  let spanChunks = (span + ${SLICE_LANES - 1}u) / ${SLICE_LANES}u;
  let perSlice = (spanChunks + ${WK - 1}u) / ${WK}u;
  for (var it = 0u; it < perSlice; it = it + 1u) {
    let c = it * ${WK}u + ${KS === 1 ? 'slice' : 'worker'};${S === 1 && KS > 1 ? `
    // THE DEAD CHUNK SKIP. Chunks are dealt round robin over the ${WK} workers whatever the
    // context, so at 5 chunks workers 5 to ${WK - 1} walk a chunk with every lane masked: every
    // load, every barrier, folded at weight exp(-3e38 - m), which is exactly zero. On the 5070 at
    // 299 positions that is 24 of 64 workgroups, on a dispatch already past the part's 32
    // workgroup knee; at 16 positions it is 56 of 64. A masked chunk leaves every accumulator
    // unchanged to the bit (correction exp(0), probabilities exp(-3e38 - m), both exact), so
    // skipping it changes no number. At one slice c is uniform across the workgroup, so the skip
    // sits above the barriers legally; at more slices c is per slice and there is no skip. The
    // DISPATCH stays (qCount, heads, kvSplits): the runtime's decode step cache holds a step's
    // dispatch fixed across positions and refuses one that moves, so the dead workers launch,
    // skip and write their empty partial rather than never existing.
    if (c >= spanChunks) { continue; }` : ''}
    let j = winLo + c * ${SLICE_LANES}u + sl;
    // Causal plus sliding mask, evaluated as data. j + windowLen > p is the inclusive
    // [p - windowLen + 1, p] window without underflowing unsigned arithmetic. A chunk past the
    // sequence, which a trailing slice sees on its last iteration, masks every lane here.
    let valid = j < params.kvLen && j <= p
      && (params.windowLen == 0u || j + params.windowLen > p);
    // Clamp the read to a real row so an invalid lane reads finite data it then discards.
    let jc = min(j, params.kvLen - 1u);
    let kBase = jc * n4;
    // THE SCORE, ON FOUR ACCUMULATORS. One accumulator makes this a dependency chain as deep as
    // the head dimension in vec4 lanes, 64 sliding and 128 global, every link waiting a full FMA
    // latency on the one before it. Round 5 measured what that costs and what it is not: the
    // global layers ran at 12.0 GB/s against the 43 GB/s the GEMV loops reach on this device, and
    // promoting every slice to its own workgroup, which is the fix if the stall were barriers or
    // scheduling, measured exactly zero on the pair (docs/ENGINE-PERF.md sections 20 and 21). The
    // stall is inside the lane. Four accumulators cut the chain by four and leave four
    // independent FMAs in flight, worth about 1.25 ms a token at the length the app runs.
    //
    // The tail loop is for correctness rather than for this model: headDim4 is 64 or 128 and both
    // are multiples of four, so it runs zero times here. Both bounds come from the uniform, so
    // control flow stays uniform across the workgroup (section 5.5 rule 6).
    //
    // THIS REASSOCIATES THE DOT PRODUCT and therefore moves tokens on near ties. Round 5 built it
    // and reverted it on one such flip, which was the lead applying a gate its own round 2 ruling
    // had already stripped of power: the corrupt a byte control made interview-300 PASS, so a
    // token verdict on a near tie carries no information. Round 6 ruling 6.1.1 in ENGINE-PLAN
    // replaces it: the arbiters are the layer 0 activation gate, the logit top-k overlap and
    // cosine, and a moved token table where a flip counts as a defect only where the reference's
    // own top two gap at that position exceeds 1.0 logit. This kernel is kept under those.
${DIMS ? dimsScoreBlock() : SA === 4 ? `    var d0 = 0.0;
    var d1 = 0.0;
    var d2 = 0.0;
    var d3 = 0.0;
    let n4x4 = (n4 / 4u) * 4u;
    for (var i = 0u; i < n4x4; i = i + 4u) {
      d0 = d0 + dot(q[qBase + i], cache[kBase + i]);
      d1 = d1 + dot(q[qBase + i + 1u], cache[kBase + i + 1u]);
      d2 = d2 + dot(q[qBase + i + 2u], cache[kBase + i + 2u]);
      d3 = d3 + dot(q[qBase + i + 3u], cache[kBase + i + 3u]);
    }
    for (var i = n4x4; i < n4; i = i + 1u) {
      d0 = d0 + dot(q[qBase + i], cache[kBase + i]);
    }
    var d = (d0 + d1) + (d2 + d3);` : scoreBlock(SA)}
    var s = ${MASKED_SCORE};
    if (valid) {
      s = d * ATTN_SCALE;
    }

${chunkReduce}

    let mNew = max(mRun, mChunk);
    let corr = exp(mRun - mNew);
    lRun = lRun * corr + lChunk * exp(mChunk - mNew);

    // Publish this lane's probability under the new running max. Unconditional store, one slot
    // per lane. Invalid lanes publish zero.
    probScratch[lid] = select(0.0, exp(s - mNew), valid);
    workgroupBarrier();

    // Weighted V accumulation in frozen order j = 0..63 within the chunk. The clamp keeps the
    // read on real rows; the zero probability of an invalid slot removes its contribution
    // without multiplying against uninitialised memory.
    acc0 = acc0 * corr;
    acc1 = acc1 * corr;
    let probBase = slice * ${SLICE_LANES}u;
${VA === 1 ? `    for (var t = 0u; t < ${SLICE_LANES}u; t = t + 1u) {
      let jt = min(winLo + c * ${SLICE_LANES}u + t, params.kvLen - 1u);
      let prob = probScratch[probBase + t];
      let vBase = params.vBase4 + jt * n4;
      acc0 = acc0 + prob * cache[vBase + sl];
      if (n4 > ${SLICE_LANES}u) {
        acc1 = acc1 + prob * cache[vBase + sl + ${SLICE_LANES}u];
      }
    }` : [
    `    // ${VA} independent chains, so ${VA} loads and ${VA} multiply adds are in flight instead`,
    '    // of one. Same positions, same probabilities, same values; only the order moves.',
    ...vaRange(VA).map((a) => `    var p${a} = vec4<f32>(0.0);`),
    ...vaRange(VA).map((a) => `    var r${a} = vec4<f32>(0.0);`),
    `    for (var t = 0u; t < ${SLICE_LANES}u; t = t + ${VA}u) {`,
    ...vaRange(VA).flatMap((a) => [
      `      let jt${a} = min(winLo + c * ${SLICE_LANES}u + t + ${a}u, params.kvLen - 1u);`,
      `      let pr${a} = probScratch[probBase + t + ${a}u];`,
      `      let vb${a} = params.vBase4 + jt${a} * n4;`,
      `      p${a} = p${a} + pr${a} * cache[vb${a} + sl];`,
    ]),
    `      if (n4 > ${SLICE_LANES}u) {`,
    ...vaRange(VA).map((a) => `        r${a} = r${a} + pr${a} * cache[vb${a} + sl + ${SLICE_LANES}u];`),
    '      }',
    '    }',
    `    acc0 = acc0 + (${vaRange(VA).map((a) => `p${a}`).join(' + ')});`,
    `    acc1 = acc1 + (${vaRange(VA).map((a) => `r${a}`).join(' + ')});`,
  ].join('\n')}
    mRun = mNew;
    // The next iteration's guarded store rewrites the scratch this one read; the barrier keeps
    // the read before the write, and it sits in uniform control flow.
    workgroupBarrier();
  }

  // The cross slice merge. Every slice publishes its triple, then slice 0 folds them in slice
  // order, which is the one place this kernel's f32 summation order differs from the round 3
  // walk. At one slice the fold is a multiply by exp(0), exactly 1.0, and nothing moves.
  if (sl == 0u) {
    sliceM[slice] = mRun;
    sliceL[slice] = lRun;
  }
  sliceAcc0[lid] = acc0;
  sliceAcc1[lid] = acc1;
  workgroupBarrier();

  if (lid < ${SLICE_LANES}u) {
    var mAll = sliceM[0];
    for (var s2 = 1u; s2 < ${S}u; s2 = s2 + 1u) {
      mAll = max(mAll, sliceM[s2]);
    }
    var lAll = 0.0;
    var o0 = vec4<f32>(0.0);
    var o1 = vec4<f32>(0.0);
    for (var s2 = 0u; s2 < ${S}u; s2 = s2 + 1u) {
      let w = exp(sliceM[s2] - mAll);
      lAll = lAll + sliceL[s2] * w;
      o0 = o0 + sliceAcc0[s2 * ${SLICE_LANES}u + lid] * w;
      o1 = o1 + sliceAcc1[s2 * ${SLICE_LANES}u + lid] * w;
    }
${KS === 1 ? `    let outBase = (qi * params.heads + h) * n4;
    let inv = 1.0 / lAll;
    if (lid < n4) {
      out[outBase + lid] = o0 * inv;
    }
    if (lid + ${SLICE_LANES}u < n4) {
      out[outBase + lid + ${SLICE_LANES}u] = o1 * inv;
    }` : `    // THE PARTIAL, NOT THE ANSWER. This workgroup saw only its ${KS}th of the window, so its
    // running max and sum are partial and the accumulator must NOT be divided here: dividing by a
    // partial sum and re-weighting afterwards is not the same number. attention-decode-merge folds
    // the ${KS} triples with the same online softmax step this kernel already uses across slices.
    let partBase = ((qi * params.heads + h) * params.kvSplits + split) * (n4 + 1u);
    if (lid < n4) {
      out[partBase + lid] = o0;
    }
    if (lid + ${SLICE_LANES}u < n4) {
      out[partBase + lid + ${SLICE_LANES}u] = o1;
    }
    if (lid == 0u) {
      out[partBase + n4] = vec4<f32>(mAll, lAll, 0.0, 0.0);
    }`}
  }
}
`;
}

/**
 * THE MERGE, which is the other half of kvSplits.
 *
 * Each split workgroup wrote an UNNORMALISED triple for its share of the window: the running max in
 * .x of the trailing lane, the running sum in .y, and the accumulator in the lanes before it. This
 * folds them with the same online softmax step the split kernel already runs across its slices:
 * take the max of the maxes, re-weight each split's sum and accumulator by exp(itsMax - theMax),
 * add, and divide once at the end.
 *
 * A split whose blocks all ran past the window carries the masked floor and a sum of zero, so its
 * weight underflows to zero and it contributes nothing. Split 0 always holds the first block of the
 * window, so the total is always positive and the single division is always safe. That is the same
 * argument the cross slice merge already rests on.
 *
 * 64 lanes, one workgroup per (position, head), because the output is at most 128 vec4 lanes and
 * each lane owns lane i and i + 64 exactly as the split kernel's store block does.
 */
export function attentionMergeWgsl(): string {
  return /* wgsl */ `struct MergeParams {
  headDim4: u32,
  heads: u32,
  qCount: u32,
  qStart: u32,
  kvLen: u32,
  windowLen: u32,
  vBase4: u32,
  kvSplits: u32,
}

@group(0) @binding(0) var<storage, read> part: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> out: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: MergeParams;

@compute @workgroup_size(${SLICE_LANES})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_index) lid: u32,
) {
  let qi = wid.x;
  let h = wid.y;
  let n4 = params.headDim4;
  let stride = n4 + 1u;
  let base = (qi * params.heads + h) * params.kvSplits * stride;

  // Every lane walks the same splits and computes the same mAll, so this loop and the next are
  // uniform across the workgroup and need no barrier.
  var mAll = ${MASKED_SCORE};
  for (var sp = 0u; sp < params.kvSplits; sp = sp + 1u) {
    mAll = max(mAll, part[base + sp * stride + n4].x);
  }
  var lAll = 0.0;
  var o0 = vec4<f32>(0.0);
  var o1 = vec4<f32>(0.0);
  for (var sp = 0u; sp < params.kvSplits; sp = sp + 1u) {
    let sbase = base + sp * stride;
    let ml = part[sbase + n4];
    let w = exp(ml.x - mAll);
    lAll = lAll + ml.y * w;
    o0 = o0 + part[sbase + lid] * w;
    if (n4 > ${SLICE_LANES}u) {
      o1 = o1 + part[sbase + lid + ${SLICE_LANES}u] * w;
    }
  }
  let outBase = (qi * params.heads + h) * n4;
  let inv = 1.0 / lAll;
  if (lid < n4) {
    out[outBase + lid] = o0 * inv;
  }
  if (lid + ${SLICE_LANES}u < n4) {
    out[outBase + lid + ${SLICE_LANES}u] = o1 * inv;
  }
}
`;
}

export const ATTENTION_MERGE_WGSL = attentionMergeWgsl();

/**
 * The split path as arithmetic, so the fold can be proved in Node without a GPU.
 *
 * It deals the window to `splits * slices` workers exactly as the WGSL does, keeps each SPLIT's
 * own (max, sum, accumulator) triple, and then folds the triples the way attentionMergeWgsl does.
 * The gate asserts this equals `attentionOracle` to within f32 reassociation, which is the property
 * the whole change rests on: splitting the window changes the summation order and NOTHING else.
 */
export function attentionSplitOracle(
  q: Float32Array,
  k: Float32Array,
  v: Float32Array,
  shape: AttentionShape,
  splits: number,
  slices = 8,
): Float32Array {
  const { heads, headDim, qCount, qStart, kvLen, window } = shape;
  const scale = shape.scale ?? ATTENTION_SCALE;
  const workers = splits * slices;
  const out = new Float32Array(qCount * heads * headDim);
  for (let qi = 0; qi < qCount; qi += 1) {
    const p = qStart + qi;
    const winLo = window > 0 ? Math.max(0, p - window + 1) : 0;
    const span = window > 0 ? Math.min(window, p + 1) : p + 1;
    const spanChunks = Math.ceil(span / SLICE_LANES);
    for (let h = 0; h < heads; h += 1) {
      const qBase = (qi * heads + h) * headDim;
      // One triple per split, folded from the slices that belong to it.
      const sm = new Float64Array(splits).fill(-Infinity);
      const sl = new Float64Array(splits);
      const sacc = new Float64Array(splits * headDim);
      for (let w = 0; w < workers; w += 1) {
        const split = Math.floor(w / slices);
        for (let c = w; c < spanChunks; c += workers) {
          for (let t = 0; t < SLICE_LANES; t += 1) {
            const j = winLo + c * SLICE_LANES + t;
            if (j >= kvLen || j > p || (window > 0 && j + window <= p)) continue;
            let d = 0;
            for (let i = 0; i < headDim; i += 1) d += q[qBase + i] * k[j * headDim + i];
            const sc = d * scale;
            const mNew = Math.max(sm[split]!, sc);
            const corr = Math.exp(sm[split]! - mNew);
            sl[split] = sl[split]! * corr + Math.exp(sc - mNew);
            const accBase = split * headDim;
            for (let i = 0; i < headDim; i += 1) {
              sacc[accBase + i] = sacc[accBase + i]! * corr + Math.exp(sc - mNew) * v[j * headDim + i]!;
            }
            sm[split] = mNew;
          }
        }
      }
      let mAll = -Infinity;
      for (let sp = 0; sp < splits; sp += 1) mAll = Math.max(mAll, sm[sp]!);
      let lAll = 0;
      const o = new Float64Array(headDim);
      for (let sp = 0; sp < splits; sp += 1) {
        const wgt = Math.exp(sm[sp]! - mAll);
        if (!Number.isFinite(wgt)) continue;
        lAll += sl[sp]! * wgt;
        for (let i = 0; i < headDim; i += 1) o[i] += sacc[sp * headDim + i]! * wgt;
      }
      const outBase = (qi * heads + h) * headDim;
      for (let i = 0; i < headDim; i += 1) out[outBase + i] = o[i]! / lAll;
    }
  }
  return out;
}

/** The default geometry builds, which are what the Node checks lint and the fallback table serves. */
export const ATTENTION_WGSL = attentionWgsl('subgroup', DEFAULT_ATTENTION_GEOMETRY);
export const ATTENTION_FALLBACK_WGSL = attentionWgsl('workgroup', DEFAULT_ATTENTION_GEOMETRY);

// ---------------------------------------------------------------------------------------------
// The CPU oracle.
// ---------------------------------------------------------------------------------------------

export interface AttentionShape {
  heads: number;
  headDim: number;
  /** Query rows. 1 for the decode shape. */
  qCount: number;
  /** Absolute position of query row 0. */
  qStart: number;
  /** Valid KV rows, covering every query position. */
  kvLen: number;
  /** Sliding window length inclusive of the query position; 0 for the global layers. */
  window: number;
  /** Defaults to ATTENTION_SCALE. Overridable only so the trap test can prove the wrong value collapses. */
  scale?: number;
}

/**
 * Reference attention: full f64 scores, max subtracted softmax, f64 weighted V sum, rounded to
 * f32 once at the end. Deliberately not a mirror of the shader's chunked reduction; the shader
 * is compared against this within a stated tolerance and against the HF reference capture by
 * cosine, per ENGINE-PLAN section 7's discipline.
 *
 * Layouts match the reference capture and the kernel:
 *   q    [qCount, heads, headDim]
 *   k, v [kvLen, headDim], the single KV head all eight query heads share
 *   out  [qCount, heads, headDim]
 */
export function attentionOracle(
  q: Float32Array,
  k: Float32Array,
  v: Float32Array,
  shape: AttentionShape,
): Float32Array {
  const { heads, headDim, qCount, qStart, kvLen, window } = shape;
  const scale = shape.scale ?? ATTENTION_SCALE;
  if (q.length < qCount * heads * headDim) throw new Error('attentionOracle: q too short');
  if (k.length < kvLen * headDim || v.length < kvLen * headDim) {
    throw new Error('attentionOracle: k or v too short');
  }
  if (qStart + qCount > kvLen) {
    throw new Error('attentionOracle: query positions extend past the valid cache');
  }
  const out = new Float32Array(qCount * heads * headDim);
  const scores = new Float64Array(kvLen);
  for (let qi = 0; qi < qCount; qi += 1) {
    const p = qStart + qi;
    const lo = window > 0 ? Math.max(0, p - window + 1) : 0;
    for (let h = 0; h < heads; h += 1) {
      const qBase = (qi * heads + h) * headDim;
      let m = -Infinity;
      for (let j = lo; j <= p; j += 1) {
        let d = 0;
        const kBase = j * headDim;
        for (let i = 0; i < headDim; i += 1) d += q[qBase + i] * k[kBase + i];
        const s = d * scale;
        scores[j] = s;
        if (s > m) m = s;
      }
      let l = 0;
      for (let j = lo; j <= p; j += 1) {
        scores[j] = Math.exp(scores[j] - m);
        l += scores[j];
      }
      const outBase = (qi * heads + h) * headDim;
      for (let i = 0; i < headDim; i += 1) {
        let a = 0;
        for (let j = lo; j <= p; j += 1) a += scores[j] * v[j * headDim + i];
        out[outBase + i] = a / l;
      }
    }
  }
  return out;
}

/**
 * The decode shape as its own entry point: one query position against the cache. Exactly
 * attentionOracle at qCount 1, and asserted equal to the matching prefill row in the checks,
 * which is the property that lets the app trust prefix reuse: a token decoded after a rewind
 * must see the same attention a full prefill would have given it.
 */
export function attentionDecodeOracle(
  q: Float32Array,
  k: Float32Array,
  v: Float32Array,
  shape: Omit<AttentionShape, 'qCount' | 'qStart'> & { qPos: number },
): Float32Array {
  const { qPos, ...rest } = shape;
  return attentionOracle(q, k, v, { ...rest, qCount: 1, qStart: qPos });
}

// ---------------------------------------------------------------------------------------------
// Registry entries.
// ---------------------------------------------------------------------------------------------

/** The params block, exported so the scheduler can restage it in place per dispatch. */
export function attnParams(p: {
  headDim4: number; heads: number; qCount: number; qStart: number;
  kvLen: number; windowLen: number; vBase4: number; kvSplits?: number;
}): ArrayBuffer {
  const words = new ArrayBuffer(32);
  const u = new Uint32Array(words);
  u[0] = p.headDim4;
  u[1] = p.heads;
  u[2] = p.qCount;
  u[3] = p.qStart;
  u[4] = p.kvLen;
  u[5] = p.windowLen;
  u[6] = p.vBase4;
  u[7] = Math.max(1, Math.trunc(p.kvSplits ?? 1));
  return words;
}

/** `split` is the kernel's identity, not the geometry's: only `attention-decode-split` passes true. */
const bindAttention = (split: boolean) => (input: KernelBindInput): KernelBindResult => bindAttentionImpl(input, split);

function bindAttentionImpl(input: KernelBindInput, split: boolean): KernelBindResult {
  const { device, inputs, output, params } = input;
  const headDim = params.headDim | 0;
  const heads = params.heads | 0;
  const qCount = params.qCount | 0;
  const qStart = params.qStart | 0;
  const kvLen = params.kvLen | 0;
  const window = params.window | 0;
  const maxContext = params.maxContext | 0;
  if (headDim <= 0 || headDim % 4 !== 0) {
    throw new Error(`attention needs params.headDim a positive multiple of 4, got ${headDim}`);
  }
  if (headDim / 4 > 2 * SLICE_LANES) {
    throw new Error(`attention accumulators cover headDim up to ${8 * SLICE_LANES}, got ${headDim}`);
  }
  if (heads <= 0 || qCount <= 0 || kvLen <= 0) {
    throw new Error('attention needs positive params.heads, params.qCount and params.kvLen');
  }
  if (qStart + qCount > kvLen) {
    throw new Error('attention: query positions extend past kvLen; store KV before attending');
  }
  const qBuf = inputs.q;
  const cacheBuf = inputs.cache;
  if (!qBuf || !cacheBuf) throw new Error('attention needs inputs named q and cache');

  const layout = kernelLayout(input, 'attention', () => device.createBindGroupLayout({
    label: 'attention',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  }));
  // The split rides on z, so an unsplit dispatch is the grid it always was and a split one is the
  // same grid with a third dimension. The kernel reads its own split from wid.z. The count is the
  // geometry's, NOT the number of chunks this position has: runtime.ts caches a resolved decode
  // step across positions and asserts its dispatch does not move, and a dispatch sized to the
  // context moves every token. Dead splits are cheap instead (the skip in attentionWgsl).
  const kvSplits = split ? kvSplitsOf(activeGeometry) : 1;
  const uniform = kernelUniform(input, 'attention params', attnParams({
    headDim4: headDim / 4,
    heads,
    qCount,
    qStart,
    kvLen,
    windowLen: window,
    vBase4: (maxContext * headDim) / 4,
    kvSplits,
  }));

  return {
    layout,
    buffers: [qBuf, cacheBuf, output, uniform.binding],
    dispatch: [qCount, heads, kvSplits],
    dispose: uniform.dispose,
  };
}

// The tolerance argument, shared by every case below. The expected buffers come from the HF
// reference forward running in bf16 (the checkpoint's dtype under dtype="auto"), while this
// kernel computes in f32 from inputs that are the exact f32 image of those bf16 tensors. The
// difference is therefore dominated by bf16's 8 bit mantissa inside the reference matmuls, about
// 1 part in 256 per product, damped by averaging over 256 or 512 wide dots and a convex softmax
// combination. Per element gates are meaningless at that floor; cosine is the right instrument
// (ENGINE-PLAN gate 1), and the k-attention check measures the f64 oracle against the same
// capture at better than 0.999 before these numbers are trusted. 0.999 here is the floor with
// margin, not a target; the measured values print in the harness and drifting DOWN from a
// measured baseline is the actual alarm.
const ATTN_MIN_COSINE = 0.999;

/**
 * The per element gate to go with it. The harness's per element tolerance defaults to ZERO on
 * both axes when a case says nothing, and these cases originally said nothing, which the
 * argument above never intended; round 1 never noticed because the kernels died at compile
 * before producing a number. First real run on the M1 (subgroup variant, Metal) measured maxAbs
 * 8.1e-6 to 1.66e-5 across all four cases at cosine 1.000000000, which is the bf16 capture floor
 * the paragraph above predicts. 1e-4 is that with a 6x margin: far above the floor's noise, far
 * below anything a wrong slot, stride or scale produces, and the cosine gate still owns
 * arrangement errors.
 */
const ATTN_TOL_ABS = 1e-4;

/**
 * The fold of a split attention dispatch, as a registry kernel.
 *
 * Present only when a device profile asks for `kvSplits`. The split build writes one UNNORMALISED
 * softmax partial per (position, head, split), each carrying its own running max and weight sum in
 * the lane past the accumulator, and this takes the max of the maxes, re-weights each partial by
 * exp(m - mAll), sums, and divides once. That order is the whole correctness argument: a partial
 * normalised by its own weight sum cannot be added to another one.
 *
 * WHY IT IS WORTH A SECOND DISPATCH ON SOME MACHINES AND NOT OTHERS. The dispatch is
 * (qCount, heads, kvSplits), so unsplit it is 8 workgroups. The M1 has 8 cores and its occupancy
 * knee is at exactly 8 workgroups, so a split there can only add this pass. The 5070 has 48 SMs
 * and its knee is at 32, and its winning configurations move the SAME work onto more and smaller
 * workgroups: 2.74x and 2.99x at slices 1 and kvSplits 8
 * (lab-results/5070-attention-split-sep04.json). That is why it is a profile field and not a
 * default, and why this kernel exists but nothing yet turns it on.
 */
export const attentionMergeKernel: Kernel = {
  name: 'attention-decode-merge',
  get wgsl(): string { return ATTENTION_MERGE_WGSL; },
  entry: 'main',
  note:
    'The fold of a split attention decode. 64 lanes a workgroup, one per (position, head). No '
    + 'subgroup operation, so the one build serves both reduce policies and it is deliberately '
    + 'absent from the fallback table.',
  cases: [
    {
      name: 'fold-2heads-2splits',
      inputs: { part: 'kattn.merge.part' },
      expected: 'kattn.merge.expected',
      params: { headDim: 8, heads: 2, qCount: 1, kvSplits: 2 },
      tolAbs: 1e-6,
      minCosine: ATTN_MIN_COSINE,
      note:
        'Two partials a head with DIFFERENT running maxes, so the re-weighting has work to do. A '
        + 'shader that summed partials which had each normalised themselves would fail this, '
        + 'which is the one thing about the split that cannot be argued from the unsplit answer.',
    },
  ],
  bind: (input: KernelBindInput): KernelBindResult => {
    const { device, inputs, output, params } = input;
    const headDim = params.headDim | 0;
    const heads = params.heads | 0;
    const qCount = params.qCount | 0;
    const kvSplits = params.kvSplits | 0;
    if (headDim <= 0 || headDim % 4 !== 0) {
      throw new Error(`attention-decode-merge needs params.headDim a positive multiple of 4, got ${headDim}`);
    }
    if (heads <= 0 || qCount <= 0) {
      throw new Error('attention-decode-merge needs positive params.heads and params.qCount');
    }
    if (kvSplits < 2) {
      throw new Error(`attention-decode-merge only exists on a split dispatch, got kvSplits ${kvSplits}`);
    }
    const part = inputs.part;
    if (!part) throw new Error('attention-decode-merge needs an input named part');
    const layout = kernelLayout(input, 'attention-decode-merge', () => device.createBindGroupLayout({
      label: 'attention-decode-merge',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    }));
    // The fold reads headDim4, heads and kvSplits; the window and the positions are the split
    // build's business and it has already finished with them.
    const uniform = kernelUniform(input, 'attention-decode-merge params', attnParams({
      headDim4: headDim / 4, heads, qCount, qStart: 0, kvLen: 1, windowLen: 0, vBase4: 0, kvSplits,
    }));
    return {
      layout,
      buffers: [part, output, uniform.binding],
      dispatch: [qCount, heads, 1],
      dispose: uniform.dispose,
    };
  },
};

export const attentionDecodeKernel: Kernel = {
  name: 'attention-decode',
  // Read at build time against the active attention geometry, so the performance rig's slice
  // sweep reaches the pipeline store through the registry like any other build. The default
  // geometry text is the ATTENTION_WGSL constant above, byte for byte.
  get wgsl(): string { return attentionWgsl('subgroup', unsplitAttentionGeometry(activeGeometry)); },
  entry: 'main',
  note:
    'K8, decode shape: one query position against the packed KV cache, per layer kind via '
    + 'params (window 512 at head_dim 256, unwindowed at 512). Scale 1.0 exactly, trap 2. The '
    + 'sequence is dealt to 64 lane slices of one workgroup and their softmax partials merged in '
    + 'f32, ENGINE-PLAN round 3 ruling 5, receipted as parity-r4.',
  cases: [
    {
      name: 'decode-sliding-256-last',
      inputs: { q: 'kattn.decode.layer00.q', cache: 'kattn.cache.layer00' },
      expected: 'kattn.decode.layer00.out',
      params: {
        headDim: 256, heads: 8, qCount: 1, qStart: 15, kvLen: 16,
        window: 512, maxContext: 16,
      },
      tolAbs: ATTN_TOL_ABS,
      minCosine: ATTN_MIN_COSINE,
      note: 'Last prompt position of the short-question capture as a decode step, layer 0.',
    },
    {
      name: 'decode-global-512-last',
      inputs: { q: 'kattn.decode.layer04.q', cache: 'kattn.cache.layer04' },
      expected: 'kattn.decode.layer04.out',
      params: {
        headDim: 512, heads: 8, qCount: 1, qStart: 15, kvLen: 16,
        window: 0, maxContext: 16,
      },
      tolAbs: ATTN_TOL_ABS,
      minCosine: ATTN_MIN_COSINE,
      note: 'Same position on the first full attention layer: head_dim 512, no window.',
    },
  ],
  bind: bindAttention(false),
};

/**
 * The split KV sibling of attention-decode, and the only attention kernel allowed to dispatch a z
 * greater than one. The plan emits this name in place of `attention-decode` exactly when it also
 * emits ROLE_ATTN_MERGE, so the partials it writes are always folded and prefill can never be
 * split by a decode profile.
 *
 * Its case is the decode entry's, run at the shipped geometry where kvSplits is 1 and the split
 * build collapses byte for byte to the text attention-decode compiles. The splits at 2, 4 and 8 are
 * proved directly in scripts/engine-check/k-attention.mjs, which drives the WGSL at an explicit
 * geometry, because a registry case runs at whatever geometry happens to be active.
 */
export const attentionDecodeSplitKernel: Kernel = {
  ...attentionDecodeKernel,
  name: ATTENTION_DECODE_SPLIT_KERNEL,
  get wgsl(): string { return attentionWgsl('subgroup', activeGeometry); },
  note: 'The split KV decode attention. Writes partials; the plan folds them.',
  bind: bindAttention(true),
};

export const attentionPrefillKernel: Kernel = {
  name: 'attention-prefill',
  get wgsl(): string { return attentionWgsl('subgroup', unsplitAttentionGeometry(activeGeometry)); },
  entry: 'main',
  note:
    'K8, prefill shape: a chunk of query positions with causal plus sliding masks computed '
    + 'structurally from positions. Same maths as decode this round; the section 5.4 GEMM '
    + 'retile is the named next step for TTFT.',
  cases: [
    {
      name: 'prefill-sliding-256',
      inputs: { q: 'attn.short-question.layer00.q', cache: 'kattn.cache.layer00' },
      expected: 'attn.short-question.layer00.out',
      params: {
        headDim: 256, heads: 8, qCount: 16, qStart: 0, kvLen: 16,
        window: 512, maxContext: 16,
      },
      tolAbs: ATTN_TOL_ABS,
      minCosine: ATTN_MIN_COSINE,
      note: 'Whole short-question prompt against the HF eager attention output, layer 0.',
    },
    {
      name: 'prefill-global-512',
      inputs: { q: 'attn.short-question.layer04.q', cache: 'kattn.cache.layer04' },
      expected: 'attn.short-question.layer04.out',
      params: {
        headDim: 512, heads: 8, qCount: 16, qStart: 0, kvLen: 16,
        window: 0, maxContext: 16,
      },
      tolAbs: ATTN_TOL_ABS,
      minCosine: ATTN_MIN_COSINE,
      note: 'Layer 4: partial RoPE arrived baked into q and k, head_dim 512, unwindowed.',
    },
  ],
  bind: bindAttention(false),
};
