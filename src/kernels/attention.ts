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
export function attentionWgsl(reduce: AttnReducePath, geometry: Readonly<AttentionGeometry> = activeGeometry): string {
  assertGeometry(geometry as AttentionGeometry);
  const S = geometry.slices;
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
  pad0: u32,
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
  let perSlice = (spanChunks + ${S - 1}u) / ${S}u;
  for (var it = 0u; it < perSlice; it = it + 1u) {
    let c = it * ${S}u + slice;
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
    var d0 = 0.0;
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
    var d = (d0 + d1) + (d2 + d3);
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
    for (var t = 0u; t < ${SLICE_LANES}u; t = t + 1u) {
      let jt = min(winLo + c * ${SLICE_LANES}u + t, params.kvLen - 1u);
      let prob = probScratch[probBase + t];
      let vBase = params.vBase4 + jt * n4;
      acc0 = acc0 + prob * cache[vBase + sl];
      if (n4 > ${SLICE_LANES}u) {
        acc1 = acc1 + prob * cache[vBase + sl + ${SLICE_LANES}u];
      }
    }
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
    let outBase = (qi * params.heads + h) * n4;
    let inv = 1.0 / lAll;
    if (lid < n4) {
      out[outBase + lid] = o0 * inv;
    }
    if (lid + ${SLICE_LANES}u < n4) {
      out[outBase + lid + ${SLICE_LANES}u] = o1 * inv;
    }
  }
}
`;
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
  kvLen: number; windowLen: number; vBase4: number;
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
  u[7] = 0;
  return words;
}

function bindAttention(input: KernelBindInput): KernelBindResult {
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
  const uniform = kernelUniform(input, 'attention params', attnParams({
    headDim4: headDim / 4,
    heads,
    qCount,
    qStart,
    kvLen,
    windowLen: window,
    vBase4: (maxContext * headDim) / 4,
  }));

  return {
    layout,
    buffers: [qBuf, cacheBuf, output, uniform.binding],
    dispatch: [qCount, heads, 1],
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

export const attentionDecodeKernel: Kernel = {
  name: 'attention-decode',
  // Read at build time against the active attention geometry, so the performance rig's slice
  // sweep reaches the pipeline store through the registry like any other build. The default
  // geometry text is the ATTENTION_WGSL constant above, byte for byte.
  get wgsl(): string { return attentionWgsl('subgroup'); },
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
  bind: bindAttention,
};

export const attentionPrefillKernel: Kernel = {
  name: 'attention-prefill',
  get wgsl(): string { return attentionWgsl('subgroup'); },
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
  bind: bindAttention,
};
