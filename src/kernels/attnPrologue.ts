// Written from ./rmsNorm.ts, ./rope.ts and ./kvStore.ts, the three kernels this file fuses and
// whose arithmetic it repeats expression for expression, docs/ENGINE-PLAN.md section 5.5, and the
// WGSL specification. No vendored bundle, no extracted kernel and no third party engine source
// was read.
//
// SPDX-License-Identifier: Apache-2.0
//
// THE ATTENTION PROLOGUE, FUSED.
//
// WHY. On a decode step the attention block spent six dispatches a producer layer and two a
// consumer layer between the projections and the attention itself: q_norm, k_norm, v_norm,
// rope q, rope k and the KV store. On the RTX 5070 each is a two to four microsecond kernel whose
// whole cost is launch, drain and one workgroup's latency over almost no bytes: 172 dispatches a
// token, about half a millisecond of a 3.7 ms token
// (lab-results/5070-dispatch-headroom-sep05.json). The two kernels here run the same arithmetic
// in one dispatch a consumer layer and two a producer layer. Prefill keeps the separate kernels:
// its chunks are wide enough that the dispatch count is not where its time goes.
//
// THE CONTRACT IS EXPRESSION FOR EXPRESSION THE THREE KERNELS. The sum of squares, the
// butterfly, the slot order, the workgroup width and the inverseSqrt are rmsNorm.ts's own text,
// imported through normReducePieces rather than copied, so they cannot drift. The apply is the
// norm kernel's two multiplies, landed in workgroup memory instead of a slot. The rotation is
// rope.ts's split half form on the same scalars, with the table row found the same way. The
// store is kvStore.ts's row placement. What changes is where the intermediate lives and how
// many times the GPU is asked to start something; no number is computed in a different order.
//
// TWO KERNELS, NOT ONE. The q rows are (position, head) rows through a weighted norm and a
// rotation into the qr slot; the k and v rows are one row each per position, k through the
// weighted norm and the rotation into the cache's K region, v through the weightless norm into
// its V region. A producer layer runs both, a consumer layer only the first, exactly as the plan
// ran the separate kernels.
import type { Kernel, KernelBindInput, KernelBindResult } from './registry';
import { kernelLayout, kernelUniform } from './binding';
import { normReducePieces, rmsNormOracle } from './rmsNorm';
import type { ReducePath } from './rmsNorm';
import { ropeOracle } from './rope';
import { kvStoreOracle } from './kvStore';
import type { KvStoreShape } from './kvStore';
import { gemvGeometry, kSplitsOf } from './qgemv';
import { applySrq } from '../quant';

/** The widest head this model has (kernels/layerGeometry.ts: 256 sliding, 512 global), the row scratch's size. */
export const MAX_HEAD_DIM = 512;

export const Q_NORM_ROPE_KERNEL = 'q-norm-rope';
export const KV_PROLOGUE_KERNEL = 'kv-norm-rope-store';
/**
 * The fold variants: the same kernels reading their q, or k and v, as the PARTIALS a split K
 * projection wrote (qgemv.ts GEMV_SPLIT_KERNEL, dst[row * splits + split], each carrying its
 * row scale and no snap), summed in the merge kernel's order and snapped with the projection's
 * own output scale before the norm sees them. The split count is read from the live 4-bit
 * geometry at build time, as the split GEMV reads it, and the plan names the fold variant
 * exactly where it names the split projection, so a split dispatch has its fold by construction.
 */
export const Q_NORM_ROPE_FOLD_KERNEL = 'q-norm-rope-fold';
export const KV_PROLOGUE_FOLD_KERNEL = 'kv-norm-rope-store-fold';

/** The fold's read: `splits` partials of element `e` summed in ascending order, then the snap the split GEMV left out. */
function foldWgsl(name: string, buffer: string, scale: string, splits: number): string {
  return `fn ${name}Snap(v: f32) -> f32 {
  if (params.${scale} == 0.0) {
    return v;
  }
  return clamp(round(v / params.${scale}), -128.0, 127.0) * params.${scale};
}
fn ${name}At(e: u32) -> f32 {
  var t = 0.0;
  for (var s = 0u; s < ${splits}u; s = s + 1u) {
    t = t + ${buffer}[e * ${splits}u + s];
  }
  return ${name}Snap(t);
}
fn ${name}Vec(j: u32) -> vec4<f32> {
  return vec4<f32>(${name}At(4u * j), ${name}At(4u * j + 1u), ${name}At(4u * j + 2u), ${name}At(4u * j + 3u));
}
`;
}

/**
 * q_norm then rope q, one workgroup per (position, head) row. `dst` is the qr slot, row major by
 * row exactly as rope.ts writes it.
 */
export function qNormRopeWgsl(reduce: ReducePath, splits = 1): string {
  const { W, head, reduce: reduceText } = normReducePieces(reduce);
  const fold = splits > 1;
  const srcRead = fold ? 'srcVec(base + i)' : 'src[base + i]';
  return /* wgsl */ `${head}
// The normalized row, as the scalars the rotation pairs up (i with i + pairCount).
var<workgroup> nrm: array<f32, ${MAX_HEAD_DIM}>;

struct QNormRopeParams {
  // Row width in vec4 lanes: 64 sliding, 128 global. The pair count is twice this.
  vec4Count: u32,
  // (position, head) rows in the dispatch. Carried for the caller to validate against, not read:
  // the dispatch is exactly one workgroup per row.
  rowCount: u32,
  eps: f32,
  // Rows sharing one position's table row: the head count, as rope.ts takes it.
  headsPerPosition: u32,${fold ? `
  // The q projection's output_activation_scale, the snap its split build left to this fold.
  outScale: f32,
  pad0: u32,
  pad1: u32,
  pad2: u32,` : ''}
}

@group(0) @binding(0) var<storage, read> src: array<${fold ? 'f32' : 'vec4<f32>'}>;
@group(0) @binding(1) var<storage, read> gain: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> cosTab: array<f32>;
@group(0) @binding(3) var<storage, read> sinTab: array<f32>;
@group(0) @binding(4) var<storage, read_write> dst: array<f32>;
@group(0) @binding(5) var<uniform> params: QNormRopeParams;
${fold ? foldWgsl('src', 'src', 'outScale', splits) : ''}
@compute @workgroup_size(${W})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_index) lid: u32,
) {
  let row = wid.x;
  let n4 = params.vec4Count;
  let base = row * n4;

  var acc = 0.0;
  for (var i = lid; i < n4; i = i + ${W}u) {
    let v = ${srcRead};
    acc = acc + dot(v, v);
  }

${reduceText}

  let inv = inverseSqrt(total / f32(n4 * 4u) + params.eps);

  // rms-norm's apply, the same two multiplies, landed in workgroup memory instead of a slot.
  for (var i = lid; i < n4; i = i + ${W}u) {
    let v = ${srcRead} * inv * gain[i];
    nrm[4u * i] = v.x;
    nrm[4u * i + 1u] = v.y;
    nrm[4u * i + 2u] = v.z;
    nrm[4u * i + 3u] = v.w;
  }
  workgroupBarrier();

  // rope's split half form over the normalized row; the table row is the position's.
  let p = n4 * 2u;
  let rowBase = row * p * 2u;
  let tab = (row / params.headsPerPosition) * p;
  for (var i = lid; i < p; i = i + ${W}u) {
    let lo = nrm[i];
    let hi = nrm[i + p];
    let c = cosTab[tab + i];
    let s = sinTab[tab + i];
    dst[rowBase + i] = lo * c - hi * s;
    dst[rowBase + i + p] = hi * c + lo * s;
  }
}
`;
}

/**
 * k_norm, v_norm, rope k and the KV store, two workgroups per position: wid.y 0 takes the K row
 * through the weighted norm and the rotation into the cache's K region, wid.y 1 takes the V row
 * through the weightless norm into the V region. Both are workgroup ids, so every branch on them
 * is uniform and the barrier between the apply and the rotation stays in uniform control flow.
 */
export function kvPrologueWgsl(reduce: ReducePath, splits = 1): string {
  const { W, head, reduce: reduceText } = normReducePieces(reduce);
  const fold = splits > 1;
  const kRead = fold ? 'kVec(base + i)' : 'k[base + i]';
  const vRead = fold ? 'vVec(base + i)' : 'v[base + i]';
  return /* wgsl */ `${head}
var<workgroup> nrm: array<f32, ${MAX_HEAD_DIM}>;

struct KvPrologueParams {
  // Row width in vec4 lanes: 64 sliding, 128 global.
  vec4Count: u32,
  // Positions in the dispatch; the dispatch is (tokenCount, 2, 1).
  tokenCount: u32,
  eps: f32,
  // Absolute position of row 0, the decode position or the prefill cursor.
  startPos: u32,
  // V region offset in f32 elements inside the packed cache: maxContext * headDim (kv.ts).
  vBase: u32,${fold ? `
  // The k and v projections' output_activation_scales, the snaps their split builds left here.
  kOutScale: f32,
  vOutScale: f32,` : `
  pad0: u32,
  pad1: u32,`}
  pad2: u32,
}

@group(0) @binding(0) var<storage, read> k: array<${fold ? 'f32' : 'vec4<f32>'}>;
@group(0) @binding(1) var<storage, read> v: array<${fold ? 'f32' : 'vec4<f32>'}>;
@group(0) @binding(2) var<storage, read> gain: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> cosTab: array<f32>;
@group(0) @binding(4) var<storage, read> sinTab: array<f32>;
@group(0) @binding(5) var<storage, read_write> cache: array<f32>;
@group(0) @binding(6) var<uniform> params: KvPrologueParams;
${fold ? foldWgsl('k', 'k', 'kOutScale', splits) + foldWgsl('v', 'v', 'vOutScale', splits) : ''}
@compute @workgroup_size(${W})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_index) lid: u32,
) {
  let row = wid.x;
  let isK = wid.y == 0u;
  let n4 = params.vec4Count;
  let base = row * n4;

  var acc = 0.0;
  for (var i = lid; i < n4; i = i + ${W}u) {
    var x: vec4<f32>;
    if (isK) { x = ${kRead}; } else { x = ${vRead}; }
    acc = acc + dot(x, x);
  }

${reduceText}

  let inv = inverseSqrt(total / f32(n4 * 4u) + params.eps);

  // rms-norm's apply on the K row, rms-norm-weightless's on the V row, into workgroup memory.
  for (var i = lid; i < n4; i = i + ${W}u) {
    var y: vec4<f32>;
    if (isK) { y = ${kRead} * inv * gain[i]; } else { y = ${vRead} * inv; }
    nrm[4u * i] = y.x;
    nrm[4u * i + 1u] = y.y;
    nrm[4u * i + 2u] = y.z;
    nrm[4u * i + 3u] = y.w;
  }
  workgroupBarrier();

  let width = n4 * 4u;
  let pos = params.startPos + row;
  if (isK) {
    // rope k, one table row per position, written straight into the cache's K row.
    let p = n4 * 2u;
    let tab = row * p;
    let kDst = pos * width;
    for (var i = lid; i < p; i = i + ${W}u) {
      let lo = nrm[i];
      let hi = nrm[i + p];
      let c = cosTab[tab + i];
      let s = sinTab[tab + i];
      cache[kDst + i] = lo * c - hi * s;
      cache[kDst + i + p] = hi * c + lo * s;
    }
  } else {
    let vDst = params.vBase + pos * width;
    for (var i = lid; i < width; i = i + ${W}u) {
      cache[vDst + i] = nrm[i];
    }
  }
}
`;
}

export const Q_NORM_ROPE_WGSL = qNormRopeWgsl('subgroup');
export const Q_NORM_ROPE_FALLBACK_WGSL = qNormRopeWgsl('workgroup');
export const KV_PROLOGUE_WGSL = kvPrologueWgsl('subgroup');
export const KV_PROLOGUE_FALLBACK_WGSL = kvPrologueWgsl('workgroup');
/** The fold variants read the live 4-bit split at build time, like the split GEMV they fold (pipeline.ts). */
export function qNormRopeFoldFallbackWgsl(): string { return qNormRopeWgsl('workgroup', kSplitsOf(gemvGeometry(4))); }
export function kvPrologueFoldFallbackWgsl(): string { return kvPrologueWgsl('workgroup', kSplitsOf(gemvGeometry(4))); }

// ---------------------------------------------------------------------------------------------
// The CPU oracles: the three kernels' oracles composed, which is the definition of these two.
// ---------------------------------------------------------------------------------------------

export function qNormRopeOracle(
  src: Float32Array,
  gain: Float32Array,
  cos: Float32Array,
  sin: Float32Array,
  rows: number,
  width: number,
  eps: number,
  headsPerPosition: number,
): Float32Array {
  const normed = rmsNormOracle(src, gain, rows, width, eps);
  return ropeOracle(normed, cos, sin, rows, width / 2, headsPerPosition);
}

/**
 * The fold as the CPU sees it: `splits` partials of every element, summed in ascending split
 * order in f32 and snapped once with the projection's output scale (quant.ts applySrq), which is
 * the merge kernel's arithmetic; then the plain oracle over the folded values.
 */
export function foldPartialsOracle(partials: Float32Array, elements: number, splits: number, outScale: number): Float32Array {
  const out = new Float32Array(elements);
  for (let e = 0; e < elements; e += 1) {
    let total = 0;
    for (let s = 0; s < splits; s += 1) total = Math.fround(total + partials[e * splits + s]!);
    out[e] = total;
  }
  return applySrq(out, Number.isFinite(outScale) ? outScale : 0, undefined, out);
}

export function qNormRopeFoldOracle(
  partials: Float32Array,
  splits: number,
  outScale: number,
  gain: Float32Array,
  cos: Float32Array,
  sin: Float32Array,
  rows: number,
  width: number,
  eps: number,
  headsPerPosition: number,
): Float32Array {
  return qNormRopeOracle(foldPartialsOracle(partials, rows * width, splits, outScale), gain, cos, sin, rows, width, eps, headsPerPosition);
}

export interface KvPrologueShape extends KvStoreShape {
  eps: number;
}

export function kvPrologueOracle(
  k: Float32Array,
  v: Float32Array,
  gain: Float32Array,
  cos: Float32Array,
  sin: Float32Array,
  shape: KvPrologueShape,
  into?: Float32Array,
): Float32Array {
  const { headDim, tokenCount, eps } = shape;
  const kn = ropeOracle(rmsNormOracle(k, gain, tokenCount, headDim, eps), cos, sin, tokenCount, headDim / 2, 1);
  const vn = rmsNormOracle(v, null, tokenCount, headDim, eps);
  return kvStoreOracle(kn, vn, shape, into);
}

// ---------------------------------------------------------------------------------------------
// Params and binds.
// ---------------------------------------------------------------------------------------------

export function qNormRopeParams(vec4Count: number, rowCount: number, eps: number, headsPerPosition: number, fold?: { outScale: number }): ArrayBuffer {
  const words = new ArrayBuffer(fold ? 32 : 16);
  const u = new Uint32Array(words);
  const f = new Float32Array(words);
  u[0] = vec4Count;
  u[1] = rowCount;
  f[2] = eps;
  u[3] = headsPerPosition;
  if (fold) f[4] = fold.outScale;
  return words;
}

export function kvPrologueParams(vec4Count: number, tokenCount: number, eps: number, startPos: number, vBase: number, fold?: { kOutScale: number; vOutScale: number }): ArrayBuffer {
  const words = new ArrayBuffer(32);
  const u = new Uint32Array(words);
  const f = new Float32Array(words);
  u[0] = vec4Count;
  u[1] = tokenCount;
  f[2] = eps;
  u[3] = startPos;
  u[4] = vBase;
  if (fold) {
    f[5] = fold.kOutScale;
    f[6] = fold.vOutScale;
  }
  return words;
}

// Built inside the binds rather than at module scope: GPUShaderStage is a browser global and the
// Node checks import this module without one.
const entries = () => ({
  readOnly: { visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' as const } },
  readWrite: { visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' as const } },
  uniformEntry: { visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' as const } },
});

function bindQNormRope(input: KernelBindInput, fold = false): KernelBindResult {
  const { device, inputs, output, params } = input;
  const { readOnly, readWrite, uniformEntry } = entries();
  const name = fold ? Q_NORM_ROPE_FOLD_KERNEL : Q_NORM_ROPE_KERNEL;
  const rows = params.rows | 0;
  const width = params.width | 0;
  const eps = params.eps ?? 1e-6;
  const headsPerPosition = (params.headsPerPosition | 0) || rows;
  if (rows <= 0) throw new Error(`${Q_NORM_ROPE_KERNEL} needs params.rows`);
  if (width <= 0 || width % 4 !== 0 || width > MAX_HEAD_DIM) {
    throw new Error(`${Q_NORM_ROPE_KERNEL} needs params.width a positive multiple of 4 up to ${MAX_HEAD_DIM}, got ${width}`);
  }
  const { src, gain, cosTab, sinTab } = inputs;
  if (!src || !gain || !cosTab || !sinTab) throw new Error(`${Q_NORM_ROPE_KERNEL} needs inputs named src, gain, cosTab and sinTab`);
  const layout = kernelLayout(input, Q_NORM_ROPE_KERNEL, () => device.createBindGroupLayout({
    label: name,
    entries: [
      { binding: 0, ...readOnly },
      { binding: 1, ...readOnly },
      { binding: 2, ...readOnly },
      { binding: 3, ...readOnly },
      { binding: 4, ...readWrite },
      { binding: 5, ...uniformEntry },
    ],
  }));
  const uniform = kernelUniform(input, `${name} params`, qNormRopeParams(width / 4, rows, eps, headsPerPosition, fold ? { outScale: params.outScale ?? 0 } : undefined));
  return {
    layout,
    buffers: [src, gain, cosTab, sinTab, output, uniform.binding],
    dispatch: [rows, 1, 1],
    dispose: uniform.dispose,
  };
}

function bindKvPrologue(input: KernelBindInput, fold = false): KernelBindResult {
  const { device, inputs, output, params } = input;
  const { readOnly, readWrite, uniformEntry } = entries();
  const name = fold ? KV_PROLOGUE_FOLD_KERNEL : KV_PROLOGUE_KERNEL;
  const headDim = params.headDim | 0;
  const tokenCount = params.tokenCount | 0;
  const startPos = params.startPos | 0;
  const maxContext = params.maxContext | 0;
  const eps = params.eps ?? 1e-6;
  if (headDim <= 0 || headDim % 4 !== 0 || headDim > MAX_HEAD_DIM) {
    throw new Error(`${KV_PROLOGUE_KERNEL} needs params.headDim a positive multiple of 4 up to ${MAX_HEAD_DIM}, got ${headDim}`);
  }
  if (tokenCount <= 0) throw new Error(`${KV_PROLOGUE_KERNEL} needs params.tokenCount > 0`);
  if (startPos < 0 || startPos + tokenCount > maxContext) {
    throw new Error(`${KV_PROLOGUE_KERNEL}: startPos + tokenCount exceeds maxContext`);
  }
  const { k, v, gain, cosTab, sinTab } = inputs;
  if (!k || !v || !gain || !cosTab || !sinTab) throw new Error(`${KV_PROLOGUE_KERNEL} needs inputs named k, v, gain, cosTab and sinTab`);
  const layout = kernelLayout(input, KV_PROLOGUE_KERNEL, () => device.createBindGroupLayout({
    label: name,
    entries: [
      { binding: 0, ...readOnly },
      { binding: 1, ...readOnly },
      { binding: 2, ...readOnly },
      { binding: 3, ...readOnly },
      { binding: 4, ...readOnly },
      { binding: 5, ...readWrite },
      { binding: 6, ...uniformEntry },
    ],
  }));
  const uniform = kernelUniform(
    input,
    `${name} params`,
    kvPrologueParams(headDim / 4, tokenCount, eps, startPos, maxContext * headDim,
      fold ? { kOutScale: params.kOutScale ?? 0, vOutScale: params.vOutScale ?? 0 } : undefined),
  );
  return {
    layout,
    buffers: [k, v, gain, cosTab, sinTab, output, uniform.binding],
    // One workgroup per (position, K or V). The count is the position count, which is the
    // dispatch's business and never the context's, so the decode step cache holds it fixed.
    dispatch: [tokenCount, 2, 1],
    dispose: uniform.dispose,
  };
}

// The tolerance argument, shared by every case below. The norm cases in rmsNorm.ts argue 8 ULP
// for inverseSqrt and the four correctly rounded steps around it; the rope cases in rope.ts argue
// 1e-6 absolute for one unrounded product under FMA contraction with every input under 8. Here
// the rotation's inputs are the norm's outputs: fixture values on the half step grid up to 4,
// over a row whose root mean square is above 1, times a gain up to 4, so under 8 in magnitude, and
// 8 ULP of a value under 8 is 3.9e-6. A rotated element is two such values times a cosine and a
// sine of magnitude at most 1, so at most 7.7e-6 of norm error plus the 1e-6 of contraction:
// 1e-5 absolute covers it with a little room. The store is a copy and adds nothing.
const FUSED_TOL_ABS = 1e-5;

export const qNormRopeKernel: Kernel = {
  name: Q_NORM_ROPE_KERNEL,
  wgsl: Q_NORM_ROPE_WGSL,
  entry: 'main',
  note:
    'q_norm and rope q in one dispatch, one workgroup per (position, head) row: the norm kernel\'s '
    + 'reduction text and apply, the rope kernel\'s split half form, the intermediate in workgroup '
    + 'memory. A decode step runs it on every layer where it ran two kernels.',
  cases: [
    {
      name: 'sliding-256-8-heads',
      inputs: { src: 'knorm.rms.qhead.src', gain: 'knorm.rms.qhead.gain', cosTab: 'krope.sliding.cos', sinTab: 'krope.sliding.sin' },
      expected: 'kfuse.qrope.sliding.expected',
      params: { rows: 8, width: 256, eps: 1e-6, headsPerPosition: 8 },
      tolAbs: FUSED_TOL_ABS,
      note: 'The norm fixture\'s eight head rows, normalized with its gain, then rotated at position 37 with the sliding tables.',
    },
    {
      name: 'global-512-8-heads',
      inputs: { src: 'knorm.rms.qhead512.src', gain: 'knorm.rms.qhead512.gain', cosTab: 'krope.global.cos', sinTab: 'krope.global.sin' },
      expected: 'kfuse.qrope.global.expected',
      params: { rows: 8, width: 512, eps: 1e-6, headsPerPosition: 8 },
      tolAbs: FUSED_TOL_ABS,
      note: 'The global head: 512 wide, 64 of 256 pairs rotating, the rest carrying cosine 1 and sine 0 in the tables.',
    },
  ],
  bind: bindQNormRope,
};

export const kvPrologueKernel: Kernel = {
  name: KV_PROLOGUE_KERNEL,
  wgsl: KV_PROLOGUE_WGSL,
  entry: 'main',
  note:
    'k_norm, v_norm, rope k and the KV store in one dispatch, two workgroups per position (K and '
    + 'V): the norm kernels\' reduction text and applies, the rope kernel\'s rotation, the store '
    + 'kernel\'s row placement, written straight into the packed cache. A decode step runs it on '
    + 'every producer layer where it ran four kernels.',
  cases: [
    {
      name: 'sliding-256-at-3',
      inputs: { k: 'kfuse.kv.sliding.k', v: 'kfuse.kv.sliding.v', gain: 'knorm.rms.qhead.gain', cosTab: 'krope.sliding.cos', sinTab: 'krope.sliding.sin' },
      expected: 'kfuse.kv.sliding.cache',
      params: { headDim: 256, tokenCount: 1, startPos: 3, maxContext: 8, eps: 1e-6 },
      tolAbs: FUSED_TOL_ABS,
      outputInit: 'zero',
      note: 'One position landing at slot 3 of an eight slot cache: K normed, rotated and stored, V normed weightlessly and stored, every other slot left zero.',
    },
    {
      name: 'global-512-at-5',
      inputs: { k: 'kfuse.kv.global.k', v: 'kfuse.kv.global.v', gain: 'knorm.rms.qhead512.gain', cosTab: 'krope.global.cos', sinTab: 'krope.global.sin' },
      expected: 'kfuse.kv.global.cache',
      params: { headDim: 512, tokenCount: 1, startPos: 5, maxContext: 8, eps: 1e-6 },
      tolAbs: FUSED_TOL_ABS,
      outputInit: 'zero',
      note: 'The same at the global head dimension of 512 and the partial rotation; the stride is per cache, not 256.',
    },
  ],
  bind: bindKvPrologue,
};

/**
 * The fold variants. Their one harness case each is the plain case: at kSplits 1 a fold over one
 * partial with no output scale is the plain kernel, so it owes the same answer exactly; the fold
 * over several partials is held to the composed oracle in the k-norms check and to parity.
 */
export const qNormRopeFoldKernel: Kernel = {
  name: Q_NORM_ROPE_FOLD_KERNEL,
  get wgsl(): string { return qNormRopeWgsl('subgroup', kSplitsOf(gemvGeometry(4))); },
  entry: 'main',
  note: 'q-norm-rope over the partials a split q_proj wrote: summed in split order, snapped with the projection output scale, then the norm and the rotation.',
  cases: [
    {
      name: 'sliding-256-8-heads-unsplit',
      inputs: { src: 'knorm.rms.qhead.src', gain: 'knorm.rms.qhead.gain', cosTab: 'krope.sliding.cos', sinTab: 'krope.sliding.sin' },
      expected: 'kfuse.qrope.sliding.expected',
      params: { rows: 8, width: 256, eps: 1e-6, headsPerPosition: 8, outScale: 0 },
      tolAbs: FUSED_TOL_ABS,
      note: 'At kSplits 1 with no output scale the fold is the plain kernel, so it owes the same answer.',
    },
  ],
  bind: (input) => bindQNormRope(input, true),
};

export const kvPrologueFoldKernel: Kernel = {
  name: KV_PROLOGUE_FOLD_KERNEL,
  get wgsl(): string { return kvPrologueWgsl('subgroup', kSplitsOf(gemvGeometry(4))); },
  entry: 'main',
  note: 'kv-norm-rope-store over the partials split k_proj and v_proj wrote: each summed in split order and snapped with its own output scale, then the norms, the rotation and the store.',
  cases: [
    {
      name: 'sliding-256-at-3-unsplit',
      inputs: { k: 'kfuse.kv.sliding.k', v: 'kfuse.kv.sliding.v', gain: 'knorm.rms.qhead.gain', cosTab: 'krope.sliding.cos', sinTab: 'krope.sliding.sin' },
      expected: 'kfuse.kv.sliding.cache',
      params: { headDim: 256, tokenCount: 1, startPos: 3, maxContext: 8, eps: 1e-6, kOutScale: 0, vOutScale: 0 },
      tolAbs: FUSED_TOL_ABS,
      outputInit: 'zero',
      note: 'At kSplits 1 with no output scales the fold is the plain kernel, so it owes the same answer.',
    },
  ],
  bind: (input) => bindKvPrologue(input, true),
};
