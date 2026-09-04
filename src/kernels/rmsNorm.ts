// Written from docs/ENGINE-PLAN.md section 5 (the architecture contract table and kernel K6),
// risk 1 quirks 1 and 4, section 5.5's reduction rules, the cost geometry in DECODE-CAMPAIGN.md
// 4.2 and 4.7, the drift record in DECODE-FUSION-FINDING.md, and the WGSL specification. No
// vendored bundle, no extracted kernel and no third party engine source was read.
//
// SPDX-License-Identifier: Apache-2.0
//
// Every RMS norm in Gemma 4, in one module, in two shapes.
//
// THE FORM, which is trap 4 of the four silent corruption hazards. This model's RMS norm is plain
//
//     y = x / rms(x) * w
//
// It does NOT use the `(1 + w)` form that Gemma 2, 3 and 3n use. A port that reuses a Gemma 3 norm
// kernel gets a wrong answer at every norm in the model, and there is a norm every few dispatches
// (ENGINE-PLAN risk 1 quirk 4). The failing test for this is in scripts/engine-check.mjs section
// k-norms: with a weight vector of all zeros this kernel must output zeros, where the `(1 + w)`
// form would output the normalized input untouched. That is a one line test that cannot pass by
// accident and it is the cheapest insurance in the engine.
//
// THE WEIGHTLESS ONE, which is trap 1. V is RMS normalized per KV head with no learnable weight,
// between the V projection and the KV cache store, on producer layers only. There is no
// `v_norm.weight` tensor in either checkpoint because a weightless norm has nothing to store.
// This project has two independent observations of it: the public finding, and this project's own
// live dispatch census, which recorded an unweighted RMS norm on the v row before the cache copy
// while looking for something else entirely (DECODE-CAMPAIGN.md 4.7). Miss it and V enters
// attention at roughly 67 times the correct magnitude; softmax self normalizes so it looks fine,
// and the residual stream drifts catastrophically across layers instead.
//
// The weightless kernel is a separate compiled module rather than the weighted one with a weight
// of ones. That is deliberate and it costs nothing: a weight-1 norm is algebraically a no-op
// rescale, which is exactly why quantizers drop it and why ports miss it, and a separate named
// kernel with a separate case in the registry cannot be dropped silently. The numerical cost of
// two modules is nil here because the drift record is unambiguous on this point: V rows never
// drifted anywhere, across an entire evening of probing, while K rows drifted at every site
// (DECODE-FUSION-FINDING.md).
//
// ONE KERNEL SERVES THREE CALL SITES. `rms-norm` is the residual stream norm at width 1536, the
// per head q norm and the per head k norm at width 256 or 512. They are the same maths over a
// different row width, and `gain` is indexed inside the row, so a per head norm passes a weight
// vector of head_dim and as many rows as there are heads. Three call sites, one module, which is
// also what ENGINE-PLAN risk 2 mitigation 4 asks for: where a scalar multiply-add chain is
// unavoidable, keep it in exactly one module so there is no second module to disagree with.

import type { Kernel, KernelBindInput, KernelBindResult } from './registry';
import { kernelLayout, kernelUniform } from './binding';
import {
  SUBGROUP_BUTTERFLY_WGSL,
  SUBGROUP_ENABLE,
  workgroupTreeWgsl,
} from './subgroupReduce';

/** 64 wide, 2 subgroups. The measured best decode workgroup width on this hardware
 *  (DECODE-CAMPAIGN.md 4.2: 64 wide with 2 subgroups measured best, 32 wide measured worse
 *  because one warp blocks cap resident warps and give the scheduler nothing to hide latency
 *  with). Nothing here is a GEMV, but the width is the one the rest of the engine uses and a
 *  second width is a second thing to tune. */
// One workgroup covers a whole row, so the row width is the only parallelism here: at 64 lanes a
// decode step ran ONE workgroup over 1536 values, which is a latency figure and not a bandwidth
// one, and 256 lanes cut the dispatch by a third to a half (docs/ENGINE-PERF.md section 18).
// 256 is the WebGPU ceiling on invocations per workgroup.
//
// THE PORTABLE PATH KEEPS 64. Widening the workgroup tree changes the summation order, and at 256
// it moved a near tie: full-r13 under the workgroup policy lost interview-300's first token, which
// r11 and r12 both matched. The subgroup butterfly at 256 held every probe the narrower one held.
// The gate is the arbiter, so the widening is kept where the receipt clears it and the fallback
// stays byte for byte the shape its last passing receipt covered.
const WORKGROUP_SIZE = 256;
const FALLBACK_WORKGROUP_SIZE = 64;
/** 32 lane groups in a workgroup of `w`, the slots the subgroup path reduces across. */
const slotsOf = (w: number): number => w / 32;

export type ReducePath = 'subgroup' | 'workgroup';

export interface RmsNormVariant {
  /** False for the weightless per KV head V norm of trap 1. */
  weighted: boolean;
  /** 'workgroup' is the fallback for a device that fails the init subgroup self test. */
  reduce: ReducePath;
}

/**
 * Build the WGSL for one norm variant.
 *
 * Structure notes, each of which is a rule from ENGINE-PLAN 5.5 rather than a preference:
 *
 *  - The sum of squares is a `dot()` over vec4 lanes. Values flowing through `dot()` intrinsics
 *    never drifted across compiled modules in a full evening of probing, while scalar chains did
 *    (DECODE-FUSION-FINDING.md), so the reduction is shaped as a dot product deliberately.
 *  - Every reduction sits above every store. The only lane divergent store in the kernel is the
 *    one slot per subgroup write into `sgPartials`, and nothing is reduced after it. Rule 1.
 *  - The tail reduction is the 32 lane butterfly from subgroupReduce.ts, never a bare
 *    `subgroupAdd`. Rule 2.
 *  - The loop bound comes from the uniform, not from a constant. Rule 6, whose cost when ignored
 *    was Metal unrolling a 256 iteration constant bound loop into a megakernel that halved decode.
 *  - Four storage bindings at most, against an adapter limit of 10. Risk 6.
 *
 * The kernel reads `src` twice, once to reduce and once to apply. That is safe in place, because
 * a lane only ever writes the element it read, and the workgroup barrier between the two loops
 * orders every read before every write. Running it in place saves an activation sized buffer,
 * which matters on the M1 where decode is streaming bound.
 */
export function rmsNormWgsl(variant: RmsNormVariant): string {
  const W = variant.reduce === 'subgroup' ? WORKGROUP_SIZE : FALLBACK_WORKGROUP_SIZE;
  const SLOTS = slotsOf(W);
  const gainBinding = variant.weighted
    ? '@group(0) @binding(1) var<storage, read> gain: array<vec4<f32>>;\n'
    : '';
  const dstBinding = variant.weighted ? 2 : 1;
  const paramsBinding = variant.weighted ? 3 : 2;
  const apply = variant.weighted ? 'src[base + i] * inv * gain[i]' : 'src[base + i] * inv';

  const head = variant.reduce === 'subgroup'
    ? `${SUBGROUP_ENABLE}\n${SUBGROUP_BUTTERFLY_WGSL}\nvar<workgroup> sgPartials: array<f32, ${SLOTS}>;\n`
    : workgroupTreeWgsl('treeScratch', W);

  const reduce = variant.reduce === 'subgroup'
    ? `  // The butterfly runs first, in uniform control flow, and only then does one lane per
  // 32 lane group store its slot. Reduce above store, never the other way round.
  let lanes = sgSum32(acc);
  if ((lid & 31u) == 0u) {
    sgPartials[lid >> 5u] = lanes;
  }
  workgroupBarrier();
  // One slot per 32 lane group, added in ascending order. Frozen, per ENGINE-PLAN risk 2
  // mitigation 2: the order is a property of the kernel, not of the schedule.
  let total = ${Array.from({ length: SLOTS }, (_, g) => `sgPartials[${g}]`).join(' + ')};`
    : `  let total = wgSum(acc, lid);`;

  return /* wgsl */ `${head}
struct NormParams {
  // Row width in vec4 lanes, so 384 for the residual stream and 64 or 128 for a head. Runtime
  // opaque, which is the whole point of it being here rather than a constant.
  vec4Count: u32,
  // Carried for the caller to validate against, not read by the shader. The dispatch is exactly
  // one workgroup per row, so a bounds guard would be dead code, and an early return before a
  // workgroupBarrier would put that barrier in non uniform control flow.
  rowCount: u32,
  eps: f32,
  pad0: u32,
}

@group(0) @binding(0) var<storage, read> src: array<vec4<f32>>;
${gainBinding}@group(0) @binding(${dstBinding}) var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(${paramsBinding}) var<uniform> params: NormParams;

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
    let v = src[base + i];
    acc = acc + dot(v, v);
  }

${reduce}

  // Plain Gemma 4 form. There is no 1.0 added to the weight anywhere in this file, and there is a
  // test that fails if somebody adds one.
  let inv = inverseSqrt(total / f32(n4 * 4u) + params.eps);

  for (var i = lid; i < n4; i = i + ${W}u) {
    dst[base + i] = ${apply};
  }
}
`;
}

export const RMS_NORM_WGSL = rmsNormWgsl({ weighted: true, reduce: 'subgroup' });
export const RMS_NORM_WEIGHTLESS_WGSL = rmsNormWgsl({ weighted: false, reduce: 'subgroup' });
export const RMS_NORM_FALLBACK_WGSL = rmsNormWgsl({ weighted: true, reduce: 'workgroup' });
export const RMS_NORM_WEIGHTLESS_FALLBACK_WGSL = rmsNormWgsl({ weighted: false, reduce: 'workgroup' });

// ---------------------------------------------------------------------------------------------
// The CPU oracle.
// ---------------------------------------------------------------------------------------------

/**
 * The known answer reference for both norm kernels.
 *
 * Accumulates in f64 and rounds once at the end, so it is a reference rather than a mirror of the
 * shader's f32 reduction. ENGINE-PLAN section 7 already settles what that means: bit identity is
 * not the contract for this shape, the contract is a stated tolerance that somebody argued for.
 * The argument for the norm cases is in scripts/engine-check.mjs.
 *
 * @param src   `rows * width` values, row major
 * @param gain  `width` values applied inside every row, or null for the weightless V norm
 * @param rows  one workgroup per row on the GPU
 * @param width row width, 1536 for the residual stream or head_dim for a per head norm
 * @param eps   1e-6 from the config
 */
export function rmsNormOracle(
  src: Float32Array,
  gain: Float32Array | null,
  rows: number,
  width: number,
  eps: number,
): Float32Array {
  if (src.length < rows * width) {
    throw new Error(`rmsNormOracle: src holds ${src.length}, needs ${rows * width}`);
  }
  if (gain && gain.length < width) {
    throw new Error(`rmsNormOracle: gain holds ${gain.length}, needs ${width}`);
  }
  const out = new Float32Array(rows * width);
  for (let r = 0; r < rows; r += 1) {
    const base = r * width;
    let sum = 0;
    for (let i = 0; i < width; i += 1) {
      const v = src[base + i];
      sum += v * v;
    }
    const inv = 1 / Math.sqrt(sum / width + eps);
    for (let i = 0; i < width; i += 1) {
      // Plain form. Multiply by the weight, never by one plus the weight.
      out[base + i] = src[base + i] * inv * (gain ? gain[i] : 1);
    }
  }
  return out;
}

/**
 * The trap 4 counter-example, kept beside the real thing on purpose.
 *
 * ENGINE-PLAN risk 1's mitigation asks for a deliberate test that flips each quirk to the wrong
 * setting and asserts the answer collapses. This is the wrong setting for the norm form: the
 * `(1 + w)` scaling that Gemma 2, 3 and 3n use. Nothing in the engine calls it, and the check
 * script asserts it differs from `rmsNormOracle` on a weight that is not all ones.
 */
export function rmsNormGemma3FormOracle(
  src: Float32Array,
  gain: Float32Array,
  rows: number,
  width: number,
  eps: number,
): Float32Array {
  const out = new Float32Array(rows * width);
  for (let r = 0; r < rows; r += 1) {
    const base = r * width;
    let sum = 0;
    for (let i = 0; i < width; i += 1) sum += src[base + i] * src[base + i];
    const inv = 1 / Math.sqrt(sum / width + eps);
    for (let i = 0; i < width; i += 1) out[base + i] = src[base + i] * inv * (1 + gain[i]);
  }
  return out;
}

// ---------------------------------------------------------------------------------------------
// Registry entries.
// ---------------------------------------------------------------------------------------------

/** The params block, exported so the scheduler can restage it in place per dispatch. */
export function normParams(vec4Count: number, rowCount: number, eps: number): ArrayBuffer {
  const words = new ArrayBuffer(16);
  const u = new Uint32Array(words);
  const f = new Float32Array(words);
  u[0] = vec4Count;
  u[1] = rowCount;
  f[2] = eps;
  u[3] = 0;
  return words;
}

function readShape(params: Readonly<Record<string, number>>): { rows: number; width: number; eps: number } {
  const rows = params.rows | 0;
  const width = params.width | 0;
  const eps = params.eps ?? 1e-6;
  if (rows <= 0) throw new Error('rms-norm needs params.rows');
  if (width <= 0 || width % 4 !== 0) {
    throw new Error(`rms-norm needs params.width a positive multiple of 4, got ${width}`);
  }
  return { rows, width, eps };
}

function bindWeighted(input: KernelBindInput): KernelBindResult {
  const { device, inputs, output, params } = input;
  const { rows, width, eps } = readShape(params);
  const src = inputs.src;
  const gain = inputs.gain;
  if (!src || !gain) throw new Error('rms-norm needs inputs named src and gain');

  const layout = kernelLayout(input, 'rms-norm', () => device.createBindGroupLayout({
    label: 'rms-norm',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  }));
  const uniform = kernelUniform(input, 'rms-norm params', normParams(width / 4, rows, eps));

  return {
    layout,
    buffers: [src, gain, output, uniform.binding],
    dispatch: [rows, 1, 1],
    dispose: uniform.dispose,
  };
}

function bindWeightless(input: KernelBindInput): KernelBindResult {
  const { device, inputs, output, params } = input;
  const { rows, width, eps } = readShape(params);
  const src = inputs.src;
  if (!src) throw new Error('rms-norm-weightless needs an input named src');

  const layout = kernelLayout(input, 'rms-norm-weightless', () => device.createBindGroupLayout({
    label: 'rms-norm-weightless',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  }));
  const uniform = kernelUniform(input, 'rms-norm-weightless params', normParams(width / 4, rows, eps));

  return {
    layout,
    buffers: [src, output, uniform.binding],
    dispatch: [rows, 1, 1],
    dispose: uniform.dispose,
  };
}

// The tolerance argument, once, since both norm kernels share it.
//
// The fixture is built so the reduction itself is exact whatever order it runs in: every input is
// a multiple of 1/2 with magnitude at most 4, so every square is a multiple of 1/4 and at most 16,
// and the running total over 1536 terms stays under 2^15, well inside the range where a multiple
// of 1/4 is exact in f32. So no error enters from the sum, and what remains is the divide by the
// width, the epsilon add, the `inverseSqrt`, and the two multiplies that apply the result. WGSL
// gives `inverseSqrt` an accuracy of 2 ULP; the other four steps are correctly rounded and
// contribute at most 1 ULP each. 8 ULP is that budget with a little room, not a shrug.
const NORM_TOL_ULP = 8;

export const rmsNormKernel: Kernel = {
  name: 'rms-norm',
  wgsl: RMS_NORM_WGSL,
  entry: 'main',
  note:
    'Gemma 4 RMS norm, plain x / rms * w with no (1 + w) term. One module for the residual stream '
    + 'norm at width 1536 and the per head q and k norms at head_dim.',
  cases: [
    {
      name: 'hidden-1536',
      inputs: { src: 'knorm.rms.hidden.src', gain: 'knorm.rms.hidden.gain' },
      expected: 'knorm.rms.hidden.expected',
      params: { rows: 4, width: 1536, eps: 1e-6 },
      tolUlp: NORM_TOL_ULP,
      note: 'Residual stream width. Fixture is exact under summation, so only inverseSqrt and the two multiplies move.',
    },
    {
      name: 'qhead-256',
      inputs: { src: 'knorm.rms.qhead.src', gain: 'knorm.rms.qhead.gain' },
      expected: 'knorm.rms.qhead.expected',
      params: { rows: 8, width: 256, eps: 1e-6 },
      tolUlp: NORM_TOL_ULP,
      note: 'Eight query heads of a sliding layer, gain shared across rows, which is the per head norm shape.',
    },
    {
      name: 'qhead-512-global',
      inputs: { src: 'knorm.rms.qhead512.src', gain: 'knorm.rms.qhead512.gain' },
      expected: 'knorm.rms.qhead512.expected',
      params: { rows: 8, width: 512, eps: 1e-6 },
      tolUlp: NORM_TOL_ULP,
      note: 'Head dimension is 512 on the seven full attention layers, not 256. Wrong quietly if assumed.',
    },
  ],
  bind: bindWeighted,
};

export const rmsNormWeightlessKernel: Kernel = {
  name: 'rms-norm-weightless',
  wgsl: RMS_NORM_WEIGHTLESS_WGSL,
  entry: 'main',
  note:
    'The per KV head V norm. No learnable weight, because there is no v_norm.weight tensor in the '
    + 'checkpoint. Runs between the V projection and the KV cache store, producer layers only.',
  cases: [
    {
      name: 'vhead-256',
      inputs: { src: 'knorm.vnorm.src' },
      expected: 'knorm.vnorm.expected',
      params: { rows: 1, width: 256, eps: 1e-6 },
      tolUlp: NORM_TOL_ULP,
      note: 'One KV head at the sliding head dimension. Trap 1: skipping this inflates V by about 67x.',
    },
    {
      name: 'vhead-512-global',
      inputs: { src: 'knorm.vnorm.src512' },
      expected: 'knorm.vnorm.expected512',
      params: { rows: 1, width: 512, eps: 1e-6 },
      tolUlp: NORM_TOL_ULP,
      note: 'The full attention KV head. Same weightless norm, different width.',
    },
  ],
  bind: bindWeightless,
};
