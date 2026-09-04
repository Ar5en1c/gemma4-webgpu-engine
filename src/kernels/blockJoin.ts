// Written from docs/ENGINE-PLAN.md section 5 (K11 and the fusion policy), section 5.5's reduction
// rules, the Hugging Face transformers reference implementation of the Gemma 4 decoder layer, the
// WGSL specification, and this engine's own rmsNorm.ts and mlpEpilogue.ts. No vendored bundle, no
// extracted kernel and no third party engine source was read.
//
// SPDX-License-Identifier: Apache-2.0
//
// The block join: `hidden = alpha * (residual + rmsNorm(src) * gain)` and, in the same pass,
// `normed = rmsNorm(hidden) * gain2`. It is the K11 epilogue (mlpEpilogue.ts), the per layer
// scalar (scale-add at alpha) and the next pre norm (rmsNorm.ts) as one dispatch, at the two
// places in every decoder layer where that chain occurs:
//
//     post_attention_layernorm + residual, then pre_feedforward_layernorm         (alpha 1)
//     post_per_layer_input_norm + residual, layer_scalar, then the next layer's
//     input_layernorm, or the final norm after the last layer on a decode step  (alpha = scalar)
//
// WHY IT EXISTS. The round 4 dispatch ledger (docs/ENGINE-PERF.md section 15) counted 731
// dispatches per decode token, about 480 of them elementwise or norm kernels of four to nine
// microseconds each, which on the M1 is launch cost rather than work: the norm kernels read 6 KB.
// This kernel removes three dispatches per layer. The plan's comment that dispatch count is not
// the binding constraint was written before the GEMV families were within ten percent of the
// incumbent's; at that point the launches are the gap.
//
// WHAT IT MUST NOT CHANGE. The arithmetic is the three kernels' arithmetic in the same order:
// the first sum of squares, the inverse square root, `residual + src * inv * gain` as the epilogue
// writes it, the multiply by alpha that scale-add did as `alpha * x + 0`, and the second norm over
// the stored hidden row with rmsNorm.ts's own expression `h * inv2 * gain2`. Both reductions use
// the same lane assignment and the same butterfly or tree as those kernels, so the frozen
// accumulation order of ENGINE-PLAN risk 2 is literally the same code, and the full model page
// is expected to read bit for bit what the unfused chain read.
//
// RULE 1, stated for the second reduction. It follows the `hidden` stores, which are unconditional
// and in uniform control flow (the loop bound is the uniform's row width, and every lane of the
// workgroup runs the loop). The per subgroup partial is written by every lane of the subgroup
// into its own slot rather than by lane 0 into a shared one, so nothing in this kernel is a lane
// divergent store, and the butterfly is the only subgroup operation. On the fallback path the
// tree's own leading barrier orders the first call's last read before the second call's first
// write.
//
// TWO OUTPUTS, ONE HARNESS SLOT. The registry contract binds one output per case, so `hidden`
// arrives as an input binding declared read_write and `normed` is the case's output. The sweep
// proves the arithmetic through `normed`, which is a function of every hidden element; that the
// hidden row lands in the residual stream is proved by the full model page, whose greedy ids
// through 35 layers cannot survive a missing residual write.

import type { Kernel, KernelBindInput, KernelBindResult } from './registry';
import { kernelLayout, kernelUniform } from './binding';
import { SUBGROUP_BUTTERFLY_WGSL, SUBGROUP_ENABLE, workgroupTreeWgsl } from './subgroupReduce';
import { rmsNormOracle } from './rmsNorm';
import { normResidualOracle } from './mlpEpilogue';

/** 64 wide, 2 subgroups, the width rmsNorm.ts and mlpEpilogue.ts run at. */
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

export type JoinReducePath = 'subgroup' | 'workgroup';

/** The registry name, and the plan's kernel name for the two fused sites. */
export const NORM_RESIDUAL_NORM = 'norm-residual-norm';

export function normResidualNormWgsl(reduce: JoinReducePath): string {
  const W = reduce === 'subgroup' ? WORKGROUP_SIZE : FALLBACK_WORKGROUP_SIZE;
  const SLOTS = slotsOf(W);
  const head = reduce === 'subgroup'
    ? `${SUBGROUP_ENABLE}\n${SUBGROUP_BUTTERFLY_WGSL}\nvar<workgroup> sgPartials: array<f32, ${W}>;\n`
    : workgroupTreeWgsl('treeScratch', W);

  // One full row reduction. On the subgroup path every lane writes the butterfly result of its
  // own 32 lane group into its own slot (no divergent store), and the total is the groups' slots
  // summed in ascending order, a fixed order so the sum is a pure function of the row.
  const reduceRow = (acc: string, total: string): string => (reduce === 'subgroup'
    ? `  workgroupBarrier();
  sgPartials[lid] = sgSum32(${acc});
  workgroupBarrier();
  let ${total} = ${Array.from({ length: SLOTS }, (_, g) => `sgPartials[${g * 32}]`).join(' + ')};`
    : `  let ${total} = wgSum(${acc}, lid);`);

  return /* wgsl */ `${head}
struct JoinParams {
  // Row width in vec4 lanes, 384 for the 1536 wide residual stream. Runtime opaque on purpose.
  vec4Count: u32,
  // Carried for the caller to validate against; the dispatch is one workgroup per row.
  rowCount: u32,
  eps: f32,
  // The per layer scalar applied to the joined row, 1.0 at the attention to MLP site.
  alpha: f32,
}

@group(0) @binding(0) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> gain: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> residual: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> gain2: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> hidden: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(6) var<uniform> params: JoinParams;

@compute @workgroup_size(${W})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_index) lid: u32,
) {
  let row = wid.x;
  let n4 = params.vec4Count;
  let base = row * n4;
  let width = f32(n4 * 4u);

  // The epilogue's norm of the block output.
  var acc = 0.0;
  for (var i = lid; i < n4; i = i + ${W}u) {
    let v = src[base + i];
    acc = acc + dot(v, v);
  }
${reduceRow('acc', 'total')}
  // Plain Gemma 4 norm form, no 1.0 added to the weight, here as in the two kernels this joins.
  let inv = inverseSqrt(total / width + params.eps);

  // The joined row: norm, then the residual, then the scalar, stored as the residual stream and
  // squared on the way past for the second norm.
  var acc2 = 0.0;
  for (var i = lid; i < n4; i = i + ${W}u) {
    let h = (residual[base + i] + src[base + i] * inv * gain[i]) * params.alpha;
    hidden[base + i] = h;
    acc2 = acc2 + dot(h, h);
  }
${reduceRow('acc2', 'total2')}
  let inv2 = inverseSqrt(total2 / width + params.eps);

  // The next block's pre norm, read back from the lane's own stores.
  for (var i = lid; i < n4; i = i + ${W}u) {
    dst[base + i] = hidden[base + i] * inv2 * gain2[i];
  }
}
`;
}

export const NORM_RESIDUAL_NORM_WGSL = normResidualNormWgsl('subgroup');
export const NORM_RESIDUAL_NORM_FALLBACK_WGSL = normResidualNormWgsl('workgroup');

// ---------------------------------------------------------------------------------------------
// The CPU oracle.
// ---------------------------------------------------------------------------------------------

/**
 * The known answer reference, by construction the composition of the two kernels this one
 * replaces: mlpEpilogue.ts's oracle, the scalar applied with one f32 rounding (what scale-add
 * stored), then rmsNorm.ts's oracle over the stored row. Returns both rows the kernel writes.
 */
export function normResidualNormOracle(
  src: Float32Array,
  gain: Float32Array,
  residual: Float32Array,
  gain2: Float32Array,
  alpha: number,
  rows: number,
  width: number,
  eps: number,
): { hidden: Float32Array; normed: Float32Array } {
  const hidden = normResidualOracle(src, gain, residual, rows, width, eps);
  for (let i = 0; i < hidden.length; i += 1) hidden[i] = Math.fround(hidden[i]! * alpha);
  const normed = rmsNormOracle(hidden, gain2, rows, width, eps);
  return { hidden, normed };
}

// ---------------------------------------------------------------------------------------------
// Registry entry.
// ---------------------------------------------------------------------------------------------

/** The params block, exported so the scheduler can restage it in place per dispatch. */
export function joinParams(vec4Count: number, rowCount: number, eps: number, alpha: number): ArrayBuffer {
  const words = new ArrayBuffer(16);
  const u = new Uint32Array(words);
  const f = new Float32Array(words);
  u[0] = vec4Count;
  u[1] = rowCount;
  f[2] = eps;
  f[3] = alpha;
  return words;
}

function bind(input: KernelBindInput): KernelBindResult {
  const { device, inputs, output, params } = input;
  const rows = params.rows | 0;
  const width = params.width | 0;
  const eps = params.eps ?? 1e-6;
  const alpha = params.alpha ?? 1;
  if (rows <= 0) throw new Error(`${NORM_RESIDUAL_NORM} needs params.rows`);
  if (width <= 0 || width % 4 !== 0) {
    throw new Error(`${NORM_RESIDUAL_NORM} needs params.width a positive multiple of 4, got ${width}`);
  }
  const { src, gain, residual, gain2, hidden } = inputs;
  if (!src || !gain || !residual || !gain2 || !hidden) {
    throw new Error(`${NORM_RESIDUAL_NORM} needs inputs named src, gain, residual, gain2 and hidden`);
  }

  const layout = kernelLayout(input, NORM_RESIDUAL_NORM, () => device.createBindGroupLayout({
    label: NORM_RESIDUAL_NORM,
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  }));
  const uniform = kernelUniform(input, `${NORM_RESIDUAL_NORM} params`, joinParams(width / 4, rows, eps, alpha));

  return {
    layout,
    buffers: [src, gain, residual, gain2, hidden, output, uniform.binding],
    dispatch: [rows, 1, 1],
    dispose: uniform.dispose,
  };
}

// The tolerance argument. The fixture makes everything but two operations exact: `src` is +1 or
// -1 on every element with eps 0, so the first sum is exactly the width and inv is exactly 1;
// `residual` sits on the 1/4 grid under 4 and `gain` on the 1/8 grid under 2, so the joined row
// is on the 1/8 grid under 6 and, at alpha 0.5, on the 1/16 grid, exact; its squares are on the
// 1/256 grid under 36 and their sum over 1536 terms stays under 2^24 on that grid, exact in any
// order. What is left is the second inverseSqrt, 2 ULP by the WGSL specification, and two
// correctly rounded multiplies. 8 ULP covers that with room, and the absolute gate beside it
// carries the elements near zero, where `gain2` crosses zero and a ULP is not a useful unit.
const JOIN_TOL_ULP = 8;
const JOIN_TOL_ABS = 4e-6;

export const normResidualNormKernel: Kernel = {
  name: NORM_RESIDUAL_NORM,
  wgsl: NORM_RESIDUAL_NORM_WGSL,
  entry: 'main',
  note:
    'The block join: alpha * (residual + rmsNorm(src) * gain) stored as the residual stream, and '
    + 'rmsNorm of that row times gain2 as the next pre norm, one dispatch for what was three. '
    + 'Serves the attention to MLP site at alpha 1 and the layer tail at alpha = layer_scalar.',
  cases: [
    {
      name: 'decode-1536',
      inputs: {
        src: 'kmlp.join.src',
        gain: 'kmlp.epi.gain',
        residual: 'kmlp.epi.residual',
        gain2: 'kmlp.join.gain2',
        hidden: 'kmlp.join.scratch',
      },
      expected: 'kmlp.join.expected',
      params: { rows: 1, width: 1536, eps: 0, alpha: 0.5 },
      tolUlp: JOIN_TOL_ULP,
      tolAbs: JOIN_TOL_ABS,
      note: 'One position, the decode shape, at a scalar of one half so the scalar is provably applied.',
    },
    {
      name: 'prefill-4x1536',
      inputs: {
        src: 'kmlp.join.src4',
        gain: 'kmlp.epi.gain',
        residual: 'kmlp.epi.residual4',
        gain2: 'kmlp.join.gain2',
        hidden: 'kmlp.join.scratch4',
      },
      expected: 'kmlp.join.expected4',
      params: { rows: 4, width: 1536, eps: 0, alpha: 0.5 },
      tolUlp: JOIN_TOL_ULP,
      tolAbs: JOIN_TOL_ABS,
      note: 'Four positions in one dispatch, the prefill shape, rows reduced independently.',
    },
  ],
  bind,
};
