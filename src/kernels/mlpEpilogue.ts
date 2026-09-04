// Written from docs/ENGINE-PLAN.md section 5 (kernel K11 and the architecture contract table),
// risk 1 quirk 4, risk 2, section 5.5's reduction rules, the ablation ledger in DECODE-CAMPAIGN.md
// 5.1, DECODE-FUSION-FINDING.md, the Hugging Face transformers reference implementation of the
// Gemma 4 decoder layer, and the WGSL specification. No vendored bundle, no extracted kernel and no
// third party engine source was read.
//
// SPDX-License-Identifier: Apache-2.0
//
// K11, the block epilogue: `dst = residual + rmsNorm(x) * w`, in one pass.
//
// WHAT THE REFERENCE ACTUALLY DOES, because the order of these three operations is a silent
// corruption hazard of the same family as ENGINE-PLAN risk 1. In the Gemma 4 decoder layer the
// residual is added **after** the post norm, not before it, at all three sites:
//
//     h = residual + post_attention_layernorm(attn_out)
//     h = residual + post_feedforward_layernorm(mlp_out)
//     h = residual + post_per_layer_input_norm(per_layer_projection_out)
//
// An implementation that adds the residual first and then norms the sum is the ordinary
// pre-norm transformer arrangement, it produces perfectly plausible text, and it is wrong at every
// block in the model. So one kernel serves all three sites, and the argument order is fixed here:
// `src` is the block output being normed, `residual` is the stream it is added back into.
//
// WHY IT IS FUSED. ENGINE-PLAN's fusion policy is precise about which fusions are in scope: K11 is
// "projection, norm and residual add in one pass", fused to avoid re-reading and re-writing the
// activation vector rather than to save a dispatch, because bandwidth is the M1's binding
// constraint and dispatch count measurably is not. The projection itself is the matmul lane's GEMV;
// this kernel is the norm and the add, which is the part that would otherwise cost two extra passes
// over the residual stream at every block. On the ablation ledger the down projection and norm pair
// was the second largest M1 cost centre at 5.13 ms against 1.13 on the 5070
// (DECODE-CAMPAIGN.md 5.1), which is why it is worth one pass rather than three.
//
// THE NORM FORM IS THE PLAIN ONE, trap 4 again: `x / rms(x) * w`, never `(1 + w)`. This file
// computes the norm itself rather than calling the norm kernel, which is what "one pass" means, so
// the trap has to be defended here too. It is: the k-mlp check runs this file's oracle against
// rmsNorm.ts's oracle plus an add and requires them to agree exactly, so the two implementations of
// the norm in this engine are pinned to each other on every run. The reduction shape is also the
// same one, taken from subgroupReduce.ts rather than rewritten, so the accumulation order that
// ENGINE-PLAN risk 2 mitigation 2 freezes is literally the same code.
//
// WHERE THE ACTIVATION ROUNDING IS, since this kernel sits immediately downstream of one and does
// not apply it. `down_proj` rounds its output with its own `output_activation_scale`, and that
// rounding belongs to `down_proj`, which is a quantized linear, so since round 2 it happens inside
// qgemv.ts's and qgemm.ts's fused epilogue and this kernel's `src` arrives already snapped. Same
// at the third call site, where `per_layer_projection` is the linear. Rounding again here would
// snap a value that is already on the grid, which is a no-op on good days and a second grid on bad
// ones. The reference capture agrees: `ref.l0.mlp.down` is what `QuantizedLinear.forward` returned
// with its output rounding included, and this file's oracle reproduces `ref.l0.mlp.out` from it.
//
// ACCUMULATORS ARE f32, NEVER f16, here as everywhere in this engine: LlamaWeb (arXiv 2605.20706)
// measured f16 accumulation producing incoherent output on Apple M-series GPUs, which is the
// hardware this ships on first. The sum of squares below is an f32 `var` reduced through
// subgroupReduce.ts, and the k-matmul lint greps every registered kernel's WGSL for an f16
// declaration so the rule is checked rather than remembered.
//
// WHAT THIS KERNEL IS NOT. Plain elementwise glue, `dst = alpha * x + y`, already exists as the
// `scale-add` kernel. Residual add without a norm is that kernel at alpha 1, and the per layer
// `layer_scalar` multiply is that kernel with a zero second operand. This file adds no second copy
// of either. If `scale-add` is ever retired as a harness canary, the engine still needs it, and
// that is a note for the lead rather than a reason to duplicate it here.

import type { Kernel, KernelBindInput, KernelBindResult } from './registry';
import { kernelLayout, kernelUniform } from './binding';
import {
  SUBGROUP_BUTTERFLY_WGSL,
  SUBGROUP_ENABLE,
  workgroupTreeWgsl,
} from './subgroupReduce';

/** 64 wide, 2 subgroups, the measured decode width on this hardware (DECODE-CAMPAIGN.md 4.2). */
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

/** 'workgroup' is the fallback for a device that fails the init subgroup self test. */
export type EpilogueReducePath = 'subgroup' | 'workgroup';

/**
 * Build the WGSL for the block epilogue.
 *
 * The structure is the norm kernel's, for the reason given above, and every rule it obeys is a rule
 * of ENGINE-PLAN 5.5 rather than a preference:
 *
 *  - The sum of squares is a `dot()` over vec4 lanes, because values flowing through `dot()` held
 *    bit identical across compiled modules for a full evening of probing while scalar chains
 *    drifted 1 to 2 ULP (DECODE-FUSION-FINDING.md).
 *  - Every reduction sits above every store, in uniform control flow, and the only lane divergent
 *    store in the kernel is the one slot per subgroup partial write, after which nothing reduces.
 *  - The tail reduction is the 32 lane butterfly, never a bare `subgroupAdd`.
 *  - The loop bound comes from the uniform, never from a large compile time constant.
 *  - Five bindings, four of them storage, against an adapter limit of 10.
 *
 * Aliasing: `dst` may be the same buffer as `src` or as `residual`. A lane only ever writes the
 * element it read, and the barrier inside the reduction orders every read of `src` before any
 * write, so both aliases are safe and both save an activation sized buffer.
 */
export function normResidualWgsl(reduce: EpilogueReducePath): string {
  const W = reduce === 'subgroup' ? WORKGROUP_SIZE : FALLBACK_WORKGROUP_SIZE;
  const SLOTS = slotsOf(W);
  const head = reduce === 'subgroup'
    ? `${SUBGROUP_ENABLE}\n${SUBGROUP_BUTTERFLY_WGSL}\nvar<workgroup> sgPartials: array<f32, ${SLOTS}>;\n`
    : workgroupTreeWgsl('treeScratch', W);

  const tail = reduce === 'subgroup'
    ? `  // Butterfly first, in uniform control flow, then one lane per 32 lane group stores its
  // slot. Reduce above store, never the other way round.
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
struct EpilogueParams {
  // Row width in vec4 lanes, 384 for the 1536 wide residual stream. Runtime opaque on purpose.
  vec4Count: u32,
  // Carried for the caller to validate against. The dispatch is one workgroup per row, so a bounds
  // guard would be dead code and an early return would put the barrier in non uniform control flow.
  rowCount: u32,
  eps: f32,
  pad0: u32,
}

@group(0) @binding(0) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> gain: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> residual: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(4) var<uniform> params: EpilogueParams;

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

${tail}

  // Plain Gemma 4 norm. No 1.0 is added to the weight anywhere in this file, and the k-mlp check
  // fails if one appears.
  let inv = inverseSqrt(total / f32(n4 * 4u) + params.eps);

  for (var i = lid; i < n4; i = i + ${W}u) {
    // The residual is added after the norm, which is the order the reference uses and the order a
    // pre-norm transformer habit would get backwards.
    dst[base + i] = residual[base + i] + src[base + i] * inv * gain[i];
  }
}
`;
}

export const NORM_RESIDUAL_WGSL = normResidualWgsl('subgroup');
export const NORM_RESIDUAL_FALLBACK_WGSL = normResidualWgsl('workgroup');

// ---------------------------------------------------------------------------------------------
// The CPU oracle.
// ---------------------------------------------------------------------------------------------

/**
 * The known answer reference for the block epilogue.
 *
 * Accumulates the sum of squares in f64 and rounds once at the end, so it is a reference rather
 * than a mirror of the shader's f32 reduction, and the case states a tolerance somebody argued for
 * (ENGINE-PLAN section 7, and the registry contract).
 *
 * @param src      the block output, `rows * width` values, row major
 * @param gain     `width` norm weights, applied inside every row
 * @param residual `rows * width` values, the stream this is added back into
 * @param rows     one workgroup per row on the GPU
 * @param width    1536 at every site in this model
 * @param eps      1e-6 from config.json
 */
export function normResidualOracle(
  src: Float32Array,
  gain: Float32Array,
  residual: Float32Array,
  rows: number,
  width: number,
  eps: number,
): Float32Array {
  const need = rows * width;
  if (src.length < need) throw new Error(`normResidualOracle: src holds ${src.length}, needs ${need}`);
  if (residual.length < need) {
    throw new Error(`normResidualOracle: residual holds ${residual.length}, needs ${need}`);
  }
  if (gain.length < width) throw new Error(`normResidualOracle: gain holds ${gain.length}, needs ${width}`);

  const out = new Float32Array(need);
  for (let r = 0; r < rows; r += 1) {
    const base = r * width;
    let sum = 0;
    for (let i = 0; i < width; i += 1) {
      const v = src[base + i];
      sum += v * v;
    }
    const inv = 1 / Math.sqrt(sum / width + eps);
    for (let i = 0; i < width; i += 1) {
      out[base + i] = residual[base + i] + src[base + i] * inv * gain[i];
    }
  }
  return out;
}

/**
 * The wrong order, kept beside the right one for the same reason rmsNorm.ts keeps the Gemma 3 norm
 * form: so the check can prove the two are distinguishable and that this engine shipped the one the
 * reference uses. This is `norm(residual + x)`, the ordinary pre-norm arrangement. Nothing calls it.
 */
export function residualThenNormOracle(
  src: Float32Array,
  gain: Float32Array,
  residual: Float32Array,
  rows: number,
  width: number,
  eps: number,
): Float32Array {
  const sum = new Float32Array(rows * width);
  for (let i = 0; i < sum.length; i += 1) sum[i] = residual[i] + src[i];
  const out = new Float32Array(rows * width);
  for (let r = 0; r < rows; r += 1) {
    const base = r * width;
    let sq = 0;
    for (let i = 0; i < width; i += 1) sq += sum[base + i] * sum[base + i];
    const inv = 1 / Math.sqrt(sq / width + eps);
    for (let i = 0; i < width; i += 1) out[base + i] = sum[base + i] * inv * gain[i];
  }
  return out;
}

// ---------------------------------------------------------------------------------------------
// Registry entry.
// ---------------------------------------------------------------------------------------------

/** The params block, exported so the scheduler can restage it in place per dispatch. */
export function epilogueParams(vec4Count: number, rowCount: number, eps: number): ArrayBuffer {
  const words = new ArrayBuffer(16);
  const u = new Uint32Array(words);
  const f = new Float32Array(words);
  u[0] = vec4Count;
  u[1] = rowCount;
  f[2] = eps;
  u[3] = 0;
  return words;
}

function bind(input: KernelBindInput): KernelBindResult {
  const { device, inputs, output, params } = input;
  const rows = params.rows | 0;
  const width = params.width | 0;
  const eps = params.eps ?? 1e-6;
  if (rows <= 0) throw new Error('norm-residual needs params.rows');
  if (width <= 0 || width % 4 !== 0) {
    throw new Error(`norm-residual needs params.width a positive multiple of 4, got ${width}`);
  }
  const src = inputs.src;
  const gain = inputs.gain;
  const residual = inputs.residual;
  if (!src || !gain || !residual) {
    throw new Error('norm-residual needs inputs named src, gain and residual');
  }

  const layout = kernelLayout(input, 'norm-residual', () => device.createBindGroupLayout({
    label: 'norm-residual',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  }));
  const uniform = kernelUniform(input, 'norm-residual params', epilogueParams(width / 4, rows, eps));

  return {
    layout,
    buffers: [src, gain, residual, output, uniform.binding],
    dispatch: [rows, 1, 1],
    dispose: uniform.dispose,
  };
}

// The tolerance argument, for both cases.
//
// The fixture makes the reduction itself exact: every `src` value is a multiple of 1/2 with
// magnitude at most 4, so every square is a multiple of 1/4 and the running total over 1536 terms
// stays far inside the range where a multiple of 1/4 is exact in f32, whatever order it runs in.
// What is left is `inverseSqrt`, which WGSL gives 2 ULP, two multiplies and one add, correctly
// rounded, at one ULP each. On a result that does not cancel, 8 ULP covers that.
//
// The absolute number is the one that matters where it does cancel. `residual` and the normed term
// are both bounded by 4 in the fixture, so a near cancellation leaves a small result carrying the
// absolute error of the terms that made it, which is at most about 6 ULP of 4, or 3e-6. 4e-6 is
// that with a little room. The two tolerances are an OR in the harness, so the ULP gate carries the
// ordinary elements and the absolute gate carries the cancelling ones, and neither is loose enough
// to let a wrong norm form or a missing residual through, both of which move the answer by order 1.
const EPILOGUE_TOL_ULP = 8;
const EPILOGUE_TOL_ABS = 4e-6;

export const normResidualKernel: Kernel = {
  name: 'norm-residual',
  wgsl: NORM_RESIDUAL_WGSL,
  entry: 'main',
  note:
    'K11 epilogue. residual + rmsNorm(x) * w in one pass, plain Gemma 4 norm form. Serves the post '
    + 'attention, post feedforward and post per layer input sites, all three of which norm before '
    + 'they add.',
  cases: [
    {
      name: 'decode-1536',
      inputs: {
        src: 'kmlp.epi.src',
        gain: 'kmlp.epi.gain',
        residual: 'kmlp.epi.residual',
      },
      expected: 'kmlp.epi.expected',
      params: { rows: 1, width: 1536, eps: 1e-6 },
      tolUlp: EPILOGUE_TOL_ULP,
      tolAbs: EPILOGUE_TOL_ABS,
      note: 'One position, the decode shape. Width is 1536 at every site in this model.',
    },
    // Google's own numbers, at both MLP kinds. `src` is what the reference's down_proj returned at
    // one position of a real prompt, `gain` is the checkpoint's own post feed forward norm weight,
    // `residual` is the stream the reference added back, and the expected buffer is its own sum
    // (.engine-ref/tools/mlp-ref.py). A kernel that normed the sum instead of summing the norm
    // fails these by a factor of order one, which is the whole point of having them.
    {
      name: 'reference-layer0',
      inputs: {
        src: 'ref.l0.mlp.down',
        gain: 'ref.l0.post_feedforward_layernorm.weight',
        residual: 'ref.l0.mlp.residual',
      },
      expected: 'ref.l0.mlp.out',
      params: { rows: 1, width: 1536, eps: 1e-6 },
      tolAbs: 3e-4,
      tolUlp: 8,
      note:
        'Layer 0. The tolerance is about a millionth of this tensor\'s peak of 293, with an 8 ULP '
        + 'budget beside it for the elements near zero. The reduction order differs between this '
        + 'kernel and the reference forward, so the absolute half is the one that carries the large '
        + 'elements.',
    },
    {
      name: 'reference-layer15',
      inputs: {
        src: 'ref.l15.mlp.down',
        gain: 'ref.l15.post_feedforward_layernorm.weight',
        residual: 'ref.l15.mlp.residual',
      },
      expected: 'ref.l15.mlp.out',
      params: { rows: 1, width: 1536, eps: 1e-6 },
      tolAbs: 3e-5,
      tolUlp: 8,
      note: 'Layer 15, a consumer layer. Same reasoning, against a peak of 29.5.',
    },
    {
      name: 'prefill-4x1536',
      inputs: {
        src: 'kmlp.epi.src4',
        gain: 'kmlp.epi.gain',
        residual: 'kmlp.epi.residual4',
      },
      expected: 'kmlp.epi.expected4',
      params: { rows: 4, width: 1536, eps: 1e-6 },
      tolUlp: EPILOGUE_TOL_ULP,
      tolAbs: EPILOGUE_TOL_ABS,
      note:
        'Four positions in one dispatch, which is the prefill shape and the case that would catch '
        + 'a row stride read as a constant.',
    },
  ],
  bind,
};
