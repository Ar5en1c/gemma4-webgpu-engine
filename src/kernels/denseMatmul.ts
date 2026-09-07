// Written from docs/ENGINE-PLAN.md sections 3 and 5 (K12), this engine's own quant.ts (the
// `plmp-bf16` family and `bf16ToF32`), the transformers reference text model's
// `project_per_layer_inputs` (Apache-2.0, an allowed source), and the WGSL specification. No
// vendored bundle, no extracted kernel and no third party engine source was read.
//
// SPDX-License-Identifier: Apache-2.0
//
// The one unquantized linear in the text stack: `model.language_model.per_layer_model_projection`,
// a plain [8960, 1536] matrix that the checkpoint's `modules_to_not_convert` list keeps out of the
// QAT scheme entirely. It is the context aware half of the per layer input, and until this kernel
// existed the engine had no way to multiply a matrix that was not packed codes plus a row scale.
//
// WHAT THE CHECKPOINT ACTUALLY STORES, checked rather than assumed. `tensors.json` carries exactly
// one tensor under this module, `per_layer_model_projection.weight`, dtype BF16, shape
// [8960, 1536], 27,525,120 bytes. There is no `weight_scale`, no `input_activation_scale` and no
// `output_activation_scale`, and `resolveQuantBits` reports `num_bits` null for it. So this kernel
// applies NO static range quantization: not because SRQ was inconvenient here, but because the
// module the reference builds for it has no scales to apply and rounding its input onto a grid
// nobody calibrated would be an invention. If a later revision of the checkpoint grows those
// scalars, this is the file that has to grow the prologue and epilogue with them.
//
// WHERE THE BF16 SHIFT HAPPENS, and why it is not in this shader. BF16 is the top 16 bits of the
// f32 pattern, so widening is a shift and is exact for every value. WGSL has no BF16 storage type
// at all, so the widening has to happen before the bytes reach a binding, and engine.ts's loader
// sink already does it for every BF16 tensor in the checkpoint through quant.ts `bf16ToF32`: the
// norm gains, the layer scalars and this matrix all arrive as f32. This kernel therefore binds
// `array<vec4<f32>>` and does no unpacking, which is the same seam the norm kernels already use
// and one fewer place for a layout to be restated. The cost is honest and worth writing down: 27.5
// MB on the wire becomes 55 MB resident.
//
// THE SCALE RIDES WITH THE DISPATCH. The reference computes
// `per_layer_model_projection(inputs_embeds) * hidden_size ** -0.5` and only then reshapes and
// norms, and RMS norm is not scale invariant once its epsilon is in play, so the multiply cannot be
// deferred past the norm. plan.ts planEmbed says the scale rides with this step rather than as a
// separate scale-add, so `alpha` is a uniform word here and the step costs one dispatch rather than
// two. `alpha` is one correctly rounded f32 multiply on both sides, so an irrational alpha does not
// cost the cases their bit identity.
//
// SHAPE. The same one invocation per output element geometry as pleMatmul.ts, for the same reason
// and with the same transposed but fully consumed access pattern: 64 lanes read word i of 64
// different rows, and those 64 cache lines then serve the next 31 iterations. No workgroup memory,
// no barriers, no subgroup calls, so no fallback build and no reduce policy sensitivity. Both step
// modes run this one kernel, a decode step at mCols 1.
//
// ACCUMULATORS ARE f32, NEVER f16: LlamaWeb (arXiv 2605.20706) measured f16 accumulation producing
// incoherent output on Apple M-series GPUs, and this matrix's 1536 term reduction is exactly the
// length where that shows up first.

import type { Kernel, KernelBindInput, KernelBindResult } from './registry';
import { kernelLayout, kernelUniform } from './binding';

/** One workgroup covers 64 output rows of one token column. */
export const DENSE_WORKGROUP_SIZE = 64;

/**
 * `hidden_size ** -0.5`, the reference's `per_layer_model_projection_scale`, rounded to the f32 the
 * uniform block will actually carry so the oracle and the shader multiply by the same number. The
 * architectural value lives in plan.ts `perLayerProjectionScale`; this is that value at the width
 * the GPU sees, and the k-ple section asserts the two agree.
 */
export const PLMP_SCALE_F32 = Math.fround(Math.pow(1536, -0.5));

/**
 * The one WGSL text. No reduction crosses a lane, so there is no subgroup variant and no workgroup
 * variant, and no entry in pipeline.ts FALLBACK_WGSL is needed or wanted.
 */
export const DENSE_MATMUL_WGSL = /* wgsl */ `
struct DenseParams {
  // vec4 lanes per weight row, K / 4. Also the vec4 stride of one activation column.
  kVec4: u32,
  numRows: u32,
  // Token columns. 1 on a decode step, the chunk length on a prefill chunk.
  mCols: u32,
  pad0: u32,
  // The architectural multiplier applied to this linear's output. hidden_size ** -0.5 at the per
  // layer model projection, 1.0 anywhere a caller wants the bare product.
  alpha: f32,
  pad1: u32,
  pad2: u32,
  pad3: u32,
}

@group(0) @binding(0) var<storage, read> w: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<uniform> params: DenseParams;

// The weight arrives as the checkpoint's own BF16 bytes, four values to a vec2<u32>, low half
// first because the file is little endian. A BF16 value is the top sixteen bits of its f32
// pattern (quant.ts bf16ToF32), so widening is one shift and the widened value is bit for bit
// the one the loader used to hand over already widened: the kernel reads half the bytes and
// computes the same numbers.
fn bf16x4(p: vec2<u32>) -> vec4<f32> {
  return vec4<f32>(
    bitcast<f32>(p.x << 16u),
    bitcast<f32>(p.x & 0xffff0000u),
    bitcast<f32>(p.y << 16u),
    bitcast<f32>(p.y & 0xffff0000u),
  );
}

@compute @workgroup_size(${DENSE_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_index) lid: u32,
) {
  // One invocation per output element. The early return is a plain guard with no barrier, no
  // reduction and no subgroup call below it.
  let row = wid.x * ${DENSE_WORKGROUP_SIZE}u + lid;
  let col = wid.y;
  if (row >= params.numRows || col >= params.mCols) {
    return;
  }
  let base = row * params.kVec4;
  let xb = col * params.kVec4;

  // f32 accumulator, never a half precision one: LlamaWeb (arXiv 2605.20706) measured half
  // precision accumulation producing incoherent output on M-series. Every product runs through dot()
  // intrinsic because values flowing through dot() held bit identical across compiled modules
  // where scalar multiply add chains drifted 1 to 2 ULP (DECODE-FUSION-FINDING.md, risk 2).
  var acc = 0.0;
  // The loop bound comes from the uniform block, never from a compile time constant: Metal
  // unrolled a constant bound 256 iteration loop into a multi second compile once already
  // (DECODE-FUSION-FINDING.md item 5), and this one runs 384 iterations at the real shape.
  for (var i = 0u; i < params.kVec4; i = i + 1u) {
    acc = acc + dot(bf16x4(w[base + i]), x[xb + i]);
  }

  dst[col * params.numRows + row] = params.alpha * acc;
}
`;

/** Four-column prefill source screened by scripts/mac-perf/dense-tiled.ts. */
export function denseMatmulTiledWgsl(): string {
  const marker = '@compute';
  const at = DENSE_MATMUL_WGSL.indexOf(marker);
  if (at < 0 || at !== DENSE_MATMUL_WGSL.lastIndexOf(marker)) {
    throw new Error('dense tiled source contract requires exactly one @compute marker');
  }
  const prefix = DENSE_MATMUL_WGSL.slice(0, at);

  return /* wgsl */ `${prefix}@compute @workgroup_size(${DENSE_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_index) lid: u32,
) {
  let row = wid.x * ${DENSE_WORKGROUP_SIZE}u + lid;

  // Decode keeps the original scalar arithmetic and does no tiled work.
  if (params.mCols == 1u) {
    let col = wid.y;
    if (row >= params.numRows || col >= params.mCols) {
      return;
    }
    let base = row * params.kVec4;
    let xb = col * params.kVec4;
    var acc = 0.0;
    for (var i = 0u; i < params.kVec4; i = i + 1u) {
      acc = acc + dot(bf16x4(w[base + i]), x[xb + i]);
    }
    dst[col * params.numRows + row] = params.alpha * acc;
    return;
  }

  let colBase = wid.y * 4u;
  if (row >= params.numRows || colBase >= params.mCols) {
    return;
  }
  let base = row * params.kVec4;
  let cLast = params.mCols - 1u;
  let xb0 = min(colBase, cLast) * params.kVec4;
  let xb1 = min(colBase + 1u, cLast) * params.kVec4;
  let xb2 = min(colBase + 2u, cLast) * params.kVec4;
  let xb3 = min(colBase + 3u, cLast) * params.kVec4;
  var acc0 = 0.0;
  var acc1 = 0.0;
  var acc2 = 0.0;
  var acc3 = 0.0;

  for (var i = 0u; i < params.kVec4; i = i + 1u) {
    let weight = bf16x4(w[base + i]);
    acc0 = acc0 + dot(weight, x[xb0 + i]);
    acc1 = acc1 + dot(weight, x[xb1 + i]);
    acc2 = acc2 + dot(weight, x[xb2 + i]);
    acc3 = acc3 + dot(weight, x[xb3 + i]);
  }

  if (colBase < params.mCols) {
    dst[colBase * params.numRows + row] = params.alpha * acc0;
  }
  if (colBase + 1u < params.mCols) {
    dst[(colBase + 1u) * params.numRows + row] = params.alpha * acc1;
  }
  if (colBase + 2u < params.mCols) {
    dst[(colBase + 2u) * params.numRows + row] = params.alpha * acc2;
  }
  if (colBase + 3u < params.mCols) {
    dst[(colBase + 3u) * params.numRows + row] = params.alpha * acc3;
  }
}
`;
}

// ---------------------------------------------------------------------------------------------
// The CPU oracle.
// ---------------------------------------------------------------------------------------------

/**
 * The known answer reference for the dense linear, in the shader's own expression order: four
 * products and three adds per vec4, one add into the accumulator, then one multiply by alpha, all
 * through `Math.fround` so the CPU carries f32 rounding rather than double rounding.
 *
 * WHERE THIS IS A MIRROR AND WHERE IT IS A REFERENCE, because the two are different claims. On a
 * fixture whose weights and activations both sit on a coarse dyadic grid, every product and every
 * partial sum is exactly representable in f32, so the answer does not depend on the order or on
 * whether the compiler contracts a multiply and an add into an FMA, and such a case gates at bit
 * identity. On real BF16 weights the products are still exact, because a BF16 value has an 8 bit
 * mantissa and the grid activations have three, but the running sum over K terms of widely
 * different exponents does round, and there an FMA contraction inside `dot()` can land a different
 * last bit. A case over real weights therefore states a ULP tolerance and says this is why.
 */
export function denseMatmulOracle(
  weights: Float32Array,
  x: Float32Array,
  numRows: number,
  k: number,
  mCols: number,
  alpha = 1,
): Float32Array {
  if (k % 4 !== 0) throw new Error(`denseMatmulOracle: K ${k} is not a multiple of 4`);
  if (weights.length < numRows * k) {
    throw new Error(`denseMatmulOracle: weights hold ${weights.length}, needs ${numRows * k}`);
  }
  if (x.length < mCols * k) throw new Error(`denseMatmulOracle: x holds ${x.length}, needs ${mCols * k}`);
  const a = Math.fround(Number.isFinite(alpha) ? alpha : 1);
  const out = new Float32Array(mCols * numRows);
  for (let col = 0; col < mCols; col += 1) {
    const xCol = x.subarray(col * k, (col + 1) * k);
    for (let r = 0; r < numRows; r += 1) {
      const rowBase = r * k;
      let sum = 0;
      for (let w = 0; w < k; w += 4) {
        let d = 0;
        for (let j = 0; j < 4; j += 1) {
          d = Math.fround(d + Math.fround(weights[rowBase + w + j]! * xCol[w + j]!));
        }
        sum = Math.fround(sum + d);
      }
      out[col * numRows + r] = Math.fround(a * sum);
    }
  }
  return out;
}

// ---------------------------------------------------------------------------------------------
// The registry entry.
// ---------------------------------------------------------------------------------------------

/** The params block, exported so the scheduler can restage it in place per dispatch. */
export function denseParams(
  kVec4: number,
  numRows: number,
  mCols: number,
  alpha = 1,
): ArrayBuffer {
  const buf = new ArrayBuffer(32);
  const u = new Uint32Array(buf);
  const f = new Float32Array(buf);
  u[0] = kVec4;
  u[1] = numRows;
  u[2] = mCols;
  u[3] = 0;
  f[4] = alpha;
  u[5] = 0;
  u[6] = 0;
  u[7] = 0;
  return buf;
}

function bindDenseMatmul(input: KernelBindInput): KernelBindResult {
  const { device, inputs, output, params } = input;
  const k = params.k | 0;
  const numRows = params.numRows | 0;
  const mCols = (params.mCols ?? 1) | 0;
  if (k <= 0 || numRows <= 0 || mCols <= 0) {
    throw new Error('dense-bf16-matmul needs params.k and params.numRows, with params.mCols absent for a decode step');
  }
  if (k % 4 !== 0) {
    throw new Error(
      `dense-bf16-matmul needs K a multiple of 4, got ${k}. The real tensor is K 1536.`,
    );
  }
  const alpha = params.alpha ?? 1;
  const w = inputs.w;
  const x = inputs.x;
  if (!w || !x) throw new Error('dense-bf16-matmul needs inputs named w and x');

  const layout = kernelLayout(input, 'dense-bf16-matmul', () => device.createBindGroupLayout({
    label: 'dense-bf16-matmul',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  }));
  const uniform = kernelUniform(
    input,
    'dense-bf16-matmul params',
    denseParams(k / 4, numRows, mCols, alpha),
  );

  return {
    layout,
    buffers: [w, x, output, uniform.binding],
    dispatch: [Math.ceil(numRows / DENSE_WORKGROUP_SIZE), mCols, 1],
    dispose: uniform.dispose,
  };
}

function bindDenseMatmulPrefill4(input: KernelBindInput): KernelBindResult {
  const bound = bindDenseMatmul(input);
  const mCols = (input.params.mCols ?? 1) | 0;
  return { ...bound, dispatch: [bound.dispatch[0], Math.ceil(mCols / 4), bound.dispatch[2]] };
}

export const denseMatmulKernel: Kernel = {
  name: 'dense-bf16-matmul',
  wgsl: DENSE_MATMUL_WGSL,
  entry: 'main',
  note:
    'K12\'s unquantized half: model.language_model.per_layer_model_projection, a plain [8960, 1536] '
    + 'BF16 linear in the checkpoint\'s modules_to_not_convert list, widened to f32 by the loader '
    + 'because WGSL has no BF16 storage type. No SRQ, because the checkpoint stores no activation '
    + 'scales for this module. The hidden_size ** -0.5 scale rides with the dispatch.',
  cases: [
    {
      name: 'synthetic-12x256-m3',
      inputs: { w: 'kple.dense.w.bf16', x: 'kple.dense.x' },
      expected: 'kple.dense.expected',
      params: { k: 256, numRows: 12, mCols: 3 },
      tolAbs: 0,
      tolUlp: 0,
      note:
        'Twelve rows against a workgroup of 64 exercise the row guard, three columns the column '
        + 'guard. Weights are on the 1/4 grid and activations on the 1/8 grid with bounded '
        + 'magnitude, so every product and every partial sum is exact in f32 whatever the '
        + 'accumulation order or FMA contraction, and the gate is bit identity.',
    },
    {
      name: 'synthetic-12x256-m3-scaled',
      inputs: { w: 'kple.dense.w.bf16', x: 'kple.dense.x' },
      expected: 'kple.dense.scaled.expected',
      params: { k: 256, numRows: 12, mCols: 3, alpha: PLMP_SCALE_F32 },
      tolAbs: 0,
      tolUlp: 0,
      note:
        'The same tile with the real hidden_size ** -0.5 multiplier. The gate stays at zero '
        + 'despite an irrational alpha because the reduction is exact on both sides and one f32 '
        + 'multiply is correctly rounded on both sides.',
    },
    {
      name: 'real-plmp-slice',
      inputs: { w: 'tensor.plmp-bf16.raw', x: 'kple.dense.x' },
      expected: 'kple.dense.real.expected',
      params: { k: 64, numRows: 1, mCols: 1, alpha: PLMP_SCALE_F32 },
      tolAbs: 0,
      tolUlp: 4,
      note:
        'Sixty four real widened BF16 values from row 0 of per_layer_model_projection, against the '
        + 'grid activation and the real scale. Unlike the synthetic cases this one cannot gate at '
        + 'zero and says why: the products are exact, but a sum of sixteen partials at widely '
        + 'different exponents rounds, so an FMA contraction inside the shader\'s dot() can move '
        + 'the last bits. Four ULP is the allowance; the observed difference on the M1 is recorded '
        + 'in the run notes rather than assumed to be zero.',
    },
  ],
  bind: bindDenseMatmul,
};

export const denseMatmulPrefill4Kernel: Kernel = {
  ...denseMatmulKernel,
  name: 'dense-bf16-prefill4',
  wgsl: denseMatmulTiledWgsl(),
  cases: [...denseMatmulKernel.cases],
  bind: bindDenseMatmulPrefill4,
};
