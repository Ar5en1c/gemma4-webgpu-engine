// Written from docs/ENGINE-PLAN.md sections 5 (K12) and 5.5, this engine's own quant.ts (the I8
// `ple-gate-8bit` family) and qlayout.ts, the round 2 ratified integer SRQ semantics as qgemv.ts
// and qgemm.ts implement them, the transformers reference decoder layer (Apache-2.0, an allowed
// source), and the WGSL specification. No vendored bundle, no extracted kernel and no third party
// engine source was read.
//
// SPDX-License-Identifier: Apache-2.0
//
// K12's two linears: the 8-bit quantized matmul serving `per_layer_input_gate` [256, 1536] and
// `per_layer_projection` [1536, 256], quant.ts's `ple-gate-8bit` family. One signed byte per code,
// no packing, no zero point: the byte IS the code, and pointing the 2-bit or 4-bit unpacker at
// these bytes produces plausible garbage rather than an error, which is why the executor refused
// these two steps by name until this kernel existed.
//
// SHAPE, and why it is not the qgemv/qgemm geometry. Those two stream hundreds of megabytes per
// token and their 64 wide, two virtual subgroup, four row workgroup with a subgroup reduction per
// row is the measured answer to that (DECODE-CAMPAIGN.md 4.2). This family moves 0.79 MB per layer,
// about 27.5 MB per token over all 35 layers against a roughly 0.6 GB token, so the design budget
// here is correctness surface rather than bandwidth: one invocation per output element, the whole K
// reduction serial in registers, no workgroup memory, no barriers, no subgroup calls, and therefore
// no fallback build and no reduce policy sensitivity at all. Both step modes run this one kernel; a
// decode step is simply mCols = 1.
//
// The access pattern that shape produces is transposed rather than wasteful, and it is worth saying
// why it is acceptable. At iteration i the 64 lanes of a workgroup read word i of 64 different rows,
// which is 64 separate cache lines rather than one. Those same 64 lines then serve iterations i+1
// through i+31, because a 128 byte line holds 32 words of one row. So the working set is 64 lines,
// 8 KB, and every byte fetched is consumed: the traffic is the tensor, once, and only the order
// differs from the GEMV's. If the 5070 round ever measures this kernel on the wrong side of a
// ledger, the qgemm tile shape is sitting right there in the file next door.
//
// STATIC RANGE QUANTIZATION, both sides, exactly as qgemv.ts fuses it: a calibrated site snaps the
// activation onto the int8 grid and reduces K as the exact integer code dot, with one closing
// multiply by input_scale times weight_scale; an uncalibrated site keeps the f32 code dot. Both of
// these modules are calibrated on every layer, so the integer path is the one that ships and the
// f32 path exists to keep one kernel honest about the other family's uncalibrated site.
//
// WHERE THE i32 ACCUMULATION IS EXACT AND WHERE IT MERELY ROUNDS ONCE. The reduction itself never
// rounds: the worst case for this family is K 1536 with both sides at the int8 limit, bounded by
// 1536 * 128 * 128 = 2.52e7, far inside i32. That bound does exceed f32's 2^24 exact integer range,
// so the single `f32(acc)` handoff at the end can round by one unit in 2.5e7, which is a relative
// error of 4e-8, below f32 epsilon and far below what an f32 dot chain over 1536 terms would
// accumulate. The fixtures below stay inside 2^24, so their cases still gate at bit identity, and
// the CPU oracle mirrors the handoff with Math.fround so the two agree past 2^24 as well.
//
// ACCUMULATORS ARE i32 OR f32, NEVER f16: LlamaWeb (arXiv 2605.20706) measured f16 accumulation
// producing incoherent output on Apple M-series GPUs. f16 in this engine is a storage and cast
// dtype only, and the k-matmul lint greps every registered kernel for an f16 declaration.

import type { Kernel, KernelBindInput, KernelBindResult } from './registry';
import { kernelLayout, kernelUniform } from './binding';
import { bytesAsWords, WGSL_UNPACK8, WGSL_UNPACK8_I, wordsPerRow, unpackWordsLikeShader } from './qlayout';
import { applySrq, srqCodes } from '../quant';

/** One workgroup covers 64 output rows of one token column. */
export const QMATMUL8_WORKGROUP_SIZE = 64;

/**
 * The one WGSL text. No reduction crosses a lane, so there is no subgroup variant and no workgroup
 * variant: the same module is correct under either engine reduce policy, which is why this kernel
 * needs no entry in pipeline.ts FALLBACK_WGSL and why the orchestrator's fallback coverage check
 * stays green without one.
 */
export const QMATMUL8_WGSL = /* wgsl */ `
struct Qm8Params {
  // u32 words per weight row, K / 4. Also the vec4 stride of one activation column, because four
  // 8-bit codes and four f32 activations cover the same K span.
  kWords: u32,
  numRows: u32,
  // Token columns. 1 on a decode step, the chunk length on a prefill chunk.
  mCols: u32,
  pad0: u32,
  // The module's input_activation_scale and output_activation_scale, quant.ts applySrq semantics:
  // 8 bits on both sides whatever the weight width is, skipped only when the stored scale is
  // exactly 0.0. Same two words, in the same role, that GemvParams and GemmParams carry.
  inScale: f32,
  outScale: f32,
  pad1: u32,
  pad2: u32,
}

@group(0) @binding(0) var<storage, read> wq: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;
@group(0) @binding(4) var<uniform> params: Qm8Params;

// int8, at every static range quantization site in this checkpoint whatever the weight width is
// (quant.ts SRQ_ACTIVATION_BITS).
const SRQ_MAX: f32 = 127.0;
const SRQ_MIN: f32 = -128.0;

// The same SRQ prologue and epilogue expressions qgemv.ts and qgemm.ts carry, because they are the
// same operation on the same kind of scales and a second spelling would be a second thing to keep
// right (quant.ts srqCodes and applySrq). round() is ties to even on both sides.
fn srqIn(v: vec4<f32>) -> vec4<i32> {
  return vec4<i32>(clamp(round(v / params.inScale), vec4<f32>(SRQ_MIN), vec4<f32>(SRQ_MAX)));
}

fn srqOut(v: f32) -> f32 {
  if (params.outScale == 0.0) {
    return v;
  }
  return clamp(round(v / params.outScale), SRQ_MIN, SRQ_MAX) * params.outScale;
}

@compute @workgroup_size(${QMATMUL8_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_index) lid: u32,
) {
  // One invocation per output element: wid.x tiles the rows, wid.y is the token column. The early
  // return is a plain guard with no barrier, no reduction and no subgroup call anywhere below it,
  // so it is legal where a reducing matmul's store guard would not be (ENGINE-PLAN 5.5 rule 1 has
  // nothing to bite on here).
  let row = wid.x * ${QMATMUL8_WORKGROUP_SIZE}u + lid;
  let col = wid.y;
  if (row >= params.numRows || col >= params.mCols) {
    return;
  }
  let base = row * params.kWords;
  let xb = col * params.kWords;

  // f32 or i32 accumulators only, never a half precision one: LlamaWeb (arXiv
  // 2605.20706) measured half precision accumulation producing incoherent output on M-series.
  var acc = 0.0;
  // Uniform branch on a uniform buffer value, the same shape the other two matmul families take.
  if (params.inScale != 0.0) {
    // The ratified integer path: codes times codes through WGSL's integer dot(), summed in i32
    // with no rounding in the reduction at all.
    var acci: i32 = 0;
    for (var i = 0u; i < params.kWords; i = i + 1u) {
      let w = wq[base + i];
${WGSL_UNPACK8_I}
      acci = acci + dot(w8i, srqIn(x[xb + i]));
    }
    acc = f32(acci);
  } else {
    // The uncalibrated f32 code dot, serial per invocation, in the order the oracle mirrors.
    for (var i = 0u; i < params.kWords; i = i + 1u) {
      let w = wq[base + i];
${WGSL_UNPACK8}
      acc = acc + dot(w8, x[xb + i]);
    }
  }

  // One closing scale multiply, the same expression qgemv.ts ends on: on the integer path the
  // input scale folds into the row's weight scale first, and on the f32 path it is 1.0, so the
  // uncalibrated arithmetic is bit for bit a plain scale times the dot.
  let xs = select(1.0, params.inScale, params.inScale != 0.0);
  dst[col * params.numRows + row] = srqOut((xs * scales[row]) * acc);
}
`;

// ---------------------------------------------------------------------------------------------
// The CPU oracle.
// ---------------------------------------------------------------------------------------------

/**
 * The known answer reference for the 8-bit matmul, in the shader's own expression order, and the
 * definition the shader implements rather than the other way round.
 *
 * A calibrated site (`inScale` not exactly 0.0) is the ratified integer semantics: snap each token
 * column of x onto the int8 grid through quant.ts's `srqCodes`, sum `xcode * wcode` over K exactly,
 * hand the integer to f32 once, and multiply by `inScale * weight_scale[row]`. The sum is taken in
 * a JavaScript double, which is exact for every integer this family can produce, and `Math.fround`
 * then reproduces the shader's `f32(acc)` bit for bit including above 2^24.
 *
 * An uncalibrated site is the serial f32 code dot in the same left to right order the shader's loop
 * takes, with each dot's four products and three adds mirrored through `Math.fround`.
 *
 * Either way `outScale` rounds each written element through the one `applySrq` implementation the
 * loader and the reference share. On grid fixtures every partial sum is exactly representable, so
 * the cases gate at zero on both axes; on real checkpoint bytes the case says what it allows.
 */
export function qmatmul8Oracle(
  packed: Uint8Array,
  scales: ArrayLike<number>,
  x: Float32Array,
  numRows: number,
  k: number,
  mCols: number,
  inScale = 0,
  outScale = 0,
): Float32Array {
  const kWords = wordsPerRow(8, k);
  if (x.length < mCols * k) throw new Error(`qmatmul8Oracle: x holds ${x.length}, needs ${mCols * k}`);
  const words = bytesAsWords(packed);
  if (words.length < numRows * kWords) {
    throw new Error(`qmatmul8Oracle: packed holds ${words.length} words, needs ${numRows * kWords}`);
  }
  const calibrated = Number.isFinite(inScale) && inScale !== 0;
  const out = new Float32Array(mCols * numRows);
  // Codes are unpacked once per row rather than once per (row, column), which is the same numbers
  // and keeps a 1536 row oracle over a real slice from being quadratic in the tile.
  const rowCodes: Int8Array[] = [];
  for (let r = 0; r < numRows; r += 1) {
    rowCodes.push(unpackWordsLikeShader(8, words.subarray(r * kWords, (r + 1) * kWords), k));
  }
  for (let col = 0; col < mCols; col += 1) {
    const xCol = x.subarray(col * k, (col + 1) * k);
    const xCodes = calibrated ? srqCodes(xCol, inScale) : null;
    for (let r = 0; r < numRows; r += 1) {
      const codes = rowCodes[r]!;
      const rawScale = Number(scales[r] ?? 0);
      const scale = Number.isFinite(rawScale) ? rawScale : 0;
      let sum = 0;
      if (xCodes) {
        // Exact integer arithmetic: products are bounded by 128 * 128 and K by 1536, so the double
        // holds the exact integer and the fround below is the shader's own f32(acc).
        for (let i = 0; i < k; i += 1) sum += codes[i]! * xCodes[i]!;
        sum = Math.fround(sum);
      } else {
        // The shader's serial vec4 loop: dot() is four multiplies and three adds per word, then
        // one add into the accumulator, all in f32.
        for (let w = 0; w < kWords; w += 1) {
          let d = 0;
          for (let j = 0; j < 4; j += 1) {
            d = Math.fround(d + Math.fround(codes[w * 4 + j]! * xCol[w * 4 + j]!));
          }
          sum = Math.fround(sum + d);
        }
      }
      const xs = calibrated ? inScale : 1;
      out[col * numRows + r] = Math.fround(Math.fround(xs * scale) * sum);
    }
  }
  return applySrq(out, Number.isFinite(outScale) ? outScale : 0, undefined, out);
}

// ---------------------------------------------------------------------------------------------
// The registry entry.
// ---------------------------------------------------------------------------------------------

/** The params block, exported so the scheduler can restage it in place per dispatch. */
export function qmatmul8Params(
  kWords: number,
  numRows: number,
  mCols: number,
  inScale = 0,
  outScale = 0,
): ArrayBuffer {
  const buf = new ArrayBuffer(32);
  const u = new Uint32Array(buf);
  const f = new Float32Array(buf);
  u[0] = kWords;
  u[1] = numRows;
  u[2] = mCols;
  u[3] = 0;
  f[4] = inScale;
  f[5] = outScale;
  u[6] = 0;
  u[7] = 0;
  return buf;
}

function bindQmatmul8(input: KernelBindInput): KernelBindResult {
  const { device, inputs, output, params } = input;
  const k = params.k | 0;
  const numRows = params.numRows | 0;
  const mCols = (params.mCols ?? 1) | 0;
  if (k <= 0 || numRows <= 0 || mCols <= 0) {
    throw new Error('qmatmul-8bit needs params.k and params.numRows, with params.mCols absent for a decode step');
  }
  if (k % 4 !== 0) {
    throw new Error(
      `qmatmul-8bit needs K a multiple of 4, got ${k}. Both real tensors are 1536 and 256, so a `
      + 'caller that does not is packing the wrong tensor.',
    );
  }
  // The module's two static range quantization scales, straight from the checkpoint. Absent means
  // 0.0, which is what the file stores for an uncalibrated site (quant.ts applySrq).
  const inScale = params.inScale ?? 0;
  const outScale = params.outScale ?? 0;
  const kWords = wordsPerRow(8, k);
  const wq = inputs.wq;
  const scales = inputs.scales;
  const x = inputs.x;
  if (!wq || !scales || !x) throw new Error('qmatmul-8bit needs inputs named wq, scales and x');

  const layout = kernelLayout(input, 'qmatmul-8bit', () => device.createBindGroupLayout({
    label: 'qmatmul-8bit',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  }));
  const uniform = kernelUniform(
    input,
    'qmatmul-8bit params',
    qmatmul8Params(kWords, numRows, mCols, inScale, outScale),
  );

  return {
    layout,
    buffers: [wq, scales, x, output, uniform.binding],
    dispatch: [Math.ceil(numRows / QMATMUL8_WORKGROUP_SIZE), mCols, 1],
    dispose: uniform.dispose,
  };
}

// The synthetic SRQ scales are k-matmul's, deliberately, so the same margin argument covers them:
// 0.1875 divides the 1/8 activation grid into thirds, whose fractional parts are 1/3 and 2/3 and
// never 1/2, and 0.0625 sits inside the fixture builder's measured output margin. The k-ple
// section measures both rather than trusting them.
const SRQ_IN_THIRDS = 0.1875;
const SRQ_IN_CLAMPING = 0.015625;
const SRQ_OUT = 0.0625;

/**
 * Layer 0's real `per_layer_input_gate` activation scales, read out of the checkpoint. They are
 * literals here because a registry case's params have to be literals, and the k-ple section pins
 * both against `ref.ple.meta` on every run so this pair cannot rot against the dump. The fixture
 * builder reads them back out of this case rather than retyping them, so the case, the shader
 * uniform and the expected buffer are one number and not three.
 */
export const PLE_GATE_L0_IN_SCALE = 3.334678888320923;
export const PLE_GATE_L0_OUT_SCALE = 0.01857776567339897;

export const qmatmul8Kernel: Kernel = {
  name: 'qmatmul-8bit',
  wgsl: QMATMUL8_WGSL,
  entry: 'main',
  note:
    'K12\'s two linears: quant.ts\'s I8 ple-gate-8bit family, per_layer_input_gate [256, 1536] and '
    + 'per_layer_projection [1536, 256]. One invocation per output element, serial K, no reduction '
    + 'and therefore no fallback build. Both step modes run it, a decode step at mCols 1, with SRQ '
    + 'fused on both sides and the ratified integer code dot in between.',
  cases: [
    {
      name: 'synthetic-10x256-m3',
      inputs: { wq: 'kple.qm8.wq', scales: 'kple.qm8.scales', x: 'kple.qm8.x' },
      expected: 'kple.qm8.expected',
      params: { k: 256, numRows: 10, mCols: 3 },
      tolAbs: 0,
      tolUlp: 0,
      note:
        'The uncalibrated f32 code dot over three token columns. Ten rows against a workgroup of '
        + '64 exercise the row guard and three columns the column guard. Codes are integers and x '
        + 'is on the 1/8 grid, so every partial sum is exact in f32 and the gate is bit identity.',
    },
    {
      name: 'synthetic-10x256-m3-srq',
      inputs: { wq: 'kple.qm8.wq', scales: 'kple.qm8.scales', x: 'kple.qm8.x' },
      expected: 'kple.qm8.srq.expected',
      params: { k: 256, numRows: 10, mCols: 3, inScale: SRQ_IN_THIRDS, outScale: SRQ_OUT },
      tolAbs: 0,
      tolUlp: 0,
      note:
        'The same bytes through the ratified integer path with both SRQ sides live, which is how '
        + 'the two real per layer linears run on every one of the 35 layers: both of their '
        + 'activation scales are calibrated in the checkpoint.',
    },
    {
      name: 'synthetic-10x256-m3-srq-clamped',
      inputs: { wq: 'kple.qm8.wq', scales: 'kple.qm8.scales', x: 'kple.qm8.x' },
      expected: 'kple.qm8.srqclamp.expected',
      params: { k: 256, numRows: 10, mCols: 3, inScale: SRQ_IN_CLAMPING, outScale: 0 },
      tolAbs: 0,
      tolUlp: 0,
      note:
        'An input scale small enough that most of the fixture lands outside int8 and clamps, so '
        + 'the clamp is covered rather than assumed, with the output side uncalibrated to cover '
        + 'the other half of the epilogue branch.',
    },
    {
      name: 'real-ple-gate-slice',
      inputs: {
        wq: 'tensor.ple-gate-8bit.raw',
        scales: 'tensor.ple-gate-8bit.scale',
        x: 'kple.x256',
      },
      expected: 'kple.qm8.real.expected',
      params: { k: 128, numRows: 1, mCols: 1 },
      tolAbs: 0,
      tolUlp: 0,
      note:
        'The first 128 real bytes of layer 0 per_layer_input_gate row 0 with their real f32 scale. '
        + 'This is what proves the sign extension against Google\'s own bytes rather than against '
        + 'a fixture: the slice contains negative codes, and an unsigned read of them would be off '
        + 'by 256 apiece.',
    },
    {
      name: 'real-ple-gate-slice-srq',
      inputs: {
        wq: 'tensor.ple-gate-8bit.raw',
        scales: 'tensor.ple-gate-8bit.scale',
        x: 'kple.xgate',
      },
      expected: 'kple.qm8.realsrq.expected',
      params: {
        k: 128,
        numRows: 1,
        mCols: 1,
        inScale: PLE_GATE_L0_IN_SCALE,
        outScale: PLE_GATE_L0_OUT_SCALE,
      },
      tolAbs: 0,
      tolUlp: 0,
      note:
        'Real bytes, real weight scale and layer 0\'s own two activation scales, which is the '
        + 'exact arithmetic the shipped forward runs at this site. The activation is built on that '
        + 'module\'s own int8 grid, which is both what the previous op\'s output rounding produces '
        + 'in the real model and what puts every quotient a full half step from a rounding '
        + 'boundary, so the gate stays at bit identity.',
    },
  ],
  bind: bindQmatmul8,
};
