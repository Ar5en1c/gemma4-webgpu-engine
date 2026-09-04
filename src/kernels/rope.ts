// Written from docs/ENGINE-PLAN.md section 5 (architecture table, kernel K7), risk 1 quirk 3,
// risk 2 mitigation 4, the drift record in DECODE-FUSION-FINDING.md, and the WGSL specification.
// No vendored bundle, no extracted kernel and no third party engine source was read.
//
// SPDX-License-Identifier: Apache-2.0
//
// RoPE, in exactly one module, for both layer types.
//
// TRAP 3, which is the reason this file is longer than the shader. The seven full attention layers
// use PARTIAL RoPE. `partial_rotary_factor` is 0.25 on those layers, so at head dimension 512 only
// the first 64 of the 256 frequency pairs rotate and the remaining 192 are identity, meaning
// cosine 1 and sine 0. Sliding layers rotate all 128 of their pairs. An implementation that
// applies full RoPE to the global layers, which is what an ordinary transformer does, corrupts
// three quarters of every global layer's positional encoding while looking perfectly plausible
// (ENGINE-PLAN risk 1 quirk 3). Note that this quirk is missing from the public third party
// documentation of the other three, so it is not one a careful reader of the literature would
// catch. It is in the config.
//
// The shader does not know about any of that. The partial rotation lives entirely in the cosine
// and sine tables, where the pairs that do not rotate carry cosine 1 and sine 0, and the shader
// applies the same two lines to every pair. `lo * 1 - hi * 0` is exactly `lo` in IEEE 754 for any
// finite `hi`, so an identity pair costs two multiplies and stays bit exact. Three things follow,
// all of them wanted: there is no lane divergent branch in the hot loop, there is one code path
// for both layer types, and the whole quirk is expressed as data that the check script can assert
// on directly.
//
// ONE MODULE, DELIBERATELY. `n0 * c - n1 * s` is the canonical FMA contraction shape, and on Metal
// contraction is the compiler's choice per compiled module: this project measured every site's K
// row drifting 1 to 2 ULP on each generation's first decode pass, through exactly this expression,
// while values flowing through `dot()` never drifted anywhere (DECODE-FUSION-FINDING.md).
// ENGINE-PLAN risk 2 mitigation 4 is therefore a hard rule and not advice: where a scalar chain is
// unavoidable, keep it in exactly one module so there is no second module to disagree with. This
// module is also why the kernel takes a table position stride rather than assuming one position:
// a separate prefill rope kernel would be a second module computing the same expression.
//
// THE FORMULATION, which was open when this file was first written and is now settled against the
// reference implementation rather than against the plan's prose.
//
// Split half form, not interleaved: pair `i` is the element pair `(x[i], x[i + head_dim/2])`, and
// the pair index runs over the whole head, so there are 256 pairs at head dimension 512. The
// alternative convention in wide use for partial rotary, where the rotating slice is taken off the
// front of the head and paired inside itself so that pair `i` is `(x[i], x[i + rotary_dim/2])` and
// everything above `rotary_dim` passes through, is NOT what this model does. The reference builds
// its angle table as `cat((freqs, freqs))` over the whole head and applies
// `x * cos + rotate_half(x) * sin`, where `rotate_half` splits at `head_dim / 2`. So the cosine at
// `i` and at `i + 256` are the same number, the split is at the middle of the head, and the two
// lines in the shader below are that expression written out. Confirmed, not assumed.
//
// What was NOT settled by the same reading, and cost this file a correction, is the frequency
// ladder itself. See `ropeInverseFrequencies`.

import type { Kernel, KernelBindInput, KernelBindResult } from './registry';
import { kernelLayout, kernelUniform } from './binding';
import { isGlobalLayer } from './layerGeometry';

const WORKGROUP_SIZE = 64;

/** Carried into the check script so the settled convention travels with the code. */
export const ROPE_PAIRING_NOTE =
  'Pair i is (x[i], x[i + headDim/2]) over the whole head, with non rotating pairs carrying '
  + 'cosine 1 and sine 0, and invFreq[i] = theta ** (-2i / headDim) with headDim as the '
  + 'denominator on both layer types. Both halves of that were read off the reference '
  + 'implementation: rotate_half splits at head_dim/2, and the proportional rope type the global '
  + 'layers use divides by head_dim and then zero pads. The rotaryDim denominator this file '
  + 'originally used was wrong by a factor of four in the exponent on the seven global layers.';

export interface RopeSpec {
  kind: 'sliding' | 'global';
  /** 256 on sliding layers, 512 on the seven full attention layers. */
  headDim: number;
  /** headDim / 2. The number of frequency pairs, 128 or 256. */
  pairCount: number;
  /**
   * headDim * partialRotaryFactor. 256 on sliding layers, 128 on global ones.
   *
   * Recorded because it is what the config expresses, and NOT used as the denominator of the
   * frequency ladder. `ropeInverseFrequencies` divides by `headDim`. See the long comment there.
   */
  rotaryDim: number;
  /** rotaryDim / 2. The number of pairs that actually rotate: 128 or 64. */
  rotaryPairs: number;
  /** 1e4 on sliding layers, 1e6 on the full attention layers. */
  theta: number;
  partialRotaryFactor: number;
}

export const ROPE_SLIDING: RopeSpec = {
  kind: 'sliding',
  headDim: 256,
  pairCount: 128,
  rotaryDim: 256,
  rotaryPairs: 128,
  theta: 1e4,
  partialRotaryFactor: 1,
};

export const ROPE_GLOBAL: RopeSpec = {
  kind: 'global',
  headDim: 512,
  pairCount: 256,
  rotaryDim: 128,
  rotaryPairs: 64,
  theta: 1e6,
  partialRotaryFactor: 0.25,
};

export function ropeSpecForLayer(layer: number): RopeSpec {
  return isGlobalLayer(layer) ? ROPE_GLOBAL : ROPE_SLIDING;
}

/**
 * Inverse frequencies for one layer type, one entry per pair.
 *
 * `invFreq[i] = theta ** (-2i / headDim)` for the pairs that rotate, and exactly 0 for the rest.
 *
 * THE DENOMINATOR IS `headDim`, NOT `rotaryDim`, AND THAT IS THE WHOLE POINT OF THIS COMMENT.
 * An earlier version of this file used `rotaryDim` on the reasoning that a partial rotary layer
 * spreads its frequency ladder across the slice it rotates, so that the global layers would reach
 * the same smallest frequency over 64 pairs that they reach over 256. That reasoning is
 * plausible, it is how several other partial rotary models do it, and it is wrong for this one.
 * The reference implementation computes the rotating slice's frequencies over the FULL head
 * dimension and then appends zeros, so a global layer's 64 rotating pairs cover only the first
 * quarter of the ladder and the fastest quarter of it at that. Over 64 pairs the two conventions
 * disagree by a factor of four in the exponent: the last rotating pair's inverse frequency is
 * about 3.2e-2 the right way and about 1.4e-6 the wrong way, which is not a subtle difference in
 * the numbers and is completely invisible in the output, since either way the result is a
 * smoothly varying rotation that produces fluent text.
 *
 * The two claims this rests on, both read from the Apache 2.0 `transformers` reference, which
 * ENGINE-PLAN section 1 lists as an allowed source:
 *
 *   - The sliding layers use the default rope type over `head_dim` 256, so all 128 pairs rotate
 *     and `rotaryDim` and `headDim` are the same number. The bug could not show there.
 *   - The full attention layers use rope type `proportional`, which the config names and which
 *     ENGINE-PLAN does not mention. It takes `rope_angles = partial_rotary_factor * head_dim / 2`,
 *     which is 64, computes those 64 inverse frequencies with `head_dim` 512 in the denominator,
 *     and concatenates 192 zeros to reach 256. Its `factor` defaults to 1 and this config does not
 *     set one, and its attention scaling is 1 and unused, so nothing else in that rope type
 *     applies here. This is trap 3 with a second floor under it.
 *
 * A zero inverse frequency gives an angle of zero at every position, so cosine 1 and sine 0 fall
 * out of the same two `Math.cos` and `Math.sin` calls with no special case. That is the zero
 * padding ENGINE-PLAN risk 1 quirk 3 describes, expressed rather than branched on, and it is also
 * exactly what the reference's concatenated zeros do.
 */
export function ropeInverseFrequencies(spec: RopeSpec): Float64Array {
  const inv = new Float64Array(spec.pairCount);
  for (let i = 0; i < spec.rotaryPairs; i += 1) {
    inv[i] = Math.pow(spec.theta, (-2 * i) / spec.headDim);
  }
  return inv;
}

/**
 * Cosine and sine tables for a run of positions, `positions.length * pairCount` values each.
 *
 * Built on the host in f64 and uploaded, rather than computed in the shader. Two reasons, both
 * from the record: a `pow` and a `sin` per element inside the hot loop would put a second scalar
 * chain in the kernel for no benefit, and the tables are tiny next to the 0.6 GB the engine
 * streams per token, so there is nothing to save.
 */
export function ropeTables(
  spec: RopeSpec,
  positions: readonly number[],
): { cos: Float32Array; sin: Float32Array } {
  const inv = ropeInverseFrequencies(spec);
  const cos = new Float32Array(positions.length * spec.pairCount);
  const sin = new Float32Array(positions.length * spec.pairCount);
  for (let p = 0; p < positions.length; p += 1) {
    const base = p * spec.pairCount;
    const pos = positions[p];
    for (let i = 0; i < spec.pairCount; i += 1) {
      const angle = pos * inv[i];
      cos[base + i] = Math.cos(angle);
      sin[base + i] = Math.sin(angle);
    }
  }
  return { cos, sin };
}

/**
 * The trap 3 counter-example. Full rotation on a layer that should be partial.
 *
 * ENGINE-PLAN risk 1's mitigation asks for a test that rotates all pairs on a global layer and
 * asserts that layer's cosine collapses while the sliding layers stay clean. This builds the wrong
 * tables for that test. Nothing in the engine calls it.
 */
export function ropeTablesFullRotary(
  spec: RopeSpec,
  positions: readonly number[],
): { cos: Float32Array; sin: Float32Array } {
  return ropeTables({ ...spec, rotaryDim: spec.headDim, rotaryPairs: spec.pairCount }, positions);
}

/**
 * The other trap 3 counter-example: the right number of rotating pairs on the wrong ladder.
 *
 * This is the version of the bug this file actually shipped for a while, and it is worth keeping a
 * name for, because it is the one a reviewer would not catch by counting. 64 of 256 pairs rotate,
 * which is correct, but their frequencies are spread over `rotaryDim` rather than `headDim`, so
 * every rotating pair below the first turns far too slowly. The check script asserts the two
 * ladders differ by orders of magnitude at the last rotating pair, so nobody can quietly restore
 * the old denominator and still see green.
 */
export function ropeInverseFrequenciesRotaryDimLadder(spec: RopeSpec): Float64Array {
  const inv = new Float64Array(spec.pairCount);
  for (let i = 0; i < spec.rotaryPairs; i += 1) {
    inv[i] = Math.pow(spec.theta, (-2 * i) / spec.rotaryDim);
  }
  return inv;
}

export const ROPE_WGSL = /* wgsl */ `
struct RopeParams {
  // Pairs per head: 128 on sliding layers, 256 on the seven full attention layers. Runtime
  // opaque, so the loop below is not unrolled into a megakernel the way a constant bound one was
  // (DECODE-FUSION-FINDING.md, item 5).
  pairCount: u32,
  // How many rows share one position's table row. Decode passes the head count and every row uses
  // table row 0; prefill passes the same head count and row r reads table row r / heads. One
  // integer divide per workgroup, which is what buys one module instead of two.
  headsPerPosition: u32,
  pad0: u32,
  pad1: u32,
}

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read> cosTab: array<f32>;
@group(0) @binding(2) var<storage, read> sinTab: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;
@group(0) @binding(4) var<uniform> params: RopeParams;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_index) lid: u32,
) {
  let row = wid.x;
  let p = params.pairCount;
  let base = row * p * 2u;
  let tab = (row / params.headsPerPosition) * p;

  // Nothing is reduced in this kernel, so ENGINE-PLAN 5.5 rules 1 and 2 have nothing to bind to
  // here. Rule 6 does, and the bound above is uniform.
  //
  // Safe in place: a lane reads and writes only the pair it owns, so dst may alias src.
  for (var i = lid; i < p; i = i + ${WORKGROUP_SIZE}u) {
    let lo = src[base + i];
    let hi = src[base + i + p];
    let c = cosTab[tab + i];
    let s = sinTab[tab + i];
    // Split half form. On a pair that does not rotate, c is 1 and s is 0, and both lines are
    // exact. This is the only scalar multiply-add chain in the engine's norm and rope family, and
    // it is here once.
    dst[base + i] = lo * c - hi * s;
    dst[base + i + p] = hi * c + lo * s;
  }
}
`;

// ---------------------------------------------------------------------------------------------
// The CPU oracle.
// ---------------------------------------------------------------------------------------------

/**
 * The known answer reference for `rope`.
 *
 * @param src              `rows * headDim` values, row major, one row per (position, head)
 * @param cosTab           `positions * pairCount` values
 * @param sinTab           the same shape
 * @param rows             number of (position, head) rows
 * @param pairCount        headDim / 2
 * @param headsPerPosition how many consecutive rows share one table row
 */
export function ropeOracle(
  src: Float32Array,
  cosTab: Float32Array,
  sinTab: Float32Array,
  rows: number,
  pairCount: number,
  headsPerPosition: number,
): Float32Array {
  const headDim = pairCount * 2;
  if (src.length < rows * headDim) {
    throw new Error(`ropeOracle: src holds ${src.length}, needs ${rows * headDim}`);
  }
  if (headsPerPosition <= 0) throw new Error('ropeOracle: headsPerPosition must be positive');
  const out = new Float32Array(rows * headDim);
  for (let r = 0; r < rows; r += 1) {
    const base = r * headDim;
    const tab = Math.floor(r / headsPerPosition) * pairCount;
    for (let i = 0; i < pairCount; i += 1) {
      const lo = src[base + i];
      const hi = src[base + i + pairCount];
      const c = cosTab[tab + i];
      const s = sinTab[tab + i];
      out[base + i] = lo * c - hi * s;
      out[base + i + pairCount] = hi * c + lo * s;
    }
  }
  return out;
}

// ---------------------------------------------------------------------------------------------
// Registry entry.
// ---------------------------------------------------------------------------------------------

/** The params block, exported so the scheduler can restage it in place per dispatch. */
export function ropeParams(pairCount: number, headsPerPosition: number): ArrayBuffer {
  const words = new ArrayBuffer(16);
  const u = new Uint32Array(words);
  u[0] = pairCount;
  u[1] = headsPerPosition;
  return words;
}

function bind(input: KernelBindInput): KernelBindResult {
  const { device, inputs, output, params } = input;
  const rows = params.rows | 0;
  const pairCount = params.pairCount | 0;
  const headsPerPosition = (params.headsPerPosition | 0) || rows;
  if (rows <= 0) throw new Error('rope needs params.rows');
  if (pairCount <= 0) throw new Error('rope needs params.pairCount');

  const src = inputs.src;
  const cosTab = inputs.cosTab;
  const sinTab = inputs.sinTab;
  if (!src || !cosTab || !sinTab) throw new Error('rope needs inputs named src, cosTab and sinTab');

  const layout = kernelLayout(input, 'rope', () => device.createBindGroupLayout({
    label: 'rope',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  }));

  const uniform = kernelUniform(input, 'rope params', ropeParams(pairCount, headsPerPosition));

  return {
    layout,
    buffers: [src, cosTab, sinTab, output, uniform.binding],
    dispatch: [rows, 1, 1],
    dispose: uniform.dispose,
  };
}

// The tolerance argument for both rope cases.
//
// ULP is the wrong instrument here and saying why matters. `lo * c - hi * s` can cancel: when the
// two products are close, the result is near zero and an absolute error that is half a ULP of the
// products is a large number of ULPs of the result. FMA contraction changes exactly one of those
// two roundings, so a ULP gate on the result would be a gate on how much cancellation the fixture
// happened to contain. An absolute gate is the honest one. The fixtures keep every input under 8
// in magnitude and cosine and sine are at most 1, so each product is at most 8, whose f32 spacing
// is 2^-20, about 9.6e-7. One unrounded product is the whole of what contraction can change, so
// 1e-6 absolute covers it with nothing to spare and nothing hidden.
const ROPE_TOL_ABS = 1e-6;

export const ropeKernel: Kernel = {
  name: 'rope',
  wgsl: ROPE_WGSL,
  entry: 'main',
  note:
    'Split half RoPE for both layer types. The partial rotation of the seven full attention '
    + 'layers lives in the tables, where non rotating pairs carry cosine 1 and sine 0, so there is '
    + 'one code path and one compiled module for a shape whose rounding is not a contract.',
  cases: [
    {
      name: 'sliding-256',
      inputs: {
        src: 'krope.sliding.src',
        cosTab: 'krope.sliding.cos',
        sinTab: 'krope.sliding.sin',
      },
      expected: 'krope.sliding.expected',
      params: { rows: 8, pairCount: 128, headsPerPosition: 8 },
      tolAbs: ROPE_TOL_ABS,
      note: 'Head dimension 256, theta 1e4, all 128 pairs rotate.',
    },
    {
      name: 'global-512-partial',
      inputs: {
        src: 'krope.global.src',
        cosTab: 'krope.global.cos',
        sinTab: 'krope.global.sin',
      },
      expected: 'krope.global.expected',
      params: { rows: 8, pairCount: 256, headsPerPosition: 8 },
      tolAbs: ROPE_TOL_ABS,
      note:
        'Head dimension 512, theta 1e6, partial_rotary_factor 0.25 so pairs 64 to 255 are identity. '
        + 'The check script asserts those pairs come back bit equal to the input at a non zero position.',
    },
    {
      name: 'prefill-4-positions',
      inputs: {
        src: 'krope.prefill.src',
        cosTab: 'krope.prefill.cos',
        sinTab: 'krope.prefill.sin',
      },
      expected: 'krope.prefill.expected',
      params: { rows: 32, pairCount: 128, headsPerPosition: 8 },
      tolAbs: ROPE_TOL_ABS,
      note: 'Four positions of eight heads through the same module, which is what the table stride buys.',
    },
  ],
  bind,
};
