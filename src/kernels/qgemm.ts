// Written from docs/ENGINE-PLAN.md sections 5.4 (prefill is a GEMM problem, 91 percent two
// kernels) and 5.5, the prefill cost ledger and retile verdict in PREFILL-CAMPAIGN.md, the decode
// geometry in DECODE-CAMPAIGN.md 4.2, this engine's own quant.ts and qlayout.ts, and the WGSL
// specification. No vendored bundle, no extracted kernel and no third party engine source was
// read.
//
// SPDX-License-Identifier: Apache-2.0
//
// K16: the M tiled GEMM variants of the two dequant GEMV families, for prefill chunks.
//
// WHY A REAL GEMM SHAPE. On the M1 a 500 token prompt spent 1399.5 ms doing GEMM shaped work in a
// GEMV shape at 10 to 18 percent of f16 peak, and 1208.5 ms in the quantized GEMM, out of 2857 ms
// total (PREFILL-CAMPAIGN.md). This engine writes the prefill path as an actual GEMM from day one
// rather than inheriting a decode shaped one, which is the whole of ENGINE-PLAN 5.4's plan.
//
// WHERE THE WIN IS EXPECTED, per the banked measurements, and where it is not: a retile of the
// upstream prefill kernels falsified the traffic hypothesis, shift 4 gave 0.91x and shift 2 gave
// 1.03x bit identical, because the kernel is instruction bound on M1 and Apple's cache hierarchy
// already absorbs weight rereads (PREFILL-CAMPAIGN.md, retile campaign result). So the design
// spends its budget on instructions per dot: each packed word is unpacked ONCE and its codes are
// reused across every token column of the tile, so the unpack cost amortizes over M dots instead
// of being paid per dot as the decode GEMV must. The activation tile is staged through workgroup
// memory so x reads are shared too. And the loop structure keeps a clean seam where a subgroup
// matrix path can replace the inner product the day Chrome ships the feature unflagged, which the
// same campaign measured at 2.1x TTFT with byte identical output behind flags.
//
// GEOMETRY. Same 64 wide, 2 virtual subgroup, 4 output row workgroup as the decode GEMV
// (DECODE-CAMPAIGN.md 4.2), extended by a column tile of M tokens: 8 at 4-bit, 4 at 2-bit, both
// chosen so the staged tile is 8 KB of the workgroup's memory and the cooperative stage is 8
// vec4 loads per lane per chunk. The K loop walks chunks of 32 words, so each iteration the 32
// lanes of a virtual subgroup touch 128 contiguous weight bytes per row, same coalescing as the
// GEMV. Accumulator pressure is rowsPerVsg times the tile, 16 f32 at 4-bit and 8 at 2-bit.
//
// STATIC RANGE QUANTIZATION IS PART OF THIS KERNEL TOO, both sides, since round 2, and it has to
// be: prefill and decode run the same linears, so if only one of them snapped its activations onto
// the calibrated int8 grid the two would disagree about the same token. The uniform block carries
// the module's input_activation_scale and output_activation_scale exactly as qgemv.ts does, the
// prologue snaps the staged tile and the epilogue snaps the written element, and a calibrated site
// reduces K as the exact integer code dot with one closing multiply by input_scale times
// weight_scale. An uncalibrated site, meaning a stored 0.0, keeps round 1's f32 code dot.
//
// ACCUMULATORS ARE f32 CARRYING EXACT INTEGERS, NEVER f16, here as everywhere in this engine.
// Round 7 moved the calibrated path off i32: the code products are small enough that their sums
// are integers an f32 holds exactly (the bound is argued at gemmDomainWgsl), and Apple's f32 fused
// multiply add units run several times the rate of i32 multiplies. Worth 1.08x on qgemm-4bit, bit
// identical. The f16 prohibition below is unchanged and is the real rule here: LlamaWeb
// (arXiv 2605.20706) measured f16 accumulation producing incoherent output on Apple M-series GPUs.
// f16 is a storage and cast dtype only (f16Cast.ts), and the k-matmul lint greps every registered
// kernel's WGSL for an f16 declaration so the rule is checked rather than remembered.
//
// SUBGROUP POLICY: identical to qgemv.ts, all reductions in uniform control flow above one
// guarded store block, butterfly tail from subgroupReduce.ts via the shared prelude, portable
// variant selected at pipeline build. Barriers all sit in uniform control flow because staging
// and compute are unconditional and the masking is data level (ENGINE-PLAN 5.5 rule 5).

import type { Kernel, KernelBindInput, KernelBindResult } from './registry';
import { kernelLayout, kernelUniform } from './binding';
import {
  foldedDispatch,
  GEMV_ROWS_PER_VSG,
  GEMV_ROWS_PER_WORKGROUP,
  GEMV_WORKGROUP_SIZE,
  matmulReducePrelude,
  qgemvOracle,
  type MatmulReduceVariant,
} from './qgemv';
import {
  CODES_PER_WORD,
  WGSL_UNPACK2,
  WGSL_UNPACK4,
  wordsPerRow,
  TILE_ROWS,
} from './qlayout';

/** Token columns per workgroup tile. Both keep the staged x tile at 8 KB. */
export const GEMM_M_TILE = { 2: 8, 4: 8 } as const;

/** Words per K chunk. 32 lanes, one word each: 256 codes at 4-bit, 512 at 2-bit. */
const CHUNK_WORDS = 32;

/**
 * One K loop, staging and compute, in one arithmetic domain.
 *
 * The two domains are emitted as two whole loops under one uniform branch rather than as one loop
 * carrying both accumulator sets. That is a register pressure decision: at 4-bit the tile is eight
 * columns and two rows, so carrying an f32 and an i32 accumulator per slot would put 32 live values
 * per lane where round 1 had 16, and occupancy is the thing this geometry was chosen for
 * (DECODE-CAMPAIGN.md 4.2). The branch reads `params.inScale`, which comes from the uniform block,
 * so it is uniform for every invocation in the workgroup and the barriers inside it stay legal
 * (ENGINE-PLAN 5.5 rule 5, and the same argument gelu-mul's uniform branch makes).
 *
 * The integer domain stages SRQ codes rather than activations, as the f32 integers round()
 * produced. Round 7 replaced an earlier scheme that carried them through the tile by `bitcast` and
 * read them back as i32: the bitcast saved one convert per staged value but bought i32 multiplies
 * in every dot of the K loop, which is the wrong side of that trade on this hardware.
 */
function gemmDomainWgsl(bits: 2 | 4, domain: 'int' | 'float'): string {
  const mTile = GEMM_M_TILE[bits];
  const vec4PerWord = CODES_PER_WORD[bits] / 4;
  const chunkVec4 = CHUNK_WORDS * vec4PerWord;
  const stagePerLane = (mTile * chunkVec4) / GEMV_WORKGROUP_SIZE;
  const int = domain === 'int';
  // BOTH domains reduce in f32, and on the calibrated one that is exact and not an approximation.
  // The values are integers: a weight code is -8..7 at 4-bit and -2..1 at 2-bit, an activation code
  // is -128..127, so one product is at most 1016 and the whole K sum is at most 1016 * K. The
  // widest K a 4-bit site sees is the producer intermediate, 6144, giving 6,289,920; a 2-bit site's
  // 12288 gives 3,121,152. Both are far inside the 16,777,216 an f32 holds exactly, so every
  // partial sum and the total are integers f32 carries with no rounding whatever the order, and
  // the kernel sweep holds the result to the i32 oracle at zero tolerance.
  //
  // Why bother: Apple's f32 fused multiply add units run several times the rate of i32 multiplies,
  // the same measurement qgemvWide.ts is built on and the reason the 2-bit GEMM below already
  // stages f32 integers through srqInF. The convert is paid once per staged value, ${stagePerLane} vec4s a
  // lane a chunk, instead of the inner loop paying i32 multiplies in all ${mTile * 2} of its dots.
  const accType = 'f32';
  const zero = '0.0';
  const unpack = bits === 4 ? WGSL_UNPACK4 : WGSL_UNPACK2;
  const tile = (n: string): string => `xTile[${n}]`;
  const names = bits === 4 ? ['wLo', 'wHi'] : ['wB0', 'wB1', 'wB2', 'wB3'];
  // Both rows' unpacked codes are held across ONE column loop, so the staged tile is read once
  // per column instead of once per column per row. See the fused loop below.
  const vecT = 'vec4<f32>';
  const codeDecls = [0, 1]
    .flatMap((r) => names.map((_, i) => `      var r${r}_${i}: ${vecT};`))
    .join('\n');
  const codeLoad = (r: number): string => [
    `      {`,
    `        let w = wq[base${r} + wi];`,
    unpack.replace(/\n+$/, ''),
    `        ${names.map((n, i) => `r${r}_${i} = ${n};`).join(' ')}`,
    `      }`,
  ].join('\n');
  const tileLoads = names
    .map((_, i) => `          let x${i} = ${tile(i === 0 ? 'tb' : `tb + ${i}u`)};`)
    .join('\n');
  // Row r at column slot ct: the same dots in the same order the two separate row blocks emitted,
  // so every accumulator is bit for bit what it was.
  const rowDots = (r: number): string => names.map((_, i) => `dot(r${r}_${i}, x${i})`).join(' + ');
  // The staged value: the raw activation, or its int8 code as the f32 integer round() produced.
  // The snap happens here, once per value, exactly as the 2-bit GEMM's srqInF does it.
  const staged = int
    ? `srqInF(x[min(col, cLast) * params.kVec4 + c * ${chunkVec4}u + off]) * f32(col < params.mCols)`
    : `x[min(col, cLast) * params.kVec4 + c * ${chunkVec4}u + off] * f32(col < params.mCols)`;

  return /* wgsl */ `  var acc0: array<${accType}, ${mTile}>;
    var acc1: array<${accType}, ${mTile}>;
    for (var i = 0u; i < ${mTile}u; i = i + 1u) { acc0[i] = ${zero}; acc1[i] = ${zero}; }

    for (var c = 0u; c < params.kIters; c = c + 1u) {
      // Stage. Every lane moves ${stagePerLane} vec4s; a column past mCols stages zeros through a
      // clamped read, which is the data level masking of ENGINE-PLAN 5.5 rule 5.
      for (var j = 0u; j < ${stagePerLane}u; j = j + 1u) {
        let t = j * ${GEMV_WORKGROUP_SIZE}u + lid;
        let ct = t / ${chunkVec4}u;
        let off = t % ${chunkVec4}u;
        let col = colBase + ct;
        xTile[t] = ${staged};
      }
      workgroupBarrier();

      // Compute. One word unpacked per row per lane, its codes reused across all ${mTile} columns,
      // which is the instruction amortization the retile campaign says this hardware wants.
      //
      // The two rows share ONE column loop. Written as two loops, each row read the whole staged
      // tile for itself: ${vec4PerWord * mTile * 2} workgroup loads to feed ${mTile * CODES_PER_WORD[bits] * 2} multiply adds, because
      // tb depends only on ct and lane and so is the same address in both. Fused, the tile is
      // read ${vec4PerWord * mTile} times and every load feeds both rows. The unpack still happens once per row per
      // word, which is the property the two loop form was written for and the reason the codes
      // are hoisted into registers here rather than the loop being turned inside out.
      let wi = c * ${CHUNK_WORDS}u + lane;
${codeDecls}
${codeLoad(0)}
${codeLoad(1)}
      for (var ct = 0u; ct < ${mTile}u; ct = ct + 1u) {
        let tb = ct * ${chunkVec4}u + lane * ${vec4PerWord}u;
${tileLoads}
          acc0[ct] = acc0[ct] + ${rowDots(0)};
          acc1[ct] = acc1[ct] + ${rowDots(1)};
      }
      workgroupBarrier();
    }

    // All reductions above all stores, then one guarded store block. ENGINE-PLAN 5.5 rule 1.
    var sum0: array<f32, ${mTile}>;
    var sum1: array<f32, ${mTile}>;
    for (var ct = 0u; ct < ${mTile}u; ct = ct + 1u) {
      sum0[ct] = mmSum(f32(acc0[ct]), lid);
      sum1[ct] = mmSum(f32(acc1[ct]), lid);
    }
    if (lane == 0u) {
      let xs = ${int ? 'params.inScale' : '1.0'};
      for (var ct = 0u; ct < ${mTile}u; ct = ct + 1u) {
        let col = colBase + ct;
        if (col < params.mCols) {
          if (row0 < params.numRows) {
            dst[col * params.numRows + row0] = srqOut((xs * scales[row0]) * sum0[ct]);
          }
          if (row1 < params.numRows) {
            dst[col * params.numRows + row1] = srqOut((xs * scales[row1]) * sum1[ct]);
          }
        }
      }
    }`;
}

/**
 * The 2-bit GEMM on the interleaved layout (qlayout.ts TILE_ROWS; qgemv.ts qgemv2TileWgsl is the
 * decode side of the same layout). Every lane owns ONE output row and GEMM_M_TILE[2] token
 * columns, walks the whole of K and reduces nothing: a workgroup of 64 lanes is four tiles of
 * sixteen rows, and its column tile is shared through workgroup memory, staged 512 k at a time.
 * The layout is what makes this shape cheap: a word holds the sixteen rows' codes at one k, so
 * the sixteen lanes of a tile read the same word (one broadcast load) and each extracts its own
 * code, and four consecutive k are one vec4<u32> load per lane.
 *
 * The arithmetic is f32 fused multiply adds on both paths. On the integer path the staged
 * activations are the int8 codes as f32 integers (the same round and clamp as the GEMV's srqIn,
 * left as f32) and the weight codes are integers in -2..1, so every product has magnitude at
 * most 384 and every partial sum is an integer under 2^24 in magnitude, exact in f32 whatever
 * the order: the lane's row sum is the integer the ratified i32 contract produces, and the
 * kernel sweep holds it to the oracle at zero tolerance. On the f32 path (an uncalibrated
 * site) the activations are raw and the accumulation order is this kernel's own, k ascending.
 *
 * No reduction, so no subgroup operation and one build for both reduce policies; ENGINE-PLAN
 * 5.5 rule 1 has nothing to bind to beyond the barriers around the staging, which every lane
 * of the workgroup reaches (uniform loop bounds from the params block). Rows past numRows and
 * columns past mCols clamp their reads and fail their store guards.
 */
function qgemm2TileWgsl(): string {
  const mTile = GEMM_M_TILE[2];
  const cols = Array.from({ length: mTile }, (_, c) => c);
  const comps = ['x', 'y', 'z', 'w'];
  // Per 4 k: the lane's four codes out of the four words, then mTile vec4 activation reads (one
  // per column, four k each) and 4 * mTile fused multiply adds.
  const body = [
    ...comps.map((c, j) => `        let c${j} = f32((v.${c} >> shift) & 3u) - 2.0;`),
    ...cols.flatMap((ct) => [
      `        { let xv = xTile[${ct}u * 128u + q];`,
      ...comps.map((c, j) => `          acc${ct} = fma(c${j}, xv.${c}, acc${ct});`),
      '        }',
    ]),
  ].join('\n');

  return /* wgsl */ `
struct GemmParams {
  kWords: u32,
  // Outer iterations of 512 k: K / 512.
  kIters: u32,
  kVec4: u32,
  numRows: u32,
  mCols: u32,
  // vec4s of x per outer iteration, 128: the K loop bound, read from the block (rule 6).
  chunkVec4: u32,
  inScale: f32,
  outScale: f32,
}

@group(0) @binding(0) var<storage, read> wq: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> scales: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;
@group(0) @binding(4) var<uniform> params: GemmParams;

const SRQ_MAX: f32 = 127.0;
const SRQ_MIN: f32 = -128.0;

// The SRQ input prologue, kept as the f32 integers round() produced: the codes the exact f32
// fused multiply adds below consume (the GEMV casts the same values to i32).
fn srqInF(v: vec4<f32>) -> vec4<f32> {
  return clamp(round(v / params.inScale), vec4<f32>(SRQ_MIN), vec4<f32>(SRQ_MAX));
}

fn srqOut(v: f32) -> f32 {
  if (params.outScale == 0.0) {
    return v;
  }
  return clamp(round(v / params.outScale), SRQ_MIN, SRQ_MAX) * params.outScale;
}

// The column tile, 512 k of ${mTile} columns as vec4s along k.
var<workgroup> xTile: array<vec4<f32>, ${mTile * 128}>;

@compute @workgroup_size(${GEMV_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
  @builtin(local_invocation_index) lid: u32,
) {
  // Four tiles per workgroup; lane r of a tile owns row 16 tile + r.
  let group = wid.x + nwg.x * wid.z;
  let tile = group * 4u + (lid >> 4u);
  let r = lid & 15u;
  let shift = 2u * r;
  let lastTile = (params.numRows - 1u) / 16u;
  let tileBase = min(tile, lastTile) * params.kWords * 4u;
  let row = tile * 16u + r;
  let colBase = wid.y * ${mTile}u;
  let cLast = params.mCols - 1u;
  let calibrated = params.inScale != 0.0;
${cols.map((ct) => `  var acc${ct} = 0.0;`).join('\n')}

  for (var c = 0u; c < params.kIters; c = c + 1u) {
    // Stage 512 k of every column: ${(mTile * 128) / GEMV_WORKGROUP_SIZE} vec4s per lane, a column past
    // mCols staged as zeros through a clamped read (ENGINE-PLAN 5.5 rule 5), and the snap
    // applied here once per value on a calibrated site.
    for (var j = 0u; j < ${(mTile * 128) / GEMV_WORKGROUP_SIZE}u; j = j + 1u) {
      let t = j * ${GEMV_WORKGROUP_SIZE}u + lid;
      let ct = t / 128u;
      let off = t % 128u;
      let col = colBase + ct;
      let raw = x[min(col, cLast) * params.kVec4 + c * 128u + off] * f32(col < params.mCols);
      xTile[t] = select(raw, srqInF(raw), calibrated);
    }
    workgroupBarrier();

    for (var q = 0u; q < params.chunkVec4; q = q + 1u) {
      let v = wq[tileBase + c * 128u + q];
${body}
    }
    workgroupBarrier();
  }

  if (row < params.numRows) {
    let xs = select(1.0, params.inScale, calibrated);
    let s = xs * scales[row];
${cols.map((ct) => `    if (colBase + ${ct}u < params.mCols) { dst[(colBase + ${ct}u) * params.numRows + row] = srqOut(s * acc${ct}); }`).join('\n')}
  }
}
`;
}

/**
 * Build the GEMM WGSL for one bit width and one reduction variant.
 *
 * Layouts: x is [M, K] row major activations, dst is [M, N] row major, weights and scales are
 * exactly the GEMV's. Rows past numRows and columns past mCols are handled by clamped reads,
 * zero padded staging and the store guard, never by branches in the hot loop.
 *
 * Static range quantization is fused here exactly as it is in the GEMV, both sides, with the same
 * two uniform scales and the same ratified integer reduction. Prefill and decode therefore run the
 * same arithmetic on the same weights, which is the only way the two paths can agree on a token.
 */
export function qgemmWgsl(bits: 2 | 4, variant: MatmulReduceVariant): string {
  if (bits === 2) return qgemm2TileWgsl();
  const mTile = GEMM_M_TILE[bits];
  const vec4PerWord = CODES_PER_WORD[bits] / 4;
  const chunkVec4 = CHUNK_WORDS * vec4PerWord;

  return /* wgsl */ `${matmulReducePrelude(variant)}
struct GemmParams {
  // Packed u32 words per weight row, K / ${CODES_PER_WORD[bits]} for this ${bits}-bit family.
  kWords: u32,
  // kWords / ${CHUNK_WORDS}: K chunk count. Runtime opaque, per registry rule 2.
  kIters: u32,
  // K / 4: vec4 stride of one token's activation row.
  kVec4: u32,
  numRows: u32,
  mCols: u32,
  pad0: u32,
  // The module's input_activation_scale and output_activation_scale, exactly as qgemv.ts carries
  // them. 0.0 means the site is uncalibrated and the rounding is skipped (quant.ts applySrq).
  inScale: f32,
  outScale: f32,
}

@group(0) @binding(0) var<storage, read> wq: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;
@group(0) @binding(4) var<uniform> params: GemmParams;

// int8, at every static range quantization site in this checkpoint (quant.ts SRQ_ACTIVATION_BITS).
const SRQ_MAX: f32 = 127.0;
const SRQ_MIN: f32 = -128.0;

// The SRQ prologue and epilogue, the same expressions qgemv.ts uses, because they are the same
// operation on the same module's scales and a second spelling would be a second thing to keep
// right (quant.ts srqCodes and applySrq).
// Kept as the f32 integers round() produced, which is what the exact f32 fused multiply adds in
// the K loop consume; qgemv.ts casts these same values to i32 for its integer reduction, and the
// 2-bit GEMM below spells this identically.
fn srqInF(v: vec4<f32>) -> vec4<f32> {
  return clamp(round(v / params.inScale), vec4<f32>(SRQ_MIN), vec4<f32>(SRQ_MAX));
}

fn srqOut(v: f32) -> f32 {
  if (params.outScale == 0.0) {
    return v;
  }
  return clamp(round(v / params.outScale), SRQ_MIN, SRQ_MAX) * params.outScale;
}

// The staged activation tile: ${mTile} token columns of one ${chunkVec4} vec4 K chunk, 8 KB. One
// tile, not one per domain: two would be 16 KB against the 16 KB workgroup storage limit before
// the fallback reduce's own scratch is counted.
var<workgroup> xTile: array<vec4<f32>, ${mTile * chunkVec4}>;

@compute @workgroup_size(${GEMV_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
  @builtin(local_invocation_index) lid: u32,
) {
  let lane = lid & 31u;
  let vsg = lid >> 5u;
  // Row groups fold over x and z so no dimension needs more than the 65535 limit; y is the
  // column tile.
  let group = wid.x + nwg.x * wid.z;
  let row0 = group * ${GEMV_ROWS_PER_WORKGROUP}u + vsg * ${GEMV_ROWS_PER_VSG}u;
  let row1 = row0 + 1u;
  let rLast = params.numRows - 1u;
  let base0 = min(row0, rLast) * params.kWords;
  let base1 = min(row1, rLast) * params.kWords;
  let colBase = wid.y * ${mTile}u;
  let cLast = params.mCols - 1u;

  // Uniform branch on a uniform buffer value, so both sides keep uniform control flow and their
  // barriers and reductions are legal where they sit.
  if (params.inScale != 0.0) {
${gemmDomainWgsl(bits, 'int')}
  } else {
${gemmDomainWgsl(bits, 'float')}
  }
}
`;
}

export const QGEMM4_WGSL = qgemmWgsl(4, 'subgroup');
// The 2-bit GEMM reduces nothing and has one build; the variant argument is ignored for it and
// it has no entry in pipeline.ts FALLBACK_WGSL.
export const QGEMM2_WGSL = qgemmWgsl(2, 'subgroup');
export const QGEMM4_FALLBACK_WGSL = qgemmWgsl(4, 'workgroup');

// ---------------------------------------------------------------------------------------------
// The CPU oracle.
// ---------------------------------------------------------------------------------------------

/**
 * The GEMM reference is the GEMV reference per token column: out[m, n] over [M, N] row major,
 * with the same exactness argument as qgemvOracle on grid fixtures.
 *
 * The two activation scales pass straight through, which is the claim worth making rather than the
 * code worth reading: a prefill column and a decode step of the same linear are the same arithmetic
 * to the last bit, so a token that prefill and decode disagree on is a bug in one of them and not a
 * property of the two shapes.
 */
export function qgemmOracle(
  bits: 2 | 4,
  packed: Uint8Array,
  scales: ArrayLike<number>,
  x: Float32Array,
  numRows: number,
  k: number,
  mCols: number,
  inScale = 0,
  outScale = 0,
): Float32Array {
  if (x.length < mCols * k) throw new Error(`qgemmOracle: x holds ${x.length}, needs ${mCols * k}`);
  const out = new Float32Array(mCols * numRows);
  for (let col = 0; col < mCols; col += 1) {
    const y = qgemvOracle(
      bits, packed, scales, x.subarray(col * k, (col + 1) * k), numRows, k, inScale, outScale,
    );
    out.set(y, col * numRows);
  }
  return out;
}

// ---------------------------------------------------------------------------------------------
// Registry entries.
// ---------------------------------------------------------------------------------------------

const K_SPAN_PER_ITER = { 2: 512, 4: 256 } as const;

/**
 * The params block, exported so the scheduler can restage it in place per dispatch. The last two
 * words are the module's two activation scales, the same pair qgemv.ts carries, so a prefill chunk
 * and a decode step describe the same linear the same way.
 */
export function gemmParams(
  kWords: number,
  kIters: number,
  kVec4: number,
  numRows: number,
  mCols: number,
  inScale = 0,
  outScale = 0,
): ArrayBuffer {
  const buf = new ArrayBuffer(32);
  const u = new Uint32Array(buf);
  const f = new Float32Array(buf);
  u[0] = kWords;
  u[1] = kIters;
  u[2] = kVec4;
  u[3] = numRows;
  u[4] = mCols;
  // vec4s of x per outer iteration, the 2-bit kernel's K loop bound; a pad word to the 4-bit one.
  u[5] = kVec4 / kIters;
  f[6] = inScale;
  f[7] = outScale;
  return buf;
}

function bindGemm(bits: 2 | 4) {
  return (input: KernelBindInput): KernelBindResult => {
    const { device, inputs, output, params } = input;
    const k = params.k | 0;
    const numRows = params.numRows | 0;
    const mCols = params.mCols | 0;
    const inScale = params.inScale ?? 0;
    const outScale = params.outScale ?? 0;
    if (k <= 0 || numRows <= 0 || mCols <= 0) {
      throw new Error('qgemm needs params.k, params.numRows and params.mCols');
    }
    if (k % K_SPAN_PER_ITER[bits] !== 0) {
      throw new Error(`qgemm-${bits}bit needs K a multiple of ${K_SPAN_PER_ITER[bits]}, got ${k}`);
    }
    const kWords = wordsPerRow(bits, k);
    const wq = inputs.wq;
    const scales = inputs.scales;
    const x = inputs.x;
    if (!wq || !scales || !x) throw new Error('qgemm needs inputs named wq, scales and x');

    const layout = kernelLayout(input, `qgemm-${bits}bit`, () => device.createBindGroupLayout({
      label: `qgemm-${bits}bit`,
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
      `qgemm-${bits}bit params`,
      gemmParams(kWords, kWords / CHUNK_WORDS, k / 4, numRows, mCols, inScale, outScale),
    );

    // The 2-bit kernel gives every lane a row, four sixteen row tiles per workgroup of 64, and
    // one column tile per workgroup; the 4-bit kernel keeps the row layout's four rows per
    // workgroup.
    const rowGroups = bits === 2 ? Math.ceil(numRows / (4 * TILE_ROWS)) : Math.ceil(numRows / GEMV_ROWS_PER_WORKGROUP);
    const colTiles = Math.ceil(mCols / GEMM_M_TILE[bits]);
    const folded = foldedDispatch(rowGroups);
    return {
      layout,
      buffers: [wq, scales, x, output, uniform.binding],
      // Row groups on x and z, column tiles on y, matching the shader's fold.
      dispatch: [folded[0], colTiles, folded[1]],
      dispose: uniform.dispose,
    };
  };
}

export const qgemm4Kernel: Kernel = {
  name: 'qgemm-4bit',
  wgsl: QGEMM4_WGSL,
  entry: 'main',
  note:
    'K16 for the 4-bit family. M tiled prefill GEMM, 8 token columns, unpack amortized across the '
    + 'tile because prefill is instruction bound on M1, not traffic bound (PREFILL-CAMPAIGN.md).',
  cases: [
    {
      name: 'synthetic-8x512-m11',
      inputs: { wq: 'kmm.gemv4.wq', scales: 'kmm.gemv4.scales', x: 'kmm.gemm4.x' },
      expected: 'kmm.gemm4.expected',
      params: { k: 512, numRows: 8, mCols: 11 },
      tolAbs: 0,
      tolUlp: 0,
      note:
        'Two K chunks, eleven columns over a tile of eight: two column tiles, five masked '
        + 'columns, same exact grid gate as the GEMV. Weights are the GEMV fixture, so the two '
        + 'kernels are proved on the same bytes.',
    },
    {
      name: 'synthetic-8x512-m11-srq',
      inputs: { wq: 'kmm.gemv4.wq', scales: 'kmm.gemv4.scales', x: 'kmm.gemm4.x' },
      expected: 'kmm.gemm4.srq.expected',
      params: { k: 512, numRows: 8, mCols: 11, inScale: 0.1875, outScale: 0.0625 },
      tolAbs: 0,
      tolUlp: 0,
      note:
        'The same tile on the ratified integer path, with both activation scales set. The scales '
        + 'match the GEMV\'s SRQ cases, so the k-matmul check can assert that prefill column and '
        + 'decode step of the same linear agree bit for bit rather than merely closely.',
    },
  ],
  bind: bindGemm(4),
};

export const qgemm2Kernel: Kernel = {
  name: 'qgemm-2bit',
  wgsl: QGEMM2_WGSL,
  entry: 'main',
  note:
    'K16 for the 2-bit family. Prefill GEMM on the interleaved tile layout: a lane per row, 8 '
    + 'token columns per lane, no reduction; exact f32 fused multiply adds on the integer path.',
  cases: [
    {
      name: 'synthetic-70x1024-m11',
      inputs: { wq: 'kmm.gemm2.tile.wq', scales: 'kmm.gemm2.scales', x: 'kmm.gemm2.x' },
      expected: 'kmm.gemm2.expected',
      params: { k: 1024, numRows: 70, mCols: 11 },
      tolAbs: 0,
      tolUlp: 0,
      note:
        'Two K chunks. Seventy rows are two workgroups of four tiles, the second holding one '
        + 'tile of six live rows and three tiles past numRows, so both the row guard and the tile '
        + 'clamp are in play; eleven columns over a tile of eight exercise the column masking and '
        + 'the second column tile.',
    },
    {
      name: 'synthetic-70x1024-m3-srq',
      inputs: { wq: 'kmm.gemm2.tile.wq', scales: 'kmm.gemm2.scales', x: 'kmm.gemm2.x' },
      expected: 'kmm.gemm2.srq.expected',
      params: { k: 1024, numRows: 70, mCols: 3, inScale: 0.1875, outScale: 0.0625 },
      tolAbs: 0,
      tolUlp: 0,
      note:
        'The same rows on the integer path over the first three columns of the same activations, '
        + 'with the row guard, the tile clamp and the column mask still in play, so a masked column '
        + 'is proved to stage as a zero code rather than as stale bits and the f32 fused multiply '
        + 'adds are proved to land on the ratified integers. Three columns rather than eleven keep '
        + 'the SRQ half integer audit at its margin over the outputs.',
    },
  ],
  bind: bindGemm(2),
};
