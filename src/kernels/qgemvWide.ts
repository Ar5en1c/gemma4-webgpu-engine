// Written from ./qgemv.ts (the shipped one column GEMV family, whose loops these kernels repeat
// column for column), docs/ENGINE-PLAN.md section 5.5 and the round 5 ceiling ledger, and the
// WGSL specification. No vendored bundle, no extracted kernel and no third party engine source
// was read.
//
// SPDX-License-Identifier: Apache-2.0
//
// THE M WIDE GEMV: one weight stream, up to GEMV_WIDE_COLS activation columns.
//
// WHY. A speculative verify pass runs the decoder over a handful of token positions at once
// (the token the model just chose plus a few drafted after it) and reads the argmax at every
// one of them. The prefill GEMM tiles are the wrong shape for that: measured on the loaded
// model, a pass of one to four tokens through them costs 148 to 158 ms against 26 ms for a
// decode token, because each tile serialises a whole K per lane at a handful of columns
// (docs/ENGINE-PLAN.md round 5 ledger). The decode GEMVs stream every weight byte once at 37
// to 41 GB/s and are bandwidth bound, so the cheapest verify pass is those same loops with the
// weight word reused across columns: the unpack is shared, only the multiply adds per column
// are new.
//
// THE CONTRACT IS COLUMN FOR COLUMN THE ONE COLUMN KERNEL. Each column's accumulation is the
// same lane assignment, the same integer products, the same reduction ladder and the same two
// closing multiplies as qgemv.ts at the same geometry, so on a calibrated site a column of
// this kernel is bit for bit the one column kernel over that column, and the kernel sweep gates
// it at zero. On the f32 head path the accumulation order per column is the tile kernel's own,
// so the head columns match the one column head bit for bit too. That identity is what lets a
// speculative decode claim the plain greedy ids by construction: the verify pass computes the
// same logits the plain step would have.
//
// LAYOUT. Activations: `cols` vectors of K values back to back, column c at c * kVec4 vec4s
// (the residual and slot layout of a multi token step, row major by position). Output: row
// major by column, dst[c * numRows + row]. `params.cols` is the live count, at most
// GEMV_WIDE_COLS; a column past it reads column cols - 1 (a clamped, harmless read) and is
// never stored. Every guard is on a uniform, so control flow stays uniform and the reductions
// all run before the one store block (ENGINE-PLAN 5.5 rule 1).
import type { Kernel, KernelBindInput, KernelBindResult } from './registry';
import { kernelLayout, kernelUniform } from './binding';
import {
  CODES_PER_WORD,
  WGSL_UNPACK4,
  WGSL_UNPACK8,
  wordsPerRow,
} from './qlayout';
import { geluMulOracle } from './geluMul';
import {
  foldedDispatch,
  gemvGeometry,
  gemvRowsPerWorkgroup,
  K_SPAN_PER_ITER,
  matmulReducePrelude,
  prologueWgsl,
  qgemvOracle,
} from './qgemv';
import type { GemvBits, GemvGeometry, GemvPrologue, MatmulReduceVariant } from './qgemv';

/**
 * Columns one dispatch carries: the chosen token plus the ones drafted after it.
 *
 * TWO, chosen by measurement in round 7 rather than by ambition. Against the one column kernel
 * on the same weight bytes, with one weight read serving every column (see the header below),
 * the 4-bit wide kernel costs 1.23x at two columns and 2.35x at four. Two columns is the physics
 * target reached: one weight stream, a second activation column nearly free. Four columns puts
 * eight activation vec4 registers a lane on top of the accumulators and gives the whole saving
 * back, which is the same register wall round 5 hit from the other direction.
 *
 * The consequence for speculation is that a verify pass covers the chosen token plus ONE draft,
 * so it wins whenever the accepted length beats 1.23, that is whenever the drafter is right more
 * than a quarter of the time. Depth four would have needed to be right 2.35 tokens deep.
 */
export const GEMV_WIDE_COLS = 2;

/** Registry names, `qgemv-<bits>bit[-gelu]-m4`. */
export function gemvWideKernelName(bits: GemvBits, prologue: GemvPrologue = 'none'): string {
  return `qgemv-${bits}bit${prologue === 'gelu' ? '-gelu' : ''}-m${GEMV_WIDE_COLS}`;
}

const WIDE_PARAMS_WGSL = `struct GemvParams {
  kWords: u32,
  kIters: u32,
  numRows: u32,
  // Live activation columns, 1 to ${GEMV_WIDE_COLS}. Uniform, so every guard on it is uniform.
  cols: u32,
  inScale: f32,
  outScale: f32,
  // vec4 stride between activation columns, K / 4.
  kVec4: u32,
  pad2: u32,
}`;

const SRQ_WGSL = `const SRQ_MAX: f32 = 127.0;
const SRQ_MIN: f32 = -128.0;

// The int8 codes as f32: exactly the integer srqIn of qgemv.ts produces, carried in f32 because
// every product and partial sum below is an integer under 2^24 and so exact in f32 under any
// order or contraction (the bound argument in qgemv.ts qgemvOracle); the f32 FMA units run
// several times the rate of i32 multiplies on Apple GPUs, which is what makes M columns fit
// under the weight stream.
fn srqInF(v: vec4<f32>) -> vec4<f32> {
  return clamp(round(v / params.inScale), vec4<f32>(SRQ_MIN), vec4<f32>(SRQ_MAX));
}

fn srqIn(v: vec4<f32>) -> vec4<i32> {
  return vec4<i32>(srqInF(v));
}

fn srqOut(v: f32) -> f32 {
  if (params.outScale == 0.0) {
    return v;
  }
  return clamp(round(v / params.outScale), SRQ_MIN, SRQ_MAX) * params.outScale;
}`;

/**
 * Columns one 32 lane subgroup carries. The 2-bit tile keeps two, because its sixteen row tile
 * already spends 50 registers a lane on accumulators at two columns and 100 at four. The 4-bit
 * and 8-bit path takes every column, for the reason in the header below.
 */
export const COLS_PER_LANE = 2;

/**
 * Rows one lane accumulates in the wide 4-bit and 8-bit kernel. The one column kernel spends
 * `rowsPerVsg` accumulators a lane; the wide kernel holds that budget by trading rows for
 * columns, so four columns run two rows where two columns ran four. Register pressure is the
 * same and the weight stream is read once instead of twice.
 */
export function gemvWideRows(geometry: Readonly<GemvGeometry>, cols: number): number {
  return Math.max(1, Math.floor((geometry.rowsPerVsg * COLS_PER_LANE) / cols));
}

/** Rows one wide 4-bit or 8-bit workgroup covers, which is what its dispatch is sized from. */
export function gemvWideRowsPerWorkgroup(geometry: Readonly<GemvGeometry>, cols: number): number {
  return (geometry.workgroupSize / 32) * gemvWideRows(geometry, cols);
}

function columnBases(cols: number): string {
  const lines = ['  let cLast = params.cols - 1u;'];
  for (let c = 0; c < cols; c += 1) {
    lines.push(`  let cb${c} = min(colBase + ${c}u, cLast) * params.kVec4;`);
  }
  return lines.join('\n');
}

/**
 * ONE WEIGHT READ, EVERY COLUMN. Round 5 built this kernel as the one column kernel replicated
 * once per column pair inside a wider workgroup, every replica walking the same weight words, on
 * the assumption that the second read would be served from cache and the DRAM stream paid once.
 * Round 7 measured that assumption and it is false. Against the one column kernel on the same
 * weight bytes, the wide kernel cost 2.02x at two live columns and 2.08x at four, and the two
 * figures being equal is the proof: the cost tracks the number of compiled column GROUPS, which
 * is two either way, not the number of live columns. Two groups over the same rows is two passes
 * over the weights, and on this device the second one is not free.
 *
 * So a lane now carries every column and the workgroup holds one group. The accumulator budget
 * is kept where it was by trading rows for columns (gemvWideRows): four columns run two rows a
 * lane where two columns ran four, so registers are unchanged, the workgroup covers half as many
 * rows, the dispatch is twice as wide, and the weight stream is read exactly once. Four columns
 * per lane at the FULL row count is what spilled in round 5 and ran three to four times slower;
 * the row trade is what makes four columns fit.
 *
 * The 2-bit tile keeps two columns a lane: its sixteen row tile already spends about 50
 * registers a lane on accumulators at two columns and would spend 100 at four, and its rows come
 * sixteen to a word so they cannot be traded away without reading each word twice instead.
 *
 * Column groups are subgroups, so a lane's columns are uniform across its subgroup and every
 * reduction below runs in uniform control flow before the one store block.
 */
export function qgemvWideWgsl(
  bits: GemvBits,
  variant: MatmulReduceVariant,
  geometry: Readonly<GemvGeometry> = gemvGeometry(bits),
  prologue: GemvPrologue = 'none',
  cols: number = GEMV_WIDE_COLS,
): string {
  if (cols % COLS_PER_LANE !== 0) throw new Error(`qgemv wide takes a multiple of ${COLS_PER_LANE} columns, got ${cols}`);
  if (bits === 2) return qgemv2TileWideWgsl(variant, geometry, prologue, cols);
  if (geometry.inner !== undefined && geometry.inner !== 'classic') {
    throw new Error(`qgemv-${bits}bit-m${cols} has only the classic inner loop, got ${geometry.inner}`);
  }
  const xr = (expr: string): string => (prologue === 'gelu' ? `act(${expr})` : `x[${expr}]`);
  // One group: every column rides the same weight read. See the header.
  const vsgPerGroup = geometry.workgroupSize / 32;
  const W = geometry.workgroupSize;
  const R = gemvWideRows(geometry, cols);
  const V = geometry.wordsPerLane;
  const rowsPerWg = gemvWideRowsPerWorkgroup(geometry, cols);
  const unpack = bits === 4 ? WGSL_UNPACK4 : WGSL_UNPACK8;
  const vec4PerWord = CODES_PER_WORD[bits] / 4;
  const wordType = V === 1 ? 'u32' : `vec${V}<u32>`;
  const wordNames = bits === 4 ? ['wLo', 'wHi'] : ['w8'];
  const cs = Array.from({ length: cols }, (_, c) => c);
  const rs = Array.from({ length: R }, (_, r) => r);
  const indent = (text: string, spaces: number): string => text
    .split('\n')
    .map((line) => (line.trim().length === 0 ? line : ' '.repeat(spaces) + line.trimStart()))
    .join('\n');

  // One lane vector at vector index `wi`: both columns' activation codes loaded once (as f32
  // integers, srqInF), then per row and word one unpack shared by the columns and one f32 dot
  // per column, exact by the bound argument on srqInF.
  const body = (snap: boolean): string => {
    const lines: string[] = [];
    lines.push(`      let xb0 = wi * ${V * vec4PerWord}u;`);
    for (const c of cs) {
      for (let cc = 0; cc < V; cc += 1) {
        for (let j = 0; j < vec4PerWord; j += 1) {
          const read = xr(`cb${c} + xb0 + ${cc * vec4PerWord + j}u`);
          lines.push(`      let x_${c}_${cc}_${j} = ${snap ? `srqInF(${read})` : read};`);
        }
      }
    }
    for (const r of rs) {
      lines.push(`      let wv${r} = wq[base${r} + wi];`);
      for (let cc = 0; cc < V; cc += 1) {
        const word = V === 1 ? `wv${r}` : `wv${r}[${cc}]`;
        lines.push('      {');
        lines.push(`        let w = ${word};`);
        lines.push(indent(unpack, 8));
        for (const c of cs) {
          const dots = wordNames.map((n, j) => `dot(${n}, x_${c}_${cc}_${j})`).join(' + ');
          lines.push(`        acc${r}_${c} = acc${r}_${c} + ${dots};`);
        }
        lines.push('      }');
      }
    }
    return lines.join('\n');
  };

  const rowDecls = rs.map((r) => [
    `  let row${r} = rowBase + ${r}u;`,
    `  let base${r} = min(row${r}, rLast) * kVec;`,
  ].join('\n')).join('\n');
  const accDecls = rs.flatMap((r) => cs.map((c) => `  var acc${r}_${c} = 0.0;`)).join('\n');
  const sums = rs.flatMap((r) => cs.map((c) => `  let sum${r}_${c} = mmSum(acc${r}_${c}, lid);`)).join('\n');
  const stores = cs.map((c) => [
    `    if (colBase + ${c}u < params.cols) {`,
    ...rs.map((r) => `      if (row${r} < params.numRows) { dst[(colBase + ${c}u) * params.numRows + row${r}] = srqOut((xs * scales[row${r}]) * sum${r}_${c}); }`),
    '    }',
  ].join('\n')).join('\n');

  return /* wgsl */ `${matmulReducePrelude(variant, W)}
${WIDE_PARAMS_WGSL}

@group(0) @binding(0) var<storage, read> wq: array<${wordType}>;
@group(0) @binding(1) var<storage, read> scales: array<f32>;
${prologueWgsl(prologue)}

${SRQ_WGSL}

@compute @workgroup_size(${W})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
  @builtin(local_invocation_index) lid: u32,
) {
  let lane = lid & 31u;
  let vsg = lid >> 5u;
  // The one column kernel's ${vsgPerGroup} row subgroups, replicated once per column pair.
  let rowVsg = vsg % ${vsgPerGroup}u;
  let colBase = (vsg / ${vsgPerGroup}u) * ${COLS_PER_LANE}u;
  let group = wid.x + nwg.x * wid.y;
  let rowBase = group * ${rowsPerWg}u + rowVsg * ${R}u;
  let rLast = params.numRows - 1u;
  let kVec = params.kWords / ${V}u;
${rowDecls}
${columnBases(cols)}
${accDecls}
  if (params.inScale != 0.0) {
    var wi = lane;
    for (var i = 0u; i < params.kIters; i = i + 1u) {
${body(true)}
      wi = wi + 32u;
    }
    if (wi < kVec) {
${body(true)}
    }
  } else {
    var wi = lane;
    for (var i = 0u; i < params.kIters; i = i + 1u) {
${body(false)}
      wi = wi + 32u;
    }
    if (wi < kVec) {
${body(false)}
    }
  }

${sums}
  if (lane == 0u) {
    let xs = select(1.0, params.inScale, params.inScale != 0.0);
${stores}
  }
}
`;
}

/**
 * The 2-bit tile kernel, M wide: one tile of sixteen rows per workgroup, one subgroup per column
 * pair, each subgroup running the tile16u loop of qgemv.ts over its two columns: the unsigned
 * pair codes of a word formed once and multiplied into both columns' pair accumulators, the
 * sixteen bit halves split per column after each block of sixteen iterations, the zero point
 * taken off with each column's own activation code sum. The f32 head path shares the unpacked
 * code per row across the two columns in the one column head's order.
 */
function qgemv2TileWideWgsl(variant: MatmulReduceVariant, geometry: Readonly<GemvGeometry>, prologue: GemvPrologue, cols: number): string {
  const xr = (expr: string): string => (prologue === 'gelu' ? `act(${expr})` : `x[${expr}]`);
  const groups = cols / COLS_PER_LANE;
  const W = 32 * groups;
  const inner = geometry.inner ?? 'tile16u';
  if (geometry.rowsPerVsg !== 16 || geometry.workgroupSize !== 32) {
    throw new Error(`qgemv-2bit-m${cols} is built on the 32 lane, sixteen row tile geometry, got ${geometry.workgroupSize}/${geometry.rowsPerVsg}`);
  }
  if (inner !== 'tile16u') throw new Error(`qgemv-2bit-m${cols} has only the tile16u loop, got ${inner}`);
  const rows = Array.from({ length: 16 }, (_, r) => r);
  const pairs = Array.from({ length: 8 }, (_, m) => m);
  const cs = [0, 1];
  const comps = ['x', 'y', 'z', 'w'];
  const shifted = (w: string, m: number): string => (m === 0 ? w : `(${w} >> ${2 * m}u)`);
  const intBody = comps.map((comp) => [
    '        {',
    `          let w = v.${comp};`,
    ...pairs.map((m) => `          let u${m} = i32(${shifted('w', m)} & 0x00030003u);`),
    ...cs.flatMap((c) => pairs.map((m) => `          p${m}_${c} = p${m}_${c} + u${m} * xk_${c}.${comp};`)),
    '        }',
  ].join('\n')).join('\n');
  const flush = cs.flatMap((c) => pairs
    .map((m) => `      { let lo = (p${m}_${c} << 16u) >> 16u; acc${m}_${c}i = acc${m}_${c}i + lo; acc${m + 8}_${c}i = acc${m + 8}_${c}i + ((p${m}_${c} - lo) >> 16u); p${m}_${c} = 0; }`))
    .join('\n');
  const headBody = comps.map((comp) => [
    '      {',
    `        let w = v.${comp};`,
    ...cs.map((c) => `        let xj_${c} = xk_${c}.${comp};`),
    ...cs.map((c) => `        sxf_${c} = sxf_${c} + xj_${c};`),
    ...rows.map((r) => `        let cw${r} = f32(${shifted('w', r)} & 3u);`),
    ...cs.flatMap((c) => rows.map((r) => `        acc${r}_${c} = acc${r}_${c} + cw${r} * xj_${c};`)),
    '      }',
  ].join('\n')).join('\n');
  const stores = cs.map((c) => [
    `    if (colBase + ${c}u < params.cols) {`,
    ...rows.map((r) => `      if (row0 + ${r}u < params.numRows) { dst[(colBase + ${c}u) * params.numRows + row0 + ${r}u] = srqOut((xs * scales[row0 + ${r}u]) * sum${r}_${c}); }`),
    '    }',
  ].join('\n')).join('\n');

  return /* wgsl */ `${matmulReducePrelude(variant, W)}
${WIDE_PARAMS_WGSL}

@group(0) @binding(0) var<storage, read> wq: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> scales: array<f32>;
${prologueWgsl(prologue)}

${SRQ_WGSL}

@compute @workgroup_size(${W})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
  @builtin(local_invocation_index) lid: u32,
) {
  let lane = lid & 31u;
  let vsg = lid >> 5u;
  // One tile per workgroup; subgroup vsg carries columns 2 vsg and 2 vsg + 1.
  let tile = wid.x + nwg.x * wid.y;
  let colBase = vsg * ${COLS_PER_LANE}u;
  let lastTile = (params.numRows - 1u) / 16u;
  let tileBase = min(tile, lastTile) * params.kWords * 4u;
  let iters = params.kWords / 8u;
${columnBases(COLS_PER_LANE)}
${cs.flatMap((c) => rows.map((r) => `  var acc${r}_${c} = 0.0;`)).join('\n')}
  if (params.inScale != 0.0) {
${cs.flatMap((c) => rows.map((r) => `    var acc${r}_${c}i: i32 = 0;`)).join('\n')}
${cs.flatMap((c) => pairs.map((m) => `    var p${m}_${c}: i32 = 0;`)).join('\n')}
${cs.map((c) => `    var sx_${c}: i32 = 0;`).join('\n')}
    var i = 0u;
    while (i < iters) {
      let stop = min(i + 16u, iters);
      for (; i < stop; i = i + 1u) {
        let v = wq[tileBase + i * 32u + lane];
${cs.map((c) => `        let xk_${c} = srqIn(${xr(`cb${c} + i * 32u + lane`)});`).join('\n')}
${cs.map((c) => `        sx_${c} = sx_${c} + xk_${c}.x + xk_${c}.y + xk_${c}.z + xk_${c}.w;`).join('\n')}
${intBody}
      }
${flush}
    }
${cs.flatMap((c) => rows.map((r) => `    acc${r}_${c} = f32(acc${r}_${c}i - 2 * sx_${c});`)).join('\n')}
  } else {
${cs.map((c) => `    var sxf_${c} = 0.0;`).join('\n')}
    for (var i = 0u; i < iters; i = i + 1u) {
      let v = wq[tileBase + i * 32u + lane];
${cs.map((c) => `      let xk_${c} = ${xr(`cb${c} + i * 32u + lane`)};`).join('\n')}
${headBody}
    }
${cs.map((c) => `    let sx2f_${c} = 2.0 * sxf_${c};`).join('\n')}
${cs.flatMap((c) => rows.map((r) => `    acc${r}_${c} = acc${r}_${c} - sx2f_${c};`)).join('\n')}
  }

${cs.flatMap((c) => rows.map((r) => `  let sum${r}_${c} = mmSum(acc${r}_${c}, lid);`)).join('\n')}
  if (lane == 0u) {
    let xs = select(1.0, params.inScale, params.inScale != 0.0);
    let row0 = tile * 16u;
${stores}
  }
}
`;
}

// ------------------------------------------------------------------------------------ oracle

/**
 * The known answer: the one column oracle over each column, laid out row major by column. This
 * is the definition; the shader's per column identity with qgemv.ts is what the sweep checks.
 */
export function qgemvWideOracle(
  bits: GemvBits,
  packed: Uint8Array,
  scales: ArrayLike<number>,
  x: Float32Array,
  cols: number,
  numRows: number,
  k: number,
  inScale = 0,
  outScale = 0,
): Float32Array {
  const out = new Float32Array(cols * numRows);
  for (let c = 0; c < cols; c += 1) {
    out.set(qgemvOracle(bits, packed, scales, x.subarray(c * k, (c + 1) * k), numRows, k, inScale, outScale), c * numRows);
  }
  return out;
}

export function qgemvWideGeluOracle(
  bits: GemvBits,
  packed: Uint8Array,
  scales: ArrayLike<number>,
  gate: Float32Array,
  up: Float32Array,
  cols: number,
  numRows: number,
  k: number,
  inScale = 0,
  outScale = 0,
): Float32Array {
  const act = new Float32Array(cols * k);
  for (let c = 0; c < cols; c += 1) {
    act.set(geluMulOracle(gate.subarray(c * k, (c + 1) * k), up.subarray(c * k, (c + 1) * k), k, inScale), c * k);
  }
  return qgemvWideOracle(bits, packed, scales, act, cols, numRows, k, inScale, outScale);
}

// -------------------------------------------------------------------------------------- bind

export function gemvWideParams(
  kWords: number,
  kIters: number,
  numRows: number,
  cols: number,
  inScale: number,
  outScale: number,
  kVec4: number,
): ArrayBuffer {
  const buf = new ArrayBuffer(32);
  const u = new Uint32Array(buf);
  const f = new Float32Array(buf);
  u[0] = kWords;
  u[1] = kIters;
  u[2] = numRows;
  u[3] = cols;
  f[4] = inScale;
  f[5] = outScale;
  u[6] = kVec4;
  u[7] = 0;
  return buf;
}

function bindGemvWide(bits: GemvBits, prologue: GemvPrologue = 'none') {
  return (input: KernelBindInput): KernelBindResult => {
    const { device, inputs, output, params } = input;
    const name = gemvWideKernelName(bits, prologue);
    const k = params.k | 0;
    const numRows = params.numRows | 0;
    const cols = params.cols | 0;
    if (k <= 0 || numRows <= 0) throw new Error(`${name} needs params.k and params.numRows`);
    if (cols <= 0 || cols > GEMV_WIDE_COLS) throw new Error(`${name} needs params.cols in 1..${GEMV_WIDE_COLS}, got ${cols}`);
    if (k % K_SPAN_PER_ITER[bits] !== 0) {
      throw new Error(`${name} needs K a multiple of ${K_SPAN_PER_ITER[bits]}, got ${k}`);
    }
    const inScale = params.inScale ?? 0;
    const outScale = params.outScale ?? 0;
    const kWords = wordsPerRow(bits, k);
    const geometry = gemvGeometry(bits);
    const kIters = Math.floor(kWords / (32 * geometry.wordsPerLane));
    const wq = inputs.wq;
    const scales = inputs.scales;
    if (!wq || !scales) throw new Error(`${name} needs inputs named wq and scales`);
    const activations: GPUBuffer[] = [];
    if (prologue === 'gelu') {
      if (!inputs.gate || !inputs.up) throw new Error(`${name} needs inputs named gate and up`);
      activations.push(inputs.gate, inputs.up);
    } else {
      if (!inputs.x) throw new Error(`${name} needs an input named x`);
      activations.push(inputs.x);
    }
    const readOnly = { visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' as const } };
    const layout = kernelLayout(input, name, () => device.createBindGroupLayout({
      label: name,
      entries: [
        { binding: 0, ...readOnly },
        { binding: 1, ...readOnly },
        ...activations.map((_, i) => ({ binding: 2 + i, ...readOnly })),
        { binding: 2 + activations.length, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' as const } },
        { binding: 3 + activations.length, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' as const } },
      ],
    }));
    const uniform = kernelUniform(
      input,
      `${name} params`,
      gemvWideParams(kWords, kIters, numRows, cols, inScale, outScale, k / 4),
    );
    return {
      layout,
      buffers: [wq, scales, ...activations, output, uniform.binding],
      // The 2-bit tile covers sixteen rows a workgroup whatever the column count; the 4-bit and
      // 8-bit path trades rows for columns, so its workgroup covers fewer rows than the one
      // column kernel's and needs proportionally more workgroups to reach numRows.
      dispatch: foldedDispatch(Math.ceil(numRows / (bits === 2
        ? gemvRowsPerWorkgroup(geometry)
        : gemvWideRowsPerWorkgroup(geometry, cols)))),
      dispose: uniform.dispose,
    };
  };
}

// ----------------------------------------------------------------------------------- registry

const SRQ_IN_THIRDS = 0.1875;
const SRQ_OUT = 0.0625;

function wideKernel(bits: GemvBits, prologue: GemvPrologue, cases: Kernel['cases']): Kernel {
  const name = gemvWideKernelName(bits, prologue);
  return {
    name,
    get wgsl(): string { return qgemvWideWgsl(bits, 'subgroup', undefined, prologue); },
    entry: 'main',
    note:
      `${name}: the ${bits}-bit decode GEMV${prologue === 'gelu' ? ' with the gelu prologue' : ''} over up to `
      + `${GEMV_WIDE_COLS} activation columns from one weight stream, column for column the one column kernel. `
      + 'The speculative verify pass runs the decoder on it (docs/ENGINE-PLAN.md round 5 ledger).',
    cases,
    bind: bindGemvWide(bits, prologue),
  };
}

export const qgemv4WideKernel: Kernel = wideKernel(4, 'none', [
  {
    name: 'synthetic-8x512-m4-srq',
    inputs: { wq: 'kmm.gemv4.wq', scales: 'kmm.gemv4.scales', x: 'kmm.x4.512' },
    expected: 'kmm.gemv4.m4.srq.expected',
    params: { k: 512, numRows: 8, cols: GEMV_WIDE_COLS, inScale: SRQ_IN_THIRDS, outScale: SRQ_OUT },
    tolAbs: 0,
    tolUlp: 0,
    note: 'Every live column is the one column kernel over its own vector, on the integer path.',
  },
  {
    name: 'synthetic-8x512-m4',
    inputs: { wq: 'kmm.gemv4.wq', scales: 'kmm.gemv4.scales', x: 'kmm.x4.512' },
    expected: 'kmm.gemv4.m4.expected',
    params: { k: 512, numRows: 8, cols: GEMV_WIDE_COLS },
    tolAbs: 0,
    tolUlp: 0,
    note: 'Every live column on the f32 path, grid activations, exact.',
  },
  {
    name: 'synthetic-8x512-m4-cols3',
    inputs: { wq: 'kmm.gemv4.wq', scales: 'kmm.gemv4.scales', x: 'kmm.x4.512' },
    expected: 'kmm.gemv4.m3.srq.expected',
    params: { k: 512, numRows: 8, cols: GEMV_WIDE_COLS - 1, inScale: SRQ_IN_THIRDS, outScale: SRQ_OUT },
    tolAbs: 0,
    tolUlp: 0,
    note: 'One live column short of the kernel width: the last column is neither read past cols nor stored.',
  },
]);

export const qgemv2WideKernel: Kernel = wideKernel(2, 'none', [
  {
    name: 'synthetic-8x1024-m4-srq',
    inputs: { wq: 'kmm.gemv2.tile.wq', scales: 'kmm.gemv2.scales', x: 'kmm.x4.1024' },
    expected: 'kmm.gemv2.m4.srq.expected',
    params: { k: 1024, numRows: 8, cols: GEMV_WIDE_COLS, inScale: SRQ_IN_THIRDS, outScale: SRQ_OUT },
    tolAbs: 0,
    tolUlp: 0,
    note: 'Every live column on the tile16u integer path.',
  },
  {
    name: 'synthetic-8x1024-m4',
    inputs: { wq: 'kmm.gemv2.tile.wq', scales: 'kmm.gemv2.scales', x: 'kmm.x4.1024' },
    expected: 'kmm.gemv2.m4.expected',
    params: { k: 1024, numRows: 8, cols: GEMV_WIDE_COLS },
    tolAbs: 0,
    tolUlp: 0,
    note: 'Every live column on the f32 head path, grid activations, exact.',
  },
]);

export const qgemv8WideKernel: Kernel = wideKernel(8, 'none', [
  {
    name: 'synthetic-10x256-m4-srq',
    inputs: { wq: 'kple.qm8.wq', scales: 'kple.qm8.scales', x: 'kple.x4.256' },
    expected: 'kple.qgemv8.m4.srq.expected',
    params: { k: 256, numRows: 10, cols: GEMV_WIDE_COLS, inScale: SRQ_IN_THIRDS, outScale: SRQ_OUT },
    tolAbs: 0,
    tolUlp: 0,
    note: 'Every live column of the I8 family on the integer path.',
  },
]);

export const qgemv4GeluWideKernel: Kernel = wideKernel(4, 'gelu', [
  {
    name: 'synthetic-8x512-gelu-m4-srq',
    inputs: { wq: 'kmm.gemv4.wq', scales: 'kmm.gemv4.scales', gate: 'kmm.gelu.gate4.512', up: 'kmm.gelu.up4.512' },
    expected: 'kmm.gemv4.gelu.m4.srq.expected',
    params: { k: 512, numRows: 8, cols: GEMV_WIDE_COLS, inScale: SRQ_IN_THIRDS, outScale: SRQ_OUT },
    tolAbs: 0,
    tolUlp: 0,
    note: 'Every live column through the gelu prologue on the integer path.',
  },
]);

export const qgemv2GeluWideKernel: Kernel = wideKernel(2, 'gelu', [
  {
    name: 'synthetic-8x1024-gelu-m4-srq',
    inputs: { wq: 'kmm.gemv2.tile.wq', scales: 'kmm.gemv2.scales', gate: 'kmm.gelu.gate4.1024', up: 'kmm.gelu.up4.1024' },
    expected: 'kmm.gemv2.gelu.m4.srq.expected',
    params: { k: 1024, numRows: 8, cols: GEMV_WIDE_COLS, inScale: SRQ_IN_THIRDS, outScale: SRQ_OUT },
    tolAbs: 0,
    tolUlp: 0,
    note: 'Every live column through the gelu prologue on the tile16u integer path.',
  },
]);

export const qgemv8GeluWideKernel: Kernel = wideKernel(8, 'gelu', [
  {
    name: 'synthetic-10x256-gelu-m4-srq',
    inputs: { wq: 'kple.qm8.wq', scales: 'kple.qm8.scales', gate: 'kple.gelu.gate4.256', up: 'kple.gelu.up4.256' },
    expected: 'kple.qgemv8.gelu.m4.srq.expected',
    params: { k: 256, numRows: 10, cols: GEMV_WIDE_COLS, inScale: SRQ_IN_THIRDS, outScale: SRQ_OUT },
    tolAbs: 0,
    tolUlp: 0,
    note: 'Every live column of the I8 family through the gelu prologue.',
  },
]);

/** The workgroup reduction builds, for the fallback table and the Node lints. */
export const QGEMV4_WIDE_FALLBACK_WGSL = qgemvWideWgsl(4, 'workgroup');
export const QGEMV2_WIDE_FALLBACK_WGSL = qgemvWideWgsl(2, 'workgroup');
export const QGEMV8_WIDE_FALLBACK_WGSL = qgemvWideWgsl(8, 'workgroup');
export const QGEMV4_GELU_WIDE_FALLBACK_WGSL = qgemvWideWgsl(4, 'workgroup', undefined, 'gelu');
export const QGEMV2_GELU_WIDE_FALLBACK_WGSL = qgemvWideWgsl(2, 'workgroup', undefined, 'gelu');
export const QGEMV8_GELU_WIDE_FALLBACK_WGSL = qgemvWideWgsl(8, 'workgroup', undefined, 'gelu');
