// Written from docs/ENGINE-PLAN.md section 5 (kernel K15) and 5.5, the miscompile record in
// PREFILL-CAMPAIGN.md rounds 2b and 3, and the WGSL specification. No vendored bundle, no
// extracted kernel and no third party engine source was read.
//
// SPDX-License-Identifier: Apache-2.0
//
// K15: argmax over the 262,144 logits, on the GPU, in two stages.
//
// WHY ON THE GPU AT ALL: keeping argmax on the GPU is what lets step N+1 be encoded without
// reading the token id back to JavaScript, which is a scheduling property worth having from day
// one rather than a late optimisation (ENGINE-PLAN K15). The head's logits never leave the
// device; only the winning id does, when the host wants text.
//
// WHY TWO STAGES: 262,144 elements over one workgroup would serialize the machine, and a single
// pass with atomics would make the answer depend on scheduling. Stage one gives each workgroup a
// contiguous segment and reduces it to one (value, index) pair; stage two reduces those pairs to
// the one winning index. Both stages use the same comparator and the same reduction ladder, so
// the result is a pure function of the logits.
//
// THE TIE RULE IS PART OF THE CONTRACT: the winner is the LOWEST index among maxima, everywhere.
// That is what a CPU scan taking the first strict maximum produces, it is what the reference's
// torch.argmax produces, and greedy decode parity (ENGINE-PLAN section 7 gate 2) is only
// meaningful if the GPU picks the same token on a tie. Every comparator below breaks value ties
// toward the smaller index, and the k-matmul check has a deliberate tie fixture that fails if
// anyone relaxes it. The comparator is select based lexicographic max on (value, -index), which
// is associative and commutative, so the answer is independent of the reduction order even
// though the order itself is still frozen as a matter of policy (ENGINE-PLAN risk 2).
//
// SUBGROUP POLICY, same as the matmul family: the 32 lane ladder uses subgroupShuffleXor with the
// mask sequence 1, 2, 4, 8, 16, matching subgroupReduce.ts, never a bare subgroup reduction
// after a divergent store (PREFILL-CAMPAIGN.md rounds 2b and 3); the portable variant performs
// the same pairing through workgroup memory and is selected at pipeline build from the device
// query plus the init self test verdict; masking is data level, with out of range slots carrying
// the LOWEST_F32 sentinel and the largest index so they cannot win.

import type { Kernel, KernelBindInput, KernelBindResult } from './registry';
import { kernelLayout, kernelUniform } from './binding';
import { SUBGROUP_ENABLE } from './subgroupReduce';

export const ARGMAX_WORKGROUP_SIZE = 64;

/** Stage one segment length. 64 workgroups cover the 262,144 vocabulary. */
export const ARGMAX_ELEMS_PER_WORKGROUP = 4096;

export type ArgmaxReduceVariant = 'subgroup' | 'workgroup';

/**
 * The (value, index) reduction ladder, both variants. `amaxPair(v, i, lid)` returns, in every
 * lane of a 32 lane virtual subgroup, the winning pair of that group: greatest value, smallest
 * index on ties.
 */
function argmaxReducePrelude(variant: ArgmaxReduceVariant): string {
  const step = (loadV: string, loadI: string) => `
    let ov = ${loadV};
    let oi = ${loadI};
    let take = (ov > bv) || ((ov == bv) && (oi < bi));
    bv = select(bv, ov, take);
    bi = select(bi, oi, take);`;

  if (variant === 'subgroup') {
    const rungs = [1, 2, 4, 8, 16]
      .map((m) => `  {${step(`subgroupShuffleXor(bv, ${m}u)`, `subgroupShuffleXor(bi, ${m}u)`)}
  }`)
      .join('\n');
    return `${SUBGROUP_ENABLE}
// Mask sequence 1, 2, 4, 8, 16, the same ladder subgroupReduce.ts uses, unrolled with constant
// masks like it, and like it never crossing an aligned 32 lane boundary, so a wider physical
// subgroup changes nothing.
fn amaxPair(v: f32, i: u32, lid: u32) -> vec2<u32> {
  var bv = v;
  var bi = i;
${rungs}
  return vec2<u32>(bitcast<u32>(bv), bi);
}
`;
  }
  return /* wgsl */ `
var<workgroup> amaxV: array<f32, ${ARGMAX_WORKGROUP_SIZE}>;
var<workgroup> amaxI: array<u32, ${ARGMAX_WORKGROUP_SIZE}>;

fn amaxPair(v: f32, i: u32, lid: u32) -> vec2<u32> {
  var bv = v;
  var bi = i;
  for (var m = 1u; m <= 16u; m = m << 1u) {
    // Order the stores of this step after the reads of the previous one, then publish, then
    // read the partner. Uniform, because amaxPair is only ever called unconditionally. An xor
    // with a mask below 32 stays inside the aligned 32 lane group by construction.
    workgroupBarrier();
    amaxV[lid] = bv;
    amaxI[lid] = bi;
    workgroupBarrier();
    let partner = lid ^ m;
${step('amaxV[partner]', 'amaxI[partner]')}
  }
  return vec2<u32>(bitcast<u32>(bv), bi);
}
`;
}

/**
 * The identity of the running max: the most negative finite f32, as an exact hex float literal.
 *
 * Not negative infinity, deliberately. WGSL has no infinity literal, and the obvious spelling,
 * `bitcast<f32>(0xff800000u)`, is a const expression: Tint evaluates it at shader creation and
 * rejects the resulting value ("value -inf cannot be represented as 'f32'"), on every backend,
 * under both reduce policies. The round 1 verifier caught exactly that, both argmax entries
 * refusing to compile anywhere.
 *
 * The finite sentinel cannot wrongly win: masked slots pair it with index 0xffffffff and the
 * comparator breaks value ties toward the smaller index, so any real element, including one equal
 * to the sentinel value, beats it. And no real input sits below it, because logits reach argmax
 * after the K14 softcap bounds them to [-30, 30], and stage two's pairs are stage one's winners
 * over those same values.
 */
const LOWEST_F32 = '-0x1.fffffep+127f';

/**
 * Stage one: each workgroup reduces its contiguous segment of the logits to one pair,
 * written as (value bits, index) into partials[workgroup].
 *
 * Lanes stride the segment by 64, so consecutive lanes read consecutive words, and a lane only
 * replaces its running best on a STRICT improvement, which keeps the earliest index within the
 * lane's ascending scan. Cross lane and cross virtual subgroup combines break ties toward the
 * smaller index, so the workgroup's pair is the segment's lexicographic maximum whatever the
 * interleave.
 */
export function argmaxPartialWgsl(variant: ArgmaxReduceVariant): string {
  return /* wgsl */ `${argmaxReducePrelude(variant)}
struct PartialParams {
  count: u32,
  elemsPerWg: u32,
  // elemsPerWg / ${ARGMAX_WORKGROUP_SIZE}: per lane scan length. Runtime opaque, registry rule 2.
  itersPerLane: u32,
  // Workgroups per row, ceil(count / elemsPerWg): row wid.y writes partials[wid.y * wgsPerRow + wid.x].
  // A verify pass carries one row of logits per position; a decode token is one row.
  wgsPerRow: u32,
}

@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> partials: array<vec2<u32>>;
@group(0) @binding(2) var<uniform> params: PartialParams;

var<workgroup> vsgBest: array<vec2<u32>, 2>;

@compute @workgroup_size(${ARGMAX_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_index) lid: u32,
) {
  let lane = lid & 31u;
  let rowBase = wid.y * params.count;
  let segBase = wid.x * params.elemsPerWg;
  let last = params.count - 1u;

  var bv = ${LOWEST_F32};
  var bi = 0xffffffffu;
  var idx = segBase + lid;
  for (var t = 0u; t < params.itersPerLane; t = t + 1u) {
    // Data level masking: an out of range slot reads a clamped element but competes as the
    // lowest finite f32 at the largest index, so it can never win against a real value.
    let inRange = idx < params.count;
    let v = select(${LOWEST_F32}, logits[rowBase + min(idx, last)], inRange);
    let i = select(0xffffffffu, idx, inRange);
    let take = (v > bv) || ((v == bv) && (i < bi));
    bv = select(bv, v, take);
    bi = select(bi, i, take);
    idx = idx + ${ARGMAX_WORKGROUP_SIZE}u;
  }

  // Ladder first, in uniform control flow. The one divergent store below it publishes each
  // virtual subgroup's pair, and nothing subgroup shaped runs after that store. Rule 1.
  let pair = amaxPair(bv, bi, lid);
  if (lane == 0u) {
    vsgBest[lid >> 5u] = pair;
  }
  workgroupBarrier();
  // Cross virtual subgroup combine in a fixed order, then one guarded store.
  let a = vsgBest[0];
  let b = vsgBest[1];
  let av = bitcast<f32>(a.x);
  let bvv = bitcast<f32>(b.x);
  let takeB = (bvv > av) || ((bvv == av) && (b.y < a.y));
  if (lid == 0u) {
    partials[wid.y * params.wgsPerRow + wid.x] = select(a, b, takeB);
  }
}
`;
}

/**
 * Stage two: one workgroup reduces the stage one pairs to the winning index. Output is two
 * words: the index, then the winning value's bits, both exact, so the case gates at zero.
 */
export function argmaxFinalWgsl(variant: ArgmaxReduceVariant): string {
  return /* wgsl */ `${argmaxReducePrelude(variant)}
struct FinalParams {
  pairCount: u32,
  itersPerLane: u32,
  pad0: u32,
  pad1: u32,
}

@group(0) @binding(0) var<storage, read> partials: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;
@group(0) @binding(2) var<uniform> params: FinalParams;

var<workgroup> vsgBest: array<vec2<u32>, 2>;

@compute @workgroup_size(${ARGMAX_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_index) lid: u32,
) {
  let lane = lid & 31u;
  let last = params.pairCount - 1u;
  // Row wid.y reads its own pairCount partials and writes out[2 wid.y], out[2 wid.y + 1].
  let rowBase = wid.y * params.pairCount;

  var bv = ${LOWEST_F32};
  var bi = 0xffffffffu;
  var idx = lid;
  for (var t = 0u; t < params.itersPerLane; t = t + 1u) {
    let inRange = idx < params.pairCount;
    let p = partials[rowBase + min(idx, last)];
    let v = select(${LOWEST_F32}, bitcast<f32>(p.x), inRange);
    let i = select(0xffffffffu, p.y, inRange);
    let take = (v > bv) || ((v == bv) && (i < bi));
    bv = select(bv, v, take);
    bi = select(bi, i, take);
    idx = idx + ${ARGMAX_WORKGROUP_SIZE}u;
  }

  let pair = amaxPair(bv, bi, lid);
  if (lane == 0u) {
    vsgBest[lid >> 5u] = pair;
  }
  workgroupBarrier();
  let a = vsgBest[0];
  let b = vsgBest[1];
  let av = bitcast<f32>(a.x);
  let bvv = bitcast<f32>(b.x);
  let takeB = (bvv > av) || ((bvv == av) && (b.y < a.y));
  let winner = select(a, b, takeB);
  if (lid == 0u) {
    out[2u * wid.y] = winner.y;
    out[2u * wid.y + 1u] = winner.x;
  }
}
`;
}

export const ARGMAX_PARTIAL_WGSL = argmaxPartialWgsl('subgroup');
export const ARGMAX_FINAL_WGSL = argmaxFinalWgsl('subgroup');
export const ARGMAX_PARTIAL_FALLBACK_WGSL = argmaxPartialWgsl('workgroup');
export const ARGMAX_FINAL_FALLBACK_WGSL = argmaxFinalWgsl('workgroup');

// ---------------------------------------------------------------------------------------------
// The CPU oracles.
// ---------------------------------------------------------------------------------------------

/**
 * First strict maximum, which is the lowest index among maxima, matching torch.argmax and the
 * shader's tie rule. NaN values never win because every comparison against them is false, which
 * is also how the shader's predicate treats them; the logits this engine feeds it are softcapped
 * to [-30, 30] so the question is theoretical, but the two implementations agreeing on it is not.
 */
export function argmaxOracle(values: Float32Array, start = 0, end = values.length): { index: number; value: number } {
  let index = -1;
  let value = Number.NEGATIVE_INFINITY;
  for (let i = start; i < end; i += 1) {
    if (values[i] > value) {
      value = values[i];
      index = i;
    }
  }
  return { index, value };
}

/**
 * Stage one's expected output: one (value bits, index) pair per segment, as an Int32Array laid
 * out [pairCount, 2] for the manifest. Order independent by the comparator's associativity, so a
 * plain ascending scan per segment is the reference.
 */
export function argmaxPartialsOracle(values: Float32Array, elemsPerWg: number): Int32Array {
  const wgs = Math.ceil(values.length / elemsPerWg);
  const out = new Int32Array(wgs * 2);
  const bits = new Int32Array(values.buffer, values.byteOffset, values.length);
  for (let w = 0; w < wgs; w += 1) {
    const { index } = argmaxOracle(values, w * elemsPerWg, Math.min((w + 1) * elemsPerWg, values.length));
    out[w * 2] = bits[index];
    out[w * 2 + 1] = index;
  }
  return out;
}

/** Stage two's expected output over stage one pairs: [winning index, winning value bits]. */
export function argmaxFinalOracle(pairs: Int32Array): Int32Array {
  const f = new Float32Array(pairs.buffer, pairs.byteOffset, pairs.length);
  let bestValue = Number.NEGATIVE_INFINITY;
  let bestIndex = 0xffffffff;
  let bestBits = 0;
  for (let p = 0; p < pairs.length; p += 2) {
    const v = f[p];
    const i = pairs[p + 1] >>> 0;
    if (v > bestValue || (v === bestValue && i < bestIndex)) {
      bestValue = v;
      bestIndex = i;
      bestBits = pairs[p];
    }
  }
  const out = new Int32Array(2);
  out[0] = bestIndex | 0;
  out[1] = bestBits;
  return out;
}

// ---------------------------------------------------------------------------------------------
// Registry entries.
// ---------------------------------------------------------------------------------------------

function uniformWords(values: number[]): ArrayBuffer {
  const buf = new ArrayBuffer(16);
  new Uint32Array(buf).set(values);
  return buf;
}

/** Stage one params, exported so the scheduler can restage the block in place per dispatch. */
export function argmaxPartialParams(count: number, elemsPerWg: number): ArrayBuffer {
  return uniformWords([count, elemsPerWg, elemsPerWg / ARGMAX_WORKGROUP_SIZE, Math.ceil(count / elemsPerWg)]);
}

/** Stage two params, exported for the same in place restaging. */
export function argmaxFinalParams(pairCount: number): ArrayBuffer {
  return uniformWords([pairCount, Math.ceil(pairCount / ARGMAX_WORKGROUP_SIZE), 0, 0]);
}

function bindPartial(input: KernelBindInput): KernelBindResult {
  const { device, inputs, output, params } = input;
  const count = params.count | 0;
  const elemsPerWg = (params.elemsPerWg | 0) || ARGMAX_ELEMS_PER_WORKGROUP;
  if (count <= 0) throw new Error('argmax-partial needs params.count');
  if (elemsPerWg % ARGMAX_WORKGROUP_SIZE !== 0) {
    throw new Error(`argmax-partial needs elemsPerWg a multiple of ${ARGMAX_WORKGROUP_SIZE}`);
  }
  const logits = inputs.logits;
  if (!logits) throw new Error('argmax-partial needs an input named logits');
  const wgs = Math.ceil(count / elemsPerWg);

  const layout = kernelLayout(input, 'argmax-partial', () => device.createBindGroupLayout({
    label: 'argmax-partial',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  }));
  const uniform = kernelUniform(input, 'argmax-partial params', argmaxPartialParams(count, elemsPerWg));

  return {
    layout,
    buffers: [logits, output, uniform.binding],
    dispatch: [wgs, Math.max(1, params.rows | 0), 1],
    dispose: uniform.dispose,
  };
}

function bindFinal(input: KernelBindInput): KernelBindResult {
  const { device, inputs, output, params } = input;
  const pairCount = params.pairCount | 0;
  if (pairCount <= 0) throw new Error('argmax-final needs params.pairCount');
  const partials = inputs.partials;
  if (!partials) throw new Error('argmax-final needs an input named partials');

  const layout = kernelLayout(input, 'argmax-final', () => device.createBindGroupLayout({
    label: 'argmax-final',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  }));
  const uniform = kernelUniform(input, 'argmax-final params', argmaxFinalParams(pairCount));

  return {
    layout,
    buffers: [partials, output, uniform.binding],
    dispatch: [1, Math.max(1, params.rows | 0), 1],
    dispose: uniform.dispose,
  };
}

export const argmaxPartialKernel: Kernel = {
  name: 'argmax-partial',
  wgsl: ARGMAX_PARTIAL_WGSL,
  entry: 'main',
  note:
    'K15 stage one. One (value, index) pair per 4096 element segment, lowest index wins ties. '
    + 'Comparisons are exact, so every case gates at zero.',
  cases: [
    {
      name: 'real-logits-262144',
      inputs: { logits: 'interview-300.logits-last' },
      expected: 'kmm.argmax.partials.expected',
      params: { count: 262144, elemsPerWg: 4096 },
      tolAbs: 0,
      note: 'The reference dump logits for the interview probe, all 64 segments.',
    },
    {
      name: 'ties-and-tail',
      inputs: { logits: 'kmm.argmax.ties' },
      expected: 'kmm.argmax.ties.partials.expected',
      params: { count: 5000, elemsPerWg: 4096 },
      tolAbs: 0,
      note:
        'A duplicated maximum inside one segment and a ragged final segment. Fails if anyone '
        + 'relaxes the lowest index tie rule or the tail masking.',
    },
  ],
  bind: bindPartial,
};

export const argmaxFinalKernel: Kernel = {
  name: 'argmax-final',
  wgsl: ARGMAX_FINAL_WGSL,
  entry: 'main',
  note:
    'K15 stage two. Reduces the stage one pairs to [index, value bits]. On the reference logits '
    + 'the index must equal the dump greedy continuation first token, which ties the whole '
    + 'pipeline to the transformers reference.',
  cases: [
    {
      name: 'real-logits-final',
      inputs: { partials: 'kmm.argmax.partials.expected' },
      expected: 'kmm.argmax.final.expected',
      params: { pairCount: 64 },
      tolAbs: 0,
      note: 'Stage two over the oracle checked stage one pairs of the interview probe logits.',
    },
  ],
  bind: bindFinal,
};
