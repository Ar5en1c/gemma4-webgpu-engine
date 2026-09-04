// Written from docs/ENGINE-PLAN.md section 3 (the tensor family table and the dequant maths),
// section 5 kernels K1 and K2, EMBEDDING-QUANT-FINDING.md (the PLE dequant, its group layout and
// the 4-bit floor), PREFILL-CAMPAIGN.md's retile result for the 2-bit packing, and the WGSL
// specification. No vendored bundle, no extracted kernel and no third party engine source was
// read.
//
// SPDX-License-Identifier: Apache-2.0
//
// The two gathers, K1 and K2, from one builder.
//
// Both are the same kernel with different constants: read a row of packed low bit codes indexed by
// a token id, dequantize four codes at a time, write f32. What differs is the bit width, the zero
// point, how many scales a row carries, and whether a constant multiplier is folded in. Writing
// them as one builder is not a saving of forty lines, it is the difference between one dequant
// expression and two that can disagree.
//
// K2, THE PLE GATHER, is the one with measured ground truth. `y = 16 * scale[id, layer] *
// (code - 8)`, 4-bit codes packed 8 to a u32, GROUP_SIZE 256, NUM_GROUPS 35, zero point 8
// (EMBEDDING-QUANT-FINDING.md). One group per layer, so a row is 35 * 256 = 8960 values and each
// group carries its own scale. The table is [262144, 8960] at 4 bits, which is 1.174 GB and the
// dominant download.
//
// DO NOT LOWER THE BIT WIDTH. This project measured PLE at 3-bit and at 2-bit by fake quantizing
// the live buffer. Easy prompts stayed fluent, grammatical and factually correct at both. A
// precise recall probe went from the right answer at 4-bit to a wrong one at 3-bit and at 2-bit,
// and back to the right one on restore (EMBEDDING-QUANT-FINDING.md). The roughly 590 MB a 2-bit
// PLE would save is not free and the failure mode is invisible to short prompts. The methodology
// lesson is worth as much as the result: easy short prompts gave a false negative.
//
// K1, THE INPUT EMBEDDING GATHER, is 2-bit at 0.10 GB, which is exactly [262144, 1536] at 2 bits
// with no slack, so there is no room in that figure for anything but codes and the scales live in
// their own tensors. Its zero point of 2 and its embed scale of sqrt(1536) are taken from
// src/engine/gemma4/quant.ts rather than restated here. The one thing this file still does not
// know is how many scales a row carries, and it does not need to: the group size reaches the
// shader through the uniform, so whatever the checkpoint's scale tensor shape turns out to be at
// M1 is a parameter and not an edit to a shader.
//
// Neither kernel streams. Both gather one row per token out of a table that stays resident, so the
// table's size drives the download and not the per token cost (DECODE-CAMPAIGN.md 1). That is why
// they are cheap and why they are also the reason the model is 2.46 GB.

import type { Kernel, KernelBindInput, KernelBindResult } from './registry';
import { kernelLayout, kernelUniform } from './binding';
import { tile16Code } from './qlayout';
import { HIDDEN_SIZE, PLE_GROUPS, PLE_GROUP_SIZE, PLE_ROW_VALUES } from './layerGeometry';
// The quant constants come from the loader lane's module rather than being restated here. Two
// families that each keep their own copy of a zero point are two families that can disagree about
// it, and the disagreement would be a wrong number rather than a build error.
import { EMBED_TOKENS_EMBED_SCALE, PLE_EMBED_SCALE, ZERO_POINT } from '../quant';
// The split planner, so a table too large for one storage binding is a plan rather than a crash.
// Nothing here reads an adapter: the plan arrives as data and this file only obeys it.
import type { TableSlice, TableSplitPlan } from '../tableSplit';

const WORKGROUP_SIZE = 64;

/**
 * Emit a WGSL f32 literal that round trips.
 *
 * Worth a function rather than a `toFixed`: the input embedding's multiplier is the square root of
 * 1536, and a literal printed to one decimal place would be a shader that quietly computes with
 * 39.2. Round to f32 first, then print the shortest decimal that reads back as the same f32.
 */
export function wgslF32(value: number): string {
  const rounded = Math.fround(value);
  const text = String(rounded);
  return /[.eE]/.test(text) ? text : `${text}.0`;
}

export interface EmbedGatherSpec {
  /** Registry kernel name. */
  name: string;
  /** 2 for embed_tokens, 4 for the PLE table. */
  bits: 2 | 4;
  /** Subtracted from each code before scaling. 8 at 4 bits, 2 at 2 bits. */
  zeroPoint: number;
  /** Values per row: 1536 for embed_tokens, 8960 for a PLE row. */
  cols: number;
  /** Values sharing one scale. 256 for the PLE, the whole row by default for embed_tokens. */
  groupSize: number;
  /** Constant folded into the scale. 16 for the PLE, 1 for embed_tokens. */
  outputScale: number;
  /** True when the numbers above are measured rather than inferred. */
  measured: boolean;
  note: string;
}

export const PLE_GATHER_SPEC: EmbedGatherSpec = {
  name: 'ple-gather-4bit',
  bits: 4,
  zeroPoint: ZERO_POINT[4],
  cols: PLE_ROW_VALUES,
  groupSize: PLE_GROUP_SIZE,
  outputScale: PLE_EMBED_SCALE,
  measured: true,
  note: 'y = 16 * scale[id, layer] * (code - 8), from EMBEDDING-QUANT-FINDING.md.',
};

export const EMBED_TOKENS_GATHER_SPEC: EmbedGatherSpec = {
  name: 'embed-gather-2bit',
  bits: 2,
  zeroPoint: ZERO_POINT[2],
  cols: HIDDEN_SIZE,
  // One scale per row is the default the fixtures use. The real number comes off the checkpoint's
  // scale tensor shape at load time and reaches the shader through the uniform, so a different
  // group layout is a parameter change and not a shader change.
  groupSize: HIDDEN_SIZE,
  outputScale: EMBED_TOKENS_EMBED_SCALE,
  measured: true,
  note:
    'Zero point 2 and the embed scale of sqrt(1536) both come from src/engine/gemma4/quant.ts, '
    + 'which the loader lane wrote against the reference implementation. One difference in '
    + 'association is recorded rather than papered over: quant.ts computes code * scale * '
    + 'embedScale, this shader folds embedScale into the scale once per group and then does one '
    + 'multiply per element. For the PLE the two are the same number because 16 is a power of two. '
    + 'For the input embedding sqrt(1536) is not, so the two orders can differ in the last bit. The '
    + 'fold is worth it here: it is one multiply per element instead of two on a gather of 1536 '
    + 'values, and ENGINE-PLAN section 7 already settles that bit identity is not the contract for '
    + 'a shape like this. Flagged to the lead so the CPU reference forward picks one order and both '
    + 'sides use it.',
};

export function codesPerWord(bits: number): number {
  return 32 / bits;
}

export function rowStrideWords(spec: EmbedGatherSpec): number {
  const per = codesPerWord(spec.bits);
  if (spec.cols % per !== 0) {
    throw new Error(`${spec.name}: ${spec.cols} columns do not pack evenly at ${spec.bits} bits`);
  }
  return spec.cols / per;
}

export function groupsPerRow(spec: EmbedGatherSpec): number {
  if (spec.cols % spec.groupSize !== 0) {
    throw new Error(`${spec.name}: group size ${spec.groupSize} does not divide ${spec.cols}`);
  }
  return spec.cols / spec.groupSize;
}

/**
 * Build the WGSL for one gather.
 *
 * Four codes are unpacked per invocation and written as one `vec4<f32>`. Four consecutive codes
 * always sit inside one 32 bit word at both widths, because 4 divides 8 and 16, so an output lane
 * does exactly one word load and no lane ever straddles a word boundary. That is what makes this
 * a coalesced read of the table rather than a scatter.
 *
 * The multiplier is folded into the scale rather than applied to the result. `16 * scale` is exact
 * because 16 is a power of two, so folding does not move a single bit, and it turns two multiplies
 * per element into one multiply per group plus one per element. The oracle folds in the same order
 * for the same reason, which is what lets these two cases gate at zero tolerance.
 *
 * No reduction, so ENGINE-PLAN 5.5 rules 1 and 2 have nothing to bind to. Four storage bindings
 * against a limit of 10 (risk 6).
 */
export interface EmbedGatherBuildOptions {
  /**
   * Build the form that reads one vocabulary range slice of a split table.
   *
   * Off by default and off on every adapter this project has measured, because the M1 reports a
   * 4 GB storage binding limit and the whole PLE table is one buffer there. When it is on, the
   * params block carries two more words naming the slice's vocabulary range, the codes buffer is
   * the slice rather than the table, and an invocation whose token id falls outside the range
   * returns without writing so another dispatch can write it. See ../tableSplit.ts for why the
   * split is by row and why the dispatch binds one slice rather than all of them.
   *
   * The unsliced build's text is unchanged by this parameter existing, to the byte. That is
   * asserted by scripts/engine-check/loader.mjs against a pinned digest, because a kernel whose
   * text moves is a kernel whose bit identity claim has to be re-earned and this change earns
   * nothing on the machine that runs it.
   */
  readonly sliced?: boolean;
}

export function embedGatherWgsl(spec: EmbedGatherSpec, options: EmbedGatherBuildOptions = {}): string {
  const per = codesPerWord(spec.bits);
  const mask = (1 << spec.bits) - 1;
  const sliced = options.sliced === true;
  // Two extra params, one guard and one rebased row index, all of them absent from the unsliced
  // build so its text is what it has always been.
  const sliceFields = sliced
    ? `
  // First vocabulary row this slice carries, and how many. Absent on the unsplit build.
  sliceStartRow: u32,
  sliceRowCount: u32,`
    : '';
  const sliceGuard = sliced
    ? `
  // The slices partition the vocabulary, so a slot outside this slice is written by another
  // dispatch. Returning is correct here and leaving the destination alone is the point.
  if (row < params.sliceStartRow || row - params.sliceStartRow >= params.sliceRowCount) {
    return;
  }
  let localRow = row - params.sliceStartRow;`
    : '';
  // Scales stay one buffer and stay addressed by the absolute row, which tableSplit.ts checks.
  const codeRow = sliced ? 'localRow' : 'row';
  return /* wgsl */ `
struct GatherParams {
  // Output vec4 lanes per row: 384 for the input embedding, 2240 for a PLE row. Runtime opaque.
  vec4Count: u32,
  // Values sharing one scale.
  groupSize: u32,
  // Scales per row, so the scale index is row * groupsPerRow + column / groupSize.
  groupsPerRow: u32,
  // Packed u32 words per row.
  rowStrideWords: u32,${sliceFields}
}

@group(0) @binding(0) var<storage, read> codes: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<f32>;
// Token ids, one per output slot. They arrive as a storage buffer rather than as a uniform
// because argmax stays on the GPU: step N+1 is encoded without reading step N's token back to JS
// (ENGINE-PLAN kernel K15). Ids are non negative, so an i32 reference buffer reads correctly here.
@group(0) @binding(2) var<storage, read> ids: array<u32>;
@group(0) @binding(3) var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(4) var<uniform> params: GatherParams;

const BITS: u32 = ${spec.bits}u;
const MASK: u32 = ${mask}u;
const PER_WORD: u32 = ${per}u;
const ZERO_POINT: f32 = ${wgslF32(spec.zeroPoint)};
const OUT_SCALE: f32 = ${wgslF32(spec.outputScale)};

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let j = gid.x;
  if (j >= params.vec4Count) {
    return;
  }
  let slot = gid.y;
  let row = ids[slot];${sliceGuard}
  let col = j * 4u;

${spec.bits === 2 ? `  // The 2-bit table is in the interleaved layout (qlayout.ts TILE_ROWS): a tile of sixteen rows
  // holds one word per column, the row's code at bit offset 2 * (row % 16), so four consecutive
  // columns are four consecutive words. A sliced table starts every slice on a tile boundary
  // (tableSplit.ts), which is what lets the slice local row index the slice's own tiles.
  let tileStride = params.rowStrideWords * 16u;
  let wordBase = (${codeRow} / 16u) * tileStride + col;
  let shift = (${codeRow} % 16u) * 2u;` : `  let word = codes[${codeRow} * params.rowStrideWords + col / PER_WORD];
  let shift = (col % PER_WORD) * BITS;`}

  // One fold per group, then one multiply per element. On the PLE this is exactly the measured
  // 16 * scale[id, layer] and the fold is bit exact because 16 is a power of two. No backtick in
  // this comment on purpose: the shader lives inside a TypeScript template string.
  let scale = OUT_SCALE * scales[row * params.groupsPerRow + col / params.groupSize];

${spec.bits === 2 ? `  let c0 = f32((codes[wordBase] >> shift) & MASK);
  let c1 = f32((codes[wordBase + 1u] >> shift) & MASK);
  let c2 = f32((codes[wordBase + 2u] >> shift) & MASK);
  let c3 = f32((codes[wordBase + 3u] >> shift) & MASK);` : `  let c0 = f32((word >> (shift)) & MASK);
  let c1 = f32((word >> (shift + BITS)) & MASK);
  let c2 = f32((word >> (shift + 2u * BITS)) & MASK);
  let c3 = f32((word >> (shift + 3u * BITS)) & MASK);`}

  dst[slot * params.vec4Count + j] = (vec4<f32>(c0, c1, c2, c3) - vec4<f32>(ZERO_POINT)) * scale;
}
`;
}

export const PLE_GATHER_WGSL = embedGatherWgsl(PLE_GATHER_SPEC);
export const EMBED_TOKENS_GATHER_WGSL = embedGatherWgsl(EMBED_TOKENS_GATHER_SPEC);

// ---------------------------------------------------------------------------------------------
// The CPU oracle, and the packer the fixtures need.
// ---------------------------------------------------------------------------------------------

/**
 * Pack codes little endian, `32 / bits` to a word, low order code first.
 *
 * Exported because a test that packs one way and a shader that unpacks another is a test of
 * nothing.
 *
 * This produces the same bytes as `packCodes` in src/engine/gemma4/quant.ts on a little endian
 * host, and that is a fact worth checking rather than assuming, so scripts/engine-check.mjs
 * section k-norms checks it. At 2 bits a word holds 16 codes at shifts 0, 2, 4 and so on, so its
 * first byte holds codes 0 to 3 at bit offsets 0, 2, 4, 6, which is exactly what `unpackInt2`
 * reads. At 4 bits a word holds 8 codes and its first byte holds codes 0 and 1 as the low then the
 * high nibble, which is what `unpackInt4` reads. The consequence is the one that matters for the
 * load path: the loader uploads the checkpoint's bytes unchanged and this shader reads them
 * correctly, with no repack step between the network and the GPU.
 */
export function packCodes(codes: Uint8Array, bits: 2 | 4): Uint32Array {
  const per = codesPerWord(bits);
  if (codes.length % per !== 0) {
    throw new Error(`packCodes: ${codes.length} codes do not fill whole words at ${bits} bits`);
  }
  const mask = (1 << bits) - 1;
  const words = new Uint32Array(codes.length / per);
  for (let i = 0; i < codes.length; i += 1) {
    const w = (i / per) | 0;
    const shift = (i % per) * bits;
    words[w] |= (codes[i] & mask) << shift;
  }
  return words;
}

/** Read one code back out, the inverse of `packCodes`. */
export function unpackCode(words: Uint32Array, index: number, bits: 2 | 4): number {
  const per = codesPerWord(bits);
  const mask = (1 << bits) - 1;
  const shift = (index % per) * bits;
  return (words[(index / per) | 0] >>> shift) & mask;
}

/**
 * The known answer reference for both gathers.
 *
 * The association matches the shader exactly: fold the multiplier into the scale first, then one
 * multiply per element. Every step is a single correctly rounded f32 operation on both sides, so
 * these cases gate at zero and mean it.
 *
 * @param codes  packed rows, `rowStrideWords` words each
 * @param scales `rows * groupsPerRow` f32 scales
 * @param ids    one row index per output slot
 */
export function embedGatherOracle(
  codes: Uint32Array,
  scales: Float32Array,
  ids: Uint32Array | Int32Array,
  spec: EmbedGatherSpec,
): Float32Array {
  const stride = rowStrideWords(spec);
  const groups = groupsPerRow(spec);
  // The shader's OUT_SCALE is an f32 literal, so the oracle folds with the f32 value too.
  const outScale = Math.fround(spec.outputScale);
  const out = new Float32Array(ids.length * spec.cols);
  for (let slot = 0; slot < ids.length; slot += 1) {
    const row = ids[slot];
    const wordBase = row * stride;
    const outBase = slot * spec.cols;
    for (let col = 0; col < spec.cols; col += 1) {
      const scale = Math.fround(outScale * scales[row * groups + Math.floor(col / spec.groupSize)]);
      // The 2-bit table is read in the interleaved layout, the 4-bit PLE table in the row layout.
      const code = spec.bits === 2
        ? tile16Code(codes, spec.cols, row, col)
        : unpackCode(codes, wordBase * codesPerWord(spec.bits) + col, spec.bits);
      out[outBase + col] = Math.fround((code - spec.zeroPoint) * scale);
    }
  }
  return out;
}

/**
 * The PLE contribution for one layer, sliced out of a gathered row.
 *
 * A PLE row is 35 groups of 256, one group per layer, so layer L's 256 values are at
 * `[L * 256, L * 256 + 256)`. Kept here rather than in the caller so the layout is stated once.
 */
export function pleLayerSlice(row: Float32Array, layer: number): Float32Array {
  if (layer < 0 || layer >= PLE_GROUPS) throw new Error(`pleLayerSlice: no layer ${layer}`);
  const start = layer * PLE_GROUP_SIZE;
  return row.subarray(start, start + PLE_GROUP_SIZE);
}

// ---------------------------------------------------------------------------------------------
// Registry entries.
// ---------------------------------------------------------------------------------------------

/**
 * The params block for a gather spec, exported for the scheduler's in place restaging.
 *
 * With a slice it is the sliced build's six word block instead of the four word one, and the two
 * extra words are the slice's vocabulary range. Without one it is exactly the block it has always
 * been, sixteen bytes, because the unsliced build's struct did not change.
 */
export function embedGatherParams(spec: EmbedGatherSpec, slice?: TableSlice): ArrayBuffer {
  const words = new ArrayBuffer(slice ? 24 : 16);
  const u = new Uint32Array(words);
  u[0] = spec.cols / 4;
  u[1] = spec.groupSize;
  u[2] = groupsPerRow(spec);
  u[3] = rowStrideWords(spec);
  if (slice) {
    u[4] = slice.startRow;
    u[5] = slice.rowCount;
  }
  return words;
}

/** One dispatch of the sliced gather: which slice buffer to bind and what to put in its uniform. */
export interface SlicedGatherDispatch {
  readonly sliceIndex: number;
  readonly slice: TableSlice;
  /** Byte range of the whole code table this slice's buffer holds. */
  readonly byteOffset: number;
  readonly byteLength: number;
  readonly params: ArrayBuffer;
}

/**
 * The dispatch list for one gather over a split table.
 *
 * One entry per slice, in slice order, each naming the buffer to bind and the uniform to stage.
 * The workgroup geometry is the same for every entry and the same as the unsliced build's, because
 * the split changes which rows a dispatch can serve and nothing about how a row is read.
 *
 * A single slice plan produces one entry whose range is the whole table, so a caller does not
 * branch on `plan.single`: it walks this list either way and the M1 walks a list of one.
 */
export function slicedGatherDispatches(
  spec: EmbedGatherSpec,
  plan: TableSplitPlan,
): SlicedGatherDispatch[] {
  return plan.slices.map((slice) => ({
    sliceIndex: slice.index,
    slice,
    byteOffset: slice.byteOffset,
    byteLength: slice.byteLength,
    params: embedGatherParams(spec, slice),
  }));
}

/**
 * Cut one slice's packed words out of a whole table, for a check or for an upload that stages the
 * table on the host. A slice starts on a row boundary, and a row is a whole number of words at
 * both bit widths, so this never straddles a word.
 */
export function sliceTableCodes(codes: Uint32Array, spec: EmbedGatherSpec, slice: TableSlice): Uint32Array {
  const stride = rowStrideWords(spec);
  const start = slice.startRow * stride;
  return codes.subarray(start, start + slice.rowCount * stride);
}

/**
 * `sliced` builds the form whose codes buffer is one vocabulary range of the table rather than the
 * whole of it. It changes nothing about the bindings, which is the point of the design in
 * ../tableSplit.ts: four storage buffers whatever the slice count is, because a dispatch binds one
 * slice. What it changes is the uniform, which gains the slice's vocabulary range, and the params
 * the caller has to supply: `sliceStartRow` and `sliceRowCount`, which the executor reads off the
 * plan (execute.ts `expandStep`) and the kernel case below states by hand.
 */
function makeBind(spec: EmbedGatherSpec, sliced = false) {
  const layoutKey = sliced ? `${spec.name}-sliced` : spec.name;
  return (input: KernelBindInput): KernelBindResult => {
    const { device, inputs, output, params } = input;
    const slots = (params.slots | 0) || 1;
    const codes = inputs.codes;
    const scales = inputs.scales;
    const ids = inputs.ids;
    if (!codes || !scales || !ids) {
      throw new Error(`${layoutKey} needs inputs named codes, scales and ids`);
    }

    const layout = kernelLayout(input, layoutKey, () => device.createBindGroupLayout({
      label: layoutKey,
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    }));

    // The sliced build's params come from the caller, because only the caller knows which slice this
    // dispatch is for. A missing range would be a dispatch that writes nothing and looks like a
    // dispatch that ran, so it is refused rather than defaulted.
    let block = embedGatherParams(spec);
    if (sliced) {
      const startRow = params.sliceStartRow;
      const rowCount = params.sliceRowCount;
      if (!Number.isFinite(startRow) || !Number.isFinite(rowCount) || (rowCount as number) <= 0) {
        throw new Error(`${layoutKey} needs params sliceStartRow and sliceRowCount naming a non empty vocabulary range`);
      }
      block = embedGatherParams(spec, {
        index: 0,
        startRow: startRow as number,
        rowCount: rowCount as number,
        byteOffset: 0,
        byteLength: 0,
      });
    }
    const uniform = kernelUniform(input, `${layoutKey} params`, block);

    return {
      layout,
      buffers: [codes, scales, ids, output, uniform.binding],
      dispatch: [Math.ceil(spec.cols / 4 / WORKGROUP_SIZE), slots, 1],
      dispose: uniform.dispose,
    };
  };
}

export const pleGatherKernel: Kernel = {
  name: PLE_GATHER_SPEC.name,
  wgsl: PLE_GATHER_WGSL,
  entry: 'main',
  // The table is 1.174 GB packed, the dominant term in the download
  // (EMBEDDING-QUANT-FINDING.md). The figure lives in this comment rather than in the note string
  // below because scripts/size-check.mjs holds src/engine/sizes.ts as the one size table and reads
  // a spoken size in live code as a number that escaped it. It is right to: a size a user ever
  // sees belongs in that table. This one is a fact about the checkpoint for whoever reads the
  // kernel, so it stays here where the rule exempts it rather than being deleted.
  note:
    'Per layer embedding gather. 35 groups of 256 for one token id out of the 4-bit table, '
    + 'dequantized in the shader at the point of use, never on the CPU and never once at upload.',
  cases: [
    {
      name: 'ple-two-ids',
      inputs: {
        codes: 'kembed.ple.codes',
        scales: 'kembed.ple.scales',
        ids: 'kembed.ple.ids',
      },
      expected: 'kembed.ple.expected',
      params: { slots: 2 },
      tolAbs: 0,
      tolUlp: 0,
      note:
        'Exact in f32: one correctly rounded multiply per element, and the fold of 16 into the '
        + 'scale is exact because 16 is a power of two. Anything softer would be hiding something.',
    },
  ],
  bind: makeBind(PLE_GATHER_SPEC),
};

/** The sliced build's name. One string, so the executor and the registry cannot disagree. */
export const PLE_GATHER_SLICED_NAME = `${PLE_GATHER_SPEC.name}-sliced`;

export const PLE_GATHER_SLICED_WGSL = embedGatherWgsl(PLE_GATHER_SPEC, { sliced: true });

/**
 * The PLE gather over one vocabulary range of a split table.
 *
 * WHY THIS IS A SECOND REGISTERED KERNEL rather than a parameter on the first. A kernel is one WGSL
 * module, and the sliced build's module is genuinely different: it carries two more words in its
 * uniform, a guard that returns for a token id outside its range, and a rebased row index. Folding
 * the two into one entry would mean either shipping the guard on every adapter, which moves the text
 * of the kernel that runs on this M1 and costs its bit identity claim its pinned digest, or
 * compiling a different module under one name, which is the thing the registry exists to prevent.
 *
 * It is dead code on every adapter this project has measured: the M1 grants a 4,294,967,292 byte
 * storage binding and the whole 1,174,405,120 byte table is one buffer there, so `expandStep`
 * resolves the ordinary build and this one never compiles. It exists for the specification floor of
 * 134,217,728, where the table is nine slices and one dispatch per slice is the only shape that fits
 * the binding budget. See ../tableSplit.ts.
 *
 * The case below is the equivalence the ship depends on: one slice covering the whole fixture table
 * has to reproduce, at zero tolerance, exactly what the unsliced build reproduces from the same
 * bytes. What it does not cover is a partial slice leaving another dispatch's slots alone, which is
 * proved value by value against the unsliced oracle in scripts/engine-check/transport.mjs, on a
 * table split three ways with a short last slice.
 */
export const pleGatherSlicedKernel: Kernel = {
  name: PLE_GATHER_SLICED_NAME,
  wgsl: PLE_GATHER_SLICED_WGSL,
  entry: 'main',
  note:
    'The per layer embedding gather over one vocabulary range of a split table. One dispatch per '
    + 'slice, four storage bindings whatever the slice count is, and an invocation whose token id '
    + 'falls outside this slice returns without writing so another dispatch can write it.',
  cases: [
    {
      name: 'ple-sliced-whole-vocabulary',
      inputs: {
        codes: 'kembed.ple.codes',
        scales: 'kembed.ple.scales',
        ids: 'kembed.ple.ids',
      },
      expected: 'kembed.ple.expected',
      // One slice covering all four rows of the fixture table, which is the plan a single slice
      // adapter produces and the case where the sliced build has to agree with the unsliced one to
      // the bit. The fixture's row count is stated here because the case has no other way to say it;
      // scripts/engine-check/k-norms.mjs builds the table at four rows.
      params: { slots: 2, sliceStartRow: 0, sliceRowCount: 4 },
      tolAbs: 0,
      tolUlp: 0,
      note:
        'The same arithmetic as ple-two-ids, reached through the sliced build with the slice set to '
        + 'the whole vocabulary. A difference here is a difference the row rebase introduced.',
    },
  ],
  bind: makeBind(PLE_GATHER_SPEC, true),
};

export const embedTokensGatherKernel: Kernel = {
  name: EMBED_TOKENS_GATHER_SPEC.name,
  wgsl: EMBED_TOKENS_GATHER_WGSL,
  entry: 'main',
  note:
    'Input embedding gather, 2-bit, one row of 1536 per token. Zero point and group layout are '
    + 'inferred rather than measured; see the file header.',
  cases: [
    {
      name: 'embed-two-ids',
      inputs: {
        codes: 'kembed.tok.codes',
        scales: 'kembed.tok.scales',
        ids: 'kembed.tok.ids',
      },
      expected: 'kembed.tok.expected',
      params: { slots: 2 },
      tolAbs: 0,
      tolUlp: 0,
      note: 'Exact in f32 for the same reason as the PLE case.',
    },
  ],
  bind: makeBind(EMBED_TOKENS_GATHER_SPEC),
};
