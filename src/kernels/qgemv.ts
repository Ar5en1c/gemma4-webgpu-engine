// Written from docs/ENGINE-PLAN.md sections 5 (kernels K4, K5 and K13), 5.5 and risks 2, 3 and 6,
// the geometry and cost ledgers in DECODE-CAMPAIGN.md 1, 4.2 and 5.1, PREFILL-CAMPAIGN.md's retile
// campaign result, DECODE-FUSION-FINDING.md, this engine's own quant.ts and qlayout.ts, and the
// WGSL specification. No vendored bundle, no extracted kernel and no third party engine source was
// read.
//
// SPDX-License-Identifier: Apache-2.0
//
// The QAT dequant GEMV family, which is the engine. K4 is the 4-bit variant: attention projections
// and producer layer MLP. K5 is the 2-bit variant: consumer layer MLP at intermediate width 12288
// and the 262,144 output logit head, which alone is about a sixth of everything the engine streams
// per token (ENGINE-PLAN risk 3).
//
// THE DESIGN TARGET, in one sentence: every packed weight byte is read exactly once, coalesced,
// and dequantized in registers at the point of use. On the M1 decode is streaming bound at a
// measured envelope near 51 GB/s, with roughly 0.6 GB moving per token, so the floor is about
// 11.8 ms per token and the whole gap between the incumbent's 26 ms and that floor is inside these
// kernels, not between them (DECODE-CAMPAIGN.md 1, DECODE-FUSION-FINDING.md). No weight is ever
// reread, no cleverness that trades bytes for dispatches.
//
// THE GEOMETRY, taken from the ledger rather than tuned from scratch: a 64 wide workgroup holding
// two 32 lane virtual subgroups, four output rows per workgroup, each row's K reduction confined
// to one 32 lane virtual subgroup. That shape measured best on the M1 for the gate and up family,
// and rows per workgroup above four collapsed occupancy; 32 wide with 2 rows measured worse
// because it caps resident warps (DECODE-CAMPAIGN.md 4.2). Start at the good shape; M5 retunes
// against the 51 GB/s envelope with the reduce order frozen.
//
// THE ACCESS PATTERN. Lane l of a virtual subgroup reads packed word l, l+32, l+64, ... of its
// row, so each iteration the 32 lanes touch 128 contiguous bytes of weights, which is the
// coalescing a streaming kernel needs as a hardware fact. The activation is read as vec4<f32> and
// shared by both rows of the virtual subgroup, so x is read once for two rows. Codes unpack in
// registers per qlayout.ts, whose CPU mirror is proved byte for byte against the loader's quant.ts
// unpackers by the k-matmul check on every run. Every dot runs through the dot() intrinsic,
// because values flowing through dot() never drifted across compiled modules in a full evening of
// probing while scalar multiply add chains drifted 1 to 2 ULP (DECODE-FUSION-FINDING.md,
// ENGINE-PLAN risk 2). The per row scale is applied once, after the reduction, matching quant.ts's
// dequantLinearRow contract of one f32 scale per output row.
//
// STATIC RANGE QUANTIZATION IS PART OF THIS KERNEL, both sides, since round 2. Every quantized
// linear in this checkpoint rounds its input onto an int8 grid with that module's own
// input_activation_scale and its output with output_activation_scale, always at 8 bits whatever
// the weight width, skipped only when the stored scale is exactly 0.0 (quant.ts applySrq, the one
// authority). The natural home is here, as a fused prologue and epilogue, because the input snap
// happens as the activation is read and the output snap as the row is written, so closing every
// SRQ site of plan.ts (round 1's list of eleven, SRQ_OPEN_SITES, now empty) adds no dispatch
// anywhere. On a calibrated site the kernel
// takes the ratified integer path: the input codes and the weight codes multiply in the integer
// domain, sum over K of xcode times wcode accumulated in i32 through WGSL's integer dot(), and
// one final multiply by input_scale times weight_scale lands the f32 answer. That accumulation is
// EXACT, not approximately right: the worst K reduction in the text stack is bounded near 6.3e6
// at 4-bit K 6144 and 3.2e6 at 2-bit K 12288, both far under 2^24, so the i32 sum has no rounding
// anywhere and is strictly more accurate than any f32 dot chain. An uncalibrated site, which in
// this checkpoint means only lm_head, keeps the f32 code dot exactly as round 1 shipped it.
//
// ACCUMULATORS ARE i32 OR f32, NEVER f16. f16 in this engine is a storage and cast dtype only
// (f16Cast.ts): LlamaWeb (arXiv 2605.20706) measured f16 accumulation producing incoherent output
// on Apple M-series GPUs, so no kernel in this family declares an f16 accumulator or sums into an
// f16 value, and the k-matmul lint greps for it.
//
// THE 2-BIT UNPACK is written for instruction count, not just coalescing: the retile campaign
// measured the 2-bit path instruction bound on M1, about 6 unpack, convert and quantize operations
// per 8 FLOP dot, landing at 15 to 20 percent of scalar ALU peak, and falsified the traffic
// hypothesis outright (PREFILL-CAMPAIGN.md, retile campaign result). The word unpack in qlayout.ts
// spends 3 scalar shifts, 4 vector shifts, 4 vector masks and 4 converts on 32 FLOPs of dot work.
// The banked next step if the logit head is still the M1 bottleneck is packed integer dot
// products, which is not bit identical and ships behind the ENGINE-PLAN section 7 fallback gate,
// not in this file.
//
// SUBGROUP POLICY, all seven rules of ENGINE-PLAN 5.5 designed in: reductions run in uniform
// control flow above the single guarded store block at the end; the tail reduction is the 32 lane
// butterfly from subgroupReduce.ts, never a bare subgroupAdd; there is no sgExact32 branch; the
// portable variant is built at pipeline creation from the device query plus the init self test
// verdict, never patched afterwards; loop bounds come from the uniform block, never from a large
// compile time constant.

import type { Kernel, KernelBindInput, KernelBindResult } from './registry';
import { kernelLayout, kernelUniform } from './binding';
import { SUBGROUP_BUTTERFLY_WGSL, SUBGROUP_ENABLE } from './subgroupReduce';
import {
  bytesAsWords,
  CODES_PER_WORD,
  unpackWordsLikeShader,
  WGSL_UNPACK2,
  WGSL_UNPACK4,
  WGSL_UNPACK4_I,
  WGSL_UNPACK8,
  WGSL_UNPACK8_I,
  wordsPerRow,
  tile16ToRowWords,
  tile16Words,
} from './qlayout';
import { applySrq, srqCodes } from '../quant';
import { PLE_GATE_L0_IN_SCALE, PLE_GATE_L0_OUT_SCALE } from './pleMatmul';
import { GELU_MUL_WGSL_FN, geluMulOracle } from './geluMul';

/** The three packed widths this family unpacks. 8 is the I8 PLE pair, one signed byte per code. */
export type GemvBits = 2 | 4 | 8;

/**
 * The round 2 decode geometry: 64 wide, 2 virtual subgroups, 4 rows (DECODE-CAMPAIGN.md 4.2). The
 * prefill GEMM in qgemm.ts still tiles its rows on these three numbers (its tile geometry was
 * falsified as a lever on M1 and is not swept); the decode GEMV's own geometry is per family below,
 * from the round 3 sweep.
 */
export const GEMV_WORKGROUP_SIZE = 64;
export const GEMV_ROWS_PER_WORKGROUP = 4;
export const GEMV_ROWS_PER_VSG = 2;

/**
 * The decode GEMV's geometry, as three numbers the round 3 performance campaign sweeps
 * (DECODE-CAMPAIGN.md 4.2 and 5, "the primary M1 lever"):
 *
 *   workgroupSize   lanes per workgroup, a multiple of 32, so workgroupSize / 32 virtual subgroups
 *   rowsPerVsg      output rows one 32 lane virtual subgroup owns, sharing its activation loads
 *   wordsPerLane    consecutive packed u32 words one lane reads per iteration: 1 reads a u32, 2 a
 *                   vec2<u32> and 4 a vec4<u32>, so the 32 lanes touch 128, 256 or 512 contiguous
 *                   bytes of one row per iteration (the vec4<u32> idiom of ENGINE-PLAN risk 3)
 *
 * What a geometry change may and may not move is the whole reason the sweep is allowed at all.
 * On the integer path, which is every calibrated site, the K reduction is an exact i32 sum of
 * integer products and is order independent by construction, so a geometry change cannot move a
 * value there and the kernel sweep gates at bit identity across every geometry. On the f32 path,
 * which in this checkpoint is only the uncalibrated logit head, the per lane partial sums round,
 * so the mapping of words to lanes is FROZEN at one word per lane whatever `wordsPerLane` says:
 * the f32 loop reads word `lane + 32 i` through the vector binding component by component. That
 * keeps the head's logits bit identical across the sweep, which is what lets the probe token
 * sequences stay a gate rather than a statistic during the campaign. Widening the head's own
 * loads is a separate lever with a separate gate.
 */
export interface GemvGeometry {
  workgroupSize: number;
  rowsPerVsg: number;
  wordsPerLane: 1 | 2 | 4;
  /**
   * The kernel's inner loop. The 4-bit and 8-bit families have one shape, 'classic': each word
   * unpacked into vec4<i32> codes and dotted against vec4<i32> activation codes, the field
   * absent or 'classic'. The 2-bit family reads the interleaved tile layout (qlayout.ts
   * TILE_ROWS, qgemv2TileWgsl below) and its field picks the loop there: 'tile16u', the shipped
   * one, unsigned codes in the pair multiply with the zero point taken off every row once at
   * the end; 'tile16', the signed form, one subtraction per pair inside the loop; 'tilefloor',
   * a diagnostic that is wrong by design, the same loads with the arithmetic dropped to one
   * multiply per word, the layout's streaming floor. The study that chose the layout and the
   * loop is docs/ENGINE-PERF.md section 13 (the performance page's `bench` set is how a
   * candidate is priced: interleaved, cycling every layer's weights, GPU mean per dispatch).
   */
  inner?: 'classic' | 'tile16u' | 'tile16' | 'tilefloor' | 'tile8u';
  /**
   * How many ways the K reduction is cut across workgroups, the flash decoding shape applied to a
   * GEMV. Absent or 1 is the shipped kernel to the byte, which the gate asserts.
   *
   * WHY THIS EXISTS, AND ONLY FOR THIS SHAPE. down_proj is 1536 rows by K 12288. At 16 rows a
   * workgroup that is 96 workgroups, and on a part with 48 SMs 96 workgroups is two apiece with
   * nothing left to hide a stall behind. Its siblings gate and up are the same bytes transposed,
   * 12288 rows by K 1536, which is 768 workgroups, and on the 5070 they run at 428 GB/s while
   * down_proj runs at 129 (lab-results/5070-per-kernel-decode-sep04.json). Same bytes, same
   * kernel, 3.3x apart, and the only thing that differs is how many workgroups the shape offers.
   *
   * Splitting K by S multiplies the workgroup count by S and costs one merge dispatch, which sums
   * S partials a row and applies the scales the split kernel deliberately does not. The partial
   * is stored RAW, before `xs * scales[row]` and before srqOut, because those are per row and
   * would otherwise be applied S times.
   *
   * This reassociates the K sum, so it is round 3 lead ruling 5's class of change: gated against
   * the oracle rather than argued, and off by default until it is priced on both machines. On the
   * M1, 96 workgroups over 8 cores is already twelve apiece, so the prediction there is a loss.
   */
  kSplits?: 1 | 2 | 4 | 8;
}

/**
 * The round 2 geometry both families shipped with, kept as the sweep's reference point and as the
 * geometry the kernel fixtures were first proved at.
 */
export const DEFAULT_GEMV_GEOMETRY: Readonly<GemvGeometry> = Object.freeze({
  workgroupSize: GEMV_WORKGROUP_SIZE,
  rowsPerVsg: GEMV_ROWS_PER_VSG,
  wordsPerLane: 1,
});

/**
 * The per family geometry the round 3 sweep chose, and the sweep it came from, measured on the M1
 * with per dispatch timestamps over 63 decode tokens of interview-300 (docs/ENGINE-PERF.md). GPU
 * milliseconds per token for the 2-bit family (391 MB per token) and the 4-bit family (357 MB):
 *
 *   WG,ROWS,WORDS    2-bit    4-bit
 *   64,2,1 (round 2) 16.5     9.4
 *   64,2,2           17.8     8.0
 *   64,2,4           20.3    10.1      the vec4<u32> idiom loses on this GPU in this shape
 *   64,4,1           13.0     8.2
 *   64,4,2           15.4     8.1
 *   64,8,1           11.9     8.7
 *   128,8,1          11.7     8.5
 *   32,8,1           12.1     8.9
 *
 * The two families want different shapes: the 2-bit rows are shorter in bytes per K, so sharing
 * one activation load across more rows is what fills the memory pipe (8 rows per virtual
 * subgroup, 33 GB/s against 24), while the 4-bit family is near the envelope at 4 rows (44 GB/s)
 * and loses again past that. Wider per lane loads never won for the 2-bit family and were a wash
 * for the 4-bit one, so both stay at one word per lane. Neither change moves a value: the integer
 * path is order independent and the f32 head keeps its one word per lane order regardless.
 */
export const GEMV_GEOMETRY_2BIT: Readonly<GemvGeometry> = Object.freeze({ workgroupSize: 32, rowsPerVsg: 16, wordsPerLane: 1, inner: 'tile16u' });
export const GEMV_GEOMETRY_4BIT: Readonly<GemvGeometry> = Object.freeze({ workgroupSize: 64, rowsPerVsg: 4, wordsPerLane: 1 });

/**
 * The 8-bit family's decode geometry (round 4 lever 3). The two PLE linears are small, 256 rows
 * of K 1536 and 1536 rows of K 256, so the question is parallelism rather than bytes per row:
 * see docs/ENGINE-PERF.md round 4 for the sweep that chose this.
 */
export const GEMV_GEOMETRY_8BIT: Readonly<GemvGeometry> = Object.freeze({ workgroupSize: 64, rowsPerVsg: 2, wordsPerLane: 1 });

const SHIPPED_GEOMETRY: Record<GemvBits, Readonly<GemvGeometry>> = {
  2: GEMV_GEOMETRY_2BIT,
  4: GEMV_GEOMETRY_4BIT,
  8: GEMV_GEOMETRY_8BIT,
};

const activeGeometry: Record<GemvBits, Readonly<GemvGeometry>> = { ...SHIPPED_GEOMETRY };

/**
 * Rows per virtual subgroup are capped at 16. Round 3 capped them at 8, which was where its sweep
 * stopped; round 4 lever 4 sweeps 12 and 16 for the 2-bit family, whose rows are short enough in
 * bytes that sharing one activation load across more of them is what fills the pipe. Each row is
 * one i32 accumulator and one packed word live per lane per iteration, so 16 is well inside the
 * register file at 64 lanes.
 */
export const GEMV_MAX_ROWS_PER_VSG = 16;

function assertGeometry(g: GemvGeometry): void {
  if (!Number.isInteger(g.workgroupSize) || g.workgroupSize % 32 !== 0 || g.workgroupSize < 32 || g.workgroupSize > 256) {
    throw new Error(`gemv geometry: workgroupSize must be a multiple of 32 in 32..256, got ${g.workgroupSize}`);
  }
  if (!Number.isInteger(g.rowsPerVsg) || g.rowsPerVsg < 1 || g.rowsPerVsg > GEMV_MAX_ROWS_PER_VSG) {
    throw new Error(`gemv geometry: rowsPerVsg must be 1..${GEMV_MAX_ROWS_PER_VSG}, got ${g.rowsPerVsg}`);
  }
  if (g.wordsPerLane !== 1 && g.wordsPerLane !== 2 && g.wordsPerLane !== 4) {
    throw new Error(`gemv geometry: wordsPerLane must be 1, 2 or 4, got ${g.wordsPerLane}`);
  }
  if (g.inner !== undefined && !['classic', 'tile16u', 'tile16', 'tilefloor', 'tile8u'].includes(g.inner)) {
    throw new Error(`gemv geometry: inner must be classic, tile16u, tile16, tilefloor or tile8u, got ${String(g.inner)}`);
  }
  const ks = (g.kSplits ?? 1) as number;
  if (ks !== 1 && ks !== 2 && ks !== 4 && ks !== 8) {
    throw new Error(`gemv geometry: kSplits must be 1, 2, 4 or 8, got ${String(ks)}`);
  }
}

/**
 * The largest split the geometry allows. The partial scratch slot is sized at this rather than at
 * whatever a profile asks for, so the slot does not change size with the device and a plan taken
 * on one machine can be replayed on another.
 */
export const GEMV_MAX_K_SPLITS = 8;

/** Splits this geometry asks for, with the default that means "the shipped kernel". */
export function kSplitsOf(geometry: Readonly<GemvGeometry>): number {
  return geometry.kSplits ?? 1;
}

/**
 * The same geometry with the K split removed. Every 2-bit site that the plan does NOT fold builds
 * through this: lm_head, the projections, and the unsplit down_proj entry. Before it existed the
 * split was read straight off the geometry singleton by every 2-bit build, so turning the split on
 * for down_proj silently split lm_head too, with no fold after it. A split dispatch is only correct
 * when a merge follows, and only the plan knows that, so the split is carried by kernel identity
 * (the `-split` entries) and never by the shared name.
 */
export function unsplitGemvGeometry(geometry: Readonly<GemvGeometry>): Readonly<GemvGeometry> {
  return kSplitsOf(geometry) === 1 ? geometry : Object.freeze({ ...geometry, kSplits: 1 });
}

/**
 * Override the decode GEMV geometry of one family, or of both when `bits` is omitted, for every
 * pipeline built afterwards. A dev seam for the performance rig's sweep: it must be called before
 * any pipeline is built, because the pipeline store caches modules by their text and the registry
 * entry's `wgsl` is read at build time. Passing null restores the family's shipped geometry.
 * Nothing in the shipping engine calls this.
 */
export function setGemvGeometry(geometry: GemvGeometry | null, bits?: GemvBits): void {
  const families: GemvBits[] = bits === undefined ? [2, 4, 8] : [bits];
  if (geometry !== null) assertGeometry(geometry);
  for (const family of families) {
    activeGeometry[family] = geometry === null ? SHIPPED_GEOMETRY[family] : Object.freeze({ ...geometry });
  }
}

export function gemvGeometry(bits: GemvBits): Readonly<GemvGeometry> {
  return activeGeometry[bits];
}

export function gemvRowsPerWorkgroup(geometry: Readonly<GemvGeometry>): number {
  return (geometry.workgroupSize / 32) * geometry.rowsPerVsg;
}

/** Which compiled reduction variant a pipeline was built with. Chosen once, at build. */
export type MatmulReduceVariant = 'subgroup' | 'workgroup';

/**
 * The reduction prelude both matmul families splice in. `mmSum(v, lid)` returns, in every lane of
 * a 32 lane virtual subgroup, the sum of `v` across that group.
 *
 * The butterfly variant delegates to sgSum32 from subgroupReduce.ts, so the engine keeps exactly
 * one subgroup reduction shape (ENGINE-PLAN 5.5 rule 3: one shape, correct everywhere). The
 * workgroup variant is for a device that fails the init self test; it reduces through workgroup
 * memory inside each aligned 32 lane half, because unlike the norm kernels a matmul workgroup
 * holds two virtual subgroups working on different rows, so a full width tree would sum across
 * rows. Its barriers are uniform because the function is only ever called unconditionally from
 * uniform control flow, which is also what Tint requires (ENGINE-PLAN 5.5 rule 5).
 */
export function matmulReducePrelude(variant: MatmulReduceVariant, workgroupSize: number = GEMV_WORKGROUP_SIZE): string {
  if (variant === 'subgroup') {
    return `${SUBGROUP_ENABLE}\n${SUBGROUP_BUTTERFLY_WGSL}
fn mmSum(v: f32, lid: u32) -> f32 {
  return sgSum32(v);
}
`;
  }
  return /* wgsl */ `
var<workgroup> mmScratch: array<f32, ${workgroupSize}>;

fn mmSum(v: f32, lid: u32) -> f32 {
  // The opening barrier orders this call's stores after every lane's reads from the previous
  // call. Uniform, because mmSum is only ever called unconditionally.
  workgroupBarrier();
  mmScratch[lid] = v;
  workgroupBarrier();
  for (var s: u32 = 16u; s > 0u; s = s >> 1u) {
    if ((lid & 31u) < s) {
      mmScratch[lid] = mmScratch[lid] + mmScratch[lid + s];
    }
    workgroupBarrier();
  }
  return mmScratch[lid & 0xffffffe0u];
}
`;
}

/**
 * Build the GEMV WGSL for one bit width, one reduction variant and one geometry.
 *
 * Row mapping: workgroup `g` (folded over a 2D grid because the logit head needs
 * ceil(262144 / rowsPerWorkgroup) workgroups against a per dimension limit of 65535) owns rows
 * `g * rowsPerWorkgroup ..`; virtual subgroup `v` owns `rowsPerVsg` consecutive rows inside that.
 * Rows past numRows read clamped, reduce garbage in uniform control flow and are dropped by the
 * store guard, which is cheaper than any masking in the hot loop.
 *
 * K loop. Each lane reads `wordsPerLane` consecutive words per iteration, so one iteration of the
 * 32 lanes covers 32 * wordsPerLane words of a row. K must fill whole words and whole lane
 * vectors (K a multiple of codesPerWord * wordsPerLane, which every text tensor's K satisfies at
 * every legal geometry, and `bind` asserts), but it need not fill whole iterations: the loop runs
 * the whole iterations the uniform block counts and one guarded tail handles the rest, so K 1536
 * at four words per lane runs one full iteration and a half one. The tail guard is per lane and
 * touches only loads and accumulation, so the reductions below it stay in uniform control flow.
 *
 * The f32 path keeps the one word per lane mapping whatever the geometry, for the reason given on
 * `GemvGeometry`: its rounding depends on the order, and the logit head must not move under a
 * sweep whose gate is the probe token sequences.
 */
/**
 * The fused activation prologue. 'gelu' binds `gate` and `up` in place of `x` and reads every
 * activation vector as `geluTanh(gate) * up`, the gelu-mul kernel's own expression (geluMul.ts
 * GELU_MUL_WGSL_FN), so the K10 pass and its round trip through the `act` slot disappear from
 * the decode token (docs/ENGINE-PERF.md section 15, the dispatch ledger). On the integer path
 * the snap is this kernel's srqIn over that product, which is the same int8 code the gelu-mul
 * pass produced and this kernel then re-snapped: round(v / s) here against round((q s) / s)
 * there with q = round(v / s), and the second form returns q because q s / s sits within a few
 * ULP of an integer under 128. So the fused prologue is bit for bit the unfused chain on every
 * calibrated site, and the full model page is the receipt. The f32 path reads the same product
 * unsnapped, as the uncalibrated gelu-mul pass would have stored it.
 */
export type GemvPrologue = 'none' | 'gelu';

/** Registry names of the fused prologue GEMVs by family, the plan's kernel names for the decode MLP tail. */
export const GEMV_GELU_KERNEL: Readonly<Record<GemvBits, string>> = Object.freeze({
  2: 'qgemv-2bit-gelu', 4: 'qgemv-4bit-gelu', 8: 'qgemv-8bit-gelu',
});

/**
 * The split K build of the 2-bit gelu-fused down_proj. A separate kernel, not a mode of the one
 * above, because the split is only correct when the plan also emits the fold (ROLE_DOWN_MERGE):
 * naming it makes a split dispatch without a merge unrepresentable rather than merely unintended.
 */
export const GEMV_GELU_SPLIT_KERNEL_2BIT = 'qgemv-2bit-gelu-split';

export function prologueWgsl(prologue: GemvPrologue): string {
  if (prologue === 'none') {
    return `@group(0) @binding(2) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;
@group(0) @binding(4) var<uniform> params: GemvParams;`;
  }
  return `@group(0) @binding(2) var<storage, read> gate: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> up: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> dst: array<f32>;
@group(0) @binding(5) var<uniform> params: GemvParams;
${GELU_MUL_WGSL_FN}
// The fused K10 prologue: one activation vector, gelu of the gate times the up projection.
fn act(i: u32) -> vec4<f32> {
  return geluTanh(gate[i]) * up[i + params.upOffset];
}`;
}

/**
 * The split K build of the classic 4-bit and 8-bit loop, its own kernel names as the 2-bit tile's
 * split is (GEMV_GELU_SPLIT_KERNEL_2BIT): the plan names them only where it also names the fold,
 * which for these families lives in the CONSUMER (kernels/attnPrologue.ts's fold variants) rather
 * than in a merge dispatch. Why: on the RTX 5070 a small GEMV's time is its walk, six dependent
 * iterations at K 1536 whatever the workgroup count, and the same rows at half the K run 0.6x
 * and at a quarter 0.45x (src/dev/gemvsweep.html?only=walk). The 2-bit split's merge dispatch
 * would give a third of that back; a fold in the consumer gives none of it back.
 */
export const GEMV_SPLIT_KERNEL: Readonly<Record<4 | 8, string>> = Object.freeze({ 4: 'qgemv-4bit-split', 8: 'qgemv-8bit-split' });

export function qgemvWgsl(
  bits: GemvBits,
  variant: MatmulReduceVariant,
  geometry: Readonly<GemvGeometry> = activeGeometry[bits],
  prologue: GemvPrologue = 'none',
  split = false,
  exactPartials = false,
  calibratedF32 = false,
): string {
  assertGeometry(geometry as GemvGeometry);
  if (calibratedF32 && (
    bits !== 4
    || split
    || exactPartials
    || (geometry.inner !== undefined && geometry.inner !== 'classic')
  )) {
    throw new Error('calibrated f32 requires classic unsplit 4-bit GEMV without exact partials');
  }
  if (exactPartials && (bits === 2 || !split)) throw new Error('unscaled partials require a 4-bit or 8-bit split');
  if (exactPartials && geometry.inner !== undefined && geometry.inner !== 'classic') {
    throw new Error('unscaled partials require the classic packed layout');
  }
  // The activation read, through the prologue when there is one.
  const xr = (expr: string): string => (prologue === 'gelu' ? `act(${expr})` : `x[${expr}]`);
  if (bits === 2) return qgemv2TileWgsl(variant, geometry, prologue);
  // The 4-bit tile loop, on the eight row interleaved layout. Priced on the sweep page before
  // any tensor is repacked to it; the shipped 4-bit loop is the classic one below until then.
  if (bits === 4 && geometry.inner === 'tile8u') return qgemv4TileWgsl(variant, geometry, prologue);
  if (geometry.inner !== undefined && geometry.inner !== 'classic') {
    throw new Error(`qgemv-${bits}bit has only the classic inner loop, got ${geometry.inner}`);
  }
  if (split && prologue !== 'none' && !exactPartials) throw new Error(`qgemv-${bits}bit: the K split has no prologue form`);
  // The split is a property of the kernel's name, never of the shared geometry: the plain build
  // reads KS 1 whatever a profile set, exactly as qgemv-2bit does beside its split sibling.
  const KS = split ? kSplitsOf(geometry) : 1;
  const W = geometry.workgroupSize;
  const R = geometry.rowsPerVsg;
  const V = geometry.wordsPerLane;
  const rowsPerWg = gemvRowsPerWorkgroup(geometry);
  // The 8-bit unpack is qlayout.ts's sign extending shift pair, the same text pleMatmul.ts
  // splices, so the GEMV and the prefill matmul of the I8 family read a word identically.
  // The 2-bit family returned above, on its own kernel; this generator serves 4 and 8 bits.
  const unpack = bits === 4 ? WGSL_UNPACK4 : WGSL_UNPACK8;
  const unpackI = bits === 4 ? WGSL_UNPACK4_I : WGSL_UNPACK8_I;
  const vec4PerWord = CODES_PER_WORD[bits] / 4;
  const wordType = V === 1 ? 'u32' : `vec${V}<u32>`;
  const wordNames = bits === 4 ? ['wLo', 'wHi'] : ['w8'];
  const wordNamesI = bits === 4 ? ['wLoI', 'wHiI'] : ['w8i'];
  const indent = (text: string, spaces: number): string => text
    .split('\n')
    .map((line) => (line.trim().length === 0 ? line : ' '.repeat(spaces) + line.trimStart()))
    .join('\n');

  // The integer path body for one lane vector at vector index `wi`: the activation codes for the
  // V * vec4PerWord vec4s the vector covers, loaded once and shared across the R rows, then one
  // unpack and one integer dot per (row, word).
  const intBody = (domain: 'i32' | 'f32' = 'i32'): string => {
    const f32 = domain === 'f32';
    const domainWordNames = f32 ? wordNames : wordNamesI;
    const domainUnpack = f32 ? unpack : unpackI;
    const lines: string[] = [];
    lines.push(`      let xb0 = wi * ${V * vec4PerWord}u;`);
    for (let c = 0; c < V; c += 1) {
      for (let j = 0; j < vec4PerWord; j += 1) {
        lines.push(`      let x_${c}_${j} = ${f32 ? 'srqInF' : 'srqIn'}(${xr(`xb0 + ${c * vec4PerWord + j}u`)});`);
      }
    }
    for (let r = 0; r < R; r += 1) {
      lines.push(`      let wv${r} = wq[base${r} + wi];`);
      for (let c = 0; c < V; c += 1) {
        const word = V === 1 ? `wv${r}` : `wv${r}[${c}]`;
        const dots = domainWordNames.map((n, j) => `dot(${n}, x_${c}_${j})`).join(' + ');
        lines.push('      {');
        lines.push(`        let w = ${word};`);
        lines.push(indent(domainUnpack, 8));
        lines.push(`        acc${r}${f32 ? '' : 'i'} = acc${r}${f32 ? '' : 'i'} + ${dots};`);
        lines.push('      }');
      }
    }
    return lines.join('\n');
  };

  // The f32 path body for one word at word index `w1`, read through the vector binding one
  // component at a time so the lane to word mapping is the one word per lane order.
  const floatBody = (): string => {
    const lines: string[] = [];
    lines.push(`      let xb0 = w1 * ${vec4PerWord}u;`);
    for (let j = 0; j < vec4PerWord; j += 1) lines.push(`      let x_${j} = ${xr(`xb0 + ${j}u`)};`);
    for (let r = 0; r < R; r += 1) {
      const word = V === 1 ? `wq[base${r} + w1]` : `wq[base${r} + (w1 / ${V}u)][w1 % ${V}u]`;
      const dots = wordNames.map((n, j) => `dot(${n}, x_${j})`).join(' + ');
      lines.push('      {');
      lines.push(`        let w = ${word};`);
      lines.push(indent(unpack, 8));
      lines.push(`        acc${r} = acc${r} + ${dots};`);
      lines.push('      }');
    }
    return lines.join('\n');
  };

  const rowDecls = Array.from({ length: R }, (_, r) => [
    `  let row${r} = rowBase + ${r}u;`,
    `  let base${r} = min(row${r}, rLast) * kVec;`,
  ].join('\n')).join('\n');
  const accDecls = Array.from({ length: R }, (_, r) => `  var acc${r} = 0.0;`).join('\n');
  const accIDecls = Array.from({ length: R }, (_, r) => `    var acc${r}i: i32 = 0;`).join('\n');
  const accHandoff = Array.from({ length: R }, (_, r) => `    acc${r} = f32(acc${r}i);`).join('\n');
  const sums = Array.from({ length: R }, (_, r) => `  let sum${r} = mmSum(acc${r}, lid);`).join('\n');
  const stores = Array.from({ length: R }, (_, r) => [
    `    if (row${r} < params.numRows) {`,
    KS === 1
      ? `      dst[row${r}] = srqOut((xs * scales[row${r}]) * sum${r});`
      // The split cut is the LAST linear step, as on the 2-bit tile: each partial carries its
      // row scale and the fold owns the snap.
      : `      dst[row${r} * ${KS}u + split] = ${exactPartials ? `sum${r}` : `(xs * scales[row${r}]) * sum${r}`};`,
    '    }',
  ].join('\n')).join('\n');
  // The band of K this workgroup walks under a split. Iteration slots are the kIters whole
  // iterations plus the guarded tail, which belongs to the band that owns slot kIters; wid.z is
  // the split and num_workgroups.z the count, so the cut needs no room in the params block.
  const band = KS === 1 ? '' : `  let split = wid.z;
  let slots = params.kIters + ${exactPartials ? `select(0u, 1u, params.kWords % ${32 * V}u != 0u)` : '1u'};
  let per = (slots + ${KS - 1}u) / ${KS}u;
  let iLo = min(split * per, slots);
  let iHi = min(iLo + per, slots);
`;
  const intFrom = KS === 1 ? '0u' : 'iLo';
  const intTo = KS === 1 ? 'params.kIters' : 'min(iHi, params.kIters)';
  const intStart = KS === 1 ? 'lane' : 'lane + iLo * 32u';
  const tailGuard = KS === 1 ? 'wi < kVec' : 'iHi > params.kIters && wi < kVec';
  const floatBand = KS === 1 ? '' : `    let oPer = (kOnes + ${KS - 1}u) / ${KS}u;
    let oLo = min(split * oPer, kOnes);
    let oHi = min(oLo + oPer, kOnes);
`;
  const floatFrom = KS === 1 ? '0u' : 'oLo';
  const floatTo = KS === 1 ? 'kOnes' : 'oHi';
  const floatStart = KS === 1 ? 'lane' : 'lane + oLo * 32u';
  const calibratedBody = calibratedF32 ? `    // At K <= 12288, |-8 * -128| = 1024 and the full sum is at most
    // 12,582,912, below 2^24, so f32 carries every integer partial and total exactly.
    if (params.kWords <= 1536u) {
      var wi = ${intStart};
      for (var i = ${intFrom}; i < ${intTo}; i = i + 1u) {
${intBody('f32')}
        wi = wi + 32u;
      }
      if (${tailGuard}) {
${intBody('f32')}
      }
    } else {
${accIDecls}
      var wi = ${intStart};
      for (var i = ${intFrom}; i < ${intTo}; i = i + 1u) {
${intBody()}
        wi = wi + 32u;
      }
      if (${tailGuard}) {
${intBody()}
      }
${accHandoff}
    }` : `${accIDecls}
    var wi = ${intStart};
    for (var i = ${intFrom}; i < ${intTo}; i = i + 1u) {
${intBody()}
      wi = wi + 32u;
    }
    // The tail: the lanes whose vector still lies inside the row. Loads and adds only, so the
    // reductions after the branch are still reached by every lane.
    if (${tailGuard}) {
${intBody()}
    }
${accHandoff}`;
  const srqInF = calibratedF32 ? `
// The same SRQ snap as srqIn, retaining the integer codes in f32 for the exact bounded path.
fn srqInF(v: vec4<f32>) -> vec4<f32> {
  return clamp(round(v / params.inScale), vec4<f32>(SRQ_MIN), vec4<f32>(SRQ_MAX));
}
` : '';

  const code = /* wgsl */ `${matmulReducePrelude(variant, W)}
struct GemvParams {
  // Packed u32 words per weight row, so K / ${CODES_PER_WORD[bits]} for this ${bits}-bit family.
  kWords: u32,
  // Whole iterations of the streaming loop at ${V} word(s) per lane: kWords / ${32 * V}, rounded
  // down; the remainder is one guarded tail. Runtime opaque, per registry rule 2
  // (DECODE-FUSION-FINDING.md item 5, where a constant bound unrolled into a megakernel).
  kIters: u32,
  numRows: u32,
  pad0: u32,
  // The consuming module's input_activation_scale. Non zero selects the integer path: the input
  // snaps to int8 codes and the K reduction is the exact integer code dot. Exactly 0.0 means the
  // site is uncalibrated (lm_head) and the f32 code dot runs instead (quant.ts applySrq).
  inScale: f32,
  // The module's output_activation_scale, applied to each written row. 0.0 means no rounding.
  outScale: f32,
  upOffset: u32,
  pad2: u32,
}

@group(0) @binding(0) var<storage, read> wq: array<${wordType}>;
@group(0) @binding(1) var<storage, read> scales: array<f32>;
${prologueWgsl(prologue)}

// int8, at every static range quantization site in this checkpoint whatever the weight width is
// (quant.ts SRQ_ACTIVATION_BITS).
const SRQ_MAX: f32 = 127.0;
const SRQ_MIN: f32 = -128.0;

// The SRQ input prologue: snap four activations onto the int8 grid and keep the CODES, which is
// what the integer dot consumes. round() is ties to even, the rule torch.round calibrated this
// checkpoint with (quant.ts srqCodes and roundTiesToEven). Clamping before the convert is what
// makes the i32 cast exact rather than implementation defined on an out of range value.
fn srqIn(v: vec4<f32>) -> vec4<i32> {
  return vec4<i32>(clamp(round(v / params.inScale), vec4<f32>(SRQ_MIN), vec4<f32>(SRQ_MAX)));
}
${srqInF}
// The SRQ output epilogue, the module's output_activation_scale applied to the row it just
// computed. A zero scale returns the value untouched, exactly as the reference treats an
// uncalibrated site. The branch is uniform: the scale comes from the uniform block.
fn srqOut(v: f32) -> f32 {
  if (params.outScale == 0.0) {
    return v;
  }
  return clamp(round(v / params.outScale), SRQ_MIN, SRQ_MAX) * params.outScale;
}

@compute @workgroup_size(${W})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
  @builtin(local_invocation_index) lid: u32,
) {
  let lane = lid & 31u;
  let vsg = lid >> 5u;
  // The 2D fold. wid.y is zero except when the row count needs more workgroups than one grid
  // dimension can carry, which in this engine means the logit head (ENGINE-PLAN risk 3).
  let group = wid.x + nwg.x * wid.y;
  let rowBase = group * ${rowsPerWg}u + vsg * ${R}u;
  // Clamped read bases keep out of range rows harmless without a branch in the hot loop. The
  // store guard at the end is what actually drops them. Bases are in lane vectors of ${V} word(s).
  let rLast = params.numRows - 1u;
  let kVec = params.kWords / ${V}u;
${rowDecls}
${band}
  // The two K loops land in these. f32 and i32 accumulators only, never f16: LlamaWeb
  // (arXiv 2605.20706) measured f16 accumulation producing incoherent output on Apple M-series.
${accDecls}
  // Uniform branch: inScale comes from the uniform block, so every lane in the dispatch takes
  // the same side and the reductions below stay in uniform control flow.
  if (params.inScale != 0.0) {
    // The ratified integer path. Codes times codes through the integer dot(), accumulated in
    // i32 with no rounding anywhere: the worst text stack reduction is bounded near 6.3e6, far
    // inside i32 and inside f32's exact integer range, so the f32 handoff below is exact too.
${calibratedBody}
  } else {
    // The uncalibrated path, the round 1 f32 code dot, byte for byte in its accumulation order:
    // one word per lane, word lane + 32 i, whatever the vector width of the binding.
    let kOnes = params.kWords / 32u;
${floatBand}    var w1 = ${floatStart};
    for (var i = ${floatFrom}; i < ${floatTo}; i = i + 1u) {
${floatBody()}
      w1 = w1 + 32u;
    }
  }

  // Both reductions run before any store, in uniform control flow, then one guarded store block
  // ends the kernel. ENGINE-PLAN 5.5 rule 1, which is the structural fix for the NVIDIA
  // miscompile (PREFILL-CAMPAIGN.md round 3).
${sums}
  if (lane == 0u) {
    // One final scale multiply: input_scale times weight_scale on the integer path, and exactly
    // the round 1 expression on the f32 path, because 1.0 times the scale is the scale.
    let xs = select(1.0, params.inScale, params.inScale != 0.0);
${stores}
  }
}
`;
  return code;
}

/**
 * The 2-bit family's kernel, on the interleaved layout of qlayout.ts (TILE_ROWS rows per word,
 * row r of the tile at bits 2 r, one word per k). One 32 lane virtual subgroup owns one tile of
 * sixteen rows and the lanes deal K between them: lane `lane` takes k = 4 lane + 128 i + j for
 * j in 0..3 through one vec4<u32> load and one vec4 activation load per iteration, so a word
 * meets one scalar activation and the sixteen rows share it.
 *
 * The integer path (a calibrated site, inScale non zero) keeps the ratified contract, the exact
 * integer code dot, and pays for it with a fifth of the classic loop's arithmetic: pair m of
 * eight, `(w >> 2m) & 0x00030003`, is the unsigned codes of rows m and m + 8 in the two 16-bit
 * halves of one u32, and one i32 multiply by the activation code forms both products at once.
 * A product is at most 3 * 128 in magnitude and a block of sixteen iterations adds 64 of them
 * per half, 24576 at most, inside signed 16 bits for the low half and inside i32 for the high
 * one; the split after each block sign extends the low half and takes it off before shifting
 * the high one down, exact. The zero point comes off once per row at the end, sum((u - 2) x)
 * being sum(u x) - 2 sum(x) with the lane's own activation code sum, so every lane partial is
 * the integer the row layout's loop produced and the reduction sees the same values. On the
 * shipped 'tile16u' form the multiply is unsigned; 'tile16' subtracts the zero point inside
 * the loop instead (the same integers, one more operation per pair); 'tilefloor' is the
 * diagnostic floor and is wrong by design (GemvGeometry.inner).
 *
 * The f32 path (the logit head, both scales 0.0) reads the same words against the unsnapped
 * activation and accumulates the sixteen rows' products in f32 in lane order, k ascending, with
 * the zero point folded out the same way. Its accumulation order is this kernel's own and is
 * different from the row layout's, so the head's logits moved at the rounding level when the
 * layout landed; the head is held by the reference anchored gate (ENGINE-PERF.md section 12),
 * not by bit identity with an earlier build, and the kernel sweep's head cases carry a
 * tolerance argued in scripts/engine-check/k-matmul.mjs.
 *
 * Reductions run before any store in uniform control flow (ENGINE-PLAN 5.5 rule 1); the loop
 * bounds and the block structure are uniform, from the params block; a tile past the row count
 * clamps its read and its rows fail the store guard. K must be a multiple of 128 (bind asserts
 * it; every 2-bit tensor's is), and one dispatch covers the head's 262,144 rows through the 2D
 * grid fold at 16 rows per workgroup.
 */
function qgemv2TileWgsl(variant: MatmulReduceVariant, geometry: Readonly<GemvGeometry>, prologue: GemvPrologue = 'none'): string {
  const xr = (expr: string): string => (prologue === 'gelu' ? `act(${expr})` : `x[${expr}]`);
  const W = geometry.workgroupSize;
  const tilesPerWg = W / 32;
  const inner = geometry.inner ?? 'tile16u';
  if (geometry.rowsPerVsg !== 16) {
    throw new Error(`qgemv-2bit reads sixteen row tiles, so rowsPerVsg must be 16, got ${geometry.rowsPerVsg}`);
  }
  if (inner === 'classic') throw new Error('qgemv-2bit has no classic loop on the interleaved layout');
  const KS = kSplitsOf(geometry);
  const rows = Array.from({ length: 16 }, (_, r) => r);
  const pairs = Array.from({ length: 8 }, (_, m) => m);
  const comps = ['x', 'y', 'z', 'w'];
  const shifted = (w: string, m: number): string => (m === 0 ? w : `(${w} >> ${2 * m}u)`);
  const pairOps = (w: string, xj: string): string => pairs
    .map((m) => `          p${m} = p${m} + ${inner === 'tile16'
      ? `(i32(${shifted(w, m)} & 0x00030003u) - 0x00020002)`
      : `i32(${shifted(w, m)} & 0x00030003u)`} * ${xj};`)
    .join('\n');
  const intBody = inner === 'tilefloor'
    ? comps.map((c, j) => `        p${j} = p${j} + i32(v.${c}) * xk.${c};`).join('\n')
    : comps.map((c) => `        {\n          let w = v.${c};\n${pairOps('w', `xk.${c}`)}\n        }`).join('\n');
  const flush = pairs
    .map((m) => `      { let lo = (p${m} << 16u) >> 16u; acc${m}i = acc${m}i + lo; acc${m + 8}i = acc${m + 8}i + ((p${m} - lo) >> 16u); p${m} = 0; }`)
    .join('\n');
  const headBody = comps.map((c) => [
    '      {',
    `        let w = v.${c};`,
    `        let xj = xk.${c};`,
    '        sxf = sxf + xj;',
    ...rows.map((r) => `        acc${r} = acc${r} + f32(${shifted('w', r)} & 3u) * xj;`),
    '      }',
  ].join('\n')).join('\n');

  return /* wgsl */ `${matmulReducePrelude(variant, W)}
struct GemvParams {
  // Packed u32 words per weight row, K / 16. The tile layout holds 16 kWords words per tile.
  kWords: u32,
  // Unused on this kernel, which derives its iteration count from kWords; kept so the params
  // block is the one gemvParams writes for every family.
  kIters: u32,
  numRows: u32,
  pad0: u32,
  // The consuming module's input_activation_scale. Non zero selects the integer path: the input
  // snaps to int8 codes and the K reduction is the exact integer code dot. Exactly 0.0 means the
  // site is uncalibrated (lm_head) and the f32 path runs instead (quant.ts applySrq).
  inScale: f32,
  // The module's output_activation_scale, applied to each written row. 0.0 means no rounding.
  outScale: f32,
  upOffset: u32,
  pad2: u32,
}

@group(0) @binding(0) var<storage, read> wq: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> scales: array<f32>;
${prologueWgsl(prologue)}

const SRQ_MAX: f32 = 127.0;
const SRQ_MIN: f32 = -128.0;

// The SRQ input prologue: snap four activations onto the int8 grid and keep the codes. round()
// is ties to even, the rule torch.round calibrated this checkpoint with (quant.ts srqCodes).
fn srqIn(v: vec4<f32>) -> vec4<i32> {
  return vec4<i32>(clamp(round(v / params.inScale), vec4<f32>(SRQ_MIN), vec4<f32>(SRQ_MAX)));
}

fn srqOut(v: f32) -> f32 {
  if (params.outScale == 0.0) {
    return v;
  }
  return clamp(round(v / params.outScale), SRQ_MIN, SRQ_MAX) * params.outScale;
}

@compute @workgroup_size(${W})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
  @builtin(local_invocation_index) lid: u32,
) {
  let lane = lid & 31u;
  let vsg = lid >> 5u;
  // The 2D fold, for the head's row count (ENGINE-PLAN risk 3).
  let group = wid.x + nwg.x * wid.y;
  let tile = group * ${tilesPerWg}u + vsg;
  let lastTile = (params.numRows - 1u) / 16u;
  // A tile holds K words, which is kWords * 16 words or kWords * 4 vec4s.
  let tileBase = min(tile, lastTile) * params.kWords * 4u;
  // 32 lanes times 4 k per iteration: kWords * 16 / 128.
  let iters = params.kWords / 8u;
${KS === 1 ? '' : `  // This workgroup's slice of K. wid.z is the split and num_workgroups.z is the split
  // count, so the cut needs no room in the params block.
  let split = wid.z;
  let per = (iters + ${KS - 1}u) / ${KS}u;
  // kLo and kHi, not lo and hi: the integer path's flush block declares its own lo inside a
  // nested scope, and two names one letter apart in the same function is how a shadowing bug
  // gets written.
  let kLo = min(split * per, iters);
  let kHi = min(kLo + per, iters);
`}${rows.map((r) => `  var acc${r} = 0.0;`).join('\n')}
  // Uniform branch: inScale comes from the uniform block.
  if (params.inScale != 0.0) {
${rows.map((r) => `    var acc${r}i: i32 = 0;`).join('\n')}
${pairs.map((m) => `    var p${m}: i32 = 0;`).join('\n')}
    var sx: i32 = 0;
    var i = ${KS === 1 ? '0u' : 'kLo'};
    while (i < ${KS === 1 ? 'iters' : 'kHi'}) {
      let stop = min(i + 16u, ${KS === 1 ? 'iters' : 'kHi'});
      for (; i < stop; i = i + 1u) {
        let v = wq[tileBase + i * 32u + lane];
        let xk = srqIn(${xr('i * 32u + lane')});
        sx = sx + xk.x + xk.y + xk.z + xk.w;
${intBody}
      }
${flush}
    }
${rows.map((r) => `    acc${r} = f32(acc${r}i - 2 * sx);`).join('\n')}
  } else {
    var sxf = 0.0;
    for (var i = ${KS === 1 ? '0u' : 'kLo'}; i < ${KS === 1 ? 'iters' : 'kHi'}; i = i + 1u) {
      let v = wq[tileBase + i * 32u + lane];
      let xk = ${xr('i * 32u + lane')};
${headBody}
    }
    let sx2f = 2.0 * sxf;
${rows.map((r) => `    acc${r} = acc${r} - sx2f;`).join('\n')}
  }

  // Both reductions above every store, in uniform control flow, then one guarded store block.
${rows.map((r) => `  let sum${r} = mmSum(acc${r}, lid);`).join('\n')}
  if (lane == 0u) {
    let xs = select(1.0, params.inScale, params.inScale != 0.0);
    let row0 = tile * 16u;
${rows.map((r) => (KS === 1
    ? `    if (row0 + ${r}u < params.numRows) { dst[row0 + ${r}u] = srqOut((xs * scales[row0 + ${r}u]) * sum${r}); }`
    // The split cut is the LAST linear step: each partial carries its own row scale, so the merge
    // has only a sum and srqOut left to do. That keeps the scales binding live and leaves the
    // rounding of the scale multiply where the shipped kernel does it.
    : `    if (row0 + ${r}u < params.numRows) { dst[(row0 + ${r}u) * ${KS}u + split] = (xs * scales[row0 + ${r}u]) * sum${r}; }`)).join('\n')}
  }
}
`;
}

/**
 * The 4-bit family's tile loop, the 2-bit tile16u trick at four bits, on an EIGHT row interleaved
 * layout: rows grouped in tiles of eight, a tile holds K words, and word k of tile t carries the
 * unsigned code (0..15, zero point 8) of row 8 t + r at bit offset 4 r. One 32 lane virtual
 * subgroup owns one tile and the lanes deal K between them exactly as the 2-bit loop does: lane
 * `lane` takes k = 4 lane + 128 i + j for j in 0..3 through one vec4<u32> load and one vec4
 * activation load per iteration, so a word meets one scalar activation and the eight rows share
 * it. The classic loop unpacks eight codes of ONE row per word and dots them against eight
 * activations; here one word carries eight rows at ONE k, so the activation traffic per multiply
 * add is an eighth and the unpack is a shift and a mask.
 *
 * The integer path: pair m of four, `(w >> 4m) & 0x000F000F`, is the unsigned codes of rows m and
 * m + 4 in the two 16-bit halves of one u32, and one i32 multiply by the activation code forms
 * both products at once. A product is at most 15 * 128 in magnitude. A BLOCK IS FOUR ITERATIONS,
 * not the 2-bit loop's sixteen: four iterations add 16 products per half, 30720 at most, inside
 * signed 16 bits for the low half (30720 < 32768) and inside i32 for the high one (30720 << 16 plus
 * the low half is under 2^31). The split after each block sign extends the low half and takes it
 * off before shifting the high one down, exact. The zero point comes off once per row at the end,
 * sum((u - 8) x) being sum(u x) - 8 sum(x) with the lane's own activation code sum.
 *
 * The f32 path (an uncalibrated site; every 4-bit site in this checkpoint is calibrated, so it
 * exists for the contract rather than for a tensor) reads the same words against the unsnapped
 * activation with the zero point folded out the same way.
 *
 * No K split in this first cut: a split geometry is refused. K must be a multiple of 128 and the
 * row count a multiple of nothing (a tile past the row count clamps its read and its rows fail
 * the store guard), and every 4-bit linear in the text stack is 256, 1536, 2048 or 6144 rows,
 * all whole tiles. This generator is priced on src/dev/gemvsweep.html?only=tile8 against the
 * classic loop with a synthetic bank before anything is repacked to read it.
 */
function qgemv4TileWgsl(variant: MatmulReduceVariant, geometry: Readonly<GemvGeometry>, prologue: GemvPrologue = 'none'): string {
  const xr = (expr: string): string => (prologue === 'gelu' ? `act(${expr})` : `x[${expr}]`);
  const W = geometry.workgroupSize;
  const tilesPerWg = W / 32;
  if (geometry.rowsPerVsg !== 8) {
    throw new Error(`qgemv-4bit tile8u reads eight row tiles, so rowsPerVsg must be 8, got ${geometry.rowsPerVsg}`);
  }
  if (kSplitsOf(geometry) !== 1) throw new Error('qgemv-4bit tile8u has no K split yet');
  const FLUSH = 4;
  const rows = Array.from({ length: 8 }, (_, r) => r);
  const pairs = Array.from({ length: 4 }, (_, m) => m);
  const comps = ['x', 'y', 'z', 'w'];
  const shifted = (w: string, m: number): string => (m === 0 ? w : `(${w} >> ${4 * m}u)`);
  const pairOps = (w: string, xj: string): string => pairs
    .map((m) => `          p${m} = p${m} + i32(${shifted(w, m)} & 0x000F000Fu) * ${xj};`)
    .join('\n');
  const intBody = comps.map((c) => `        {\n          let w = v.${c};\n${pairOps('w', `xk.${c}`)}\n        }`).join('\n');
  const flush = pairs
    .map((m) => `      { let lo = (p${m} << 16u) >> 16u; acc${m}i = acc${m}i + lo; acc${m + 4}i = acc${m + 4}i + ((p${m} - lo) >> 16u); p${m} = 0; }`)
    .join('\n');
  const headBody = comps.map((c) => [
    '      {',
    `        let w = v.${c};`,
    `        let xj = xk.${c};`,
    '        sxf = sxf + xj;',
    ...rows.map((r) => `        acc${r} = acc${r} + f32(${shifted('w', r)} & 15u) * xj;`),
    '      }',
  ].join('\n')).join('\n');

  return /* wgsl */ `${matmulReducePrelude(variant, W)}
struct GemvParams {
  // Packed u32 words per weight row in the ROW layout, K / 8. The tile layout holds 8 kWords words
  // per tile, which is K words, one per k.
  kWords: u32,
  // Unused on this kernel, which derives its iteration count from kWords.
  kIters: u32,
  numRows: u32,
  pad0: u32,
  inScale: f32,
  outScale: f32,
  upOffset: u32,
  pad2: u32,
}

@group(0) @binding(0) var<storage, read> wq: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> scales: array<f32>;
${prologueWgsl(prologue)}

const SRQ_MAX: f32 = 127.0;
const SRQ_MIN: f32 = -128.0;

fn srqIn(v: vec4<f32>) -> vec4<i32> {
  return vec4<i32>(clamp(round(v / params.inScale), vec4<f32>(SRQ_MIN), vec4<f32>(SRQ_MAX)));
}

fn srqOut(v: f32) -> f32 {
  if (params.outScale == 0.0) {
    return v;
  }
  return clamp(round(v / params.outScale), SRQ_MIN, SRQ_MAX) * params.outScale;
}

@compute @workgroup_size(${W})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
  @builtin(local_invocation_index) lid: u32,
) {
  let lane = lid & 31u;
  let vsg = lid >> 5u;
  let group = wid.x + nwg.x * wid.y;
  let tile = group * ${tilesPerWg}u + vsg;
  let lastTile = (params.numRows - 1u) / 8u;
  // A tile holds K words, which is kWords * 8 words or kWords * 2 vec4s.
  let tileBase = min(tile, lastTile) * params.kWords * 2u;
  // 32 lanes times 4 k per iteration: kWords * 8 / 128.
  let iters = params.kWords / 16u;
${rows.map((r) => `  var acc${r} = 0.0;`).join('\n')}
  // Uniform branch: inScale comes from the uniform block.
  if (params.inScale != 0.0) {
${rows.map((r) => `    var acc${r}i: i32 = 0;`).join('\n')}
${pairs.map((m) => `    var p${m}: i32 = 0;`).join('\n')}
    var sx: i32 = 0;
    var i = 0u;
    while (i < iters) {
      let stop = min(i + ${FLUSH}u, iters);
      for (; i < stop; i = i + 1u) {
        let v = wq[tileBase + i * 32u + lane];
        let xk = srqIn(${xr('i * 32u + lane')});
        sx = sx + xk.x + xk.y + xk.z + xk.w;
${intBody}
      }
${flush}
    }
${rows.map((r) => `    acc${r} = f32(acc${r}i - 8 * sx);`).join('\n')}
  } else {
    var sxf = 0.0;
    for (var i = 0u; i < iters; i = i + 1u) {
      let v = wq[tileBase + i * 32u + lane];
      let xk = ${xr('i * 32u + lane')};
${headBody}
    }
    let sx8f = 8.0 * sxf;
${rows.map((r) => `    acc${r} = acc${r} - sx8f;`).join('\n')}
  }

  // Both reductions above every store, in uniform control flow, then one guarded store block.
${rows.map((r) => `  let sum${r} = mmSum(acc${r}, lid);`).join('\n')}
  if (lane == 0u) {
    let xs = select(1.0, params.inScale, params.inScale != 0.0);
    let row0 = tile * 8u;
${rows.map((r) => `    if (row0 + ${r}u < params.numRows) { dst[row0 + ${r}u] = srqOut((xs * scales[row0 + ${r}u]) * sum${r}); }`).join('\n')}
  }
}
`;
}

/**
 * The merge half of a split K GEMV: sum the S partials a row and apply the output snap.
 *
 * The split kernel already applied `inScale * scales[row]` to each partial, so all that is left
 * here is a sum of S floats and srqOut. One lane a row, 64 a workgroup, so down_proj's 1536 rows
 * are 24 workgroups and the pass is far below the timestamp clock on either machine.
 *
 * `params` is the same block gemvParams writes, with the split count in the slot the GEMV kernels
 * do not read (u[3]).
 */
export function qgemvMergeWgsl(): string {
  return /* wgsl */ `struct GemvParams {
  kWords: u32,
  kIters: u32,
  numRows: u32,
  kSplits: u32,
  inScale: f32,
  outScale: f32,
  pad1: u32,
  pad2: u32,
}

@group(0) @binding(0) var<storage, read> part: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> params: GemvParams;

const SRQ_MAX: f32 = 127.0;
const SRQ_MIN: f32 = -128.0;

fn srqOut(v: f32) -> f32 {
  if (params.outScale == 0.0) {
    return v;
  }
  return clamp(round(v / params.outScale), SRQ_MIN, SRQ_MAX) * params.outScale;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  // Rows past the count sum a clamped base and are dropped by the store guard, the same shape the
  // GEMV kernels use, so there is no early return above a store.
  let r = min(row, params.numRows - 1u);
  let base = r * params.kSplits;
  var total = 0.0;
  for (var s = 0u; s < params.kSplits; s = s + 1u) {
    total = total + part[base + s];
  }
  if (row < params.numRows) {
    dst[row] = srqOut(total);
  }
}
`;
}

/** The one build, since this kernel has no geometry of its own. */
export const QGEMV_MERGE_WGSL = qgemvMergeWgsl();

/** The grid a split K GEMV dispatches: the folded row grid, with the split count on z. */
export function foldedSplitDispatch(groups: number, splits: number): readonly [number, number, number] {
  const [x, y] = foldedDispatch(groups);
  return [x, y, splits];
}

/** The default geometry builds, which are what the Node checks read and the fallback table serves. */
export const QGEMV4_WGSL = qgemvWgsl(4, 'subgroup', GEMV_GEOMETRY_4BIT);
export const QGEMV2_WGSL = qgemvWgsl(2, 'subgroup', GEMV_GEOMETRY_2BIT);
export const QGEMV4_FALLBACK_WGSL = qgemvWgsl(4, 'workgroup', GEMV_GEOMETRY_4BIT);
export const QGEMV2_FALLBACK_WGSL = qgemvWgsl(2, 'workgroup', GEMV_GEOMETRY_2BIT);
export const QGEMV8_WGSL = qgemvWgsl(8, 'subgroup', GEMV_GEOMETRY_8BIT);
export const QGEMV8_FALLBACK_WGSL = qgemvWgsl(8, 'workgroup', GEMV_GEOMETRY_8BIT);
export const QGEMV4_GELU_WGSL = qgemvWgsl(4, 'subgroup', GEMV_GEOMETRY_4BIT, 'gelu');
export const QGEMV2_GELU_WGSL = qgemvWgsl(2, 'subgroup', GEMV_GEOMETRY_2BIT, 'gelu');
export const QGEMV8_GELU_WGSL = qgemvWgsl(8, 'subgroup', GEMV_GEOMETRY_8BIT, 'gelu');
export const QGEMV4_GELU_FALLBACK_WGSL = qgemvWgsl(4, 'workgroup', GEMV_GEOMETRY_4BIT, 'gelu');
export const QGEMV2_GELU_FALLBACK_WGSL = qgemvWgsl(2, 'workgroup', GEMV_GEOMETRY_2BIT, 'gelu');
export const QGEMV8_GELU_FALLBACK_WGSL = qgemvWgsl(8, 'workgroup', GEMV_GEOMETRY_8BIT, 'gelu');

// ---------------------------------------------------------------------------------------------
// The CPU oracle.
// ---------------------------------------------------------------------------------------------

/**
 * The known answer reference for both GEMV kernels, and the definition the shader implements
 * rather than the other way round.
 *
 * A calibrated site, meaning `inScale` is not exactly 0.0, is the ratified integer semantics:
 * snap x onto the int8 grid through quant.ts's `srqCodes`, sum `xcode * wcode` over K as exact
 * integers, and multiply once at the end by `inScale * weight_scale[r]`. That accumulation is
 * exact rather than approximately right. The largest reduction in this text stack is the 4-bit
 * producer down projection at K 6144, bounded by 128 * 8 * 6144 = 6.3e6, and the 2-bit consumer
 * down projection at K 12288 is bounded by 3.2e6; both are far inside i32 and inside f32's exact
 * integer range, so nothing rounds anywhere in the reduction and the result is strictly more
 * accurate than an f32 dot chain over dequantized values.
 *
 * An uncalibrated site, which in this checkpoint means only lm_head, keeps round 1's f32 code
 * dot: sum `code * x` and apply the row scale once after the reduction.
 *
 * Either way `outScale` then rounds the written row onto its own int8 grid through the same
 * `applySrq` the loader and the reference use, and a zero scale leaves it alone.
 *
 * Codes come through qlayout.ts's shader mirror so the oracle and the shader read the packed words
 * identically, and the k-matmul check proves that mirror against quant.ts's byte unpackers on real
 * checkpoint slices.
 *
 * WHERE THE ORACLE IS A MIRROR AND WHERE IT IS A REFERENCE. On the harness fixtures it is a
 * mirror and the cases gate at zero on both axes. On the integer path every product is an integer
 * and the sum is exact under any accumulation order, so only the two closing multiplies can round
 * and they are correctly rounded on both sides; the one place a GPU could disagree is the divide
 * inside the rounding, which WGSL allows 2.5 ULP on, so the fixtures are built with quotients that
 * sit far from a half integer and the k-matmul check measures that margin rather than assuming it.
 * On the f32 path the fixture codes are integers of magnitude at most 8 against x on the 1/8 grid,
 * so every partial sum is a multiple of 1/8 far inside f32's exact range. On arbitrary inputs it is
 * a reference, not a mirror, and such a case states a tolerance somebody argued for.
 */
export function qgemvOracle(
  bits: GemvBits,
  packed: Uint8Array,
  scales: ArrayLike<number>,
  x: Float32Array,
  numRows: number,
  k: number,
  inScale = 0,
  outScale = 0,
): Float32Array {
  const kWords = wordsPerRow(bits, k);
  if (x.length < k) throw new Error(`qgemvOracle: x holds ${x.length}, needs ${k}`);
  if (bits === 2 && bytesAsWords(packed).length < tile16Words(numRows, k)) {
    throw new Error(`qgemvOracle: packed holds ${bytesAsWords(packed).length} words, the tile layout of ${numRows} rows by ${k} needs ${tile16Words(numRows, k)}`);
  }
  // The 2-bit family's fixtures and buffers are in the interleaved layout (qlayout.ts); the
  // oracle reads them back to row words so the arithmetic below is the one the row layout had.
  const words = bits === 2 ? tile16ToRowWords(bytesAsWords(packed), numRows, k) : bytesAsWords(packed);
  if (words.length < numRows * kWords) {
    throw new Error(`qgemvOracle: packed holds ${words.length} words, needs ${numRows * kWords}`);
  }
  const calibrated = Number.isFinite(inScale) && inScale !== 0;
  // The SRQ input prologue runs once for the whole dispatch, not once per row, because every row
  // of a linear reads the same activation vector through the same input_activation_scale.
  const xCodes = calibrated ? srqCodes(x.subarray(0, k), inScale) : null;
  const out = new Float32Array(numRows);
  for (let r = 0; r < numRows; r += 1) {
    const rowWords = words.subarray(r * kWords, (r + 1) * kWords);
    const codes = unpackWordsLikeShader(bits, rowWords, k);
    const rawScale = Number(scales[r] ?? 0);
    const scale = Number.isFinite(rawScale) ? rawScale : 0;
    let sum = 0;
    if (xCodes) {
      for (let i = 0; i < k; i += 1) sum += codes[i] * xCodes[i];
    } else {
      for (let i = 0; i < k; i += 1) sum += codes[i] * x[i];
    }
    // The shader's closing expression, in its order: one f32 multiply to fold the input scale into
    // the row scale, then one more against the reduced sum. On the f32 path the input scale is 1.0
    // and this is bit for bit round 1's `scales[row] * sum`.
    const xs = calibrated ? inScale : 1;
    out[r] = Math.fround(Math.fround(xs * scale) * Math.fround(sum));
  }
  // The SRQ output epilogue, delegated to the one implementation (quant.ts applySrq).
  return applySrq(out, Number.isFinite(outScale) ? outScale : 0, undefined, out);
}

/**
 * The fused prologue's oracle, by construction the chain it replaces: the gelu-mul oracle with
 * the consuming linear's input scale (which stores the snapped values as that pass did), then
 * the GEMV oracle over them, which snaps again to the same codes. `k` is the activation width.
 */
/**
 * The CPU model of a split K dispatch and its merge, for the fold gate.
 *
 * It mirrors where the shader cuts: by ITERATION, not by k. One iteration of the 2-bit tile loop
 * is 32 lanes by 4 k, so a split owns whole 128 wide bands of K and the last one may be short.
 * Each split closes with the shipped `xs * scale * sum` and the merge sums those in f32 and
 * applies the output snap once, which is the only place the arithmetic differs from `qgemvOracle`
 * and the whole reason this is gated rather than argued.
 */
export function qgemvSplitOracle(
  bits: GemvBits,
  packed: Uint8Array,
  scales: ArrayLike<number>,
  x: Float32Array,
  numRows: number,
  k: number,
  splits: number,
  inScale = 0,
  outScale = 0,
): Float32Array {
  if (bits !== 2) throw new Error(`qgemvSplitOracle models the 2-bit tile loop, got ${bits}`);
  const kWords = wordsPerRow(bits, k);
  const words = tile16ToRowWords(bytesAsWords(packed), numRows, k);
  const calibrated = Number.isFinite(inScale) && inScale !== 0;
  const xCodes = calibrated ? srqCodes(x.subarray(0, k), inScale) : null;
  const xs = calibrated ? inScale : 1;
  // The shader's own band arithmetic: iters = kWords / 8, per = ceil(iters / splits), and one
  // iteration spans 128 of K.
  const iters = Math.floor(kWords / 8);
  const per = Math.ceil(iters / splits);
  const out = new Float32Array(numRows);
  for (let r = 0; r < numRows; r += 1) {
    const rowWords = words.subarray(r * kWords, (r + 1) * kWords);
    const codes = unpackWordsLikeShader(bits, rowWords, k);
    const rawScale = Number(scales[r] ?? 0);
    const scale = Number.isFinite(rawScale) ? rawScale : 0;
    let total = 0;
    for (let sIdx = 0; sIdx < splits; sIdx += 1) {
      const lo = Math.min(sIdx * per, iters) * 128;
      const hi = Math.min(Math.min(sIdx * per, iters) + per, iters) * 128;
      let sum = 0;
      if (xCodes) {
        for (let i = lo; i < hi; i += 1) sum += codes[i] * xCodes[i];
      } else {
        for (let i = lo; i < hi; i += 1) sum += codes[i] * x[i];
      }
      total = Math.fround(total + Math.fround(Math.fround(xs * scale) * Math.fround(sum)));
    }
    out[r] = total;
  }
  return applySrq(out, Number.isFinite(outScale) ? outScale : 0, undefined, out);
}

export function qgemvGeluOracle(
  bits: GemvBits,
  packed: Uint8Array,
  scales: ArrayLike<number>,
  gate: Float32Array,
  up: Float32Array,
  numRows: number,
  k: number,
  inScale = 0,
  outScale = 0,
): Float32Array {
  const act = geluMulOracle(gate, up, k, inScale);
  return qgemvOracle(bits, packed, scales, act, numRows, k, inScale, outScale);
}

// ---------------------------------------------------------------------------------------------
// Registry entries.
// ---------------------------------------------------------------------------------------------

// 32 lanes times the codes one word holds: the K span one iteration of the loop at one word per
// lane covers, and the multiple every K of this family's tensors satisfies (1536 and 256 at
// 8-bit, and the real slice fixture's 128).
export const K_SPAN_PER_ITER = { 2: 512, 4: 256, 8: 128 } as const;

/**
 * The params block, exported so the scheduler can restage it in place per dispatch.
 *
 * The two activation scales are part of the dispatch, not part of the pipeline: the same compiled
 * module serves q_proj and lm_head, and what separates them is two f32s in this block. That is why
 * closing every SRQ site adds no dispatch and no second pipeline anywhere.
 */
export function gemvParams(
  kWords: number,
  kIters: number,
  numRows: number,
  inScale = 0,
  outScale = 0,
  kSplits = 1,
  upOffset = 0,
): ArrayBuffer {
  const buf = new ArrayBuffer(32);
  const u = new Uint32Array(buf);
  const f = new Float32Array(buf);
  u[0] = kWords;
  u[1] = kIters;
  u[2] = numRows;
  // The split count, read by the merge kernel alone. The GEMV kernels take their split from
  // wid.z and never look here, so an unsplit dispatch writes the 0 it always wrote.
  u[3] = kSplits === 1 ? 0 : kSplits;
  f[4] = inScale;
  f[5] = outScale;
  u[6] = upOffset;
  u[7] = 0;
  return buf;
}

/** Fold `groups` workgroups into a grid that respects the 65535 per dimension limit. */
export function foldedDispatch(groups: number): readonly [number, number, number] {
  if (groups <= 65535) return [groups, 1, 1];
  const x = 32768;
  return [x, Math.ceil(groups / x), 1];
}

export function bindGemv(bits: GemvBits, prologue: GemvPrologue = 'none', split = false, geometryOverride?: () => Readonly<GemvGeometry>) {
  return (input: KernelBindInput): KernelBindResult => {
    const { device, inputs, output, params } = input;
    const k = params.k | 0;
    const numRows = params.numRows | 0;
    if (k <= 0 || numRows <= 0) throw new Error('qgemv needs params.k and params.numRows');
    if (k % K_SPAN_PER_ITER[bits] !== 0) {
      throw new Error(
        `qgemv-${bits}bit needs K a multiple of ${K_SPAN_PER_ITER[bits]}, got ${k}. Every text `
        + 'tensor satisfies this; a caller that does not is packing the wrong tensor.',
      );
    }
    // The module's two static range quantization scales, straight from the checkpoint. Absent
    // means 0.0, which is what the file stores for an uncalibrated site and what the reference
    // reads as "no rounding here" (quant.ts applySrq).
    const inScale = params.inScale ?? 0;
    const outScale = params.outScale ?? 0;
    const kWords = wordsPerRow(bits, k);
    // Whole iterations at the active geometry's words per lane; the shader's guarded tail takes
    // the remainder (qgemvWgsl). kWords is a multiple of the lane vector width because K is a
    // multiple of K_SPAN_PER_ITER, which is at least 8 words at every legal width.
    const geometry = geometryOverride?.() ?? gemvGeometry(bits);
    const kIters = Math.floor(kWords / (32 * geometry.wordsPerLane));
    // The K split, on the 2-bit tile loop and on the classic 4-bit and 8-bit loops. It rides on
    // z, so a split dispatch is the same grid with a third dimension and an unsplit one is the
    // grid it always was; only a kernel named for the split reads it.
    const kSplits = split ? kSplitsOf(geometry) : 1;
    const wq = inputs.wq;
    const scales = inputs.scales;
    if (!wq || !scales) throw new Error('qgemv needs inputs named wq and scales');
    // The activation bindings: x, or the fused prologue's gate and up.
    const activations: GPUBuffer[] = [];
    if (prologue === 'gelu') {
      if (!inputs.gate || !inputs.up) throw new Error(`${GEMV_GELU_KERNEL[bits]} needs inputs named gate and up`);
      activations.push(inputs.gate, inputs.up);
    } else {
      if (!inputs.x) throw new Error('qgemv needs an input named x');
      activations.push(inputs.x);
    }
    const name = prologue === 'gelu' ? GEMV_GELU_KERNEL[bits] : `qgemv-${bits}bit`;
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
      gemvParams(kWords, kIters, numRows, inScale, outScale, kSplits, params.upOffset ?? 0),
    );

    return {
      layout,
      buffers: [wq, scales, ...activations, output, uniform.binding],
      dispatch: foldedSplitDispatch(Math.ceil(numRows / gemvRowsPerWorkgroup(geometry)), kSplits),
      dispose: uniform.dispose,
    };
  };
}

// The tolerance argument, shared by every matmul case below: the fixtures put codes on the
// integer grid and activations on the 1/8 grid with bounded magnitude, so the K sum is exact in
// f32 whatever order and whatever contraction the compiler picks, and the one scale multiply is
// correctly rounded identically on CPU and GPU. Zero on both axes is therefore a real claim
// about the dispatch, exactly as the scale-add canary argues. Real weight slices keep the same
// gate because the codes are still integers and x is still the fixture grid; only the scale is a
// real f32, and one correctly rounded multiply is still deterministic.
//
// The static range quantization cases keep the same gate for a longer reason, and it is worth
// spelling out because it is the one place a GPU is allowed to disagree with the CPU here. The
// integer reduction cannot round at all, and the two closing multiplies are correctly rounded on
// both sides. What is not correctly rounded is the divide inside `round(v / scale)`: WGSL allows
// f32 division 2.5 ULP. A 2.5 ULP error only changes the rounded integer when the exact quotient
// sits within that of a half integer, so the k-matmul fixture builder MEASURES the smallest
// distance from a half integer over every quotient in these fixtures and asserts it clears a wide
// margin. The scales below were chosen to make that true rather than hoped to be: 0.1875 divides
// the 1/8 activation grid into thirds, whose fractional parts are 1/3 and 2/3 and never 1/2.

/** The input scale of the rounding cases. See the margin argument above. */
const SRQ_IN_THIRDS = 0.1875;
/** An input scale small enough that the fixture activations reach the int8 clamp. */
const SRQ_IN_CLAMPING = 0.015625;
/** The output scale of the rounding cases, kept in the fixture builder's measured margin. */
const SRQ_OUT = 0.0625;

/**
 * The merge half of a split K GEMV, as a registry kernel.
 *
 * It exists only when a device profile asks for `kSplits` on the 2-bit family. The split build
 * writes `numRows * kSplits` partials that already carry their row scale, and this sums the
 * kSplits of each row and applies the output snap. One lane a row, 64 a workgroup, so down_proj's
 * 1536 rows are 24 workgroups.
 *
 * WHY IT IS WORTH A SECOND DISPATCH. down_proj is 1536 rows by K 12288, which at 16 rows a
 * workgroup is 96 workgroups; its siblings gate and up are the transpose and so 768. On the 5070
 * that costs 3.9x on identical bytes, 0.042141 ms against 0.010806, and kSplits 8 recovers it to
 * 0.011879 including this pass, which is within 10 percent of the sibling that never had the
 * problem (lab-results/5070-downproj-shape-vs-prologue-sep04.json). The merge itself is 1.2 us of
 * that 11.9, about a tenth, and it buys the other 280 percent.
 */
export const qgemvMergeKernel: Kernel = {
  name: 'qgemv-merge',
  get wgsl(): string { return QGEMV_MERGE_WGSL; },
  entry: 'main',
  note:
    'The fold of a split K GEMV. No subgroup operations and no workgroup memory, so the one build '
    + 'serves both reduce policies and it is deliberately absent from the fallback table, whose '
    + 'entries all differ from their registry build by design.',
  cases: [
    {
      name: 'fold-8x4',
      inputs: { part: 'kmm.merge.part' },
      expected: 'kmm.merge.expected',
      params: { numRows: 8, kSplits: 4, outScale: SRQ_OUT },
      tolAbs: 0,
      tolUlp: 0,
      note:
        'Four partials a row, already carrying their row scale as the split build writes them, '
        + 'summed in the order the shader sums them and snapped once. Zero tolerance: the fold is '
        + 'three f32 adds and one srqOut, and the fixture is built by the same applySrq the '
        + 'shader mirrors.',
    },
    {
      name: 'fold-8x4-uncalibrated',
      inputs: { part: 'kmm.merge.part' },
      expected: 'kmm.merge.raw.expected',
      params: { numRows: 8, kSplits: 4, outScale: 0 },
      tolAbs: 0,
      tolUlp: 0,
      note:
        'The same partials with outScale 0.0, the uncalibrated site, where the fold is the sum '
        + 'and nothing else. This is the case that would catch a shader applying the snap '
        + 'unconditionally.',
    },
  ],
  bind: (input: KernelBindInput): KernelBindResult => {
    const { device, inputs, output, params } = input;
    const numRows = params.numRows | 0;
    const kSplits = params.kSplits | 0;
    if (numRows <= 0) throw new Error('qgemv-merge needs a positive params.numRows');
    if (kSplits < 2) {
      throw new Error(`qgemv-merge only exists on a split dispatch, got kSplits ${kSplits}`);
    }
    const part = inputs.part;
    if (!part) throw new Error('qgemv-merge needs an input named part');
    const layout = kernelLayout(input, 'qgemv-merge', () => device.createBindGroupLayout({
      label: 'qgemv-merge',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    }));
    // kWords and kIters mean nothing to this kernel; it reads numRows, kSplits and outScale.
    const uniform = kernelUniform(input, 'qgemv-merge params', gemvParams(
      0, 0, numRows, params.inScale ?? 0, params.outScale ?? 0, kSplits,
    ));
    return {
      layout,
      buffers: [part, output, uniform.binding],
      dispatch: foldedDispatch(Math.ceil(numRows / 64)),
      dispose: uniform.dispose,
    };
  },
};

export const qgemv4Kernel: Kernel = {
  name: 'qgemv-4bit',
  // Read at build time against the active geometry, so the performance rig's sweep reaches the
  // pipeline store through the registry like any other build. The default geometry text is the
  // QGEMV4_WGSL constant above, byte for byte.
  get wgsl(): string { return qgemvWgsl(4, 'subgroup'); },
  entry: 'main',
  note:
    'K4. Fused 4-bit dequant GEMV at the measured decode geometry, 64 wide, 2 virtual subgroups, '
    + '4 rows per workgroup (DECODE-CAMPAIGN.md 4.2). Streams each weight byte once, coalesced, '
    + 'with both sides of the module\'s static range quantization fused in as a prologue and an '
    + 'epilogue and the K reduction taken as the exact integer code dot.',
  cases: [
    {
      name: 'synthetic-8x512',
      inputs: { wq: 'kmm.gemv4.wq', scales: 'kmm.gemv4.scales', x: 'kmm.x512' },
      expected: 'kmm.gemv4.expected',
      params: { k: 512, numRows: 8 },
      tolAbs: 0,
      tolUlp: 0,
      note:
        'Two iterations of the streaming loop. Codes and grid x make the K sum exact in f32, so '
        + 'the gate is bit identity.',
    },
    {
      name: 'real-q-proj-slice',
      inputs: {
        wq: 'tensor.attn-q-proj-4bit.raw',
        scales: 'tensor.attn-q-proj-4bit.scale',
        x: 'kmm.x256',
      },
      expected: 'kmm.gemv4.real.expected',
      params: { k: 256, numRows: 1 },
      tolAbs: 0,
      tolUlp: 0,
      note:
        'One row of real checkpoint bytes with its real scale, x on the exact grid. Rows 1 to 3 '
        + 'of the workgroup exercise the store guard.',
    },
    {
      name: 'synthetic-8x512-srq',
      inputs: { wq: 'kmm.gemv4.wq', scales: 'kmm.gemv4.scales', x: 'kmm.x512' },
      expected: 'kmm.gemv4.srq.expected',
      params: { k: 512, numRows: 8, inScale: SRQ_IN_THIRDS, outScale: SRQ_OUT },
      tolAbs: 0,
      tolUlp: 0,
      note:
        'The same weights and activation as the case above, taken through the ratified integer '
        + 'path: the input snaps to int8 codes, the K reduction is the exact integer code dot, and '
        + 'the written row snaps again on the output scale. Both sides of one quantized linear in '
        + 'one dispatch, which is what closing every SRQ site of plan.ts means.',
    },
    {
      name: 'synthetic-8x512-srq-clamped',
      inputs: { wq: 'kmm.gemv4.wq', scales: 'kmm.gemv4.scales', x: 'kmm.x512' },
      expected: 'kmm.gemv4.srqclamp.expected',
      params: { k: 512, numRows: 8, inScale: SRQ_IN_CLAMPING, outScale: 0 },
      tolAbs: 0,
      tolUlp: 0,
      note:
        'The input scale is small enough that most of the fixture lands outside int8 and clamps, '
        + 'so the clamp is covered rather than assumed. The output side is uncalibrated here, '
        + 'which is the other half of the branch.',
    },
  ],
  bind: bindGemv(4),
};

export const qgemv2Kernel: Kernel = {
  name: 'qgemv-2bit',
  get wgsl(): string { return qgemvWgsl(2, 'subgroup', unsplitGemvGeometry(gemvGeometry(2))); },
  entry: 'main',
  note:
    'K5 and K13. Fused 2-bit dequant GEMV, unpack written for instruction count because the 2-bit '
    + 'path is instruction bound on M1 (PREFILL-CAMPAIGN.md retile). The 262,144 row logit head '
    + 'runs here through the 2D grid fold, on the uncalibrated f32 path, while the consumer MLP '
    + 'runs the same module on the integer path.',
  cases: [
    {
      name: 'synthetic-8x1024',
      inputs: { wq: 'kmm.gemv2.tile.wq', scales: 'kmm.gemv2.scales', x: 'kmm.x1024' },
      expected: 'kmm.gemv2.expected',
      params: { k: 1024, numRows: 8 },
      tolAbs: 0,
      tolUlp: 0,
      note: 'Two iterations of the streaming loop, same exact grid argument as the 4-bit case.',
    },
    {
      name: 'real-lm-head-slice',
      inputs: {
        wq: 'tensor.lm-head-2bit.tile',
        scales: 'tensor.lm-head-2bit.scale',
        x: 'kmm.x512',
      },
      expected: 'kmm.gemv2.real.expected',
      params: { k: 512, numRows: 1 },
      tolAbs: 0,
      tolUlp: 0,
      note: 'Real logit head bytes and scale. The full head is this kernel at numRows 262144.',
    },
    {
      name: 'synthetic-8x1024-srq',
      inputs: { wq: 'kmm.gemv2.tile.wq', scales: 'kmm.gemv2.scales', x: 'kmm.x1024' },
      expected: 'kmm.gemv2.srq.expected',
      params: { k: 1024, numRows: 8, inScale: SRQ_IN_THIRDS, outScale: SRQ_OUT },
      tolAbs: 0,
      tolUlp: 0,
      note:
        'The 2-bit family on the integer path, which is the consumer MLP shape. The logit head '
        + 'stays on the f32 path above because both of its activation scales are 0.0 in the '
        + 'checkpoint, so this kernel compiles both and the uniform block picks.',
    },
  ],
  bind: bindGemv(2),
};

/**
 * K12's two linears on a decode step, round 4 lever 3. `qmatmul-8bit` (pleMatmul.ts) keeps the
 * prefill shape, one invocation per output element over a chunk's columns; on a decode step that
 * shape is 4 workgroups of serial K over 14 MB per token and read at 5.6 GB/s, so execute.ts
 * routes the one column case here, onto the GEMV geometry with the same fused SRQ and the same
 * exact integer dot. The fixtures are column 0 of the qmatmul-8bit fixtures and the real slice
 * cases are the same expected buffers, so the two shapes are held to the same bits.
 */
export const qgemv8Kernel: Kernel = {
  name: 'qgemv-8bit',
  get wgsl(): string { return qgemvWgsl(8, 'subgroup'); },
  entry: 'main',
  note:
    'K12 on a decode step: per_layer_input_gate [256, 1536] and per_layer_projection [1536, 256], '
    + 'quant.ts\'s I8 ple-gate-8bit family, on the GEMV geometry with both SRQ sides fused. The '
    + 'prefill shape stays qmatmul-8bit; the two agree column for column at bit identity.',
  cases: [
    {
      name: 'synthetic-10x256',
      inputs: { wq: 'kple.qm8.wq', scales: 'kple.qm8.scales', x: 'kple.qm8.x' },
      expected: 'kple.qgemv8.expected',
      params: { k: 256, numRows: 10 },
      tolAbs: 0,
      tolUlp: 0,
      note:
        'Column 0 of the qmatmul-8bit synthetic fixture, uncalibrated f32 code dot. Ten rows '
        + 'against a workgroup that owns four exercise the store guard.',
    },
    {
      name: 'synthetic-10x256-srq',
      inputs: { wq: 'kple.qm8.wq', scales: 'kple.qm8.scales', x: 'kple.qm8.x' },
      expected: 'kple.qgemv8.srq.expected',
      params: { k: 256, numRows: 10, inScale: SRQ_IN_THIRDS, outScale: SRQ_OUT },
      tolAbs: 0,
      tolUlp: 0,
      note: 'The same bytes through the ratified integer path with both SRQ sides live.',
    },
    {
      name: 'synthetic-10x256-srq-clamped',
      inputs: { wq: 'kple.qm8.wq', scales: 'kple.qm8.scales', x: 'kple.qm8.x' },
      expected: 'kple.qgemv8.srqclamp.expected',
      params: { k: 256, numRows: 10, inScale: SRQ_IN_CLAMPING, outScale: 0 },
      tolAbs: 0,
      tolUlp: 0,
      note: 'An input scale small enough that most of the activation clamps at int8.',
    },
    {
      name: 'real-ple-gate-slice',
      inputs: { wq: 'tensor.ple-gate-8bit.raw', scales: 'tensor.ple-gate-8bit.scale', x: 'kple.x256' },
      expected: 'kple.qm8.real.expected',
      params: { k: 128, numRows: 1 },
      tolAbs: 0,
      tolUlp: 0,
      note:
        'The first 128 real bytes of layer 0 per_layer_input_gate row 0 with their real scale, '
        + 'the same expected buffer the qmatmul-8bit case reproduces: the sign extension against '
        + 'Google\'s own bytes, on this shape too.',
    },
    {
      name: 'real-ple-gate-slice-srq',
      inputs: { wq: 'tensor.ple-gate-8bit.raw', scales: 'tensor.ple-gate-8bit.scale', x: 'kple.xgate' },
      expected: 'kple.qm8.realsrq.expected',
      params: { k: 128, numRows: 1, inScale: PLE_GATE_L0_IN_SCALE, outScale: PLE_GATE_L0_OUT_SCALE },
      tolAbs: 0,
      tolUlp: 0,
      note:
        'Real bytes, real weight scale and layer 0\'s own two activation scales, the exact '
        + 'arithmetic the shipped decode step runs at this site.',
    },
  ],
  bind: bindGemv(8),
};

// The fused prologue GEMVs. One case each on the integer path at zero tolerance, which is the
// shape every site in the model runs (down_proj and per_layer_projection are calibrated), and
// one on the f32 path for the 4-bit family with a tolerance argued at the case. The gate and up
// fixtures are built so every product's quotient by the input scale clears a half integer by a
// wide margin (k-matmul.mjs), so the f32 tanh on the GPU and the f64 one in the oracle snap to
// the same code, and the integer reduction below is then exact as every GEMV case argues.
const GELU_F32_TOL_ABS = 1e-3;

export const qgemv4GeluKernel: Kernel = {
  name: GEMV_GELU_KERNEL[4],
  get wgsl(): string { return qgemvWgsl(4, 'subgroup', gemvGeometry(4), 'gelu'); },
  entry: 'main',
  note: 'K16 with the K10 prologue: gelu(gate) * up read in place of x. The decode down_proj of the 4-bit layers.',
  cases: [
    {
      name: 'synthetic-8x512-gelu-srq',
      inputs: { wq: 'kmm.gemv4.wq', scales: 'kmm.gemv4.scales', gate: 'kmm.gelu.gate512', up: 'kmm.gelu.up512' },
      expected: 'kmm.gemv4.gelu.srq.expected',
      params: { k: 512, numRows: 8, inScale: SRQ_IN_THIRDS, outScale: SRQ_OUT },
      tolAbs: 0,
      tolUlp: 0,
      note: 'The GEMV fixture weights, the activation as gelu(gate) * up snapped by the prologue.',
    },
    {
      name: 'synthetic-8x512-gelu',
      inputs: { wq: 'kmm.gemv4.wq', scales: 'kmm.gemv4.scales', gate: 'kmm.gelu.gate512', up: 'kmm.gelu.up512' },
      expected: 'kmm.gemv4.gelu.expected',
      params: { k: 512, numRows: 8 },
      tolAbs: GELU_F32_TOL_ABS,
      note:
        'The uncalibrated f32 path, which no site in the model runs. The GPU tanh is not the '
        + 'oracle\'s f64 one, so the products differ by a few ULP and the K sum of 512 of them '
        + 'against codes under 8 by at most about 1e-4 in a result of order 10; 1e-3 covers it.',
    },
  ],
  bind: bindGemv(4, 'gelu'),
};

export const qgemv2GeluKernel: Kernel = {
  name: GEMV_GELU_KERNEL[2],
  get wgsl(): string { return qgemvWgsl(2, 'subgroup', unsplitGemvGeometry(gemvGeometry(2)), 'gelu'); },
  entry: 'main',
  note: 'The tile GEMV with the K10 prologue. The decode down_proj of the 2-bit layers.',
  cases: [
    {
      name: 'synthetic-8x1024-gelu-srq',
      inputs: { wq: 'kmm.gemv2.tile.wq', scales: 'kmm.gemv2.scales', gate: 'kmm.gelu.gate1024', up: 'kmm.gelu.up1024' },
      expected: 'kmm.gemv2.gelu.srq.expected',
      params: { k: 1024, numRows: 8, inScale: SRQ_IN_THIRDS, outScale: SRQ_OUT },
      tolAbs: 0,
      tolUlp: 0,
      note: 'The tile fixture weights, the activation as gelu(gate) * up snapped by the prologue.',
    },
  ],
  bind: bindGemv(2, 'gelu'),
};

/**
 * The split K sibling of the entry above, and the only 2-bit GEMV that is allowed to dispatch a z
 * greater than one. The plan emits this name in place of `qgemv-2bit-gelu` exactly when it also
 * emits ROLE_DOWN_MERGE, so the partials it writes are always folded.
 *
 * Its case is the unsplit sibling's, run at the shipped geometry where kSplits is 1 and the split
 * build collapses to the text the sibling compiles: that proves the collapse and nothing more. The
 * split builds at 2, 4 and 8 are proved directly against qgemvSplitOracle by the split K block in
 * scripts/engine-check/k-matmul.mjs, which drives the WGSL at an explicit geometry rather than
 * through a registry case, because a case runs at whatever geometry is active.
 */
export const qgemv2GeluSplitKernel: Kernel = {
  name: GEMV_GELU_SPLIT_KERNEL_2BIT,
  get wgsl(): string { return qgemvWgsl(2, 'subgroup', gemvGeometry(2), 'gelu'); },
  entry: 'main',
  note: 'The split K decode down_proj of the 2-bit layers. Writes partials; the plan folds them.',
  cases: [
    {
      name: 'synthetic-8x1024-gelu-srq-unsplit',
      inputs: { wq: 'kmm.gemv2.tile.wq', scales: 'kmm.gemv2.scales', gate: 'kmm.gelu.gate1024', up: 'kmm.gelu.up1024' },
      expected: 'kmm.gemv2.gelu.srq.expected',
      params: { k: 1024, numRows: 8, inScale: SRQ_IN_THIRDS, outScale: SRQ_OUT },
      tolAbs: 0,
      tolUlp: 0,
      note: 'At kSplits 1 the split build is the unsplit build, so it owes the same answer exactly.',
    },
  ],
  bind: bindGemv(2, 'gelu', true),
};

/**
 * The split K builds of the classic loops (GEMV_SPLIT_KERNEL). Each carries the unsplit case:
 * at kSplits 1 the split build is the plain build to the byte, so it owes the same answer
 * exactly; the split itself is held to the k-matmul text checks and to parity, as the 2-bit
 * split is. Their fold lives in the consumer (kernels/attnPrologue.ts), never in a merge.
 */
export const qgemv4SplitKernel: Kernel = {
  name: GEMV_SPLIT_KERNEL[4],
  get wgsl(): string { return qgemvWgsl(4, 'subgroup', gemvGeometry(4), 'none', true); },
  entry: 'main',
  note: 'The split K decode q, k and v projections of the 4-bit family. Writes partials; the fused attention prologue folds them.',
  cases: [
    {
      name: 'synthetic-8x512-unsplit',
      inputs: { wq: 'kmm.gemv4.wq', scales: 'kmm.gemv4.scales', x: 'kmm.x512' },
      expected: 'kmm.gemv4.expected',
      params: { k: 512, numRows: 8 },
      tolAbs: 0,
      tolUlp: 0,
      note: 'At kSplits 1 the split build is the plain build, so it owes the same answer exactly.',
    },
  ],
  bind: bindGemv(4, 'none', true),
};

export const qgemv8SplitKernel: Kernel = {
  name: GEMV_SPLIT_KERNEL[8],
  get wgsl(): string { return qgemvWgsl(8, 'subgroup', gemvGeometry(8), 'none', true); },
  entry: 'main',
  note: 'The split K decode per_layer_input_gate of the I8 family. Writes partials; the per layer projection folds them in its prologue.',
  cases: [
    {
      name: 'synthetic-10x256-unsplit',
      inputs: { wq: 'kple.qm8.wq', scales: 'kple.qm8.scales', x: 'kple.qm8.x' },
      expected: 'kple.qgemv8.expected',
      params: { k: 256, numRows: 10 },
      tolAbs: 0,
      tolUlp: 0,
      note: 'At kSplits 1 the split build is the plain build, so it owes the same answer exactly.',
    },
  ],
  bind: bindGemv(8, 'none', true),
};

export const qgemv8GeluKernel: Kernel = {
  name: GEMV_GELU_KERNEL[8],
  get wgsl(): string { return qgemvWgsl(8, 'subgroup', gemvGeometry(8), 'gelu'); },
  entry: 'main',
  note: 'The I8 GEMV with the K10 prologue. The decode per_layer_projection, over gelu(gate) times the per layer row.',
  cases: [
    {
      name: 'synthetic-10x256-gelu-offset-row34',
      inputs: { wq: 'kple.qm8.wq', scales: 'kple.qm8.scales', gate: 'kple.gelu.gate256', up: 'kple.gelu.up35x256' },
      expected: 'kple.qgemv8.gelu.srq.expected',
      params: { k: 256, numRows: 10, inScale: SRQ_IN_THIRDS, outScale: SRQ_OUT, upOffset: 34 * 64 },
      tolAbs: 0,
      tolUlp: 0,
      note: 'Reads the final layer row in place. Every preceding row contains a deliberately different value.',
    },
    {
      name: 'synthetic-10x256-gelu-srq',
      inputs: { wq: 'kple.qm8.wq', scales: 'kple.qm8.scales', gate: 'kple.gelu.gate256', up: 'kple.gelu.up256' },
      expected: 'kple.qgemv8.gelu.srq.expected',
      params: { k: 256, numRows: 10, inScale: SRQ_IN_THIRDS, outScale: SRQ_OUT },
      tolAbs: 0,
      tolUlp: 0,
      note: 'The k-ple I8 fixture weights, the activation as gelu(gate) * up snapped by the prologue.',
    },
  ],
  bind: bindGemv(8, 'gelu'),
};
