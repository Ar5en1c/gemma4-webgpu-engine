// Written from docs/ENGINE-PLAN.md section 5.5 and risk 5, the measurement record in
// PREFILL-CAMPAIGN.md rounds 2b and 3, and the WGSL specification's subgroups section. No third
// party engine source and no extracted kernel was read.
//
// SPDX-License-Identifier: Apache-2.0
//
// The one reduction shape, in one file, so there is exactly one of it.
//
// ENGINE-PLAN 5.5 rule 3 is the reason this is a file rather than a snippet copied into each
// kernel: the engine has no `sgExact32` branch and no adapter width gate. There is one reduction
// shape and it is correct everywhere. Two kernels that each grew their own butterfly are two
// reduction shapes, whatever the comments say.
//
// Why a butterfly rather than `subgroupAdd`. On NVIDIA D3D12 and Blackwell a bare `subgroupAdd`
// executed after a lane divergent store miscompiles: 53 compiled kernels passed and 6 failed, all
// six the same template, all six at N_ROWS=2, and the first bad element was always at row 1
// (PREFILL-CAMPAIGN.md round 2b). Two fixes were proven independently on the real kernels: the five
// step `subgroupShuffleXor` butterfly below, and hoisting every reduction above the store into
// uniform control flow (round 3). This engine does both, always, and a later refinement of the
// trigger does not change that: the miscompile needs the straight line unrolled tail shape, and
// loop back edges dodge it, which makes the trigger a property of generated code shape and
// therefore not something a kernel author can rule out by reading their own source.
//
// Why 32 and not the adapter's reported width. `subgroupShuffleXor` with masks below 32 never
// crosses a 32 lane aligned boundary, so on a 64 or 128 lane subgroup this reduces within each
// aligned 32 lane group rather than across the whole subgroup. That is the "32 lane virtual
// subgroup" the measured best decode geometry already uses, where each output row's K reduction
// stays inside one such group so the accumulation order per output element is fixed
// (DECODE-CAMPAIGN.md 4.2). The runtime reported `subgroup_size` builtin was 32 in every
// configuration on Blackwell even though the adapter advertised a maximum of 128
// (PREFILL-CAMPAIGN.md bisect round 1), so the advertised range is advisory and this code does not
// read it. What it does instead is behaviour: ENGINE-PLAN 5.5 rule 4 requires a known answer self
// test at init, and a device that fails it takes the subgroup free path below.
//
// The case this shape does not cover is a subgroup narrower than 32, where the mask 16 shuffle
// leaves the subgroup. That device fails the init self test and gets `workgroupTreeWgsl`.

/** The enable directive. A module that calls `sgSum32` must carry it. */
export const SUBGROUP_ENABLE = 'enable subgroups;';

/**
 * `sgSum32(v)` returns, in every lane, the sum of `v` across that lane's aligned group of 32.
 *
 * Five steps, masks 1, 2, 4, 8, 16. The accumulation order is fixed by the mask sequence and is
 * the same in every lane, which is what makes it a contract rather than a tuning knob
 * (ENGINE-PLAN risk 2, mitigation 2: reduce order is frozen, the safe knobs are workgroup width,
 * rows per workgroup, grid mapping, tile size and activation staging).
 */
export const SUBGROUP_BUTTERFLY_WGSL = /* wgsl */ `
fn sgSum32(v: f32) -> f32 {
  var x = v;
  x = x + subgroupShuffleXor(x, 1u);
  x = x + subgroupShuffleXor(x, 2u);
  x = x + subgroupShuffleXor(x, 4u);
  x = x + subgroupShuffleXor(x, 8u);
  x = x + subgroupShuffleXor(x, 16u);
  return x;
}
`;

/**
 * The subgroup free fallback, for a device that fails the init self test.
 *
 * A workgroup memory tree over `size` lanes. Its accumulation order differs from the butterfly's,
 * so it is a different number in the last bits, and this engine says so rather than pretending
 * otherwise: parity claims in ENGINE-PLAN section 7 are per device and, here, per reduction path.
 * A device on the fallback path is a device whose output is compared against its own baseline.
 *
 * The barrier sits outside the `if`, in uniform control flow, and the loop bound is uniform. Tint
 * hard errors on subgroup operations in non uniform control flow and the same discipline is what
 * keeps a barrier legal (DECODE-FUSION-FINDING.md, what did not work, item 7).
 *
 * The entry barrier, which is the one worth explaining. A tree that only barriers on the way down
 * is correct exactly once. Every lane's last act is to read `scratch[0]`, and nothing orders that
 * read against the next call's `scratch[lid] = v`, so a lane that reaches the second call first
 * can overwrite slot 0 while a slower lane is still reading it. No kernel in the tree calls the
 * same tree twice today, and the two attention calls use two separate arrays, so this is a hazard
 * rather than a live defect. It is fixed here anyway, because a precondition that only holds while
 * nobody writes a second reducing loop is not a property, it is a wait. The cost is one barrier per
 * call on the fallback path, which no measured device takes.
 *
 * @param scratch name of the `var<workgroup>` array this declares and uses
 * @param size    workgroup width, a power of two
 */
export function workgroupTreeWgsl(scratch: string, size: number): string {
  if (size < 2 || (size & (size - 1)) !== 0) {
    throw new Error(`workgroupTreeWgsl needs a power of two width, got ${size}`);
  }
  return /* wgsl */ `
var<workgroup> ${scratch}: array<f32, ${size}>;

fn wgSum(v: f32, lid: u32) -> f32 {
  // Orders every lane's read of slot 0 from a previous call before this call overwrites it.
  workgroupBarrier();
  ${scratch}[lid] = v;
  workgroupBarrier();
  for (var s: u32 = ${size >> 1}u; s > 0u; s = s >> 1u) {
    if (lid < s) {
      ${scratch}[lid] = ${scratch}[lid] + ${scratch}[lid + s];
    }
    workgroupBarrier();
  }
  return ${scratch}[0];
}
`;
}

/**
 * The max butterfly, added by the attention lane, in this file because this file is where
 * reduction shapes live and a second file would be a second shape rule.
 *
 * `sgMax32(v)` returns, in every lane, the maximum of `v` across that lane's aligned group of
 * 32. Softmax needs a running maximum before it can exponentiate anything, so the attention
 * kernels reduce max first and sum second, both above any divergent store, per rule 1. Same mask
 * ladder as `sgSum32` and the same 32 lane alignment argument; max is additionally insensitive
 * to accumulation order, so the frozen order contract of ENGINE-PLAN risk 2 is satisfied
 * trivially here.
 */
export const SUBGROUP_BUTTERFLY_MAX_WGSL = /* wgsl */ `
fn sgMax32(v: f32) -> f32 {
  var x = v;
  x = max(x, subgroupShuffleXor(x, 1u));
  x = max(x, subgroupShuffleXor(x, 2u));
  x = max(x, subgroupShuffleXor(x, 4u));
  x = max(x, subgroupShuffleXor(x, 8u));
  x = max(x, subgroupShuffleXor(x, 16u));
  return x;
}
`;

/**
 * The subgroup free max fallback, mirror of `workgroupTreeWgsl`, for the same device class that
 * fails the init self test. Declares its own scratch array, so a kernel using both trees passes
 * two distinct scratch names.
 */
export function workgroupTreeMaxWgsl(scratch: string, size: number): string {
  if (size < 2 || (size & (size - 1)) !== 0) {
    throw new Error(`workgroupTreeMaxWgsl needs a power of two width, got ${size}`);
  }
  return /* wgsl */ `
var<workgroup> ${scratch}: array<f32, ${size}>;

fn wgMax(v: f32, lid: u32) -> f32 {
  ${scratch}[lid] = v;
  workgroupBarrier();
  for (var s: u32 = ${size >> 1}u; s > 0u; s = s >> 1u) {
    if (lid < s) {
      ${scratch}[lid] = max(${scratch}[lid], ${scratch}[lid + s]);
    }
    workgroupBarrier();
  }
  return ${scratch}[0];
}
`;
}

/**
 * The known answer self test of ENGINE-PLAN 5.5 rule 4, as WGSL.
 *
 * Every lane contributes its own index, so the answer for a 32 lane group starting at `g` is
 * `sum(g .. g+31)`, which is exact in f32 and independent of accumulation order. The host checks
 * it and, on a mismatch, routes every reduction in the engine to the workgroup tree. Behaviour
 * verified, never vendor sniffed: the mitigation for a driver bug nobody has met yet is the same
 * test, which is portable by construction.
 *
 * Written here rather than in the device module so the test and the thing it tests cannot drift.
 */
export const SUBGROUP_SELFTEST_WGSL = /* wgsl */ `${SUBGROUP_ENABLE}
${SUBGROUP_BUTTERFLY_WGSL}
@group(0) @binding(0) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>) {
  let idx = wid.x * 64u + lid;
  out[idx] = sgSum32(f32(idx));
}
`;

/** The value `SUBGROUP_SELFTEST_WGSL` must write at `index`, given a 64 wide workgroup. */
export function subgroupSelfTestExpected(index: number): number {
  const group = Math.floor(index / 32) * 32;
  // Sum of 32 consecutive integers starting at `group`.
  return 32 * group + (31 * 32) / 2;
}
