// Written from docs/ENGINE-PLAN.md sections 5, 5.5 and 7, this lane's harness brief, and the WGSL
// and WebGPU specifications.
//
// SPDX-License-Identifier: Apache-2.0
//
// The kernel registry. One entry per kernel, and the entry carries everything the dev harness
// needs to run that kernel against reference data and everything the engine needs to dispatch it.
// A kernel that is not in here cannot be proved, so it does not ship.
//
// The shape of an entry is deliberately small. A kernel owns its WGSL, its entry point, and the
// function that turns a set of named buffers into a bind group layout, an ordered buffer list and
// a dispatch size. It does not own device creation, buffer upload, readback, or comparison. Those
// belong to the caller, so the same entry serves the harness page and the engine's scheduler.

// The binding value helpers live in ./binding.ts, not here: kernels import them as values, and a
// value import back into this module, which imports every kernel, would be a runtime import
// cycle. The types are re-exported below so callers can keep treating this file as the seam.
import type { KernelBinding, UniformSlice } from './binding';

export type { KernelBinding, UniformSlice } from './binding';

/** What the harness hands a kernel's `bind`. Buffers are already uploaded and correctly sized. */
export interface KernelBindInput {
  device: GPUDevice;
  /**
   * Input buffers by the binding name the kernel's own case declares. Created with
   * STORAGE | COPY_DST and filled from the manifest before `bind` is called.
   */
  inputs: Readonly<Record<string, GPUBuffer>>;
  /**
   * The single output buffer, created with STORAGE | COPY_SRC | COPY_DST and sized from the
   * case's expected entry. The caller reads this buffer back and nothing else, so a kernel with
   * several outputs is expressed as several cases, one per output, or as one packed output.
   */
  output: GPUBuffer;
  /** Scalars from the case. Kernel specific, named by the kernel. */
  params: Readonly<Record<string, number>>;
  /**
   * Stage a params block into a caller owned uniform arena and get the slice to bind. When
   * present, `bind` creates no uniform buffer of its own; when absent, it falls back to a
   * throwaway buffer it disposes. BufferManager.stageUniform in buffers.ts is the intended
   * implementation, which is how the decode loop updates a uniform in place instead of paying a
   * create, write, destroy per dispatch. The `kernelUniform` helper in ./binding.ts is the one
   * place that chooses.
   */
  stageUniform?: (data: ArrayBuffer | ArrayBufferView) => UniformSlice;
  /**
   * Fetch a bind group layout from a caller owned cache, keyed by kernel name. When present,
   * layouts are created once per kernel per device rather than once per bind. The `kernelLayout`
   * helper in ./binding.ts is the one place that chooses.
   */
  layoutFor?: (key: string, make: () => GPUBindGroupLayout) => GPUBindGroupLayout;
}

export interface KernelBindResult {
  /** The explicit bind group layout. Explicit, not `auto`, so the binding budget is legible. */
  layout: GPUBindGroupLayout;
  /**
   * The bindings, in binding order: index `n` of this array is `@binding(n)`. Anything the
   * kernel created or staged for itself, a uniform block for instance, appears here too. Entries
   * may be whole buffers or `{ buffer, offset, size }` sub ranges.
   */
  buffers: KernelBinding[];
  /** Workgroup counts for `dispatchWorkgroups`. */
  dispatch: readonly [number, number, number];
  /** Pipeline overridable constants, when the kernel specializes at compile time. */
  constants?: Record<string, number>;
  /** Release anything `bind` allocated. Called after the run whether it passed or failed. */
  dispose?: () => void;
}

/**
 * One provable run of a kernel: which manifest entries feed it, which one it has to reproduce,
 * and the tolerance it is allowed.
 *
 * Tolerances default to zero on both axes. That is not optimism, it is the only default that makes
 * a green result mean something. A kernel whose maths is exact in f32, which is most known answer
 * canaries, passes at zero. A kernel that genuinely cannot, a quantized GEMV for instance, states
 * its floor here as a number somebody argued for, and the argument goes in `note`.
 */
export interface KernelCase {
  name: string;
  /** Manifest entry id per input binding name. */
  inputs: Record<string, string>;
  /** Manifest entry id of the expected output. */
  expected: string;
  params?: Record<string, number>;
  /**
   * How the harness initializes the output buffer before the dispatch. Default 'sentinel': the
   * wrote-nothing detector's fill, for kernels that write every element of their output. A kernel
   * case that legitimately leaves part of its output untouched and asserts those slots hold zero,
   * the KV append shape for instance, declares 'zero' instead: round 1's verifier proved the two
   * conventions are mutually exclusive, because the untouched slots read back as the sentinel and
   * can never equal the expected zeros. With 'zero' the sentinel scan is skipped and the silent
   * no-dispatch guard is carried by the comparison itself, which the harness enforces by refusing
   * a 'zero' case whose expected buffer is all zeros.
   */
  outputInit?: 'sentinel' | 'zero';
  /** Absolute difference allowed per element. */
  tolAbs?: number;
  /** ULP distance allowed per element, f32 only. */
  tolUlp?: number;
  /** Cosine similarity floor, when the case is a quantized one and per element diffs are noise. */
  minCosine?: number;
  note?: string;
}

export interface Kernel {
  /** Stable name. The harness runs kernels by this, so do not rename one casually. */
  name: string;
  /** The WGSL source, as an exported template string so tsc sees the file. */
  wgsl: string;
  /** Entry point name inside `wgsl`. */
  entry: string;
  /** Reference cases this kernel is proved against. At least one, or it is not registered. */
  cases: KernelCase[];
  bind(input: KernelBindInput): KernelBindResult;
  note?: string;
}

// Kernels, in the order ENGINE-PLAN section 8 milestone M2 builds them. Other lanes append here.
//
// Two rules from ENGINE-PLAN 5.5 bind every entry in this list, and they are stated here rather
// than in each kernel because they are properties of the registry, not of any one kernel:
//
//   1. No reduction after a lane divergent store in straight line code. Hoist reductions above
//      stores, into uniform control flow, and end with one guarded store block. Where a reduction
//      has to sit in a tail, use the 32 lane subgroupShuffleXor butterfly, never a bare
//      subgroupAdd (PREFILL-CAMPAIGN.md rounds 2b and 3, where a bare subgroupAdd after a
//      divergent store miscompiled on NVIDIA D3D12 in six of 59 compiled kernels, always at row 1).
//   2. No compile time constant loop bound where the trip count is large. Metal fully unrolled a
//      256 iteration constant bound loop into a megakernel with a multi second compile that halved
//      decode; a runtime opaque uniform bound fixed it (DECODE-FUSION-FINDING.md, item 5).
//
// And one budget: the M1 Chrome adapter caps maxStorageBuffersPerShaderStage at 10 and requesting
// more is impossible (DECODE-CAMPAIGN.md 4.7). The harness counts every kernel's bindings against
// the live limit on each run, so the budget is checked rather than remembered.
import { pleGateSplitKernel, pleFoldProjectionKernel } from './pleSplit';
import { exactSplitKernel, exactMergeKernel } from './exactSplit';
import { scaleAddKernel } from './scaleAdd';
import { rmsNormKernel, rmsNormWeightlessKernel } from './rmsNorm';
import { ropeKernel } from './rope';
import { embedTokensGatherKernel, pleGatherKernel, pleGatherSlicedKernel } from './embedGather';
import { qkvBatchKernel, gateUpBatch4Kernel, gateUpBatch2Kernel } from './projectionBatch';
import { qgemv2Kernel, qgemv4Kernel, qgemv8Kernel, qgemv4GeluKernel, qgemv2GeluKernel, qgemv8GeluKernel, qgemv2GeluSplitKernel, qgemvMergeKernel, qgemv4SplitKernel, qgemv8SplitKernel } from './qgemv';
import {
  qgemv2WideKernel, qgemv4WideKernel, qgemv8WideKernel, qgemv4GeluWideKernel, qgemv2GeluWideKernel, qgemv8GeluWideKernel,
  qgemv2GeluSplitWideKernel,
} from './qgemvWide';
import { qNormRopeKernel, kvPrologueKernel, qNormRopeFoldKernel, kvPrologueFoldKernel } from './attnPrologue';
import { qgemm2Kernel, qgemm4Kernel } from './qgemm';
import { qmatmul8Kernel } from './pleMatmul';
import { denseMatmulKernel } from './denseMatmul';
import { argmaxFinalKernel, argmaxPartialKernel } from './argmax';
import { attentionDecodeKernel, attentionPrefillKernel, attentionDecodeSplitKernel, attentionMergeKernel } from './attention';
import { kvCacheStoreKernel } from './kvStore';
import { geluMulKernel } from './geluMul';
import { normResidualKernel } from './mlpEpilogue';
import { normResidualNormKernel } from './blockJoin';
import { logitSoftcapKernel } from './logitSoftcap';
import { f16PackKernel, f16UnpackKernel } from './f16Cast';
import { repack2Kernel } from './repack';

export const KERNELS: Kernel[] = [
  scaleAddKernel,
  // The norm, rope and embedding family. Three of the four silent corruption traps of
  // ENGINE-PLAN risk 1 live in these five entries: the weightless per KV head V norm, the partial
  // RoPE of the seven full attention layers, and the plain norm form with no (1 + weight). The
  // fourth, the attention scale of 1.0, belongs to the attention lane.
  rmsNormKernel,
  rmsNormWeightlessKernel,
  ropeKernel,
  embedTokensGatherKernel,
  pleGatherKernel,
  // The PLE gather over one vocabulary range of a table too large for one buffer on this adapter.
  // Registered rather than conditional: a registry that depends on the device is a registry the
  // Node checks cannot walk. It compiles only when a plan names it, which happens only where the
  // split is planned, so on every adapter with room for the whole table this entry costs a module
  // that is never created. See ./embedGather.ts and ../tableSplit.ts.
  pleGatherSlicedKernel,
  // The QAT dequant matmul family, K4, K5, K13, K16 and K15, which is where the per token
  // bandwidth goes on the M1 (DECODE-CAMPAIGN.md 1) and therefore where the engine's performance
  // lives. Checked by scripts/engine-check/k-matmul.mjs.
  qgemv4Kernel,
  exactSplitKernel, exactMergeKernel,
  pleGateSplitKernel, pleFoldProjectionKernel,
  qkvBatchKernel, gateUpBatch4Kernel, gateUpBatch2Kernel,
  qgemv2Kernel,
  qgemm4Kernel,
  qgemm2Kernel,
  qgemv4GeluKernel,
  qgemv2GeluKernel,
  qgemv2GeluSplitKernel,
  // Only reachable on a device profile that asks for kSplits; unreferenced by the plan otherwise.
  qgemvMergeKernel,
  qgemv8GeluKernel,
  qgemv4WideKernel,
  qgemv2WideKernel,
  qgemv8WideKernel,
  qgemv4GeluWideKernel,
  qgemv2GeluWideKernel,
  qgemv2GeluSplitWideKernel,
  qNormRopeKernel,
  kvPrologueKernel,
  // The 4-bit attention split and its fold, only reachable on a geometry that asks for kSplits
  // on the 4-bit family; the 8-bit split is registered for the PLE gate's turn.
  qgemv4SplitKernel,
  qgemv8SplitKernel,
  qNormRopeFoldKernel,
  kvPrologueFoldKernel,
  qgemv8GeluWideKernel,
  repack2Kernel,
  // The per layer embedding path's own two matmul families, K12, which round 1 did not plan and
  // round 2's dispatch lane could only refuse by name. Neither reduces across lanes, so neither
  // takes a FALLBACK_WGSL entry: the one module is correct under either reduce policy.
  //   qmatmul-8bit      per_layer_input_gate and per_layer_projection, quant.ts's I8
  //                     ple-gate-8bit family, which the 2-bit and 4-bit unpackers would have read
  //                     as plausible garbage rather than refused. The prefill shape since round 4;
  //                     a decode step runs the same two linears as qgemv-8bit on the GEMV
  //                     geometry (lever 3), which does reduce across lanes and so has a fallback.
  //   dense-bf16-matmul per_layer_model_projection, the one tensor in modules_to_not_convert that
  //                     a forward actually multiplies by. No SRQ, because it stores no scales.
  qmatmul8Kernel,
  qgemv8Kernel,
  denseMatmulKernel,
  argmaxPartialKernel,
  argmaxFinalKernel,
  // The attention family: K9 then K8, in store-before-attend order because that is also the
  // scheduler's contract (KV for the query position is written before attention reads it).
  // Trap 2 lives here: the attention scale is 1.0 exactly, a named constant in attention.ts,
  // asserted against the reference capture by scripts/engine-check/k-attention.mjs.
  kvCacheStoreKernel,
  attentionDecodeKernel,
  attentionDecodeSplitKernel,
  // Only reachable on a device profile that asks for kvSplits; unreferenced by the plan otherwise.
  attentionMergeKernel,
  attentionPrefillKernel,
  // The MLP and glue family, K10, K11 and K14, plus the storage cast pair. The gated activation is
  // gelu_pytorch_tanh, which config.json names and scripts/engine-check/k-mlp.mjs asserts against
  // the erf form; the epilogue norms before it adds the residual, which is the order the reference
  // uses and the one an ordinary pre-norm habit gets backwards. Plain elementwise glue is
  // scale-add above, at alpha 1 for a residual add and with a zero operand for a scale, so there is
  // no second copy of it here.
  geluMulKernel,
  normResidualKernel,
  normResidualNormKernel,
  logitSoftcapKernel,
  f16PackKernel,
  f16UnpackKernel,
];

export function kernelByName(name: string): Kernel | null {
  return KERNELS.find((k) => k.name === name) ?? null;
}

export function kernelNames(): string[] {
  return KERNELS.map((k) => k.name);
}
