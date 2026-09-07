// Written from docs/ENGINE-PLAN.md section 5.5 (the subgroup policy), the WebGPU specification,
// and the measurement record in ../../../../gemma4-kernels-lab/PREFILL-CAMPAIGN.md (the NVIDIA
// miscompile and its self test) and DECODE-FUSION-FINDING.md (asynchronous pipeline errors). No
// vendored bundle, no extracted kernel and no third party engine source was read.
//
// SPDX-License-Identifier: Apache-2.0
//
// Pipeline construction for the engine: which reduction variant every kernel compiles with, the
// known answer self test that decides it, and a cache of built pipelines so warmup() compiles the
// decode set once and generate() never compiles anything.
//
// The policy, restated from ENGINE-PLAN 5.5 because this file is where it becomes code:
//
//  - Variant selection happens once, at build time, per device, from two inputs only: whether the
//    `subgroups` feature was granted, and whether the known answer self test PASSED on this
//    device. Behaviour verified, never vendor sniffed. There is no `sgExact32` style branch on
//    the adapter's reported subgroup width anywhere in this engine (rule 3), because the reported
//    range is advisory: a Blackwell adapter advertised a maximum of 128 while the builtin
//    measured 32 in every configuration (PREFILL-CAMPAIGN.md, bisect round 1).
//  - The subgroup variant is the 32 lane shuffleXor butterfly of kernels/subgroupReduce.ts, which
//    is immune to the reconvergence miscompile a bare subgroupAdd after a divergent store
//    triggers on NVIDIA D3D12 (PREFILL-CAMPAIGN.md rounds 2b and 3). The fallback is the
//    workgroup memory tree, correct on every adapter with no subgroup feature at all.
//  - One reduction shape per variant, engine wide. A kernel is never mixed and matched.

import { createComputePipelineChecked, withErrorScopes, type Gemma4Device } from './device';
import { pleGateSplitWgsl, pleFoldProjectionWgsl } from './kernels/pleSplit';
import { exactSplitWgsl } from './kernels/exactSplit';
import { projectionBatchWgsl } from './kernels/projectionBatch';
import { kernelByName, type Kernel } from './kernels/registry';
import { SUBGROUP_SELFTEST_WGSL, subgroupSelfTestExpected } from './kernels/subgroupReduce';
import type { MatmulReduceVariant } from './kernels/qgemv';
import { qgemvWgsl, gemvGeometry, unsplitGemvGeometry, GEMV_GELU_SPLIT_KERNEL_2BIT, GEMV_SPLIT_KERNEL } from './kernels/qgemv';
import { gemvWideKernelName, qgemv2GeluSplitWideFallbackWgsl, qgemvWideWgsl } from './kernels/qgemvWide';
import { QGEMM4_FALLBACK_WGSL, QGEMM4_PREFILL4_FALLBACK_WGSL, QGEMM4_SINGLE_FALLBACK_WGSL } from './kernels/qgemm';
import { RMS_NORM_FALLBACK_WGSL, RMS_NORM_WEIGHTLESS_FALLBACK_WGSL } from './kernels/rmsNorm';
import {
  KV_PROLOGUE_FALLBACK_WGSL, KV_PROLOGUE_FOLD_KERNEL, KV_PROLOGUE_KERNEL, Q_NORM_ROPE_FALLBACK_WGSL, Q_NORM_ROPE_FOLD_KERNEL, Q_NORM_ROPE_KERNEL,
  kvPrologueFoldFallbackWgsl, qNormRopeFoldFallbackWgsl,
} from './kernels/attnPrologue';
import { ARGMAX_FINAL_FALLBACK_WGSL, ARGMAX_PARTIAL_FALLBACK_WGSL } from './kernels/argmax';
import { attentionWgsl, attentionGeometry, unsplitAttentionGeometry, ATTENTION_DECODE_SPLIT_KERNEL } from './kernels/attention';
import { NORM_RESIDUAL_FALLBACK_WGSL } from './kernels/mlpEpilogue';
import { NORM_RESIDUAL_NORM_FALLBACK_WGSL } from './kernels/blockJoin';

export type ReduceVariant = MatmulReduceVariant; // 'subgroup' | 'workgroup'

/**
 * The registry entries whose WGSL carries a reduction, mapped to the workgroup tree build of the
 * same kernel. Every other registered kernel has no reduction and compiles identically under
 * either policy. A kernel added to the registry with a reduction must be added here too; the
 * orchestrator check cross references this table against the registry so a miss fails a check
 * rather than shipping a subgroup kernel to a device that failed the self test.
 */
export const FALLBACK_WGSL: Readonly<Record<string, string>> = Object.freeze({
  get 'qkv-batch-4bit'(): string { return projectionBatchWgsl(4, 3, 'workgroup'); },
  get 'gate-up-batch-4bit'(): string { return projectionBatchWgsl(4, 2, 'workgroup'); },
  get 'gate-up-batch-2bit'(): string { return projectionBatchWgsl(2, 2, 'workgroup'); },
  get 'qgemv-4bit-gelu-exact-split'(): string { return exactSplitWgsl('workgroup'); },
  get 'ple-gate-split'(): string { return pleGateSplitWgsl('workgroup'); },
  get 'ple-fold-projection'(): string { return pleFoldProjectionWgsl('workgroup'); },
  'rms-norm': RMS_NORM_FALLBACK_WGSL,
  'rms-norm-weightless': RMS_NORM_WEIGHTLESS_FALLBACK_WGSL,
  // The fused attention prologue reduces through the norm kernel's own pieces, so its portable
  // build is the norm's workgroup tree (kernels/attnPrologue.ts).
  [Q_NORM_ROPE_KERNEL]: Q_NORM_ROPE_FALLBACK_WGSL,
  [KV_PROLOGUE_KERNEL]: KV_PROLOGUE_FALLBACK_WGSL,
  // Read at build time against the active decode GEMV geometry (qgemv.ts setGemvGeometry), the
  // same way the registry entries read theirs, so the performance rig's sweep compiles the
  // portable build of the geometry under test too. At the default geometry these are the
  // QGEMV4_FALLBACK_WGSL and QGEMV2_FALLBACK_WGSL constants byte for byte.
  get 'qgemv-4bit'(): string { return qgemvWgsl(4, 'workgroup'); },
  get 'qgemv-2bit'(): string { return qgemvWgsl(2, 'workgroup', unsplitGemvGeometry(gemvGeometry(2))); },
  get 'qgemv-8bit'(): string { return qgemvWgsl(8, 'workgroup'); },
  get 'qgemv-4bit-gelu'(): string { return qgemvWgsl(4, 'workgroup', undefined, 'gelu'); },
  get 'qgemv-2bit-gelu'(): string { return qgemvWgsl(2, 'workgroup', unsplitGemvGeometry(gemvGeometry(2)), 'gelu'); },
  // The split siblings read the live geometry, which is the whole point of them.
  get [GEMV_GELU_SPLIT_KERNEL_2BIT](): string { return qgemvWgsl(2, 'workgroup', gemvGeometry(2), 'gelu'); },
  get [GEMV_SPLIT_KERNEL[4]](): string { return qgemvWgsl(4, 'workgroup', gemvGeometry(4), 'none', true); },
  get [GEMV_SPLIT_KERNEL[8]](): string { return qgemvWgsl(8, 'workgroup', gemvGeometry(8), 'none', true); },
  get [Q_NORM_ROPE_FOLD_KERNEL](): string { return qNormRopeFoldFallbackWgsl(); },
  get [KV_PROLOGUE_FOLD_KERNEL](): string { return kvPrologueFoldFallbackWgsl(); },
  get 'qgemv-8bit-gelu'(): string { return qgemvWgsl(8, 'workgroup', undefined, 'gelu'); },
  // Keyed off gemvWideKernelName rather than written out, because the wide family's column count
  // is a measured constant (GEMV_WIDE_COLS) and round 7 moved it from four to two. Spelling the
  // names here by hand meant the table silently stopped matching the registry when it moved, and
  // the portable build of six kernels went untested until device-profile caught it.
  get [gemvWideKernelName(4)](): string { return qgemvWideWgsl(4, 'workgroup'); },
  get [gemvWideKernelName(2)](): string { return qgemvWideWgsl(2, 'workgroup'); },
  get [gemvWideKernelName(8)](): string { return qgemvWideWgsl(8, 'workgroup'); },
  get [gemvWideKernelName(4, 'gelu')](): string { return qgemvWideWgsl(4, 'workgroup', undefined, 'gelu'); },
  get [gemvWideKernelName(2, 'gelu')](): string { return qgemvWideWgsl(2, 'workgroup', undefined, 'gelu'); },
  get [gemvWideKernelName(8, 'gelu')](): string { return qgemvWideWgsl(8, 'workgroup', undefined, 'gelu'); },
  // The wide split sibling reads the live geometry, as the one column split above does.
  get [gemvWideKernelName(2, 'gelu', true)](): string { return qgemv2GeluSplitWideFallbackWgsl(); },
  'qgemm-4bit': QGEMM4_FALLBACK_WGSL,
  'qgemm-4bit-single': QGEMM4_SINGLE_FALLBACK_WGSL,
  'qgemm-4bit-prefill4': QGEMM4_PREFILL4_FALLBACK_WGSL,
  // qgemm-2bit reduces nothing (a lane per output row) and has one build under both policies.
  'argmax-partial': ARGMAX_PARTIAL_FALLBACK_WGSL,
  'argmax-final': ARGMAX_FINAL_FALLBACK_WGSL,
  // The decode and prefill attention entries share one module text and differ by entry point,
  // so both map to the same fallback build. Read at build time against the active attention
  // geometry (attention.ts setAttentionGeometry) for the same reason as the GEMV entries above;
  // at the default geometry this is the ATTENTION_FALLBACK_WGSL constant byte for byte.
  get 'attention-decode'(): string { return attentionWgsl('workgroup', unsplitAttentionGeometry(attentionGeometry())); },
  get 'attention-prefill'(): string { return attentionWgsl('workgroup', unsplitAttentionGeometry(attentionGeometry())); },
  get [ATTENTION_DECODE_SPLIT_KERNEL](): string { return attentionWgsl('workgroup'); },
  'norm-residual': NORM_RESIDUAL_FALLBACK_WGSL,
  'norm-residual-norm': NORM_RESIDUAL_NORM_FALLBACK_WGSL,
});

/**
 * The build time decision of ENGINE-PLAN 5.5 rule 4. The subgroup path is taken only when the
 * feature was granted AND the self test came back exactly right; anything else, including a self
 * test that could not run, is the portable workgroup tree. A granted feature is permission to
 * compile, never permission to trust.
 */
export function chooseReduceVariant(subgroupsGranted: boolean, selfTestPassed: boolean | null): ReduceVariant {
  return subgroupsGranted && selfTestPassed === true ? 'subgroup' : 'workgroup';
}

/** The WGSL a kernel compiles with under a policy. Registry text for 'subgroup', table for 'workgroup'. */
export function wgslForKernel(kernel: Kernel, variant: ReduceVariant): string {
  if (variant === 'workgroup') {
    const fallback = FALLBACK_WGSL[kernel.name];
    if (fallback !== undefined) return fallback;
  }
  return kernel.wgsl;
}

/** Workgroups the self test dispatches; 64 lanes each, so 256 values checked. */
const SELFTEST_WORKGROUPS = 4;

/**
 * Run the known answer subgroup self test on a live device. Returns true only when every value
 * matches subgroupSelfTestExpected exactly; any GPU error, compile failure or mismatch is false,
 * and false routes every reduction in the engine to the workgroup tree. The test is behaviour
 * verified, which is why it also covers driver bugs nobody has met yet (ENGINE-PLAN risk 5).
 */
export async function runSubgroupSelfTest(gpu: Gemma4Device): Promise<boolean> {
  if (!gpu.features.subgroups) return false;
  const device = gpu.device;
  const count = SELFTEST_WORKGROUPS * 64;
  const { pipeline, error } = await createComputePipelineChecked(device, {
    label: 'subgroup-selftest',
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ label: 'subgroup-selftest', code: SUBGROUP_SELFTEST_WGSL }),
      entryPoint: 'main',
    },
  });
  if (!pipeline || error) return false;

  const out = device.createBuffer({
    label: 'subgroup-selftest:out',
    size: count * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const staging = device.createBuffer({
    label: 'subgroup-selftest:staging',
    size: count * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  try {
    const { error: runError } = await withErrorScopes(device, async () => {
      const bindGroup = device.createBindGroup({
        label: 'subgroup-selftest',
        layout: pipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: out } }],
      });
      const encoder = device.createCommandEncoder({ label: 'subgroup-selftest' });
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(SELFTEST_WORKGROUPS);
      pass.end();
      encoder.copyBufferToBuffer(out, 0, staging, 0, count * 4);
      device.queue.submit([encoder.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
    });
    if (runError) return false;
    const values = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    for (let i = 0; i < count; i += 1) {
      if (values[i] !== subgroupSelfTestExpected(i)) return false;
    }
    return true;
  } catch {
    return false;
  } finally {
    out.destroy();
    staging.destroy();
  }
}

export interface BuiltPipeline {
  readonly kernel: Kernel;
  readonly variant: ReduceVariant;
  readonly pipeline: GPUComputePipeline;
}

/**
 * Builds and caches compute pipelines under one reduce policy. Shader modules are cached by their
 * source text, because the weighted and weightless norms share nothing but the two matmuls of one
 * bit width share their whole module. Creation failures are values, not exceptions, per
 * ENGINE-PLAN 5.5 rule 7: pipeline errors are asynchronous and try catch never sees them.
 */
export class PipelineStore {
  private readonly gpu: Gemma4Device;
  readonly variant: ReduceVariant;
  private readonly modules = new Map<string, GPUShaderModule>();
  private readonly pipelines = new Map<string, BuiltPipeline>();

  constructor(gpu: Gemma4Device, variant: ReduceVariant) {
    this.gpu = gpu;
    this.variant = variant;
  }

  private moduleFor(label: string, code: string): GPUShaderModule {
    let module = this.modules.get(code);
    if (!module) {
      module = this.gpu.device.createShaderModule({ label, code });
      this.modules.set(code, module);
    }
    return module;
  }

  /**
   * The built pipeline for a registered kernel, compiling on first use.
   *
   * `bindGroupLayout` is required, and that is round 2's correction to this class. A pipeline built
   * with `layout: 'auto'` owns a layout nobody else can name, so a bind group made from a kernel's
   * own `bind` result is rejected against it. Round 1 built every pipeline here with 'auto' and had
   * no caller that made bind groups, so nothing noticed; the moment the scheduler binds, an 'auto'
   * pipeline is not a pipeline it can use. The layout a kernel's `bind` returns is stable per
   * kernel, which is why one is enough and why the cache is still keyed by kernel name alone.
   */
  /**
   * The synchronous fast path: the built pipeline when it is already in the store, else undefined.
   * The decode loop binds 556 steps a token, and an `await` per step costs a microtask and a
   * promise each even when the store hits, which is host time on the critical path (docs/
   * ENGINE-PERF.md section 15). A miss falls back to `get`.
   */
  peek(kernelName: string): BuiltPipeline | undefined {
    return this.pipelines.get(kernelName);
  }

  async get(kernelName: string, bindGroupLayout: GPUBindGroupLayout): Promise<BuiltPipeline> {
    const cached = this.pipelines.get(kernelName);
    if (cached) return cached;
    const kernel = kernelByName(kernelName);
    if (!kernel) {
      throw new Error(`PipelineStore: ${kernelName} is not in the kernel registry, so it cannot ship`);
    }
    const code = wgslForKernel(kernel, this.variant);
    const { pipeline, error } = await createComputePipelineChecked(this.gpu.device, {
      label: `${kernel.name}:${this.variant}`,
      layout: this.gpu.device.createPipelineLayout({
        label: `${kernel.name}:layout`,
        bindGroupLayouts: [bindGroupLayout],
      }),
      compute: { module: this.moduleFor(kernel.name, code), entryPoint: kernel.entry },
    });
    if (!pipeline) {
      const info = await this.moduleFor(kernel.name, code).getCompilationInfo();
      const diagnostics = info.messages.map(m => `${m.lineNum}:${m.linePos} ${m.message}`).join('\n');
      throw new Error(`PipelineStore: ${kernelName} failed to build under ${this.variant}: ${error ?? 'no message'}\n${diagnostics}`);
    }
    const built: BuiltPipeline = { kernel, variant: this.variant, pipeline };
    this.pipelines.set(kernelName, built);
    return built;
  }

  /**
   * `buildAll` is gone. Compiling ahead of the first token is still what `warmup()` does, but it
   * now goes through the executor's `prepare`, which is the only caller that can produce the bind
   * group layout `get` needs. Compiling a set of names here would have to invent layouts, and a
   * pipeline built against an invented layout is a pipeline the scheduler cannot bind to.
   */
  get builtCount(): number {
    return this.pipelines.size;
  }
}
