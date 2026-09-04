// Written from docs/ENGINE-PLAN.md sections 3, 5.5 and 7, the WebGPU specification, and the
// adapter facts recorded in ../../../../gemma4-kernels-lab/DECODE-CAMPAIGN.md and PREFILL-CAMPAIGN.md.
//
// SPDX-License-Identifier: Apache-2.0
//
// The device seam. Nothing in here knows about kernels, the harness page, or the model. The dev
// harness uses it today and the engine's loader will use the same function later, which is the
// point of it living beside the engine rather than inside the page.

/** Adapter identity, flattened, because `GPUAdapterInfo` is not a plain object. */
export interface Gemma4AdapterInfo {
  vendor: string;
  architecture: string;
  device: string;
  description: string;
  /**
   * The adapter's advertised subgroup width range, when the browser reports one. Reporting only:
   * a live device report should be able to say what the adapter advertised, but per ENGINE-PLAN
   * 5.5 rule 3 no code path may branch on it, because the range is advisory. PREFILL-CAMPAIGN.md
   * bisect round 1 measured a subgroup_size builtin of 32 on a part that advertised a maximum
   * of 128.
   */
  subgroupMinSize?: number;
  subgroupMaxSize?: number;
}

/** The limits every kernel is designed against. Read them, do not assume them. */
export interface Gemma4DeviceLimits {
  maxStorageBuffersPerShaderStage: number;
  maxStorageBufferBindingSize: number;
  maxBufferSize: number;
  maxComputeWorkgroupStorageSize: number;
  maxComputeInvocationsPerWorkgroup: number;
  maxComputeWorkgroupSizeX: number;
  maxComputeWorkgroupSizeY: number;
  maxComputeWorkgroupSizeZ: number;
  maxComputeWorkgroupsPerDimension: number;
}

export interface Gemma4Device {
  adapter: GPUAdapter;
  device: GPUDevice;
  info: Gemma4AdapterInfo;
  /** Feature policy as it was actually granted, not as it was asked for. */
  features: {
    /** Always true. The plan makes f16 a precondition, so a device without it is never returned. */
    f16: boolean;
    /**
     * Whether the `subgroups` feature was requested and granted. False by default. A granted
     * feature is permission to compile a subgroup path, never permission to trust one: per
     * ENGINE-PLAN 5.5 rule 4 a subgroup kernel ships only behind a known answer self test at
     * init, with a subgroup free fallback when it fails.
     */
    subgroups: boolean;
    /**
     * Whether `timestamp-query` was requested and granted. False unless the caller asked, and only
     * the performance rig asks: a shipping engine has no use for GPU timestamps and a feature that
     * is not requested cannot cost a device request on an adapter that lacks it.
     */
    timestampQuery: boolean;
  };
  limits: Gemma4DeviceLimits;
  /** Resolves if the device is ever lost. Read it, do not await it on the load path. */
  lost: Promise<GPUDeviceLostInfo>;
  /**
   * Set when `lost` resolves, so a caller can ask SYNCHRONOUSLY whether the device has died.
   * A lost device silently accepts every submit and runs none of them, which is how an iPhone
   * came to report 3705 tok/s on empty output (ENGINE-PERF 28.6). Anything that reports a
   * measurement must read this first.
   */
  lostReason: string | null;
  /** Uncaptured GPU errors, newest last. A non empty list invalidates any number measured after it. */
  errors: string[];
  destroy(): void;
}

export interface Gemma4DeviceOptions {
  /**
   * Ask for the `subgroups` feature when the adapter offers it. Default false. Turning this on
   * does not turn on any subgroup code path, see `features.subgroups`.
   */
  subgroups?: boolean;
  /**
   * Ask for `timestamp-query` when the adapter offers it. Default false. Granted on the M1 the
   * performance campaign runs on; the dev rig reads per dispatch GPU time through it (runtime.ts
   * timing mode) and nothing in the forward path depends on it.
   */
  timestamps?: boolean;
  label?: string;
}

export class Gemma4DeviceError extends Error {
  readonly reason: 'no-webgpu' | 'no-adapter' | 'no-f16' | 'no-device' | 'device-lost';
  constructor(reason: Gemma4DeviceError['reason'], message: string) {
    super(message);
    this.name = 'Gemma4DeviceError';
    this.reason = reason;
  }
}

/**
 * The nine compute limits a standalone device has to raise to the adapter maximum before it can
 * compile the same kernels the engine compiles.
 *
 * This is not a precaution. PREFILL-CAMPAIGN.md round 2b records a standalone A/B device that ran
 * on default limits and took two pipeline creation errors on a single QKV projection kernel, which
 * would have been read as a kernel bug rather than as a device configuration bug. Raise them once,
 * here, and no kernel author ever meets that failure.
 */
const REQUESTED_LIMITS = [
  'maxStorageBuffersPerShaderStage',
  'maxStorageBufferBindingSize',
  'maxBufferSize',
  'maxComputeWorkgroupStorageSize',
  'maxComputeInvocationsPerWorkgroup',
  'maxComputeWorkgroupSizeX',
  'maxComputeWorkgroupSizeY',
  'maxComputeWorkgroupSizeZ',
  'maxComputeWorkgroupsPerDimension',
] as const;

function readLimits(source: GPUSupportedLimits): Gemma4DeviceLimits {
  const bag = source as unknown as Record<string, number>;
  const out = {} as Record<string, number>;
  for (const name of REQUESTED_LIMITS) out[name] = bag[name] ?? 0;
  return out as unknown as Gemma4DeviceLimits;
}

function readInfo(adapter: GPUAdapter): Gemma4AdapterInfo {
  const info = (adapter as unknown as { info?: Partial<Gemma4AdapterInfo> }).info;
  const out: Gemma4AdapterInfo = {
    vendor: info?.vendor ?? '',
    architecture: info?.architecture ?? '',
    device: info?.device ?? '',
    description: info?.description ?? '',
  };
  if (typeof info?.subgroupMinSize === 'number') out.subgroupMinSize = info.subgroupMinSize;
  if (typeof info?.subgroupMaxSize === 'number') out.subgroupMaxSize = info.subgroupMaxSize;
  return out;
}

/**
 * The adapter's own facts, with no device created and nothing allocated.
 *
 * This exists so the loader page can size the IndexedDB chunk ceiling from the device profile
 * this machine actually matches without becoming a second place that asks for a GPU. There is one
 * device seam in this engine and scripts/engine-check/integration.mjs asserts it by grepping for
 * the request calls, which is the right shape of check and is why this function lives here rather
 * than beside its one caller.
 *
 * Returns null when there is no WebGPU at all or no adapter, because a caller that only wants
 * limits has somewhere sensible to go from there and an exception would be the wrong shape.
 */
export async function readAdapterFacts(): Promise<{ info: Gemma4AdapterInfo; limits: Gemma4DeviceLimits } | null> {
  const gpu = (globalThis.navigator as unknown as { gpu?: GPU } | undefined)?.gpu;
  if (!gpu) return null;
  const adapter = await gpu.requestAdapter();
  if (!adapter) return null;
  return { info: readInfo(adapter), limits: readLimits(adapter.limits) };
}

/**
 * Request the device the engine runs on.
 *
 * Feature policy, straight from ENGINE-PLAN section 2 and 5.5:
 *
 *  - `shader-f16` is required, not preferred. `checkWebGpu()` in src/engine/llm/client.ts already
 *    refuses before a single byte is downloaded when it is missing, so an engine that reaches this
 *    function without it is in a state nothing downstream is written for. Throw rather than
 *    silently compile an f32 engine that is four times slower and not the thing that was measured.
 *  - `subgroups` is off unless asked for, and asking for it grants nothing but the right to
 *    compile. There is deliberately no `sgExact32` style branch on the adapter reported subgroup
 *    width anywhere in this engine (ENGINE-PLAN 5.5 rule 3), and the reported range is advisory in
 *    any case: PREFILL-CAMPAIGN.md bisect round 1 measured a `subgroup_size` builtin of 32 in every
 *    configuration on a Blackwell part whose adapter advertised a maximum of 128.
 *  - The nine compute limits above go to the adapter maximum.
 */
export async function requestGemma4Device(options: Gemma4DeviceOptions = {}): Promise<Gemma4Device> {
  const gpu = navigator.gpu;
  if (!gpu) {
    throw new Gemma4DeviceError('no-webgpu', 'This browser does not expose WebGPU.');
  }
  const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) {
    throw new Gemma4DeviceError('no-adapter', 'No GPU adapter was offered.');
  }
  if (!adapter.features.has('shader-f16')) {
    throw new Gemma4DeviceError(
      'no-f16',
      'This adapter does not support shader-f16, which this engine requires.',
    );
  }

  const wantSubgroups = options.subgroups === true && adapter.features.has('subgroups');
  const requiredFeatures: GPUFeatureName[] = ['shader-f16'];
  if (wantSubgroups) requiredFeatures.push('subgroups' as GPUFeatureName);
  const wantTimestamps = options.timestamps === true && adapter.features.has('timestamp-query' as GPUFeatureName);
  if (wantTimestamps) requiredFeatures.push('timestamp-query' as GPUFeatureName);

  const adapterLimits = readLimits(adapter.limits);
  const requiredLimits: Record<string, number> = {};
  for (const name of REQUESTED_LIMITS) {
    const value = (adapterLimits as unknown as Record<string, number>)[name];
    if (value > 0) requiredLimits[name] = value;
  }

  let device: GPUDevice;
  try {
    device = await adapter.requestDevice({
      label: options.label ?? 'gemma4',
      requiredFeatures,
      requiredLimits,
    });
  } catch (err) {
    throw new Gemma4DeviceError('no-device', `The adapter refused a device: ${String(err)}`);
  }

  const errors: string[] = [];
  device.addEventListener('uncapturederror', (event) => {
    const detail = (event as GPUUncapturedErrorEvent).error;
    errors.push(`${detail.constructor.name}: ${detail.message}`);
  });

  const resolved: Gemma4Device = {
    adapter,
    device,
    info: readInfo(adapter),
    features: {
      f16: true,
      subgroups: device.features.has('subgroups' as GPUFeatureName),
      timestampQuery: device.features.has('timestamp-query' as GPUFeatureName),
    },
    limits: readLimits(device.limits),
    lost: device.lost,
    lostReason: null,
    errors,
    destroy: () => device.destroy(),
  };
  // Recorded rather than awaited: the load path must not block on a promise that resolves only on
  // failure, but everything downstream needs to be able to ask without awaiting.
  void device.lost.then((info) => {
    resolved.lostReason = `${info.reason ?? 'unknown'}: ${info.message || 'no message'}`;
  }).catch(() => { resolved.lostReason = 'unknown: the lost promise rejected'; });
  return resolved;
}

export interface PipelineResult {
  pipeline: GPUComputePipeline | null;
  error: string | null;
}

/**
 * Create a compute pipeline and actually find out whether it worked.
 *
 * Pipeline creation errors are asynchronous. `try`/`catch` never sees them, and the uncaptured
 * error handler logs a `GPUValidationError` with no message attached, which is how an afternoon
 * gets spent on a kernel that never compiled (DECODE-FUSION-FINDING.md, debug history item 1, and
 * ENGINE-PLAN 5.5 rule 7). Wrap creation in an error scope, return the failure as a value, and let
 * the caller treat it as a fallback rather than a crash.
 */
export async function createComputePipelineChecked(
  device: GPUDevice,
  descriptor: GPUComputePipelineDescriptor,
): Promise<PipelineResult> {
  device.pushErrorScope('validation');
  device.pushErrorScope('internal');
  let pipeline: GPUComputePipeline | null = null;
  let thrown: string | null = null;
  try {
    pipeline = device.createComputePipeline(descriptor);
  } catch (err) {
    thrown = String(err);
  }
  const internal = await device.popErrorScope();
  const validation = await device.popErrorScope();
  const error = thrown ?? internal?.message ?? validation?.message ?? null;
  if (error !== null) return { pipeline: null, error };
  return { pipeline, error: null };
}

/** Run `body` inside validation, internal and out of memory scopes. Returns the first error seen. */
export async function withErrorScopes<T>(
  device: GPUDevice,
  body: () => T | Promise<T>,
): Promise<{ value: T | null; error: string | null }> {
  device.pushErrorScope('out-of-memory');
  device.pushErrorScope('validation');
  device.pushErrorScope('internal');
  let value: T | null = null;
  let thrown: string | null = null;
  try {
    value = await body();
  } catch (err) {
    thrown = String(err);
  }
  const internal = await device.popErrorScope();
  const validation = await device.popErrorScope();
  const oom = await device.popErrorScope();
  const error = thrown ?? internal?.message ?? validation?.message ?? oom?.message ?? null;
  return { value, error };
}
