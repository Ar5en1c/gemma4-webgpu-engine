// Written from docs/ENGINE-PLAN.md sections 3, 5, 5.5 and 6, the resolver in ./execute.ts, the
// binding seam in ./kernels/binding.ts and ./kernels/registry.ts, the KV layout in ./kv.ts, and the
// WebGPU specification. No vendored bundle, no extracted kernel and no third party engine source
// was read.
//
// SPDX-License-Identifier: Apache-2.0
//
// The impure half of the forward. `execute.ts` says what every planned step binds and dispatches;
// this file owns the buffers, builds the pipelines, encodes the command buffer and reads back
// exactly one thing, the argmax token, at the end of a step that has a head.
//
// Three properties are deliberate and each of them is a decision somebody could reverse by
// accident, so they are stated here:
//
//   1. One readback per token, and it is eight bytes. Argmax runs on the GPU (ENGINE-PLAN K15) so
//      the next step's ids can be written from a number the host already has, rather than the host
//      mapping a 262,144 element logit buffer to find a maximum a workgroup already found.
//   2. Pipelines are built before anything is encoded. Pipeline creation is asynchronous and its
//      failures do not reach a try/catch (ENGINE-PLAN 5.5 rule 7), so it cannot sit inside the
//      encode loop; `prepare` does it once per kernel per device and `warmup()` calls it early so
//      the first real token is not the slow one.
//   3. The uniform arena is written for a whole chunk and reset only after that chunk's submitted
//      work is done. Resetting inside a submission would let a `writeBuffer` land on bytes a queued
//      dispatch still reads, which is exactly the class of bug that reads as a wrong number rather
//      than as an error.

import { pleGateSplits } from './kernels/pleSplit';
import { producerDownSplits } from './kernels/exactSplit';
import { BufferManager, type BufferLike, type DeviceLike } from './buffers';
import {
  ACTIVATION_BYTES,
  PendingKernelError,
  kvLengthAfter,
  resolveStep,
  slotElements,
  type BufferRef,
  type ForwardGeometry,
  type ResolvedStep,
} from './execute';
import { GEMV_WIDE_COLS } from './kernels/qgemvWide';
import type { Gemma4Device } from './device';
import type { PipelineStore } from './pipeline';
import { createKvLayout, kvRoleForLayer, type KvLayout } from './kv';
import {
  GEMMA4_E2B,
  planDecodeStep,
  planPrefillChunk,
  type DispatchStep,
  type GatherSliceRef,
  type Gemma4Arch,
} from './plan';
import { bindingBuffer, bindingEntry } from './kernels/binding';
import { kernelByName, type KernelBindResult } from './kernels/registry';
import { ROPE_GLOBAL, ROPE_SLIDING, ropeTables } from './kernels/rope';
import { gemvGeometry, kSplitsOf } from './kernels/qgemv';
import { attentionGeometry, kvSplitsOf } from './kernels/attention';

/**
 * The seam between the executor and whatever produced the weights. The engine's loader implements
 * it over `BufferManager`; the dev harness implements it over a handful of tensors sliced out of
 * the pinned snapshot, which is how a truncated model runs the same code path as a whole one.
 */
export interface ExecutorResources {
  /** A resident weight tensor, packed, by the name the checkpoint gives it. */
  weight(name: string): GPUBuffer;
  /** True when the tensor is resident, so a caller can report a missing one in its own words. */
  hasWeight(name: string): boolean;
  /** A four byte checkpoint scalar by name, or 0 when the checkpoint does not carry it. */
  scalar(name: string): number;
}

/** The seam the engine's greedy loop drives. Implemented here and by the dry recorder. */
export interface ForwardExecutor {
  prefill(tokens: readonly number[], startPosition: number, isLast: boolean): Promise<number>;
  decode(prevToken: number, position: number): Promise<number>;
  /**
   * Optional: run the decode step at `position` whose input token is the one the GPU already
   * holds from the previous step, without waiting for that token to reach the host. An executor
   * that offers this lets the greedy loop keep one step in flight (plan.ts runGreedyLoop).
   */
  decodeAhead?(position: number): Promise<number>;
  /**
   * Optional: run one wide pass over `tokens` placed at `position` onward and answer the model's
   * argmax after each of them. An executor that offers this can be driven by the speculative loop
   * (draft.ts runSpeculativeLoop); the dry recorder does not offer it and runs greedy.
   */
  verify?(tokens: readonly number[], position: number): Promise<number[]>;
}

export interface GpuExecutorOptions {
  /**
   * False keeps the six separate attention prologue kernels on a decode step instead of the
   * fused two (kernels/attnPrologue.ts), for a control run. Default true.
   */
  attentionPrologue?: boolean;
  batchProjections?: boolean;
  singleTokenGemm?: boolean;
  readonly prefillGemm4Tile?: 4 | 8;
  readonly prefillDenseTile?: 1 | 4;
  /**
   * On a one token step the PLE projection reads its layer's row in place out of the per layer
   * input, through an offset in its params, instead of copying the row into a slot of its own
   * first: 35 buffer copies a token gone, and each copy closed the compute pass around itself.
   * Default true; false keeps the copies, for a control run. Token identical either way
   * (lab-results/5070-attention-prologue-and-queue-sep05.json, parity r6).
   */
  directPle?: boolean;
  gpu: Gemma4Device;
  pipelines: PipelineStore;
  buffers: BufferManager;
  resources: ExecutorResources;
  arch?: Gemma4Arch;
  kvLayout?: KvLayout;
  /** Longest prefill chunk this executor will be asked to run. Slots are sized from it. */
  maxChunkTokens: number;
  /** Aborted between chunks and between tokens, which is the contract's one token granularity. */
  isAborted?: () => boolean;
  /**
   * Vocabulary ranges of the PLE table, when this adapter's buffers are too small to hold it whole
   * and engine.ts has planned a split. Absent or of length one means the whole table is one buffer,
   * which is what every adapter with a limit above 1,174,405,120 bytes reports, and then the plans
   * built below are byte for byte the ones this engine has always built.
   */
  pleSlices?: readonly GatherSliceRef[];
}

/** Bytes of uniform arena. 731 steps at the 256 byte alignment is about 187 KB; this is room. */
const ARENA_BYTES = 1 << 20;

/**
 * One dispatch's GPU time, from the timestamp pair written around its own compute pass. The step
 * identity travels with the number so a reader can aggregate by kernel, by role or by phase
 * without a second lookup into the plan.
 */
export interface StepTiming {
  kernel: string;
  role: string;
  phase: string;
  layer: number;
  /** GPU nanoseconds between the pass's beginning and end timestamps, as the browser reports them. */
  ns: number;
}

/**
 * What one timed `runSteps` measured. `wallMs` is the host's own clock from just before the submit
 * to the readback landing, so the two totals can be read against each other: the GPU sum says how
 * long the dispatches ran, the wall clock says how long the token took, and the gap is everything
 * that is not a dispatch (encode, submission, inter pass gaps, the readback hop).
 */
export interface RunTiming {
  mode: 'gemv' | 'gemm';
  tokens: number;
  startPosition: number;
  steps: StepTiming[];
  totalNs: number;
  wallMs: number;
}

/**
 * The spec's ceiling on a query set. A decode token is 731 dispatches and a prefill chunk 736 with
 * the head, so two timestamps each is under half of it; the ceiling is checked rather than assumed
 * because a plan that grew past it would fail inside createQuerySet with a validation error that
 * never reaches a try/catch.
 */
const MAX_QUERY_COUNT = 4096;

/** One dispatch of a pass, bound and ready to encode. */
interface Encoded {
  resolved: ResolvedStep;
  pipeline: GPUComputePipeline;
  bindGroup: GPUBindGroup;
  dispatch: readonly [number, number, number];
  copies: { src: GPUBuffer; srcOffset: number; dst: GPUBuffer; dstOffset: number; bytes: number }[];
}

/** A cached decode step whose params block carries the position: restaged every token where it was. */
interface DynamicStep {
  index: number;
  offset: number;
  size: number;
}

/** One recorded decode token, see the decode step cache in the executor. */
interface DecodePlanCache {
  steps: number;
  generation: number;
  encoded: Encoded[];
  dynamic: DynamicStep[];
  /** Residual pair names to flip, in plan order, one per flipping step. */
  flips: string[];
}

export class GpuExecutor implements ForwardExecutor {
  private readonly attentionPrologue: boolean;
  private readonly batchProjections: boolean;
  private readonly singleTokenGemm: boolean;
  private readonly prefillGemm4Tile: 4 | 8;
  private readonly prefillDenseTile: 1 | 4;
  private readonly directPle: boolean;
  readonly arch: Gemma4Arch;
  readonly kvLayout: KvLayout;
  private readonly gpu: Gemma4Device;
  private readonly pipelines: PipelineStore;
  private readonly buffers: BufferManager;
  /** Empty on every adapter that fits the PLE table in one buffer. See GpuExecutorOptions. */
  private readonly pleSlices: readonly GatherSliceRef[];
  private readonly resources: ExecutorResources;
  private readonly maxChunkTokens: number;
  private readonly isAborted: () => boolean;
  private readonly layouts = new Map<string, GPUBindGroupLayout>();
  private readonly prepared = new Set<string>();
  private readonly ropeUploaded = new Map<string, number>();
  private arenaReady = false;
  /** Dispatches actually encoded, which the harness reports so a run cannot be silently empty. */
  dispatchesEncoded = 0;
  /** Buffer to buffer slice copies encoded, the two sites execute.ts's SliceCopy covers. */
  copiesEncoded = 0;

  // ------------------------------------------------------------------- timing mode
  //
  // The performance rig's instrument. When it is on, every dispatch is encoded in a compute pass
  // of its own with a timestamp written at the pass's beginning and end, the query set is resolved
  // into a buffer at the end of the command buffer, and the pairs are read back beside the token.
  // That is the only way WebGPU exposes per dispatch GPU time: `timestampWrites` is a property of
  // a pass, not of a dispatch, so per kernel numbers cost one pass per kernel. Off, the executor
  // encodes one pass per submission exactly as it always has, and the tokens per second the
  // campaign reports are measured with it off. The timed run's own wall clock is recorded with the
  // GPU sum so the cost of the instrument is visible rather than assumed.
  private timing = false;
  /**
   * Coarse timing: one timestamp pair per compute PASS instead of one per dispatch, so the number
   * is the token's GPU time as the loop pays it, drains between dependent dispatches included,
   * and the per kernel table is empty. Per dispatch timing gives every dispatch its own pass and
   * so adds a pass boundary to each; the difference between the two readings is that overhead.
   */
  private timingCoarse = false;
  private querySet: GPUQuerySet | null = null;
  private queryResolve: GPUBuffer | null = null;
  private queryStaging: GPUBuffer | null = null;
  private queryCapacity = 0;
  /** Every timed `runSteps` since timing was turned on, in order. Cleared by `takeTimings`. */
  private timings: RunTiming[] = [];

  /**
   * Turn per dispatch GPU timing on or off. Returns the state actually reached: on is refused when
   * the device was not created with the `timestamp-query` feature (device.ts, `timestamps`).
   */
  setTiming(on: boolean, mode: 'dispatch' | 'coarse' = 'dispatch'): boolean {
    if (on && !this.gpu.features.timestampQuery) {
      this.timing = false;
      this.timingCoarse = false;
      return false;
    }
    this.timing = on;
    this.timingCoarse = on && mode === 'coarse';
    return this.timing;
  }

  get timingEnabled(): boolean {
    return this.timing;
  }

  /** Hand back every timing recorded since the last call and forget them. */
  takeTimings(): RunTiming[] {
    const out = this.timings;
    this.timings = [];
    return out;
  }

  private ensureQueries(count: number): GPUQuerySet {
    if (count > MAX_QUERY_COUNT) {
      throw new Error(
        `gemma4 engine: timing needs ${count} timestamps for one submission, over the ${MAX_QUERY_COUNT} `
        + 'a query set may hold. Split the step list or time fewer dispatches.',
      );
    }
    if (this.querySet && this.queryCapacity >= count) return this.querySet;
    this.querySet?.destroy();
    this.queryResolve?.destroy();
    this.queryStaging?.destroy();
    // Sized to the ceiling once rather than regrown per step list: a decode token and a prefill
    // chunk ask for different counts and reallocating between them would be garbage on the loop.
    const capacity = MAX_QUERY_COUNT;
    this.querySet = this.device.createQuerySet({ label: 'gemma4:timestamps', type: 'timestamp', count: capacity });
    this.queryResolve = this.device.createBuffer({
      label: 'gemma4:timestamps:resolve',
      size: capacity * 8,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    this.queryStaging = this.device.createBuffer({
      label: 'gemma4:timestamps:staging',
      size: capacity * 8,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    this.queryCapacity = capacity;
    return this.querySet;
  }

  constructor(options: GpuExecutorOptions) {
    this.attentionPrologue = options.attentionPrologue ?? true;
    this.batchProjections = options.batchProjections ?? false;
    this.singleTokenGemm = options.singleTokenGemm ?? false;
    this.prefillGemm4Tile = options.prefillGemm4Tile ?? 8;
    this.prefillDenseTile = options.prefillDenseTile ?? 1;
    this.directPle = options.directPle ?? true;
    this.gpu = options.gpu;
    this.pipelines = options.pipelines;
    this.buffers = options.buffers;
    this.pleSlices = options.pleSlices ?? [];
    this.resources = options.resources;
    this.arch = options.arch ?? GEMMA4_E2B;
    this.kvLayout = options.kvLayout ?? createKvLayout();
    this.maxChunkTokens = Math.max(1, options.maxChunkTokens);
    this.isAborted = options.isAborted ?? (() => false);
  }

  // --------------------------------------------------------------------------- resources

  private get device(): GPUDevice {
    return this.gpu.device;
  }

  private ensureArena(): void {
    if (this.arenaReady) return;
    this.buffers.createUniformArena(ARENA_BYTES);
    this.arenaReady = true;
  }

  /**
   * Allocate every activation slot at the widest geometry, once.
   *
   * Idempotent, and it has to be, because it is the fix for a real ordering hazard rather than a
   * tidy-up. Every slot is sized here from the architecture's maxima at `maxChunkTokens`, and
   * `BufferManager.storage` refuses a later larger ask as a plan bug rather than resizing. So any
   * path that touches a slot before this has run creates that slot at whatever width it happened to
   * want first, and the next wider ask throws. `stageInputs` is exactly such a path: it uploads the
   * rope tables for the positions in play, and a sixteen position chunk sizes `rope.cos.sliding` at
   * a quarter of what a sixty four position chunk needs. engine.ts never met this because
   * `generate()` awaits `warmup()`, which calls `prepare`, which calls this first; a caller driving
   * the executor directly did meet it. Every entry point now calls this before touching a slot.
   */
  allocateSlots(): void {
    if (this.slotsReady) return;
    const sizes = slotElements(this.arch, this.maxChunkTokens);
    for (const [key, elements] of Object.entries(sizes)) {
      this.buffers.storage(key, Math.max(4, elements * 4));
    }
    for (const cache of this.kvLayout.caches) {
      this.buffers.storage(`kv:${cache.layer}`, cache.bytes);
    }
    // The residual ping-pong pair, at the widest chunk, for the same reason as the slots above and
    // caught the same way: it is created on first touch, and a sixteen token forward followed by a
    // sixty four token one sized it at a quarter of what the second needed and threw. Nothing else
    // allocates it, because resolveRef asks for it at whatever width the step in hand has.
    this.buffers.writeSide('hidden', this.maxChunkTokens * this.arch.hiddenSize * ACTIVATION_BYTES);
    this.ensureArena();
    this.slotsReady = true;
  }

  private slotsReady = false;

  private slotBuffer(key: string, elements: number): GPUBuffer {
    return this.buffers.storage(key, Math.max(4, elements * 4)) as unknown as GPUBuffer;
  }

  private resolveRef(ref: BufferRef): GPUBuffer {
    switch (ref.kind) {
      case 'weight':
        return this.resources.weight(ref.name);
      case 'slot':
        return this.slotBuffer(ref.name, ref.elements);
      case 'residual-read':
        return this.buffers.readSide(ref.name, Math.max(4, ref.elements * 4)) as unknown as GPUBuffer;
      case 'residual-write':
        return this.buffers.writeSide(ref.name, Math.max(4, ref.elements * 4)) as unknown as GPUBuffer;
      case 'kv': {
        const layer = Number(ref.name);
        const desc = this.kvLayout.caches[layer];
        if (!desc) throw new Error(`gemma4 engine: layer ${layer} owns no KV cache`);
        return this.buffers.storage(`kv:${layer}`, desc.bytes) as unknown as GPUBuffer;
      }
      default:
        throw new Error(`gemma4 engine: unknown buffer ref kind ${(ref as BufferRef).kind}`);
    }
  }

  // -------------------------------------------------------------------------- preparation

  /**
   * Build every pipeline a step list needs, and check each kernel's storage binding count against
   * the adapter's limit while a representative bind group layout is in hand. Idempotent, so the
   * decode set built by warmup() is not rebuilt per token.
   */
  async prepare(steps: readonly DispatchStep[], geometry: ForwardGeometry): Promise<void> {
    this.allocateSlots();
    // A step list already prepared under this shape is skipped whole: resolving 626 steps a token
    // only to find every kernel prepared was 626 resolves a token, and the arena discard at the
    // end retired the decode step cache every token (docs/ENGINE-PERF.md section 15).
    const shape = `${geometry.mode}|${geometry.tokens}|${steps.length}|${geometry.idsFromTokenSlot ? 'slot' : 'ids'}`;
    if (this.preparedShapes.has(shape)) return;
    for (const step of steps) {
      let resolved: ResolvedStep;
      try {
        resolved = resolveStep(step, geometry);
      } catch (err) {
        if (err instanceof PendingKernelError) continue;
        throw err;
      }
      // Already prepared kernels are skipped outright rather than rebound and re-checked. This is
      // what makes warmup() mean something: after it, a decode token does no pipeline work and no
      // binding budget arithmetic, it only binds the steps it is about to encode.
      if (this.prepared.has(resolved.kernel)) continue;
      this.prepared.add(resolved.kernel);
      const bound = this.bindKernel(resolved);
      const storage = bound.buffers.filter(
        (b) => (bindingBuffer(b).usage & GPUBufferUsage.STORAGE) === GPUBufferUsage.STORAGE,
      ).length;
      const limit = this.gpu.limits.maxStorageBuffersPerShaderStage;
      if (storage > limit) {
        throw new Error(
          `gemma4 engine: ${resolved.kernel} binds ${storage} storage buffers against an adapter `
          + `limit of ${limit} (ENGINE-PLAN risk 6, DECODE-CAMPAIGN.md 4.7)`,
        );
      }
      await this.pipelines.get(resolved.kernel, bound.layout);
    }
    // Nothing was encoded, so the params the binds staged are dropped rather than flushed.
    this.buffers.discardStagedUniforms();
    this.preparedShapes.add(shape);
  }

  /** Step list shapes prepare() has already walked, see prepare. */
  private readonly preparedShapes = new Set<string>();

  /** The arena slice the most recent bindKernel staged, for the decode step cache to record. */
  private lastStaged: { offset: number; size: number } | null = null;

  private bindKernel(resolved: ResolvedStep, restageAt?: number): KernelBindResult {
    const kernel = kernelByName(resolved.kernel);
    if (!kernel) {
      throw new Error(
        `gemma4 engine: step "${resolved.step.role}" wants kernel ${resolved.kernel}, which is not `
        + 'in the registry. A kernel that is not registered cannot be proved, so it does not ship.',
      );
    }
    const inputs: Record<string, GPUBuffer> = {};
    for (const [name, ref] of Object.entries(resolved.inputs)) inputs[name] = this.resolveRef(ref);
    const bound = kernel.bind({
      device: this.device,
      inputs,
      output: this.resolveRef(resolved.output),
      params: resolved.params,
      stageUniform: (data) => {
        const view = ArrayBuffer.isView(data) ? data : new Uint8Array(data);
        const slice = restageAt === undefined
          ? this.buffers.stageUniform(view)
          : this.buffers.restageUniform(restageAt, view);
        this.lastStaged = { offset: slice.offset, size: slice.size };
        return slice as unknown as { buffer: GPUBuffer; offset: number; size: number };
      },
      layoutFor: (key, make) => {
        let layout = this.layouts.get(key);
        if (!layout) {
          layout = make();
          this.layouts.set(key, layout);
        }
        return layout;
      },
    });
    return bound;
  }

  // ------------------------------------------------------------------------------ the run

  private uploadIds(tokens: readonly number[]): void {
    const ids = new Int32Array(this.maxChunkTokens);
    for (let i = 0; i < tokens.length; i += 1) ids[i] = tokens[i]!;
    const buffer = this.slotBuffer('ids', this.maxChunkTokens);
    this.device.queue.writeBuffer(buffer, 0, ids);
  }

  /**
   * Upload the cosine and sine tables for a run of positions, once per distinct run. Two pairs,
   * because the sliding and global layers rotate different numbers of pairs at different thetas
   * and over different head dimensions; rope.ts owns the frequency ladder and this only places
   * what it returns.
   */
  private uploadRope(startPosition: number, tokens: number): void {
    const cacheKey = `${startPosition}:${tokens}`;
    if (this.ropeCacheKey === cacheKey) return;
    const positions: number[] = [];
    for (let i = 0; i < tokens; i += 1) positions.push(startPosition + i);
    for (const [kind, spec] of [['sliding', ROPE_SLIDING], ['global', ROPE_GLOBAL]] as const) {
      const { cos, sin } = ropeTables(spec, positions);
      this.device.queue.writeBuffer(this.slotBuffer(`rope.cos.${kind}`, cos.length), 0, cos);
      this.device.queue.writeBuffer(this.slotBuffer(`rope.sin.${kind}`, sin.length), 0, sin);
    }
    this.ropeCacheKey = cacheKey;
  }

  private ropeCacheKey = '';

  private geometryFor(mode: 'gemv' | 'gemm', tokens: number, startPosition: number): ForwardGeometry {
    return {
      directPle: this.directPle,
      singleTokenGemm: this.singleTokenGemm,
      prefillGemm4Tile: this.prefillGemm4Tile,
      prefillDenseTile: this.prefillDenseTile,
      arch: this.arch,
      mode,
      tokens,
      startPosition,
      maxContext: this.kvLayout.maxContext,
      scalar: (name) => this.resources.scalar(name),
    };
  }

  // ------------------------------------------------------------------- the host pipeline
  //
  // Round 3's timing table put a decode token at 41.5 ms of GPU time inside 55 ms of wall clock:
  // a quarter of every token was the host binding and encoding 731 dispatches, submitting, and
  // waiting on an eight byte readback before it could start the next one. The GPU sat idle for
  // that quarter. The fix is not to encode faster but to encode EARLIER: the next step's input
  // token is already on the GPU (the argmax wrote it to the token slot), so with the gathers
  // reading that slot (execute.ts idsFromTokenSlot) the next step can be bound, encoded and
  // submitted while the current one runs, and the readback only has to arrive before the host
  // needs the token for text and stop detection. `decodeAhead` is that entry point and
  // `runGreedyLoop` keeps one step in flight.
  //
  // Two things make it safe. The host side of a step (arena reset, uniform staging, residual
  // flips, bind groups, encode, submit) runs to completion under `encodeChain` before the next
  // step's host side starts, so two steps never interleave their arena slices. And every write
  // the host issues for step N+1, the uniform arena and the rope tables included, goes through
  // queue.writeBuffer, which the queue orders after step N's submitted command buffer, so a
  // slice step N still reads is never overwritten underneath it. The readback buffers are a
  // pool, one per step in flight, because a buffer with a pending map cannot be a copy target.
  private encodeChain: Promise<unknown> = Promise.resolve();
  private readonly readbackPool: GPUBuffer[] = [];
  /** The last timed step's readback; timing mode owns one query staging buffer, so it serializes. */
  private lastTimed: Promise<void> = Promise.resolve();
  /** Host milliseconds spent binding, encoding and submitting, summed over `submissions`. */
  hostEncodeMs = 0;
  submissions = 0;
  /**
   * The same host time split by phase, summed over `submissions`: resolving and binding the
   * steps (resolveStep, kernel.bind, uniform staging), the pipeline lookups, the bind group
   * cache, and the command encoding with its submit. The performance page prints them; the
   * host encode path is a ruled lever (docs/ENGINE-PLAN.md, round 4 close, the block join).
   */
  hostPhaseMs = { bind: 0, pipeline: 0, bindGroup: 0, encode: 0 };
  /** Decode step cache traffic, for the performance page. */
  decodeCacheBuilds = 0;
  decodeCacheHits = 0;

  private acquireReadback(): GPUBuffer {
    const pooled = this.readbackPool.pop();
    if (pooled) return pooled;
    return this.device.createBuffer({
      label: 'gemma4:token-readback',
      size: 8 * GEMV_WIDE_COLS,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
  }

  /**
   * Encode and submit one chunk. `readToken` asks for the eight byte argmax readback at the end,
   * which only a step list carrying the head phase can answer. `stage` runs under the encode chain
   * just before binding, which is where a caller's per step uploads (ids, rope tables) belong once
   * more than one step can be in flight: issued any earlier they would land on the slot a queued
   * step has not read yet.
   */
  async runSteps(
    steps: readonly DispatchStep[],
    geometry: ForwardGeometry,
    readToken: boolean,
    stage?: () => void,
  ): Promise<number> {
    const submitted = this.encodeChain.then(() => this.encodeAndSubmit(steps, geometry, readToken, stage));
    this.encodeChain = submitted.then(() => undefined, () => undefined);
    const { finished } = await submitted;
    const pairs = await finished;
    return pairs ? pairs[0]! : -1;
  }

  /**
   * `runSteps` for a verify pass: the argmax of every position, in position order. A gemv mode
   * step over `tokens` positions writes one (index, value) pair per position to the token slot.
   */
  async runStepsWide(
    steps: readonly DispatchStep[],
    geometry: ForwardGeometry,
    stage?: () => void,
  ): Promise<number[]> {
    const submitted = this.encodeChain.then(() => this.encodeAndSubmit(steps, geometry, true, stage));
    this.encodeChain = submitted.then(() => undefined, () => undefined);
    const { finished } = await submitted;
    const pairs = await finished;
    if (!pairs) return [];
    const out: number[] = [];
    for (let i = 0; i < geometry.tokens; i += 1) out.push(pairs[2 * i]!);
    return out;
  }

  /**
   * The full host pass: resolve, bind and build a bind group for every step, in plan order,
   * flipping the residual pair as the plan says so bind group N+1 sees the halves N left behind.
   * Nothing is encoded here, which keeps pipeline creation out of the encode loop. With a cache
   * key the list is recorded for later tokens (see the decode step cache above).
   */
  private async buildEncoded(
    steps: readonly DispatchStep[],
    geometry: ForwardGeometry,
    lap: (phase: 'bind' | 'pipeline' | 'bindGroup' | 'encode') => void,
    cacheKey: string | null,
  ): Promise<Encoded[]> {
    this.buffers.resetUniformArena();
    const generation = this.buffers.arenaGeneration;
    const encoded: Encoded[] = [];
    const flips: string[] = [];
    const dynamic: DynamicStep[] = [];
    // The same steps one position on, to find the params that carry the position.
    const ahead: ForwardGeometry | null = cacheKey !== null
      ? { ...geometry, startPosition: geometry.startPosition + 1 }
      : null;
    for (let index = 0; index < steps.length; index += 1) {
      const step = steps[index]!;
      const resolved = resolveStep(step, geometry);
      this.lastStaged = null;
      const copies = resolved.copies.map((copy) => ({
        src: this.resolveRef(copy.from),
        srcOffset: copy.fromOffset * 4,
        dst: this.resolveRef(copy.to),
        dstOffset: copy.toOffset * 4,
        bytes: copy.elements * 4,
      }));
      const bound = this.bindKernel(resolved);
      // A dispatch dimension over the adapter's cap is a validation error on the command buffer,
      // which WebGPU reports on the device's uncapturederror event and nowhere a caller awaits:
      // the whole submission is dropped, the token slot keeps its stale value, and the step
      // "succeeds" in 126 ms with the wrong token. Round 4 met exactly that when the prefill chunk
      // went to 512 and scale-add over 512 times 8960 elements asked for 71,680 workgroups in one
      // dimension. Refused here, before encoding, with the kernel named.
      const cap = this.gpu.limits.maxComputeWorkgroupsPerDimension;
      for (const dim of bound.dispatch) {
        if (dim > cap || dim <= 0) {
          throw new Error(
            `gemma4 engine: ${resolved.kernel} (${resolved.step.role}) asks for a dispatch of `
            + `[${bound.dispatch.join(', ')}] against the adapter's ${cap} workgroups per dimension`,
          );
        }
      }
      lap('bind');
      // The synchronous lookup first: an await per step is a microtask and a promise each, 626
      // times a token, on the host's critical path. The await only happens on a store miss.
      const built = this.pipelines.peek(resolved.kernel) ?? await this.pipelines.get(resolved.kernel, bound.layout);
      lap('pipeline');
      const bindGroup = this.buffers.cachedBindGroup(
        `${resolved.kernel}|${resolved.step.role}`,
        bound.layout,
        bound.buffers.map((b, binding) => {
          const entry = bindingEntry(b);
          const spec: { binding: number; buffer: BufferLike; offset?: number; size?: number } = {
            binding,
            buffer: entry.buffer as unknown as BufferLike,
          };
          if (entry.offset !== undefined) spec.offset = entry.offset;
          if (entry.size !== undefined) spec.size = entry.size;
          return spec;
        }),
      ) as GPUBindGroup;
      encoded.push({ resolved, pipeline: built.pipeline, bindGroup, dispatch: bound.dispatch, copies });
      lap('bindGroup');
      if (resolved.flipsResidual) {
        // The residual write side is the output on the plain epilogues and an input on the fused
        // joins, whose output is the next block's normed row.
        const written = [resolved.output, ...Object.values(resolved.inputs)]
          .find((ref) => ref.kind === 'residual-write');
        if (!written) {
          throw new Error(`gemma4 engine: step "${resolved.step.role}" flips the residual but writes no residual side`);
        }
        this.buffers.flip(written.name);
        flips.push(written.name);
      }
      if (ahead) {
        const staged = this.lastStaged as { offset: number; size: number } | null;
        if (staged === null) {
          throw new Error(`gemma4 engine: decode step cache: ${resolved.kernel} (${resolved.step.role}) staged no params block`);
        }
        const next = resolveStep(step, ahead);
        if (JSON.stringify(next.params) !== JSON.stringify(resolved.params)) {
          dynamic.push({ index, offset: staged.offset, size: staged.size });
        }
      }
    }
    if (cacheKey !== null) {
      this.decodeCache.set(cacheKey, { steps: steps.length, generation, encoded, dynamic, flips });
      this.decodeCacheBuilds += 1;
    }
    return encoded;
  }

  // ------------------------------------------------------------------- the decode step cache
  //
  // A decode token is the same 626 dispatches every time: the same kernels, the same buffers (the
  // residual pair flips an even number of times a token, so it is back where it started), the
  // same dispatch sizes, and the same params blocks except the few that carry the position
  // (attention's kvLen, the KV store's slot). Resolving, binding and building bind groups for
  // all of them every token was 17 ms of host time in the performance pane (docs/ENGINE-PERF.md
  // section 15; the pane runs the page on efficiency cores, which is why every host microsecond
  // there counts five times). So the first token under a given key builds the list once and
  // records, for every step, the arena offset its params block took and whether that block
  // depends on the position (found by resolving the same step at the next position and comparing
  // params, not by naming kernels). Every later token restages only those blocks at their
  // recorded offsets, applies the residual flips the plan asks for, and encodes the recorded list.
  //
  // Invalidation is explicit. The key carries the plan's identity, the residual parity at the
  // start, the ids source and the timing mode; the entry carries the arena generation it was
  // built under, and an arena reset by any other pass (a prefill chunk) retires it, because that
  // pass overwrote the blocks it recorded. Buffers never move underneath it: slots never grow
  // silently, weights are resident for the model's life, the KV caches are fixed at load.
  private readonly decodeCache = new Map<string, DecodePlanCache>();

  private async encodeAndSubmit(
    steps: readonly DispatchStep[],
    geometry: ForwardGeometry,
    readToken: boolean,
    stage: (() => void) | undefined,
  ): Promise<{ finished: Promise<Int32Array | null> }> {
    await this.prepare(steps, geometry);
    if (this.timing) await this.lastTimed;
    const hostStart = performance.now();
    stage?.();

    const phases = this.hostPhaseMs;
    let phaseStart = performance.now();
    const lap = (phase: keyof typeof phases): void => {
      const now = performance.now();
      phases[phase] += now - phaseStart;
      phaseStart = now;
    };

    // A verify pass (gemv mode over a few positions) is cached under its own width, so a token
    // and a four position pass keep separate recorded lists.
    const cacheKey = geometry.mode === 'gemv'
      ? `${steps.length}|${geometry.tokens}|${geometry.idsFromTokenSlot ? 'slot' : 'ids'}|${this.buffers.pairParity('hidden')}|${this.timing ? 't' : 'u'}`
      : null;
    const hit = cacheKey !== null ? this.decodeCache.get(cacheKey) : undefined;
    let encoded: Encoded[];
    if (hit && hit.generation === this.buffers.arenaGeneration && hit.steps === steps.length) {
      // The cached token: restage the position dependent params where they were, flip as recorded.
      for (const d of hit.dynamic) {
        const resolved = resolveStep(steps[d.index]!, geometry);
        const bound = this.bindKernel(resolved, d.offset);
        if (this.lastStaged === null || this.lastStaged.offset !== d.offset || this.lastStaged.size !== d.size) {
          throw new Error(`gemma4 engine: decode step cache: ${resolved.kernel} (${resolved.step.role}) restaged a different block than it recorded`);
        }
        for (let i = 0; i < bound.dispatch.length; i += 1) {
          if (bound.dispatch[i] !== hit.encoded[d.index]!.dispatch[i]) {
            throw new Error(`gemma4 engine: decode step cache: ${resolved.kernel} (${resolved.step.role}) changed its dispatch with the position`);
          }
        }
        hit.encoded[d.index]!.resolved = resolved;
      }
      for (const name of hit.flips) this.buffers.flip(name);
      encoded = hit.encoded;
      this.decodeCacheHits += 1;
      lap('bind');
    } else {
      encoded = await this.buildEncoded(steps, geometry, lap, cacheKey);
    }

    const encoder = this.device.createCommandEncoder({ label: `gemma4:${geometry.mode}` });
    const timed = this.timing;
    const coarse = this.timingCoarse;
    const querySet = timed ? this.ensureQueries(encoded.length * 2) : null;
    let pass: GPUComputePassEncoder | null = null;
    // Coarse timing numbers the passes rather than the dispatches: a pass reopens only around a
    // buffer copy, so a token is a handful of pairs and the sum is its GPU time with every drain
    // between dependent dispatches inside it.
    let passCount = 0;
    const openPass = (index: number): GPUComputePassEncoder => {
      if (!pass) {
        const descriptor: GPUComputePassDescriptor = { label: `gemma4:${geometry.mode}` };
        if (querySet) {
          const pair = coarse ? passCount++ : index;
          descriptor.timestampWrites = {
            querySet,
            beginningOfPassWriteIndex: pair * 2,
            endOfPassWriteIndex: pair * 2 + 1,
          };
        }
        pass = encoder.beginComputePass(descriptor);
      }
      return pass;
    };
    const closePass = (): void => {
      if (pass) {
        pass.end();
        pass = null;
      }
    };
    for (let index = 0; index < encoded.length; index += 1) {
      const item = encoded[index]!;
      if (item.copies.length > 0) {
        // A buffer to buffer copy is a command, not a dispatch, so the pass closes around it. The
        // reopened pass is a new pass on the same encoder, which costs nothing measurable and
        // keeps the copy's ordering against the dispatch that reads it explicit.
        closePass();
        for (const copy of item.copies) {
          encoder.copyBufferToBuffer(copy.src, copy.srcOffset, copy.dst, copy.dstOffset, copy.bytes);
          this.copiesEncoded += 1;
        }
      }
      const active = openPass(index);
      active.setPipeline(item.pipeline);
      active.setBindGroup(0, item.bindGroup);
      active.dispatchWorkgroups(item.dispatch[0], item.dispatch[1], item.dispatch[2]);
      this.dispatchesEncoded += 1;
      // Timing mode: one pass per dispatch, because the timestamps belong to the pass.
      if (timed && !coarse) closePass();
    }
    closePass();

    const pairs = coarse ? passCount : encoded.length;
    if (querySet) {
      encoder.resolveQuerySet(querySet, 0, pairs * 2, this.queryResolve!, 0);
      encoder.copyBufferToBuffer(this.queryResolve!, 0, this.queryStaging!, 0, pairs * 16);
    }
    const wallStart = performance.now();
    const recordTiming = async (): Promise<void> => {
      if (!querySet) return;
      const staging = this.queryStaging!;
      await staging.mapAsync(GPUMapMode.READ, 0, pairs * 16);
      const stamps = new BigUint64Array(staging.getMappedRange(0, pairs * 16).slice(0));
      staging.unmap();
      const wallMs = performance.now() - wallStart;
      const steps: StepTiming[] = [];
      let totalNs = 0;
      for (let i = 0; i < pairs; i += 1) {
        const begin = stamps[i * 2]!;
        const end = stamps[i * 2 + 1]!;
        const ns = end >= begin ? Number(end - begin) : 0;
        if (!coarse) {
          const step = encoded[i]!.resolved.step;
          steps.push({ kernel: encoded[i]!.resolved.kernel, role: step.role, phase: step.phase, layer: step.layer, ns });
        }
        totalNs += ns;
      }
      this.timings.push({
        mode: geometry.mode, tokens: geometry.tokens, startPosition: geometry.startPosition, steps, totalNs, wallMs,
      });
    };

    const readback = readToken ? this.acquireReadback() : null;
    const readRows = geometry.mode === 'gemv' ? geometry.tokens : 1;
    if (readback) encoder.copyBufferToBuffer(this.slotBuffer('token', 2 * GEMV_WIDE_COLS), 0, readback, 0, 8 * readRows);
    // Every params block this pass staged, in one write, ordered by the queue ahead of the submit.
    this.buffers.flushUniformArena();
    this.device.queue.submit([encoder.finish()]);
    lap('encode');
    this.hostEncodeMs += performance.now() - hostStart;
    this.submissions += 1;

    // The wait starts now, outside the encode chain, so the next step's host side can begin while
    // this one runs. The readback buffer goes back to the pool only once it is unmapped.
    const finish = async (): Promise<Int32Array | null> => {
      if (!readback) {
        await this.device.queue.onSubmittedWorkDone();
        await recordTiming();
        return null;
      }
      await readback.mapAsync(GPUMapMode.READ);
      const pairs = new Int32Array(readback.getMappedRange().slice(0));
      readback.unmap();
      this.readbackPool.push(readback);
      await recordTiming();
      return pairs;
    };
    const finished = finish();
    if (querySet) this.lastTimed = finished.then(() => undefined, () => undefined);
    return { finished };
  }

  /**
   * The per chunk inputs every forward needs before its steps can run: the token ids the gathers
   * read, and the cosine and sine tables for the positions in play. `prefill` and `decode` call
   * this for themselves; a caller driving `runSteps` directly, which is what the truncated model
   * page does, has to call it too, and the reason it is public rather than folded into `runSteps`
   * is that a probe often wants to run a step list at a position its own ids did not produce.
   */
  stageInputs(tokenIds: readonly number[], startPosition: number): void {
    // Before anything, because both uploads below touch slots and a narrow first touch would size
    // them for the chunk in hand rather than for the widest one. See allocateSlots.
    this.allocateSlots();
    this.uploadIds(tokenIds);
    this.uploadRope(startPosition, Math.max(1, tokenIds.length));
  }

  /**
   * Seed one activation slot. Dev instrument only, and the reason it exists is worth stating: the
   * truncated model page proves a step kind by starting the forward from a reference capture
   * rather than from the thirty four layers that would otherwise have to be resident to produce
   * it. Nothing in the decode path calls this.
   */
  writeSlot(key: string, data: Float32Array): void {
    this.device.queue.writeBuffer(this.slotBuffer(key, data.length), 0, data);
  }

  /** Seed the half of the residual pair the next reader will read. Same dev-only argument. */
  writeResidual(data: Float32Array): void {
    const buffer = this.buffers.readSide('hidden', data.byteLength) as unknown as GPUBuffer;
    this.device.queue.writeBuffer(buffer, 0, data);
  }

  /** Read one activation slot back as f32. Debug instrument for the harness, not the decode path. */
  async readSlot(key: string, elements: number): Promise<Float32Array> {
    const source = this.slotBuffer(key, elements);
    const bytes = elements * 4;
    const staging = this.device.createBuffer({
      label: `gemma4:read:${key}`,
      size: bytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      const encoder = this.device.createCommandEncoder({ label: `gemma4:read:${key}` });
      encoder.copyBufferToBuffer(source, 0, staging, 0, bytes);
      this.device.queue.submit([encoder.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const copy = new Float32Array(staging.getMappedRange().slice(0));
      staging.unmap();
      return copy;
    } finally {
      staging.destroy();
    }
  }

  /** The half of the residual pair a reader would see right now, read back as f32. */
  async readResidual(elements: number): Promise<Float32Array> {
    const source = this.buffers.readSide('hidden', elements * 4) as unknown as GPUBuffer;
    const staging = this.device.createBuffer({
      label: 'gemma4:read:hidden',
      size: elements * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      const encoder = this.device.createCommandEncoder({ label: 'gemma4:read:hidden' });
      encoder.copyBufferToBuffer(source, 0, staging, 0, elements * 4);
      this.device.queue.submit([encoder.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const copy = new Float32Array(staging.getMappedRange().slice(0));
      staging.unmap();
      return copy;
    } finally {
      staging.destroy();
    }
  }

  // ------------------------------------------------------------------- the ForwardExecutor

  async prefill(tokens: readonly number[], startPosition: number, isLast: boolean): Promise<number> {
    if (tokens.length === 0) return -1;
    if (tokens.length > this.maxChunkTokens) {
      throw new Error(
        `gemma4 engine: a prefill chunk of ${tokens.length} exceeds the ${this.maxChunkTokens} `
        + 'this executor sized its activation slots for',
      );
    }
    const geometry = this.geometryFor('gemm', tokens.length, startPosition);
    this.allocateSlots();
    const steps = prefillStepsFor(this.arch, isLast, this.pleSlices);
    return this.runSteps(steps, geometry, isLast, () => {
      this.uploadIds(tokens);
      this.uploadRope(startPosition, tokens.length);
    });
  }

  async decode(prevToken: number, position: number): Promise<number> {
    if (this.isAborted()) return -1;
    const geometry = this.geometryFor('gemv', 1, position);
    this.allocateSlots();
    return this.runSteps(decodeStepsFor(this.arch, this.pleSlices, this.attentionPrologue, this.batchProjections && geometry.tokens === 1, geometry.tokens === 1), geometry, true, () => {
      this.uploadIds([prevToken]);
      this.uploadRope(position, 1);
    });
  }

  /**
   * The lookahead decode step (see the host pipeline note above): the gathers read the token the
   * previous step's argmax left in the token slot, so nothing about this step waits on the host
   * having seen that token. The returned promise resolves to this step's own argmax once its
   * readback lands, exactly as `decode` does.
   */
  /**
   * The verify pass of a speculative decode: `tokens[0]` is the token the model chose last and
   * `tokens[1..]` the drafted continuation, placed at `position` onward, and the answer is the
   * model's argmax after every one of them, in order. `answer[i]` is the token that follows
   * tokens[i]; the caller accepts the longest prefix on which answer[i] equals tokens[i + 1]
   * and reads the first disagreeing answer as the model's own choice, which is exactly the plain
   * greedy decode's choice at that position, because each column is the decode step's arithmetic
   * (kernels/qgemvWide.ts). The KV rows for every position are written, accepted or not; the
   * next pass overwrites the unaccepted ones by position.
   */
  async verify(tokens: readonly number[], position: number): Promise<number[]> {
    if (this.isAborted()) return [];
    if (tokens.length < 1 || tokens.length > GEMV_WIDE_COLS) {
      throw new Error(`gemma4 engine: a verify pass takes 1 to ${GEMV_WIDE_COLS} tokens, got ${tokens.length}`);
    }
    const geometry = this.geometryFor('gemv', tokens.length, position);
    this.allocateSlots();
    return this.runStepsWide(decodeStepsFor(this.arch, this.pleSlices, this.attentionPrologue, this.batchProjections && geometry.tokens === 1, geometry.tokens === 1), geometry, () => {
      this.uploadIds(tokens);
      this.uploadRope(position, tokens.length);
    });
  }

  decodeAhead(position: number): Promise<number> {
    if (this.isAborted()) return Promise.resolve(-1);
    const geometry: ForwardGeometry = { ...this.geometryFor('gemv', 1, position), idsFromTokenSlot: true };
    this.allocateSlots();
    return this.runSteps(decodeStepsFor(this.arch, this.pleSlices, this.attentionPrologue, this.batchProjections && geometry.tokens === 1, geometry.tokens === 1), geometry, true, () => {
      this.uploadRope(position, 1);
    });
  }

  dispose(): void {
    for (const buffer of this.readbackPool) buffer.destroy();
    this.readbackPool.length = 0;
    this.querySet?.destroy();
    this.queryResolve?.destroy();
    this.queryStaging?.destroy();
    this.querySet = null;
    this.queryResolve = null;
    this.queryStaging = null;
    this.queryCapacity = 0;
  }
}

/**
 * The step lists, taken from plan.ts's builders on every call rather than cached, so a plan edit
 * reaches the executor without anybody remembering to invalidate anything. Building 731 small
 * objects is nothing next to the dispatches they describe.
 */
function decodeStepsFor(arch: Gemma4Arch, pleSlices: readonly GatherSliceRef[], attentionPrologue = true, batchProjections = false, allowProducerSplit = true): DispatchStep[] {
  // The split is read HERE, from the live geometry, rather than passed down from a caller,
  // because plan.ts has no imports and this is the one place every real decode plan is built. A
  // profile that turned kSplits on before any pipeline was compiled gets the fold steps; every
  // other run gets the plan this engine has always built, to the object. A verify pass runs the
  // same plan: the wide family carries the split as qgemv-2bit-gelu-split-m2 (execute.ts
  // gemvWide), and the decode step cache keys on the position count, so a token and a pass never
  // share an entry.
  return planDecodeStep(arch, pleSlices, {
    attentionPrologue,
    batchProjections,
    producerDownKSplits: allowProducerSplit ? producerDownSplits() : 1,
    pleGateKSplits: allowProducerSplit ? pleGateSplits() : 1,
    downKSplits: kSplitsOf(gemvGeometry(2)),
    attnKvSplits: kvSplitsOf(attentionGeometry()),
    attnKSplits: kSplitsOf(gemvGeometry(4)),
  });
}

function prefillStepsFor(
  arch: Gemma4Arch,
  isLast: boolean,
  pleSlices: readonly GatherSliceRef[],
): DispatchStep[] {
  return planPrefillChunk(arch, isLast, pleSlices);
}

/** Re-exported so a caller can size a BufferManager without importing execute.ts as well. */
export { slotElements, kvLengthAfter, kvRoleForLayer };

/**
 * The one unsafe conversion in the engine's GPU wiring, named so it is greppable. `GPUDevice`
 * satisfies `DeviceLike` at runtime; tsc balks only on the bind group entry variance, because
 * `BufferLike` is narrower than `GPUBuffer`. The conversion is one way and safe: the manager only
 * ever hands back buffers this device created.
 */
export function asDeviceLike(device: GPUDevice): DeviceLike {
  return device as unknown as DeviceLike;
}
