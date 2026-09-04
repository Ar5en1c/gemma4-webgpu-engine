// Written from docs/ENGINE-PLAN.md sections 2, 3, 5, 6 and 8, the API contract in
// src/engine/llm/gemma4-engine.d.ts, and this engine's own modules. No vendored bundle, no
// extracted kernel and no third party engine source was read.
//
// SPDX-License-Identifier: Apache-2.0
//
// The engine shell: the module that makes the kernels a model. It implements exactly the API the
// app consumes through `@gemma4/engine` (gemma4-engine.d.ts), so the eventual swap is one alias
// change in vite.config.ts. This round the shell compiles, loads weights, tokenizes, schedules
// and proves its orchestration in Node against fakes; the dispatches that turn a scheduled step
// into logits land next round, and until they do the executor below says so in plain words
// instead of producing plausible garbage.

import { repackTile16, tile16Target } from './kernels/repack';
import type { BufferLike } from './buffers';
import { Gemma4DecodeStream, Gemma4Tokenizer, type Gemma4Message } from './tokenizer';
import { KvCacheState, createKvLayout, type KvLayout } from './kv';
import {
  DEFAULT_PREFILL_CHUNK,
  GEMMA4_E2B,
  PENDING_KERNELS,
  buildForwardPlan,
  censusByKernel,
  openingWeightsEvent,
  planDecodeStep,
  runGreedyLoop,
  weightsEvent,
  type DispatchStep,
  type ForwardPlan,
  type Gemma4Arch,
  type ProgressEvent,
} from './plan';
import { BufferManager, type BufferManagerStats } from './buffers';
import { PipelineStore, chooseReduceVariant, runSubgroupSelfTest, type ReduceVariant } from './pipeline';
import { GpuExecutor, asDeviceLike, type ExecutorResources, type ForwardExecutor } from './runtime';
import { bf16StaysPacked, bf16ToF32 } from './quant';
import { Gemma4DeviceError, requestGemma4Device, type Gemma4Device } from './device';
import { resolveUrl, type SafetensorsDirectory } from './safetensors';
import { WeightCache, defaultFetch, loadWeights, loaderConcurrencyFor, type LoadReceipt } from './cache';
import { resolveDeviceProfile, withLiveLimits } from './deviceProfile';
import { planGatherSplit } from './tableSplit';
import { PLE_CODES } from './execute';
import type { GatherSliceRef } from './plan';
import { PLE_GROUPS } from './kernels/layerGeometry';

// ----------------------------------------------------------------------------- public types

/** Re-exported so the engine module presents the same names the ambient declaration does. */
export type { Gemma4Message };

export type Gemma4Progress = ProgressEvent;

/**
 * Device request options, declared because gemma4-engine.d.ts declares them and this engine
 * claims to implement that contract in full.
 *
 * Neither field does anything here, and that is an answer rather than an oversight.
 * `disabledFeatures` and `host` are in the contract because the incumbent took them. This engine
 * owns its device through the one seam in ./device.ts, which requires shader-f16, grants subgroups
 * only the right to compile, and proves the subgroup path with a known answer self test rather
 * than trusting a caller's feature list (ENGINE-PLAN 5.5 rule 4). Accepting the field and ignoring
 * it keeps the eventual swap a one line change; honouring it would mean letting a caller switch
 * off a feature the engine has already tested for itself.
 */
export interface Gemma4RuntimeOptions {
  disabledFeatures?: string[] | string;
  host?: unknown;
}

export interface Gemma4LoadOptions {
  onProgress?: (progress: Gemma4Progress) => void;
  /** Accepted for contract compatibility and deliberately unused. See `Gemma4RuntimeOptions`. */
  runtimeOptions?: Gemma4RuntimeOptions;
  cache?: boolean;
  force?: boolean;
  /** IndexedDB store name. Defaults to this engine's own `purrview-gemma4-weights-v1`. */
  cacheName?: string;
  signal?: AbortSignal;
  /** Only needed for gated repos. The Gemma 4 E2B weights are public, so this is normally unset. */
  accessToken?: string;
  revision?: string;
  fetch?: typeof fetch;
}

export interface Gemma4GenerateOptions {
  maxNewTokens?: number;
  eosTokenId?: number | number[];
  signal?: AbortSignal;
}

export interface Gemma4Chunk {
  token: number;
  delta: string;
  text: string;
}

export interface Gemma4DeviceInfo {
  vendor: string;
  architecture: string;
  device: string;
  description: string;
  isFallbackAdapter: boolean;
  subgroupMinSize?: number;
  subgroupMaxSize?: number;
  features: {
    shaderF16: boolean;
    subgroups: boolean;
    subgroupMatrix: boolean;
    timestampQuery: boolean;
  };
}

export const DEFAULT_MODEL_ID = 'google/gemma-4-E2B-it-qat-mobile-transformers';

/**
 * Repo id to a snapshot base URL, matching the contract in gemma4-engine.d.ts: absolute URLs and
 * paths pass through unchanged, a bare repo id resolves against the Hugging Face resolve route
 * whose CDN honours ranged reads (ENGINE-PLAN section 3).
 */
export function resolveModelRoot(modelId: string | null, revision = 'main'): string {
  const id = modelId ?? DEFAULT_MODEL_ID;
  if (/^https?:\/\//.test(id) || id.startsWith('/') || id.startsWith('.')) return id;
  return resolveUrl(id, revision, '').replace(/\/$/, '');
}

/** Default generation budget when the caller gives none. client.ts always passes its own. */
export const DEFAULT_MAX_NEW_TOKENS = 256;

// ----------------------------------------------------------------------- forward execution

/**
 * The seam between the schedule and the GPU, defined in runtime.ts beside the executor that
 * implements it and re-exported here because this module is the one callers import.
 */
export type { ForwardExecutor };

/**
 * The executor before `warmup()` has run. It no longer stands in for a missing forward path, which
 * is what round 1's version did; it stands in for a device whose pipelines have not been built,
 * and every public entry point that needs a forward awaits `warmup()` first, so reaching this is a
 * programming error rather than a state a caller can be in.
 */
class UnpreparedExecutor implements ForwardExecutor {
  private refuse(): never {
    const pending = PENDING_KERNELS.length > 0
      ? ` Kernel families still owed by other lanes: ${PENDING_KERNELS.join(', ')}.`
      : '';
    throw new Error(
      'gemma4 engine: the forward ran before warmup() built its pipelines. Call warmup(), or use '
      + `generate(), which awaits it.${pending}`,
    );
  }

  prefill(): Promise<number> {
    this.refuse();
  }

  decode(): Promise<number> {
    this.refuse();
  }
}

/**
 * The dry executor: walks the forward plan against fake device objects and records the dispatch
 * sequence instead of running it. This is what the orchestrator check section and the dev
 * harness's census view drive. Tokens are synthesized outside the EOS set so a dry generate
 * exercises the whole loop.
 */
export class DryRunExecutor implements ForwardExecutor {
  readonly dispatches: DispatchStep[] = [];
  private readonly plan: ForwardPlan;
  private chunkIndex = 0;

  constructor(plan: ForwardPlan) {
    this.plan = plan;
  }

  prefill(tokens: readonly number[], _startPosition: number, isLast: boolean): Promise<number> {
    const steps = this.plan.prefillSteps[this.chunkIndex] ?? this.plan.prefillSteps[this.plan.prefillSteps.length - 1] ?? [];
    this.chunkIndex += 1;
    this.dispatches.push(...steps);
    return Promise.resolve(isLast ? 7 + (tokens.length % 5) : -1);
  }

  decode(prevToken: number, position: number): Promise<number> {
    this.dispatches.push(...this.plan.decodeSteps);
    return Promise.resolve(7 + ((prevToken + position) % 5));
  }
}

// ------------------------------------------------------------------------------- the engine

interface LoadedState {
  root: string;
  repo: string | null;
  revision: string;
  tokenizer: Gemma4Tokenizer;
  arch: Gemma4Arch;
  directory: SafetensorsDirectory | null;
  gpu: Gemma4Device | null;
  buffers: BufferManager | null;
  cache: WeightCache | null;
  /**
   * The checkpoint's four byte tensors, kept on the host rather than uploaded: every quantized
   * linear's `input_activation_scale` and `output_activation_scale`, and each layer's
   * `layer_scalar`. They are uniform values, not storage, and the executor reads them through
   * `ExecutorResources.scalar`. A name this map does not carry reads 0, which is the checkpoint's
   * own way of saying a site is uncalibrated (quant.ts applySrq).
   */
  scalars: Map<string, number>;
  /**
   * Vocabulary ranges the PLE table is carried in, when this adapter's buffers are too small to
   * hold its 1,174,405,120 bytes whole. Empty on every adapter that fits it, which is every one
   * this project has measured except iOS, whose maxBufferSize is 1,073,741,824. Filled by the
   * upload sink from tableSplit.ts's plan and handed to the executor, which turns each range into
   * its own gather dispatch. See ENGINE-PERF 28.7 for what the absent split cost.
   */
  pleSlices: GatherSliceRef[];
  /** The union of every stop set the checkpoint's files name. See `unionStopTokens`. */
  stopTokenIds: readonly number[];
  /**
   * What the load actually did on the wire and in the cache: requests by kind, runs resumed
   * against runs fetched, the storage readiness answer, and the signed URL that served it. Kept
   * because a load that quietly re-downloaded two gigabytes and a load that resumed cleanly are
   * indistinguishable from the outside otherwise.
   */
  loadReceipt: LoadReceipt | null;
}

export class Gemma4Mobile {
  /**
   * Plan and declare the PLE table's split, if this adapter needs one, before its first byte lands.
   *
   * THE FACT THIS EXISTS FOR. The table is [262144, 8960] at 4 bits, which is 1,174,405,120 bytes.
   * iOS grants a `maxBufferSize` of 1,073,741,824, and WebGPU does not throw on an oversized
   * `createBuffer`: it hands back an invalid buffer, every write into it fails the same silent way,
   * and the load reports success with the table empty. That is exactly what every iPhone did, and
   * the model emitted fluent garbage rather than failing (ENGINE-PERF 28.7).
   *
   * The split is by vocabulary row so that a gather still reads one row out of one buffer, and the
   * gather runs once per slice with the slice's row range in its uniform (./tableSplit.ts). At iOS's
   * 1 GiB the plan is two slices, so the cost is one extra dispatch per gather per forward, and on
   * every adapter that fits the table this function plans one slice, declares nothing, and the engine
   * runs the plan it has always run.
   *
   * The geometry comes off the checkpoint's own header rather than from `arch`, because a split
   * planned from a constant is a split a revision bump can silently invalidate.
   */
  private declarePleSplit(entry: { shape: readonly number[]; byteLength: number }): void {
    const buffers = this.state.buffers;
    const gpu = this.state.gpu;
    if (!buffers || !gpu || buffers.isSplit(PLE_CODES)) return;
    const rows = entry.shape[0] ?? 0;
    if (rows <= 0 || entry.byteLength % rows !== 0) {
      throw new Error(
        `gemma4 engine: ${PLE_CODES} has shape [${entry.shape.join(', ')}] and ${entry.byteLength} `
        + 'bytes, which is not a whole number of bytes per vocabulary row, so its split cannot be '
        + 'planned. A table that cannot be split cannot be loaded on an adapter that needs one.',
      );
    }
    const rowBytes = entry.byteLength / rows;
    const plans = planGatherSplit(PLE_CODES, rows, rowBytes, PLE_GROUPS, {
      maxStorageBufferBindingSize: gpu.limits.maxStorageBufferBindingSize,
      maxBufferSize: gpu.limits.maxBufferSize,
      maxStorageBuffersPerShaderStage: gpu.limits.maxStorageBuffersPerShaderStage,
    });
    // One slice is the whole table in one buffer, which is the fast path and the only path this
    // engine had before. Declaring nothing is what keeps it byte for byte what it was.
    if (plans.codes.single) return;
    if (!plans.codes.feasible) throw new Error(`gemma4 engine: ${plans.codes.note}`);
    // The sliced shader indexes the scales by the absolute vocabulary row, so a split scale table
    // would read past the end of a slice and quietly return wrong numbers. No adapter reports a limit
    // small enough to reach this, and it is refused rather than assumed away.
    if (!plans.scalesSingle) throw new Error(`gemma4 engine: ${plans.note}`);
    buffers.declareSplit(PLE_CODES, plans.codes.slices.map((slice) => ({
      index: slice.index,
      byteOffset: slice.byteOffset,
      byteLength: slice.byteLength,
    })));
    this.state.pleSlices = plans.codes.slices.map((slice) => ({
      index: slice.index,
      startRow: slice.startRow,
      rowCount: slice.rowCount,
    }));
    console.warn(`gemma4 engine: ${plans.codes.note}`);
  }

  static readonly DEFAULT_MODEL_ID: string = DEFAULT_MODEL_ID;

  private readonly state: LoadedState;
  private readonly kvState = new KvCacheState();
  private readonly kvLayout: KvLayout;
  private pipelines: PipelineStore | null = null;
  private gpuExecutor: GpuExecutor | null = null;
  private executor: ForwardExecutor;
  private disposed = false;

  private constructor(state: LoadedState) {
    this.state = state;
    this.kvLayout = createKvLayout();
    this.executor = new UnpreparedExecutor();
  }

  /**
   * Load the model. Progress events follow the exact shapes engineHost.ts switches on: `init`,
   * `tokenizer`, `weights` with `kind: 'bytes'` then `kind: 'tensors'`, then `ready`, with the
   * opening weights event carrying `loaded: 0` and no `fromCache` so the host's download latch
   * reads the load correctly (ENGINE-PLAN section 3).
   */
  static async load(modelId?: string | null, options: Gemma4LoadOptions = {}): Promise<Gemma4Mobile> {
    const revision = options.revision ?? 'main';
    const root = resolveModelRoot(modelId ?? null, revision);
    const repo = /^https?:\/\//.test(root) && root.includes('huggingface.co/')
      ? root.split('huggingface.co/')[1]!.split('/resolve/')[0]!
      : null;
    const emit = (event: Gemma4Progress): void => options.onProgress?.(event);
    // `accessToken` is only needed for a gated repo, which the Gemma 4 E2B repos are not
    // (ENGINE-PLAN section 9, read 31 August 2026). It is honoured anyway because it is in the
    // contract and because a mirror or a private fork of the checkpoint is a reasonable thing for
    // a caller to point this engine at. A caller's own `fetch` still wins, since a caller that
    // supplies both has already decided how its requests are made.
    const doFetch = options.fetch ?? bearerFetch(options.accessToken);

    emit({ status: 'init', loaded: 0 });

    // The device first, so a machine that cannot run the model refuses before any download.
    // f16 is a precondition, not a preference (ENGINE-PLAN section 2).
    const gpu = await requestGemma4Device({ subgroups: true, label: 'gemma4-engine' });

    emit({ status: 'tokenizer' });
    const tokenizer = await Gemma4Tokenizer.load(root, { fetch: doFetch, signal: options.signal });

    // config.json, validated against the architecture contract so a checkpoint swap that changes
    // the geometry fails here in words rather than later in a wrong shaped dispatch.
    const arch = GEMMA4_E2B;
    const config = await fetchConfig(root, doFetch, options.signal);
    const stopTokenIds = unionStopTokens(
      tokenizer.defaultStopTokenIds,
      config.eosTokenIds,
      await fetchGenerationStopIds(root, doFetch, options.signal),
    );
    if (config.numHiddenLayers !== arch.layerCount || config.numKvSharedLayers !== arch.layerCount - arch.kvProducerLayers) {
      throw new Error(
        `gemma4 engine: config.json reports ${config.numHiddenLayers} layers with `
        + `${config.numKvSharedLayers} shared, but this engine was built for ${arch.layerCount} `
        + `and ${arch.layerCount - arch.kvProducerLayers}`,
      );
    }

    // The chunk ceiling the weight cache writes at is a measurement rather than a constant: the
    // round 2 IndexedDB probe found that a value above it writes successfully and then fails its
    // own read once the origin's accounted usage catches its quota (deviceProfile.ts,
    // idbMaxValueBytes). Resolve the profile against this adapter so the number comes from the
    // machine rather than from a default.
    const profile = withLiveLimits(
      resolveDeviceProfile({ vendor: gpu.info.vendor, architecture: gpu.info.architecture }),
      gpu.limits,
    );

    const cache = options.cache === false
      ? null
      : await WeightCache.open(options.cacheName, profile.idbMaxValueBytes.value);
    if (cache && options.force) {
      await cache.clearScope(repo ?? root, revision);
    }

    const engine = new Gemma4Mobile({
      root,
      repo,
      revision,
      tokenizer,
      arch,
      // Filled in from the load receipt below, which carries the directory with the lm_head tie
      // already resolved, so a consumer asking for lm_head.weight gets the embed entry whose bytes
      // were actually fetched.
      directory: null,
      gpu,
      buffers: new BufferManager(asDeviceLike(gpu.device), { maxBufferSize: gpu.limits.maxBufferSize }),
      cache,
      scalars: new Map<string, number>(),
      pleSlices: [],
      stopTokenIds,
      loadReceipt: null,
    });

    // The whole network path lives in cache.ts: one resolve, the two request manifest read, the tie
    // guard's 256 KiB before the allow list is built, then pooled single range GETs against the
    // signed CDN URL with per run resume and profile sized IndexedDB chunks. This module supplies
    // the sink and nothing else about how the bytes arrive.
    let uploaded = 0;
    // Source bytes delivered per tensor. A tensor larger than the loader's piece size arrives in
    // several calls, and cache.ts fetches pieces through a CONCURRENT POOL and delivers them in
    // COMPLETION order, not in offset order (its `deliver` chain serializes the sink, it does not
    // sort it). So "this piece ends at byteLength" does not mean "the tensor is whole", and the
    // 2-bit repack below must not fire until it is. See ENGINE-PERF section 28.5.
    const deliveredBytes = new Map<string, number>();
    let tensorCount = 0;
    // How many response bodies may be on the heap at once. Scaled down on a device whose adapter
    // caps a buffer at a gigabyte, because the pool's default 384 MiB of ArrayBuffers is what a
    // phone cannot afford alongside two gigabytes going resident. See cache.ts.
    const concurrency = loaderConcurrencyFor(gpu.limits.maxBufferSize);
    const receipt = await loadWeights({
      poolSize: concurrency.poolSize,
      maxBytesInFlight: concurrency.maxBytesInFlight,
      fileUrl: `${root}/model.safetensors`,
      repo: repo ?? root,
      revision,
      numHiddenLayers: config.numHiddenLayers,
      numKvSharedLayers: config.numKvSharedLayers,
      cache,
      // The storage guard refuses a load that `navigator.storage.estimate()` says will not fit,
      // before the first payload byte, because a 2 GB download that cannot be kept is a download
      // the user pays for twice. A caller that passed `cache: false` has already said it is not
      // keeping anything, so there is nothing for the guard to protect and refusing on storage
      // grounds would refuse a load that asks the origin for no storage at all. That is not a
      // hypothetical: this project's own browser reports a 443 MB quota on a brand new origin
      // (the round 3 loader lane's finding), so `cache: false` is the only way the engine runs
      // here at all, and it is the path the round 3 battery is measured on.
      requireStorage: options.cache !== false,
      fetchImpl: doFetch,
      signal: options.signal,
      onPlan: (info) => {
        tensorCount = info.plan.tensorCount;
        if (!info.tie.tied) {
          // The reason is a complete prose sentence written for exactly this line. Surfacing it is
          // deliberate: a revision bump silently costing 101.7 MB should be visible in the console.
          console.warn(`gemma4 engine: ${info.tie.reason}`);
        }
        emit(openingWeightsEvent(info.totalBytes));
      },
      onBytes: (loaded, total, fromCache) => emit(weightsEvent('bytes', loaded, total, fromCache)),
      sink: (name, entry, bytes, offset) => {
        // Upload straight to the GPU, packed. Nothing quantized is dequantized on the CPU
        // (ENGINE-PLAN section 3), and nothing is retained on the JS heap. Two narrow exceptions,
        // both about storage rather than about quantization:
        //
        //  - A one element tensor is a uniform value, not a buffer. Every activation scale and
        //    every layer_scalar lands in the host side scalar map instead of in a storage buffer,
        //    because that is what a kernel's params block needs it as.
        //  - BF16 has no WGSL storage type. The norm gains and per_layer_model_projection are
        //    stored BF16 and are widened to f32 on the way in, which is exact: every BF16 value is
        //    an f32 value. This is not a dequantization; quant.ts's TEXT_STORAGE_FAMILIES records
        //    these families as unquantized and there is nothing to unpack.
        //
        // A tensor over the loader's piece size arrives in several calls, each with the byte offset
        // of its own span, and the buffer manager writes each into the same buffer at that offset.
        // The two that do are the PLE table and the embed table, both quantized and neither BF16 or
        // scalar, so the widening below never sees a partial tensor; it is offset anyway, because a
        // widened offset is the source offset doubled and nothing else, and a rule that holds by
        // arithmetic is worth more than a rule that holds by which tensors happen to be big.
        // Before this tensor's first byte reaches a buffer: if it does not fit in one buffer on
        // this adapter, declare the split now so every piece is routed into a slice. This is here
        // rather than beside the BufferManager's construction because the tensor's true byte length
        // comes off the checkpoint's own header, and a split planned from a constant would be a
        // split that a revision bump could silently invalidate.
        if (name === PLE_CODES) engine.declarePleSplit(entry);
        const scalar = readScalarTensor(entry.dtype, entry.elementCount, bytes);
        if (scalar !== null) {
          engine.state.scalars.set(name, scalar);
        } else if (entry.dtype === 'BF16' && !bf16StaysPacked(name)) {
          const widened = bf16ToF32(bytes);
          engine.state.buffers!.uploadWeight(
            name,
            new Uint8Array(widened.buffer, widened.byteOffset, widened.byteLength),
            { offset: offset * 2, totalBytes: entry.byteLength * 2 },
          );
        } else {
          engine.state.buffers!.uploadWeight(name, bytes, { offset, totalBytes: entry.byteLength });
        }
        // Counted when the tensor is complete rather than per call, so the tensors progress event
        // still counts tensors and a resumed load that takes eighteen requests for one table does
        // not report eighteen of 1,439.
        //
        // Completeness is the SUM of the source bytes delivered for this tensor, not the end
        // offset of the piece in hand. Delivery is serialized so this accumulator needs no
        // atomicity, but it is not ordered, so the piece that ends at byteLength can arrive with
        // earlier pieces still in flight.
        const soFar = (deliveredBytes.get(name) ?? 0) + bytes.byteLength;
        deliveredBytes.set(name, soFar);
        if (soFar >= entry.byteLength) {
          deliveredBytes.delete(name);
          // A 2-bit codes tensor is read in the interleaved tile layout (kernels/qlayout.ts):
          // once the WHOLE tensor is queued it is repacked on the GPU into a new buffer that takes
          // its name, and the uploaded one is released. The cache keeps the checkpoint bytes.
          const target = tile16Target(name, entry.shape, entry.dtype);
          if (target) {
            const buffers = engine.state.buffers!;
            const tiled = repackTile16(
              engine.state.gpu!.device,
              buffers.weight(name) as unknown as GPUBuffer,
              target.rows,
              target.k,
              name,
            );
            buffers.replaceWeight(name, tiled as unknown as BufferLike);
          }
          uploaded += 1;
          emit(weightsEvent('tensors', uploaded, tensorCount));
        }
      },
    });
    engine.state.directory = receipt.directory;
    engine.state.loadReceipt = receipt;

    emit({ status: 'ready' });
    return engine;
  }

  /** Tokenize a rendered chat prompt. Length is the prompt budget in tokens for client.ts. */
  encodePrompt(messages: Gemma4Message[]): number[] {
    this.assertLive();
    return this.state.tokenizer.encodePrompt(messages);
  }

  /**
   * Stream a completion. Greedy argmax, `{ token, delta, text }` with cumulative text, EOS as a
   * set, abort within one token, and the section 6 prefix rewind deciding how much of the KV
   * cache survives the new prompt. The first token comes from the prefill's final logits, so
   * `maxNewTokens: 1` runs zero decode passes.
   */
  async *generate(messages: Gemma4Message[], options: Gemma4GenerateOptions = {}): AsyncGenerator<Gemma4Chunk, void, void> {
    this.assertLive();
    this.assertDeviceLive();
    const signal = options.signal ?? null;
    const isAborted = (): boolean => signal?.aborted === true;
    // Idempotent, so the ordinary path pays for it once. A caller that already called warmup()
    // returns from it immediately; a caller that did not gets its pipelines built here rather than
    // an executor that refuses.
    await this.warmup();
    const ids = this.encodePrompt(messages);

    // The rewind. kv.ts owns the decision; the engine just obeys it. Reuse is what makes the
    // app's second turn cost 437 ms instead of 1,863 (ENGINE-PLAN section 6).
    const reuse = this.kvState.acceptPrompt(ids);
    const plan = buildForwardPlan(this.state.arch, reuse.prefillFrom, ids.length, DEFAULT_PREFILL_CHUNK);

    let firstToken = -1;
    for (let i = 0; i < plan.chunks.length; i += 1) {
      if (isAborted()) return;
      const chunk = plan.chunks[i]!;
      const tokens = ids.slice(chunk.start, chunk.end);
      const isLast = i === plan.chunks.length - 1;
      const result = await this.executor.prefill(tokens, chunk.start, isLast);
      this.kvState.append(tokens);
      if (isLast) firstToken = result;
    }
    if (plan.chunks.length === 0 || firstToken < 0) {
      // Nothing to prefill only happens on an empty suffix, which the reuse plan maps to a full
      // reset, so a well formed prompt always reaches here with a token.
      return;
    }

    const eosIds = eosSet(options.eosTokenId, this.state.stopTokenIds);
    const stream = this.state.tokenizer.decodeStream({ skipSpecialTokens: true });
    const loop = runGreedyLoop({
      firstToken,
      maxNewTokens: options.maxNewTokens ?? DEFAULT_MAX_NEW_TOKENS,
      eosIds,
      startPosition: ids.length,
      isAborted,
      decode: (prevToken, position) => {
        // The generated token enters the cached transcript as its KV rows are written, so the
        // next turn's longest common prefix covers the reply the app echoes back.
        this.kvState.append([prevToken]);
        return this.executor.decode(prevToken, position);
      },
      // The GPU executor offers the lookahead step; the dry and unprepared executors do not, and
      // the loop runs serially over `decode` for them. Under the lookahead the transcript append
      // moves to `onAdvance`, which fires for exactly the tokens whose step was submitted.
      decodeAhead: this.executor.decodeAhead ? (position) => this.executor.decodeAhead!(position) : undefined,
      onAdvance: (token) => { this.kvState.append([token]); },
    });
    for await (const step of loop) {
      const delta = stream.push(step.token);
      yield { token: step.token, delta, text: stream.text };
    }
  }

  /** Run `generate` to the end and return the final cumulative text. */
  async complete(messages: Gemma4Message[], options: Gemma4GenerateOptions = {}): Promise<string> {
    let text = '';
    for await (const chunk of this.generate(messages, options)) text = chunk.text;
    return text;
  }

  /**
   * Compile the decode kernels so the first real token is not the slow one. Also where the
   * reduce variant is decided, once, from the granted feature plus the known answer self test
   * (ENGINE-PLAN 5.5 rule 4). Idempotent.
   */
  async warmup(): Promise<void> {
    this.assertLive();
    if (this.gpuExecutor || !this.state.gpu) return;
    const passed = await runSubgroupSelfTest(this.state.gpu);
    const variant = chooseReduceVariant(this.state.gpu.features.subgroups, passed);
    this.pipelines = new PipelineStore(this.state.gpu, variant);
    const executor = new GpuExecutor({
      gpu: this.state.gpu,
      pipelines: this.pipelines,
      buffers: this.state.buffers!,
      resources: this.resources(),
      arch: this.state.arch,
      kvLayout: this.kvLayout,
      maxChunkTokens: DEFAULT_PREFILL_CHUNK,
      pleSlices: this.state.pleSlices,
    });
    // Compiling the decode set is what makes the first real token not the slow one, and it now
    // goes through the executor because a pipeline needs the bind group layout its kernel declares
    // (pipeline.ts get). `prepare` also allocates every activation slot and every KV cache buffer,
    // and it checks each kernel's storage binding count against the adapter's limit of 10
    // (ENGINE-PLAN risk 6), which is a check that could only ever run here.
    await executor.prepare(
      planDecodeStep(this.state.arch, this.state.pleSlices),
      {
        arch: this.state.arch,
        mode: 'gemv',
        tokens: 1,
        startPosition: 0,
        maxContext: this.kvLayout.maxContext,
        scalar: (name) => this.state.scalars.get(name) ?? 0,
      },
    );
    this.gpuExecutor = executor;
    this.executor = executor;
  }

  /** The weight and scalar seam the executor reads. One object, built once, no copies. */
  private resources(): ExecutorResources {
    const buffers = this.state.buffers;
    if (!buffers) throw new Error('gemma4 engine: no buffer manager, so no weights are resident');
    return {
      weight: (name) => {
        if (!buffers.hasWeight(name)) {
          throw new Error(
            `gemma4 engine: the forward asked for tensor ${name}, which is not resident. Either `
            + 'the loader allow list in safetensors.ts does not fetch it or the executor names it '
            + 'differently from the checkpoint.',
          );
        }
        return buffers.weight(name) as unknown as GPUBuffer;
      },
      hasWeight: (name) => buffers.hasWeight(name),
      scalar: (name) => this.state.scalars.get(name) ?? 0,
    };
  }

  /** Drop the KV cache and the cached transcript. */
  /**
   * A lost device is not an error anywhere in WebGPU: every submit is accepted and none of it runs,
   * every readback returns zeros, and a greedy loop happily yields maxNewTokens empty strings at
   * thousands of tokens a second. That is worse than a crash, because it looks like a measurement.
   * An iPhone reported exactly that (ENGINE-PERF 28.6), so generate() refuses to start on a dead
   * device and says why.
   */
  private assertDeviceLive(): void {
    const gpu = this.state.gpu;
    if (gpu?.lostReason) {
      throw new Gemma4DeviceError('device-lost',
        `the GPU device was lost (${gpu.lostReason}); every dispatch since is a no op, so any `
        + 'number measured after it is meaningless. Reload the page to get a new device.');
    }
  }

  reset(): void {
    this.kvState.reset();
  }

  /** Release GPU resources. The instance is unusable afterwards. */
  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.gpuExecutor?.dispose();
    this.state.buffers?.dispose();
    this.state.cache?.close();
    this.state.gpu?.destroy();
  }

  /**
   * What the last load did on the wire and in the cache. Null before `load()` has finished, and
   * an addition to the contract in gemma4-engine.d.ts rather than a change to it: the incumbent
   * has no equivalent and nothing in the app calls this. The dev loader page and the receipt
   * surface read it.
   */
  loadReceipt(): LoadReceipt | null {
    return this.state.loadReceipt;
  }

  /**
   * Which reduce policy `warmup()` chose for this instance, or null before it has run. The same
   * kind of addition as `loadReceipt()` above: read only, absent from the contract in
   * gemma4-engine.d.ts, and called by nothing in the app.
   *
   * It exists because ENGINE-PLAN 5.5 rule 4 makes the policy a property of the machine rather
   * than of the build, so a receipt taken through the shipping class had no way to say which of
   * the two kernel sets produced its tokens. Every other page under dev/ builds its own
   * PipelineStore and therefore already knows; this class builds its own and kept the answer.
   */
  /**
   * What the GPU has to say about itself: whether the device has been lost, and every uncaptured
   * error since load. A lost device accepts submits and runs none of them, so a caller that reports
   * timings MUST read this. See ENGINE-PERF 28.6.
   */
  deviceErrors(): { lostReason: string | null; errors: readonly string[] } {
    const gpu = this.state.gpu;
    if (!gpu) return { lostReason: null, errors: [] };
    return { lostReason: gpu.lostReason, errors: [...gpu.errors] };
  }

  reducePolicy(): ReduceVariant | null {
    return this.pipelines?.variant ?? null;
  }

  /**
   * The capped logits the last forward step left on the GPU, or null when no GPU executor has run
   * one. Dev instrument, like `GpuExecutor.readSlot` which it is one line over: nothing in the app
   * calls it and the decode path never reads it back.
   *
   * It exists because round 2's ruling two says token identity is a reported statistic and never a
   * verdict on its own, and every probe has to report the top two gap at the position where it
   * diverges. A gap is a property of the logits, and generate() only ever yields the argmax of
   * them, so a battery page driving the shipping class had no way to say whether a divergence was
   * a near tie or a real disagreement. Run generate() with `maxNewTokens: k + 1` and read this,
   * and the answer is the logits of exactly the step that produced token k.
   */
  async lastLogits(): Promise<Float32Array | null> {
    this.assertLive();
    if (!this.gpuExecutor) return null;
    return this.gpuExecutor.readSlot('logits.capped', this.state.arch.vocabSize);
  }

  /**
   * What this engine is actually holding on the GPU: weight buffers and their bytes, activation
   * buffers and theirs.
   *
   * Exposed because "it crashed on that device" is not a bug report and "it was holding 2,384 MiB
   * when the device stopped at 1,536" is. An adapter's allocation ladder says what a device will
   * grant; this says what the engine asked for, and only the two together say whether a device can
   * run the model. Null before the buffer manager exists, which is before `load`.
   */
  bufferStats(): BufferManagerStats | null {
    return this.state.buffers?.snapshotStats() ?? null;
  }

  deviceInfo(): Gemma4DeviceInfo {
    this.assertLive();
    const gpu = this.state.gpu;
    if (!gpu) throw new Gemma4DeviceError('no-device', 'deviceInfo() before a device exists');
    const adapterFeatures = gpu.adapter.features;
    const info: Gemma4DeviceInfo = {
      vendor: gpu.info.vendor,
      architecture: gpu.info.architecture,
      device: gpu.info.device,
      description: gpu.info.description,
      isFallbackAdapter: (gpu.adapter as { isFallbackAdapter?: boolean }).isFallbackAdapter === true,
      features: {
        shaderF16: gpu.features.f16,
        subgroups: gpu.features.subgroups,
        subgroupMatrix: adapterFeatures.has('chromium-experimental-subgroup-matrix' as GPUFeatureName),
        timestampQuery: adapterFeatures.has('timestamp-query' as GPUFeatureName),
      },
    };
    const limits = gpu.adapter as unknown as { info?: { subgroupMinSize?: number; subgroupMaxSize?: number } };
    if (typeof limits.info?.subgroupMinSize === 'number') info.subgroupMinSize = limits.info.subgroupMinSize;
    if (typeof limits.info?.subgroupMaxSize === 'number') info.subgroupMaxSize = limits.info.subgroupMaxSize;
    return info;
  }

  // ------------------------------------------------------------------------- dry run seam

  /**
   * Swap the executor for the dry recorder over a fresh forward plan and return it. The dev
   * harness and the verifier use this to walk a whole generate() against fake dispatches; the
   * pure census maths is in plan.ts where Node can reach it without a class.
   */
  attachDryExecutor(prefillFrom: number, promptLength: number): DryRunExecutor {
    const plan = buildForwardPlan(this.state.arch, prefillFrom, promptLength, DEFAULT_PREFILL_CHUNK);
    const dry = new DryRunExecutor(plan);
    this.executor = dry;
    return dry;
  }

  /** The engine's own census: dispatches per steady decode token, by kernel. */
  static decodeCensus(): { total: number; byKernel: Record<string, number> } {
    const steps = planDecodeStep(GEMMA4_E2B);
    return { total: steps.length, byKernel: censusByKernel(steps) };
  }

  private assertLive(): void {
    if (this.disposed) throw new Error('gemma4 engine: this instance was disposed');
  }
}

// -------------------------------------------------------------------------------- internals

/**
 * A one element tensor's value, or null when the tensor is a buffer rather than a scalar.
 *
 * The activation scales are F32 with shape `[]`, which is a scalar with no axes at all rather than
 * a length one vector, and `layer_scalar` is BF16 with shape `[1]`. Both are one value and both
 * belong in a uniform, so this keys off the element count and not off the shape's rank.
 */
export function readScalarTensor(
  dtype: string,
  elementCount: number,
  bytes: Uint8Array,
): number | null {
  if (elementCount !== 1) return null;
  if (dtype === 'F32' && bytes.byteLength >= 4) {
    return new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getFloat32(0, true);
  }
  if (dtype === 'BF16' && bytes.byteLength >= 2) {
    return bf16ToF32(bytes.subarray(0, 2))[0]!;
  }
  return null;
}

/**
 * The stop set for one generation. Absent means the checkpoint's own set; a number or an array
 * means exactly what the caller named, and that includes the empty array.
 *
 * The empty case is the round 3 seam lane's correction, and it is a contract difference rather
 * than a preference. `eosTokenId: []` is how both engines are asked to run a fixed number of
 * tokens past their own stop token, which is how the reference probes are taken: the incumbent
 * honours it and produced all sixteen on every probe, and this engine read `[]` as "unset" and
 * stopped at the end of turn after nine on short-question. Two engines that answer the same option
 * differently are not swappable behind one alias, whatever their tokens do, and the app never
 * passes the option at all, so aligning on the incumbent's reading costs the app nothing.
 */
export function eosSet(option: number | number[] | undefined, fallback: readonly number[]): Set<number> {
  if (typeof option === 'number') return new Set([option]);
  if (Array.isArray(option)) return new Set(option);
  return new Set(fallback);
}

interface ParsedConfig {
  numHiddenLayers: number;
  numKvSharedLayers: number;
  eosTokenIds: number[];
}

/**
 * The stop set, unioned from every file that names one. Integrator's ruling on the question the
 * tokenizer lane raised, decided the way that lane argued for it.
 *
 * Three sources disagree and none of them is wrong. `tokenizer_config.json` names the two tokens
 * the tokenizer itself knows, ids 1 and 106. `config.json` gives `eos_token_id` as [1, 106].
 * `generation_config.json` gives [1, 106, 50], where 50 is the tool response opener, and it is the
 * only file carrying that third id. A text only v1 that emits a tool response opener has begun a
 * call into a void and should stop rather than narrate one, which makes the union the right set
 * rather than merely the largest one available.
 *
 * A caller's explicit `eosTokenId` still replaces this outright, per the d.ts contract. That is a
 * caller saying it knows better, which is a different thing from three files each knowing a part.
 */
export function unionStopTokens(...sources: readonly (readonly number[])[]): number[] {
  const seen = new Set<number>();
  for (const source of sources) {
    for (const id of source) {
      if (Number.isInteger(id) && id >= 0) seen.add(id);
    }
  }
  return [...seen].sort((a, b) => a - b);
}

/**
 * `fetch` with a bearer token attached, for `Gemma4LoadOptions.accessToken`. Returns the global
 * `fetch` untouched when there is no token, so the ordinary public path adds no wrapper and no
 * header.
 */
export function bearerFetch(accessToken?: string): typeof fetch {
  // Bound, because an unbound global fetch throws "Illegal invocation" the moment it is called
  // through a binding in a browser. See cache.ts defaultFetch.
  if (!accessToken) return defaultFetch();
  return (input: RequestInfo | URL, init?: RequestInit) => {
    const headers = new Headers(init?.headers ?? undefined);
    if (!headers.has('Authorization')) headers.set('Authorization', `Bearer ${accessToken}`);
    return fetch(input, { ...init, headers });
  };
}

/**
 * `generation_config.json`. Optional on purpose: a checkpoint without one is not an error, it just
 * contributes nothing to the union above.
 */
async function fetchGenerationStopIds(root: string, doFetch: typeof fetch, signal?: AbortSignal): Promise<number[]> {
  try {
    const response = await doFetch(`${root}/generation_config.json`, { signal });
    if (!response.ok) return [];
    const raw = (await response.json()) as Record<string, unknown>;
    const eos = raw['eos_token_id'];
    if (typeof eos === 'number') return [eos];
    if (Array.isArray(eos)) return eos.filter((v): v is number => typeof v === 'number');
    return [];
  } catch {
    // A missing or malformed optional file contributes nothing. An abort is caught here too and
    // the load's next awaited step rethrows it, so nothing is swallowed that matters.
    return [];
  }
}

async function fetchConfig(root: string, doFetch: typeof fetch, signal?: AbortSignal): Promise<ParsedConfig> {
  const response = await doFetch(`${root}/config.json`, { signal });
  if (!response.ok) throw new Error(`gemma4 engine: config.json fetch failed with ${response.status}`);
  const raw = (await response.json()) as Record<string, unknown>;
  const text = (raw['text_config'] ?? raw) as Record<string, unknown>;
  const layers = text['num_hidden_layers'];
  const shared = text['num_kv_shared_layers'];
  if (typeof layers !== 'number' || typeof shared !== 'number') {
    throw new Error('gemma4 engine: config.json lacks num_hidden_layers or num_kv_shared_layers');
  }
  const eosRaw = text['eos_token_id'] ?? raw['eos_token_id'];
  const eosTokenIds = typeof eosRaw === 'number' ? [eosRaw] : Array.isArray(eosRaw) ? eosRaw.filter((v): v is number => typeof v === 'number') : [];
  return { numHiddenLayers: layers, numKvSharedLayers: shared, eosTokenIds };
}

export default Gemma4Mobile;
