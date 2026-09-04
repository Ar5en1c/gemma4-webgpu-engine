// Written from docs/ENGINE-PLAN.md sections 3, 5, 5.5 and 6, the architecture and step plan in
// ./plan.ts, the binding seam in ./kernels/binding.ts, the transformers reference implementation's
// decoder layer and per layer input path (Apache-2.0, an allowed source), and the WebGPU
// specification. No vendored bundle, no extracted kernel and no third party engine source was read.
//
// SPDX-License-Identifier: Apache-2.0
//
// THE LIVE FORWARD. Round 1 left a dry walk that recorded a `DispatchStep` sequence against fake
// device objects; this file is what turns one of those steps into a bind group, a pipeline and a
// `dispatchWorkgroups`, and what turns a whole sequence into a submitted command buffer whose only
// readback is the argmax token at the end.
//
// The file is split in two on purpose, and the split is the reason any of this is testable without
// a GPU:
//
//   `resolveStep` is pure. It takes a planned step plus the geometry of the forward it belongs to
//   and returns which registered kernel runs, which buffers it reads by name, which buffer it
//   writes, and the scalar params. It touches no device, allocates nothing and can be walked over
//   all 731 steps of a decode token in Node, which the orchestrator check section does. Every
//   claim this engine makes about which tensor feeds which dispatch is a claim about this function.
//
//   `GpuExecutor` is the impure half. It owns the resources, encodes what `resolveStep` describes,
//   and is as small as that split can make it.
//
// Two rules from the plan bind everything here. Accumulators are f32 or i32 and never f16, which is
// a property of the kernels rather than of this file but is restated where the executor chooses
// buffer element types: LlamaWeb (arXiv 2605.20706) measured f16 accumulation producing incoherent
// output on Apple M-series, and this engine stores f32 activations for that reason. And a step
// whose kernel is in `PENDING_KERNELS` is refused with the family it wanted named in the message,
// never substituted with a near neighbour, because a 4-bit unpacker pointed at an I8 tensor
// produces plausible numbers rather than an error.

import {
  ROLE_GELU_DOWN,
  ROLE_GELU_PLE,
  ROLE_JOIN_FINAL,
  ROLE_JOIN_MLP,
  ROLE_JOIN_TAIL,
  GEMMA4_E2B,
  PENDING_KERNELS,
  PER_LAYER_INPUT_SCALE,
  headDimForLayer,
  isGlobalLayer,
  mlpIntermediateForLayer,
  perLayerProjectionScale,
  type DispatchStep,
  type Gemma4Arch,
} from './plan';
import { kvRoleForLayer, type KvLayout } from './kv';
import { FINAL_LOGIT_SOFTCAP } from './kernels/logitSoftcap';
import { ARGMAX_ELEMS_PER_WORKGROUP } from './kernels/argmax';
import { GEMV_WIDE_COLS } from './kernels/qgemvWide';
import { PLE_GATHER_SLICED_NAME } from './kernels/embedGather';
import { sliceName } from './buffers';

// ------------------------------------------------------------------------------- buffer names

/**
 * Where a binding's bytes come from. Names are namespaced by `kind` rather than by a string
 * prefix, so a typo in a slot key cannot silently resolve as a tensor name.
 *
 *   weight    a checkpoint tensor, resident packed, addressed by its own name
 *   slot      a scratch activation buffer this executor owns, addressed by a stable key
 *   residual  the ping-pong pair carrying the hidden state, whichever half the step needs
 *   kv        one producer layer's packed K and V cache, addressed by the producer's layer index
 */
export type BufferRefKind = 'weight' | 'slot' | 'residual-read' | 'residual-write' | 'kv';

export interface BufferRef {
  readonly kind: BufferRefKind;
  /** Tensor name, slot key, or the producer layer index as text for a kv cache. */
  readonly name: string;
  /** Element count the slot must hold. Zero for weights and caches, which are sized elsewhere. */
  readonly elements: number;
}

/** Bytes per activation element. f32 everywhere, never f16. See the file header. */
export const ACTIVATION_BYTES = 4;

/**
 * A slice moved into a slot of its own before a dispatch reads it.
 *
 * Two sites in the forward want a strided or offset view of a buffer, and the kernel seam cannot
 * carry one: `KernelBindInput.inputs` is a map of whole `GPUBuffer`s, so a kernel's `bind` has no
 * way to say "this binding starts 6144 bytes in". The alternative to a copy would be widening that
 * seam and then rewriting fourteen `bind` functions around it, which is the kernel lanes' file set
 * and not this one's. A buffer to buffer copy is a few hundred bytes of traffic on a step that
 * moves megabytes, it is encoded outside the compute pass where it is legal, and it keeps the
 * seam's one shape.
 */
export interface SliceCopy {
  readonly from: BufferRef;
  readonly fromOffset: number;
  readonly to: BufferRef;
  readonly toOffset: number;
  readonly elements: number;
  readonly why: string;
}

function slot(name: string, elements: number): BufferRef {
  return { kind: 'slot', name, elements };
}

function weight(name: string): BufferRef {
  return { kind: 'weight', name, elements: 0 };
}

/** One resolved dispatch: everything the executor needs and nothing about the device. */
export interface ResolvedStep {
  readonly step: DispatchStep;
  /** Registered kernel name. Never a PENDING_KERNELS member; those throw in `resolveStep`. */
  readonly kernel: string;
  /** Input buffers by the binding name the kernel's `bind` declares. */
  readonly inputs: Readonly<Record<string, BufferRef>>;
  readonly output: BufferRef;
  readonly params: Readonly<Record<string, number>>;
  /** Slices staged into their own slots before the dispatch, encoded outside the compute pass. */
  readonly copies: readonly SliceCopy[];
  /** True when the step's output is the residual stream, so the executor flips the pair after it. */
  readonly flipsResidual: boolean;
}

// ------------------------------------------------------------------------ forward geometry

export interface ForwardGeometry {
  readonly arch: Gemma4Arch;
  /** 'gemv' for a decode step, 'gemm' for a prefill chunk. Matches the step plan's mode. */
  readonly mode: 'gemv' | 'gemm';
  /** Tokens in this chunk. 1 on a decode step. */
  readonly tokens: number;
  /** Absolute position of this chunk's first token. */
  readonly startPosition: number;
  /** Cache capacity in positions, from kv.ts's layout. */
  readonly maxContext: number;
  /**
   * A checkpoint scalar by tensor name: the per layer `layer_scalar` and the two activation scales
   * of every quantized linear. These are four byte tensors that belong in a uniform rather than in
   * a storage buffer, so the loader keeps them on the host. A name with no stored value reads 0,
   * which for an activation scale is exactly the checkpoint's own way of saying "uncalibrated".
   */
  scalar(name: string): number;
  /**
   * Decode only: the two embedding gathers read their one token id from the `token` slot the
   * previous step's argmax wrote, instead of from the `ids` slot the host uploads. This is what
   * lets the executor submit the next step before the previous token has been read back
   * (runtime.ts decodeAhead): the token never leaves the GPU on its way to the next gather. The
   * slot holds a u32 at element 0, which is exactly what the gathers index with.
   */
  readonly idsFromTokenSlot?: boolean;
}

/** Valid cache positions once this chunk's KV is stored, which is what attention is told. */
export function kvLengthAfter(geometry: ForwardGeometry): number {
  return geometry.startPosition + geometry.tokens;
}

// ------------------------------------------------------------------------------ tensor names

const LM = 'model.language_model';

function layerTensor(layer: number, suffix: string): string {
  return `${LM}.layers.${layer}.${suffix}`;
}

/**
 * The module a planned step's `role` names, as the path the checkpoint stores it under. This is the
 * whole mapping from the plan's prose roles to the checkpoint, in one table, so a role rename that
 * silently detaches a dispatch from its weights fails the orchestrator's cross reference instead of
 * running against the wrong tensor.
 */
export const ROLE_MODULE: Readonly<Record<string, string>> = Object.freeze({
  q_proj: 'self_attn.q_proj',
  k_proj: 'self_attn.k_proj',
  v_proj: 'self_attn.v_proj',
  o_proj: 'self_attn.o_proj',
  gate_proj: 'mlp.gate_proj',
  up_proj: 'mlp.up_proj',
  down_proj: 'mlp.down_proj',
  per_layer_input_gate: 'per_layer_input_gate',
  per_layer_projection: 'per_layer_projection',
});

/** The norm gain a norm shaped role reads, as a tensor name relative to its layer. */
export const ROLE_NORM_GAIN: Readonly<Record<string, string>> = Object.freeze({
  input_layernorm: 'input_layernorm.weight',
  'q_norm per head': 'self_attn.q_norm.weight',
  'k_norm per head': 'self_attn.k_norm.weight',
  'post_attention_layernorm plus residual': 'post_attention_layernorm.weight',
  pre_feedforward_layernorm: 'pre_feedforward_layernorm.weight',
  'post_feedforward_layernorm plus residual': 'post_feedforward_layernorm.weight',
  'post_per_layer_input_norm plus residual': 'post_per_layer_input_norm.weight',
});

/**
 * The head's weight names. `lm_head.weight` and `lm_head.weight_scale` are byte identical to the
 * embed_tokens pair in this checkpoint and the loader never puts them on the wire
 * (safetensors.ts TIED_TENSORS), so the head reads the embedding tensors by their own names and the
 * tie is a fact about the download rather than a special case in the forward.
 */
export const LM_HEAD_CODES = `${LM}.embed_tokens.embedding_quantized`;
export const LM_HEAD_SCALES = `${LM}.embed_tokens.embedding_scale`;

/**
 * The PLE table's two tensors, named once so the loader's split and this file's gather cannot
 * disagree about which tensor is the 1,174,405,120 byte one.
 */
export const PLE_CODES = `${LM}.embed_tokens_per_layer.embedding_quantized`;
export const PLE_SCALES = `${LM}.embed_tokens_per_layer.embedding_scale`;

// -------------------------------------------------------------------------------- slot sizing

/**
 * Every activation slot the forward uses, in elements, at the widest geometry any layer asks for.
 * Sized once from the architecture rather than grown per layer: the seven global layers are the
 * widest attention shape and the twenty consumer layers are the widest MLP, so one sizing pass over
 * the maxima leaves nothing to resize mid forward, which is what makes `BufferManager.storage`
 * able to refuse a later larger ask as a plan bug.
 */
/** slotElements per (arch, tokens), memoized: resolveStep runs 626 times a token and the table is the same each time. */
const SLOT_SIZES = new WeakMap<Gemma4Arch, Map<number, Record<string, number>>>();
function slotElementsCached(arch: Gemma4Arch, maxTokens: number): Record<string, number> {
  let byTokens = SLOT_SIZES.get(arch);
  if (!byTokens) {
    byTokens = new Map();
    SLOT_SIZES.set(arch, byTokens);
  }
  let sizes = byTokens.get(maxTokens);
  if (!sizes) {
    sizes = slotElements(arch, maxTokens);
    byTokens.set(maxTokens, sizes);
  }
  return sizes;
}

export function slotElements(arch: Gemma4Arch, maxTokens: number): Record<string, number> {
  const t = Math.max(1, maxTokens);
  const wideHead = arch.globalHeadDim;
  const q = arch.queryHeads * wideHead;
  const inter = arch.consumerIntermediate;
  const pleRow = arch.layerCount * arch.pleDim;
  const argmaxPairs = Math.ceil(arch.vocabSize / ARGMAX_ELEMS_PER_WORKGROUP) * 2;
  return {
    normed: t * arch.hiddenSize,
    q: t * q,
    qn: t * q,
    qr: t * q,
    k: t * wideHead,
    kn: t * wideHead,
    kr: t * wideHead,
    v: t * wideHead,
    vn: t * wideHead,
    attn: t * q,
    proj: t * arch.hiddenSize,
    gate: t * inter,
    up: t * inter,
    act: t * inter,
    'ple.identity': t * pleRow,
    'ple.projection': t * pleRow,
    'ple.normed': t * pleRow,
    'ple.sum': t * pleRow,
    'ple.input': t * pleRow,
    'ple.gate': t * arch.pleDim,
    'ple.act': t * arch.pleDim,
    // One layer group of the per layer input, gathered out of the [tokens, 35, 256] tensor.
    'ple.row': t * arch.pleDim,
    // The last position's hidden state, which is the only row the head reads.
    'head.in': arch.hiddenSize,
    // The additive identity, never written. WebGPU zero initializes every buffer it creates, so a
    // slot nothing writes is a zero vector by construction, which is what turns scale-add's
    // alpha * x + y into a plain scale.
    zero: Math.max(t * pleRow, t * arch.hiddenSize, t * inter),
    ids: t,
    // Two table pairs, not one: the sliding layers rotate 128 pairs at theta 1e4 over head
    // dimension 256, the seven global layers rotate 64 of 256 pairs at theta 1e6 over head
    // dimension 512 with the rest carrying cosine 1 and sine 0 (ENGINE-PLAN risk 1 quirk 3, and
    // the proportional frequency ladder of the round 1 corrections). One shared pair would make a
    // global layer read a sliding table, which is the shape of that trap exactly.
    'rope.cos.sliding': t * (arch.slidingHeadDim / 2),
    'rope.sin.sliding': t * (arch.slidingHeadDim / 2),
    'rope.cos.global': t * (wideHead / 2),
    'rope.sin.global': t * (wideHead / 2),
    // The head slots carry one row per position of a verify pass (GEMV_WIDE_COLS at most); a
    // decode token and a prefill chunk use row 0.
    logits: GEMV_WIDE_COLS * arch.vocabSize,
    'logits.capped': GEMV_WIDE_COLS * arch.vocabSize,
    'argmax.partials': GEMV_WIDE_COLS * argmaxPairs,
    token: 2 * GEMV_WIDE_COLS,
  };
}

// ----------------------------------------------------------------------------- step resolution

export class PendingKernelError extends Error {
  readonly kernel: string;
  readonly role: string;

  constructor(kernel: string, role: string, detail: string) {
    super(
      `gemma4 engine: the forward reached ${role}, which needs the ${kernel} family, and that `
      + `family is not in the kernel registry yet. ${detail}`,
    );
    this.name = 'PendingKernelError';
    this.kernel = kernel;
    this.role = role;
  }
}

/**
 * Empty, because `PENDING_KERNELS` is empty: both families this table used to explain have landed
 * in the registry. It stays as the seam a future entry writes into, so a step that refuses says
 * what it wanted and why rather than naming a kernel and stopping.
 */
const PENDING_DETAIL: Readonly<Record<string, string>> = Object.freeze({});

function requireRegistered(kernel: string, role: string): void {
  if (PENDING_KERNELS.includes(kernel)) {
    throw new PendingKernelError(kernel, role, PENDING_DETAIL[kernel] ?? 'See plan.ts PENDING_KERNELS.');
  }
}

/** The quantized matmul params a linear site passes, SRQ scales included. */
function linearParams(
  geometry: ForwardGeometry,
  module: string,
  k: number,
  numRows: number,
): Record<string, number> {
  const params: Record<string, number> = {
    k,
    numRows,
    // The static range quantize scales of ENGINE-PLAN's round 1 correction 6. They travel with
    // every quantized matmul step whether or not the kernel reads them yet, so closing a site is a
    // change in the kernel and not a change here. quant.ts is the authority for what they mean:
    // 8 bits on both sides, skipped only when the stored scale is exactly 0.0.
    inScale: geometry.scalar(`${module}.input_activation_scale`),
    outScale: geometry.scalar(`${module}.output_activation_scale`),
  };
  if (geometry.mode === 'gemm') params.mCols = geometry.tokens;
  else if (geometry.tokens > 1) params.cols = geometry.tokens;
  return params;
}

/** The I8 family's shape by mode: the GEMV on a one token decode step, the per element matmul over a chunk. */
function pleKernel(geometry: ForwardGeometry, stepKernel: string): string {
  if (stepKernel !== 'qmatmul-8bit') return gemvWide(geometry, stepKernel);
  return geometry.mode === 'gemv' ? gemvWide(geometry, 'qgemv-8bit') : stepKernel;
}

/**
 * A verify pass: a gemv mode step over more than one position runs the M wide form of every
 * decode GEMV (kernels/qgemvWide.ts), column for column the one column kernel, with
 * `params.cols` the position count. A one token step keeps the one column kernel.
 */
function gemvWide(geometry: ForwardGeometry, kernel: string): string {
  if (geometry.mode !== 'gemv' || geometry.tokens === 1 || !kernel.startsWith('qgemv-')) return kernel;
  if (geometry.tokens > GEMV_WIDE_COLS) {
    throw new Error(`gemma4 engine: a verify pass carries at most ${GEMV_WIDE_COLS} positions, got ${geometry.tokens}`);
  }
  return `${kernel}-m${GEMV_WIDE_COLS}`;
}

function matmulKernel(geometry: ForwardGeometry, stepKernel: string): string {
  // The plan already chose the family and the mode; this only asserts the two agree, because a
  // gemv kernel handed a multi token chunk writes one row and leaves the rest as noise.
  const wantsGemm = geometry.mode === 'gemm';
  if (wantsGemm && stepKernel.startsWith('qgemv-')) {
    return stepKernel.replace('qgemv-', 'qgemm-');
  }
  return gemvWide(geometry, stepKernel);
}

/**
 * Resolve one planned step into the dispatch that runs it. Pure.
 *
 * Throws `PendingKernelError` for a step whose family has not landed, and a plain Error for a role
 * the tables above do not cover, which is the case a plan edit that invents a role hits.
 */
export function resolveStep(step: DispatchStep, geometry: ForwardGeometry): ResolvedStep {
  const { arch, tokens } = geometry;
  const hidden = arch.hiddenSize;
  const pleRow = arch.layerCount * arch.pleDim;
  const layer = step.layer;
  const headDim = layer >= 0 ? headDimForLayer(arch, layer) : arch.slidingHeadDim;
  const heads = arch.queryHeads;
  const inter = layer >= 0 ? mlpIntermediateForLayer(arch, layer) : arch.producerIntermediate;
  const ropeKind = layer >= 0 && isGlobalLayer(arch, layer) ? 'global' : 'sliding';
  const sizes = slotElementsCached(arch, tokens);
  const s = (key: string): BufferRef => slot(key, sizes[key] ?? 0);
  // Logit rows the head phase produces: every position of a verify pass, the last one otherwise.
  const headRows = geometry.mode === 'gemv' ? tokens : 1;
  // Where the gathers read their token ids: the host's upload, or on a lookahead decode step the
  // token slot the previous step's argmax wrote (ForwardGeometry.idsFromTokenSlot). A prefill
  // chunk has many ids and always reads the upload; the flag is refused there rather than ignored.
  const idsRef = (): BufferRef => {
    if (!geometry.idsFromTokenSlot) return s('ids');
    if (geometry.mode !== 'gemv' || tokens !== 1) {
      throw new Error('gemma4 engine: idsFromTokenSlot is a one token decode step property, not a prefill chunk one');
    }
    return s('token');
  };
  const base = (suffix: string): string => layerTensor(layer, suffix);
  const residual = (write: boolean): BufferRef => ({
    kind: write ? 'residual-write' : 'residual-read',
    name: 'hidden',
    elements: tokens * hidden,
  });

  requireRegistered(step.kernel, step.role);

  const make = (
    kernel: string,
    inputs: Record<string, BufferRef>,
    output: BufferRef,
    params: Record<string, number>,
    extra?: { copies?: SliceCopy[]; flipsResidual?: boolean },
  ): ResolvedStep => ({
    step,
    kernel,
    inputs,
    output,
    params,
    copies: extra?.copies ?? [],
    flipsResidual: extra?.flipsResidual ?? false,
  });

  switch (step.role) {
    // ------------------------------------------------------------------ the embed phase
    case 'embed_tokens gather':
      return make('embed-gather-2bit', {
        codes: weight(`${LM}.embed_tokens.embedding_quantized`),
        scales: weight(`${LM}.embed_tokens.embedding_scale`),
        ids: idsRef(),
      }, residual(true), { slots: tokens }, { flipsResidual: true });

    case 'per layer embedding gather': {
      const codesName = PLE_CODES;
      // On an adapter with room for the whole 1,174,405,120 byte table this is the single step it
      // has always been and this branch is not taken. Where the table is split, planEmbed has
      // already emitted one step per vocabulary range, each carrying the range it serves, so this
      // stays one step to one dispatch and the decode loop above never learns the difference.
      const slice = step.gatherSlice;
      if (slice) {
        return make(PLE_GATHER_SLICED_NAME, {
          codes: weight(sliceName(codesName, slice.index)),
          // Scales stay one buffer and stay addressed by the absolute vocabulary row, which is what
          // the sliced shader assumes and what tableSplit.ts's `scalesSingle` checks.
          scales: weight(PLE_SCALES),
          ids: idsRef(),
        }, s('ple.identity'), {
          slots: tokens,
          sliceStartRow: slice.startRow,
          sliceRowCount: slice.rowCount,
        });
      }
      return make('ple-gather-4bit', {
        codes: weight(codesName),
        scales: weight(PLE_SCALES),
        ids: idsRef(),
      }, s('ple.identity'), { slots: tokens });
    }

    case 'per_layer_model_projection': {
      // The context aware half of the per layer input, and the one unquantized linear a forward
      // multiplies by. The reference scales this module's output by hidden_size ** -0.5 BEFORE the
      // per layer group norm sees it, and RMS norm is not scale invariant once its epsilon is in
      // play, so the multiply cannot be deferred past the norm. It rides in this dispatch's uniform
      // rather than costing a scale-add of its own (plan.ts planEmbed).
      const params: Record<string, number> = {
        k: hidden,
        numRows: pleRow,
        alpha: perLayerProjectionScale(arch),
      };
      if (geometry.mode === 'gemm' || tokens > 1) params.mCols = tokens;
      return make('dense-bf16-matmul', {
        w: weight(`${LM}.per_layer_model_projection.weight`),
        x: residual(false),
      }, s('ple.projection'), params);
    }

    case 'per_layer_projection_norm':
      // 35 rows of 256 per token, not one row of 8960: per_layer_projection_norm is sized
      // hidden_size_per_layer_input in the reference, so each layer group normalizes on its own.
      return make('rms-norm', {
        src: s('ple.projection'),
        gain: weight(`${LM}.per_layer_projection_norm.weight`),
      }, s('ple.normed'), { rows: tokens * arch.layerCount, width: arch.pleDim, eps: arch.rmsEps });

    case 'per layer inputs add':
      return make('scale-add', { x: s('ple.normed'), y: s('ple.identity') }, s('ple.sum'), {
        n: tokens * pleRow, alpha: 1,
      });

    case 'per layer inputs scale':
      return make('scale-add', { x: s('ple.sum'), y: s('zero') }, s('ple.input'), {
        n: tokens * pleRow, alpha: PER_LAYER_INPUT_SCALE,
      });

    // ------------------------------------------------------------------ the attention block
    case 'input_layernorm':
    case 'pre_feedforward_layernorm':
      return make('rms-norm', {
        src: residual(false),
        gain: weight(base(ROLE_NORM_GAIN[step.role]!)),
      }, s('normed'), { rows: tokens, width: hidden, eps: arch.rmsEps });

    case 'q_proj':
    case 'k_proj':
    case 'v_proj': {
      const module = ROLE_MODULE[step.role]!;
      const rows = step.role === 'q_proj' ? heads * headDim : headDim;
      const out = step.role === 'q_proj' ? 'q' : step.role === 'k_proj' ? 'k' : 'v';
      return make(matmulKernel(geometry, step.kernel), {
        wq: weight(base(`${module}.weight`)),
        scales: weight(base(`${module}.weight_scale`)),
        x: s('normed'),
      }, s(out), linearParams(geometry, base(module), hidden, rows));
    }

    case 'q_norm per head':
      return make('rms-norm', {
        src: s('q'),
        gain: weight(base(ROLE_NORM_GAIN[step.role]!)),
      }, s('qn'), { rows: tokens * heads, width: headDim, eps: arch.rmsEps });

    case 'k_norm per head':
      return make('rms-norm', {
        src: s('k'),
        gain: weight(base(ROLE_NORM_GAIN[step.role]!)),
      }, s('kn'), { rows: tokens, width: headDim, eps: arch.rmsEps });

    case 'v_norm weightless':
      // ENGINE-PLAN risk 1 quirk 1. One KV head, so one row per token, and no gain binding at all
      // because there is no v_norm.weight tensor to bind.
      return make('rms-norm-weightless', { src: s('v') }, s('vn'), {
        rows: tokens, width: headDim, eps: arch.rmsEps,
      });

    case 'rope q':
      return make('rope', {
        src: s('qn'), cosTab: s(`rope.cos.${ropeKind}`), sinTab: s(`rope.sin.${ropeKind}`),
      }, s('qr'), { rows: tokens * heads, pairCount: headDim / 2, headsPerPosition: heads });

    case 'rope k':
      return make('rope', {
        src: s('kn'), cosTab: s(`rope.cos.${ropeKind}`), sinTab: s(`rope.sin.${ropeKind}`),
      }, s('kr'), { rows: tokens, pairCount: headDim / 2, headsPerPosition: 1 });

    case 'kv store':
      return make('kv-cache-store', { k: s('kr'), v: s('vn') }, {
        kind: 'kv', name: String(layer), elements: 0,
      }, {
        headDim,
        startPos: geometry.startPosition,
        tokenCount: tokens,
        maxContext: geometry.maxContext,
      });

    case 'attention': {
      const role = kvRoleForLayer(layer);
      const cacheLayer = role.kind === 'producer' ? role.layer : role.readsFrom;
      return make(step.kernel, {
        q: s('qr'),
        cache: { kind: 'kv', name: String(cacheLayer), elements: 0 },
      }, s('attn'), {
        headDim,
        heads,
        qCount: tokens,
        qStart: geometry.startPosition,
        kvLen: kvLengthAfter(geometry),
        window: isGlobalLayer(arch, layer) ? 0 : arch.slidingWindow,
        maxContext: geometry.maxContext,
      });
    }

    case 'o_proj':
      return make(matmulKernel(geometry, step.kernel), {
        wq: weight(base('self_attn.o_proj.weight')),
        scales: weight(base('self_attn.o_proj.weight_scale')),
        x: s('attn'),
      }, s('proj'), linearParams(geometry, base('self_attn.o_proj'), heads * headDim, hidden));

    // The fused joins (kernels/blockJoin.ts). `hidden` is the residual write side and the
    // kernel's `normed` output is the next block's input; the flip is on the residual, which
    // the runtime finds by kind rather than by assuming it is the output.
    case ROLE_JOIN_MLP:
      return make('norm-residual-norm', {
        src: s('proj'),
        gain: weight(base('post_attention_layernorm.weight')),
        residual: residual(false),
        gain2: weight(base('pre_feedforward_layernorm.weight')),
        hidden: residual(true),
      }, s('normed'), { rows: tokens, width: hidden, eps: arch.rmsEps, alpha: 1 }, { flipsResidual: true });

    case ROLE_JOIN_TAIL:
    case ROLE_JOIN_FINAL:
      // Trap 5 rides here: alpha is the layer's own `layer_scalar` from the checkpoint.
      return make('norm-residual-norm', {
        src: s('proj'),
        gain: weight(base('post_per_layer_input_norm.weight')),
        residual: residual(false),
        gain2: step.role === ROLE_JOIN_FINAL
          ? weight(`${LM}.norm.weight`)
          : weight(layerTensor(layer + 1, 'input_layernorm.weight')),
        hidden: residual(true),
      }, s('normed'), {
        rows: tokens, width: hidden, eps: arch.rmsEps, alpha: geometry.scalar(base('layer_scalar')),
      }, { flipsResidual: true });

    case 'post_attention_layernorm plus residual':
    case 'post_feedforward_layernorm plus residual':
    case 'post_per_layer_input_norm plus residual':
      return make('norm-residual', {
        src: s('proj'),
        gain: weight(base(ROLE_NORM_GAIN[step.role]!)),
        residual: residual(false),
      }, residual(true), { rows: tokens, width: hidden, eps: arch.rmsEps }, { flipsResidual: true });

    // ------------------------------------------------------------------------- the MLP block
    case 'gate_proj':
    case 'up_proj': {
      const module = ROLE_MODULE[step.role]!;
      return make(matmulKernel(geometry, step.kernel), {
        wq: weight(base(`${module}.weight`)),
        scales: weight(base(`${module}.weight_scale`)),
        x: s('normed'),
      }, s(step.role === 'gate_proj' ? 'gate' : 'up'), linearParams(geometry, base(module), hidden, inter));
    }

    case 'gelu times up':
      // The down projection's input rounding rides in this pass rather than in a dispatch of its
      // own. The down_proj matmul's own fused prologue snaps the same values again, which is
      // idempotent on the grid (plan.ts SRQ_OPEN_SITES carries the rule).
      return make('gelu-mul', { gate: s('gate'), up: s('up') }, s('act'), {
        count: tokens * inter,
        srqScale: geometry.scalar(base('mlp.down_proj.input_activation_scale')),
      });

    case ROLE_GELU_DOWN:
      // The fused K10 prologue (decode only): the GEMV reads gelu(gate) * up itself, snapped by its
      // own input scale, which is the scale the separate pass carried.
      return make(gemvWide(geometry, step.kernel), {
        wq: weight(base('mlp.down_proj.weight')),
        scales: weight(base('mlp.down_proj.weight_scale')),
        gate: s('gate'),
        up: s('up'),
      }, s('proj'), linearParams(geometry, base('mlp.down_proj'), inter, hidden));

    case 'down_proj':
      return make(matmulKernel(geometry, step.kernel), {
        wq: weight(base('mlp.down_proj.weight')),
        scales: weight(base('mlp.down_proj.weight_scale')),
        x: s('act'),
      }, s('proj'), linearParams(geometry, base('mlp.down_proj'), inter, hidden));

    // ---------------------------------------------------------------------- the PLE inject
    //
    // Both linears here are quant.ts's I8 `ple-gate-8bit` family. A prefill chunk runs
    // `qmatmul-8bit`, one invocation per output element over the chunk's columns; a decode step
    // runs the same two linears as `qgemv-8bit` on the GEMV geometry (round 4 lever 3), because
    // the per element shape at one column was 4 workgroups of serial K reading 14 MB per token at
    // 5.6 GB/s. The plan names the family and this file picks the shape by mode, the same split
    // `matmulKernel` makes for the 2-bit and 4-bit families. Their SRQ scales come from
    // `linearParams` like every other quantized linear's, which is what keeps closing a site a
    // property of the kernel rather than of this file.
    case 'per_layer_input_gate': {
      const module = ROLE_MODULE[step.role]!;
      return make(pleKernel(geometry, step.kernel), {
        wq: weight(base(`${module}.weight`)),
        scales: weight(base(`${module}.weight_scale`)),
        x: residual(false),
      }, s('ple.gate'), linearParams(geometry, base(module), hidden, arch.pleDim));
    }

    case 'per_layer_projection': {
      const module = ROLE_MODULE[step.role]!;
      // Writes `proj`, which is the slot `post_per_layer_input_norm plus residual` reads, the same
      // slot o_proj and down_proj hand to their own epilogues.
      return make(pleKernel(geometry, step.kernel), {
        wq: weight(base(`${module}.weight`)),
        scales: weight(base(`${module}.weight_scale`)),
        x: s('ple.act'),
      }, s('proj'), linearParams(geometry, base(module), arch.pleDim, hidden));
    }

    case ROLE_GELU_PLE: {
      // The same gather as the unfused step below, then the I8 GEMV over gelu(gate) * row.
      const module = ROLE_MODULE['per_layer_projection']!;
      const dim = arch.pleDim;
      const copies: SliceCopy[] = [];
      for (let t = 0; t < tokens; t += 1) {
        copies.push({
          from: s('ple.input'),
          fromOffset: (t * arch.layerCount + layer) * dim,
          to: s('ple.row'),
          toOffset: t * dim,
          elements: dim,
          why: `layer ${layer}'s group of token ${t}'s per layer input`,
        });
      }
      return make(gemvWide(geometry, step.kernel), {
        wq: weight(base(`${module}.weight`)),
        scales: weight(base(`${module}.weight_scale`)),
        gate: s('ple.gate'),
        up: s('ple.row'),
      }, s('proj'), linearParams(geometry, base(module), dim, hidden), { copies });
    }

    case 'gate times per layer row': {
      // The per layer input for layer L lives at columns L * 256 of a [tokens, 35, 256] tensor, so
      // one token's group is contiguous and a chunk's is not. The groups are gathered into a dense
      // [tokens, 256] slot first and the multiply then runs once over the whole chunk.
      const dim = arch.pleDim;
      const copies: SliceCopy[] = [];
      for (let t = 0; t < tokens; t += 1) {
        copies.push({
          from: s('ple.input'),
          fromOffset: (t * arch.layerCount + layer) * dim,
          to: s('ple.row'),
          toOffset: t * dim,
          elements: dim,
          why: `layer ${layer}'s group of token ${t}'s per layer input`,
        });
      }
      return make('gelu-mul', { gate: s('ple.gate'), up: s('ple.row') }, s('ple.act'), {
        count: tokens * dim,
        srqScale: geometry.scalar(base('per_layer_projection.input_activation_scale')),
      }, { copies });
    }

    // --------------------------------------------------------------------------- the head
    case 'final norm':
      // Only the last position's logits are ever read, so the head phase runs on one row whatever
      // the chunk length was, over a copy of the residual stream's last row.
      return make('rms-norm', {
        src: s('head.in'),
        gain: weight(`${LM}.norm.weight`),
      }, s('normed'), { rows: 1, width: hidden, eps: arch.rmsEps }, {
        copies: [{
          from: residual(false),
          fromOffset: (tokens - 1) * hidden,
          to: s('head.in'),
          toOffset: 0,
          elements: hidden,
          why: 'the head reads the last position only',
        }],
      });

    case 'lm_head': {
      // `x` binds the whole `normed` slot and the kernel reads its first 1536 values, which is the
      // one row the final norm just wrote. Nothing downstream reads the rest of that slot. On a
      // verify pass the tail join wrote one normed row per position and the M wide head reads
      // them all, one logits row each.
      const headParams: Record<string, number> = {
        k: hidden,
        numRows: arch.vocabSize,
        inScale: geometry.scalar('lm_head.input_activation_scale'),
        outScale: geometry.scalar('lm_head.output_activation_scale'),
      };
      if (headRows > 1) headParams.cols = headRows;
      return make(gemvWide(geometry, 'qgemv-2bit'), {
        wq: weight(LM_HEAD_CODES),
        scales: weight(LM_HEAD_SCALES),
        x: s('normed'),
      }, s('logits'), headParams);
    }

    case 'tanh(logits / 30) * 30':
      return make('logit-softcap', { src: s('logits') }, s('logits.capped'), {
        count: headRows * arch.vocabSize,
        cap: arch.finalSoftcap === 0 ? 0 : FINAL_LOGIT_SOFTCAP,
      });

    case 'argmax partials':
      return make('argmax-partial', { logits: s('logits.capped') }, s('argmax.partials'), {
        count: arch.vocabSize,
        elemsPerWg: ARGMAX_ELEMS_PER_WORKGROUP,
        rows: headRows,
      });

    case 'argmax final':
      return make('argmax-final', { partials: s('argmax.partials') }, s('token'), {
        pairCount: Math.ceil(arch.vocabSize / ARGMAX_ELEMS_PER_WORKGROUP),
        rows: headRows,
      });

    // ------------------------------------------------------------------- the layer scalar
    case 'layer_scalar':
      // Trap 5. `hidden_states *= layer_scalar`, a per layer BF16 value that is 0.0272 on layer 0,
      // expressed as alpha * x + 0 against the never written zero slot.
      return make('scale-add', { x: residual(false), y: s('zero') }, residual(true), {
        n: tokens * hidden,
        alpha: geometry.scalar(base('layer_scalar')),
      }, { flipsResidual: true });

    default:
      throw new Error(
        `gemma4 engine: no dispatch is defined for the planned role "${step.role}" `
        + `(kernel ${step.kernel}, phase ${step.phase}, layer ${step.layer}). Either plan.ts grew a `
        + 'step this executor does not know, or a role was renamed on one side only.',
      );
  }
}

/**
 * Every checkpoint tensor a forward binds, derived from the step plan rather than listed. The
 * loader's allow list is built from the architecture and this set is built from the dispatches, so
 * comparing the two is a real cross check: a tensor the forward wants and the loader never fetches
 * shows up as a name in here that is not in there.
 */
export function weightsReferencedBy(
  steps: readonly DispatchStep[],
  geometry: ForwardGeometry,
): string[] {
  const names = new Set<string>();
  for (const step of steps) {
    let resolved: ResolvedStep;
    try {
      resolved = resolveStep(step, geometry);
    } catch (err) {
      if (err instanceof PendingKernelError) continue;
      throw err;
    }
    for (const ref of Object.values(resolved.inputs)) {
      if (ref.kind === 'weight') names.add(ref.name);
    }
    if (resolved.output.kind === 'weight') names.add(resolved.output.name);
  }
  return [...names].sort();
}
