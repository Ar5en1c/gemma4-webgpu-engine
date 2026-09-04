// Written from docs/ENGINE-PLAN.md sections 2, 3, 5 and 6, the API contract in
// src/engine/llm/gemma4-engine.d.ts, and the measurement record in
// ../../../../gemma4-kernels-lab/DECODE-CAMPAIGN.md and DECODE-FUSION-FINDING.md. No vendored
// bundle, no extracted kernel and no third party engine source was read.
//
// SPDX-License-Identifier: Apache-2.0
//
// The orchestrator's pure half. Everything in this file is arithmetic over plain values: which
// dispatches a forward pass consists of, how a prompt splits into prefill chunks, what a progress
// event looks like, and when the greedy loop stops. Nothing here touches a GPU, the DOM or the
// network, and the file deliberately has no imports at all so that scripts/engine-check.mjs can
// transpile and run it in Node as it is. engine.ts composes these functions with the device, the
// buffers and the kernels; the orchestrator check section runs them against fakes.
//
// One duplication is accepted knowingly and checked mechanically: the architecture constants below
// repeat values that kernels/layerGeometry.ts also exports for the norm and rope family. This file
// cannot import that one and stay Node loadable through the check runner's single file transpile,
// so the orchestrator check section loads both modules and asserts they agree, which turns the
// duplication into a tested invariant rather than a drift hazard. If the lead hoists the
// architecture into one shared module, this table is the candidate.

// ------------------------------------------------------------------------------ architecture

/**
 * The architecture contract of ENGINE-PLAN section 5, as typed data. Values were read from
 * config.json and the Hugging Face transformers reference implementation at M0 and cross checked
 * against the lab's census. Disagreements resolve in favour of the config.
 */
export interface Gemma4Arch {
  readonly layerCount: number;
  readonly hiddenSize: number;
  readonly queryHeads: number;
  readonly kvHeads: number;
  /** Head dimension on sliding layers. The census's 256 is this value, not a universal one. */
  readonly slidingHeadDim: number;
  /** Head dimension on the seven full attention layers. */
  readonly globalHeadDim: number;
  /** Full attention layer indices. One in five, last layer forced global. */
  readonly globalLayers: readonly number[];
  /** Window in positions, inclusive of the current one, so [q - 511, q]. */
  readonly slidingWindow: number;
  /** Layers 0..kvProducerLayers-1 own KV caches, the rest read a producer's. */
  readonly kvProducerLayers: number;
  /** MLP intermediate width on producer layers. */
  readonly producerIntermediate: number;
  /** MLP intermediate width on consumer layers, double the producer width. */
  readonly consumerIntermediate: number;
  readonly vocabSize: number;
  /** Per layer input dimension of the PLE path: 35 groups of 256 in the table row. */
  readonly pleDim: number;
  readonly rmsEps: number;
  /** Final logit softcap. There is no attention logit softcapping in the text model. */
  readonly finalSoftcap: number;
  /** RoPE theta on sliding layers; all 128 pairs rotate. */
  readonly ropeThetaSliding: number;
  /** RoPE theta on global layers; only partialRotaryFactor of the 256 pairs rotate. */
  readonly ropeThetaGlobal: number;
  /** 0.25 on global layers: 64 of 256 frequency pairs rotate, the rest are identity. */
  readonly partialRotaryFactor: number;
}

/** google/gemma-4-E2B-it-qat-mobile-transformers, per the ENGINE-PLAN section 5 table. */
export const GEMMA4_E2B: Gemma4Arch = Object.freeze({
  layerCount: 35,
  hiddenSize: 1536,
  queryHeads: 8,
  kvHeads: 1,
  slidingHeadDim: 256,
  globalHeadDim: 512,
  globalLayers: Object.freeze([4, 9, 14, 19, 24, 29, 34]),
  slidingWindow: 512,
  kvProducerLayers: 15,
  producerIntermediate: 6144,
  consumerIntermediate: 12288,
  vocabSize: 262144,
  pleDim: 256,
  rmsEps: 1e-6,
  finalSoftcap: 30.0,
  ropeThetaSliding: 1e4,
  ropeThetaGlobal: 1e6,
  partialRotaryFactor: 0.25,
});

export function isGlobalLayer(arch: Gemma4Arch, layer: number): boolean {
  return arch.globalLayers.includes(layer);
}

export function isKvProducer(arch: Gemma4Arch, layer: number): boolean {
  return layer >= 0 && layer < arch.kvProducerLayers;
}

export function headDimForLayer(arch: Gemma4Arch, layer: number): number {
  return isGlobalLayer(arch, layer) ? arch.globalHeadDim : arch.slidingHeadDim;
}

export function mlpIntermediateForLayer(arch: Gemma4Arch, layer: number): number {
  return isKvProducer(arch, layer) ? arch.producerIntermediate : arch.consumerIntermediate;
}

/**
 * Bit width of a quantized matmul site, per the tensor family table of ENGINE-PLAN section 3 plus
 * the round 2 corrections: attention projections and producer MLP are the 4-bit family, consumer
 * MLP and the logit head are the 2-bit family, and the PLE path's two little linears are the
 * 8-bit family.
 *
 * ROUND 2 CORRECTION, and it was a live defect rather than a tidy-up. This function answered 4 for
 * the `ple` site, which is what round 1 believed and what the plan's own comment then defended as
 * "the quantization rule, not the storage family". That distinction does not survive contact with
 * a dispatch: the only thing this answer is used for is choosing which kernel unpacks the bytes,
 * and quant.ts's TEXT_STORAGE_FAMILIES records `per_layer_input_gate` [256, 1536] and
 * `per_layer_projection` [1536, 256] as `ple-gate-8bit`, I8 storage, one signed byte per code with
 * no packing and no zero point. A 4-bit unpacker pointed at those bytes reads two codes where
 * there is one and subtracts a zero point that is not there, and it does not fail: it returns
 * numbers. Answering 8 here routes both to `qmatmul-8bit`, which is the kernel that reads them.
 */
export function gemvBitsForSite(arch: Gemma4Arch, layer: number, site: 'attn' | 'mlp' | 'ple'): 2 | 4 | 8 {
  if (site === 'ple') return 8;
  if (site === 'mlp' && !isKvProducer(arch, layer)) return 2;
  return 4;
}

export const LM_HEAD_BITS = 2;

// -------------------------------------------------------------------------- prefill chunking

/**
 * Default prefill chunk in tokens. A multiple of both GEMM M tiles (8 at 4-bit, 4 at 2-bit, from
 * kernels/qgemm.ts) so no chunk but the last carries a partial tile.
 *
 * Why 64 and not the whole prompt. ENGINE-PLAN round 3 ruling 6 asked for the chunk to move to
 * the whole prompt up to 512 positions on the argument that a 299 token prompt in five chunks
 * streamed the weights five times. Round 4 measured it on the same tree and the same morning:
 * interview-300 time to first token 7069 ms at 512 against 7157 ms at 64, inside the run to run
 * noise (docs/ENGINE-PERF.md section 10). PREFILL-CAMPAIGN.md had already found M1 prefill compute
 * bound rather than traffic bound, and this is that finding again: the GEMM unpacks each weight
 * once per tile of token columns, so what the long prompts pay for is the unpack per tile, not the
 * bytes per chunk. The width changes no value (the full model page runs interview-300 at 64 and at
 * 299 and both produce the same sixteen tokens, .engine-ref/parity-r3-integrator.json).
 *
 * What 512 would cost, measured before the constant moved back: the activation staging of every
 * slot of execute.ts slotElements plus the residual pair, sized once at this width by
 * runtime.ts allocateSlots, is 33.0 MB at 64, 146.3 MB at 299 and 249.0 MB at 512, so 216 MB more
 * resident beside a 2,006,857,726 byte model for nothing measurable. The KV cache is 75.5 MB
 * regardless. The TTFT gap on long prompts is a GEMM tiling question, taken up in the plan.
 */
export const DEFAULT_PREFILL_CHUNK = 64;

export interface PrefillChunk {
  /** Absolute position of the chunk's first token. */
  readonly start: number;
  /** One past the last position. */
  readonly end: number;
  readonly length: number;
}

/**
 * Split positions [from, promptLength) into chunks of at most chunkSize. Contiguous, in order,
 * covering exactly the suffix the KV reuse plan said to prefill. An empty suffix is zero chunks.
 */
export function prefillChunks(from: number, promptLength: number, chunkSize: number = DEFAULT_PREFILL_CHUNK): PrefillChunk[] {
  if (!Number.isInteger(from) || from < 0) throw new Error(`prefillChunks: bad from ${from}`);
  if (!Number.isInteger(promptLength) || promptLength < from) {
    throw new Error(`prefillChunks: promptLength ${promptLength} below from ${from}`);
  }
  if (!Number.isInteger(chunkSize) || chunkSize <= 0) {
    throw new Error(`prefillChunks: chunkSize must be a positive integer, got ${chunkSize}`);
  }
  const chunks: PrefillChunk[] = [];
  // THE LAST CHUNK IS ONE POSITION, and every chunk before it therefore stops at the KV producers
  // (planPrefillChunk). Only the last chunk runs all 35 layers, and only the last position's
  // logits are ever read, so making that chunk exactly one token is what turns "20 of 35 layers
  // are wasted on all but the final position" into "20 of 35 layers run for the final position
  // only". Splitting a 483 token prompt as 482 producer positions plus one full position, rather
  // than 448 plus a full chunk of 35, is worth another 1.1x cold.
  //
  // It matters far more on the warm path, which is the one an interview actually walks. A second
  // turn reuses the KV prefix (kv.ts acceptPrompt) and prefills only the new suffix, so before
  // this the common case was a suffix under `chunkSize`: one chunk, which is the last chunk, which
  // ran all 35 layers and saved nothing at all. Now a 50 token suffix is 49 producer positions and
  // one full position, and the warm turn gets the same 59 percent the cold one does.
  //
  // A one token gemm chunk is not a new shape: a prompt whose length is one above a multiple of
  // the chunk size already produced one (app.clarifyFallback at 385 is 6 chunks of 64 and a chunk
  // of 1), and it is the cheapest chunk the head can run on.
  const last = promptLength - 1;
  for (let start = from; start < last; start += chunkSize) {
    const end = Math.min(start + chunkSize, last);
    chunks.push({ start, end, length: end - start });
  }
  if (promptLength > from) chunks.push({ start: last, end: promptLength, length: 1 });
  return chunks;
}

// ------------------------------------------------------------------------ progress accounting

/**
 * The Gemma4Progress shape of src/engine/llm/gemma4-engine.d.ts, restated structurally so this
 * file needs no import. engine.ts type checks its events against the real declaration too.
 */
export interface ProgressEvent {
  status: 'init' | 'tokenizer' | 'weights' | 'ready';
  kind?: 'bytes' | 'tensors';
  loaded?: number;
  total?: number | null;
  fraction?: number;
  fromCache?: boolean;
  message?: string;
}

function fractionOf(loaded: number, total: number | null): number | undefined {
  if (total === null || !(total > 0)) return undefined;
  return Math.min(1, Math.max(0, loaded / total));
}

/**
 * A weights phase event. Two rules from ENGINE-PLAN section 3, both load bearing for
 * src/engine/llm/engineHost.ts:
 *
 *  - The host treats `loaded > 0` with `fromCache !== true` as proof the load is a download and
 *    latches that for the rest of the load. So the `fromCache` key is only present when true;
 *    a `fromCache: false` on cached bytes would be harmless, but omitting it entirely keeps the
 *    event byte compatible with the incumbent's.
 *  - The opening event of the phase must carry `loaded === 0` and no `fromCache`, which
 *    `openingWeightsEvent` builds and the orchestrator check asserts.
 */
export function weightsEvent(
  kind: 'bytes' | 'tensors',
  loaded: number,
  total: number | null,
  fromCache?: boolean,
): ProgressEvent {
  const event: ProgressEvent = { status: 'weights', kind, loaded, total };
  const fraction = fractionOf(loaded, total);
  if (fraction !== undefined) event.fraction = fraction;
  if (fromCache === true) event.fromCache = true;
  return event;
}

/** The first weights event of a load. loaded 0, no fromCache, per the host's download latch. */
export function openingWeightsEvent(totalBytes: number | null): ProgressEvent {
  return weightsEvent('bytes', 0, totalBytes);
}

// ----------------------------------------------------------------------------- forward plan

/**
 * One planned dispatch. `kernel` names a registry entry when the family exists this round and a
 * PENDING_KERNELS member when it lands next round. `role` says what the dispatch is for, so a
 * recorded sequence reads as a story rather than as a histogram.
 */
export interface DispatchStep {
  readonly kernel: string;
  readonly phase: 'embed' | 'attn' | 'ple' | 'mlp' | 'head';
  /** Decoder layer index, or -1 for the embed and head phases. */
  readonly layer: number;
  readonly role: string;
  /**
   * Head dimension the step operates at, on the attention chain steps that have one: 256 on
   * sliding layers, 512 on the seven global layers. Round 1's dry walk could not distinguish a
   * global layer from its sliding neighbours because no geometry travelled with a step; with
   * these two fields planLayer produces four distinct sequences (producer and consumer, sliding
   * and global) instead of two, and the orchestrator check asserts it.
   */
  readonly headDim?: number;
  /** Attention window in positions, on the same steps: 512 inclusive on sliding layers, 0 meaning unwindowed on global layers. */
  readonly window?: number;
  /** MLP intermediate width, on the MLP matmul and activation steps: 6144 producer, 12288 consumer. */
  readonly intermediate?: number;
}

/**
 * Kernel families the forward plan references that are not in the registry yet. The orchestrator
 * check asserts that every kernel a plan references is either registered or on this list, and the
 * live executor refuses a step whose kernel is on it rather than substituting anything. The two
 * entries and why they are here are argued directly below the list.
 */
export const PENDING_KERNELS: readonly string[] = Object.freeze([]);

/**
 * THE LIST IS EMPTY AND THE MECHANISM STAYS. Round 2 put two families here, both findings rather
 * than oversights, and both have now landed in the registry:
 *
 *  - `dense-bf16-matmul` is `model.language_model.per_layer_model_projection`, a plain BF16 linear
 *    of [8960, 1536] that quant.ts records in modules_to_not_convert. It is the context aware half
 *    of the per layer input, and no kernel in this engine multiplied an unquantized matrix until
 *    kernels/denseMatmul.ts. The loader widens BF16 to f32 on the way in, because WGSL has no BF16
 *    storage type, so the kernel binds f32 and the widening is one shift in quant.ts.
 *  - `qmatmul-8bit` is `per_layer_input_gate` [256, 1536] and `per_layer_projection` [1536, 256],
 *    which quant.ts's TEXT_STORAGE_FAMILIES records as the I8 family: signed bytes, no packing and
 *    no zero point, so the byte is the code. kernels/pleMatmul.ts reads them with the same two
 *    sided SRQ and the same exact integer accumulation the other two matmul families use.
 *
 * The list itself is kept rather than deleted, and so is the executor's refusal that consults it,
 * because the property worth keeping is not "these two are missing" but "a planned step whose
 * kernel does not exist refuses by name instead of being substituted with a near neighbour". A
 * 4-bit unpacker pointed at I8 bytes returns numbers rather than an error, which is the failure
 * this mechanism exists to make loud, and the next family somebody plans before they build should
 * meet it already in place.
 */
const QMATMUL8 = 'qmatmul-8bit';
const DENSE_BF16 = 'dense-bf16-matmul';

/**
 * THE SRQ RULE, recorded as data so it is greppable rather than as a comment somebody scrolls
 * past. Integrator's note, rewritten at the round 2 merge.
 *
 * Every `QuantizedLinear` in this checkpoint rounds its input onto an int8 grid with that
 * module's own `input_activation_scale`, and its output with `output_activation_scale`, always at
 * 8 bits whatever the weight width is, skipped only when the stored scale is exactly 0.0
 * (quant.ts `applySrq`, which is the one implementation). This is arithmetic the model was
 * calibrated with. It is not a dispatch count question and it is not a rounding question:
 * snapping layer 0's MLP input onto `gate_proj`'s grid at scale 0.9407 moves that vector by 3.2
 * percent in relative L2, and layer 15's by 1.5 percent, measured against the reference capture
 * in .engine-ref/data/ref.mlp.meta.json. Thirty five layers compound it.
 *
 * Round 1 closed one site and listed eleven open ones here. Round 2 replaced the list with the
 * rule, and the rule is enforced in the kernels rather than at the call sites: every quantized
 * matmul kernel (qgemv, qgemm and qmatmul-8bit) carries the module's two scales as uniform words
 * and runs the input snap as a fused prologue and the output snap as a fused epilogue, both
 * unconditional and scale driven. `execute.ts` `linearParams` stages the two scales from the
 * checkpoint for every quantized linear step, so a site cannot be left open by omission; only a
 * stored scale of exactly 0.0, which is the checkpoint's own word for "uncalibrated", turns a
 * side off. The list of eleven was also short: gate_proj and up_proj carry a non zero
 * output_activation_scale (0.618 at layer 0, 0.0305 at layer 15) and were not on it, and the
 * k-matmul section now asserts against the reference capture that every element of the
 * reference's own gate and up outputs sits on that grid. An enumeration of sites is a thing that
 * can be short; the rule cannot be.
 *
 * This export is kept, empty, so the orchestrator check can assert that nothing is open and so
 * a future site somebody discovers has a place to be named while it is being closed.
 *
 * ENGINE-PLAN K3 names the fix as "RMS norm plus activation quantize plus zero point sum" and
 * describes fusing it into the norm. Read that as two claims: fusing saves a dispatch, applying
 * it at all changes the numbers. Only the second one gates correctness, and the k-matmul section
 * measures the 3.2 percent effect appearing and then vanishing under the fused kernels.
 */
export const SRQ_OPEN_SITES: readonly string[] = Object.freeze([]);

// The quantized matmul kernel for a bit width, in each step mode. The 8-bit family has one entry
// serving both modes, because kernels/pleMatmul.ts is one kernel whose decode shape is mCols 1;
// the other two widths are genuinely different kernels with different tile geometry.
const GEMV: Record<2 | 4 | 8, string> = { 2: 'qgemv-2bit', 4: 'qgemv-4bit', 8: QMATMUL8 };
const GEMM: Record<2 | 4 | 8, string> = { 2: 'qgemm-2bit', 4: 'qgemm-4bit', 8: QMATMUL8 };

/**
 * The dispatches of one decoder layer, in execution order, for a decode step (mode 'gemv') or a
 * prefill chunk (mode 'gemm').
 *
 * The order inside the layer follows the reference implementation's block structure: attention
 * with its pre norm and its post norm before the residual add, then the per layer embedding path,
 * then the MLP with its pre and post norms. The PLE path's exact position is provisional until
 * the M1 CPU reference forward pins it against per layer activations; it is a placement question,
 * not a count question, so the census is stable either way.
 *
 * Deliberately less fused than the incumbent where fusion is not yet earned (ENGINE-PLAN K3,
 * K10, K12): the pre norms run on their own, gate and up are separate GEMVs, and the PLE glue is
 * spelled out, because the measured marginal cost of a within pass dispatch is single digit
 * microseconds on the M1 and about 4 on the 5070 (DECODE-FUSION-FINDING.md, DECODE-CAMPAIGN.md
 * 5.2), and separate dispatches are provable one kernel at a time against reference activations.
 * The K11 shape ships fused already because the MLP lane built it that way
 * (kernels/mlpEpilogue.ts); the rest of the plan's fusions come back in the performance round
 * with the identity gates watching.
 *
 * Nothing is missing here numerically any more. See `SRQ_OPEN_SITES` above: the activation
 * quantize on both sides of every quantized linear is a property of the matmul kernels, not of
 * this step list, which is why closing every site changed no row of the census.
 */
/**
 * The fused joins of kernels/blockJoin.ts, by role. The first replaces the post attention epilogue
 * and the MLP pre norm; the second replaces the post PLE epilogue, the layer scalar and the NEXT
 * layer's input norm; the third is the second at the last layer of a decode step, where the norm
 * that follows is the head's final norm. Round 4 dispatch fusion, docs/ENGINE-PERF.md section 15.
 */
export const ROLE_JOIN_MLP = 'post_attention_layernorm plus residual, pre_feedforward_layernorm';
export const ROLE_JOIN_TAIL = 'post_per_layer_input_norm plus residual, layer_scalar, next input_layernorm';
export const ROLE_JOIN_FINAL = 'post_per_layer_input_norm plus residual, layer_scalar, final norm';

/**
 * The fused K10 prologues of the decode token (kernels/qgemv.ts GemvPrologue): gelu(gate) * up
 * read inside the consuming GEMV instead of stored to a slot and read back. Decode only; a
 * prefill chunk keeps gelu-mul and the GEMM.
 */
export const ROLE_GELU_DOWN = 'gelu times up, down_proj';
export const ROLE_GELU_PLE = 'gate times per layer row, per_layer_projection';
const GEMV_GELU: Record<2 | 4 | 8, string> = { 2: 'qgemv-2bit-gelu', 4: 'qgemv-4bit-gelu', 8: 'qgemv-8bit-gelu' };

/**
 * A layer's input norm as a step of its own. Layer 0 runs it (nothing precedes layer 0 but the
 * embed phase); every later layer receives its input norm from the previous layer's tail join, so
 * a layer plan run in ISOLATION with a seeded residual must be preceded by this step or its
 * q_proj reads a stale `normed` slot. The dev pages that run one layer alone do exactly that.
 */
export function inputNormStep(layer: number): DispatchStep {
  return { kernel: 'rms-norm', phase: 'attn', layer, role: 'input_layernorm' };
}

/** The head's final norm as a step, for the prefill head and for a dev page that seeds the residual. */
export const FINAL_NORM_STEP: DispatchStep = Object.freeze({ kernel: 'rms-norm', phase: 'head', layer: -1, role: 'final norm' });

export function planLayer(arch: Gemma4Arch, layer: number, mode: 'gemv' | 'gemm'): DispatchStep[] {
  const producer = isKvProducer(arch, layer);
  const attnBits = gemvBitsForSite(arch, layer, 'attn');
  const mlpBits = gemvBitsForSite(arch, layer, 'mlp');
  const mm = mode === 'gemv' ? GEMV : GEMM;
  const steps: DispatchStep[] = [];
  const push = (
    kernel: string,
    phase: DispatchStep['phase'],
    role: string,
    extra?: { headDim?: number; window?: number; intermediate?: number },
  ): void => {
    steps.push({ kernel, phase, layer, role, ...extra });
  };

  // The layer's attention geometry, attached to every step of the chain that depends on it, so a
  // recorded sequence carries what the dispatch actually needs: head dimension 256 or 512, and a
  // window of 512 inclusive positions or 0 meaning unwindowed, per the ENGINE-PLAN section 5
  // table. Round 1 shipped steps with no geometry, which made the seven global layers
  // indistinguishable from their sliding neighbours in the dry walk.
  const geo = {
    headDim: headDimForLayer(arch, layer),
    window: isGlobalLayer(arch, layer) ? 0 : arch.slidingWindow,
  };
  const width = { intermediate: mlpIntermediateForLayer(arch, layer) };

  // Attention. Scale is 1.0, not 1/sqrt(head_dim); that constant belongs to the attention
  // kernels and is restated here only so nobody reads this plan and adds the standard one.
  // Layer 0's input norm; every later layer's arrived with the previous layer's tail join.
  if (layer === 0) steps.push(inputNormStep(layer));
  push(mm[attnBits], 'attn', 'q_proj', geo);
  if (producer) {
    push(mm[attnBits], 'attn', 'k_proj', geo);
    push(mm[attnBits], 'attn', 'v_proj', geo);
  }
  push('rms-norm', 'attn', 'q_norm per head', geo);
  if (producer) {
    push('rms-norm', 'attn', 'k_norm per head', geo);
    // The weightless per KV head V norm before the cache store, ENGINE-PLAN risk 1 quirk 1,
    // independently observed by the lab's own census (DECODE-CAMPAIGN.md 4.7).
    push('rms-norm-weightless', 'attn', 'v_norm weightless', geo);
  }
  push('rope', 'attn', 'rope q', geo);
  if (producer) {
    push('rope', 'attn', 'rope k', geo);
    // K9: written into the slot directly, one dispatch for both regions, our own layout
    // (kv.ts), instead of the incumbent's stage and copy pattern.
    push('kv-cache-store', 'attn', 'kv store', geo);
  }
  // K8 is one dispatch per layer with the partials and merge inside it, per the attention
  // lane's kernel, and the prefill shape is its own entry (ENGINE-PLAN 5.4).
  push(mode === 'gemv' ? 'attention-decode' : 'attention-prefill', 'attn', 'attention', geo);
  push(mm[attnBits], 'attn', 'o_proj', geo);
  // K11 shape: the epilogue norms the block output and then adds the residual, fused because
  // re-reading the activation vector is bandwidth (ENGINE-PLAN section 5).
  // Fused with the MLP pre norm since the round 4 dispatch fusion (kernels/blockJoin.ts): the
  // epilogue's row is stored as the residual and normed again for the MLP in the same pass.
  push('norm-residual-norm', 'mlp', ROLE_JOIN_MLP);

  // MLP, gelu_pytorch_tanh, intermediate 6144 or 12288 by layer kind. Gate and up are separate
  // GEMVs this round; the K10 single pass fusion is a performance round move behind the
  // identity gates, not a correctness round one.
  push(mm[mlpBits], 'mlp', 'gate_proj', width);
  push(mm[mlpBits], 'mlp', 'up_proj', width);
  if (mode === 'gemv') {
    push(GEMV_GELU[mlpBits], 'mlp', ROLE_GELU_DOWN, width);
  } else {
    push('gelu-mul', 'mlp', 'gelu times up', width);
    push(mm[mlpBits], 'mlp', 'down_proj', width);
  }
  push('norm-residual', 'mlp', 'post_feedforward_layernorm plus residual');

  // The per layer embedding path, K12: gate from the hidden state, gelu it, multiply against the
  // gathered per layer row, project back up, norm, and add back into the residual.
  //
  // PLACEMENT IS NO LONGER PROVISIONAL. It was a guess when this lane wrote it, sitting between
  // attention and the MLP, and the guess was wrong. Settled at integration against the
  // transformers reference decoder layer (Apache-2.0, an allowed source), whose order is
  // attention block, then MLP block, then this block, then the scalar below. The block itself is
  // per_layer_input_gate, act_fn, multiply by the per layer row, per_layer_projection,
  // post_per_layer_input_norm, residual add, which is why the norm and the add are one
  // `norm-residual` here rather than a norm and a `scale-add`: it is the same shape this layer
  // already runs twice, so it uses the same kernel.
  // The two linears here are NOT the 4-bit family, which is what round 1 planned them as.
  // quant.ts's TEXT_STORAGE_FAMILIES has them as `ple-gate-8bit`: I8 storage, one byte per code,
  // no packing and no zero point. `gemvBitsForSite(arch, layer, 'ple')` now answers 8 and the
  // width goes through the same `mm` table as the other two sites, so the routing is one rule
  // rather than a constant sitting next to a function that disagrees with it.
  const pleBits = gemvBitsForSite(arch, layer, 'ple');
  push(mm[pleBits], 'ple', 'per_layer_input_gate');
  if (mode === 'gemv') {
    push(GEMV_GELU[pleBits], 'ple', ROLE_GELU_PLE);
  } else {
    push('gelu-mul', 'ple', 'gate times per layer row');
    push(mm[pleBits], 'ple', 'per_layer_projection');
  }
  // The tail: the PLE epilogue, the layer scalar below and the next layer's input norm are one
  // dispatch (kernels/blockJoin.ts). On a decode step the last layer's tail carries the head's
  // final norm instead; on a prefill chunk the last layer keeps the unfused pair, because the head
  // norms one copied row (the chunk's last) and the join would norm every row into the slot the
  // head reads from row 0.
  const lastLayer = layer + 1 === arch.layerCount;
  if (!lastLayer) {
    push('norm-residual-norm', 'ple', ROLE_JOIN_TAIL);
    return steps;
  }
  if (mode === 'gemv') {
    push('norm-residual-norm', 'ple', ROLE_JOIN_FINAL);
    return steps;
  }
  push('norm-residual', 'ple', 'post_per_layer_input_norm plus residual');

  // The fifth silent corruption trap, and it is not in ENGINE-PLAN's contract table. Every
  // decoder layer ends with `hidden_states *= layer_scalar`, a per layer scalar stored in the
  // checkpoint as one BF16 value (35 `layers.N.layer_scalar` tensors, confirmed in the header).
  // The values are not ones: layer 0 is 0.0272 and the range is 0.0272 to 0.8945, so dropping it
  // inflates layer 0's output by about 37x. This lane originally spent the name on the PLE
  // residual add, which is a different operation; the reference does the add first and then this.
  // `scale-add` covers it at alpha = layer_scalar with a zero second operand.
  push('scale-add', 'mlp', 'layer_scalar');

  return steps;
}

/**
 * The embed phase. Round 1 planned two gathers here and that was wrong by four dispatches, which
 * is a correctness hole rather than a census one: the per layer input a decoder layer multiplies
 * against is not the PLE table row. It is the token identity half and the context aware half
 * combined, and the combination is where the second half comes from.
 *
 * Settled against the transformers reference (Apache-2.0, an allowed source), whose text model
 * builds it in `get_per_layer_inputs` and `project_per_layer_inputs`:
 *
 *   identity   = embed_tokens_per_layer(ids)                    scaled by sqrt(256), the gather
 *   projection = per_layer_model_projection(inputs_embeds) * hidden_size ** -0.5
 *   normed     = per_layer_projection_norm(projection)          RMS norm over 256, per layer group
 *   per_layer  = (normed + identity) * 2 ** -0.5
 *
 * The gather's own scale of 16 is sqrt(256), so the identity half is complete as the gather leaves
 * it. The projection's scale rides with the pending dense step, because a kernel that does not
 * exist yet can carry its own scale when it arrives and a separate scale-add for it would be a
 * dispatch invented to work around an absence.
 *
 * The norm runs over the [tokens, 35, 256] tensor as 35 rows of 256 per token, not as one row of
 * 8960, which is what `per_layer_projection_norm` being sized `hidden_size_per_layer_input` means.
 * The combine is two scale-adds rather than one, because scale-add computes alpha * x + y and
 * (a + b) * s is not that shape; the first adds, the second scales by 2 ** -0.5 against a zero
 * operand. Fusing the pair is a performance round move, and it is the same trade planLayer's
 * comment argues for the norms.
 */
export function planEmbed(arch: Gemma4Arch): DispatchStep[] {
  return [
    { kernel: 'embed-gather-2bit', phase: 'embed', layer: -1, role: 'embed_tokens gather' },
    { kernel: 'ple-gather-4bit', phase: 'embed', layer: -1, role: 'per layer embedding gather' },
    { kernel: DENSE_BF16, phase: 'embed', layer: -1, role: 'per_layer_model_projection' },
    { kernel: 'rms-norm', phase: 'embed', layer: -1, role: 'per_layer_projection_norm', intermediate: arch.pleDim },
    { kernel: 'scale-add', phase: 'embed', layer: -1, role: 'per layer inputs add' },
    { kernel: 'scale-add', phase: 'embed', layer: -1, role: 'per layer inputs scale' },
  ];
}

/**
 * `2 ** -0.5`, the reference's `per_layer_input_scale`, applied once to the combined per layer
 * input.
 *
 * `Math.SQRT1_2` rather than `Math.pow(2, -0.5)`, and the difference is measured rather than
 * stylistic: `Math.pow` returns 0.7071067811865475 here, which is one unit in the last place below
 * the correctly rounded double, while Python's `2.0 ** -0.5` and `Math.SQRT1_2` both give
 * 0.7071067811865476. Both round to the same f32, so this cannot move a token; it is fixed because
 * the k-ple section compares this constant against the reference's own value and a constant that
 * disagrees with the reference by a ULP is a constant nobody can use as a gate.
 */
export const PER_LAYER_INPUT_SCALE = Math.SQRT1_2;

/** `hidden_size ** -0.5`, the reference's `per_layer_model_projection_scale`. */
export function perLayerProjectionScale(arch: Gemma4Arch): number {
  return Math.pow(arch.hiddenSize, -0.5);
}

/**
 * The head phase of a decode step: final norm, the 2-bit logit head over the whole vocabulary,
 * the monotonic softcap, and GPU side argmax so step N+1 encodes without a readback
 * (ENGINE-PLAN K13, K14, K15).
 */
export function planHead(arch: Gemma4Arch, mode: 'gemv' | 'gemm' = 'gemm'): DispatchStep[] {
  // On a decode step the final norm rides in the last layer's tail join (planLayer).
  return [
    ...(mode === 'gemv' ? [] : [FINAL_NORM_STEP]),
    { kernel: GEMV[LM_HEAD_BITS], phase: 'head', layer: -1, role: 'lm_head' },
    { kernel: 'logit-softcap', phase: 'head', layer: -1, role: 'tanh(logits / 30) * 30' },
    { kernel: 'argmax-partial', phase: 'head', layer: -1, role: 'argmax partials' },
    { kernel: 'argmax-final', phase: 'head', layer: -1, role: 'argmax final' },
  ];
}

/** Every dispatch of one steady state decode token, in order. */
export function planDecodeStep(arch: Gemma4Arch): DispatchStep[] {
  const steps: DispatchStep[] = [...planEmbed(arch)];
  for (let layer = 0; layer < arch.layerCount; layer += 1) {
    steps.push(...planLayer(arch, layer, 'gemv'));
  }
  steps.push(...planHead(arch, 'gemv'));
  return steps;
}

/**
 * Every dispatch of one prefill chunk. The head runs only on the chunk that carries the prompt's
 * last position, because its logits are the only ones a greedy decode reads.
 *
 * A CHUNK THAT IS NOT THE LAST ONE STOPS AFTER THE KV PRODUCERS, layers 0 to
 * `arch.kvProducerLayers - 1`. This is exact, not an approximation, and the reason is the KV
 * sharing this checkpoint is built on. Only producer layers write anything that outlives the
 * chunk: `planLayer` pushes `k_proj`, `v_proj`, `k_norm`, `v_norm`, `rope k` and
 * `kv-cache-store` under `if (producer)`, and `kv.ts kvRoleForLayer` has every consumer layer
 * reading producer 13 or 14 rather than a cache of its own. So a consumer layer's output for a
 * token reaches exactly two places: the next layer, for that same token, and the head. The head
 * runs on the last chunk only, and tokens never talk to each other except through the KV cache.
 * Everything layers 15 to 34 compute for a token in an earlier chunk is therefore read by
 * nothing, and the residual stream is re-seeded from the embedding at the top of every chunk, so
 * no state crosses the boundary either.
 *
 * That is 20 of 35 layers, and by round 7's per kernel measurement about 59 percent of a chunk:
 * the whole of `qgemm-2bit` (the consumer MLP is the only 2-bit prefill site), the `q_proj` and
 * `o_proj` of 20 layers inside `qgemm-4bit`, and 20/35 of the PLE and attention families. On a
 * 483 token prompt 7 of the 8 chunks take this path. See ENGINE-PLAN "Round 8 lead item"; vLLM
 * ships the same structure for this model family as `--kv-sharing-fast-prefill`.
 *
 * The last chunk still runs all 35 layers over all of its positions. Trimming that one to the
 * single position the head reads is the remaining 8 percent and needs the residual seeded at row
 * 0, which is a second submission; it is deliberately not done here.
 */
export function planPrefillChunk(arch: Gemma4Arch, isLastChunk: boolean): DispatchStep[] {
  const steps: DispatchStep[] = [...planEmbed(arch)];
  const layers = isLastChunk ? arch.layerCount : arch.kvProducerLayers;
  for (let layer = 0; layer < layers; layer += 1) {
    steps.push(...planLayer(arch, layer, 'gemm'));
  }
  if (isLastChunk) steps.push(...planHead(arch, 'gemm'));
  return steps;
}

export interface ForwardPlan {
  readonly chunks: readonly PrefillChunk[];
  /** Dispatches per prefill chunk, aligned with `chunks`. */
  readonly prefillSteps: readonly (readonly DispatchStep[])[];
  /** Dispatches of one decode token. Decode repeats this sequence per token. */
  readonly decodeSteps: readonly DispatchStep[];
  /** Dispatch count of one steady state decode token. The census number of this engine. */
  readonly dispatchesPerDecodeToken: number;
}

/**
 * The whole forward story for one prompt: the prefill chunk schedule from the KV reuse plan's
 * `prefillFrom`, and the decode step sequence each generated token repeats.
 *
 * For the record, against the incumbent's measured 315 dispatches per steady decode token
 * (DECODE-CAMPAIGN.md 4.7): this plan produces more, deliberately, because norms, residuals and
 * elementwise glue are unfused this round. See planLayer's comment for the measured argument.
 */
export function buildForwardPlan(
  arch: Gemma4Arch,
  prefillFrom: number,
  promptLength: number,
  chunkSize: number = DEFAULT_PREFILL_CHUNK,
): ForwardPlan {
  const chunks = prefillChunks(prefillFrom, promptLength, chunkSize);
  const prefillSteps = chunks.map((_, i) => planPrefillChunk(arch, i === chunks.length - 1));
  const decodeSteps = planDecodeStep(arch);
  return {
    chunks,
    prefillSteps,
    decodeSteps,
    dispatchesPerDecodeToken: decodeSteps.length,
  };
}

/** Dispatch counts by kernel name, for the census assertions and the dry run report. */
export function censusByKernel(steps: readonly DispatchStep[]): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const step of steps) counts[step.kernel] = (counts[step.kernel] ?? 0) + 1;
  return counts;
}

// -------------------------------------------------------------------------- the greedy loop

/**
 * The greedy decode loop as a pure schedule. The first token comes from the prefill's final
 * logits, so `maxNewTokens: 1` yields one token and runs zero decode passes, which is the
 * documented contract of the API (gemma4-engine.d.ts) and the reason client.ts clamps to 2.
 *
 * The abort contract: `isAborted` is consulted exactly once per token, before the decode pass
 * that would produce it, so an abort lands within one token (ENGINE-PLAN section 2). EOS is a
 * set, not a scalar, because config.json says [1, 106] and the generation config says
 * [1, 106, 50] (ENGINE-PLAN section 4).
 *
 * A STOP TOKEN IS NOT YIELDED. The loop ends on it and the caller never sees it, so a chunk count
 * is a count of the tokens that carry the reply. Until 2026-09-02 it was yielded, with an empty
 * delta because the decode stream skips special tokens, which left this engine's count one higher
 * than the incumbent's for identical text (ENGINE-PLAN round 3 worklist item 7, settled by round
 * 4's lead ruling 4: the two engines answer the shared contract identically). Nothing about the
 * decoding moved, only what the generator hands back. The stop token is still what ends the loop,
 * it still ends it at the same step, it still consumes the budget it consumed, and a caller timing
 * a rate over the chunks now times the tokens that produced text.
 *
 * `decode(prevToken, position)` runs one forward step with prevToken at `position` and resolves
 * to the argmax of its logits.
 *
 * `decodeAhead(position)`, when the executor offers it, runs the step at `position` whose input
 * token the GPU already holds from the previous step (runtime.ts decodeAhead). With it the loop
 * keeps ONE step in flight: the step for position p + 1 is submitted before the token at p has
 * been read back, so the host's binding and encoding of a token overlaps the GPU's run of the
 * previous one, which is the round 3 host overhead lever. The tokens are the same tokens: the
 * argmax the GPU feeds forward is the argmax the host reads back a step later. What changes is
 * the accounting: `onAdvance(token, position)` fires as soon as a token is known to be the input
 * of a submitted step, which is where a caller appends it to its cached transcript, and an abort
 * or an EOS can leave one submitted step whose result is discarded. The abort still lands within
 * one yielded token, and the stop set is still honoured on the token that carries it; the
 * discarded step only ever wrote KV rows past the transcript's end, which the next prefill's
 * rewind treats as absent. The decode pass count can therefore be one above the serial loop's.
 */
export async function* runGreedyLoop(options: {
  firstToken: number;
  maxNewTokens: number;
  eosIds: ReadonlySet<number>;
  startPosition: number;
  isAborted: () => boolean;
  decode: (prevToken: number, position: number) => Promise<number>;
  decodeAhead?: (position: number) => Promise<number>;
  onAdvance?: (token: number, position: number) => void;
}): AsyncGenerator<{ token: number; index: number }, void, void> {
  const budget = Math.max(0, Math.floor(options.maxNewTokens));
  if (budget === 0 || options.isAborted()) return;
  // The stop check sits before the yield rather than after it, at all three sites below. A prompt
  // whose very first token is a stop token therefore yields nothing at all, which is the honest
  // answer: the model wrote no reply.
  if (options.eosIds.has(options.firstToken)) return;
  yield { token: options.firstToken, index: 0 };

  if (!options.decodeAhead) {
    let prev = options.firstToken;
    for (let index = 1; index < budget; index += 1) {
      if (options.isAborted()) return;
      const token = await options.decode(prev, options.startPosition + index - 1);
      if (options.eosIds.has(token)) return;
      yield { token, index };
      prev = token;
    }
    return;
  }

  // The pipelined loop. `pending` is the step whose token is due next; `position` is the position
  // of that step's input token. The step after it is submitted before `pending` is awaited.
  const ahead = options.decodeAhead;
  let position = options.startPosition;
  if (budget < 2 || options.isAborted()) return;
  options.onAdvance?.(options.firstToken, position);
  let pending = ahead(position);
  for (let index = 1; index < budget; index += 1) {
    const next = index + 1 < budget && !options.isAborted() ? ahead(position + 1) : null;
    const token = await pending;
    // A stop token's step may already be in flight. It is drained here and the token never enters
    // the transcript, exactly as the serial loop never decoded past it, and the stop token itself
    // is not yielded on either loop.
    if (options.eosIds.has(token)) {
      // Drain the lookahead so its readback settles before the caller moves on; its token is not
      // part of the reply.
      if (next) await next.then(() => undefined, () => undefined);
      return;
    }
    if (next) options.onAdvance?.(token, position + 1);
    yield { token, index };
    if (next === null) return;
    if (options.isAborted()) {
      await next.then(() => undefined, () => undefined);
      return;
    }
    pending = next;
    position += 1;
  }
}
