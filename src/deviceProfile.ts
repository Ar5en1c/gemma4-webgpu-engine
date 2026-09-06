// Written from docs/ENGINE-PLAN.md sections 3, 5.3, 5.5 and risk 6, the WebGPU and WGSL
// specifications, the measurement record in ../../../../gemma4-kernels-lab/DECODE-CAMPAIGN.md and
// PREFILL-CAMPAIGN.md, and a live adapter probe run on this project's M1 on 2026-09-01. No vendored
// bundle, no extracted kernel and no third party engine source was read.
//
// SPDX-License-Identifier: Apache-2.0
//
// The device profile: what this engine believes about a class of hardware, as data, with the
// reason for every belief attached to the belief.
//
// Why this file exists at all. Round 1 read the adapter's limits and stopped there, which meant
// every tuning decision in the tree was either a constant with a comment or a number remembered
// from a lab document. A limit and a tuning choice are different things and they fail differently.
// A limit is a fact the adapter states and the engine must obey. A tuning choice is a number
// somebody measured on one machine, and the only honest way to carry it is with the machine and
// the measurement attached, so the next person can see whether it was ever measured on theirs.
// Hence `tuningReason` on every field: a profile whose reasons read "nobody has measured this"
// is still useful, and it is useful precisely because it says so.
//
// The rule this file does not break. ENGINE-PLAN 5.5 rule 3 forbids gating a correctness path on
// the adapter's reported subgroup width, and nothing here does. `subgroupWidth` below is the width
// the 32 lane butterfly's aligned groups assume, recorded so a device that reports something else
// is visible in a report; the reduce policy is still chosen by the known answer self test in
// pipeline.ts and by nothing else. A profile never selects a kernel. It informs geometry, buffer
// sizing and what a report says about the machine.

import type { Gemma4AdapterInfo, Gemma4DeviceLimits } from './device';

// --------------------------------------------------------------------------- the shape

/** One tuned value and the reason it has that value. The reason is not optional on purpose. */
export interface Tuned<T> {
  readonly value: T;
  /**
   * Why this number. Name the measurement and the machine where there was one, and say plainly
   * that there was not where there was not. "It seemed fine" is not a reason and neither is a
   * bare citation with no number in it.
   */
  readonly tuningReason: string;
}

/** Everything a profile carries. Each field is a `Tuned`, so each field carries its own reason. */
export interface Gemma4ProfileTuning {
  /**
   * The lane count the 32 lane butterfly's aligned groups assume. Advisory, and deliberately not
   * a branch: `subgroupShuffleXor` with masks below 32 never crosses a 32 lane aligned boundary,
   * so the butterfly reduces within each aligned group of 32 whatever the real width is.
   */
  readonly subgroupWidth: Tuned<number>;
  /** Whether `@subgroup_size(n)` can be written at all on this stack. */
  readonly subgroupSizeControl: Tuned<boolean>;
  /** Whether `dot4I8Packed` and friends are available as an optional fast path. */
  readonly packedInt8Dot: Tuned<boolean>;
  /** Risk 6's number. Every kernel's binding budget is designed against it. */
  readonly maxStorageBuffersPerShaderStage: Tuned<number>;
  /** The largest single storage binding, which is what caps the PLE table as one buffer. */
  readonly maxStorageBufferBindingSize: Tuned<number>;
  /** The largest single buffer allocation. */
  readonly maxBufferSize: Tuned<number>;
  /** Workgroup memory per workgroup, which caps a GEMM's staged M tile. */
  readonly maxComputeWorkgroupStorageSize: Tuned<number>;
  /** Invocations per workgroup, which caps the decode GEMV's width times rows. */
  readonly maxComputeInvocationsPerWorkgroup: Tuned<number>;
  /** Decode GEMV workgroup width. A geometry knob, safe under ENGINE-PLAN risk 2 mitigation 2. */
  readonly decodeGemvWorkgroupWidth: Tuned<number>;
  /**
   * The decode attention shape, three fields, applied by `applyProfileGeometry` before any
   * pipeline is built.
   *
   * These are NOT like the two decodeGemv fields below them, which are recorded and deliberately
   * not applied: those name one workgroup width and one row count, and the engine has had a
   * separate geometry per quantization family since round 3, so a single pair cannot describe it
   * and applying it would push the 2-bit family off its measured 32 by 16 and the 8-bit off its
   * 64 by 2. These three each name exactly one knob of one kernel.
   *
   * `attentionSlices` is how many 64 lane workers share a workgroup, `attentionKvSplits` how many
   * workgroups share the KV span, and their product is how many workers deal it. The two are the
   * same axis from either end, which is why they live together: the 5070's winning shape holds
   * the worker count at the shipped 8 and moves it onto four times the workgroups.
   */
  readonly attentionSlices: Tuned<number>;
  readonly attentionKvSplits: Tuned<number>;
  /** Independent accumulators the weighted V loop carries. A latency knob, not a scheduling one. */
  readonly attentionVAccumulators: Tuned<number>;
  /** Independent accumulators the score loop carries. 4 is the shipped text to the byte. */
  readonly attentionScoreAccumulators: Tuned<number>;
  /**
   * How the score loop is laid across the lanes: 'rows', the shipped loop, each lane walking its
   * own K row; or 'dims', the lanes spanning the head dimension with one butterfly a position
   * (kernels/attention.ts scoreLayout). A string, the one non numeric geometry field.
   */
  readonly attentionScoreLayout: Tuned<string>;
  /** How many ways the 2-bit down_proj's K reduction is cut across workgroups. */
  readonly decodeGemv2KSplits: Tuned<number>;
  /**
   * Decode steps the greedy loop keeps in flight (plan.ts runGreedyLoop lookaheadDepth). 2 is the
   * loop this engine has always run: one step awaited, one submitted behind it. A deeper queue
   * covers a readback that takes longer than one step; every queued step reads its input token
   * from the GPU's own slot, so no token is guessed and the ids cannot move. Applied by
   * engine.ts's generate loop, not by applyProfileGeometry: it is a loop shape, not a kernel one.
   */
  readonly decodeLookahead: Tuned<number>;
  /** Independent Q/K/V and gate/up producers share dispatches without changing their dot loops. */
  readonly decodeBatchProjections: Tuned<boolean>;
  /** Workgroup width of batched 2-bit gate/up only; down and head retain their geometry. */
  readonly decodeBatch2WorkgroupWidth: Tuned<number>;
  /** Calibrated producer down dots split before scaling, then fold exactly. */
  readonly decodeProducerDownKSplits: Tuned<number>;
  /** PLE gate integer partials folded by its projection consumer. */
  readonly decodePleGateKSplits: Tuned<number>;
  /** Decode GEMV output rows per workgroup. Same class of knob. */
  readonly decodeGemvRowsPerWorkgroup: Tuned<number>;
  /** The largest single IndexedDB value the weight cache may write. */
  readonly idbMaxValueBytes: Tuned<number>;
}

export type Gemma4ProfileField = keyof Gemma4ProfileTuning;

/**
 * What a profile matches on. A superset of `Gemma4AdapterInfo`, because Chrome also puts the
 * subgroup size range on `adapter.info` and a profile is allowed to read it for reporting even
 * though no correctness path may branch on it.
 */
export interface Gemma4ProfileInput extends Partial<Gemma4AdapterInfo> {
  readonly subgroupMinSize?: number;
  readonly subgroupMaxSize?: number;
  /** Adapter feature names, lowercase, as `[...adapter.features]` gives them. */
  readonly features?: readonly string[];
  /** `[...navigator.gpu.wgslLanguageFeatures]`. */
  readonly wgslLanguageFeatures?: readonly string[];
}

export interface Gemma4DeviceProfile {
  /** Stable id. Reports quote it, so do not rename one casually. */
  readonly id: string;
  readonly note: string;
  /** ISO date of the live probe behind the numbers, or null when nobody has run one. */
  readonly measuredOn: string | null;
  /** True when this profile claims the adapter. Evaluated in registry order, first match wins. */
  matches(info: Gemma4ProfileInput): boolean;
  readonly tuning: Gemma4ProfileTuning;
}

/**
 * A partial override. Either a bare value, which records that nobody gave a reason, or a whole
 * `Tuned` when the caller has one. Unnamed fields keep the profile's value, which is the merge
 * this engine wants: an override says what is different, never what is the same.
 */
export type Gemma4ProfileOverride = {
  readonly [K in Gemma4ProfileField]?: Gemma4ProfileTuning[K]['value'] | Gemma4ProfileTuning[K];
};

export interface ResolvedDeviceProfile extends Gemma4ProfileTuning {
  readonly id: string;
  readonly note: string;
  readonly measuredOn: string | null;
  /** How this profile was reached: the matched profile's id, or 'fallback', or 'explicit'. */
  readonly matchedBy: string;
  /** Fields the caller overrode, in the order the merge saw them. */
  readonly overridden: readonly Gemma4ProfileField[];
}

// --------------------------------------------------------------------------- the profiles

/**
 * The raw live probe this file's first profile is built from, kept verbatim so the numbers are
 * auditable rather than remembered. Run in the Browser pane against the dev server on 2026-09-01,
 * on the project's development machine, an Apple M1 with 16 GB, in the Claude browser build of
 * Chrome 148. Every field below was read back off the adapter in that session, and the ones that
 * are behaviour rather than metadata were re-run a second time in the same session before being
 * written down, because a limit somebody remembers is not a limit somebody measured.
 *
 * Four results in here are the interesting ones and all four are recorded because they close
 * questions rather than because they are pretty:
 *
 *  - `subgroupMinSize` and `subgroupMaxSize` are both 32, and a dispatch that read the
 *    `subgroup_size` builtin came back 32 in every lane, with `subgroup_invocation_id` running
 *    0 to 31 twice across a 64 wide workgroup. So the butterfly's aligned group of 32 is the
 *    whole subgroup here, exactly, and not an approximation of it, and a 64 wide workgroup is
 *    exactly the two virtual subgroups the measured decode geometry assumes.
 *  - `subgroups` is granted and there is no subgroup size control feature of any name, so
 *    `@subgroup_size(32)` cannot be written on this stack at all. `subgroupSizeControlAttempts`
 *    below carries all three spellings and the compiler's exact words for each.
 *  - Every compute limit requested at the adapter maximum was granted unchanged, and buffer
 *    allocations up to 2 GiB landed with no validation and no out of memory error.
 *  - The IndexedDB ceiling is not the flat number this project had been carrying. See
 *    `indexedDb` and the `idbMaxValueBytes` reason.
 */
export const M1_ADAPTER_PROBE = Object.freeze({
  when: '2026-09-01',
  // The machine is an Apple M1 with 16 GB of unified memory, this project's only development and
  // measurement machine. The memory is a fact about the machine rather than about the adapter, so
  // it is recorded here and not inside the profile's own fields.
  machine: 'Apple M1, Chrome 148.0.7778.280 in the Claude browser build',
  info: Object.freeze({
    vendor: 'apple',
    architecture: 'metal-3',
    // Chrome hands an Apple adapter no device and no description string at all. Recorded as the
    // empty strings they actually are, so nobody writes a profile matcher against them.
    device: '',
    description: '',
    subgroupMinSize: 32,
    subgroupMaxSize: 32,
  }),
  /** Every adapter feature, unfiltered, exactly as `[...adapter.features].sort()` gave them. */
  features: Object.freeze([
    'bgra8unorm-storage',
    'clip-distances',
    'core-features-and-limits',
    'depth-clip-control',
    'depth32float-stencil8',
    'dual-source-blending',
    'float32-blendable',
    'float32-filterable',
    'indirect-first-instance',
    'primitive-index',
    'rg11b10ufloat-renderable',
    'shader-f16',
    'subgroups',
    'texture-component-swizzle',
    'texture-compression-astc',
    'texture-compression-astc-sliced-3d',
    'texture-compression-bc',
    'texture-compression-bc-sliced-3d',
    'texture-compression-etc2',
    'texture-formats-tier1',
    'texture-formats-tier2',
    'timestamp-query',
  ] as const),
  /** `[...navigator.gpu.wgslLanguageFeatures].sort()`, unfiltered. */
  wgslLanguageFeatures: Object.freeze([
    'linear_indexing',
    'packed_4x8_integer_dot_product',
    'pointer_composite_access',
    'readonly_and_readwrite_storage_textures',
    'subgroup_id',
    'subgroup_uniformity',
    'texture_and_sampler_let',
    'uniform_buffer_standard_layout',
    'unrestricted_pointer_parameters',
  ] as const),
  limits: Object.freeze({
    maxStorageBuffersPerShaderStage: 10,
    maxStorageBufferBindingSize: 4294967292,
    maxBufferSize: 4294967292,
    maxComputeWorkgroupStorageSize: 32768,
    maxComputeInvocationsPerWorkgroup: 1024,
    maxComputeWorkgroupSizeX: 1024,
    maxComputeWorkgroupSizeY: 1024,
    maxComputeWorkgroupSizeZ: 64,
    maxComputeWorkgroupsPerDimension: 65535,
    maxUniformBufferBindingSize: 65536,
    minUniformBufferOffsetAlignment: 256,
    minStorageBufferOffsetAlignment: 256,
  }),
  /**
   * A device requested with all nine compute limits at the adapter maximum was granted all nine
   * unchanged. So `device.ts` asking for the adapter maxima is not a hopeful request on this
   * stack, it is one that lands, and the report carries the exact requiredLimits snippet.
   */
  requestedMaximaGranted: true,
  /**
   * Storage buffer allocations, each created and destroyed under both a validation and an out of
   * memory error scope. All five landed clean, so the granted `maxBufferSize` is not merely
   * advertised. Sizes in MiB.
   */
  bufferAllocationsMib: Object.freeze([64, 256, 512, 1024, 2048] as const),
  /**
   * The `subgroup_size` builtin, read in every lane of a 64 wide workgroup. One distinct value,
   * 32, and `subgroup_invocation_id` covering 0 to 31 in each half.
   */
  subgroupSizeBuiltin: Object.freeze({
    workgroupWidth: 64,
    distinctSizes: Object.freeze([32] as const),
    invocationIdMax: 31,
  }),
  /**
   * Why `@subgroup_size(32)` is not this engine's primary reduce path on Apple. Three spellings
   * were compiled through `getCompilationInfo` under a validation error scope, and all three were
   * refused. The strings are quoted from the live compiler, not paraphrased.
   */
  subgroupSizeControlAttempts: Object.freeze([
    Object.freeze({
      form: '@subgroup_size(32) with no directive',
      error: "use of '@subgroup_size' requires enabling extension "
        + "'chromium_experimental_subgroup_size_control'",
    }),
    Object.freeze({
      form: 'enable chromium_experimental_subgroup_size_control;',
      error: "extension 'chromium_experimental_subgroup_size_control' is not allowed in the "
        + 'current environment',
    }),
    Object.freeze({
      form: 'requires chromium_experimental_subgroup_size_control;',
      error: "feature 'chromium_experimental_subgroup_size_control' is not supported",
    }),
  ] as const),
  /**
   * The IndexedDB probe, which was run to turn this project's inherited 240 MB folklore figure
   * into a receipt and turned up something more useful than a number.
   *
   * On a healthy origin, single values of 1 MiB, 64 MiB and 256 MiB each wrote and read back byte
   * correct, and the origin's reported usage rose by exactly the value's bytes plus about 110 of
   * bookkeeping. Timings on that first pass, incompressible data, write then read: 1 MiB 3.6 and
   * 2.3 ms, 64 MiB 44.0 and 42.6 ms, 256 MiB 235.1 and 239.2 ms. So 240 MB is not a limit here.
   *
   * Then the same probe kept going, and the interesting part is what it broke. Writing 512 MiB
   * round tripped. Writing 1 GiB completed its transaction in 328.7 ms and then FAILED THE READ
   * with `UnknownError: Failed to read large IndexedDB value`. From that point the failing size
   * walked downward through the session: 768 MiB still read back, 640 MiB did not, and by the end
   * of the session even 128 MiB did not, while 96 MiB still did. `indexedDB.databases()` reported
   * no databases at all, every probe database having been deleted and awaited, yet
   * `navigator.storage.estimate()` reported 8.51 GB of usage attributed entirely to indexedDB,
   * which is very close to the total this probe had ever written, and quota had converged to
   * exactly that usage.
   *
   * The mechanism a cache has to survive, stated as behaviour rather than as a theory about
   * Chrome: deleting a database does not promptly release the origin's accounted usage, and an
   * origin whose accounted usage has caught up with its quota still ACCEPTS a large write and
   * then fails the matching read. A weight cache that trusts a completed write transaction as
   * proof the bytes are retrievable will report a healthy cache and then fail to load a model.
   */
  indexedDb: Object.freeze({
    healthyRoundTrips: Object.freeze([
      Object.freeze({ mib: 1, writeMs: 3.6, readMs: 2.3, usageDeltaBytes: 1048685 }),
      Object.freeze({ mib: 64, writeMs: 44.0, readMs: 42.6, usageDeltaBytes: 67108980 }),
      Object.freeze({ mib: 256, writeMs: 235.1, readMs: 239.2, usageDeltaBytes: 268435579 }),
    ] as const),
    /** The size that never failed a read in any state this probe reached, in MiB. */
    alwaysReadableMib: 64,
    /** The largest size that read back on a healthy origin, in MiB. */
    healthyCeilingMib: 768,
    readFailureName: 'UnknownError',
    readFailureMessage: 'Failed to read large IndexedDB value',
    /** True when a failing read followed a transaction that had reported success. */
    writeSucceedsWhenReadFails: true,
    /** Reported usage, in bytes, after every probe database had been deleted and awaited. */
    usageAfterDeletingEverything: 8507164764,
  }),
});

const M1_PROFILE: Gemma4DeviceProfile = {
  id: 'apple-metal3',
  measuredOn: M1_ADAPTER_PROBE.when,
  note:
    'Apple silicon on Metal 3, measured on the M1 that is this project\'s only development '
    + 'and measurement machine. Chrome reports no device or description string for an Apple '
    + 'adapter, only vendor and architecture, so this profile cannot tell an M1 from an M4 and '
    + 'does not pretend to: it is the Apple default until somebody probes a second Apple part and '
    + 'splits it.',
  matches: (info) =>
    (info.vendor ?? '').toLowerCase() === 'apple'
    && (info.architecture ?? '').toLowerCase().startsWith('metal'),
  tuning: {
    subgroupWidth: {
      value: 32,
      tuningReason:
        'adapter.info reported subgroupMinSize 32 and subgroupMaxSize 32, and a live dispatch '
        + 'reading the subgroup_size builtin returned 32 in every lane of a 64 wide workgroup '
        + '(probe of 2026-09-01). The butterfly assumes aligned groups of 32 and here that is the '
        + 'whole subgroup.',
    },
    subgroupSizeControl: {
      value: false,
      tuningReason:
        'no subgroup size control feature is offered under any name, and all three ways of asking '
        + 'were refused by the live compiler on 2026-09-01: the bare attribute wants an extension '
        + 'enabled, enabling that extension is "not allowed in the current environment", and '
        + 'requiring it is "not supported" (subgroupSizeControlAttempts carries the exact words). '
        + 'So a declared @subgroup_size(32) primary path is not available to write here at all, '
        + 'and the 32 lane butterfly is the only path rather than the fallback half of a pair.',
    },
    packedInt8Dot: {
      value: true,
      tuningReason:
        'navigator.gpu.wgslLanguageFeatures carries packed_4x8_integer_dot_product on this stack. '
        + 'It stays an optional fast path behind a feature check and is never required: the i32 '
        + 'accumulation contract is written with plain converts so it compiles everywhere.',
    },
    maxStorageBuffersPerShaderStage: {
      value: 10,
      tuningReason:
        'read from the adapter, 10, and requesting it in requiredLimits was granted. This is the '
        + 'number of ENGINE-PLAN risk 6 and DECODE-CAMPAIGN.md 4.7, confirmed live rather than '
        + 'recalled. Every kernel is designed against it from the first sketch.',
    },
    maxStorageBufferBindingSize: {
      value: 4294967292,
      tuningReason:
        'read from the adapter and granted in requiredLimits, four bytes short of four gibibytes. '
        + 'The whole PLE table therefore fits in a single binding with room to spare, so no split '
        + 'is needed on this class of device.',
    },
    maxBufferSize: {
      value: 4294967292,
      tuningReason:
        'read from the adapter and granted. Every allocation in the probe\'s own '
        + 'bufferAllocationsMib ladder succeeded with no validation and no out of memory error '
        + '(probe of 2026-09-01), so the granted limit is not merely advertised.',
    },
    maxComputeWorkgroupStorageSize: {
      value: 32768,
      tuningReason:
        'read from the adapter and granted, twice the WebGPU default of 16384. A prefill GEMM may '
        + 'stage twice the M tile here that the spec floor allows, which is a real degree of '
        + 'freedom for section 5.4 and is recorded so that lane does not have to rediscover it.',
    },
    maxComputeInvocationsPerWorkgroup: {
      value: 1024,
      tuningReason:
        'read from the adapter and granted, four times the WebGPU default of 256. The decode GEMV '
        + 'geometry below uses 64, so this is headroom rather than a constraint.',
    },
    attentionSlices: {
      value: 8,
      tuningReason:
        'the shipped shape, and the measured one here. This machine has 8 GPU cores and its '
        + 'attention occupancy knee is at exactly 8 workgroups '
        + '(lab-results/m1-attention-occupancy-sep04.json), so the dispatch already fills it and '
        + 'there is no width for a narrower workgroup to spread onto.',
    },
    attentionKvSplits: {
      value: 1,
      tuningReason:
        'no split. Measured across four contexts here, a split is a loss at every one of them '
        + 'except the long global shape, where it beats vAccumulators alone by three percent and '
        + 'costs a fold dispatch a layer to do it (lab-results/m1-attention-split-by-context-'
        + 'sep04.json). Not worth a second dispatch on a machine already at its knee.',
    },
    attentionVAccumulators: {
      value: 1,
      tuningReason:
        'Retain the shipped reduction order. September 6 end-to-end V=4 changes generated IDs '
        + 'on all three M1 prompts. The older model-page record checked tokenizer IDs, not '
        + 'generated output parity. See lab-results/m1-decode-sep06.json.',
    },
    attentionScoreAccumulators: {
      value: 4,
      tuningReason:
        'Retain four accumulators. September 6 paired M1 measurements of eight change generated '
        + 'IDs on all three prompts and show no consistent speed gain; see '
        + 'lab-results/m1-decode-sep06.json. Sixteen remains unmeasured here.',
    },
    attentionScoreLayout: {
      value: 'rows',
      tuningReason: 'The shipped loop. The dims layout was built for the 5070 and has not been run here.',
    },
    decodeGemv2KSplits: {
      value: 1,
      tuningReason:
        'no split, and this one is measured rather than deferred. Wired end to end it costs 1.7 '
        + 'percent at 2 and 23 percent at 4 here, with the token ids unchanged, and the isolated '
        + 'bench that predicted a win was wrong by 3.7x in the other direction. 8 cores cannot '
        + 'use the workgroups a split creates.',
    },
    decodeBatch2WorkgroupWidth: { value: 32, tuningReason: 'The September 6 M1 paired 32 versus 64 comparison did not establish a consistent advantage for 64. Retain 32; see lab-results/m1-decode-sep06.json.' },
    decodeBatchProjections: {
      value: true,
      tuningReason:
        'Measured on the 8-core Apple M1 with Chrome 152, September 6: independent Q/K/V and '
        + 'gate/up batching preserves the control token IDs and improves paired decode throughput. '
        + 'Keep the existing 32-thread width. See lab-results/m1-decode-sep06.json. Other Apple '
        + 'chips share this profile but their performance has not been measured.',
    },
    decodeProducerDownKSplits: { value: 1, tuningReason: 'September 6 paired M1 testing of the exact integer split at two preserves IDs but has no throughput gain. Retain one; see lab-results/m1-decode-sep06.json.' },
    decodePleGateKSplits: { value: 1, tuningReason: 'September 6 paired M1 testing of the exact integer split at two preserves IDs but is mixed across prompts. Retain one; see lab-results/m1-decode-sep06.json.' },
    decodeLookahead: {
      value: 2,
      tuningReason:
        'Retain two steps in flight. September 6 paired M1 testing of depth four preserves '
        + 'generated IDs but does not improve throughput. Timestamp diagnostics also add '
        + 'submission and readback overhead; see lab-results/m1-decode-sep06.json.',
    },
    decodeGemvWorkgroupWidth: {
      value: 64,
      tuningReason:
        'DECODE-CAMPAIGN.md 4.2 measured 64 wide with 2 subgroups and 4 rows per workgroup as the '
        + 'best decode GEMV shape on this machine, against 32 wide with 2 rows which caps resident '
        + 'warps and measured worse. Geometry is a safe knob under ENGINE-PLAN risk 2 mitigation '
        + '2; reduce order is not, and is frozen elsewhere.',
    },
    decodeGemvRowsPerWorkgroup: {
      value: 4,
      tuningReason: 'the rows half of the same measured shape (DECODE-CAMPAIGN.md 4.2).',
    },
    idbMaxValueBytes: {
      value: 67108864,
      tuningReason:
        'measured, and the measurement moved this number down rather than up. On a healthy origin '
        + 'every size the probe tried round tripped byte correct, so the folklore ceiling this '
        + 'project carried is not a limit on this stack. But the same probe then found the failure '
        + 'mode that matters: once the origin\'s accounted usage catches up with its quota, a '
        + 'large value still writes successfully and the matching read fails with "UnknownError: '
        + 'Failed to read large IndexedDB value", and the size at which that starts walks downward '
        + 'through the session. This value is the largest that never failed a read in any state '
        + 'reached, so it is the chunk ceiling. A cache reading it must still treat a failed read '
        + 'as a miss and refetch, because no size is a guarantee here. The whole ladder, with its '
        + 'timings and the exact failure, is in M1_ADAPTER_PROBE.indexedDb.',
    },
  },
};

/**
 * The profile for a device nobody has measured, which is every device except one.
 *
 * Its numbers are the WebGPU specification's guaranteed floors rather than optimistic ones, so a
 * kernel sized against this profile compiles on the weakest conforming adapter. Every reason here
 * says the same thing in different words, and that is the point: this profile is a statement that
 * the engine is running somewhere it has never been measured, and a report that quotes it is
 * telling the reader exactly that.
 */
const GENERIC_PROFILE: Gemma4DeviceProfile = {
  id: 'generic-webgpu',
  measuredOn: null,
  note:
    'The unmeasured default. Specification floors throughout, so nothing sized against it can be '
    + 'too large for a conforming adapter. Reaching this profile is not a fault; shipping a '
    + 'performance number measured under it as if it were tuned would be.',
  matches: () => true,
  tuning: {
    subgroupWidth: {
      value: 32,
      tuningReason:
        'the butterfly reduces within aligned groups of 32 on any width of 32 or more, and a '
        + 'device narrower than 32 fails the known answer self test and takes the workgroup tree, '
        + 'so 32 is the right thing to record for a device nobody has probed.',
    },
    subgroupSizeControl: {
      value: false,
      tuningReason: 'assumed absent until a live probe on the device says otherwise.',
    },
    packedInt8Dot: {
      value: false,
      tuningReason: 'assumed absent until a live probe says otherwise; it is never required.',
    },
    maxStorageBuffersPerShaderStage: {
      value: 8,
      tuningReason: 'the WebGPU specification default. Not measured on this device.',
    },
    maxStorageBufferBindingSize: {
      value: 134217728,
      tuningReason:
        'the WebGPU specification default, which is the floor a conservative adapter reports. '
        + 'Not measured. Note that the whole PLE table does not fit in one binding at this floor, '
        + 'so a device that really reports it needs a split table rather than a smaller model.',
    },
    maxBufferSize: {
      value: 268435456,
      tuningReason: 'the WebGPU specification default. Not measured on this device.',
    },
    maxComputeWorkgroupStorageSize: {
      value: 16384,
      tuningReason: 'the WebGPU specification default. Not measured on this device.',
    },
    maxComputeInvocationsPerWorkgroup: {
      value: 256,
      tuningReason: 'the WebGPU specification default. Not measured on this device.',
    },
    attentionSlices: {
      value: 8,
      tuningReason: 'the shipped shape. Nobody has measured the occupancy knee on this device.',
    },
    attentionKvSplits: {
      value: 1,
      tuningReason:
        'no split, which is the shape that needs no second dispatch. A device that wants one has '
        + 'to be measured into a profile of its own; defaulting an unmeasured machine into an '
        + 'extra dispatch a layer is the wrong direction to guess in.',
    },
    attentionVAccumulators: {
      value: 1,
      tuningReason: 'the shipped chain. Not measured on this device.',
    },
    attentionScoreAccumulators: {
      value: 4,
      tuningReason: 'the shipped text. Not measured on this device.',
    },
    attentionScoreLayout: {
      value: 'rows',
      tuningReason: 'the shipped loop. Not measured on this device.',
    },
    decodeGemv2KSplits: {
      value: 1,
      tuningReason: 'no split, for the same reason as attentionKvSplits. Not measured on this device.',
    },
    decodeBatch2WorkgroupWidth: { value: 32, tuningReason: 'Retain the existing 2-bit producer width outside the measured Blackwell profile.' },
    decodeBatchProjections: { value: false, tuningReason: 'Not measured on this device family; retain separate projections.' },
    decodeProducerDownKSplits: { value: 1, tuningReason: 'Not measured on this device family; retain the original producer down dot.' },
    decodePleGateKSplits: { value: 1, tuningReason: 'Not measured on this device family; retain the original PLE gate dot.' },
    decodeLookahead: {
      value: 2,
      tuningReason:
        'the shipped loop, one step awaited and one behind it. A deeper queue only pays where a '
        + 'token is shorter than the readback hop, which has been measured on one discrete part '
        + 'and on no unknown adapter; two is the shape every measured device is correct at.',
    },
    decodeGemvWorkgroupWidth: {
      value: 64,
      tuningReason:
        'carried over from the one machine where it was measured (DECODE-CAMPAIGN.md 4.2) because '
        + 'a starting shape has to be something, and 64 fits inside the specification floor of 256 '
        + 'invocations per workgroup. Nobody has measured it here.',
    },
    decodeGemvRowsPerWorkgroup: {
      value: 4,
      tuningReason: 'same, and 64 times 4 is inside the 256 invocation floor.',
    },
    idbMaxValueBytes: {
      value: 33554432,
      tuningReason:
        'half of the size that never failed a read on the one machine anybody probed, as a '
        + 'conservative chunk ceiling for a browser and disk nobody has measured. The probe that '
        + 'set that number found large value reads failing after large value writes had reported '
        + 'success, so erring small here costs a few more keys and buys a cache that can be read '
        + 'back. Raise it with a measurement, never with an assumption.',
    },
  },
};

/**
 * The RTX 5070, measured on 2026-09-04 and 2026-09-05 on the machine that owns it, with every
 * geometry field below carrying the family that set it (lab-results/5070-*.json) and the parity
 * receipt that cleared it (lab-results/5070-parity-r5-sep05.json, ENGINE-PLAN ruling 6.1.1).
 *
 * Chrome reports vendor 'nvidia' and architecture 'blackwell' for this adapter and nothing that
 * tells a 5070 from the rest of the family, so this profile is the Blackwell default the way the
 * Apple profile is the Apple default: the 5070's measured shape is the nearest measured shape for
 * a 5060 or a 5090 until somebody probes one, and a 5090 with three and a half times the SMs will
 * want more splits than this, not fewer. The limits are the ones the sweep page read off this
 * adapter; the two features nobody probed say so.
 */
const RTX5070_PROFILE: Gemma4DeviceProfile = {
  id: 'nvidia-blackwell',
  measuredOn: '2026-09-05',
  note:
    'NVIDIA Blackwell, measured on an RTX 5070 (48 SMs, 672 gigabytes per second) under Chrome on Windows on '
    + '2026-09-04 and 2026-09-05. Every geometry value below is the winner of a pre-registered '
    + 'family with controls at both ends, and the whole configuration is token identical to the '
    + 'reference over the sixteen judged tokens on all three probes (parity r5). The profile '
    + 'matches every Blackwell adapter because Chrome exposes nothing finer; other parts inherit '
    + 'the 5070 shape as the nearest measured one.',
  matches: (info) =>
    (info.vendor ?? '').toLowerCase() === 'nvidia'
    && (info.architecture ?? '').toLowerCase().startsWith('blackwell'),
  tuning: {
    subgroupWidth: {
      value: 32,
      tuningReason:
        'the perf page runs the 32 lane known answer self test before every round and it passed on '
        + 'every round of every family on this adapter. Warps are 32 wide on this part.',
    },
    subgroupSizeControl: {
      value: false,
      tuningReason: 'not probed on this adapter. Assumed absent, as on the one machine where it was.',
    },
    packedInt8Dot: {
      value: false,
      tuningReason: 'not probed on this adapter. Never required.',
    },
    maxStorageBuffersPerShaderStage: {
      value: 8,
      tuningReason: 'the specification default; withLiveLimits reads the live value at load. Not probed here.',
    },
    maxStorageBufferBindingSize: {
      value: 2147483644,
      tuningReason: 'read off the adapter by src/dev/gemvsweep.html on 2026-09-05.',
    },
    maxBufferSize: {
      value: 2147483648,
      tuningReason: 'read off the adapter by src/dev/gemvsweep.html on 2026-09-05, and every perf round records the same.',
    },
    maxComputeWorkgroupStorageSize: {
      value: 16384,
      tuningReason: 'the specification default; withLiveLimits reads the live value at load. Not probed here.',
    },
    maxComputeInvocationsPerWorkgroup: {
      value: 256,
      tuningReason: 'the specification default; withLiveLimits reads the live value at load. Not probed here.',
    },
    attentionSlices: {
      value: 1,
      tuningReason:
        'one 64 lane worker a workgroup, with the KV split below carrying the parallelism. The '
        + 'attention occupancy curve on this part has its knee at 32 workgroups against the 8 the '
        + 'shipped shape dispatches (lab-results/5070-attention-occupancy-sep04.json), and slices 1 '
        + 'with kvSplits 8 was the best row of the three axis grid at 2.74x and 2.99x '
        + '(5070-attention-split-sep04.json). attn=2 on the same split lost 3.6 percent on '
        + 'interview-300 and moved its first token (5070-geometry-sweep-e-family-sep04.json).',
    },
    attentionKvSplits: {
      value: 8,
      tuningReason:
        'eight workgroups a head share the window. Worth 26 percent on interview-300 alone and '
        + 'additive with the down_proj split (5070-both-splits-d1-d5-sep04.json). kvSplits 4 ties '
        + 'at 16 and 38 prompt tokens and loses 15.6 percent at 299; kvSplits 16 loses 2 to 3 '
        + 'percent up to 483 tokens and wins 3.3 percent at 563, so 8 is the best fixed value over '
        + 'the contexts the app runs (5070-kvsplit-at-length-f-family-sep04.json).',
    },
    attentionVAccumulators: {
      value: 8,
      tuningReason:
        'eight chains in the V loop, worth 4.9 percent over four and saturating there: sixteen is '
        + '5.0 alone and spills beside sixteen score accumulators '
        + '(5070-attention-chain-depth-o-family-sep04.json).',
    },
    attentionScoreAccumulators: {
      value: 16,
      tuningReason:
        'sixteen score chains beside eight V chains is the best arm of the o family at +6.7 '
        + 'percent mean over eight contexts, ahead of va=8 alone on eight of eight. Alone, deeper '
        + 'score chains are SLOWER on this part, 8.2 percent at sixteen, so this value is only '
        + 'right beside vAccumulators 8 (5070-attention-chain-depth-o-family-sep04.json).',
    },
    attentionScoreLayout: {
      value: 'dims',
      tuningReason:
        'the transposed score loop, worth 0.7 percent at 299 prompt tokens rising to 4.6 at 483, '
        + 'positive on eight of eight contexts, and identical to the reference where the rows loop '
        + 'had drifted at interview-300 position 13 '
        + '(5070-attention-transposed-score-u-family-sep04.json, 5070-parity-r5-sep05.json).',
    },
    decodeGemv2KSplits: {
      value: 8,
      tuningReason:
        'the 2-bit down_proj K split. Its shape is 96 workgroups on a 48 SM part and it ran at 129 '
        + 'GB/s against its transposed sibling at 428 on the same bytes; the split is worth 12 to '
        + '16 percent and moved not one token in 192 (5070-both-splits-d1-d5-sep04.json). '
        + 'kSplits 4 loses 1.2 to 1.8 percent to 8 (5070-geometry-sweep-e-family-sep04.json).',
    },
    decodeBatch2WorkgroupWidth: { value: 64, tuningReason: 'RTX 5070, cx74-cx76: wider batched 2-bit producers improve eight-prompt median by about 1 percent; 64, 128 and 256 are close, so keep 64. Per-projection arithmetic and IDs unchanged. lab-results/5070-codex-projection-splits-sep06.json.' },
    decodeBatchProjections: { value: true, tuningReason: 'RTX 5070 Chrome 152, cx20 between cx15/cx21 controls: about 5 percent faster across eight prompts, all 2560 generated IDs identical. GPU subgroup and workgroup fixtures exact, full parity r8.' },
    decodeProducerDownKSplits: { value: 2, tuningReason: 'RTX 5070 cx33-cx36: about 1 percent over batched projections. Split integer dots before scaling to preserve every sum; full parity r9.' },
    decodePleGateKSplits: { value: 2, tuningReason: 'RTX 5070 cx42: median 293.5 across eight prompts with batching and producer splits; 2560 IDs identical to cx15, full parity r10. All 35 checkpoint gate row bounds below 2^24.' },
    decodeLookahead: {
      value: 4,
      tuningReason:
        'four decode steps in flight against the shipped two. Worth 3.3 percent over the two deep '
        + 'loop on the same tree, 275.4 to 284.6 tok/s median over the eight prompts, ids identical '
        + 'to the control on every run of every prompt, on this machine in the headed instrument '
        + '(lab-results/5070-attention-prologue-and-queue-sep05.json, rounds l2 and l3); first '
        + 'seen at depth 4 in a headless instrument by a second session (rounds cx05 and cx06). '
        + 'A token is under 4 ms here and the readback hop is about 3, so two steps leave the '
        + 'GPU waiting on the host whenever the hop runs long; four never do.',
    },
    decodeGemvWorkgroupWidth: {
      value: 64,
      tuningReason:
        'recorded, not applied (see applyProfileGeometry). The 4-bit family at 64 lanes by 4 rows '
        + 'is the optimum of sixteen shapes on this part and every direction away from it is '
        + 'slower, a reassociation, or broken (5070-gemv4-geometry-h-family-sep04.json).',
    },
    decodeGemvRowsPerWorkgroup: {
      value: 4,
      tuningReason: 'recorded, not applied. Same measurement as the width.',
    },
    idbMaxValueBytes: {
      value: 33554432,
      tuningReason:
        'the generic ceiling. IndexedDB was not probed on this machine: every load here is a ranged '
        + 'read of the pinned snapshot from the dev server. Raise it with a measurement.',
    },
  },
};

/** Registry order is match order, most specific first, with the catch all last. */
export const DEVICE_PROFILES: readonly Gemma4DeviceProfile[] = Object.freeze([
  M1_PROFILE,
  RTX5070_PROFILE,
  GENERIC_PROFILE,
]);

export function profileById(id: string): Gemma4DeviceProfile | null {
  return DEVICE_PROFILES.find((p) => p.id === id) ?? null;
}

// --------------------------------------------------------------------------- resolution

const FIELDS: readonly Gemma4ProfileField[] = Object.freeze([
  'subgroupWidth',
  'subgroupSizeControl',
  'packedInt8Dot',
  'maxStorageBuffersPerShaderStage',
  'maxStorageBufferBindingSize',
  'maxBufferSize',
  'maxComputeWorkgroupStorageSize',
  'maxComputeInvocationsPerWorkgroup',
  'attentionSlices',
  'attentionKvSplits',
  'attentionVAccumulators',
  'attentionScoreAccumulators',
  'attentionScoreLayout',
  'decodeGemv2KSplits',
  'decodeLookahead',
  'decodeBatchProjections',
  'decodeBatch2WorkgroupWidth',
  'decodeProducerDownKSplits',
  'decodePleGateKSplits',
  'decodeGemvWorkgroupWidth',
  'decodeGemvRowsPerWorkgroup',
  'idbMaxValueBytes',
]);

function isTuned(value: unknown): value is Tuned<number | boolean | string> {
  return typeof value === 'object' && value !== null && 'value' in value && 'tuningReason' in value;
}

export interface ResolveOptions {
  /** Force a profile by id, skipping the match. Throws when the id is not registered. */
  profileId?: string;
  /** Field by field override, merged over whichever profile was chosen. */
  override?: Gemma4ProfileOverride;
}

/**
 * Choose a profile for an adapter and merge any override over it.
 *
 * Resolution is total: `GENERIC_PROFILE` matches everything, so this never returns null and no
 * caller needs a null branch. What a caller does need to read is `matchedBy` and `measuredOn`,
 * because a resolved profile whose `measuredOn` is null is guesswork wearing a data structure,
 * and a report that does not say so is misleading.
 */
export function resolveDeviceProfile(
  info: Gemma4ProfileInput,
  options: ResolveOptions = {},
): ResolvedDeviceProfile {
  let base: Gemma4DeviceProfile;
  let matchedBy: string;
  if (options.profileId !== undefined) {
    const forced = profileById(options.profileId);
    if (!forced) {
      throw new Error(
        `resolveDeviceProfile: no profile with id ${options.profileId}. `
        + `Known: ${DEVICE_PROFILES.map((p) => p.id).join(', ')}`,
      );
    }
    base = forced;
    matchedBy = 'explicit';
  } else {
    base = DEVICE_PROFILES.find((p) => p.matches(info)) ?? GENERIC_PROFILE;
    matchedBy = base === GENERIC_PROFILE ? 'fallback' : base.id;
  }

  const merged = { ...base.tuning } as Record<Gemma4ProfileField, Tuned<number | boolean | string>>;
  const overridden: Gemma4ProfileField[] = [];
  const override = options.override;
  if (override) {
    for (const field of FIELDS) {
      const supplied = override[field];
      if (supplied === undefined) continue;
      overridden.push(field);
      if (isTuned(supplied)) {
        merged[field] = { value: supplied.value, tuningReason: supplied.tuningReason };
      } else {
        merged[field] = {
          value: supplied,
          tuningReason:
            `explicit override of ${base.id}.${field} to ${String(supplied)} with no reason `
            + 'recorded by the caller',
        };
      }
    }
  }

  return {
    ...(merged as unknown as Gemma4ProfileTuning),
    id: base.id,
    note: base.note,
    measuredOn: base.measuredOn,
    matchedBy,
    overridden,
  };
}

/**
 * Clamp a resolved profile's capacity caps against what the live device actually granted.
 *
 * A profile is guidance and the device's limits are truth. Where they disagree the smaller number
 * wins and the clamp is recorded in the reason, so a report shows both the belief and the fact
 * rather than quietly replacing one with the other. This is the direction that matters: a profile
 * promising more than the adapter grants is how a kernel gets written against a binding count that
 * cannot be requested, which is the failure ENGINE-PLAN risk 6 exists to prevent.
 */
export function withLiveLimits(
  profile: ResolvedDeviceProfile,
  limits: Gemma4DeviceLimits,
): ResolvedDeviceProfile {
  const caps: readonly (Gemma4ProfileField & keyof Gemma4DeviceLimits)[] = [
    'maxStorageBuffersPerShaderStage',
    'maxStorageBufferBindingSize',
    'maxBufferSize',
    'maxComputeWorkgroupStorageSize',
    'maxComputeInvocationsPerWorkgroup',
  ];
  const out = { ...profile } as Record<string, unknown>;
  for (const field of caps) {
    const believed = profile[field].value;
    const granted = limits[field];
    if (typeof granted !== 'number' || granted <= 0 || granted === believed) continue;
    out[field] = {
      value: Math.min(believed, granted),
      tuningReason:
        `${profile[field].tuningReason} Clamped against the live device, which granted `
        + `${granted}${granted < believed ? ', below the profile' : ', above the profile'}.`,
    } satisfies Tuned<number>;
  }
  // Q/K/V batching binds ten storage buffers. A matched adapter family alone is insufficient:
  // browser/device limits may be lower than the hardware probe on which the profile was based.
  if (profile.decodeBatchProjections.value && limits.maxStorageBuffersPerShaderStage < 10) {
    out.decodeBatchProjections = {
      value: false,
      tuningReason: `${profile.decodeBatchProjections.tuningReason} Disabled because Q/K/V batching `
        + `requires 10 storage buffers and the live device granted ${limits.maxStorageBuffersPerShaderStage}.`,
    } satisfies Tuned<boolean>;
  }
  return out as unknown as ResolvedDeviceProfile;
}

/** Every field with its value and reason, for a report. One line per field, stable order. */
/**
 * Apply the profile's kernel geometry, before any pipeline is built.
 *
 * THE FIELDS ABOVE WERE RECORDED AND NEVER READ. Every geometry number in this file has been a
 * note to a future reader since it was written: nothing called a setter with them, so a profile
 * could name a shape the engine did not run and no gate would notice. That is the hole this
 * closes, and it closes it for the four fields that can be closed honestly.
 *
 * `decodeGemvWorkgroupWidth` and `decodeGemvRowsPerWorkgroup` are deliberately still not applied.
 * They name ONE width and ONE row count, and since round 3 the engine has run a different geometry
 * per quantization family: 32 by 16 on the 2-bit tile, 64 by 4 on the 4-bit, 64 by 2 on the 8-bit.
 * A single pair cannot express that, and pushing it onto all three would move two families off
 * their measured shapes. Fixing that means splitting those two fields per family, which is a
 * change to what the profile MEANS and wants its own measurement; until then they stay what they
 * have always been, which is a record.
 *
 * Returns what it changed, so a caller can say so and a gate can assert that a profile carrying
 * the shipped values changes nothing at all.
 */
export function applyProfileGeometry(
  profile: ResolvedDeviceProfile,
  setters: {
    setAttentionGeometry: (g: {
      slices: 1 | 2 | 4 | 8;
      kvSplits?: 1 | 2 | 4 | 8 | 16;
      vAccumulators?: 1 | 2 | 4 | 8 | 16;
      scoreAccumulators?: 4 | 8 | 16;
      scoreLayout?: 'rows' | 'dims';
    } | null) => void;
    setGemvGeometry: (g: GemvGeometryLike | null, bits: 2 | 4 | 8) => void;
    gemvGeometry: (bits: 2 | 4 | 8) => Readonly<GemvGeometryLike>;
  },
): string[] {
  const changed: string[] = [];
  const slices = profile.attentionSlices.value as 1 | 2 | 4 | 8;
  const kvSplits = profile.attentionKvSplits.value as 1 | 2 | 4 | 8 | 16;
  const vAcc = profile.attentionVAccumulators.value as 1 | 2 | 4 | 8 | 16;
  const sAcc = profile.attentionScoreAccumulators.value as 4 | 8 | 16;
  const layout = profile.attentionScoreLayout.value as 'rows' | 'dims';
  // The shipped attention shape is slices 8 with none of the other four, so a profile that names
  // exactly that is left alone rather than being set to an equal value. Setting it would be
  // harmless and would still read as a change in the log, which is the thing worth avoiding.
  if (slices !== 8 || kvSplits !== 1 || vAcc !== 1 || sAcc !== 4 || layout !== 'rows') {
    setters.setAttentionGeometry({
      slices,
      ...(kvSplits === 1 ? {} : { kvSplits }),
      ...(vAcc === 1 ? {} : { vAccumulators: vAcc }),
      ...(sAcc === 4 ? {} : { scoreAccumulators: sAcc }),
      ...(layout === 'rows' ? {} : { scoreLayout: layout }),
    });
    changed.push(`attention slices ${slices}, kvSplits ${kvSplits}, vAccumulators ${vAcc}, scoreAccumulators ${sAcc}, scoreLayout ${layout}`);
  }
  const kSplits = profile.decodeGemv2KSplits.value as 1 | 2 | 4 | 8;
  if (kSplits !== 1) {
    setters.setGemvGeometry({ ...setters.gemvGeometry(2), kSplits }, 2);
    changed.push(`2-bit decode GEMV kSplits ${kSplits}`);
  }
  return changed;
}

/** The shape applyProfileGeometry needs from a GEMV geometry, without importing the kernel lane. */
export interface GemvGeometryLike {
  readonly workgroupSize: number;
  readonly rowsPerVsg: number;
  readonly wordsPerLane: 1 | 2 | 4;
  readonly inner?: 'classic' | 'tile16u' | 'tile16' | 'tilefloor' | 'tile8u';
  readonly kSplits?: 1 | 2 | 4 | 8;
}

export function describeProfile(profile: ResolvedDeviceProfile): string[] {
  const head = `${profile.id} (matched ${profile.matchedBy}, measured `
    + `${profile.measuredOn ?? 'never'})`;
  const lines = [head];
  for (const field of FIELDS) {
    lines.push(`  ${field} = ${String(profile[field].value)}  ${profile[field].tuningReason}`);
  }
  if (profile.overridden.length > 0) {
    lines.push(`  overridden: ${profile.overridden.join(', ')}`);
  }
  return lines;
}
