// Written from docs/ENGINE-PLAN.md's research corrections, section 5 kernel budget item 4 and
// risk 6, the WebGPU specification's limit table, and this project's own adapter probe recorded in
// ./deviceProfile.ts. No vendored bundle, no extracted kernel and no third party engine source was
// read.
//
// SPDX-License-Identifier: Apache-2.0
//
// The adapter floor: how a table too large for one storage binding is carried anyway.
//
// THE PROBLEM, stated as arithmetic rather than as a worry. The PLE table is [262144, 8960] at
// 4 bits, which is 4,480 packed bytes per vocabulary row and 1,174,405,120 bytes in total
// (EMBEDDING-QUANT-FINDING.md). WebGPU's default `maxStorageBufferBindingSize` is 134,217,728
// bytes and the practical range across adapters runs from that floor to about 4 GB. On the M1 this
// project develops on, the adapter reports and grants 4,294,967,292, so the whole table is one
// binding with room to spare and nothing here changes anything. On an adapter at the specification
// floor the table does not fit in one binding at all, and it is not a smaller model that is
// needed, it is a split one.
//
// THE SHAPE OF THE SPLIT, and why it is by vocabulary row. A gather reads one whole row per token
// id. Slicing by row keeps every read inside one slice, so a slice is addressable with the same
// arithmetic the unsliced table uses minus an offset, and no invocation ever straddles two
// bindings. Slicing by column would put a single row's 4,480 bytes across two buffers and turn one
// word load into a branch, which is the shape to avoid on the kernel that runs 35 groups deep.
//
// THE BINDING BUDGET, which is what decides the dispatch shape rather than the buffer shape. Nine
// slices at the 128 MiB floor against the ten `maxStorageBuffersPerShaderStage` this project
// measured on M1 Chrome would leave one binding for the scales, the ids and the destination
// together, which is three bindings short. So the gather does NOT bind every slice at once. It
// binds exactly one slice per dispatch and runs once per slice, with the slice's vocabulary range
// in its uniform and every invocation whose token id falls outside that range returning without
// writing. The slices partition the vocabulary, so every output slot is written by exactly one of
// those dispatches, and the storage binding count is four whatever the slice count is.
//
// The cost is dispatch count, not bandwidth: at decode a single token gathers from one slice and
// the other eight dispatches return immediately, which is eight launches of a kernel that reads
// nothing. On the machine where this matters at all that is the price of running, and on every
// adapter that reports a real limit the plan is one slice and one dispatch, exactly as today.

/** One contiguous run of vocabulary rows, carried in its own storage buffer. */
export interface TableSlice {
  readonly index: number;
  /** First vocabulary row this slice carries. */
  readonly startRow: number;
  /** How many rows it carries. The last slice is short whenever the count does not divide. */
  readonly rowCount: number;
  /** Byte offset of the slice inside the whole table, which is `startRow * rowBytes`. */
  readonly byteOffset: number;
  readonly byteLength: number;
}

export interface TableSplitInput {
  /** The tensor this plan is for, quoted in the plan's own note. */
  readonly name: string;
  /** Vocabulary rows. 262,144 for both gathers on this checkpoint. */
  readonly rows: number;
  /** Packed bytes per row. 4,480 for the PLE codes, 384 for the input embedding codes. */
  readonly rowBytes: number;
  /** `adapter.limits.maxStorageBufferBindingSize`, never a constant (kernel budget item 4). */
  readonly maxStorageBufferBindingSize: number;
  /** `adapter.limits.maxBufferSize`. A slice has to be allocatable as well as bindable. */
  readonly maxBufferSize: number;
  /** `adapter.limits.maxStorageBuffersPerShaderStage`, risk 6's number. */
  readonly maxStorageBuffersPerShaderStage: number;
  /**
   * Largest slice this planner will cut, once it has decided a split is needed at all. Defaults to
   * `DEFAULT_MAX_SLICE_BYTES`.
   *
   * WHY A SLICE IS NOT SIMPLY AS LARGE AS THE LIMIT ALLOWS. The first version of this packed each
   * slice to `min(maxStorageBufferBindingSize, maxBufferSize)`, which on iOS produced one buffer of
   * 1,073,694,720 bytes, forty seven kilobytes under that device's stated maximum. An adapter's
   * advertised maximum is what it will validate, not what it can still find when a gigabyte and a
   * half is already live, and a single allocation that large has to be backed contiguously. The
   * instrumented load shows the whole table arriving as one 1,074 MB step at 300 MB resident, and
   * the phone dying as the writes into it begin, at the same byte on two browsers and two builds.
   *
   * Smaller slices cost one gather dispatch each and ask the driver for something it can actually
   * place. This is the same trade the header already accepted at nine slices on the specification
   * floor.
   */
  readonly maxSliceBytes?: number;
}

/**
 * The cap on one slice once a table has to be split, chosen to sit alongside the engine's other
 * large weights rather than at the adapter's ceiling: the next biggest buffers this model allocates
 * are `lm_head.weight` and `embed_tokens.embedding_quantized` at about 101 MB each.
 */
export const DEFAULT_MAX_SLICE_BYTES = 256 * 1024 * 1024;

export interface TableSplitPlan {
  readonly name: string;
  readonly rows: number;
  readonly rowBytes: number;
  readonly totalBytes: number;
  /** The smaller of the binding and allocation limits, which is what actually caps a slice. */
  readonly limitBytes: number;
  /** Rows that fit in one slice under that cap. */
  readonly rowsPerSlice: number;
  readonly slices: readonly TableSlice[];
  /** True when the whole table is one buffer, which is the M1's answer and today's fast path. */
  readonly single: boolean;
  /**
   * Storage bindings one gather dispatch uses: codes, scales, ids, destination. Four, and four
   * whatever the slice count is, because a dispatch binds one slice.
   */
  readonly storageBindingsPerDispatch: number;
  readonly bindingBudget: number;
  /** One dispatch per slice. Equal to `slices.length`. */
  readonly dispatchesPerGather: number;
  /** False only when a single row does not fit in one binding, which no adapter reports. */
  readonly feasible: boolean;
  readonly note: string;
}

/** Storage bindings the gather kernel declares: codes, scales, ids, destination. */
export const GATHER_STORAGE_BINDINGS = 4;

/**
 * Plan the split for one table against one adapter's reported limits.
 *
 * The limits go in as numbers rather than as a device, so this is a pure function a Node check can
 * drive with the specification floor, with the M1's granted maxima, and with anything in between,
 * and prove both plans from the same code the browser runs.
 */
export function planTableSplit(input: TableSplitInput): TableSplitPlan {
  const rows = Math.max(0, Math.trunc(input.rows));
  const rowBytes = Math.max(1, Math.trunc(input.rowBytes));
  const totalBytes = rows * rowBytes;
  const limitBytes = Math.max(
    0,
    Math.min(
      Math.trunc(input.maxStorageBufferBindingSize),
      Math.trunc(input.maxBufferSize),
    ),
  );
  const bindingBudget = Math.trunc(input.maxStorageBuffersPerShaderStage);

  // The whole table in one buffer stays exactly that, on every adapter with room for it. The cap
  // below applies only once a split is unavoidable, so an adapter that never needed one is
  // untouched by this and keeps the single buffer and the single dispatch it has always had.
  const fitsWhole = totalBytes > 0 && totalBytes <= limitBytes;
  const sliceCap = Math.max(
    1,
    Math.min(limitBytes, Math.trunc(input.maxSliceBytes ?? DEFAULT_MAX_SLICE_BYTES)),
  );
  const rowsPerSlice = Math.floor((fitsWhole ? limitBytes : sliceCap) / rowBytes);
  if (rowsPerSlice < 1) {
    return {
      name: input.name,
      rows,
      rowBytes,
      totalBytes,
      limitBytes,
      rowsPerSlice: 0,
      slices: [],
      single: false,
      storageBindingsPerDispatch: GATHER_STORAGE_BINDINGS,
      bindingBudget,
      dispatchesPerGather: 0,
      feasible: false,
      note:
        `${input.name}: one row is ${rowBytes} bytes and the adapter caps a storage binding at `
        + `${limitBytes}, so a single row does not fit in a binding. A row level split would be a `
        + 'different kernel, and no adapter this engine has seen reports a limit anywhere near '
        + 'that small.',
    };
  }

  // Slices start on a sixteen row boundary: the 2-bit table is read in the interleaved tile
  // layout (kernels/qlayout.ts TILE_ROWS) and a slice's local row indexes the slice's own
  // tiles. Rounding down costs at most fifteen rows of capacity per slice on any table.
  const aligned = rowsPerSlice >= 16 ? rowsPerSlice - (rowsPerSlice % 16) : rowsPerSlice;
  const perSlice = Math.min(rows, aligned);
  const slices: TableSlice[] = [];
  for (let start = 0; start < rows; start += perSlice) {
    const rowCount = Math.min(perSlice, rows - start);
    slices.push({
      index: slices.length,
      startRow: start,
      rowCount,
      byteOffset: start * rowBytes,
      byteLength: rowCount * rowBytes,
    });
  }
  if (slices.length === 0) {
    slices.push({ index: 0, startRow: 0, rowCount: 0, byteOffset: 0, byteLength: 0 });
  }

  const single = slices.length === 1;
  const note = single
    ? `${input.name}: ${totalBytes} bytes fit in one binding of ${limitBytes}, so the whole table `
      + 'is one buffer and one dispatch. This is the M1\'s answer and it is a fact about that '
      + 'adapter rather than a design decision.'
    : `${input.name}: ${totalBytes} bytes against a binding cap of ${limitBytes} needs `
      + `${slices.length} slices of at most ${perSlice} vocabulary rows. The gather binds one `
      + `slice per dispatch and runs ${slices.length} times, so it uses `
      + `${GATHER_STORAGE_BINDINGS} storage bindings of the adapter's ${bindingBudget} rather than `
      + `${slices.length + 3}, which would not fit.`;

  return {
    name: input.name,
    rows,
    rowBytes,
    totalBytes,
    limitBytes,
    rowsPerSlice: perSlice,
    slices,
    single,
    storageBindingsPerDispatch: GATHER_STORAGE_BINDINGS,
    bindingBudget,
    dispatchesPerGather: slices.length,
    feasible: GATHER_STORAGE_BINDINGS <= bindingBudget,
    note,
  };
}

/** Which slice carries a vocabulary row, or -1 when no slice does. */
export function sliceForRow(plan: TableSplitPlan, row: number): number {
  for (const slice of plan.slices) {
    if (row >= slice.startRow && row < slice.startRow + slice.rowCount) return slice.index;
  }
  return -1;
}

/** The limits half of a device profile, in the shape `planTableSplit` wants. */
export interface GatherLimits {
  readonly maxStorageBufferBindingSize: number;
  readonly maxBufferSize: number;
  readonly maxStorageBuffersPerShaderStage: number;
}

export interface GatherSplitPlans {
  /** The packed code table, which is the one that can need splitting. */
  readonly codes: TableSplitPlan;
  /** The scale table, planned by the same function so a tiny binding limit is visible rather than assumed. */
  readonly scales: TableSplitPlan;
  /**
   * True when the scale table is one buffer, which every kernel here assumes: a sliced code
   * buffer is addressed by a local row and the scales are still addressed by the absolute row.
   * False is not a crash, it is a plan the gather kernel does not implement yet, and the note says
   * so rather than the shader reading past the end of a slice.
   */
  readonly scalesSingle: boolean;
  readonly note: string;
}

/**
 * Plan both tables one gather needs.
 *
 * The two are planned separately because they hit the cap at completely different sizes. On this
 * checkpoint the PLE codes are 1,174,405,120 bytes and its scales are 262,144 rows of 35 f32,
 * which is 36,700,160 bytes and inside the specification floor by a factor of three. So the split
 * that matters is the codes, and the scales are planned anyway so that an adapter reporting
 * something smaller than the specification floor shows up as a plan rather than as a wrong answer.
 */
export function planGatherSplit(
  name: string,
  rows: number,
  codeRowBytes: number,
  scalesPerRow: number,
  limits: GatherLimits,
): GatherSplitPlans {
  const codes = planTableSplit({
    name: `${name} codes`,
    rows,
    rowBytes: codeRowBytes,
    ...limits,
  });
  const scales = planTableSplit({
    name: `${name} scales`,
    rows,
    rowBytes: scalesPerRow * 4,
    ...limits,
  });
  const scalesSingle = scales.single;
  return {
    codes,
    scales,
    scalesSingle,
    note: scalesSingle
      ? `${codes.note} ${scales.name}: one buffer, which is what the gather kernel assumes.`
      : `${codes.note} ${scales.name} does NOT fit in one binding on this adapter, and the sliced `
        + 'gather addresses scales by the absolute vocabulary row, so this configuration needs a '
        + 'second slice index in the uniform before it is correct. It has never been reached on '
        + 'any adapter this project has probed.',
  };
}
