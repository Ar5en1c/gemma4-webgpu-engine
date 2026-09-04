// Written from docs/ENGINE-PLAN.md sections 3 and 5, the WebGPU specification (buffer usage
// flags and the 256 byte minUniformBufferOffsetAlignment default), and the residency facts in
// ../../../../gemma4-kernels-lab/DECODE-CAMPAIGN.md. No vendored bundle, no extracted kernel and
// no third party engine source was read.
//
// SPDX-License-Identifier: Apache-2.0
//
// The buffer manager: weight buffers by tensor name, activation ping-pong pairs, a uniform arena
// with offset discipline, and a bind group cache. Everything GPU shaped is typed structurally
// against the small DeviceLike surface below rather than against GPUDevice directly, so the same
// code runs against the real device in the engine and against a fake recording device in the
// Node checks and the dry forward walk. The offset arithmetic lives in its own pure class for the
// same reason. This file has no imports, so the check runner can transpile and load it directly.

// ------------------------------------------------------------------- platform constants

/**
 * GPUBufferUsage bit values, transcribed from the WebGPU IDL. Restated as runtime values here
 * because the browser global of the same name does not exist under Node, where the checks and the
 * dry walk run. webgpuFlags.d.ts declares the browser global for files that only run in a page.
 */
export const USAGE = Object.freeze({
  MAP_READ: 0x0001,
  MAP_WRITE: 0x0002,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  INDEX: 0x0010,
  VERTEX: 0x0020,
  UNIFORM: 0x0040,
  STORAGE: 0x0080,
  INDIRECT: 0x0100,
  QUERY_RESOLVE: 0x0200,
});

/**
 * The spec's default minUniformBufferOffsetAlignment and minStorageBufferOffsetAlignment are both
 * 256. The engine aligns to the spec default rather than the adapter's possibly smaller value,
 * because a stricter alignment is valid everywhere and saves nothing worth having here.
 */
export const UNIFORM_OFFSET_ALIGN = 256;

/** Shared zero length payload, so allocating a slice a piece did not reach costs no garbage. */
const EMPTY_PIECE = new Uint8Array(0);

/** Round `value` up to a multiple of `align`, which must be a positive power of two. */
export function alignTo(value: number, align: number): number {
  if (!Number.isInteger(value) || value < 0) throw new Error(`alignTo: bad value ${value}`);
  if (!Number.isInteger(align) || align <= 0 || (align & (align - 1)) !== 0) {
    throw new Error(`alignTo: alignment must be a positive power of two, got ${align}`);
  }
  return (value + align - 1) & ~(align - 1);
}

// ------------------------------------------------------------------- structural GPU types

/** The slice of GPUBuffer this module needs. GPUBuffer satisfies it; so does a fake. */
export interface BufferLike {
  readonly size: number;
  destroy(): void;
}

export interface BufferDescriptorLike {
  label?: string;
  size: number;
  usage: number;
}

/** The slice of GPUDevice this module needs. Fakes implement it; GPUDevice satisfies it. */
export interface DeviceLike {
  createBuffer(descriptor: BufferDescriptorLike): BufferLike;
  queue: {
    writeBuffer(
      buffer: BufferLike,
      bufferOffset: number,
      data: ArrayBufferView | ArrayBuffer,
      dataOffset?: number,
      size?: number,
    ): void;
  };
  createBindGroup(descriptor: {
    label?: string;
    layout: unknown;
    entries: { binding: number; resource: { buffer: BufferLike; offset?: number; size?: number } }[];
  }): unknown;
}

// ------------------------------------------------------------------------- uniform arena

/**
 * Pure offset bookkeeping for a bump allocated arena. Separate from the GPU side so the
 * arithmetic is Node testable on its own, which the orchestrator check section does.
 *
 * Discipline: every allocation starts on a UNIFORM_OFFSET_ALIGN boundary, because dynamic and
 * static uniform binding offsets must respect minUniformBufferOffsetAlignment, and one arena
 * serves every kernel's params block in a pass. reset() recycles the whole arena between
 * submitted passes; offsets are never reused inside one, so no write can land on a region a
 * queued dispatch still reads.
 */
export class ArenaPlan {
  readonly capacity: number;
  readonly align: number;
  private cursor = 0;
  private count = 0;

  constructor(capacity: number, align: number = UNIFORM_OFFSET_ALIGN) {
    if (!Number.isInteger(capacity) || capacity <= 0) {
      throw new Error(`ArenaPlan: capacity must be a positive integer, got ${capacity}`);
    }
    this.capacity = capacity;
    this.align = align;
    // Validate the alignment eagerly rather than on first alloc.
    alignTo(0, align);
  }

  get used(): number {
    return this.cursor;
  }

  get allocations(): number {
    return this.count;
  }

  /** Reserve `bytes` and return the aligned offset. Throws rather than wrapping on overflow. */
  alloc(bytes: number): number {
    if (!Number.isInteger(bytes) || bytes <= 0) {
      throw new Error(`ArenaPlan.alloc: bytes must be a positive integer, got ${bytes}`);
    }
    const offset = alignTo(this.cursor, this.align);
    if (offset + bytes > this.capacity) {
      throw new Error(
        `ArenaPlan.alloc: ${bytes} bytes at offset ${offset} overflows a ${this.capacity} byte arena`,
      );
    }
    this.cursor = offset + bytes;
    this.count += 1;
    return offset;
  }

  reset(): void {
    this.cursor = 0;
    this.count = 0;
  }
}

// ------------------------------------------------------------------------ buffer manager

/**
 * One contiguous byte range of a tensor that is carried in its own buffer.
 *
 * This is the byte shaped half of ../tableSplit.ts's `TableSlice`: the planner works in vocabulary
 * rows because that is what makes a gather addressable, and the buffer manager works in bytes
 * because that is what a piece of a download is. The two are the same partition.
 */
export interface WeightSliceRange {
  readonly index: number;
  readonly byteOffset: number;
  readonly byteLength: number;
}

/** The name slice `index` of tensor `name` is resident under. One function, so nothing can disagree. */
export function sliceName(name: string, index: number): string {
  return `${name}#${index}`;
}

export interface UniformSlice {
  buffer: BufferLike;
  offset: number;
  size: number;
}

export interface BindGroupEntrySpec {
  binding: number;
  buffer: BufferLike;
  offset?: number;
  size?: number;
}

export interface BufferManagerStats {
  weightBuffers: number;
  weightBytes: number;
  activationBuffers: number;
  activationBytes: number;
  bindGroupsCreated: number;
  bindGroupsReused: number;
}

/**
 * Owns every buffer the engine allocates, so dispose() is one call and the residency story is one
 * number. Three families:
 *
 *  - Weights, one STORAGE buffer per tensor, uploaded packed. Nothing is dequantized on the CPU;
 *    dequant happens in the shader at the point of use, which is what keeps the model inside a
 *    16 GB M1 (ENGINE-PLAN section 3).
 *  - Activations, ping-pong pairs by named slot, so layer N's output buffer is layer N+1's input
 *    without a copy and without ever aliasing a buffer a queued dispatch still reads.
 *  - One uniform arena for every kernel's params block, bump allocated per pass.
 *
 * Bind groups are cached by layout and buffer identity, because creating one per dispatch per
 * token is avoidable garbage on a loop that runs hundreds of dispatches per token. Buffers get
 * stable numeric ids at creation; the cache key is the layout's key plus the id, offset and size
 * of every entry.
 */
export class BufferManager {
  private readonly device: DeviceLike;
  private readonly weights = new Map<string, BufferLike>();
  /** Tensors delivered whole in one call, which is what the double upload guard is about. */
  private readonly wholeUploads = new Set<string>();
  /**
   * Tensors that have received at least one byte. The double upload guard used to read "this name
   * has a buffer", which meant the same thing until `declareSplit` began allocating a split
   * tensor's slices before any of its bytes arrive. Now a slice buffer can exist and be empty, and
   * a piece that happens to cover one whole slice is an ordinary first write rather than a second
   * upload, so the guard asks what was written instead of what was allocated.
   */
  private readonly written = new Set<string>();
  private readonly pairs = new Map<string, { buffers: [BufferLike, BufferLike]; active: 0 | 1 }>();
  private readonly slots = new Map<string, BufferLike>();
  private readonly bufferIds = new Map<BufferLike, number>();
  private readonly bindGroups = new Map<string, unknown>();
  /** The numeric fast path of cachedBindGroup: recent groups per layout key, by entry signature. */
  private readonly recentBindGroups = new Map<string, { sig: number[]; group: unknown }[]>();
  private nextBufferId = 1;
  private arenaPlan: ArenaPlan | null = null;
  /** Host copy of the arena; stageUniform writes here and flushUniformArena pushes it once. */
  private arenaShadow: Uint8Array | null = null;
  /** Bytes of the shadow staged since the last flush, from offset 0. */
  private arenaDirty = 0;
  /** Arena resets so far; see arenaGeneration. */
  private arenaResets = 0;
  private arenaBuffer: BufferLike | null = null;
  private stats: BufferManagerStats = {
    weightBuffers: 0,
    weightBytes: 0,
    activationBuffers: 0,
    activationBytes: 0,
    bindGroupsCreated: 0,
    bindGroupsReused: 0,
  };

  /**
   * `maxBufferSize` is the device's granted limit, when the caller knows it. WebGPU does not throw
   * on an over sized createBuffer: it returns an INVALID buffer, every writeBuffer into it fails
   * the same way, and the load reports success with the tensor never populated. That is how the
   * PLE table (1,174,405,120 bytes) silently produced garbage on every device with a 1 GiB limit,
   * every iPhone among them, while the M1's 4 GiB limit hid it. See ENGINE-PERF 28.7.
   */
  constructor(device: DeviceLike, limits?: { maxBufferSize?: number }) {
    this.device = device;
    this.maxBufferSize = limits?.maxBufferSize ?? 0;
  }

  /** Zero means the caller did not say, and the guard below stands down. */
  private readonly maxBufferSize: number;

  /**
   * Tensors carried as several buffers because no one buffer on this adapter is large enough.
   * Keyed by the tensor's checkpoint name; the buffers live in `weights` under `sliceName`.
   *
   * Empty on every adapter that grants a limit larger than the biggest tensor, which is every
   * desktop adapter this project has measured, and non empty on iOS, where `maxBufferSize` is
   * 1,073,741,824 and the PLE table is 1,174,405,120. See ../tableSplit.ts for why the split is by
   * vocabulary row and ENGINE-PERF 28.7 for what the missing split cost.
   */
  private readonly splits = new Map<string, readonly WeightSliceRange[]>();

  /**
   * Declare that one tensor arrives as several buffers, before any of its bytes do.
   *
   * The ranges partition the tensor and are given in offset order; `uploadWeight` then routes each
   * incoming piece into whichever slices it covers, rebasing the offset, and splitting a piece that
   * straddles a boundary. The loader is unchanged by this: it keeps fetching and delivering pieces
   * of the whole tensor and never learns that the destination is more than one buffer.
   */
  declareSplit(name: string, ranges: readonly WeightSliceRange[]): void {
    if (this.weights.has(name) || this.splits.has(name)) {
      throw new Error(`BufferManager.declareSplit: ${name} is already declared or resident`);
    }
    if (ranges.length === 0) throw new Error(`BufferManager.declareSplit: ${name} needs at least one range`);
    let expect = 0;
    for (const r of ranges) {
      if (r.byteOffset !== expect) {
        throw new Error(
          `BufferManager.declareSplit: ${name} slice ${r.index} starts at ${r.byteOffset}, not at `
          + `${expect}; the ranges have to partition the tensor in offset order`,
        );
      }
      if (r.byteOffset % 4 !== 0) {
        throw new Error(`BufferManager.declareSplit: ${name} slice ${r.index} starts at ${r.byteOffset}, which is not four byte aligned`);
      }
      if (this.maxBufferSize > 0 && r.byteLength > this.maxBufferSize) {
        throw new Error(
          `BufferManager.declareSplit: ${name} slice ${r.index} is ${r.byteLength} bytes against a `
          + `maxBufferSize of ${this.maxBufferSize}, so the split does not go far enough`,
        );
      }
      expect += r.byteLength;
    }
    this.splits.set(name, ranges.map((r) => ({ ...r })));
    // Allocate every slice now, at its full length, rather than on the piece that first reaches it.
    // Two reasons, and the second is the one a test found. A tensor whose last pieces all land in
    // earlier slices would otherwise leave a tail slice with no buffer at all, which a kernel would
    // meet as a missing tensor at the first dispatch. And allocating lazily meant the router had to
    // visit slices a piece did not touch, which made a slice that one piece happened to cover whole
    // look like the double upload the guard below refuses. Allocating here makes the router pure:
    // it writes intersections and never creates anything.
    for (const r of ranges) {
      this.uploadOneBuffer(sliceName(name, r.index), EMPTY_PIECE, { offset: 0, totalBytes: r.byteLength });
    }
  }

  /** True when this tensor is carried as several buffers on this adapter. */
  isSplit(name: string): boolean {
    return this.splits.has(name);
  }

  /** The ranges `declareSplit` recorded, for a residency report or a check. */
  splitOf(name: string): readonly WeightSliceRange[] | null {
    return this.splits.get(name) ?? null;
  }

  private idOf(buffer: BufferLike): number {
    let id = this.bufferIds.get(buffer);
    if (id === undefined) {
      id = this.nextBufferId;
      this.nextBufferId += 1;
      this.bufferIds.set(buffer, id);
    }
    return id;
  }

  // Weights.

  /**
   * Create the buffer for one tensor and upload its packed bytes. Buffer sizes and writeBuffer
   * extents must both be multiples of four bytes, so a tensor whose packed length is not gets
   * copied once into a zero padded staging array; no kernel reads the pad because every kernel
   * indexes by its own row geometry.
   *
   * A TENSOR MAY ARRIVE IN PIECES. `options.offset` is where these bytes belong inside the tensor
   * and `options.totalBytes` is how long the whole tensor is, so the loader can fetch a 1.17 GB
   * table in 64 MiB requests and resume at the request rather than at the table
   * (safetensors.ts `splitRangeIntoPieces`). The buffer is created at the full length by whichever
   * piece arrives first, which means pieces may arrive in any order; the second piece of a tensor
   * is a write into the existing buffer rather than the double upload the guard below refuses.
   *
   * Two properties make the offset write safe rather than merely convenient. `writeBuffer` needs a
   * four byte aligned destination offset, and every offset except a final short piece's is a whole
   * multiple of the piece size, which the loader keeps a multiple of four. And the zero pad a short
   * tail needs can only run past the end of the tensor, never over the next piece, because the only
   * piece that is not a multiple of four bytes long is the last one.
   */
  uploadWeight(
    name: string,
    bytes: Uint8Array,
    options: { offset?: number; totalBytes?: number } = {},
  ): BufferLike {
    const ranges = this.splits.get(name);
    if (ranges) return this.uploadSplitWeight(name, ranges, bytes, options);
    return this.uploadOneBuffer(name, bytes, options);
  }

  /**
   * Route one piece of a split tensor into the slices it covers.
   *
   * A piece is a byte range of the whole tensor and a slice is another, so the intersection is the
   * arithmetic and there is nothing else to it. A piece that lands inside one slice is one write;
   * a piece that straddles a boundary is two, each rebased to its own buffer's offset. Both are
   * four byte aligned, because `declareSplit` refuses a range that does not start on a word and the
   * loader's piece offsets are multiples of the piece size (safetensors.ts `splitRangeIntoPieces`).
   *
   * Returns the buffer of the first slice this piece touched, so the signature is the one every
   * caller already has. No caller uses the return value for a split tensor: the kernels reach the
   * slices by `sliceName`, which is what the sliced gather binds one of per dispatch.
   */
  private uploadSplitWeight(
    name: string,
    ranges: readonly WeightSliceRange[],
    bytes: Uint8Array,
    options: { offset?: number; totalBytes?: number },
  ): BufferLike {
    const pieceStart = Math.max(0, Math.trunc(options.offset ?? 0));
    const pieceEnd = pieceStart + bytes.byteLength;
    let first: BufferLike | null = null;
    for (const range of ranges) {
      const rangeEnd = range.byteOffset + range.byteLength;
      const from = Math.max(pieceStart, range.byteOffset);
      const to = Math.min(pieceEnd, rangeEnd);
      // Slices this piece does not reach are already allocated by declareSplit and have nothing to
      // receive, so there is nothing to do for them here.
      if (to <= from) continue;
      const buffer = this.uploadOneBuffer(
        sliceName(name, range.index),
        bytes.subarray(from - pieceStart, to - pieceStart),
        { offset: from - range.byteOffset, totalBytes: range.byteLength },
      );
      if (!first) first = buffer;
    }
    if (!first) {
      // A piece outside every declared range is a partition that does not cover the tensor, which
      // declareSplit's own ordering check should already have made impossible.
      throw new Error(
        `BufferManager.uploadWeight: ${name} piece [${pieceStart}, ${pieceEnd}) falls outside every `
        + 'declared slice',
      );
    }
    return first;
  }

  private uploadOneBuffer(
    name: string,
    bytes: Uint8Array,
    options: { offset?: number; totalBytes?: number } = {},
  ): BufferLike {
    const offset = Math.max(0, Math.trunc(options.offset ?? 0));
    const totalBytes = Math.max(offset + bytes.byteLength, Math.trunc(options.totalBytes ?? 0));
    if (offset % 4 !== 0) {
      throw new Error(`BufferManager.uploadWeight: ${name} piece at offset ${offset} is not four byte aligned`);
    }
    const size = alignTo(Math.max(4, totalBytes), 4);
    let buffer = this.weights.get(name);
    // What a double upload means once a tensor can arrive in pieces: a call that covers the whole
    // tensor, against a tensor that already has a buffer. Pieces are allowed in any order and each
    // covers part of one, so the guard cannot be "this name has a buffer" any more without refusing
    // the second piece of every split tensor.
    const coversWhole = offset === 0 && bytes.byteLength > 0 && bytes.byteLength >= totalBytes;
    if ((coversWhole && this.written.has(name)) || this.wholeUploads.has(name)) {
      throw new Error(`BufferManager.uploadWeight: ${name} was already uploaded`);
    }
    if (buffer && buffer.size < offset + bytes.byteLength) {
      throw new Error(
        `BufferManager.uploadWeight: ${name} holds ${buffer.size} bytes and a piece ends at `
        + `${offset + bytes.byteLength}`,
      );
    }
    if (!buffer && this.maxBufferSize > 0 && size > this.maxBufferSize) {
      throw new Error(
        `BufferManager.uploadWeight: ${name} needs ${size} bytes in one buffer and this device's `
        + `maxBufferSize is ${this.maxBufferSize}. WebGPU would return an invalid buffer here and `
        + 'every write into it would fail, so the tensor would be empty and the model would emit '
        + 'garbage. A tensor this large has to be declared with declareSplit before its bytes '
        + 'arrive, which engine.ts does for the PLE table from tableSplit.ts\'s plan. Reaching '
        + 'this message means a different tensor outgrew the adapter and nothing plans its split.',
      );
    }
    if (!buffer) {
      buffer = this.device.createBuffer({
        label: `weight:${name}`,
        size,
        usage: USAGE.STORAGE | USAGE.COPY_DST,
      });
      this.weights.set(name, buffer);
      this.stats.weightBuffers += 1;
      this.stats.weightBytes += size;
    }
    if (bytes.byteLength > 0) {
      let payload = bytes;
      if (bytes.byteLength % 4 !== 0) {
        payload = new Uint8Array(alignTo(bytes.byteLength, 4));
        payload.set(bytes);
      }
      this.device.queue.writeBuffer(buffer, offset, payload);
      this.written.add(name);
    }
    if (coversWhole) this.wholeUploads.add(name);
    return buffer;
  }

  /**
   * Swap a resident weight for another buffer under the same name, the old one already
   * destroyed or owned elsewhere. The 2-bit repack (kernels/repack.ts) is the one caller: the
   * uploaded row layout buffer is consumed and the interleaved one takes its name.
   */
  replaceWeight(name: string, buffer: BufferLike): void {
    if (!this.weights.has(name)) throw new Error(`BufferManager.replaceWeight: ${name} is not resident`);
    this.weights.set(name, buffer);
  }

  weight(name: string): BufferLike {
    const buffer = this.weights.get(name);
    if (!buffer) throw new Error(`BufferManager.weight: no tensor named ${name}`);
    return buffer;
  }

  hasWeight(name: string): boolean {
    return this.weights.has(name);
  }

  // Activations.

  /**
   * The ping-pong pair for a named activation slot, created on first use at `bytes` and reused
   * after. Asking for a bigger size than the pair was created with is a plan bug, not a resize.
   */
  private pair(key: string, bytes: number): { buffers: [BufferLike, BufferLike]; active: 0 | 1 } {
    let entry = this.pairs.get(key);
    if (!entry) {
      const size = alignTo(Math.max(4, bytes), 4);
      const make = (half: 0 | 1): BufferLike => this.device.createBuffer({
        label: `activation:${key}:${half}`,
        size,
        usage: USAGE.STORAGE | USAGE.COPY_SRC | USAGE.COPY_DST,
      });
      entry = { buffers: [make(0), make(1)], active: 0 };
      this.pairs.set(key, entry);
      this.stats.activationBuffers += 2;
      this.stats.activationBytes += 2 * size;
    }
    if (entry.buffers[0].size < bytes) {
      throw new Error(
        `BufferManager: slot ${key} holds ${entry.buffers[0].size} bytes, asked for ${bytes}`,
      );
    }
    return entry;
  }

  /**
   * A single storage buffer under a name, created on first use and reused after. The ping-pong
   * pair above is for the one slot that is genuinely read and written by the same dispatch, which
   * is the residual stream; every other activation in the forward has a distinct source and
   * destination, and giving each of those a pair would double the activation residency to protect
   * against an aliasing that the schedule does not contain.
   *
   * Asking for a bigger size than the slot was created with is a plan bug rather than a resize, on
   * the same argument `pair` makes: the executor sizes every slot from the architecture's maxima
   * once, so a later larger ask means the geometry moved under it.
   */
  storage(key: string, bytes: number): BufferLike {
    let buffer = this.slots.get(key);
    if (!buffer) {
      const size = alignTo(Math.max(4, bytes), 4);
      buffer = this.device.createBuffer({
        label: `slot:${key}`,
        size,
        usage: USAGE.STORAGE | USAGE.COPY_SRC | USAGE.COPY_DST,
      });
      this.slots.set(key, buffer);
      this.stats.activationBuffers += 1;
      this.stats.activationBytes += size;
      return buffer;
    }
    if (buffer.size < bytes) {
      throw new Error(`BufferManager: slot ${key} holds ${buffer.size} bytes, asked for ${bytes}`);
    }
    return buffer;
  }

  hasStorage(key: string): boolean {
    return this.slots.has(key);
  }

  /** The buffer a kernel writing slot `key` writes into this step. */
  writeSide(key: string, bytes: number): BufferLike {
    const entry = this.pair(key, bytes);
    return entry.buffers[entry.active];
  }

  /** The buffer a kernel reading slot `key` reads, which is the other half of the pair. */
  readSide(key: string, bytes: number): BufferLike {
    const entry = this.pair(key, bytes);
    return entry.buffers[entry.active === 0 ? 1 : 0];
  }

  /** Make the freshly written half the readable one. Called after the dispatch that wrote it. */
  flip(key: string): void {
    const entry = this.pairs.get(key);
    if (!entry) throw new Error(`BufferManager.flip: no slot named ${key}`);
    entry.active = entry.active === 0 ? 1 : 0;
  }

  // Uniform arena.

  /** Allocate the arena once. Capacity is a whole pass worth of params blocks. */
  createUniformArena(capacity: number): void {
    if (this.arenaBuffer) throw new Error('BufferManager: the uniform arena already exists');
    this.arenaPlan = new ArenaPlan(capacity);
    this.arenaShadow = new Uint8Array(capacity);
    this.arenaDirty = 0;
    this.arenaBuffer = this.device.createBuffer({
      label: 'uniform-arena',
      size: capacity,
      usage: USAGE.UNIFORM | USAGE.COPY_DST,
    });
  }

  /**
   * Write one params block into the arena and return its slice for binding. The block must be a
   * whole number of four byte words, which every params struct is; writeBuffer is called without
   * the trailing extent arguments because those count elements, not bytes, for typed array data.
   */
  stageUniform(data: ArrayBufferView): UniformSlice {
    if (!this.arenaPlan || !this.arenaBuffer || !this.arenaShadow) {
      throw new Error('BufferManager.stageUniform: createUniformArena was never called');
    }
    const size = data.byteLength;
    if (size % 4 !== 0) {
      throw new Error(`BufferManager.stageUniform: ${size} bytes is not a whole number of words`);
    }
    const offset = this.arenaPlan.alloc(size);
    // Into the host shadow only. One writeBuffer per pass from flushUniformArena, not one per
    // dispatch: the round 4 host phase timers put 626 writeBuffer calls a token at 11 ms of the
    // 18 the host spent per token (docs/ENGINE-PERF.md section 15), all of it IPC.
    this.arenaShadow.set(new Uint8Array(data.buffer, data.byteOffset, size), offset);
    this.arenaDirty = Math.max(this.arenaDirty, offset + size);
    return { buffer: this.arenaBuffer, offset, size };
  }

  /**
   * Overwrite one params block at an offset a previous stageUniform handed out, for a pass that
   * reuses the arena layout of an earlier one and only refreshes the blocks that changed (the
   * runtime's decode step cache). The block must fit where it was.
   */
  restageUniform(offset: number, data: ArrayBufferView): UniformSlice {
    if (!this.arenaPlan || !this.arenaBuffer || !this.arenaShadow) {
      throw new Error('BufferManager.restageUniform: createUniformArena was never called');
    }
    const size = data.byteLength;
    if (offset < 0 || offset + size > this.arenaShadow.byteLength) {
      throw new Error(`BufferManager.restageUniform: ${size} bytes at ${offset} is outside the arena`);
    }
    this.arenaShadow.set(new Uint8Array(data.buffer, data.byteOffset, size), offset);
    this.arenaDirty = Math.max(this.arenaDirty, offset + size);
    return { buffer: this.arenaBuffer, offset, size };
  }

  /**
   * Counts the arena resets, so a caller that recorded offsets into one layout of the arena can
   * tell when another pass has reused it and its recorded blocks are gone.
   */
  get arenaGeneration(): number {
    return this.arenaResets;
  }

  /** Which half of a ping pong pair is currently the write side, 0 or 1; 0 for a pair not yet made. */
  pairParity(key: string): number {
    return this.pairs.get(key)?.active ?? 0;
  }

  /**
   * Push every params block staged since the last reset to the GPU in one write. Call before the
   * submit that binds them; the queue orders the write ahead of that command buffer.
   */
  flushUniformArena(): void {
    if (!this.arenaBuffer || !this.arenaShadow || this.arenaDirty === 0) return;
    this.device.queue.writeBuffer(this.arenaBuffer, 0, this.arenaShadow, 0, this.arenaDirty);
    this.arenaDirty = 0;
  }

  /**
   * Drop staged params that will never be submitted (a pipeline warm up binds every step to build
   * its pipelines and encodes nothing), then recycle the arena. The ordinary path is flush, submit,
   * reset; this is the one caller that may skip the flush and has to say so.
   */
  discardStagedUniforms(): void {
    // Nothing staged means nothing to discard, and the arena's recorded layout survives.
    if (this.arenaDirty === 0 && (this.arenaPlan?.used ?? 0) === 0) return;
    this.arenaDirty = 0;
    this.arenaResets += 1;
    this.arenaPlan?.reset();
  }

  /** Recycle the arena between submitted passes. Never inside one. */
  resetUniformArena(): void {
    if (this.arenaDirty !== 0) {
      throw new Error('BufferManager.resetUniformArena: staged params were never flushed to the GPU');
    }
    this.arenaResets += 1;
    this.arenaPlan?.reset();
  }

  // Bind groups.

  /**
   * A bind group for `layout` over `entries`, cached by layout key and entry identity. The caller
   * names the layout with a stable `layoutKey` because layout objects themselves are opaque.
   */
  cachedBindGroup(layoutKey: string, layout: unknown, entries: readonly BindGroupEntrySpec[]): unknown {
    // The numeric fast path first: the last few groups built under this layout key, compared by
    // buffer id, offset and size without building a string. A decode token binds 626 groups and
    // the string key alone was 5 ms of host time per token (docs/ENGINE-PERF.md section 15).
    const recent = this.recentBindGroups.get(layoutKey);
    if (recent) {
      outer: for (const candidate of recent) {
        if (candidate.sig.length !== entries.length * 3) continue;
        for (let i = 0; i < entries.length; i += 1) {
          const e = entries[i]!;
          if (candidate.sig[i * 3] !== this.idOf(e.buffer)
            || candidate.sig[i * 3 + 1] !== (e.offset ?? 0)
            || candidate.sig[i * 3 + 2] !== (e.size ?? -1)) continue outer;
        }
        this.stats.bindGroupsReused += 1;
        return candidate.group;
      }
    }
    const key = `${layoutKey}|${entries
      .map((e) => `${e.binding}:${this.idOf(e.buffer)}:${e.offset ?? 0}:${e.size ?? -1}`)
      .join(',')}`;
    const cached = this.bindGroups.get(key);
    if (cached !== undefined) {
      this.stats.bindGroupsReused += 1;
      this.remember(layoutKey, entries, cached);
      return cached;
    }
    const group = this.device.createBindGroup({
      label: layoutKey,
      layout,
      entries: entries.map((e) => {
        const resource: { buffer: BufferLike; offset?: number; size?: number } = { buffer: e.buffer };
        if (e.offset !== undefined) resource.offset = e.offset;
        if (e.size !== undefined) resource.size = e.size;
        return { binding: e.binding, resource };
      }),
    });
    this.bindGroups.set(key, group);
    this.stats.bindGroupsCreated += 1;
    this.remember(layoutKey, entries, group);
    return group;
  }

  /** Keep a group in the numeric fast path under its layout key, four per key, oldest out. */
  private remember(layoutKey: string, entries: readonly BindGroupEntrySpec[], group: unknown): void {
    const sig = new Array<number>(entries.length * 3);
    for (let i = 0; i < entries.length; i += 1) {
      const e = entries[i]!;
      sig[i * 3] = this.idOf(e.buffer);
      sig[i * 3 + 1] = e.offset ?? 0;
      sig[i * 3 + 2] = e.size ?? -1;
    }
    let recent = this.recentBindGroups.get(layoutKey);
    if (!recent) {
      recent = [];
      this.recentBindGroups.set(layoutKey, recent);
    }
    recent.push({ sig, group });
    if (recent.length > 4) recent.shift();
  }

  snapshotStats(): BufferManagerStats {
    return { ...this.stats };
  }

  /** Destroy everything this manager created. The engine's dispose() calls this once. */
  dispose(): void {
    for (const buffer of this.weights.values()) buffer.destroy();
    this.weights.clear();
    this.wholeUploads.clear();
    this.written.clear();
    this.splits.clear();
    for (const entry of this.pairs.values()) {
      entry.buffers[0].destroy();
      entry.buffers[1].destroy();
    }
    this.pairs.clear();
    for (const buffer of this.slots.values()) buffer.destroy();
    this.slots.clear();
    this.arenaBuffer?.destroy();
    this.arenaBuffer = null;
    this.arenaPlan = null;
    this.bindGroups.clear();
    this.bufferIds.clear();
  }
}
