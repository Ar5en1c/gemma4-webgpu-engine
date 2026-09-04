// Written from the safetensors container format, the model.safetensors header of
// google/gemma-4-E2B-it-qat-mobile-transformers read directly over HTTPS, and docs/ENGINE-PLAN.md section 3.
// SPDX-License-Identifier: Apache-2.0
//
// The container is 8 bytes of little endian header length N, then N bytes of JSON mapping tensor
// name to dtype, shape and data_offsets, then the payload. Offsets in the JSON are relative to byte
// 8 + N rather than to the start of the file, and the end offset is one past the last byte. That is
// a property of the format, so this file reads it and nothing else.
//
// Nothing here touches a GPU and nothing here dequantizes. It turns bytes into a directory, and it
// turns a directory into a list of HTTP ranges to ask for. Dequantization lives in quant.ts.
//
// One thing this module decides rather than merely reports: the lm_head tie. The checkpoint stores
// the output head twice, once as lm_head and once as embed_tokens, byte for byte. The allow list
// fetches one copy and resolves the other name onto it, guarded by a pinned sample digest. See the
// tied tensors section below for the receipt and the fallback.

/** The safetensors dtype strings. Only the ones this checkpoint uses are exercised, but the table is complete. */
export type SafetensorsDtype =
  | 'BOOL'
  | 'U8'
  | 'I8'
  | 'F8_E4M3'
  | 'F8_E5M2'
  | 'U16'
  | 'I16'
  | 'F16'
  | 'BF16'
  | 'U32'
  | 'I32'
  | 'F32'
  | 'U64'
  | 'I64'
  | 'F64';

/** Bytes per element for every safetensors dtype. */
export const DTYPE_BYTES: Readonly<Record<SafetensorsDtype, number>> = Object.freeze({
  BOOL: 1,
  U8: 1,
  I8: 1,
  F8_E4M3: 1,
  F8_E5M2: 1,
  U16: 2,
  I16: 2,
  F16: 2,
  BF16: 2,
  U32: 4,
  I32: 4,
  F32: 4,
  U64: 8,
  I64: 8,
  F64: 8,
});

/** The fixed size of the little endian header length that opens the file. */
export const HEADER_PREFIX_BYTES = 8;

/**
 * Refuse a header length that could only come from a corrupt or hostile file. The mobile
 * checkpoint's header is 375,392 bytes and the base checkpoint's is 263,952, so 64 MiB is several
 * hundred times more slack than any real Gemma 4 export needs.
 */
export const MAX_HEADER_BYTES = 64 * 1024 * 1024;

export type SafetensorsErrorCode =
  | 'short-prefix'
  | 'bad-header-length'
  | 'truncated-header'
  | 'bad-json'
  | 'bad-entry'
  | 'bad-dtype'
  | 'bad-shape'
  | 'bad-offsets'
  | 'size-mismatch'
  | 'overlap'
  | 'duplicate-tensor'
  | 'short-range';

/** Every rejection this module raises carries a code, so callers can branch without matching prose. */
export class SafetensorsError extends Error {
  readonly code: SafetensorsErrorCode;

  constructor(code: SafetensorsErrorCode, message: string) {
    super(message);
    this.name = 'SafetensorsError';
    this.code = code;
  }
}

/** One tensor in the directory. Payload offsets and absolute file offsets are both kept, because the header speaks the first language and HTTP Range speaks the second. */
export interface TensorEntry {
  readonly name: string;
  readonly dtype: SafetensorsDtype;
  readonly shape: readonly number[];
  /** Offset of the first byte, relative to the start of the payload. */
  readonly dataStart: number;
  /** One past the last byte, relative to the start of the payload. */
  readonly dataEnd: number;
  /** Offset of the first byte in the file. */
  readonly fileStart: number;
  /** One past the last byte in the file. */
  readonly fileEnd: number;
  readonly byteLength: number;
  readonly elementCount: number;
}

/** The parsed header: a name to tensor map plus the geometry needed to fetch payload bytes. */
export interface SafetensorsDirectory {
  readonly headerLength: number;
  /** Byte offset of the payload, which is HEADER_PREFIX_BYTES + headerLength. */
  readonly payloadStart: number;
  /** Length of the payload implied by the largest end offset. */
  readonly payloadLength: number;
  /** Total file length implied by the header. The mobile checkpoint reports 2,458,111,846. */
  readonly fileLength: number;
  readonly metadata: Readonly<Record<string, string>> | null;
  /** Every tensor, sorted by payload offset. */
  readonly tensors: readonly TensorEntry[];
  readonly byName: ReadonlyMap<string, TensorEntry>;
}

function isSafeCount(value: unknown): value is number {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0;
}

/**
 * Read the little endian u64 header length from the first eight bytes of the file.
 * Two HTTP requests give the whole manifest: this one, then the header JSON it sizes.
 */
export function readHeaderLength(prefix: Uint8Array): number {
  if (prefix.length < HEADER_PREFIX_BYTES) {
    throw new SafetensorsError(
      'short-prefix',
      `need ${HEADER_PREFIX_BYTES} bytes to read the header length, got ${prefix.length}`,
    );
  }
  const view = new DataView(prefix.buffer, prefix.byteOffset, HEADER_PREFIX_BYTES);
  const raw = view.getBigUint64(0, true);
  if (raw > BigInt(MAX_HEADER_BYTES)) {
    throw new SafetensorsError('bad-header-length', `header length ${raw} exceeds the ${MAX_HEADER_BYTES} byte ceiling`);
  }
  const length = Number(raw);
  if (length <= 0) {
    throw new SafetensorsError('bad-header-length', `header length ${length} is not a positive integer`);
  }
  return length;
}

export interface ParseHeaderOptions {
  /**
   * Throw when two tensors claim the same payload byte. Overlaps are corrupt rather than merely
   * unusual, so this defaults on. Gaps are reported by coverageGaps instead of thrown, because a
   * padded export would still be readable.
   */
  readonly rejectOverlaps?: boolean;
}

/**
 * Parse a header out of the opening bytes of the file. `bytes` must hold at least the eight byte
 * length plus the JSON it names; anything shorter is rejected as a truncated header rather than
 * guessed at.
 */
export function parseHeader(bytes: Uint8Array, options: ParseHeaderOptions = {}): SafetensorsDirectory {
  const rejectOverlaps = options.rejectOverlaps !== false;
  const headerLength = readHeaderLength(bytes);
  const needed = HEADER_PREFIX_BYTES + headerLength;
  if (bytes.length < needed) {
    throw new SafetensorsError(
      'truncated-header',
      `header claims ${headerLength} JSON bytes so ${needed} are needed, got ${bytes.length}`,
    );
  }

  const json = new TextDecoder('utf-8', { fatal: false }).decode(
    bytes.subarray(HEADER_PREFIX_BYTES, needed),
  );
  let parsed: unknown;
  try {
    parsed = JSON.parse(json);
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    throw new SafetensorsError('bad-json', `header JSON did not parse: ${detail}`);
  }
  if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new SafetensorsError('bad-json', 'header JSON is not an object');
  }

  const record = parsed as Record<string, unknown>;
  let metadata: Record<string, string> | null = null;
  const raw = record.__metadata__;
  if (raw !== undefined) {
    if (raw === null || typeof raw !== 'object' || Array.isArray(raw)) {
      throw new SafetensorsError('bad-entry', '__metadata__ is present but is not an object');
    }
    metadata = {};
    for (const [key, value] of Object.entries(raw as Record<string, unknown>)) {
      metadata[key] = typeof value === 'string' ? value : String(value);
    }
  }

  const payloadStart = needed;
  const tensors: TensorEntry[] = [];
  for (const [name, value] of Object.entries(record)) {
    if (name === '__metadata__') continue;
    if (value === null || typeof value !== 'object' || Array.isArray(value)) {
      throw new SafetensorsError('bad-entry', `tensor ${name} is not an object`);
    }
    const entry = value as Record<string, unknown>;

    const dtype = entry.dtype;
    if (typeof dtype !== 'string' || !(dtype in DTYPE_BYTES)) {
      throw new SafetensorsError('bad-dtype', `tensor ${name} has dtype ${String(dtype)}`);
    }
    const typedDtype = dtype as SafetensorsDtype;

    const shapeRaw = entry.shape;
    if (!Array.isArray(shapeRaw) || !shapeRaw.every(isSafeCount)) {
      throw new SafetensorsError('bad-shape', `tensor ${name} has a shape that is not a list of counts`);
    }
    const shape = shapeRaw as number[];
    let elementCount = 1;
    for (const dim of shape) elementCount *= dim;
    if (!Number.isSafeInteger(elementCount)) {
      throw new SafetensorsError('bad-shape', `tensor ${name} has an element count that overflows`);
    }

    const offsets = entry.data_offsets;
    if (!Array.isArray(offsets) || offsets.length !== 2 || !offsets.every(isSafeCount)) {
      throw new SafetensorsError('bad-offsets', `tensor ${name} has data_offsets that are not a pair of counts`);
    }
    const dataStart = offsets[0] as number;
    const dataEnd = offsets[1] as number;
    if (dataEnd < dataStart) {
      throw new SafetensorsError('bad-offsets', `tensor ${name} ends at ${dataEnd} before it starts at ${dataStart}`);
    }
    const byteLength = dataEnd - dataStart;
    const expected = elementCount * DTYPE_BYTES[typedDtype];
    if (byteLength !== expected) {
      throw new SafetensorsError(
        'size-mismatch',
        `tensor ${name} spans ${byteLength} bytes but ${typedDtype} ${JSON.stringify(shape)} needs ${expected}`,
      );
    }

    tensors.push({
      name,
      dtype: typedDtype,
      shape,
      dataStart,
      dataEnd,
      fileStart: payloadStart + dataStart,
      fileEnd: payloadStart + dataEnd,
      byteLength,
      elementCount,
    });
  }

  tensors.sort((a, b) => (a.dataStart - b.dataStart) || (a.dataEnd - b.dataEnd) || a.name.localeCompare(b.name));

  if (rejectOverlaps) {
    for (let i = 1; i < tensors.length; i += 1) {
      const previous = tensors[i - 1];
      const current = tensors[i];
      if (current.byteLength > 0 && previous.byteLength > 0 && current.dataStart < previous.dataEnd) {
        throw new SafetensorsError(
          'overlap',
          `tensor ${current.name} starts at ${current.dataStart} inside ${previous.name} which ends at ${previous.dataEnd}`,
        );
      }
    }
  }

  let payloadLength = 0;
  for (const tensor of tensors) payloadLength = Math.max(payloadLength, tensor.dataEnd);

  const byName = new Map<string, TensorEntry>();
  for (const tensor of tensors) byName.set(tensor.name, tensor);

  return {
    headerLength,
    payloadStart,
    payloadLength,
    fileLength: payloadStart + payloadLength,
    metadata,
    tensors,
    byName,
  };
}

/** A run of payload bytes no tensor claims. The Gemma 4 exports have none, and this is how we check that rather than assume it. */
export interface CoverageGap {
  readonly start: number;
  readonly end: number;
}

/** Payload spans no tensor claims, in payload relative coordinates. */
export function coverageGaps(directory: SafetensorsDirectory): CoverageGap[] {
  const gaps: CoverageGap[] = [];
  let cursor = 0;
  for (const tensor of directory.tensors) {
    if (tensor.byteLength === 0) continue;
    if (tensor.dataStart > cursor) gaps.push({ start: cursor, end: tensor.dataStart });
    cursor = Math.max(cursor, tensor.dataEnd);
  }
  return gaps;
}

// The ranged fetch plan.
//
// Section 3 of the plan asks for ranged reads over exactly the tensors we want. Issuing 1,601
// separate requests for the text tensors would be slower than the transfer, so neighbouring spans
// are coalesced into one range when the bytes between them are cheaper to download than a second
// round trip. maxGapBytes is the price we are willing to pay per merge and maxRangeBytes keeps any
// single response inside a size a browser will buffer comfortably.

export interface PlannedTensor {
  readonly name: string;
  readonly entry: TensorEntry;
  /** Offset of this tensor inside the range's response body. */
  readonly offsetInRange: number;
  readonly byteLength: number;
}

export interface PlannedRange {
  readonly index: number;
  /** Absolute file offset of the first byte to request. */
  readonly start: number;
  /** One past the last byte to request. */
  readonly end: number;
  readonly byteLength: number;
  readonly tensors: readonly PlannedTensor[];
}

export interface RangePlan {
  readonly ranges: readonly PlannedRange[];
  /** Bytes that belong to a requested tensor. */
  readonly tensorBytes: number;
  /** Bytes actually asked for, including anything swallowed by a coalesced gap. */
  readonly fetchedBytes: number;
  /** fetchedBytes minus tensorBytes. The price of the merges. */
  readonly wastedBytes: number;
  readonly tensorCount: number;
}

export interface RangePlanOptions {
  /** Merge two spans when at most this many unwanted bytes sit between them. */
  readonly maxGapBytes?: number;
  /** Never grow a single range past this. A tensor larger than this still gets its own range. */
  readonly maxRangeBytes?: number;
}

/**
 * 64 KiB of slack per merge and 64 MiB per request, both measured against the real header rather
 * than picked. Over the 1,439 text tensors of the mobile checkpoint with the lm_head tie taken,
 * cap held at 64 MiB:
 *
 *   gap 0        90 ranges, 2,006,857,726 bytes, nothing wasted
 *   gap 4 KiB    50 ranges, 2,006,907,198 bytes, 49,472 wasted
 *   gap 64 KiB   50 ranges, 2,006,907,198 bytes, 49,472 wasted
 *   gap 1 MiB    16 ranges, 2,016,591,038 bytes, 9,733,312 wasted
 *
 * A megabyte of slack buys 34 fewer requests and pays 9.68 MB for them, most of it the dead
 * consumer k_proj and v_proj the allow list exists to avoid, so the default sits at 64 KiB. The cap
 * is about memory rather than request count: the PLE table alone is 1.174 GB
 * (EMBEDDING-QUANT-FINDING.md) and has to arrive in pieces whatever the gap budget is.
 *
 * The numbers before the tie, kept because two of them are load bearing elsewhere: 1,441 tensors,
 * 91 ranges and 2,108,569,598 bytes at gap 0, 51 ranges and 2,108,619,070 at the default, 17 ranges
 * and 2,118,302,910 at 1 MiB. 2,108,569,598 is exactly the reference lane's independent count of
 * the bytes the text stack needs before deduplication. 2,118,302,910 is, to the byte, the wire
 * total VENDOR.md observed from the incumbent, which is that same stack plus the 9,486,656 bytes of
 * dead consumer k_proj and v_proj a loose coalescer swallows. Subtracting the 101,711,872 byte tied
 * pair from 2,108,569,598 gives 2,006,857,726, which is 1.8690 GiB and is the "about 1.87 GB
 * resident" ENGINE-PLAN.md section 3 recorded as an open question. It is not open any more.
 */
export const DEFAULT_RANGE_PLAN_OPTIONS: Required<RangePlanOptions> = Object.freeze({
  maxGapBytes: 64 << 10,
  maxRangeBytes: 64 << 20,
});

/**
 * Group tensors into contiguous byte ranges. Input order does not matter; the result is sorted by
 * file offset, every requested tensor appears in exactly one range, and no range overlaps another.
 */
export function buildRangePlan(entries: readonly TensorEntry[], options: RangePlanOptions = {}): RangePlan {
  const maxGapBytes = Math.max(0, options.maxGapBytes ?? DEFAULT_RANGE_PLAN_OPTIONS.maxGapBytes);
  const maxRangeBytes = Math.max(1, options.maxRangeBytes ?? DEFAULT_RANGE_PLAN_OPTIONS.maxRangeBytes);

  const seen = new Set<string>();
  for (const entry of entries) {
    if (seen.has(entry.name)) {
      throw new SafetensorsError('duplicate-tensor', `tensor ${entry.name} appears twice in the plan input`);
    }
    seen.add(entry.name);
  }

  const sorted = [...entries].sort(
    (a, b) => (a.fileStart - b.fileStart) || (a.fileEnd - b.fileEnd) || a.name.localeCompare(b.name),
  );

  const ranges: PlannedRange[] = [];
  let current: { start: number; end: number; tensors: PlannedTensor[] } | null = null;
  let tensorBytes = 0;

  const close = (): void => {
    if (!current) return;
    ranges.push({
      index: ranges.length,
      start: current.start,
      end: current.end,
      byteLength: current.end - current.start,
      tensors: current.tensors,
    });
    current = null;
  };

  for (const entry of sorted) {
    tensorBytes += entry.byteLength;
    if (current) {
      const gap = entry.fileStart - current.end;
      const grown = Math.max(current.end, entry.fileEnd) - current.start;
      // A gap below zero means the caller handed us overlapping tensors, which is still safe to
      // merge: the range covers both and each keeps its own offset.
      if (gap <= maxGapBytes && grown <= maxRangeBytes) {
        current.tensors.push({
          name: entry.name,
          entry,
          offsetInRange: entry.fileStart - current.start,
          byteLength: entry.byteLength,
        });
        current.end = Math.max(current.end, entry.fileEnd);
        continue;
      }
      close();
    }
    current = {
      start: entry.fileStart,
      end: entry.fileEnd,
      tensors: [{ name: entry.name, entry, offsetInRange: 0, byteLength: entry.byteLength }],
    };
  }
  close();

  let fetchedBytes = 0;
  for (const range of ranges) fetchedBytes += range.byteLength;

  return {
    ranges,
    tensorBytes,
    fetchedBytes,
    wastedBytes: fetchedBytes - tensorBytes,
    tensorCount: sorted.length,
  };
}

// Sub-splitting an oversized range.
//
// `buildRangePlan` caps a range at `maxRangeBytes` except in one case it cannot: a single tensor
// bigger than the cap still gets its own range, because a tensor is the smallest thing the plan can
// name. Two of this checkpoint's ranges are exactly that, the 1,174,405,120 byte PLE table and the
// 100,663,296 byte embed table the head shares. The loader's resume unit is the range, so a
// connection dropped 1.1 GB into the PLE table costs the whole table.
//
// The fix is to make the REQUEST smaller than the range without making the range smaller than the
// tensor. A piece is one request over a byte span of one tensor, and it is shaped as a PlannedRange
// so that `rangeHeaderValue`, the fetch path and `sliceTensorFromRange` all keep working on it
// unchanged. What the piece adds is the byte offset of its span inside the tensor, which is what
// the sink needs to write it into the right part of a buffer it may already have created.
//
// Only a single tensor range is split, and only when its tensor starts at the range's first byte,
// which is what `buildRangePlan` produces for an oversized tensor. A coalesced range holding
// several tensors is left whole: cutting one at an arbitrary byte would put half a tensor in a
// piece, and the resume that buys is not worth a second addressing scheme. Every range in this
// checkpoint's plan is either under the cap or one of the two single tensor ones, so nothing is
// left unsplit that a dropped connection could make expensive.

/** One request over part of one range, and where its bytes belong inside the tensor. */
export interface RangePiece {
  /** The request. A whole range when `count` is 1, a byte span of one tensor when it is not. */
  readonly range: PlannedRange;
  /** This piece's index inside its run, from 0. */
  readonly index: number;
  /** Pieces the run has. 1 means the run is one request and nothing about it changed. */
  readonly count: number;
  /**
   * Byte offset of this piece inside the tensor it carries, which is what the sink is handed. Zero
   * for an unsplit run, where every tensor arrives whole and starts at its own beginning.
   */
  readonly offsetInTensor: number;
}

/**
 * Cut one planned range into requests of at most `pieceBytes`.
 *
 * Returns a single piece covering the whole range when it does not need splitting, so a caller
 * walks pieces and never asks whether it is in the split case.
 */
export function splitRangeIntoPieces(range: PlannedRange, pieceBytes: number): RangePiece[] {
  const limit = Math.max(1, Math.trunc(pieceBytes));
  const only = range.tensors.length === 1 ? range.tensors[0] : undefined;
  if (range.byteLength <= limit || !only || only.offsetInRange !== 0) {
    return [{ range, index: 0, count: 1, offsetInTensor: 0 }];
  }
  const count = Math.ceil(range.byteLength / limit);
  const pieces: RangePiece[] = [];
  for (let i = 0; i < count; i += 1) {
    const start = range.start + i * limit;
    const end = Math.min(range.end, start + limit);
    pieces.push({
      // The index is the RUN's index, deliberately. A piece is a request inside run n, not a run of
      // its own, so a receipt that counts runs still counts what the plan planned.
      range: {
        index: range.index,
        start,
        end,
        byteLength: end - start,
        tensors: [{ name: only.name, entry: only.entry, offsetInRange: 0, byteLength: end - start }],
      },
      index: i,
      count,
      offsetInTensor: i * limit,
    });
  }
  return pieces;
}

/** The value for an HTTP Range request header. The end is inclusive on the wire and exclusive here. */
export function rangeHeaderValue(range: PlannedRange): string {
  const last = range.byteLength > 0 ? range.end - 1 : range.start;
  return `bytes=${range.start}-${last}`;
}

/** Cut one tensor's bytes out of a range's response body, without copying. */
export function sliceTensorFromRange(range: PlannedRange, body: Uint8Array, name: string): Uint8Array {
  const planned = range.tensors.find((candidate) => candidate.name === name);
  if (!planned) {
    throw new SafetensorsError('bad-entry', `tensor ${name} is not in range ${range.index}`);
  }
  const end = planned.offsetInRange + planned.byteLength;
  if (body.length < end) {
    throw new SafetensorsError(
      'short-range',
      `range ${range.index} body is ${body.length} bytes but ${name} ends at ${end}`,
    );
  }
  return body.subarray(planned.offsetInRange, end);
}

/** What a plan did and did not cover. Used by the loader check rather than at runtime. */
export interface PlanCoverage {
  readonly missing: string[];
  readonly duplicated: string[];
  readonly overlappingRanges: number;
}

/** Confirm a plan names every wanted tensor exactly once and that its ranges do not overlap. */
export function planCoverage(plan: RangePlan, wanted: readonly TensorEntry[]): PlanCoverage {
  const counts = new Map<string, number>();
  for (const range of plan.ranges) {
    for (const planned of range.tensors) {
      counts.set(planned.name, (counts.get(planned.name) ?? 0) + 1);
    }
  }
  const missing: string[] = [];
  const duplicated: string[] = [];
  for (const entry of wanted) {
    const count = counts.get(entry.name) ?? 0;
    if (count === 0) missing.push(entry.name);
    if (count > 1) duplicated.push(entry.name);
  }
  for (const [name, count] of counts) {
    if (count > 1 && !duplicated.includes(name)) duplicated.push(name);
  }

  let overlappingRanges = 0;
  const ordered = [...plan.ranges].sort((a, b) => a.start - b.start);
  for (let i = 1; i < ordered.length; i += 1) {
    if (ordered[i].start < ordered[i - 1].end) overlappingRanges += 1;
  }

  return { missing, duplicated, overlappingRanges };
}

// Tied tensors.
//
// `config.json` says `tie_word_embeddings: false` and its `quantization_config` lists `lm_head` and
// `model.language_model.embed_tokens` as two separate 2-bit modules. The bytes say otherwise. On
// revision dd693ff40353f057ca5f07e945ad867f4afbf2ec of the mobile checkpoint:
//
//   lm_head.weight                                        U8  [262144, 384]  at [359273726, 459937022]
//   model.language_model.embed_tokens.embedding_quantized  U8  [262144, 384]  at [538580222, 639243518]
//
// are the same 100,663,296 bytes, and
//
//   lm_head.weight_scale                                  F32 [262144, 1]    at [8, 1048584]
//   model.language_model.embed_tokens.embedding_scale     F32 [262144, 1]    at [77711432, 78760008]
//
// are the same 1,048,576 bytes. Verified three times independently on 2026-09-01: the research
// lane hashed 20 evenly spaced 256 KB windows across the weight (20 of 20 identical) and both
// scale tensors in full; a second party re-fetched the live file and compared four 64 KB windows
// at intra-tensor offsets 0, 33554432, 67108864 and 100597760 (4 of 4 identical); and this module's
// own check hashed both tensors end to end out of the pinned snapshot. All three agree, and the
// scale digest 177b4a1ab87a451b1239839c7fb00a664f52530f84843cc7c4c1c4d63b3f0d56 is the same on
// every reading.
//
// So the allow list fetches the embed pair and never the lm_head pair, and the lm_head names
// resolve to the embed entries through lookupTensor. That is 101,711,872 bytes, a visible 100 MB,
// off a consent gated download.
//
// The risk this takes is narrow and it is guarded rather than assumed. If Google ever re-exports
// the checkpoint with a genuinely untied head, an engine that aliased the two would compute logits
// against the input embedding and produce fluent, wrong text: exactly the silent corruption class
// ENGINE-PLAN section 10 is about. The guard below is what stops that.

/** One tensor the checkpoint stores twice. The alias is never fetched; it resolves to the source. */
export interface TiedTensor {
  readonly alias: string;
  readonly source: string;
  readonly byteLength: number;
}

/** The ties this checkpoint carries. Read from the header and confirmed byte for byte, not inferred from config. */
export const TIED_TENSORS: readonly TiedTensor[] = Object.freeze([
  Object.freeze({
    alias: 'lm_head.weight',
    source: 'model.language_model.embed_tokens.embedding_quantized',
    byteLength: 100_663_296,
  }),
  Object.freeze({
    alias: 'lm_head.weight_scale',
    source: 'model.language_model.embed_tokens.embedding_scale',
    byteLength: 1_048_576,
  }),
]);

/** Bytes the tie keeps off the wire: the two lm_head tensors that duplicate the embed pair. */
export const TIED_PAIR_BYTES = 101_711_872;

const TIED_ALIAS_TO_SOURCE: ReadonlyMap<string, string> = new Map(
  TIED_TENSORS.map((tie) => [tie.alias, tie.source] as const),
);

/** True for a name the allow list deliberately never fetches because another tensor holds its bytes. */
export function isTiedAlias(name: string): boolean {
  return TIED_ALIAS_TO_SOURCE.has(name);
}

/** The name whose bytes actually arrive. Anything that is not an alias comes back unchanged. */
export function resolveTiedTensorName(name: string): string {
  return TIED_ALIAS_TO_SOURCE.get(name) ?? name;
}

/**
 * Look one tensor up, following the tie. This is the tie aware door into a directory that is still
 * a literal reading of the header, and it is what `applyTies` is built out of.
 */
export function lookupTensor(directory: SafetensorsDirectory, name: string): TensorEntry | undefined {
  const direct = directory.byName.get(resolveTiedTensorName(name));
  if (direct) return direct;
  return directory.byName.get(name);
}

/**
 * The directory a consumer should hold once the ties are taken.
 *
 * A free `lookupTensor` is not enough on its own, because the natural thing to write is
 * `directory.byName.get('lm_head.weight')`, and on a literal directory that returns the entry for
 * bytes the allow list deliberately never fetched. Reading it would hand a kernel a buffer nobody
 * filled. So the alias names are remapped in `byName` to point at the source entry, and every
 * consumer gets the right bytes whether or not it knows the tie exists.
 *
 * `tensors` is left exactly as the header reads it, because it describes the file rather than our
 * fetch, and `coverageGaps` has to keep seeing a fully indexed payload. Only the lookup changes.
 */
export function applyTies(
  directory: SafetensorsDirectory,
  ties: readonly TiedTensor[] = TIED_TENSORS,
): SafetensorsDirectory {
  if (ties.length === 0) return directory;
  const byName = new Map(directory.byName);
  for (const tie of ties) {
    const source = directory.byName.get(tie.source);
    if (!source) {
      throw new SafetensorsError(
        'bad-entry',
        `tie ${tie.alias} points at ${tie.source} and this checkpoint does not carry it`,
      );
    }
    byName.set(tie.alias, source);
  }
  return { ...directory, byName };
}

// The tie guard.
//
// The guard hashes the first 256 KiB of the embed tensor and compares it against a digest pinned to
// the revision the tie was verified on. It is a fingerprint of the file, not a proof that the two
// tensors match today: the byte for byte comparison happened once, offline, over both tensors in
// full, and pinning the fingerprint is how a load decides whether it is looking at that same file.
// A checkpoint whose embed bytes differ from the pin is a checkpoint the tie was never checked
// against, and the loader must not assume anything about it.
//
// The fallback when the digest misses is a plain two fetch load: put both lm_head tensors back on
// the allow list, pay the 101,711,872 bytes, and carry on correct but 100 MB heavier. That path is
// one boolean, `fetchTiedPair`, so it cannot rot: the same allow list builder produces both shapes.

/** The revision the tie was verified on, byte for byte, over both tensors in full. */
export const TIE_GUARD_REVISION = 'dd693ff40353f057ca5f07e945ad867f4afbf2ec';

/** How much of the source tensor the guard hashes. */
export const TIE_GUARD_SAMPLE_BYTES = 256 * 1024;

/** The tensor the guard samples. It is the one that stays on the allow list. */
export const TIE_GUARD_SOURCE_TENSOR = 'model.language_model.embed_tokens.embedding_quantized';

/**
 * SHA-256 of the first 256 KiB of `model.language_model.embed_tokens.embedding_quantized` at
 * revision dd693ff40353f057ca5f07e945ad867f4afbf2ec. The same 256 KiB of `lm_head.weight` hashes to
 * the same value, which is the tie in one line. Anyone with the checkpoint can reproduce this with
 * eight lines of Python, which is the point of pinning a digest rather than a byte count.
 */
export const TIE_GUARD_SAMPLE_SHA256 = '2bb8a288ec1d80f80f79eef5c447e37658037d281335079d99db2fd85a146fb5';

export interface TieGuardResult {
  /** Whether the ties may be taken. False means load the aliases as their own tensors. */
  readonly tied: boolean;
  /** The digest of the sample handed in. */
  readonly digest: string;
  /** The digest that was expected. */
  readonly expected: string;
  /** Plain prose for a log line or a thrown message. */
  readonly reason: string;
}

/**
 * The absolute byte range a caller fetches to feed the guard. Shaped as a PlannedRange so
 * `rangeHeaderValue` builds the header for it, which keeps one definition of how a range becomes
 * an HTTP header.
 */
export function tieGuardRange(directory: SafetensorsDirectory): PlannedRange {
  const entry = directory.byName.get(TIE_GUARD_SOURCE_TENSOR);
  if (!entry) {
    throw new SafetensorsError(
      'bad-entry',
      `tie guard needs ${TIE_GUARD_SOURCE_TENSOR} and this checkpoint does not carry it`,
    );
  }
  const length = Math.min(TIE_GUARD_SAMPLE_BYTES, entry.byteLength);
  return {
    index: 0,
    start: entry.fileStart,
    end: entry.fileStart + length,
    byteLength: length,
    tensors: [{ name: entry.name, entry, offsetInRange: 0, byteLength: length }],
  };
}

/**
 * Decide whether the ties hold for this load. Both conditions have to pass: the revision has to be
 * the one the tie was verified on, and the sample has to hash to the pinned digest.
 */
export function checkTieGuard(sample: Uint8Array, revision: string): TieGuardResult {
  const expected = TIE_GUARD_SAMPLE_SHA256;
  if (revision !== TIE_GUARD_REVISION) {
    return {
      tied: false,
      digest: '',
      expected,
      reason:
        `tie guard: revision ${revision} is not the pinned ${TIE_GUARD_REVISION}, `
        + 'so the lm_head tie was never verified against this checkpoint. Fetching both tensors.',
    };
  }
  if (sample.length !== TIE_GUARD_SAMPLE_BYTES) {
    return {
      tied: false,
      digest: '',
      expected,
      reason:
        `tie guard: sample is ${sample.length} bytes, expected ${TIE_GUARD_SAMPLE_BYTES}. `
        + 'Fetching both tensors.',
    };
  }
  const digest = sha256Hex(sample);
  if (digest !== expected) {
    return {
      tied: false,
      digest,
      expected,
      reason:
        `tie guard: ${TIE_GUARD_SOURCE_TENSOR} sample hashes to ${digest}, pinned ${expected}. `
        + 'The checkpoint is not the one the tie was verified on. Fetching both tensors.',
    };
  }
  return { tied: true, digest, expected, reason: 'tie guard: sample matches the pinned digest' };
}

// SHA-256, because the guard needs a digest a human can reproduce with a shell one liner and both
// halves of this project can compute. Web Crypto is async and Node's crypto is not in a browser, so
// this is 40 lines of the FIPS 180-4 compression function rather than a branch on the environment.
// It runs once per load over 256 KiB, so its speed is not interesting.

const SHA256_K = new Uint32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]);

function rotr32(value: number, bits: number): number {
  return ((value >>> bits) | (value << (32 - bits))) >>> 0;
}

/** SHA-256 of a byte range, lower case hexadecimal. */
export function sha256Hex(bytes: Uint8Array): string {
  const state = new Uint32Array([
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
  ]);
  const paddedLength = (bytes.length + 9 + 63) & ~63;
  const padded = new Uint8Array(paddedLength);
  padded.set(bytes);
  padded[bytes.length] = 0x80;
  const view = new DataView(padded.buffer);
  // The length goes in as a big endian u64 of bits, split because a byte count past 512 MB
  // overflows the low word.
  const bitLength = bytes.length * 8;
  view.setUint32(paddedLength - 8, Math.floor(bitLength / 0x1_0000_0000), false);
  view.setUint32(paddedLength - 4, bitLength >>> 0, false);

  const w = new Uint32Array(64);
  for (let offset = 0; offset < paddedLength; offset += 64) {
    for (let i = 0; i < 16; i += 1) w[i] = view.getUint32(offset + i * 4, false);
    for (let i = 16; i < 64; i += 1) {
      const x = w[i - 15];
      const y = w[i - 2];
      const s0 = (rotr32(x, 7) ^ rotr32(x, 18) ^ (x >>> 3)) >>> 0;
      const s1 = (rotr32(y, 17) ^ rotr32(y, 19) ^ (y >>> 10)) >>> 0;
      w[i] = (w[i - 16] + s0 + w[i - 7] + s1) >>> 0;
    }
    let a = state[0];
    let b = state[1];
    let c = state[2];
    let d = state[3];
    let e = state[4];
    let f = state[5];
    let g = state[6];
    let h = state[7];
    for (let i = 0; i < 64; i += 1) {
      const s1 = (rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25)) >>> 0;
      const ch = ((e & f) ^ (~e & g)) >>> 0;
      const temp1 = (h + s1 + ch + SHA256_K[i] + w[i]) >>> 0;
      const s0 = (rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22)) >>> 0;
      const maj = ((a & b) ^ (a & c) ^ (b & c)) >>> 0;
      const temp2 = (s0 + maj) >>> 0;
      h = g;
      g = f;
      f = e;
      e = (d + temp1) >>> 0;
      d = c;
      c = b;
      b = a;
      a = (temp1 + temp2) >>> 0;
    }
    state[0] = (state[0] + a) >>> 0;
    state[1] = (state[1] + b) >>> 0;
    state[2] = (state[2] + c) >>> 0;
    state[3] = (state[3] + d) >>> 0;
    state[4] = (state[4] + e) >>> 0;
    state[5] = (state[5] + f) >>> 0;
    state[6] = (state[6] + g) >>> 0;
    state[7] = (state[7] + h) >>> 0;
  }

  let hex = '';
  for (let i = 0; i < 8; i += 1) hex += state[i].toString(16).padStart(8, '0');
  return hex;
}

// Text tensor selection.
//
// Section 3 of the plan asks for an allow list built from the layer count rather than a deny list,
// so a tower added upstream is silently ignored instead of silently downloaded. The names below
// were read from the checkpoint's own header.

export interface TextTensorSelectionConfig {
  /** text_config.num_hidden_layers, 35 on this checkpoint. */
  readonly numHiddenLayers: number;
  /** text_config.num_kv_shared_layers, 20 on this checkpoint, so layers 0 to 14 produce KV. */
  readonly numKvSharedLayers: number;
  /**
   * Put the tied lm_head pair back on the wire as its own 101,711,872 bytes. This is the two fetch
   * fallback and the tie guard is the only thing that should ever set it. Default false.
   */
  readonly fetchTiedPair?: boolean;
}

/**
 * The suffixes every quantized linear carries. weight is the packed codes, weight_scale is one f32
 * per output row, and the two activation scales are the SRQ calibration scalars.
 */
const LINEAR_SUFFIXES = ['weight', 'weight_scale', 'input_activation_scale', 'output_activation_scale'] as const;

export interface TextTensorSelection {
  /** Names to fetch, in checkpoint order once resolved against a directory. */
  readonly names: string[];
  /** Names present in the file that we deliberately leave behind. */
  readonly skipped: string[];
  /** Names we asked for that the file does not have. Empty means the allow list matches the export. */
  readonly missing: string[];
  /**
   * Ties in force for this selection: names the engine knows but never fetches, because another
   * tensor in `names` holds their bytes. Empty when the two fetch fallback is on. These are not in
   * `skipped`, because skipped means dead and a tied tensor is very much alive.
   */
  readonly tied: readonly TiedTensor[];
  /**
   * The directory to hand consumers, with `applyTies` already applied for the ties in force. Hold
   * this one rather than the literal header reading, and `byName.get('lm_head.weight')` returns the
   * embed entry whose bytes the plan actually fetches. Identical to the input directory when the
   * two fetch fallback is on, because then there is nothing to alias.
   */
  readonly resolvedDirectory: SafetensorsDirectory;
}

/** Build the allow list from the layer count alone, with no reference to the file. */
export function textTensorNames(config: TextTensorSelectionConfig): string[] {
  const layers = Math.max(0, Math.trunc(config.numHiddenLayers));
  const shared = Math.max(0, Math.trunc(config.numKvSharedLayers));
  // Producers own real KV caches. Consumers read a producer's cache, and the reference
  // implementation drops their k_proj, v_proj, k_norm and v_norm as unexpected keys on load, so we
  // never put them on the wire. The checkpoint ships k_proj and v_proj for them anyway.
  const producerLayers = Math.max(0, layers - shared);

  const names: string[] = [
    // lm_head.weight and lm_head.weight_scale are absent on purpose. They are byte identical to the
    // embed_tokens pair two lines down, and a consumer holding the selection's resolvedDirectory
    // reaches them under their own names anyway. The two activation scales below are lm_head's own,
    // four bytes each, and are not tied to anything.
    'lm_head.input_activation_scale',
    'lm_head.output_activation_scale',
    'model.language_model.embed_tokens.embedding_quantized',
    'model.language_model.embed_tokens.embedding_scale',
    'model.language_model.embed_tokens_per_layer.embedding_quantized',
    'model.language_model.embed_tokens_per_layer.embedding_scale',
    'model.language_model.norm.weight',
    'model.language_model.per_layer_model_projection.weight',
    'model.language_model.per_layer_projection_norm.weight',
  ];

  for (let layer = 0; layer < layers; layer += 1) {
    const base = `model.language_model.layers.${layer}`;
    names.push(
      `${base}.input_layernorm.weight`,
      `${base}.post_attention_layernorm.weight`,
      `${base}.pre_feedforward_layernorm.weight`,
      `${base}.post_feedforward_layernorm.weight`,
      `${base}.post_per_layer_input_norm.weight`,
      `${base}.layer_scalar`,
      `${base}.self_attn.q_norm.weight`,
      `${base}.self_attn.k_cache_scale`,
      `${base}.self_attn.v_cache_scale`,
    );
    const isProducer = layer < producerLayers;
    if (isProducer) names.push(`${base}.self_attn.k_norm.weight`);

    const linears = ['self_attn.q_proj', 'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj', 'per_layer_input_gate', 'per_layer_projection'];
    if (isProducer) linears.push('self_attn.k_proj', 'self_attn.v_proj');
    for (const linear of linears) {
      for (const suffix of LINEAR_SUFFIXES) names.push(`${base}.${linear}.${suffix}`);
    }
  }

  // The two fetch fallback. The guard sets this when the sample digest misses, and then the aliases
  // come down as their own bytes and nothing resolves through the tie.
  if (config.fetchTiedPair) {
    for (const tie of TIED_TENSORS) names.push(tie.alias);
  }

  return names;
}

/** Resolve the allow list against a parsed directory, reporting what is left behind and what is absent. */
export function selectTextTensors(
  directory: SafetensorsDirectory,
  config: TextTensorSelectionConfig,
): TextTensorSelection {
  const wanted = new Set(textTensorNames(config));
  const tied = config.fetchTiedPair ? [] : TIED_TENSORS.filter((tie) => directory.byName.has(tie.alias));
  const tiedAliases = new Set(tied.map((tie) => tie.alias));
  const names: string[] = [];
  const skipped: string[] = [];
  for (const tensor of directory.tensors) {
    if (wanted.has(tensor.name)) names.push(tensor.name);
    else if (!tiedAliases.has(tensor.name)) skipped.push(tensor.name);
  }
  const present = new Set(names);
  const missing = [...wanted].filter((name) => !present.has(name));
  return { names, skipped, missing, tied, resolvedDirectory: applyTies(directory, tied) };
}

/**
 * Look tensor names up in a directory, rejecting any the file does not carry.
 *
 * Pass the literal directory here, not the selection's `resolvedDirectory`. Its input is a fetch
 * list, and on an aliased directory a list naming both an alias and its source would resolve to the
 * same entry twice and hand buildRangePlan a duplicate. Ties are a consumer side concern, and
 * `resolvedDirectory` is the door for them.
 */
export function resolveEntries(
  directory: SafetensorsDirectory,
  names: readonly string[],
): TensorEntry[] {
  const entries: TensorEntry[] = [];
  for (const name of names) {
    const entry = directory.byName.get(name);
    if (!entry) throw new SafetensorsError('bad-entry', `tensor ${name} is not in this checkpoint`);
    entries.push(entry);
  }
  return entries;
}

/** The resolve URL a ranged fetch targets. The 302 to the CDN keeps Accept-Ranges and the CORS headers. */
export function resolveUrl(repo: string, revision: string, path: string): string {
  return `https://huggingface.co/${repo}/resolve/${encodeURIComponent(revision)}/${path}`;
}
