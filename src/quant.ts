// Written from the Hugging Face transformers reference implementation of Google's "gemma" quant
// method (src/transformers/integrations/gemma_quant.py, Apache 2.0), the quantization_config block
// of the checkpoint's own config.json, and EMBEDDING-QUANT-FINDING.md.
// SPDX-License-Identifier: Apache-2.0
//
// This file is the documented source of truth for what the shaders implement. Every kernel that
// unpacks a weight is implementing one of the functions below, and the loader check dequantizes
// real bytes through here and compares against the reference dump. If a shader and this file ever
// disagree, this file is right and the shader is the bug.
//
// The three storage families, all row major with the packed axis last:
//
//   4-bit  two codes per byte, low nibble first, unsigned 0..15, zero point 8
//   2-bit  four codes per byte at bit offsets 0, 2, 4, 6, unsigned 0..3, zero point 2
//   8-bit  one signed byte per code, no packing
//
// A quantized linear stores one f32 scale per output row. A quantized embedding stores one f32
// scale per block of the row, where the block size is the embedding dimension divided by the number
// of scale columns: 1536 over 1 for embed_tokens, and 8960 over 35 for the PLE table, which is the
// 256 wide per layer group EMBEDDING-QUANT-FINDING.md measured.
//
// Nothing here throws on bad input. A dequant that raises in the middle of a 2 GB load turns a
// recoverable byte problem into a dead engine, so short buffers read as zero codes and out of range
// scale lookups fall back to the last scale. The loader validates lengths; this layer stays total.

/** The bit widths this checkpoint uses. Nothing in the text model is stored at any other width. */
export type QuantBits = 2 | 4 | 8;

/** Codes are unsigned in the file and shifted by this to land on the signed grid. */
export const ZERO_POINT: Readonly<Record<QuantBits, number>> = Object.freeze({ 2: 2, 4: 8, 8: 0 });

/** Codes per stored byte. */
export const CODES_PER_BYTE: Readonly<Record<QuantBits, number>> = Object.freeze({ 2: 4, 4: 2, 8: 1 });

/** The PLE table is 35 groups of 256, one group per decoder layer, with its own scale column each. */
export const PLE_NUM_GROUPS = 35;
export const PLE_GROUP_SIZE = 256;
export const PLE_BITS: QuantBits = 4;

/**
 * The architectural embed scale the PLE gather multiplies in, which is
 * sqrt(hidden_size_per_layer_input) with hidden_size_per_layer_input 256. This is the 16 in
 * EMBEDDING-QUANT-FINDING.md's `y = 16 * scale[id, layer] * (code - 8)`.
 */
export const PLE_EMBED_SCALE = 16;

/**
 * The input embedding's architectural embed scale, sqrt(hidden_size) with hidden_size 1536. The
 * reference keeps this in f32 and casts the product to the model dtype, and the Gemma family has a
 * known bf16 downcast wrinkle at this exact multiply, so parity work uses roundToBf16 rather than
 * assuming the product survives.
 */
export const EMBED_TOKENS_EMBED_SCALE = Math.sqrt(1536);

/** Bytes needed to store `width` codes at `bits`, matching the reference's ceiling division. */
export function packedBytesForWidth(bits: QuantBits, width: number): number {
  const codes = CODES_PER_BYTE[bits];
  const safeWidth = Number.isFinite(width) && width > 0 ? Math.trunc(width) : 0;
  return Math.ceil(safeWidth / codes);
}

/**
 * Infer the bit width from a stored row length and the logical width. The checkpoint's shapes make
 * this exact: q_proj is [2048, 768] over 1536 inputs so 4 bits, the consumer gate_proj is
 * [12288, 384] over the same 1536 so 2 bits, and down_proj is [1536, 3072] over 6144 inputs on a
 * producer and over 12288 on a consumer, which is the same byte count at two different widths.
 * Returns null when the numbers do not land on one of the three families.
 */
export function bitsFromPackedRow(packedBytes: number, width: number): QuantBits | null {
  if (!Number.isFinite(packedBytes) || !Number.isFinite(width) || width <= 0 || packedBytes <= 0) return null;
  for (const bits of [2, 4, 8] as QuantBits[]) {
    if (packedBytesForWidth(bits, width) === Math.trunc(packedBytes)) return bits;
  }
  return null;
}

const EMPTY_BYTES = new Uint8Array(0);

function byteAt(packed: Uint8Array, index: number): number {
  return index >= 0 && index < packed.length ? packed[index] : 0;
}

/**
 * Unpack 4-bit codes. Two per byte, low nibble first, then the high nibble, shifted from unsigned
 * 0..15 to signed -8..7. Trailing codes past `width` are dropped exactly as the reference does.
 */
export function unpackInt4(packed: Uint8Array, width: number, out?: Int8Array): Int8Array {
  const count = Number.isFinite(width) && width > 0 ? Math.trunc(width) : 0;
  const values = out && out.length >= count ? out : new Int8Array(count);
  for (let i = 0; i < count; i += 1) {
    const byte = byteAt(packed, i >> 1);
    const nibble = (i & 1) === 0 ? (byte & 0x0f) : (byte >> 4) & 0x0f;
    values[i] = nibble - 8;
  }
  return values.length === count ? values : values.subarray(0, count);
}

/**
 * Unpack 2-bit codes. Four per byte at bit offsets 0, 2, 4 and 6, shifted from unsigned 0..3 to
 * signed -2..1. The lab's 0x03030303 mask with four chunks per word is the same layout seen a u32
 * at a time (PREFILL-CAMPAIGN.md, retile campaign result): masking a word lifts code j of each of
 * the four bytes at once.
 */
export function unpackInt2(packed: Uint8Array, width: number, out?: Int8Array): Int8Array {
  const count = Number.isFinite(width) && width > 0 ? Math.trunc(width) : 0;
  const values = out && out.length >= count ? out : new Int8Array(count);
  for (let i = 0; i < count; i += 1) {
    const byte = byteAt(packed, i >> 2);
    const shift = (i & 3) * 2;
    values[i] = ((byte >> shift) & 0x03) - 2;
  }
  return values.length === count ? values : values.subarray(0, count);
}

/**
 * Read 8-bit codes, which are stored as signed bytes with nothing to unpack and no zero point to
 * undo. This family is stored as safetensors I8 rather than U8, so the byte is already the code,
 * which is why ZERO_POINT[8] is 0 and not 128. Confirmed against the real header: the two 8-bit
 * text modules are per_layer_input_gate at [256, 1536] and per_layer_projection at [1536, 256].
 */
export function unpackInt8(packed: Uint8Array, width: number, out?: Int8Array): Int8Array {
  const count = Number.isFinite(width) && width > 0 ? Math.trunc(width) : 0;
  const values = out && out.length >= count ? out : new Int8Array(count);
  for (let i = 0; i < count; i += 1) {
    const byte = byteAt(packed, i);
    values[i] = byte > 127 ? byte - 256 : byte;
  }
  return values.length === count ? values : values.subarray(0, count);
}

/** Dispatch to the right unpacker. Any width the family does not know falls back to 8-bit. */
export function unpackCodes(bits: QuantBits, packed: Uint8Array, width: number, out?: Int8Array): Int8Array {
  if (bits === 2) return unpackInt2(packed, width, out);
  if (bits === 4) return unpackInt4(packed, width, out);
  return unpackInt8(packed, width, out);
}

/** Pack signed codes back into the stored layout. Only the loader check uses this, to prove the unpackers round trip. */
export function packCodes(bits: QuantBits, codes: ArrayLike<number>, width: number): Uint8Array {
  const count = Number.isFinite(width) && width > 0 ? Math.trunc(width) : 0;
  const bytes = new Uint8Array(packedBytesForWidth(bits, count));
  const zero = ZERO_POINT[bits];
  const perByte = CODES_PER_BYTE[bits];
  const mask = bits === 2 ? 0x03 : bits === 4 ? 0x0f : 0xff;
  for (let i = 0; i < count; i += 1) {
    const raw = Number(codes[i] ?? 0);
    const code = (Number.isFinite(raw) ? Math.trunc(raw) : 0) + zero;
    const shift = (i % perByte) * bits;
    bytes[Math.floor(i / perByte)] |= (code & mask) << shift;
  }
  return bytes;
}

/**
 * Dequantize one row of a quantized linear: `code * weight_scale[row]`, one f32 scale for the whole
 * row. This is the maths K4 and K5 implement in the shader at the point of use, which is what keeps
 * residency near 1.87 GB instead of the dequantized size.
 */
export function dequantLinearRow(
  bits: QuantBits,
  packedRow: Uint8Array,
  inFeatures: number,
  scale: number,
  out?: Float32Array,
): Float32Array {
  const count = Number.isFinite(inFeatures) && inFeatures > 0 ? Math.trunc(inFeatures) : 0;
  const values = out && out.length >= count ? out.subarray(0, count) : new Float32Array(count);
  const safeScale = Number.isFinite(scale) ? scale : 0;
  const codes = unpackCodes(bits, packedRow, count);
  for (let i = 0; i < count; i += 1) values[i] = codes[i] * safeScale;
  return values;
}

/**
 * Dequantize a whole quantized linear into a row major f32 matrix of [outFeatures, inFeatures].
 * Intended for tests and for the CPU reference forward, never for the load path: doing this to the
 * consumer MLP would turn 0.5 GB of packed weights into several times that in f32.
 */
export function dequantLinearMatrix(
  bits: QuantBits,
  packed: Uint8Array,
  outFeatures: number,
  inFeatures: number,
  scales: ArrayLike<number>,
): Float32Array {
  const rows = Number.isFinite(outFeatures) && outFeatures > 0 ? Math.trunc(outFeatures) : 0;
  const cols = Number.isFinite(inFeatures) && inFeatures > 0 ? Math.trunc(inFeatures) : 0;
  const stride = packedBytesForWidth(bits, cols);
  const values = new Float32Array(rows * cols);
  const scratch = new Int8Array(cols);
  for (let row = 0; row < rows; row += 1) {
    const start = row * stride;
    const packedRow = start < packed.length ? packed.subarray(start, Math.min(start + stride, packed.length)) : EMPTY_BYTES;
    const codes = unpackCodes(bits, packedRow, cols, scratch);
    const rawScale = Number(scales[Math.min(row, scales.length - 1)] ?? 0);
    const scale = Number.isFinite(rawScale) ? rawScale : 0;
    const base = row * cols;
    for (let i = 0; i < cols; i += 1) values[base + i] = codes[i] * scale;
  }
  return values;
}

/**
 * Dequantize one row of a quantized embedding table. The scale row carries one f32 per block, where
 * the block size is `embeddingDim / scaleRow.length`. `embedScale` is the architectural multiplier
 * the embedding forward applies after dequant; pass 1 to get the raw table row.
 */
export function dequantEmbeddingRow(
  bits: QuantBits,
  packedRow: Uint8Array,
  embeddingDim: number,
  scaleRow: ArrayLike<number>,
  embedScale = 1,
  out?: Float32Array,
): Float32Array {
  const dim = Number.isFinite(embeddingDim) && embeddingDim > 0 ? Math.trunc(embeddingDim) : 0;
  const values = out && out.length >= dim ? out.subarray(0, dim) : new Float32Array(dim);
  const groups = scaleRow.length > 0 ? scaleRow.length : 1;
  const blockSize = Math.max(1, Math.floor(dim / groups));
  const factor = Number.isFinite(embedScale) ? embedScale : 1;
  const codes = unpackCodes(bits, packedRow, dim);
  for (let i = 0; i < dim; i += 1) {
    const group = Math.min(groups - 1, Math.floor(i / blockSize));
    const raw = Number(scaleRow[group] ?? 0);
    const scale = Number.isFinite(raw) ? raw : 0;
    values[i] = codes[i] * scale * factor;
  }
  return values;
}

/**
 * The PLE gather for one token id: 35 groups of 256 at 4 bits, each group carrying its own scale
 * column, with the embed scale of 16 folded in. This is exactly
 * `y = 16 * scale[id, layer] * (code - 8)` from EMBEDDING-QUANT-FINDING.md, and it is the K2
 * contract. Output is laid out layer major, so layer L occupies [L * 256, L * 256 + 256).
 */
export function dequantPleRow(packedRow: Uint8Array, scaleRow: ArrayLike<number>, out?: Float32Array): Float32Array {
  return dequantEmbeddingRow(
    PLE_BITS,
    packedRow,
    PLE_NUM_GROUPS * PLE_GROUP_SIZE,
    scaleRow,
    PLE_EMBED_SCALE,
    out,
  );
}

/**
 * The width static range quantization always rounds to, whatever the neighbouring weight width is.
 * The reference's `apply_srq` defaults to 8 and nothing in this checkpoint passes anything else, so
 * a 2-bit consumer MLP still sees int8 activations. One constant, here, because quant.ts is the
 * authority and a second copy in a kernel file is a second thing to keep right.
 */
export const SRQ_ACTIVATION_BITS = 8;

/**
 * Static range quantization of an activation, which is the rounding the checkpoint was calibrated
 * with and the reason a V row is already on an int8 grid before it reaches the cache
 * (DECODE-CAMPAIGN.md 4.3). A scale of zero means the site was never calibrated and the reference
 * makes it a no-op, so this does too.
 *
 * Three details the reference dump pinned down, all of which a shader has to carry. The rounding is
 * always 8 bits regardless of the weight bit width sitting next to it, so a 2-bit consumer MLP still
 * sees int8 activations. It runs on **both** sides of every quantized linear, using that module's
 * own input_activation_scale and output_activation_scale, which are F32 scalars in the file rather
 * than anything derived. And a stored 0.0 means uncalibrated rather than a scale of zero, which is
 * why lm_head, whose two scales are both 0.0, applies no activation rounding at all.
 *
 * A fourth detail, settled at integration: the rounding breaks ties toward the even integer, which
 * is what `torch.round` produced in the reference and what WGSL's `round` does in the shaders. See
 * `roundTiesToEven` below.
 *
 * Round 2 split the rounding in two. The codes are the thing the matmul kernels actually want, so
 * `srqCodes` below computes them and this function is the same operation with the scale multiplied
 * back on. Same numbers as before, one implementation instead of the two the integer path would
 * otherwise have needed.
 */
export function applySrq(values: Float32Array, scale: number, bits = SRQ_ACTIVATION_BITS, out?: Float32Array): Float32Array {
  const result = out && out.length >= values.length ? out.subarray(0, values.length) : new Float32Array(values.length);
  const safeScale = Number.isFinite(scale) ? scale : 0;
  if (safeScale === 0) {
    result.set(values);
    return result;
  }
  const codes = srqCodes(values, safeScale, bits);
  for (let i = 0; i < values.length; i += 1) result[i] = codes[i] * safeScale;
  return result;
}

/**
 * The same rounding as `applySrq`, stopped one step earlier: the integer CODES rather than the
 * codes multiplied back by the scale. This is what the round 2 integer semantics consume.
 *
 * A quantized linear's ratified arithmetic is `sum over K of xcode * wcode` accumulated in i32 and
 * scaled once at the end by `input_scale * weight_scale`. That needs the input's codes, not its
 * dequantized image, so the two functions are split here and `applySrq` is defined in terms of this
 * one. The identity `applySrq(v, s) === srqCodes(v, s) * s` therefore holds by construction rather
 * than by two implementations agreeing, and the k-matmul check asserts it anyway on real slices.
 *
 * A scale of exactly 0.0 means the site was never calibrated, and an uncalibrated site has no code
 * representation at all. This returns zeros for that case rather than throwing, because nothing in
 * this file throws, and every caller branches on the scale before it gets here.
 *
 * The codes come back as integer valued f32 rather than as an integer array, which is not a
 * looseness: `round` of a small negative value is negative zero, WGSL's `round` produces negative
 * zero there too, and the sign survives the multiply back in `applySrq` and in the gelu-mul
 * shader's own epilogue. An integer array would quietly flatten it and move a gated fixture.
 */
export function srqCodes(values: Float32Array, scale: number, bits = SRQ_ACTIVATION_BITS, out?: Float32Array): Float32Array {
  const result = out && out.length >= values.length ? out.subarray(0, values.length) : new Float32Array(values.length);
  const safeBits = Number.isFinite(bits) && bits >= 2 ? Math.trunc(bits) : SRQ_ACTIVATION_BITS;
  const safeScale = Number.isFinite(scale) ? scale : 0;
  if (safeScale === 0) {
    result.fill(0);
    return result;
  }
  const maxValue = 2 ** (safeBits - 1) - 1;
  const minValue = -maxValue - 1;
  for (let i = 0; i < values.length; i += 1) {
    // THE DIVISION IS ROUNDED TO f32 BEFORE THE ROUND, and that is a correctness fix rather than a
    // flourish. Both sides this oracle stands in for divide in f32: `apply_srq` divides in x.dtype,
    // which is f32 on this reference, and `srqIn` and `srqOut` in the shaders divide f32 by f32.
    // JavaScript divides in f64, and the two disagree exactly where this engine's parity arguments
    // live. A real element of this checkpoint, layer 0 gate_proj at position 112 of interview-300:
    // the accumulator is -11.435039520263672 and the scale 0.6181102395057678, whose f64 quotient
    // is -18.500000144645664 and rounds to -19, while the f32 quotient is exactly -18.5 and the
    // ties to even rule sends it to -18. Without the fround this oracle and the kernel it checks
    // return different codes for that element, and the disagreement is invisible on any fixture
    // that has no element sitting on a boundary.
    const rounded = roundTiesToEven(Math.fround(values[i] / safeScale));
    result[i] = rounded < minValue ? minValue : rounded > maxValue ? maxValue : rounded;
  }
  return result;
}

/**
 * Round half to even, which is what both sides of this engine actually do.
 *
 * `Math.round` breaks a tie toward positive infinity. `torch.round`, which is what produced the
 * reference activations, and WGSL's `round`, which is what the shaders call, both break a tie
 * toward the even integer. Integrator's call, taken because the two sides disagreed and the
 * reference is the tiebreaker: quant.ts is the layout authority, so quant.ts moves to the rule the
 * reference and the shader share rather than the shaders moving to this file's rule.
 *
 * The change is only ever visible when `value / scale` lands exactly on a half, which the k-mlp
 * check measures rather than assumes: it counts the ties in every gated activation fixture and
 * asserts both rules resolve them identically, so this edit moved no gated number. It removes the
 * case where they would not.
 */
export function roundTiesToEven(x: number): number {
  if (!Number.isFinite(x)) return x;
  const truncated = Math.trunc(x);
  const fraction = x - truncated;
  if (fraction !== 0.5 && fraction !== -0.5) return Math.round(x);
  // Exactly on a half. Step to whichever neighbour is even.
  return truncated % 2 === 0 ? truncated : truncated + Math.sign(fraction);
}

// Reading stored bytes into typed arrays.
//
// A tensor sliced out of a coalesced range starts wherever the file put it, so the byte offset is
// almost never four byte aligned. Constructing a Float32Array over such a view throws, which is the
// one place a loader would fall over on entirely valid data, so these copy when they have to and
// alias when they can.

/** View or copy raw little endian bytes as f32. */
export function asFloat32(bytes: Uint8Array): Float32Array {
  const count = Math.floor(bytes.length / 4);
  if ((bytes.byteOffset & 3) === 0) {
    return new Float32Array(bytes.buffer, bytes.byteOffset, count);
  }
  const copy = new Float32Array(count);
  const view = new DataView(bytes.buffer, bytes.byteOffset, count * 4);
  for (let i = 0; i < count; i += 1) copy[i] = view.getFloat32(i * 4, true);
  return copy;
}

/** View or copy raw bytes as signed int8. */
export function asInt8(bytes: Uint8Array): Int8Array {
  return new Int8Array(bytes.buffer, bytes.byteOffset, bytes.length);
}

/**
 * Decode bf16 to f32. The norm weights, the layer scalar and per_layer_model_projection are stored
 * as bf16, which is the top 16 bits of the f32 pattern, so this is a shift rather than a conversion.
 */
/**
 * Which BF16 tensors the GPU reads PACKED, two values to a u32, instead of widened to f32 on the
 * way in. A BF16 value is the top sixteen bits of its f32 pattern, so a shader can widen it with
 * one shift, and a tensor big enough for its bytes to matter should pay that shift rather than
 * twice the traffic. per_layer_model_projection is the only one: 27.5 MB on the wire against 55 MB
 * widened, and a decode token reads all of it (docs/ENGINE-PERF.md section 17). The norm gains and
 * the layer scalars stay widened because the norm kernels read `array<f32>` and their bytes round
 * to nothing. Every loader asks this, so the rule has one home rather than three.
 */
export function bf16StaysPacked(name: string): boolean {
  return name.endsWith('per_layer_model_projection.weight');
}

export function bf16ToF32(bytes: Uint8Array, out?: Float32Array): Float32Array {
  const count = Math.floor(bytes.length / 2);
  const values = out && out.length >= count ? out.subarray(0, count) : new Float32Array(count);
  const scratch = new DataView(new ArrayBuffer(4));
  for (let i = 0; i < count; i += 1) {
    const lo = bytes[i * 2];
    const hi = bytes[i * 2 + 1];
    scratch.setUint32(0, ((hi << 8) | lo) << 16, true);
    values[i] = scratch.getFloat32(0, true);
  }
  return values;
}

/**
 * Round an f32 to the nearest bf16 and return it as f32, ties to even. The reference casts several
 * products to bf16 on the way through, so a CPU reference forward that wants to match it needs this
 * rather than staying in f32 the whole way.
 */
export function roundToBf16(value: number): number {
  if (!Number.isFinite(value)) return value;
  const scratch = new DataView(new ArrayBuffer(4));
  scratch.setFloat32(0, value, true);
  const bits = scratch.getUint32(0, true);
  const lower = bits & 0xffff;
  let upper = bits >>> 16;
  if (lower > 0x8000 || (lower === 0x8000 && (upper & 1) === 1)) upper += 1;
  scratch.setUint32(0, (upper & 0xffff) << 16, true);
  return scratch.getFloat32(0, true);
}

// Which family a module belongs to.
//
// Transcribed from the checkpoint's own config.json quantization_config. The reference joins these
// patterns into one alternation and takes the leftmost match, preferring the earlier pattern on a
// tie, which is what resolveQuantBits reproduces. The ordering matters exactly once: layers 0 to 14
// take 4-bit MLP weights from the first mlp rule and layers 15 and up fall through to the second at
// 2-bit, which is the producer and consumer split the census measured (DECODE-CAMPAIGN.md 4.7).

export interface QuantRule {
  readonly pattern: string;
  readonly bits: QuantBits;
}

/** The module_quant_configs table, in config.json order. */
export const QUANT_RULES: readonly QuantRule[] = Object.freeze([
  { pattern: '^lm_head$', bits: 2 },
  { pattern: 'audio_tower(?!.*lconv1d\\.linear_start)', bits: 2 },
  { pattern: 'audio_tower\\.layers\\.\\d+\\.lconv1d\\.linear_start\\.', bits: 4 },
  { pattern: 'language_model\\.embed_tokens$', bits: 2 },
  { pattern: 'language_model\\.embed_tokens_per_layer$', bits: 4 },
  { pattern: 'language_model\\.layers\\.(\\d|1[0-4])\\.mlp\\.', bits: 4 },
  { pattern: 'language_model\\.layers\\.\\d+\\.mlp\\.', bits: 2 },
  { pattern: 'language_model\\.layers\\.\\d+\\.per_layer_input_gate$', bits: 8 },
  { pattern: 'language_model\\.layers\\.\\d+\\.per_layer_projection$', bits: 8 },
  { pattern: 'language_model\\.layers\\.\\d+\\.self_attn\\.', bits: 4 },
  { pattern: 'vision_tower', bits: 8 },
]);

/** The default when no rule matches, from quantization_config.num_bits. */
export const DEFAULT_QUANT_BITS: QuantBits = 4;

/** Modules the checkpoint leaves unquantized, from quantization_config.modules_to_not_convert. */
export const MODULES_TO_NOT_CONVERT: readonly string[] = Object.freeze([
  'model.vision_tower.patch_embedder',
  'model.audio_tower.subsample_conv_projection',
  'model.audio_tower.output_proj',
  'relative_k_proj',
  'model.embed_audio',
  'model.embed_vision',
  'per_layer_model_projection',
]);

/**
 * Leaves that are never a quantized linear, whatever the rules say. The rule table answers "if this
 * module were a quantized linear, at what width", and its fallthrough is 4-bit, so asking it about a
 * norm weight gets a confident wrong answer. The reference only ever replaces Linear modules, so a
 * norm, the per layer scalar and the two KV cache scales stay in their stored dtype. Every norm in
 * this model ends its module path with `norm`, which covers the four block norms, the per layer
 * input norm, the final norm, and q_norm and k_norm.
 */
const NEVER_QUANTIZED_LEAVES: readonly string[] = Object.freeze(['layer_scalar', 'k_cache_scale', 'v_cache_scale']);

/** True when the checkpoint stores this module unquantized, so its bytes are read in their own dtype. */
export function isNeverQuantized(modulePath: string): boolean {
  for (const skip of MODULES_TO_NOT_CONVERT) {
    if (modulePath.includes(skip)) return true;
  }
  const leaf = modulePath.slice(modulePath.lastIndexOf('.') + 1);
  return leaf.endsWith('norm') || NEVER_QUANTIZED_LEAVES.includes(leaf);
}

const COMPILED_RULES = QUANT_RULES.map((rule) => ({ bits: rule.bits, regex: new RegExp(rule.pattern) }));

/**
 * The bit width for a module path such as `model.language_model.layers.20.mlp.gate_proj`. Pass the
 * module path, not the tensor name, so drop any trailing `.weight` or `.weight_scale` first.
 * Returns null for a module the checkpoint stores unquantized.
 */
export function resolveQuantBits(modulePath: string): QuantBits | null {
  if (isNeverQuantized(modulePath)) return null;
  let bestIndex = Number.POSITIVE_INFINITY;
  let bestBits: QuantBits = DEFAULT_QUANT_BITS;
  let matched = false;
  for (const rule of COMPILED_RULES) {
    const found = rule.regex.exec(modulePath);
    if (!found) continue;
    if (found.index < bestIndex) {
      bestIndex = found.index;
      bestBits = rule.bits;
      matched = true;
    }
  }
  return matched ? bestBits : DEFAULT_QUANT_BITS;
}

// The storage families, as they actually appear in the file.
//
// ENGINE-PLAN.md section 3 lists four families. The real header has more, and the reference lane
// recorded the difference as a discrepancy rather than smoothing it over: the two per layer PLE
// linears are 8-bit I8, and per_layer_model_projection is not quantized at all. This table is the
// one place a kernel author should have to look to know what a tensor's bytes mean, and the loader
// check asserts every row of it against the real header rather than trusting the prose.
//
// Every row was confirmed by dequantizing a real slice and comparing to the reference f32 dump:
// each family reduces to `code * scale` exactly, with no architectural embed scale folded into the
// stored values. The embed scale below is what the forward multiplies in afterwards.

export interface TextStorageFamily {
  /** The reference dump's id for this family, so a failure can be traced back to a slice. */
  readonly id: string;
  /** One real tensor of the family, named exactly as the header names it. */
  readonly example: string;
  /** The safetensors dtype the bytes are stored under. */
  readonly dtype: 'U8' | 'I8' | 'BF16';
  /** Code width, or null when the family is stored unquantized. */
  readonly bits: QuantBits | null;
  /** The example tensor's shape, as the header gives it. */
  readonly shape: readonly number[];
  /** Columns in the matching scale tensor. 1 is one scale per row, 35 is the PLE's per layer blocking, 0 is no scale tensor. */
  readonly scaleColumns: number;
  /** The architectural multiplier the forward applies after dequant, which is not stored anywhere. */
  readonly embedScale: number;
  readonly note: string;
}

/** Every storage family in the text stack. Verified against the checkpoint's own header at 375,392 bytes and 2,780 tensors. */
export const TEXT_STORAGE_FAMILIES: readonly TextStorageFamily[] = Object.freeze([
  {
    id: 'ple-table-4bit',
    example: 'model.language_model.embed_tokens_per_layer.embedding_quantized',
    dtype: 'U8',
    bits: 4,
    shape: [262144, 4480],
    scaleColumns: PLE_NUM_GROUPS,
    embedScale: PLE_EMBED_SCALE,
    // 262,144 rows of 4,480 bytes is 1.174 GB packed, the dominant term in the download
    // (EMBEDDING-QUANT-FINDING.md). Stated in this comment rather than in the note string, because
    // scripts/size-check.mjs keeps src/engine/sizes.ts as the one size table and reads a spoken
    // size in live code as one that escaped it.
    note: '4480 bytes hold 8960 codes, which is 35 layers of 256. One scale per layer group, so the '
      + 'gather for layer L reads column L. Folding the two steps gives EMBEDDING-QUANT-FINDING.md\'s '
      + 'y = 16 * scale * (code - 8). The dominant download.',
  },
  {
    id: 'embed-tokens-2bit',
    example: 'model.language_model.embed_tokens.embedding_quantized',
    dtype: 'U8',
    bits: 2,
    shape: [262144, 384],
    scaleColumns: 1,
    embedScale: EMBED_TOKENS_EMBED_SCALE,
    note: '384 bytes hold 1536 codes, one scale for the whole row, then the forward multiplies by '
      + 'sqrt(1536). Gathered rather than streamed, so its cost is the download and not the token.',
  },
  {
    id: 'lm-head-2bit',
    example: 'lm_head.weight',
    dtype: 'U8',
    bits: 2,
    shape: [262144, 384],
    scaleColumns: 1,
    embedScale: 1,
    note: 'One row per vocabulary entry, one f32 scale per row, so a block major repack for the GEMV '
      + 'is a load time transform of this layout. Both of its activation scales are 0.0, meaning the '
      + 'head is uncalibrated and applies no activation rounding.',
  },
  {
    id: 'attn-q-proj-4bit',
    example: 'model.language_model.layers.0.self_attn.q_proj.weight',
    dtype: 'U8',
    bits: 4,
    shape: [2048, 768],
    scaleColumns: 1,
    embedScale: 1,
    note: '2048 outputs is 8 query heads of head_dim 256, and 768 bytes hold the 1536 inputs. Every '
      + 'attention projection on every layer is 4-bit. Unlike the embedding families the scale is '
      + 'never blocked along the row.',
  },
  {
    id: 'mlp-gate-2bit',
    example: 'model.language_model.layers.15.mlp.gate_proj.weight',
    dtype: 'U8',
    bits: 2,
    shape: [12288, 384],
    scaleColumns: 1,
    embedScale: 1,
    note: 'Layer 15 is the first KV consumer, and consumers double the MLP width to 12288 and drop to '
      + '2-bit. This is the family PREFILL-CAMPAIGN.md measured as instruction bound rather than '
      + 'traffic bound on M1, so the unpack is written for instruction count.',
  },
  {
    id: 'ple-gate-8bit',
    example: 'model.language_model.layers.0.per_layer_input_gate.weight',
    dtype: 'I8',
    bits: 8,
    shape: [256, 1536],
    scaleColumns: 1,
    embedScale: 1,
    note: 'A family ENGINE-PLAN.md section 3 does not list. Signed bytes, no packing, no zero point: '
      + 'the byte is the code. per_layer_projection is the same family transposed at [1536, 256].',
  },
  {
    id: 'norm-weight-bf16',
    example: 'model.language_model.layers.0.input_layernorm.weight',
    dtype: 'BF16',
    bits: null,
    shape: [1536],
    scaleColumns: 0,
    embedScale: 1,
    note: 'Norm weights are never quantized. The norm is x * pow(mean(x^2) + 1e-6, -0.5) * weight '
      + 'with no (1 + weight) form, which is ENGINE-PLAN.md section 10 quirk 4. layer_scalar is the '
      + 'same storage as a single bf16 value per layer.',
  },
  {
    id: 'plmp-bf16',
    example: 'model.language_model.per_layer_model_projection.weight',
    dtype: 'BF16',
    bits: null,
    shape: [8960, 1536],
    scaleColumns: 0,
    embedScale: 1,
    // 8,960 by 1,536 at two bytes is 27.5 MB, the largest unquantized tensor in the download. In
    // this comment rather than the note string for the size-check rule described above.
    note: 'In the checkpoint\'s modules_to_not_convert list, so it stays a plain bf16 linear of '
      + '[8960, 1536] with no weight_scale and no activation scales. It is the context aware half '
      + 'of the PLE path, and it is the largest unquantized tensor we download.',
  },
]);

/** Strip the stored suffix off a tensor name to get the module path resolveQuantBits wants. */
export function modulePathForTensor(tensorName: string): string {
  const suffixes = [
    '.weight_scale',
    '.input_activation_scale',
    '.output_activation_scale',
    '.embedding_quantized',
    '.embedding_scale',
    '.weight',
  ];
  for (const suffix of suffixes) {
    if (tensorName.endsWith(suffix)) return tensorName.slice(0, -suffix.length);
  }
  return tensorName;
}
