// Written from src/engine/gemma4/quant.ts (this project's documented source of truth for the
// stored quant layout) and the measurement record in gemma4-kernels-lab/EMBEDDING-QUANT-FINDING.md
// and PREFILL-CAMPAIGN.md line 145 (2-bit mask 0x03030303, four chunks per word).
// SPDX-License-Identifier: Apache-2.0
//
// The word level view of quant.ts's byte level layout, which is what the matmul shaders consume.
// quant.ts is the contract; this file is its restatement for a shader that reads storage as u32.
// If the two ever disagree, quant.ts is right and this file is the bug, and the k-matmul section
// of scripts/engine-check.mjs compares them numerically on every run so a disagreement is a red
// check rather than a wrong token three kernels later.
//
// How the byte layout composes into u32 words, on the little endian buffers WebGPU mandates:
//
//   4-bit  quant.ts: two codes per byte, low nibble first, unsigned 0..15, zero point 8.
//          As a u32: code i of the word sits at bit offset 4 * i, because byte i/2 contributes
//          bits 8 * (i/2) and the low nibble comes first. Eight codes per word, ascending k.
//   2-bit  quant.ts: four codes per byte at bit offsets 0, 2, 4, 6, unsigned 0..3, zero point 2.
//          As a u32: code i of the word sits at bit offset 8 * (i / 4) + 2 * (i % 4). Sixteen
//          codes per word; byte b holds the four consecutive codes 4 b .. 4 b + 3. The lab's
//          0x03030303 mask with CHUNKS 4 is this same layout read chunk-wise: chunk c lifts code
//          c of each byte, that is k offsets c, c + 4, c + 8, c + 12 at once.
//
// Scales are one f32 per output row for every quantized linear (quant.ts, dequantLinearRow), so
// the shaders apply the scale once per output element, after the K reduction, not per group.
// Per group scales exist only on the embedding tables, which are gather kernels, not matmuls.

/** Codes per u32 word, by bit width. The 8-bit I8 family holds four signed bytes per word. */
export const CODES_PER_WORD = { 2: 16, 4: 8, 8: 4 } as const;

/** u32 words per weight row. K must divide evenly, which every text tensor's K does. */
export function wordsPerRow(bits: 2 | 4 | 8, k: number): number {
  const per = CODES_PER_WORD[bits];
  if (k % per !== 0) throw new Error(`K ${k} does not fill whole ${bits}-bit words`);
  return k / per;
}

/**
 * K span one 32 lane iteration covers when each lane reads one word: 256 codes at 4-bit,
 * 512 at 2-bit. The GEMV loop bound is wordsPerRow / 32, so K must be a multiple of this.
 */
export const K_SPAN = { 2: 512, 4: 256 } as const;

/**
 * WGSL snippet: unpack the 8 codes of a 4-bit word `w` into two vec4<f32> of (code - 8).
 * `wLo` holds k offsets 0..3, `wHi` holds 4..7. The subtraction bakes in quant.ts's zero
 * point of 8, and the dots downstream run through the dot() intrinsic because that is the
 * shape whose rounding held bit identical across modules all evening while scalar multiply
 * add chains drifted (DECODE-FUSION-FINDING.md, ENGINE-PLAN.md risk 2).
 */
export const WGSL_UNPACK4 = `
    let sh4 = vec4<u32>(0u, 4u, 8u, 12u);
    let wLo = vec4<f32>((vec4<u32>(w) >> sh4) & vec4<u32>(0xfu)) - vec4<f32>(8.0);
    let wHi = vec4<f32>((vec4<u32>(w >> 16u) >> sh4) & vec4<u32>(0xfu)) - vec4<f32>(8.0);
`;

/**
 * WGSL snippet: unpack the 16 codes of a 2-bit word `w` into four vec4<f32> of (code - 2),
 * byte by byte so each vec4 covers four consecutive k offsets: `wB0` is k 0..3 (byte 0),
 * `wB1` is 4..7, `wB2` is 8..11, `wB3` is 12..15. Zero point 2 per quant.ts. The unpack is
 * written for instruction count because the 2-bit path is instruction bound on M1, not
 * traffic bound: the retile campaign measured about 6 unpack, convert and srq ops per 8 FLOP
 * dot at 15 to 20 percent of scalar ALU peak (PREFILL-CAMPAIGN.md, retile campaign result).
 * Three scalar shifts, four vector shifts, four vector masks and four converts serve 32 FLOPs
 * of dot work. The banked next step if this is still the M1 bottleneck is packed integer dot
 * products, which is not bit identical and ships behind the ENGINE-PLAN section 7 fallback
 * gate, not this round.
 */
export const WGSL_UNPACK2 = `
    let sh2 = vec4<u32>(0u, 2u, 4u, 6u);
    let wB0 = vec4<f32>((vec4<u32>(w) >> sh2) & vec4<u32>(3u)) - vec4<f32>(2.0);
    let wB1 = vec4<f32>((vec4<u32>(w >> 8u) >> sh2) & vec4<u32>(3u)) - vec4<f32>(2.0);
    let wB2 = vec4<f32>((vec4<u32>(w >> 16u) >> sh2) & vec4<u32>(3u)) - vec4<f32>(2.0);
    let wB3 = vec4<f32>((vec4<u32>(w >> 24u) >> sh2) & vec4<u32>(3u)) - vec4<f32>(2.0);
`;

/**
 * WGSL snippet: the 4-bit unpack kept in the integer domain, for the SRQ integer dot path.
 * Same word layout and zero point as WGSL_UNPACK4, but the codes stay vec4<i32> so the dot
 * against int8 activation codes runs through WGSL's integer dot() and the K accumulation is
 * exact integer arithmetic, which is the round 2 ratified semantics for a calibrated site.
 * The k-matmul check proves both domains against quant.ts on the same bytes.
 */
export const WGSL_UNPACK4_I = `
    let sh4 = vec4<u32>(0u, 4u, 8u, 12u);
    let wLoI = vec4<i32>((vec4<u32>(w) >> sh4) & vec4<u32>(0xfu)) - vec4<i32>(8);
    let wHiI = vec4<i32>((vec4<u32>(w >> 16u) >> sh4) & vec4<u32>(0xfu)) - vec4<i32>(8);
`;

/**
 * WGSL snippet: unpack the 4 codes of an 8-bit I8 word `w` into one vec4<f32>. quant.ts stores
 * this family as signed bytes with no packing and zero point 0, so the byte IS the code and the
 * unpack is a sign extension: shift each byte to the top and arithmetic shift it back down, which
 * `>>` does on i32 by definition. Byte b of the little endian word is k offset b, ascending, same
 * as the other two families. No subtraction, because ZERO_POINT[8] is 0 and not 128.
 */
export const WGSL_UNPACK8 = `
    let sh8 = vec4<u32>(24u, 16u, 8u, 0u);
    let w8 = vec4<f32>(bitcast<vec4<i32>>(vec4<u32>(w) << sh8) >> vec4<u32>(24u));
`;

/** WGSL snippet: the 8-bit unpack in the integer domain, companion to WGSL_UNPACK8. */
export const WGSL_UNPACK8_I = `
    let sh8 = vec4<u32>(24u, 16u, 8u, 0u);
    let w8i = bitcast<vec4<i32>>(vec4<u32>(w) << sh8) >> vec4<u32>(24u);
`;

/** WGSL snippet: the 2-bit unpack in the integer domain, companion to WGSL_UNPACK2. */
export const WGSL_UNPACK2_I = `
    let sh2 = vec4<u32>(0u, 2u, 4u, 6u);
    let wB0i = vec4<i32>((vec4<u32>(w) >> sh2) & vec4<u32>(3u)) - vec4<i32>(2);
    let wB1i = vec4<i32>((vec4<u32>(w >> 8u) >> sh2) & vec4<u32>(3u)) - vec4<i32>(2);
    let wB2i = vec4<i32>((vec4<u32>(w >> 16u) >> sh2) & vec4<u32>(3u)) - vec4<i32>(2);
    let wB3i = vec4<i32>((vec4<u32>(w >> 24u) >> sh2) & vec4<u32>(3u)) - vec4<i32>(2);
`;

/**
 * CPU mirror of the six WGSL snippets above, which are three layouts in two arithmetic domains and
 * not six layouts, bit for bit: given packed words, return the
 * signed codes in k order, exactly as the shader's vec4 lanes see them. The k-matmul check
 * compares this against quant.ts's unpackCodes over the same bytes, which is the mechanical
 * proof that the shader layout and the loader layout agree. Kept in this file, beside the
 * snippets it mirrors, so the two cannot drift without the diff showing them side by side.
 * The integer domain snippets produce the same codes by construction, only as i32, and the
 * check asserts the two WGSL texts share their shift constants.
 */
export function unpackWordsLikeShader(bits: 2 | 4 | 8, words: Uint32Array, count: number): Int8Array {
  const out = new Int8Array(count);
  if (bits === 8) {
    for (let i = 0; i < count; i++) {
      const byte = (words[i >> 2] >>> ((i & 3) * 8)) & 0xff;
      out[i] = byte > 127 ? byte - 256 : byte;
    }
    return out;
  }
  if (bits === 4) {
    for (let i = 0; i < count; i++) {
      const w = words[i >> 3];
      out[i] = ((w >>> ((i & 7) * 4)) & 0xf) - 8;
    }
    return out;
  }
  for (let i = 0; i < count; i++) {
    const w = words[i >> 4];
    const inWord = i & 15;
    out[i] = ((w >>> ((inWord >> 2) * 8 + (inWord & 3) * 2)) & 0x3) - 2;
  }
  return out;
}

/** View packed bytes as the u32 words the shader binds. Copies when the view is unaligned. */
export function bytesAsWords(bytes: Uint8Array): Uint32Array {
  if ((bytes.byteOffset & 3) === 0 && (bytes.byteLength & 3) === 0) {
    return new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength >> 2);
  }
  const padded = new Uint8Array((bytes.byteLength + 3) & ~3);
  padded.set(bytes);
  return new Uint32Array(padded.buffer);
}

// ---------------------------------------------------------------------------------------------
// The row interleaved layout of the 2-bit family, docs/ENGINE-PERF.md section 13.
// ---------------------------------------------------------------------------------------------
//
// The row layout above is the checkpoint's own and stays what the loader downloads, checksums
// and caches. What the 2-bit kernels READ is a repack of it done on the GPU at load: rows are
// grouped in tiles of TILE_ROWS consecutive rows, a tile holds K words, one per k, and word k of
// tile t carries the code of row 16 t + r at bit offset 2 r. The codes themselves are the same
// unsigned 0..3 with zero point 2; only their address moves. A word therefore meets ONE scalar
// activation in a matmul, which is what lets two rows share one integer multiply in the 16-bit
// halves of a u32 (qgemv.ts). Rows pad to a whole tile with zero codes, and every consumer
// guards its stores by the true row count. The 4-bit and 8-bit families are untouched.

/** Rows per tile of the interleaved 2-bit layout. */
export const TILE_ROWS = 16;

/** u32 words a 2-bit tensor of `rows` by `k` occupies in the interleaved layout. */
export function tile16Words(rows: number, k: number): number {
  if (k % 16 !== 0) throw new Error(`tile16Words: k must be a multiple of 16, got ${k}`);
  return Math.ceil(rows / TILE_ROWS) * k;
}

/**
 * The CPU mirror of the GPU repack (kernels/repack.ts): the row layout's words of a 2-bit
 * tensor to the interleaved layout's words. The kernel sweep holds the GPU to this at bit
 * identity, and the 2-bit oracles read their fixtures back through tile16ToRowWords, so the
 * expected values are the same numbers the row layout produced.
 */
export function rowWordsToTile16(rowWords: Uint32Array, rows: number, k: number): Uint32Array {
  const kWords = k / 16;
  if (rowWords.length < rows * kWords) {
    throw new Error(`rowWordsToTile16: ${rowWords.length} words hold fewer than ${rows} rows of k ${k}`);
  }
  const out = new Uint32Array(tile16Words(rows, k));
  for (let row = 0; row < rows; row += 1) {
    const tile = Math.floor(row / TILE_ROWS);
    const shift = 2 * (row % TILE_ROWS);
    const dstBase = tile * k;
    const srcBase = row * kWords;
    for (let w = 0; w < kWords; w += 1) {
      const word = rowWords[srcBase + w]!;
      for (let i = 0; i < 16; i += 1) {
        const code = (word >>> (2 * i)) & 3;
        out[dstBase + w * 16 + i] = (out[dstBase + w * 16 + i]! | (code << shift)) >>> 0;
      }
    }
  }
  return out;
}

/** The inverse of rowWordsToTile16, for the oracles: interleaved words back to row words. */
export function tile16ToRowWords(tileWords: Uint32Array, rows: number, k: number): Uint32Array {
  const kWords = k / 16;
  if (tileWords.length < tile16Words(rows, k)) {
    throw new Error(`tile16ToRowWords: ${tileWords.length} words hold fewer than ${rows} rows of k ${k}`);
  }
  const out = new Uint32Array(rows * kWords);
  for (let row = 0; row < rows; row += 1) {
    const tile = Math.floor(row / TILE_ROWS);
    const shift = 2 * (row % TILE_ROWS);
    const srcBase = tile * k;
    const dstBase = row * kWords;
    for (let kk = 0; kk < k; kk += 1) {
      const code = (tileWords[srcBase + kk]! >>> shift) & 3;
      const w = kk >> 4;
      out[dstBase + w] = (out[dstBase + w]! | (code << (2 * (kk & 15)))) >>> 0;
    }
  }
  return out;
}

/** One code of an interleaved 2-bit tensor, unsigned 0..3, by row and k. */
export function tile16Code(tileWords: Uint32Array, k: number, row: number, kk: number): number {
  return (tileWords[Math.floor(row / TILE_ROWS) * k + kk]! >>> (2 * (row % TILE_ROWS))) & 3;
}
