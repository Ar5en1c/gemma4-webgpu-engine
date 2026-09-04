// Written from docs/ENGINE-PLAN.md sections 5 (kernel K9) and 6, the layout in ../kv.ts, and the
// WGSL specification. No vendored bundle, no extracted kernel and no third party engine source
// was read.
//
// SPDX-License-Identifier: Apache-2.0
//
// The KV cache store, kernel K9. Writes new K and V rows, already per head normed and, for K,
// already rotated (the store runs after K6 and K7 in the producer chain), into the packed cache
// buffer that kv.ts describes: K region at position * headDim, V region at vBase + the same.
//
// The measured incumbent stages K and V and then issues about 30 strided copies per token
// (PREFILL-CAMPAIGN.md). This engine writes into the slot directly instead, which ENGINE-PLAN K9
// is explicit about: that is a design freedom our own cache layout buys, not a fusion
// optimisation. One dispatch stores a whole prefill chunk or a single decode row; `startPos` is
// the absolute position of the first incoming row, so the same kernel serves prefill (chunk at
// the prefill cursor) and decode (one row at the current position), and a rewind needs no
// clearing pass because overwriting the slot is the only write this layout ever does.
//
// There is no reduction anywhere in this kernel, so the subgroup rules of ENGINE-PLAN 5.5 are
// satisfied by having nothing to apply them to. The loop bound is the uniform's headDim4, runtime
// opaque, per rule 6. Three storage buffers plus one uniform, against the adapter budget of 10.
//
// V is stored as f32 in v1. ENGINE-PLAN section 6 banks the option of caching V as its quantized
// codes (bit identical reconstruction, four times less V traffic); that lands here as a second
// variant writing codes plus scale to a narrower V region, and the layout in kv.ts already
// carries the stride per cache so nothing outside this file and kv.ts would move.

import type { Kernel, KernelBindInput, KernelBindResult } from './registry';
import { kernelLayout, kernelUniform } from './binding';

/** 64 wide, the engine's one workgroup width (DECODE-CAMPAIGN.md 4.2). */
const WORKGROUP_SIZE = 64;

export const KV_STORE_WGSL = /* wgsl */ `
struct KvStoreParams {
  // Row width in vec4 lanes: 64 sliding, 128 global.
  headDim4: u32,
  // Absolute position of incoming row 0. The prefill cursor, or the decode position.
  startPos: u32,
  // Incoming rows. The dispatch is one workgroup per row and the caller sizes both to match.
  tokenCount: u32,
  // V region offset in vec4 lanes: maxContext * headDim / 4, from the kv.ts layout.
  vBase4: u32,
}

@group(0) @binding(0) var<storage, read> kNew: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> vNew: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> cache: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> params: KvStoreParams;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_index) lid: u32,
) {
  // One workgroup per incoming row. No bounds guard on the row: the dispatch equals tokenCount
  // exactly, and the in-row loop bound below is uniform, so control flow stays uniform.
  let row = wid.x;
  let n4 = params.headDim4;
  let src = row * n4;
  let kDst = (params.startPos + row) * n4;
  let vDst = params.vBase4 + kDst;
  for (var i = lid; i < n4; i = i + ${WORKGROUP_SIZE}u) {
    cache[kDst + i] = kNew[src + i];
    cache[vDst + i] = vNew[src + i];
  }
}
`;

// ---------------------------------------------------------------------------------------------
// The CPU oracle.
// ---------------------------------------------------------------------------------------------

export interface KvStoreShape {
  headDim: number;
  startPos: number;
  tokenCount: number;
  maxContext: number;
}

/**
 * Reference for the store: writes k and v rows into a packed cache image and returns it. Pass
 * `into` to model consecutive stores against one cache (the append case); omitted, the cache
 * starts zeroed, which is also what the harness's fresh output buffer holds.
 *
 * A pure copy, exact by construction, so the kernel's tolerance against this is zero: a store
 * kernel that is off by one ULP is a store kernel reading the wrong slot.
 */
export function kvStoreOracle(
  k: Float32Array,
  v: Float32Array,
  shape: KvStoreShape,
  into?: Float32Array,
): Float32Array {
  const { headDim, startPos, tokenCount, maxContext } = shape;
  if (startPos < 0 || tokenCount < 0 || startPos + tokenCount > maxContext) {
    throw new Error(
      `kvStoreOracle: rows ${startPos}..${startPos + tokenCount - 1} outside 0..${maxContext - 1}`,
    );
  }
  if (k.length < tokenCount * headDim || v.length < tokenCount * headDim) {
    throw new Error('kvStoreOracle: k or v shorter than tokenCount * headDim');
  }
  const vBase = maxContext * headDim;
  const cache = into ?? new Float32Array(2 * vBase);
  if (cache.length !== 2 * vBase) {
    throw new Error(`kvStoreOracle: cache holds ${cache.length}, layout says ${2 * vBase}`);
  }
  for (let t = 0; t < tokenCount; t += 1) {
    cache.set(k.subarray(t * headDim, (t + 1) * headDim), (startPos + t) * headDim);
    cache.set(v.subarray(t * headDim, (t + 1) * headDim), vBase + (startPos + t) * headDim);
  }
  return cache;
}

// ---------------------------------------------------------------------------------------------
// Registry entry.
// ---------------------------------------------------------------------------------------------

/** The params block, exported so the scheduler can restage it in place per dispatch. */
export function storeParams(headDim4: number, startPos: number, tokenCount: number, vBase4: number): ArrayBuffer {
  const words = new ArrayBuffer(16);
  const u = new Uint32Array(words);
  u[0] = headDim4;
  u[1] = startPos;
  u[2] = tokenCount;
  u[3] = vBase4;
  return words;
}

function bindStore(input: KernelBindInput): KernelBindResult {
  const { device, inputs, output, params } = input;
  const headDim = params.headDim | 0;
  const startPos = params.startPos | 0;
  const tokenCount = params.tokenCount | 0;
  const maxContext = params.maxContext | 0;
  if (headDim <= 0 || headDim % 4 !== 0) {
    throw new Error(`kv-cache-store needs params.headDim a positive multiple of 4, got ${headDim}`);
  }
  if (tokenCount <= 0) throw new Error('kv-cache-store needs params.tokenCount > 0');
  if (startPos < 0 || startPos + tokenCount > maxContext) {
    throw new Error('kv-cache-store: startPos + tokenCount exceeds maxContext');
  }
  const k = inputs.k;
  const v = inputs.v;
  if (!k || !v) throw new Error('kv-cache-store needs inputs named k and v');

  const layout = kernelLayout(input, 'kv-cache-store', () => device.createBindGroupLayout({
    label: 'kv-cache-store',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  }));
  const uniform = kernelUniform(
    input,
    'kv-cache-store params',
    storeParams(headDim / 4, startPos, tokenCount, (maxContext * headDim) / 4),
  );

  return {
    layout,
    buffers: [k, v, output, uniform.binding],
    dispatch: [tokenCount, 1, 1],
    dispose: uniform.dispose,
  };
}

export const kvCacheStoreKernel: Kernel = {
  name: 'kv-cache-store',
  wgsl: KV_STORE_WGSL,
  entry: 'main',
  note:
    'K9. Direct slot store of post-norm post-rope K and weightless-normed V rows into the packed '
    + 'per producer cache of kv.ts. Prefill chunk and decode row are the same kernel at a '
    + 'different startPos.',
  cases: [
    {
      name: 'prefill-sliding-256',
      inputs: { k: 'attn.short-question.layer00.k', v: 'attn.short-question.layer00.v' },
      expected: 'kattn.cache.layer00',
      params: { headDim: 256, startPos: 0, tokenCount: 16, maxContext: 16 },
      tolAbs: 0,
      note: 'Whole 16 token prompt into an exactly sized cache. Exact copy, zero tolerance.',
    },
    {
      name: 'prefill-global-512',
      inputs: { k: 'attn.short-question.layer04.k', v: 'attn.short-question.layer04.v' },
      expected: 'kattn.cache.layer04',
      params: { headDim: 512, startPos: 0, tokenCount: 16, maxContext: 16 },
      tolAbs: 0,
      note: 'Same store at the global head dimension of 512. The stride is per cache, not 256.',
    },
    {
      name: 'append-at-offset',
      inputs: { k: 'kattn.append.k', v: 'kattn.append.v' },
      expected: 'kattn.cache.append',
      params: { headDim: 256, startPos: 12, tokenCount: 4, maxContext: 24 },
      tolAbs: 0,
      // outputInit 'zero' because this case deliberately writes only 2048 of the 12288 cache
      // elements and asserts the other 10240 hold zero. The harness's default sentinel fill and
      // that assertion are mutually exclusive, which round 1's verifier proved on hardware: every
      // untouched slot read back as the sentinel bit pattern, -6.267e18 as f32, never as zero.
      // With a zeroed output the silent no-dispatch guard is the comparison itself, since the
      // expected buffer's 2048 written elements are not all zero.
      outputInit: 'zero',
      note:
        'Four rows landing at positions 12..15 of a larger cache, the decode append shape. '
        + 'Slots outside the appended range stay zero, matching the zero initialized cache the '
        + 'case declares via outputInit.',
    },
  ],
  bind: bindStore,
};
