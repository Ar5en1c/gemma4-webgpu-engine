// Written from docs/ENGINE-PLAN.md sections 2, 3 and 5 (the dtype and residency policy) and
// DECODE-CAMPAIGN.md 1 and 4.3, plus the WGSL specification's data packing builtins and IEEE 754
// binary16. No vendored bundle, no extracted kernel and no third party engine source was read.
//
// SPDX-License-Identifier: Apache-2.0
//
// The cast between f16 storage and f32 compute, both directions, as two kernels.
//
// THE POLICY THIS IMPLEMENTS, stated plainly because it is easy to read more into it than is there.
// Compute and every reference activation in this engine are f32: the reference data contract says
// activations are f32 whatever the kernel computes in, the GEMV reads its activation vector as
// `vec4<f32>` and writes f32, and the norms accumulate in f32. `shader-f16` is a precondition for
// the engine rather than a fallback question, because `checkWebGpu()` refuses before any download
// starts without it, but that feature buys f16 *arithmetic* and this file is not about arithmetic.
// It is about storage: a buffer that is written once and read once, or read many times by a kernel
// that immediately widens it, can be held as f16 and halve its traffic. Decode on the M1 is
// streaming bound at a measured 51 GB/s envelope with about 0.6 GB moving per token
// (DECODE-CAMPAIGN.md 1), so halving a buffer's traffic is the only kind of saving that shows up
// there at all.
//
// NOTHING IN THE ENGINE IS COMMITTED TO F16 STORAGE YET, and this file does not commit it. The
// candidate is the KV cache, and the plan's own banked idea for that is sharper than f16: V rows
// arrive already snapped to an int8 grid by the activation quantize, so storing codes and a scale
// is bit identical and four times smaller, while K does not qualify because norm and RoPE run after
// the quantize (DECODE-CAMPAIGN.md 4.3, ENGINE-PLAN section 6). f16 is the general purpose fallback
// for a staging buffer that has no such structure. A site adopts it with a measurement, and these
// two kernels are what it adopts.
//
// WHY `pack2x16float` RATHER THAN AN `f16` TYPED BUFFER. The two builtins used here are core WGSL
// and need no enable directive, so the cast path compiles on a device that never got `shader-f16`,
// which keeps the storage decision independent of the arithmetic one. It also keeps the storage
// layout explicit: two halves packed low first into one `u32`, which is exactly what a later kernel
// reading the buffer as `array<u32>` and unpacking in registers will see.
//
// ROUNDING, and the one thing the GPU cases deliberately do not assert. f32 to f16 loses bits, and
// the tie rule is the place where implementations differ in principle: this file's oracle rounds to
// nearest with ties to even, which is IEEE 754's default and what `pack2x16float` is expected to
// do, but a tie is one ULP of a 10 bit mantissa and it is not worth building a gate that could go
// amber for a reason that is not a bug. So the GPU fixtures hold no ties: half the values are
// exactly representable in f16 and cast losslessly, and the rest are at least a quarter of an ULP
// away from any halfway point, so the nearest value is unambiguous and both cases gate at bit
// identity. The tie behaviour itself is exercised against the oracle in `scripts/engine-check.mjs`
// section k-mlp, on the CPU, where the rule is ours to state.

import type { Kernel, KernelBindInput, KernelBindResult } from './registry';
import { kernelLayout, kernelUniform } from './binding';

/** 64 wide, the width the rest of the engine uses (DECODE-CAMPAIGN.md 4.2). Nothing reduces here. */
const WORKGROUP_SIZE = 64;

export const F16_PACK_WGSL = /* wgsl */ `
struct CastParams {
  // Number of packed words, so half the f32 count. Runtime opaque, per the registry's loop bound
  // rule, even though this kernel has no loop: the guard reads it and a constant here would be a
  // module per buffer size.
  wordCount: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
}

@group(0) @binding(0) var<storage, read> src: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> params: CastParams;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.wordCount) {
    return;
  }
  // Low half first, which is what unpack2x16float reverses and what a reader of this buffer as
  // array<u32> has to assume.
  dst[i] = pack2x16float(src[i]);
}
`;

export const F16_UNPACK_WGSL = /* wgsl */ `
struct CastParams {
  wordCount: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
}

@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: CastParams;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.wordCount) {
    return;
  }
  // Widening is exact: every binary16 is a binary32, so this direction has no tolerance question
  // at all and its case gates at bit identity.
  dst[i] = unpack2x16float(src[i]);
}
`;

// ---------------------------------------------------------------------------------------------
// The CPU oracles.
// ---------------------------------------------------------------------------------------------

const castView = new DataView(new ArrayBuffer(4));

/**
 * One f32 to its IEEE 754 binary16 bit pattern, round to nearest, ties to even.
 *
 * Written out rather than delegated to `Math.f16round`, which this repo's Node does not have, and
 * rather than to a `Float16Array`, for the same reason. The k-mlp check cross checks against both
 * when the runtime happens to provide them, so the day Node grows them this becomes a checked
 * implementation rather than an unchecked one.
 *
 * Overflow saturates to an infinity and underflow to a signed zero, which is what the hardware
 * does. The engine's own use never approaches either, because activations in this model live within
 * a couple of orders of magnitude of 1 and f16 reaches 65504.
 */
export function f32ToF16Bits(value: number): number {
  castView.setFloat32(0, value, true);
  const x = castView.getUint32(0, true);
  const sign = (x >>> 31) << 15;
  const exp = (x >>> 23) & 0xff;
  const mant = x & 0x7fffff;

  if (exp === 0xff) {
    // Infinity keeps its sign; a NaN stays a NaN with a set quiet bit, since a NaN payload is not
    // something binary16 can carry and is not something this engine relies on.
    return (sign | 0x7c00 | (mant !== 0 ? 0x200 : 0)) & 0xffff;
  }

  // 127 - 15 is the exponent bias difference, so this is the binary16 biased exponent.
  const e = exp - 112;
  if (e >= 0x1f) return (sign | 0x7c00) & 0xffff;

  if (e > 0) {
    let bits = sign | (e << 10) | (mant >>> 13);
    const rem = mant & 0x1fff;
    // Ties to even. A carry out of the mantissa runs into the exponent field on its own, which is
    // the correct answer and the reason this is an add rather than a field write.
    if (rem > 0x1000 || (rem === 0x1000 && ((mant >>> 13) & 1) === 1)) bits += 1;
    return bits & 0xffff;
  }

  // Subnormal, or smaller than the smallest subnormal. e < -10 cannot round up to anything.
  if (e < -10) return sign & 0xffff;
  const m = mant | 0x800000;
  const shift = 14 - e;
  let half = m >>> shift;
  const rem = m & ((1 << shift) - 1);
  const halfway = 1 << (shift - 1);
  if (rem > halfway || (rem === halfway && (half & 1) === 1)) half += 1;
  return (sign | half) & 0xffff;
}

/** One binary16 bit pattern back to f32. Exact in this direction, always. */
export function f16BitsToF32(bits: number): number {
  const sign = (bits & 0x8000) !== 0 ? -1 : 1;
  const exp = (bits >>> 10) & 0x1f;
  const mant = bits & 0x3ff;
  if (exp === 0) return sign * mant * 2 ** -24;
  if (exp === 0x1f) return mant === 0 ? sign * Infinity : NaN;
  return sign * (1 + mant / 1024) * 2 ** (exp - 15);
}

/** What `f32ToF16Bits` then `f16BitsToF32` does to a value, which is what a round trip costs. */
export function f16Round(value: number): number {
  return f16BitsToF32(f32ToF16Bits(value));
}

/**
 * The known answer reference for `f16-pack`: `count` f32 values to `count / 2` packed words, low
 * half of each word first. Returned as i32 because that is what the manifest carries for raw 32 bit
 * words and what the harness reads back.
 */
export function f16PackOracle(values: Float32Array, count: number): Int32Array {
  if (count % 2 !== 0) throw new Error(`f16PackOracle: count ${count} is not a pair count`);
  if (values.length < count) {
    throw new Error(`f16PackOracle: src holds ${values.length}, needs ${count}`);
  }
  const out = new Int32Array(count / 2);
  for (let i = 0; i < count / 2; i += 1) {
    const lo = f32ToF16Bits(values[2 * i]);
    const hi = f32ToF16Bits(values[2 * i + 1]);
    out[i] = ((hi << 16) | lo) | 0;
  }
  return out;
}

/** The known answer reference for `f16-unpack`, the exact direction. */
export function f16UnpackOracle(words: Int32Array, wordCount: number): Float32Array {
  if (words.length < wordCount) {
    throw new Error(`f16UnpackOracle: src holds ${words.length}, needs ${wordCount}`);
  }
  const out = new Float32Array(wordCount * 2);
  for (let i = 0; i < wordCount; i += 1) {
    const w = words[i] >>> 0;
    out[2 * i] = f16BitsToF32(w & 0xffff);
    out[2 * i + 1] = f16BitsToF32(w >>> 16);
  }
  return out;
}

// ---------------------------------------------------------------------------------------------
// Registry entries.
// ---------------------------------------------------------------------------------------------

/** The params block, exported so the scheduler can restage it in place per dispatch. */
export function castParams(wordCount: number): ArrayBuffer {
  const words = new ArrayBuffer(16);
  const u = new Uint32Array(words);
  u[0] = wordCount;
  u[1] = 0;
  u[2] = 0;
  u[3] = 0;
  return words;
}

function bindCast(label: string) {
  return (input: KernelBindInput): KernelBindResult => {
    const { device, inputs, output, params } = input;
    const wordCount = params.wordCount | 0;
    if (wordCount <= 0) throw new Error(`${label} needs params.wordCount`);
    const src = inputs.src;
    if (!src) throw new Error(`${label} needs an input named src`);

    const layout = kernelLayout(input, label, () => device.createBindGroupLayout({
      label,
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    }));
    const uniform = kernelUniform(input, `${label} params`, castParams(wordCount));

    return {
      layout,
      buffers: [src, output, uniform.binding],
      dispatch: [Math.ceil(wordCount / WORKGROUP_SIZE), 1, 1],
      dispose: uniform.dispose,
    };
  };
}

export const f16PackKernel: Kernel = {
  name: 'f16-pack',
  wgsl: F16_PACK_WGSL,
  entry: 'main',
  note:
    'f32 compute to f16 storage, two halves per word, low half first. Core WGSL packing builtins, '
    + 'so the cast path needs no shader-f16 and the storage decision stays independent of the '
    + 'arithmetic one.',
  cases: [
    {
      name: 'exact-1024',
      inputs: { src: 'kmlp.f16.exact.f32' },
      expected: 'kmlp.f16.exact.packed',
      params: { wordCount: 512 },
      tolAbs: 0,
      note:
        'Every value is exactly representable in f16, so the cast is lossless and the gate is bit '
        + 'identity with no rounding rule involved at all.',
    },
    {
      name: 'rounded-1024',
      inputs: { src: 'kmlp.f16.rounded.f32' },
      expected: 'kmlp.f16.rounded.packed',
      params: { wordCount: 512 },
      tolAbs: 0,
      note:
        'Values that need rounding but sit at least a quarter of an ULP from any halfway point, so '
        + 'the nearest f16 is unambiguous and the gate is still bit identity. Ties are checked on '
        + 'the CPU, where the rule is ours to state.',
    },
  ],
  bind: bindCast('f16-pack'),
};

export const f16UnpackKernel: Kernel = {
  name: 'f16-unpack',
  wgsl: F16_UNPACK_WGSL,
  entry: 'main',
  note:
    'f16 storage back to f32 compute. Exact in this direction, so it gates at bit identity on any '
    + 'input, including the rounded fixture.',
  cases: [
    {
      name: 'widen-512-words',
      inputs: { src: 'kmlp.f16.rounded.packed' },
      expected: 'kmlp.f16.rounded.widened',
      params: { wordCount: 512 },
      tolAbs: 0,
      tolUlp: 0,
      note: 'The packed fixture widened back, which also proves the two kernels agree on the layout.',
    },
  ],
  bind: bindCast('f16-unpack'),
};
