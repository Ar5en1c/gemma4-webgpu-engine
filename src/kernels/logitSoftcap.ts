// Written from docs/ENGINE-PLAN.md section 5 (kernel K14 and the architecture contract table),
// .engine-ref/config.json, the Hugging Face transformers reference implementation of the Gemma 4
// causal LM head, the ablation ledger in DECODE-CAMPAIGN.md 5.1, and the WGSL specification. No
// vendored bundle, no extracted kernel and no third party engine source was read.
//
// SPDX-License-Identifier: Apache-2.0
//
// K14, the final logit softcap: `tanh(logits / cap) * cap`, over the 262,144 wide logit vector.
//
// THE CONSTANT IS IN THE CONFIG AND IS READ FROM IT. `config.json` gives
// `text_config.final_logit_softcapping` as 30.0, and there is no attention logit softcapping in
// this text model. The number below is the default this engine ships with, the loader passes the
// value it actually read, and the k-mlp check asserts the two agree with the config file rather
// than letting a constant here quietly outlive a config change.
//
// WHY IMPLEMENT IT AT ALL, since greedy decode cannot see it. `tanh` is strictly monotonic, so the
// cap cannot move an argmax and this engine's own product behaviour would be identical without it.
// Three reasons it ships anyway, in the order they matter. Parity against the reference logits
// needs it, and gate 1 of ENGINE-PLAN section 7 compares logits, not just tokens: the reference
// dump's `<probe>.logits-last` is captured after the cap, so an engine without it disagrees with
// the reference by up to the cap itself on the largest logits. A sampling mode, which is out of
// scope for v1 and not out of scope forever, is not monotone invariant and would silently change
// distribution without it. And an uncapped logit is a larger number going into any future softmax,
// which is a numerical hazard for nothing gained.
//
// WHAT THIS FILE DELIBERATELY DOES NOT CONTAIN. The final RMS norm that feeds `lm_head` is
// `model.language_model.norm`, a plain 1536 wide Gemma 4 RMS norm, and it is dispatched as
// rmsNorm.ts's `rms-norm` kernel with one row, width 1536 and that tensor as the gain. There is no
// second norm module here. ENGINE-PLAN risk 2 mitigation 4 is explicit that a scalar multiply-add
// chain belongs in exactly one module so there is no second module to disagree with it, and a
// "final norm" kernel that was a copy of the norm kernel with a different name would be exactly the
// second module. What this lane owns instead is the proof that the composition is right: the k-mlp
// check runs the final norm oracle into the 2-bit head into this cap and compares the result
// against the reference logits from the checkpoint itself.
//
// COST. The cap is one pass over 262,144 f32, about 1 MB read and 1 MB written per token, against
// the roughly 0.6 GB per token this engine streams (DECODE-CAMPAIGN.md 1). That is 0.3 percent, and
// the head that produced the vector is itself only +0.23 ms on the 5070 ablation
// (DECODE-CAMPAIGN.md 5.1). Nothing here is worth fusing into the head; if a later round wants to
// anyway, the fused form is one line inside the GEMV's store guard and the argument for it has to
// be a measurement, not this comment.

import type { Kernel, KernelBindInput, KernelBindResult } from './registry';
import { kernelLayout, kernelUniform } from './binding';

/** 64 wide, the width the rest of the engine uses (DECODE-CAMPAIGN.md 4.2). Nothing reduces here. */
const WORKGROUP_SIZE = 64;

/**
 * `text_config.final_logit_softcapping` from config.json. The default, not the authority: the
 * loader reads the config and passes the value, and a config that carries no cap passes 0, which
 * this kernel treats as "no cap" rather than as a division by zero.
 */
export const FINAL_LOGIT_SOFTCAP = 30;

/** Vocabulary, which is the logit vector's length (config.json, and the lm_head row count). */
export const LOGIT_COUNT = 262144;

export const LOGIT_SOFTCAP_WGSL = /* wgsl */ `
struct SoftcapParams {
  // Logit count in vec4 lanes, 65536 at this vocabulary. Runtime opaque, per the registry's loop
  // bound rule, and it is what lets one compiled module serve a shortened test vector.
  vec4Count: u32,
  // The cap. Exactly 0.0 means the config carried none, and then this kernel is a copy.
  cap: f32,
  pad0: u32,
  pad1: u32,
}

@group(0) @binding(0) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: SoftcapParams;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.vec4Count) {
    return;
  }
  let v = src[i];
  // Uniform branch: the cap comes from the uniform block, so every lane takes the same side.
  if (params.cap == 0.0) {
    dst[i] = v;
    return;
  }
  // Divide, tanh, multiply, in that order, which is the order the reference applies them. tanh
  // saturates rather than overflowing, so no clamp is needed here: the argument is a logit over 30
  // and the largest logits in the reference dump are single digits after the cap.
  dst[i] = tanh(v / params.cap) * params.cap;
}
`;

// ---------------------------------------------------------------------------------------------
// The CPU oracle.
// ---------------------------------------------------------------------------------------------

/**
 * The known answer reference for the cap. f64 inside, rounded once per element.
 *
 * A cap of 0, meaning a config with no `final_logit_softcapping`, copies. That mirrors the
 * reference, which skips the three lines entirely when the field is None, and it is the reason the
 * shader's branch exists.
 */
export function softcapOracle(
  logits: Float32Array,
  cap: number = FINAL_LOGIT_SOFTCAP,
  out?: Float32Array,
): Float32Array {
  const result = out && out.length >= logits.length
    ? out.subarray(0, logits.length)
    : new Float32Array(logits.length);
  if (!Number.isFinite(cap) || cap === 0) {
    result.set(logits);
    return result;
  }
  for (let i = 0; i < logits.length; i += 1) {
    result[i] = Math.tanh(logits[i] / cap) * cap;
  }
  return result;
}

/**
 * The inverse, for reading a capped dump back onto the raw logit scale.
 *
 * Used by the k-mlp check to say how far inside the cap the reference logits sit, which is the
 * cheap way to notice that a dump was captured without the cap: uncapped Gemma 4 logits run well
 * past 30, and every capped value is strictly inside it. Not used by the engine.
 */
export function softcapInverse(capped: number, cap: number = FINAL_LOGIT_SOFTCAP): number {
  if (!Number.isFinite(cap) || cap === 0) return capped;
  return Math.atanh(capped / cap) * cap;
}

// ---------------------------------------------------------------------------------------------
// Registry entry.
// ---------------------------------------------------------------------------------------------

/** The params block, exported so the scheduler can restage it in place per dispatch. */
export function softcapParams(vec4Count: number, cap: number): ArrayBuffer {
  const words = new ArrayBuffer(16);
  const u = new Uint32Array(words);
  const f = new Float32Array(words);
  u[0] = vec4Count;
  f[1] = cap;
  u[2] = 0;
  u[3] = 0;
  return words;
}

/** Per dimension workgroup limit. The full vocabulary needs 1024 groups, so there is room. */
const MAX_WORKGROUPS_PER_DIM = 65535;

function bind(input: KernelBindInput): KernelBindResult {
  const { device, inputs, output, params } = input;
  const count = params.count | 0;
  const cap = params.cap ?? FINAL_LOGIT_SOFTCAP;
  if (count <= 0 || count % 4 !== 0) {
    throw new Error(`logit-softcap needs params.count a positive multiple of 4, got ${count}`);
  }
  const groups = Math.ceil(count / 4 / WORKGROUP_SIZE);
  if (groups > MAX_WORKGROUPS_PER_DIM) {
    throw new Error(
      `logit-softcap wants ${groups} workgroups against a per dimension limit of `
      + `${MAX_WORKGROUPS_PER_DIM}. At four logits per lane the 262144 vocabulary needs 1024, so a `
      + 'count this large means the caller is passing something that is not a logit vector.',
    );
  }
  const src = inputs.src;
  if (!src) throw new Error('logit-softcap needs an input named src');

  const layout = kernelLayout(input, 'logit-softcap', () => device.createBindGroupLayout({
    label: 'logit-softcap',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  }));
  const uniform = kernelUniform(input, 'logit-softcap params', softcapParams(count / 4, cap));

  return {
    layout,
    buffers: [src, output, uniform.binding],
    dispatch: [groups, 1, 1],
    dispose: uniform.dispose,
  };
}

// The tolerance argument.
//
// One divide, one tanh and one multiply, on inputs bounded by 64 in the fixture. WGSL specifies
// tanh's accuracy through exp, whose error is 3 + 2|x| ULP; the fixture's largest argument is about
// 2 in magnitude, where that is a handful of ULP of a number near 7, so the relative error in the
// tanh is under a part in a million. Times a cap of 30 that is 3e-5, and the divide and the
// multiply add one correctly rounded step each. 1e-4 covers it, which is about three parts per
// million of the cap. The pass-through case, where the config carries no cap, is a copy and gates
// at exactly zero, because a copy that is off by anything is not a copy.
const SOFTCAP_TOL_ABS = 1e-4;

export const logitSoftcapKernel: Kernel = {
  name: 'logit-softcap',
  wgsl: LOGIT_SOFTCAP_WGSL,
  entry: 'main',
  note:
    'K14. tanh(logits / 30) * 30 over the 262144 wide logit vector, with the cap read from the '
    + 'config rather than assumed. Monotonic, so it cannot move an argmax; it ships because parity '
    + 'against the reference logits needs it.',
  cases: [
    {
      name: 'cap-30',
      inputs: { src: 'kmlp.cap.src' },
      expected: 'kmlp.cap.expected',
      params: { count: 4096, cap: FINAL_LOGIT_SOFTCAP },
      tolAbs: SOFTCAP_TOL_ABS,
      note:
        'A 4096 element slice rather than the whole vocabulary: this kernel is elementwise with a '
        + 'runtime bound, so the full 262144 exercises nothing the slice does not.',
    },
    {
      name: 'no-cap',
      inputs: { src: 'kmlp.cap.src' },
      expected: 'kmlp.cap.src',
      params: { count: 4096, cap: 0 },
      tolAbs: 0,
      tolUlp: 0,
      note:
        'A config with no final_logit_softcapping. The expected buffer is the input itself, so this '
        + 'case is bit identity against a copy and nothing softer.',
    },
  ],
  bind,
};
