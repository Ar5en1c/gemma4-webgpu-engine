// Written from the WGSL specification and this lane's harness brief. No model maths here.
//
// SPDX-License-Identifier: Apache-2.0
//
// The harness's own known answer canary: an elementwise `dst[i] = alpha * x[i] + y[i]`.
//
// This kernel exists to prove the rig, not the model. It is the smallest thing that exercises
// every joint the real kernels use: manifest inputs, an explicit bind group layout, a uniform
// block the kernel allocates for itself, a guarded pipeline creation, a dispatch, a readback, and
// a diff that has to come out at exactly zero.
//
// Why the fixture values are what they are, which is the only interesting decision in this file.
// `alpha * x + y` is a scalar multiply-add chain, and that is precisely the shape whose rounding
// is not a contract: on Metal the compiler chooses FMA contraction per compiled module, so the
// same expression can round differently in two modules that compute the same thing
// (DECODE-FUSION-FINDING.md, and ENGINE-PLAN.md risk 2). A canary that demanded bit identity from
// this shape over arbitrary inputs would be asserting something the hardware does not promise, and
// would eventually go amber for a reason that is not a bug.
//
// So the fixture is built from values where contraction cannot matter. `x` is a multiple of 1/4,
// `y` a multiple of 1/8, `alpha` is 3/2, and every magnitude stays under 8. Every product and
// every sum is exactly representable in f32, so the rounded and the contracted result are the same
// number, and a zero difference is a real claim about the dispatch rather than a claim about the
// compiler's mood. `scripts/engine-check.mjs` generates the fixture and re-derives the expected
// buffer from that rule, so the two cannot drift apart.

import type { Kernel, KernelBindInput, KernelBindResult } from './registry';
import { kernelLayout, kernelUniform } from './binding';
import { foldedDispatch } from './qgemv';

export const SCALE_ADD_WGSL = /* wgsl */ `
struct Params {
  n: u32,
  alpha: f32,
  pad0: u32,
  pad1: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> y: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

// 64 wide, which is the workgroup width the decode GEMV geometry settled on for this hardware
// (DECODE-CAMPAIGN.md 4.2: 64 wide, 2 subgroups, 4 rows per workgroup measured best; 32 wide with
// 2 rows measured worse because it caps resident warps). There is nothing to reduce here, so this
// kernel has no opinion on subgroups and takes no subgroup path.
//
// The grid is folded over two dimensions, the same fold the logit head uses (qgemv.ts
// foldedDispatch), because one dimension is capped at 65535 workgroups and this kernel runs over
// tokens times 8960 per layer inputs in the embed phase: at the round 4 prefill chunk of 512 that
// is 71,680 workgroups, and a dispatch over the cap is a validation error that drops the whole
// command buffer without a token of output changing shape. Round 4 found it as a 483 token prompt
// prefilling in 126 ms and writing the wrong first token (docs/ENGINE-PERF.md, round 4 lever 1).
@compute @workgroup_size(64)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
  @builtin(local_invocation_index) lid: u32,
) {
  let i = (wid.x + nwg.x * wid.y) * 64u + lid;
  // The bound comes from the uniform, not from a constant, per the registry's rule 2. It costs
  // nothing here and it keeps the habit.
  if (i >= params.n) {
    return;
  }
  dst[i] = params.alpha * x[i] + y[i];
}
`;

const WORKGROUP_SIZE = 64;

/**
 * The params block, exported so the engine's scheduler can restage it in place per dispatch
 * instead of rebinding through a fresh buffer. Uniform blocks round up to 16 bytes in the uniform
 * address space, hence the two pad words in the struct above.
 */
export function scaleAddParams(n: number, alpha: number): ArrayBuffer {
  const words = new ArrayBuffer(16);
  new Uint32Array(words, 0, 1)[0] = n;
  new Float32Array(words, 4, 1)[0] = alpha;
  return words;
}

function bind(input: KernelBindInput): KernelBindResult {
  const { device, inputs, output, params } = input;
  const n = params.n | 0;
  const alpha = params.alpha ?? 1;

  const layout = kernelLayout(input, 'scale-add', () => device.createBindGroupLayout({
    label: 'scale-add',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  }));

  const uniform = kernelUniform(input, 'scale-add params', scaleAddParams(n, alpha));

  const x = inputs.x;
  const y = inputs.y;
  if (!x || !y) throw new Error('scale-add needs inputs named x and y');

  return {
    layout,
    buffers: [x, y, output, uniform.binding],
    dispatch: foldedDispatch(Math.ceil(n / WORKGROUP_SIZE)),
    dispose: uniform.dispose,
  };
}

export const scaleAddKernel: Kernel = {
  name: 'scale-add',
  wgsl: SCALE_ADD_WGSL,
  entry: 'main',
  note: 'Harness canary. Elementwise alpha * x + y over an exactly representable fixture.',
  cases: [
    {
      name: 'axpy-1024',
      inputs: { x: 'harness.scaleAdd.x', y: 'harness.scaleAdd.y' },
      expected: 'harness.scaleAdd.expected',
      params: { n: 1024, alpha: 1.5 },
      tolAbs: 0,
      tolUlp: 0,
      note: 'Exact in f32 by construction, so the gate is bit identity and nothing softer.',
    },
  ],
  bind,
};
