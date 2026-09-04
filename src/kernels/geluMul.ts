// Written from docs/ENGINE-PLAN.md section 5 (the architecture contract table and kernel K10),
// .engine-ref/config.json, the Hugging Face transformers reference implementation of Gemma 4 and
// its Apache 2.0 gemma_quant integration, the cost ledgers in DECODE-CAMPAIGN.md 1 and 5.1 and
// DECODE-FUSION-FINDING.md, this engine's own quant.ts, and the WGSL specification. No vendored
// bundle, no extracted kernel and no third party engine source was read.
//
// SPDX-License-Identifier: Apache-2.0
//
// K10, the gated activation: `gelu(gate) * up`, in one pass, with the down projection's activation
// quantize folded into the same pass.
//
// WHICH GELU. `config.json` gives `text_config.hidden_activation` as `gelu_pytorch_tanh`, so this
// is the tanh approximation and not the erf form:
//
//     gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//
// The reference resolves that name to a module whose forward is `gelu(x, approximate="tanh")`, and
// the two forms are not interchangeable: they differ by up to 4.7e-4 near x = -2.7, which is twenty
// times this kernel's own tolerance. `scripts/engine-check.mjs` section k-mlp asserts the config
// string, checks this file's oracle against a table computed independently in Python, and asserts
// that the erf form would fail that table, so "we picked tanh" is a checked claim rather than a
// comment. The counter-example oracle lives beside the real one in this file for the same reason
// rmsNorm.ts keeps the Gemma 3 norm form beside the Gemma 4 one.
//
// TWO CALL SITES, ONE KERNEL. The reference applies this exact shape twice per decoder layer:
// once as the MLP's gated activation at intermediate width, 6144 on the producer layers and 12288
// on the consumer layers, and once on the per layer input path, where the gate projection's output
// is passed through the same activation and multiplied by the PLE row at width 256. Same maths,
// different width and a different second operand, so it is one module with a runtime width. That
// also satisfies ENGINE-PLAN risk 2 mitigation 4: a scalar multiply-add chain lives in exactly one
// module so there is no second module to disagree with it.
//
// THE ACTIVATION QUANTIZE IS PART OF THIS PASS. Every quantized linear in this checkpoint rounds
// its input onto an int8 grid first, using that module's own `input_activation_scale`, and rounds
// its output with `output_activation_scale` (quant.ts `applySrq`, and the reference's
// `QuantizedLinear.forward`). The consumer of this kernel's output is `down_proj`, so its input
// rounding is this vector's rounding, and doing it here costs nothing while doing it in a separate
// dispatch would read and rewrite a 12288 element vector for no reason. That is the same argument
// ENGINE-PLAN makes for K3: fuse to avoid re-reading an activation, not to save a dispatch. A scale
// of 0 means the site was never calibrated and the rounding is a no-op, exactly as the reference
// has it, which is why the shader applies it unconditionally through a uniform.
//
// WHY THE GATE AND UP PROJECTIONS ARE NOT IN THIS KERNEL, stated with the number rather than as a
// preference. Fusing them in would save writing and re-reading the two projection outputs: about
// 196 KB per consumer layer and 98 KB per producer layer, so roughly 5.4 MB per token against the
// 0.6 GB per token this engine streams (DECODE-CAMPAIGN.md 1). Under one percent, and the ledger is
// unambiguous that dispatch count is not the lever on either machine: 39 dispatches removed moved
// M1 GPU time by at most 0.1 to 0.3 ms, and the 5070 recovered about 4 microseconds per removed
// boundary (DECODE-FUSION-FINDING.md, DECODE-CAMPAIGN.md 5.2). The gate and up projections are the
// matmul lane's GEMV, this is the activation, and they meet through the registry rather than
// through a copy of each other's source. If M5 ever wants the fused variant, it composes qgemv.ts's
// template with `GELU_MUL_WGSL_FN` below rather than growing a second gelu.

import type { Kernel, KernelBindInput, KernelBindResult } from './registry';
import { kernelLayout, kernelUniform } from './binding';
import { applySrq, SRQ_ACTIVATION_BITS } from '../quant';

/** 64 wide, the width the rest of the engine uses (DECODE-CAMPAIGN.md 4.2). Nothing reduces here. */
const WORKGROUP_SIZE = 64;

/**
 * sqrt(2/pi) and the cubic coefficient, as the reference writes them.
 *
 * Both are given here at f64 precision and rounded once by the WGSL front end, rather than being
 * pre-rounded to f32 in this file, so the shader and the oracle start from the same real number.
 */
export const GELU_TANH_COEF = 0.7978845608028654;
export const GELU_TANH_CUBIC = 0.044715;

/**
 * The tanh argument is clamped to +/-10 before the call.
 *
 * In the shader's own arithmetic the clamp is free: tanh(10) is 1 - 4.1e-9, the nearest f32 to that
 * is exactly 1.0, and so is the nearest f32 to the tanh of anything larger, so every argument the
 * clamp touches was going to produce +/-1 in f32 anyway. The clamp is reached only above about
 * x = 5.4, which the check asserts, so the part of the range where this activation does anything
 * interesting is untouched.
 *
 * What it buys is that no backend computing tanh as (exp(2x) - 1) / (exp(2x) + 1) can reach an
 * infinity over an infinity on a large activation, which is a NaN in the residual stream and a dead
 * generation rather than a slightly wrong number.
 *
 * The oracle clamps identically, which costs it a little accuracy in f64 and is worth it: at
 * x = -12 the clamped f64 answer is -2.5e-8 where the unclamped one is around -1e-32. That is three
 * orders of magnitude inside this kernel's tolerance and the shader, working in f32, produces the
 * exact zero anyway. Keeping the two structurally identical is worth more than an f64 tail nobody
 * can observe.
 */
export const GELU_TANH_CLAMP = 10;

/**
 * Static range quantization is always 8 bit here, whatever the neighbouring weight width is. The
 * value comes from quant.ts, which is the authority, rather than being written out again.
 */
export const SRQ_BITS = SRQ_ACTIVATION_BITS;

/**
 * The activation as a WGSL function, exported on its own so a later fused gate-up-gelu kernel can
 * splice in this exact expression rather than write a second one.
 */
export const GELU_MUL_WGSL_FN = /* wgsl */ `
const GELU_COEF: f32 = ${GELU_TANH_COEF};
const GELU_CUBIC: f32 = ${GELU_TANH_CUBIC};
const GELU_CLAMP: f32 = ${GELU_TANH_CLAMP}.0;

// gelu_pytorch_tanh, from config.json's hidden_activation. Not the erf form.
fn geluTanh(x: vec4<f32>) -> vec4<f32> {
  let cube = x * x * x;
  let inner = clamp(
    GELU_COEF * (x + GELU_CUBIC * cube),
    vec4<f32>(-GELU_CLAMP),
    vec4<f32>(GELU_CLAMP)
  );
  return 0.5 * x * (vec4<f32>(1.0) + tanh(inner));
}
`;

export const GELU_MUL_WGSL = /* wgsl */ `${GELU_MUL_WGSL_FN}
struct ActParams {
  // Row width in vec4 lanes: 3072 at the consumer intermediate width of 12288, 1536 at the
  // producer width, 64 on the per layer input path. Runtime opaque, per the registry's rule, so
  // one compiled module serves all three.
  vec4Count: u32,
  // The consuming linear's input_activation_scale. Exactly 0.0 means uncalibrated, which the
  // reference treats as no rounding at all, so this shader does too (quant.ts applySrq).
  srqScale: f32,
  pad0: u32,
  pad1: u32,
}

// int8 range, which is what static range quantization uses at every site in this checkpoint.
const SRQ_MAX: f32 = 127.0;
const SRQ_MIN: f32 = -128.0;

@group(0) @binding(0) var<storage, read> gate: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> up: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> params: ActParams;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.vec4Count) {
    return;
  }

  var v = geluTanh(gate[i]) * up[i];

  // Uniform control flow: srqScale comes from the uniform block, so every lane in every workgroup
  // takes the same branch. There is no reduction and no barrier in this kernel, so ENGINE-PLAN 5.5
  // rules 1 and 5 have nothing to bite on, but the branch is uniform anyway because a per lane
  // branch here would be the habit that eventually lands next to a reduction.
  if (params.srqScale != 0.0) {
    let q = clamp(round(v / params.srqScale), vec4<f32>(SRQ_MIN), vec4<f32>(SRQ_MAX));
    v = q * params.srqScale;
  }

  dst[i] = v;
}
`;

// ---------------------------------------------------------------------------------------------
// The CPU oracles.
// ---------------------------------------------------------------------------------------------

/**
 * `gelu_pytorch_tanh` on one value, in f64, rounded once at the end.
 *
 * The clamp mirrors the shader's. Everything else is the reference formula written straight out.
 */
export function geluTanhScalar(x: number): number {
  const inner0 = GELU_TANH_COEF * (x + GELU_TANH_CUBIC * x * x * x);
  const inner = inner0 > GELU_TANH_CLAMP
    ? GELU_TANH_CLAMP
    : inner0 < -GELU_TANH_CLAMP ? -GELU_TANH_CLAMP : inner0;
  return 0.5 * x * (1 + Math.tanh(inner));
}

/**
 * The counter-example, kept beside the real thing on purpose.
 *
 * This is the exact gelu, `0.5 * x * (1 + erf(x / sqrt(2)))`, which is what `gelu` means when a
 * config does not say `tanh`. Nothing in the engine calls it. The k-mlp check asserts that it
 * disagrees with `geluTanhScalar` by far more than the kernel's tolerance, which is what makes the
 * tolerance a real test of which variant shipped.
 *
 * The erf is Abramowitz and Stegun 7.1.26, accurate to about 1.5e-7, which is three orders of
 * magnitude tighter than the 4.7e-4 gap it is being used to demonstrate.
 */
export function geluErfScalar(x: number): number {
  const t = 1 / (1 + 0.3275911 * Math.abs(x) / Math.SQRT2);
  const poly = t * (0.254829592
    + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
  const erfAbs = 1 - poly * Math.exp(-(x * x) / 2);
  const erf = x >= 0 ? erfAbs : -erfAbs;
  return 0.5 * x * (1 + erf);
}

/**
 * The known answer reference for `gelu-mul`.
 *
 * The activation runs in f64 and rounds once per element, which is a reference rather than a mirror
 * of the shader's f32 chain, exactly as ENGINE-PLAN section 7 asks: the case states a tolerance
 * somebody argued for. The static range quantize afterwards is not approximate at all, so it is
 * delegated to quant.ts's `applySrq` rather than reimplemented here. That is the loader's contract
 * for this operation and a second copy of it would be a second thing to keep right.
 *
 * There used to be a documented difference between the oracle and the shader at a rounding tie, and
 * there is not any more: `applySrq` breaks a tie toward the even integer, which is what WGSL's
 * `round` and `torch.round` both do, so the two sides agree on every input including the ties. The
 * k-mlp check still counts the ties in each fixture and asserts both rules resolve them the same
 * way, because a measured zero is worth more than a settled argument.
 *
 * @param gate     the gate projection's output
 * @param up       the up projection's output, or the PLE row on the per layer input path
 * @param count    elements to process, a multiple of 4 because the shader works in vec4 lanes
 * @param srqScale the consuming linear's `input_activation_scale`, 0 for an uncalibrated site
 */
export function geluMulOracle(
  gate: Float32Array,
  up: Float32Array,
  count: number,
  srqScale = 0,
): Float32Array {
  if (gate.length < count) throw new Error(`geluMulOracle: gate holds ${gate.length}, needs ${count}`);
  if (up.length < count) throw new Error(`geluMulOracle: up holds ${up.length}, needs ${count}`);
  const out = new Float32Array(count);
  for (let i = 0; i < count; i += 1) {
    out[i] = geluTanhScalar(gate[i]) * up[i];
  }
  return applySrq(out, srqScale, SRQ_BITS, out);
}

// ---------------------------------------------------------------------------------------------
// Registry entry.
// ---------------------------------------------------------------------------------------------

/** The params block, exported so the scheduler can restage it in place per dispatch. */
export function actParams(vec4Count: number, srqScale: number): ArrayBuffer {
  const words = new ArrayBuffer(16);
  const u = new Uint32Array(words);
  const f = new Float32Array(words);
  u[0] = vec4Count;
  f[1] = srqScale;
  u[2] = 0;
  u[3] = 0;
  return words;
}

function bind(input: KernelBindInput): KernelBindResult {
  const { device, inputs, output, params } = input;
  const count = params.count | 0;
  const srqScale = params.srqScale ?? 0;
  if (count <= 0 || count % 4 !== 0) {
    throw new Error(`gelu-mul needs params.count a positive multiple of 4, got ${count}`);
  }
  const gate = inputs.gate;
  const up = inputs.up;
  if (!gate || !up) throw new Error('gelu-mul needs inputs named gate and up');

  const layout = kernelLayout(input, 'gelu-mul', () => device.createBindGroupLayout({
    label: 'gelu-mul',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  }));
  const uniform = kernelUniform(input, 'gelu-mul params', actParams(count / 4, srqScale));

  return {
    layout,
    buffers: [gate, up, output, uniform.binding],
    dispatch: [Math.ceil(count / 4 / WORKGROUP_SIZE), 1, 1],
    dispose: uniform.dispose,
  };
}

// The tolerance argument, once, since every case here shares it.
//
// The fixture bounds gate at 8 and up at 4, so the product is at most 32 in magnitude. Three things
// can move: the polynomial inside the tanh, the tanh itself, and the two multiplies that finish the
// expression. WGSL specifies tanh's accuracy through exp, whose error is 3 + 2|x| ULP; at the
// clamped argument range that is under two parts in a million of the tanh result, and the derivative
// of gelu with respect to that argument is at most |x| / 2, so the activation carries well under
// 1e-5 absolute at these magnitudes. The remaining f32 roundings are one ULP each on values up to
// 32, which is 4e-6. 2e-5 covers that with room and is still twenty times below the 4.7e-4 gap to
// the erf form, so a kernel that shipped the wrong variant fails this gate rather than squeaking
// through it. That last sentence is the whole reason the number is not simply "generous".
const ACT_TOL_ABS = 2e-5;

export const geluMulKernel: Kernel = {
  name: 'gelu-mul',
  wgsl: GELU_MUL_WGSL,
  entry: 'main',
  note:
    'K10. gelu_pytorch_tanh of the gate times the up projection, with the consuming linear\'s '
    + 'static range quantize folded into the same pass. Serves the MLP at intermediate width and '
    + 'the per layer input gate at width 256.',
  cases: [
    {
      name: 'producer-6144',
      inputs: { gate: 'kmlp.act.gate6144', up: 'kmlp.act.up6144' },
      expected: 'kmlp.act.expected6144',
      params: { count: 6144, srqScale: 0 },
      tolAbs: ACT_TOL_ABS,
      note: 'Producer layer intermediate width, uncalibrated site, so no activation rounding.',
    },
    {
      name: 'consumer-12288-srq',
      inputs: { gate: 'kmlp.act.gate12288', up: 'kmlp.act.up12288' },
      expected: 'kmlp.act.expected12288',
      params: { count: 12288, srqScale: 0.125 },
      tolAbs: ACT_TOL_ABS,
      note:
        'Consumer layer width with the down projection\'s activation rounding applied. The scale is '
        + 'a power of two so the grid is exact in f32, and at this one about eight percent of the '
        + 'fixture sits at the int8 clamp, so the clamp is covered rather than assumed.',
    },
    // The two cases that use Google's own numbers rather than a fixture this lane invented. The
    // gate and up buffers are what the reference's gate_proj and up_proj actually returned at one
    // position of a real prompt, and the expected buffer is what its own activation module produced
    // from them (.engine-ref/tools/mlp-ref.py). They are also two very different scales: layer 0's
    // activation peaks near 3652 and layer 15's near 2, which is exactly the spread that makes a
    // single absolute tolerance meaningless and is why each case states its own.
    {
      name: 'reference-layer0',
      inputs: { gate: 'ref.l0.mlp.gate', up: 'ref.l0.mlp.up' },
      expected: 'ref.l0.mlp.act',
      params: { count: 6144, srqScale: 0 },
      tolAbs: 4e-3,
      tolUlp: 8,
      note:
        'Layer 0, a 4-bit producer layer at intermediate width 6144. The tolerance is about a '
        + 'millionth of this tensor\'s peak of 3652, paired with an 8 ULP budget for the elements '
        + 'near zero where an absolute number says nothing.',
    },
    {
      name: 'reference-layer15',
      inputs: { gate: 'ref.l15.mlp.gate', up: 'ref.l15.mlp.up' },
      expected: 'ref.l15.mlp.act',
      params: { count: 12288, srqScale: 0 },
      tolAbs: 2e-6,
      tolUlp: 8,
      note:
        'Layer 15, the first 2-bit consumer layer, where use_double_wide_mlp doubles the width to '
        + '12288. Same reasoning, against a peak of 2.04.',
    },
    // The same two positions again with the down projection's real activation rounding on, gated
    // against what the reference's own `apply_srq` returned. This is the pair that proves the SRQ
    // epilogue at the scales the checkpoint actually calibrated rather than at a fixture's tidy
    // power of two, and it is the GPU half of a claim k-mlp already makes on the CPU.
    //
    // THE TOLERANCE IS A GRID STEP, deliberately, and the cosine floor is what carries the case.
    // After the rounding every element is an exact multiple of the scale, so the only difference a
    // correct kernel can produce is one element landing on the neighbouring code where the value
    // sat within a rounding of the boundary, and that difference is exactly one scale step. An
    // absolute tolerance below a step would gate on the tanh's last bit rather than on the
    // rounding; one above a step would let a systematically wrong kernel through, which is what the
    // cosine floor of 0.99999 is there to catch. k-mlp measures how many boundary elements there
    // actually are, and the answer on this dump is that the oracle reproduces the capture inside 4
    // ULP everywhere, so a step is headroom rather than an expectation.
    {
      name: 'reference-layer0-srq',
      inputs: { gate: 'ref.l0.mlp.gate', up: 'ref.l0.mlp.up' },
      expected: 'ref.l0.mlp.act_q',
      // down_proj's input_activation_scale at layer 0, from .engine-ref/data/ref.mlp.meta.json.
      // The k-mlp check asserts this number equals the one in that file, so the copy cannot rot.
      params: { count: 6144, srqScale: 27.842519760131836 },
      tolAbs: 27.842519760131836,
      minCosine: 0.99999,
      note:
        'Layer 0 with down_proj\'s activation rounding applied, against the reference\'s own '
        + 'act_q capture. The down projection\'s input side, closed here and again in the matmul.',
    },
    {
      name: 'reference-layer15-srq',
      inputs: { gate: 'ref.l15.mlp.gate', up: 'ref.l15.mlp.up' },
      expected: 'ref.l15.mlp.act_q',
      // The same scale at layer 15, the first 2-bit consumer layer. Three orders of magnitude
      // smaller than layer 0's, which is why each case states its own numbers.
      params: { count: 12288, srqScale: 0.02005414292216301 },
      tolAbs: 0.02005414292216301,
      minCosine: 0.99999,
      note:
        'Layer 15, double wide, with its own down_proj input scale. The pair with the case above '
        + 'is what makes the rounding a property of the module rather than of one layer.',
    },
    {
      name: 'per-layer-gate-256',
      inputs: { gate: 'kmlp.act.gate256', up: 'kmlp.act.up256' },
      expected: 'kmlp.act.expected256',
      params: { count: 256, srqScale: 0 },
      tolAbs: ACT_TOL_ABS,
      note:
        'The per layer input path: the same activation applied to the input gate\'s output and '
        + 'multiplied by the PLE row, at hidden_size_per_layer_input.',
    },
  ],
  bind,
};
