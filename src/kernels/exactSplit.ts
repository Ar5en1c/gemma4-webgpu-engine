// SPDX-License-Identifier: Apache-2.0
// Split calibrated 4-bit projections before scaling. K <= 16384 and int8 activations
// bound the entire signed dot by 2^24, so every partial and every fold is exact in f32.
import type { Kernel, KernelBindInput } from './registry';
import { kernelLayout, kernelUniform } from './binding';
import { bindGemv, gemvGeometry, qgemvWgsl, qgemv4GeluKernel, type MatmulReduceVariant } from './qgemv';

let splits: 1 | 2 | 4 | 8 = 1;
export function producerDownSplits(): number { return splits; }
export function setProducerDownSplits(value: number): void {
  if (![1, 2, 4, 8].includes(value)) throw new Error('producer down splits must be 1, 2, 4 or 8');
  splits = value as typeof splits;
}
const geometry = () => ({ ...gemvGeometry(4), kSplits: splits === 1 ? 2 : splits });
export function exactSplitWgsl(variant: MatmulReduceVariant): string {
  return qgemvWgsl(4, variant, geometry(), 'gelu', true, true);
}
const bind = bindGemv(4, 'gelu', true, geometry);
export const exactSplitKernel: Kernel = {
  name: 'qgemv-4bit-gelu-exact-split',
  get wgsl() { return exactSplitWgsl('subgroup'); },
  entry: 'main',
  get cases() {
    const c = qgemv4GeluKernel.cases[0]!;
    return [{ ...c, name: 'integer-partials', expected: `kmm.gemv4.exact.part${geometry().kSplits}` }];
  },
  bind(input: KernelBindInput) {
    if (!(input.params.inScale! > 0) || input.params.k! > 16384) throw new Error('exact split needs calibrated int8 input and K <= 16384');
    return bind(input);
  },
};
export const exactMergeKernel: Kernel = {
  name: 'qgemv-exact-merge', entry: 'main',
  cases: [2, 4, 8].map(kSplits => ({
    name: `integer-fold-${kSplits}`,
    inputs: { part: `kmm.gemv4.exact.part${kSplits}`, scales: 'kmm.gemv4.scales' },
    expected: qgemv4GeluKernel.cases[0]!.expected,
    params: { ...qgemv4GeluKernel.cases[0]!.params, kSplits }, tolAbs: 0, tolUlp: 0,
  })),
  wgsl: `struct Params { rows: u32, splits: u32, inScale: f32, outScale: f32 }
@group(0) @binding(0) var<storage, read> part: array<f32>;
@group(0) @binding(1) var<storage, read> scales: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= params.rows) { return; }
  var total = 0.0;
  for (var s = 0u; s < params.splits; s++) { total += part[row * params.splits + s]; }
  var v = (params.inScale * scales[row]) * total;
  if (params.outScale != 0.0) {
    v = clamp(round(v / params.outScale), -128.0, 127.0) * params.outScale;
  }
  dst[row] = v;
}`,
  bind(input) {
    const { device, params, inputs, output } = input;
    const data = new ArrayBuffer(16), u = new Uint32Array(data), f = new Float32Array(data);
    u[0] = params.numRows!; u[1] = params.kSplits!;
    f[2] = params.inScale!; f[3] = params.outScale!;
    const layout = kernelLayout(input, 'qgemv-exact-merge', () => device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ] }));
    const uniform = kernelUniform(input, 'qgemv-exact-merge params', data);
    return { layout, buffers: [inputs.part!, inputs.scales!, output, uniform.binding],
      dispatch: [Math.ceil(params.numRows! / 64), 1, 1], dispose: uniform.dispose };
  },
};
