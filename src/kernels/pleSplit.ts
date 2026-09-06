// SPDX-License-Identifier: Apache-2.0
// PLE gate K split, with its unscaled partials folded by the projection.
// The pinned checkpoint has max(row L1) * 128 = 7,001,088 across all 35 gates,
// below 2^24. Thus all int8 dot partials and folds are exact for these weights.
// This is a checkpoint property, not a claim for arbitrary full-range I8 matrices.
import type { Kernel } from './registry';
import { kernelLayout, kernelUniform } from './binding';
import { bindGemv, gemvGeometry, gemvParams, gemvRowsPerWorkgroup, foldedDispatch, qgemvWgsl, qgemv8Kernel, qgemv8GeluKernel, type MatmulReduceVariant } from './qgemv';

let splits: 1 | 2 | 4 | 8 = 1;
export function pleGateSplits(): number { return splits; }
export function setPleGateSplits(value: number): void {
  if (![1, 2, 4, 8].includes(value)) throw new Error('PLE splits must be 1, 2, 4 or 8');
  splits = value as typeof splits;
}
const activeSplits = () => splits === 1 ? 2 : splits;
const geometry = () => ({ ...gemvGeometry(8), kSplits: activeSplits() });
export function pleGateSplitWgsl(variant: MatmulReduceVariant): string {
  const n = activeSplits();
  return qgemvWgsl(8, variant, geometry(), 'none', true, true)
    .replace(/dst\[row(\d+) \* \d+u \+ split\]/g,
      (_, r: string) => `dst[(row${r} / 4u) * ${4 * n}u + split * 4u + row${r} % 4u]`);
}
export function pleFoldProjectionWgsl(variant: MatmulReduceVariant): string {
  const n = activeSplits();
  return qgemvWgsl(8, variant, gemvGeometry(8), 'gelu')
    .replace('  pad2: u32,\n}', '  pad2: u32,\n  gateInScale: f32,\n  gateOutScale: f32,\n  pad3: u32,\n  pad4: u32,\n}')
    .replace('@group(0) @binding(4) var<storage, read_write> dst:',
      '@group(0) @binding(4) var<storage, read> gateScales: array<vec4<f32>>;\n@group(0) @binding(5) var<storage, read_write> dst:')
    .replace('@binding(5) var<uniform>', '@binding(6) var<uniform>')
    .replace('geluTanh(gate[i])', 'geluTanh(foldGate(i))') + `
fn foldGate(i: u32) -> vec4<f32> {
  var total = vec4<f32>(0.0);
${Array.from({ length: n }, (_, s) => `  total = total + gate[i * ${n}u + ${s}u];`).join('\n')}
  var v = (params.gateInScale * gateScales[i]) * total;
  if (params.gateOutScale != 0.0) {
    v = clamp(round(v / params.gateOutScale), vec4<f32>(-128.0), vec4<f32>(127.0)) * params.gateOutScale;
  }
  return v;
}`;
}
const bindGate = bindGemv(8, 'none', true, geometry);
export const pleGateSplitKernel: Kernel = {
  name: 'ple-gate-split', entry: 'main',
  get cases() {
    return qgemv8Kernel.cases.slice(1, 3).map((c, i) => ({ ...c,
      params: { ...c.params, numRows: 8 },
      expected: `kple.split.raw${activeSplits()}.${i}`, tolAbs: 0, tolUlp: 0,
    }));
  },
  get wgsl() { return pleGateSplitWgsl('subgroup'); },
  bind(input) {
    if (input.params.numRows! % 4 !== 0) throw new Error('PLE split row count must be divisible by four');
    if (!(input.params.inScale! > 0)) throw new Error('PLE split needs calibrated input');
    return bindGate(input);
  },
};
export const pleFoldProjectionKernel: Kernel = {
  name: 'ple-fold-projection', entry: 'main',
  get cases() {
    return qgemv8GeluKernel.cases.flatMap(c => [0, 0.125].map(gateOutScale => ({ ...c,
      name: `${c.name}-gate-srq-${gateOutScale}`,
      inputs: { ...c.inputs, gate: `kple.split.fold${activeSplits()}`, gateScales: 'kple.split.scales' },
      params: { ...c.params, gateInScale: 0.125, gateOutScale },
    })));
  },
  get wgsl() { return pleFoldProjectionWgsl('subgroup'); },
  bind(input) {
    const { device, inputs, params, output } = input;
    const g = gemvGeometry(8), kWords = params.k! / 4;
    const data = new ArrayBuffer(48);
    new Uint8Array(data).set(new Uint8Array(gemvParams(kWords, Math.floor(kWords / (32 * g.wordsPerLane)),
      params.numRows!, params.inScale!, params.outScale!, 1, params.upOffset ?? 0)));
    const f = new Float32Array(data); f[8] = params.gateInScale!; f[9] = params.gateOutScale!;
    const layout = kernelLayout(input, 'ple-fold-projection', () => device.createBindGroupLayout({ entries: [
      ...Array.from({ length: 5 }, (_, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' as const } })),
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ] }));
    const uniform = kernelUniform(input, 'ple-fold-projection params', data);
    return { layout, buffers: [inputs.wq!, inputs.scales!, inputs.gate!, inputs.up!, inputs.gateScales!, output, uniform.binding],
      dispatch: foldedDispatch(Math.ceil(params.numRows! / gemvRowsPerWorkgroup(g))), dispose: uniform.dispose };
  },
};
