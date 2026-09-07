// SPDX-License-Identifier: Apache-2.0
// Batch independent projections without repacking weights or changing their arithmetic.
// The generated dot loop is qgemv.ts's own loop; only resource selection and grid placement
// change. Q/K/V share one dispatch on producer layers; gate/up share one on every layer.
import type { Kernel, KernelBindInput, KernelBindResult, KernelCase } from './registry';
import { kernelLayout, kernelUniform } from './binding';
import { qgemv2Kernel, qgemv4Kernel, gemvGeometry, gemvRowsPerWorkgroup, qgemvWgsl, unsplitGemvGeometry, type GemvBits, type MatmulReduceVariant } from './qgemv';
import { wordsPerRow } from './qlayout';

let batchWorkgroup = 0;
/** Zero follows the family geometry; other widths apply only to the batched 2-bit producers. */
export function setBatchWorkgroup(value: number): void {
  if (![0, 32, 64, 128, 256].includes(value)) throw new Error('batch workgroup must be 0, 32, 64, 128 or 256');
  batchWorkgroup = value;
}
const batchGeometry = (bits: GemvBits) => {
  const g = unsplitGemvGeometry(gemvGeometry(bits));
  return bits === 2 && batchWorkgroup ? { ...g, workgroupSize: batchWorkgroup } : g;
};

export function projectionBatchWgsl(
  bits: GemvBits,
  count: 2 | 3,
  variant: MatmulReduceVariant,
  calibratedF32 = false,
): string {
  const geometry = batchGeometry(bits);
  let code = qgemvWgsl(bits, variant, geometry, 'none', false, false, calibratedF32);
  const type = bits === 2 ? 'vec4<u32>' : geometry.wordsPerLane === 1 ? 'u32' : `vec${geometry.wordsPerLane}<u32>`;
  const expected = ['struct GemvParams {', 'let group = wid.x + nwg.x * wid.y;'];
  for (const needle of expected) if (!code.includes(needle)) throw new Error(`projection batch source contract changed: ${needle}`);
  code = code.replace(/struct GemvParams \{[\s\S]*?\n\}/, `struct GemvParams {
  kWords: u32,
  kIters: u32,
  firstGroups: u32,
  secondGroups: u32,
  rowCounts: vec4<u32>,
  inScales: vec4<f32>,
  outScales: vec4<f32>,
}`);
  // All five declarations belong to the plain GEMV interface. Fail if that interface changes.
  const declarations = code.match(/^@group\(0\) @binding\(\d+\).*;$/gm) ?? [];
  if (declarations.length !== 5) throw new Error('projection batch requires the five-binding plain GEMV');
  code = code.replace(/^@group\(0\) @binding\(\d+\).*;\n?/gm, '');
  // Specialize the resource choice outside the dot loop. Every workgroup enters one
  // original GEMV body with static weight bindings and its own SRQ scale pair.
  const original = qgemvWgsl(bits, variant, geometry, 'none', false, false, calibratedF32);
  const header = original.slice(0, original.indexOf('struct GemvParams {'));
  const uniformStruct = code.slice(code.indexOf('struct GemvParams {'), code.indexOf('struct GemvParams {') + code.slice(code.indexOf('struct GemvParams {')).indexOf('}') + 1);
  let binding = 0;
  const resources: string[] = [];
  for (let i = 0; i < count; i++) resources.push(
    `@group(0) @binding(${binding++}) var<storage, read> packedWeightBatch${i}: array<${type}>;`,
    `@group(0) @binding(${binding++}) var<storage, read> rowScaleBatch${i}: array<f32>;`,
  );
  resources.push(`@group(0) @binding(${binding++}) var<storage, read> x: array<vec4<f32>>;`);
  for (let i = 0; i < count; i++) resources.push(`@group(0) @binding(${binding++}) var<storage, read_write> out${i}: array<f32>;`);
  resources.push(`@group(0) @binding(${binding}) var<uniform> params: GemvParams;`);
  const bodies = Array.from({ length: count }, (_, i) => {
    let body = original.slice(header.length).replace(/struct GemvParams \{[\s\S]*?\n\}/, '')
      .replace(/^@group\(0\) @binding\(\d+\).*;\n?/gm, '')
      .replace(/@compute @workgroup_size\(\d+\)/, '')
      .replace(/@builtin\(\w+\) /g, '')
      .replace(/\bparams\.numRows\b/g, `params.rowCounts[${i}]`)
      .replace(/\bparams\.inScale\b/g, `params.inScales[${i}]`)
      .replace(/\bparams\.outScale\b/g, `params.outScales[${i}]`)
      .replace(/\bwq\b/g, `packedWeightBatch${i}`).replace(/\bscales\b/g, `rowScaleBatch${i}`).replace(/\bdst\b/g, `out${i}`);
    const names = [...body.matchAll(/(?:fn|const) (\w+)/g)].map(m => m[1]!);
    for (const name of names) body = body.replace(new RegExp(`\\b${name}\\b`, 'g'), `${name}_${i}`);
    return body;
  });
  const call = (i: number, group: string) => `main_${i}(vec3<u32>(${group}, 0u, 0u), vec3<u32>(1u), lid);`;
  const dispatch = count === 2
    ? `if (wid.y == 0u) { ${call(0, 'wid.x')} } else { ${call(1, 'wid.x')} }`
    : `if (wid.x < params.firstGroups) { ${call(0, 'wid.x')} }
      else if (wid.x < params.firstGroups + params.secondGroups) { ${call(1, 'wid.x - params.firstGroups')} }
      else { ${call(2, 'wid.x - params.firstGroups - params.secondGroups')} }`;
  return header + uniformStruct + '\n' + resources.join('\n') + bodies.join('\n') + `
@compute @workgroup_size(${geometry.workgroupSize})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  ${dispatch}
}`;
}

function bindProjectionBatch(bits: GemvBits, count: 2 | 3) {
  return (input: KernelBindInput): KernelBindResult => {
    const { device, inputs, output, params } = input;
    const geometry = batchGeometry(bits);
    const k = params.k;
    if (!Number.isInteger(k) || k <= 0 || k % 128 !== 0) throw new Error('batched projection K must be a positive multiple of 128');
    const rows = Array.from({ length: count }, (_, i) => params[`rows${i}`]);
    if (rows.some(r => !Number.isInteger(r) || r <= 0)) throw new Error('batched projection rows must be positive integers');
    if (count === 2 && rows[0] !== rows[1]) throw new Error('gate/up batch requires equal row counts');
    const groups = rows.map(r => Math.ceil(r / gemvRowsPerWorkgroup(geometry)));
    const data = new ArrayBuffer(64);
    const u = new Uint32Array(data);
    const f = new Float32Array(data);
    u[0] = wordsPerRow(bits, k);
    u[1] = Math.floor(u[0]! / (32 * geometry.wordsPerLane));
    u[2] = groups[0]!;
    u[3] = groups[1]!;
    for (let i = 0; i < count; i++) {
      u[4 + i] = rows[i]!;
      f[8 + i] = params[`inScale${i}`] ?? 0;
      f[12 + i] = params[`outScale${i}`] ?? 0;
    }
    const storage: GPUBuffer[] = [];
    for (let i = 0; i < count; i++) storage.push(inputs[`w${i}`]!, inputs[`s${i}`]!);
    storage.push(inputs.x!);
    const outputIndex = params.outputIndex ?? count - 1;
    if (!Number.isInteger(outputIndex) || outputIndex < 0 || outputIndex >= count) throw new Error('invalid batch output index');
    for (let i = 0; i < count; i++) storage.push(i === outputIndex ? output : inputs[`out${i}`]!);
    if (storage.some(x => !x)) throw new Error('batched projection is missing an input buffer');
    const name = count === 3 ? 'qkv-batch-4bit' : `gate-up-batch-${bits}bit`;
    const layout = kernelLayout(input, name, () => device.createBindGroupLayout({
      label: name,
      entries: [
        ...storage.map((_, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE,
          buffer: { type: binding < 2 * count + 1 ? 'read-only-storage' as const : 'storage' as const } })),
        { binding: storage.length, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' as const } },
      ],
    }));
    const uniform = kernelUniform(input, `${name} params`, data);
    return { layout, buffers: [...storage, uniform.binding],
      dispatch: count === 2 ? [groups[0]!, 2, 1] : [groups.reduce((a, b) => a + b, 0), 1, 1],
      dispose: uniform.dispose };
  };
}

function cases(bits: GemvBits, count: 2 | 3): KernelCase[] {
  const source = bits === 2 ? qgemv2Kernel.cases : qgemv4Kernel.cases;
  // Each output is independently read back with the harness's unwritten sentinel. Mix
  // uncalibrated, SRQ and clamped SRQ within one dispatch to expose scale routing errors.
  const selections = [Array.from({ length: count }, (_, i) => source[i === 0 ? 0 : i === 1 ? 2 : 3]!),
    Array.from({ length: count }, () => source[1]!)];
  return selections.flatMap((selection, set) => selection.map((selected, outputIndex) => {
    const inputs: Record<string, string> = { x: selected.inputs.x! };
    const params: Record<string, number> = { k: selected.params!.k!, outputIndex };
    selection.forEach((c, i) => {
      inputs[`w${i}`] = c.inputs.wq!;
      inputs[`s${i}`] = c.inputs.scales!;
      if (i !== outputIndex) inputs[`out${i}`] = c.expected;
      params[`rows${i}`] = c.params!.numRows!;
      params[`inScale${i}`] = c.params!.inScale ?? 0;
      params[`outScale${i}`] = c.params!.outScale ?? 0;
    });
    return { name: `batch-${set}-output-${outputIndex}`, inputs, expected: selected.expected,
      params, tolAbs: 0, tolUlp: 0 };
  }));
}

function kernel(bits: GemvBits, count: 2 | 3): Kernel {
  return {
    name: count === 3 ? 'qkv-batch-4bit' : `gate-up-batch-${bits}bit`,
    get wgsl() { return projectionBatchWgsl(bits, count, 'subgroup'); },
    entry: 'main', cases: cases(bits, count),
    note: 'Independent projections in one dispatch, original packed weights and original per-projection SRQ scales.',
    bind: bindProjectionBatch(bits, count),
  };
}
export const qkvBatchKernel = kernel(4, 3);
export const gateUpBatch4Kernel = kernel(4, 2);
export const gateUpBatch2Kernel = kernel(2, 2);
