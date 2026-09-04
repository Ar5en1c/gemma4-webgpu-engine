// SPDX-License-Identifier: Apache-2.0
//
// The 2-bit repack: the checkpoint's row layout to the interleaved tile layout of qlayout.ts,
// run once per 2-bit codes tensor on the GPU when the tensor has finished uploading. One
// invocation transposes one 16 row by 16 k block: sixteen row words in, sixteen tile words out,
// every byte read once and written once, so the whole 2-bit family (about 390 MB) repacks in a
// few milliseconds and the checkpoint bytes on disk and in the cache are never touched.
// docs/ENGINE-PERF.md section 13 is the measurement that made the layout worth it.

import type { Kernel, KernelBindInput, KernelBindResult } from './registry';
import { kernelLayout, kernelUniform } from './binding';
import { TILE_ROWS } from './qlayout';
import { resolveQuantBits } from '../quant';

const WORKGROUP = 64;

export const REPACK2_WGSL = /* wgsl */ `
struct RepackParams {
  // Rows of the tensor, so a tile past the end reads zero codes rather than another tensor.
  rows: u32,
  // u32 words per row in the row layout, K / 16.
  kWords: u32,
  // Whole tiles of ${TILE_ROWS} rows, the last one padded.
  tiles: u32,
  pad0: u32,
}

@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> params: RepackParams;

@compute @workgroup_size(${WORKGROUP})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
  @builtin(local_invocation_index) lid: u32,
) {
  let block = (wid.x + nwg.x * wid.y) * ${WORKGROUP}u + lid;
  if (block >= params.tiles * params.kWords) {
    return;
  }
  let tile = block / params.kWords;
  let w = block % params.kWords;
  // The sixteen row words of this block. A row past the tensor contributes zero codes.
${Array.from({ length: TILE_ROWS }, (_, r) => `  var in${r}: u32 = 0u;
  if (tile * ${TILE_ROWS}u + ${r}u < params.rows) { in${r} = src[(tile * ${TILE_ROWS}u + ${r}u) * params.kWords + w]; }`).join('\n')}
  let out = tile * params.kWords * 16u + w * 16u;
  for (var i = 0u; i < 16u; i = i + 1u) {
    let sh = 2u * i;
    dst[out + i] = ${Array.from({ length: TILE_ROWS }, (_, r) => `(((in${r} >> sh) & 3u) << ${2 * r}u)`).join('\n      | ')};
  }
}
`;

/** The params block: rows, kWords, tiles. */
export function repackParams(rows: number, k: number): ArrayBuffer {
  const buf = new ArrayBuffer(16);
  const u = new Uint32Array(buf);
  u[0] = rows;
  u[1] = k / 16;
  u[2] = Math.ceil(rows / TILE_ROWS);
  u[3] = 0;
  return buf;
}

/** Fold `groups` workgroups into a grid under the 65535 per dimension limit. */
function fold(groups: number): readonly [number, number, number] {
  if (groups <= 65535) return [groups, 1, 1];
  const x = 32768;
  return [x, Math.ceil(groups / x), 1];
}

function bindRepack(input: KernelBindInput): KernelBindResult {
  const { device, inputs, output, params } = input;
  const rows = params.rows | 0;
  const k = params.k | 0;
  if (rows <= 0 || k <= 0 || k % 128 !== 0) {
    throw new Error(`repack-2bit needs params.rows and params.k, k a multiple of 128, got ${rows} by ${k}`);
  }
  const src = inputs.src;
  if (!src) throw new Error('repack-2bit needs an input named src');
  const layout = kernelLayout(input, 'repack-2bit', () => device.createBindGroupLayout({
    label: 'repack-2bit',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  }));
  const uniform = kernelUniform(input, 'repack-2bit params', repackParams(rows, k));
  const blocks = Math.ceil(rows / TILE_ROWS) * (k / 16);
  return {
    layout,
    buffers: [src, output, uniform.binding],
    dispatch: fold(Math.ceil(blocks / WORKGROUP)),
    dispose: uniform.dispose,
  };
}

export const repack2Kernel: Kernel = {
  name: 'repack-2bit',
  wgsl: REPACK2_WGSL,
  entry: 'main',
  note:
    'The 2-bit row layout to the interleaved tile layout, once per tensor at load. The case '
    + 'holds it to qlayout.ts rowWordsToTile16 at bit identity, rows padded to a whole tile.',
  cases: [
    {
      name: 'synthetic-8x1024',
      inputs: { src: 'kmm.gemv2.wq' },
      expected: 'kmm.gemv2.tile.wq',
      params: { rows: 8, k: 1024 },
      tolAbs: 0,
      tolUlp: 0,
      note: 'Eight rows into one padded tile of sixteen; the same words the 2-bit GEMV case then reads.',
    },
  ],
  bind: bindRepack,
};

// ---------------------------------------------------------------------------------------------
// The loader side helper.
// ---------------------------------------------------------------------------------------------

const pipelines = new WeakMap<GPUDevice, { pipeline: GPUComputePipeline; layout: GPUBindGroupLayout }>();

function repackPipeline(device: GPUDevice): { pipeline: GPUComputePipeline; layout: GPUBindGroupLayout } {
  let built = pipelines.get(device);
  if (built) return built;
  const layout = device.createBindGroupLayout({
    label: 'repack-2bit',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  });
  const pipeline = device.createComputePipeline({
    label: 'repack-2bit',
    layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
    compute: { module: device.createShaderModule({ label: 'repack-2bit', code: REPACK2_WGSL }), entryPoint: 'main' },
  });
  built = { pipeline, layout };
  pipelines.set(device, built);
  return built;
}

/**
 * Repack one uploaded 2-bit codes tensor into a new buffer in the interleaved layout and return
 * it. The source is destroyed once the repack is submitted; queue order guarantees every later
 * dispatch on this queue reads the repacked words. The destination holds a whole number of
 * tiles, so a row count that is not a multiple of ${TILE_ROWS} grows by up to fifteen rows of
 * zero codes.
 */
export function repackTile16(device: GPUDevice, src: GPUBuffer, rows: number, k: number, label: string): GPUBuffer {
  if (k % 128 !== 0) throw new Error(`repackTile16: ${label} has k ${k}, which is not a multiple of 128`);
  const tiles = Math.ceil(rows / TILE_ROWS);
  const dst = device.createBuffer({
    label: `${label}:tile16`,
    size: tiles * k * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  const params = device.createBuffer({
    label: `${label}:tile16 params`,
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(params, 0, repackParams(rows, k));
  const { pipeline, layout } = repackPipeline(device);
  const bindGroup = device.createBindGroup({
    layout,
    entries: [
      { binding: 0, resource: { buffer: src } },
      { binding: 1, resource: { buffer: dst } },
      { binding: 2, resource: { buffer: params } },
    ],
  });
  const encoder = device.createCommandEncoder({ label: `${label}:tile16` });
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  const grid = fold(Math.ceil((tiles * (k / 16)) / WORKGROUP));
  pass.dispatchWorkgroups(grid[0], grid[1], grid[2]);
  pass.end();
  device.queue.submit([encoder.finish()]);
  src.destroy();
  void device.queue.onSubmittedWorkDone().then(() => params.destroy());
  return dst;
}

/**
 * Whether an uploaded tensor is a 2-bit codes tensor the interleaved layout applies to, and its
 * shape as rows by K if so. The codes tensors are U8 with four codes per byte, so K is four
 * times the packed width; the module path is the tensor name less its `.weight` or
 * `.embedding_quantized` suffix, resolved through quant.ts's rules (the embed table, and the
 * MLP linears of the 2-bit layers). Scales, norms, BF16 and the other families return null.
 */
export function tile16Target(name: string, shape: readonly number[], dtype: string): { rows: number; k: number } | null {
  if (dtype !== 'U8' || shape.length !== 2) return null;
  let module: string;
  if (name.endsWith('.embedding_quantized')) module = name.slice(0, -'.embedding_quantized'.length);
  else if (name.endsWith('.weight')) module = name.slice(0, -'.weight'.length);
  else return null;
  if (resolveQuantBits(module) !== 2) return null;
  const rows = shape[0]!;
  const k = shape[1]! * 4;
  if (k % 128 !== 0) return null;
  return { rows, k };
}
