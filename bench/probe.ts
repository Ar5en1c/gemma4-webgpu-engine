// SPDX-License-Identifier: Apache-2.0
//
// Cheap device probes that run with no model download, because a 70 second load that ends in
// silence tells you nothing and an iPhone reporting 3705 tok/s on empty output tells you worse
// than nothing. These answer two questions before any weight is fetched: does this device compute
// WGSL correctly at all, and how much memory will it actually hand over.

export interface SmokeResult {
  computeOk: boolean;
  computeDetail: string;
  errors: string[];
}

/** Does a trivial compute shader run and read back the right answer on this device. */
export async function smokeTest(device: GPUDevice): Promise<SmokeResult> {
  const errors: string[] = [];
  const onErr = (e: Event) => {
    errors.push(String((e as GPUUncapturedErrorEvent).error.message));
  };
  device.addEventListener('uncapturederror', onErr);
  device.pushErrorScope('validation');
  device.pushErrorScope('internal');

  const N = 1024;
  const out = device.createBuffer({ size: N * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const read = device.createBuffer({ size: N * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const mod = device.createShaderModule({
    code: `@group(0) @binding(0) var<storage, read_write> o : array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g : vec3<u32>) {
  var acc = 0.0;
  for (var i = 0u; i < 16u; i = i + 1u) { acc = acc + f32(g.x) * 0.5 + f32(i); }
  o[g.x] = acc;
}`,
  });
  const info = await mod.getCompilationInfo();
  for (const m of info.messages) if (m.type === 'error') errors.push(`WGSL ${m.lineNum}: ${m.message}`);

  const pipe = device.createComputePipeline({ layout: 'auto', compute: { module: mod, entryPoint: 'main' } });
  const bg = device.createBindGroup({
    layout: pipe.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: out } }],
  });
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipe);
  pass.setBindGroup(0, bg);
  pass.dispatchWorkgroups(N / 64);
  pass.end();
  enc.copyBufferToBuffer(out, 0, read, 0, N * 4);
  device.queue.submit([enc.finish()]);

  await read.mapAsync(GPUMapMode.READ);
  const got = new Float32Array(read.getMappedRange().slice(0)) as Float32Array;
  read.unmap();

  const internal = await device.popErrorScope();
  const validation = await device.popErrorScope();
  if (internal) errors.push(`internal: ${internal.message}`);
  if (validation) errors.push(`validation: ${validation.message}`);
  device.removeEventListener('uncapturederror', onErr);

  // expected[x] = 16 * (x * 0.5) + (0 + 1 + ... + 15)
  let bad = -1;
  for (let x = 0; x < N; x += 1) {
    const want = 16 * (x * 0.5) + 120;
    if (Math.abs((got[x] ?? NaN) - want) > 1e-3) { bad = x; break; }
  }
  out.destroy();
  read.destroy();

  const computeOk = bad === -1 && errors.length === 0;
  return {
    computeOk,
    computeDetail: bad === -1
      ? `all ${N} values correct`
      : `wrong at index ${bad}: got ${got[bad]}, expected ${16 * (bad * 0.5) + 120}`,
    errors,
  };
}

export interface AllocResult {
  /** What the device SAYS it allows for one buffer, from its own limits. */
  maxBufferSizeMiB: number;
  largestSingleMiB: number;
  totalMiB: number;
  neededSingleMiB: number;
  neededTotalMiB: number;
  fitsSingle: boolean;
  fitsTotal: boolean;
  note: string;
}

/**
 * How large a single buffer this device grants, and how much it grants in total. The model needs
 * roughly 2.1 GB resident, and its largest single tensor is the PLE table at about 1.17 GB, which
 * the engine splits when the adapter cannot bind it whole (src/tableSplit.ts).
 */
export async function allocationLadder(device: GPUDevice): Promise<AllocResult> {
  const MiB = 1024 * 1024;
  const NEEDED_SINGLE = 128;   // after the table split, the largest binding the engine asks for
  const NEEDED_TOTAL = 2200;   // the whole resident set, approximately

  const alive: GPUBuffer[] = [];
  // BOTH scopes. A size over `maxBufferSize` is a VALIDATION error, not an out of memory one, and
  // an out of memory scope alone silently reports success for a buffer the device refused. That
  // bug had this probe claiming a 2048 MiB buffer on an iPhone whose limit is 1024.
  const tryAlloc = async (bytes: number): Promise<boolean> => {
    device.pushErrorScope('out-of-memory');
    device.pushErrorScope('validation');
    let buf: GPUBuffer | null = null;
    try {
      buf = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE });
    } catch {
      await device.popErrorScope();
      await device.popErrorScope();
      return false;
    }
    const validation = await device.popErrorScope();
    const oom = await device.popErrorScope();
    if (validation || oom) { buf.destroy(); return false; }
    alive.push(buf);
    return true;
  };

  // Largest single binding, doubling, and never past what the device says it allows: asking for
  // more only produces a validation error we already know the answer to.
  const cap = Math.floor(device.limits.maxBufferSize / MiB);
  let largest = 0;
  for (const mib of [64, 128, 256, 512, 1024, 2048]) {
    if (mib > cap) break;
    const ok = await tryAlloc(mib * MiB);
    if (!ok) break;
    largest = mib;
    const b = alive.pop();
    b?.destroy();
  }

  // total, in 128 MiB chunks held at once
  let total = 0;
  for (let i = 0; i < 24; i += 1) {
    const ok = await tryAlloc(128 * MiB);
    if (!ok) break;
    total += 128;
  }
  for (const b of alive) b.destroy();
  alive.length = 0;

  return {
    maxBufferSizeMiB: Math.floor(device.limits.maxBufferSize / MiB),
    largestSingleMiB: largest,
    totalMiB: total,
    neededSingleMiB: NEEDED_SINGLE,
    neededTotalMiB: NEEDED_TOTAL,
    fitsSingle: largest >= NEEDED_SINGLE,
    fitsTotal: total >= NEEDED_TOTAL,
    // Creation granted is not residency proven: a device can accept a buffer and fail when the
    // whole working set is live. Read this as an upper bound.
    note: total >= NEEDED_TOTAL
      ? 'this device can hold the model'
      : `this device granted ${total} MiB of the roughly ${NEEDED_TOTAL} MiB the model needs`,
  };
}
