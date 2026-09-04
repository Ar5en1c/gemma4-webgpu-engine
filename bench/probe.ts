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
  /** Buffers the device HANDED OUT, which is not the same as memory it can hold. See below. */
  totalMiB: number;
  /**
   * Memory the device actually took writes into, chunk by chunk, before it refused or the page
   * died. This is the number that predicts whether the model loads.
   *
   * `createBuffer` succeeding proves nothing on iOS: the handle comes back and the pages are
   * committed lazily, on first write. So the granted ladder above happily reported thousands of
   * megabytes on a device that is killed partway through the actual upload, which is a probe that
   * says "this device can run the engine" and then watches the tab die. Writing to each chunk is
   * what turns a promise into a measurement.
   */
  committedMiB: number;
  /** True when the committing ladder stopped because the device refused, rather than finishing. */
  committedRefused: boolean;
  /**
   * True when the ladder stopped on its own after proving enough headroom. `committedMiB` is then a
   * floor, not a ceiling: the device took at least that much and was not asked for more.
   */
  committedStoppedEarly: boolean;
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
export async function allocationLadder(
  device: GPUDevice,
  onChunk?: (aboutToCommitMiB: number) => void,
): Promise<AllocResult> {
  const MiB = 1024 * 1024;
  const NEEDED_SINGLE = 128;   // after the table split, the largest binding the engine asks for
  const NEEDED_TOTAL = 2200;   // the whole resident set, approximately
  const CHUNK = 128;           // MiB per rung, on both ladders
  // One staging array, reused. Allocating a 128 MiB host array per chunk would measure the JS heap
  // as much as the device, and on a phone it would be the thing that fails.
  const stage = new Uint8Array(8 * MiB);

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

  // Granted: buffers the device hands out and holds at once, without a byte written to them.
  let total = 0;
  for (let i = 0; i < 24; i += 1) {
    const ok = await tryAlloc(CHUNK * MiB);
    if (!ok) break;
    total += CHUNK;
  }
  for (const b of alive) b.destroy();
  alive.length = 0;

  // Committed: the same ladder, but every chunk is written end to end before the next is asked
  // for, so the pages are really backed. onChunk is called BEFORE each rung, so a trail written by
  // the caller names the rung that killed the page rather than the last one that survived.
  // Stop once there is clear headroom over what the model needs, rather than climbing until the
  // device refuses. The question this ladder answers is "will 2.1 GB fit", not "where is the
  // ceiling", and every extra rung is memory held in the same process that is about to try the
  // real load. The first phone to run this committed every rung to the old cap of 3072 MiB and
  // then died during the load, which is a ladder that took a bite out of the thing it was
  // measuring.
  const COMMIT_CEILING = NEEDED_TOTAL + 384;
  const maxRungs = Math.ceil(COMMIT_CEILING / CHUNK);
  let committed = 0;
  let committedRefused = false;
  for (let i = 0; i < maxRungs; i += 1) {
    onChunk?.(committed + CHUNK);
    if (!(await tryAlloc(CHUNK * MiB))) { committedRefused = true; break; }
    const buf = alive[alive.length - 1]!;
    device.pushErrorScope('out-of-memory');
    for (let off = 0; off < CHUNK * MiB; off += stage.byteLength) {
      device.queue.writeBuffer(buf, off, stage);
    }
    // onSubmittedWorkDone is the only honest way to wait for the writes to land; without it the
    // ladder measures how fast a queue accepts work, which every device is instantly good at.
    await device.queue.onSubmittedWorkDone();
    if (await device.popErrorScope()) { committedRefused = true; break; }
    committed += CHUNK;
  }
  for (const b of alive) b.destroy();
  alive.length = 0;

  return {
    maxBufferSizeMiB: Math.floor(device.limits.maxBufferSize / MiB),
    largestSingleMiB: largest,
    totalMiB: total,
    neededSingleMiB: NEEDED_SINGLE,
    neededTotalMiB: NEEDED_TOTAL,
    committedMiB: committed,
    committedRefused,
    /** True when the ladder stopped because it had proved enough, not because the device refused. */
    committedStoppedEarly: !committedRefused && committed >= COMMIT_CEILING,
    fitsSingle: largest >= NEEDED_SINGLE,
    // The committed ladder decides this, not the granted one. A device that hands out buffers it
    // cannot back is exactly the device this probe exists to catch.
    fitsTotal: committed >= NEEDED_TOTAL,
    note: committed >= NEEDED_TOTAL
      ? `this device took writes into at least ${committed} MiB, enough for the model`
      : `this device granted ${total} MiB of buffers but only took writes into ${committed} MiB of `
        + `the roughly ${NEEDED_TOTAL} MiB the model needs`
        + (total > committed
          ? '. Buffer creation succeeding is not memory: the pages are committed on first write, '
            + 'which is where it stopped.'
          : ''),
  };
}

/**
 * How much HOST memory this tab is given, measured the same way the GPU ladder measures the device.
 *
 * WHY THIS IS A SEPARATE NUMBER. The allocation ladder above measures GPU buffers, and on the phone
 * that could not load the model it reached 2,688 MiB without complaint. The load then died at about
 * 1,479 MB, and a second engine from a different project died at about the same figure on the same
 * phone. Two engines failing at one number is a property of the platform, not of either engine, but
 * "the tab has a budget" and "the tab has a budget for GPU buffers" are different claims and only
 * one of them is consistent with a ladder that reached 2.7 GB.
 *
 * So this allocates ordinary ArrayBuffers and writes a byte to every page, which is what makes an
 * allocation real rather than reserved, and reports where it stopped. Run against the GPU figure it
 * says whether the two share one budget or hold separate ones, and that decides whether a 2.1 GB
 * model can be made to fit here at all or whether the resident set has to come down.
 *
 * This is expected to end the tab on a phone. That is the measurement, and the caller writes a
 * breadcrumb before each step so the number survives the process that produced it.
 */
export interface HostMemoryResult {
  reachedMiB: number;
  refused: boolean;
  note: string;
}

export async function hostMemoryLadder(
  ceilingMiB: number,
  onChunk?: (aboutToHoldMiB: number) => void,
): Promise<HostMemoryResult> {
  const MiB = 1024 * 1024;
  const CHUNK = 128;
  const held: Uint8Array[] = [];
  let reached = 0;
  let refused = false;
  for (let i = 0; i < Math.ceil(ceilingMiB / CHUNK); i += 1) {
    onChunk?.(reached + CHUNK);
    let block: Uint8Array;
    try {
      block = new Uint8Array(CHUNK * MiB);
    } catch {
      refused = true;
      break;
    }
    // Touch every 4 KiB page. An untouched allocation can be a reservation the system never backs,
    // which is the same mistake the GPU ladder made before it started writing to its buffers.
    for (let off = 0; off < block.byteLength; off += 4096) block[off] = 1;
    held.push(block);
    reached += CHUNK;
    // Yield, so the page can paint and the breadcrumb write is not starved by a tight loop.
    await new Promise((r) => setTimeout(r, 0));
  }
  held.length = 0;
  return {
    reachedMiB: reached,
    refused,
    note: refused
      ? `this tab refused a host allocation past ${reached} MiB`
      : `this tab held ${reached} MiB of host memory, the ceiling this probe was asked to try`,
  };
}
