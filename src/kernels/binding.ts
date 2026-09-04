// Written from docs/ENGINE-PLAN.md sections 5 and 8, the WebGPU specification, and the round 1
// verifier's seam finding (KernelBindResult could not express sub range bindings and every bind
// created a throwaway layout and uniform). No vendored bundle, no extracted kernel and no third
// party engine source was read.
//
// SPDX-License-Identifier: Apache-2.0
//
// The binding seam shared by every kernel's `bind` and by both of its callers, the harness page
// and the engine's scheduler. This lives outside registry.ts on purpose: registry.ts imports the
// kernel modules, the kernel modules import these helpers as values, and a value import back into
// registry.ts would be a runtime import cycle, which the Node check loaders refuse by design.
// Types flow through registry.ts as type only imports, which are erased.

import type { KernelBindInput } from './registry';

/**
 * One binding slot. A bare buffer binds whole; the object form binds a sub range, which is what
 * the engine's scheduler needs for arena staged uniforms and sliced activations. Round 1's
 * verifier found the bare `GPUBuffer[]` could not express that, which made registry.ts's one seam
 * claim false; this union is the fix, and plain buffers stay assignable so no kernel changed
 * for it.
 */
export type KernelBinding = GPUBuffer | { buffer: GPUBuffer; offset?: number; size?: number };

/** A staged uniform slice inside a caller owned arena (buffers.ts BufferManager.stageUniform). */
export interface UniformSlice {
  buffer: GPUBuffer;
  offset: number;
  size: number;
}

/** Normalize a KernelBinding to the bind group entry resource shape. */
export function bindingEntry(b: KernelBinding): { buffer: GPUBuffer; offset?: number; size?: number } {
  return typeof (b as { buffer?: unknown }).buffer === 'object' && (b as { buffer?: unknown }).buffer !== null
    ? (b as { buffer: GPUBuffer; offset?: number; size?: number })
    : { buffer: b as GPUBuffer };
}

/** The underlying buffer of a KernelBinding, for usage checks and budget counting. */
export function bindingBuffer(b: KernelBinding): GPUBuffer {
  return bindingEntry(b).buffer;
}

/** A layout through the caller's cache when one was provided, or freshly made when not. */
export function kernelLayout(
  input: KernelBindInput,
  key: string,
  make: () => GPUBindGroupLayout,
): GPUBindGroupLayout {
  return input.layoutFor ? input.layoutFor(key, make) : make();
}

/**
 * A params block as a binding: staged into the caller's arena when the input carries one, or a
 * throwaway uniform buffer with a dispose when it does not. Every kernel's bind routes its params
 * through here, so the harness and the scheduler exercise the same code and round 1's
 * create-then-destroy-per-dispatch pattern has exactly one fallback home instead of thirteen
 * unconditional ones.
 */
export function kernelUniform(
  input: KernelBindInput,
  label: string,
  words: ArrayBuffer,
): { binding: KernelBinding; dispose?: () => void } {
  if (input.stageUniform) {
    return { binding: input.stageUniform(words) };
  }
  const buffer = input.device.createBuffer({
    label,
    size: words.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  input.device.queue.writeBuffer(buffer, 0, words);
  return { binding: buffer, dispose: () => buffer.destroy() };
}
