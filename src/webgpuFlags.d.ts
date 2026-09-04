// Written from the WebGPU specification, section "GPUBufferUsage", "GPUShaderStage" and
// "GPUMapMode". Numbers transcribed from the IDL, which is a fact about the platform.
//
// SPDX-License-Identifier: Apache-2.0
//
// TypeScript 7's DOM library declares the WebGPU interfaces, `GPUDevice` and `GPUBuffer` and the
// rest, but not the three namespace objects that carry the usage flags, so every file that says
// `GPUBufferUsage.STORAGE` fails to compile against a browser that has had the value since 2023.
// These are the platform's own constants, declared here once for the whole engine rather than cast
// away at each of the several dozen call sites.
//
// If a future TypeScript ships these, delete this file rather than keeping both.

declare const GPUBufferUsage: {
  readonly MAP_READ: number;
  readonly MAP_WRITE: number;
  readonly COPY_SRC: number;
  readonly COPY_DST: number;
  readonly INDEX: number;
  readonly VERTEX: number;
  readonly UNIFORM: number;
  readonly STORAGE: number;
  readonly INDIRECT: number;
  readonly QUERY_RESOLVE: number;
};

declare const GPUShaderStage: {
  readonly VERTEX: number;
  readonly FRAGMENT: number;
  readonly COMPUTE: number;
};

declare const GPUMapMode: {
  readonly READ: number;
  readonly WRITE: number;
};
