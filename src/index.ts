// Written from docs/ENGINE-PLAN.md section 9, the API contract in
// src/engine/llm/gemma4-engine.d.ts, and this engine's own modules. No vendored bundle, no
// extracted kernel and no third party engine source was read.
//
// SPDX-License-Identifier: Apache-2.0
//
// The engine's one public entry point, written by the integrator because section 8 gives the
// entry module to the integrator and nobody else.
//
// ENGINE-PLAN section 9 says the eventual swap is a single alias change in vite.config.ts, from
// the vendored path to `src/engine/gemma4/index.ts`, with the bare specifier `@gemma4/engine`
// unchanged. That only works if this file exists and its exported surface is exactly what
// gemma4-engine.d.ts declares, so this file is that surface and nothing else. It adds no
// behaviour: everything below is re-exported from the module that owns it.
//
// The alias is NOT flipped this round and this file is not imported by the app. It is here so the
// swap is a one line change on the day gate 2 and gate 3 pass, rather than a scramble to invent an
// entry point under schedule pressure. scripts/engine-check/integration.mjs asserts that every
// name gemma4-engine.d.ts declares is exported from here, so the two cannot drift apart quietly.

export {
  Gemma4Mobile,
  Gemma4Mobile as default,
  DEFAULT_MODEL_ID,
  resolveModelRoot,
} from './engine';

export type {
  Gemma4Message,
  Gemma4Progress,
  Gemma4RuntimeOptions,
  Gemma4LoadOptions,
  Gemma4GenerateOptions,
  Gemma4Chunk,
  Gemma4DeviceInfo,
} from './engine';
