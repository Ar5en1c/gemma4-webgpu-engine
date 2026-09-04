# Gemma 4 WebGPU Engine

A clean-room WebGPU inference engine for Google's **Gemma 4 E2B**, running entirely in the browser.
No server, no WASM runtime, no ONNX. It fetches the quantized weights from Hugging Face, keeps them
in IndexedDB, and does every matmul in WGSL compute shaders.

**Status: alpha.** It runs, it is measured against the transformers reference, and it is not yet a
package on npm. The API is small and will change.

## Why it exists

It was written to replace a vendored third-party engine in a product that promises nothing leaves
the machine. That promise is only worth what the code behind it is worth, so the engine was written
from scratch, from Google's published architecture rather than from anyone else's implementation.
See [NOTICE](NOTICE).

## Where it stands

Measured on one M1 MacBook, both engines interleaved in the same session, eight app prompts,
against the engine it was written to replace:

| | result |
|---|---|
| time to first token | **1.73x to 2.33x faster**, all eight prompts |
| decode tokens/sec | 1.005x to 1.024x faster, all eight prompts |
| warm turn, end to end | **1.89x faster** |

Read those as ratios on one machine in one session, which is what an interleaved A/B licenses. They
are not a portable claim about your hardware, which is exactly what the bench page in this repo is
for.

The time to first token result comes from one architectural property rather than from kernel
tuning. Gemma 4 E2B computes its KV cache in 15 of its 35 layers and the other 20 only consume it,
so during prefill every position except the last needs only the producer half. Skipping the rest
removes **59.0 percent of a prefill chunk**, exactly and bit identically. Three months of kernel
work before that moved the same number about 1.15x.

To be clear about what is and is not new: this optimization is known, and vLLM and onnxruntime both
ship a version of it. As far as we can tell this is the first **browser** engine to do it.

## Requirements

- WebGPU with the `shader-f16` feature. Chrome 121+ or Edge on Windows and macOS; Safari 26+;
  Chrome on Android with WebGPU enabled.
- About **2 GB** of download on first run, cached afterwards, and roughly 3 GB of free memory.
  Most phones will not have the memory even where the browser supports WebGPU.

## Try it

The bench page profiles your GPU first, with no download, and only fetches weights if you ask it to.

```bash
npm install
npm run dev
```

## Use it

```ts
import { Gemma4Mobile } from 'gemma4-webgpu-engine';

const engine = await Gemma4Mobile.load(null, {
  onProgress: (p) => console.log(p.loaded, p.total),
});

for await (const chunk of engine.generate(
  [{ role: 'user', content: 'Explain cache invalidation in two sentences.' }],
  { maxNewTokens: 128 },
)) {
  process.stdout.write(chunk.delta);
}
```

`load(null, ...)` uses `google/gemma-4-E2B-it-qat-mobile-transformers`. Pass a repo id, or a URL, to
point it somewhere else.

## How it is built

39 source files, about 18,500 lines of TypeScript and WGSL, no runtime dependencies.

- `plan.ts` builds a forward pass as a list of dispatches before anything touches the GPU, which is
  what makes the prefill exit a scheduling change rather than a kernel change.
- `kernels/` holds the WGSL. Every reduction kernel exists in a subgroup and a workgroup form, and
  the engine proves the subgroup path with a known answer self test at load rather than trusting
  the adapter's feature list.
- `quant.ts` and `safetensors.ts` read Google's `wNa8o8` mobile quantization directly, including
  the static activation scales, with ranged fetches so nothing downloads twice.

## Provenance

Written as an independent implementation from Google's published Gemma 4 architecture and config
files, the safetensors format, the Hugging Face `transformers` reference (Apache-2.0), and this
project's own measurement record. **It contains no code from any third-party inference engine**, and
no such engine's source or shaders were read during development. That rule was enforced throughout,
including while benchmarking against one.

## Licence

Apache-2.0, see [LICENSE](LICENSE). The Gemma 4 weights are Google's and are also Apache-2.0, with
no acceptance gate, so this page fetches them directly.
