# Gemma 4 WebGPU Engine

A WebGPU inference engine for Google's **Gemma 4 E2B**, running in the browser.
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

Speed figures published here earlier have been withdrawn, and a new measurement is not published
yet. The bench page in this repo measures your own hardware.

One optimization in the engine comes from the model's structure. Gemma 4 E2B computes its KV cache
in 15 of its 35 layers and the other 20 only consume it, so during prefill every position except the
last needs only the producer half, and the engine skips the rest. The generated tokens do not change.
The time this saves has not been measured on the current build.

This optimization is not new: vLLM and onnxruntime both ship a version of it.

## Requirements

- WebGPU with the `shader-f16` feature. Chrome 121+ or Edge on Windows and macOS; Safari 26+.
- **A `maxBufferSize` of 128 MiB or more**, which is the WebGPU specification floor, so every
  conformant adapter qualifies. The per layer embedding table is 1,174,405,120 bytes and no adapter
  holds that in one buffer: iOS grants exactly 1 GiB and the specification default is 256 MiB. The
  engine splits the table by vocabulary row and gathers once per slice, so the table's size stops
  being a limit on which devices can run it (`tableSplit.ts`). At iOS's 1 GiB that is two slices and
  one extra dispatch per gather. Adapters with room for the whole table allocate it whole and run
  the plan they always ran.
- About **2 GB** of download on first run, cached afterwards, and roughly 3 GB of free memory.

## Current engine snapshot

The engine is synced from the private Purrview repository. The exact revision is recorded in
[engine-source.json](engine-source.json) and included in copied benchmark results. This update
includes the latest projection batching, exact integer splits, and device profile tuning.
Blackwell-specific settings remain confined to that profile; Mac hardware uses its Apple profile.

On the RTX 5070, the latest eight-prompt runs measured 296.5 to 296.95 tokens/sec against
272.75 to 273.25 in matching controls, with identical generated IDs. Those numbers use the
Purrview benchmark harness, not this page's three-prompt test.

## Try it

Open the [live benchmark](https://ar5en1c.github.io/gemma4-webgpu-engine/) on your Mac, choose
**Profile this device**, then **Run benchmark**. Keep the page in front until it finishes and use
**Copy result JSON** to save the build, GPU details, generated text, and timings. If the page
reports an older cached build, use its **Load the current build** link.

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

TypeScript and WGSL source under `src/`, with no runtime dependencies.

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
project's own measurement record. It was written without copying code from any other inference
engine.

## Licence

Apache-2.0, see [LICENSE](LICENSE). The Gemma 4 weights are Google's and are also Apache-2.0, with
no acceptance gate, so this page fetches them directly.
