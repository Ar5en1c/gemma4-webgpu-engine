// SPDX-License-Identifier: Apache-2.0
//
// The bench page's driver. Two stages, deliberately separate: a profile that any device can run
// with no download, and a benchmark that needs the weights. Every number this page reports is
// measured here in the page, and the prompts are fixed so two devices can be compared.

import { Gemma4Mobile, DEFAULT_MODEL_ID } from '../src/index';
import type { Gemma4Message, Gemma4Progress } from '../src/index';
import { smokeTest, allocationLadder, type SmokeResult, type AllocResult } from './probe';

const $ = (id: string): HTMLElement => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing #${id}`);
  return el;
};

/** Fixed so two devices are comparable. Short, medium and long prefill, one generation length. */
const PROMPTS: { name: string; text: string; expect?: string[] }[] = [
  // `expect` is a coherence canary. A run that reports 44 tok/s on fluent nonsense passes a rate
  // check and a non empty check; it does not pass this. The words are ones any correct answer to
  // the prompt contains, matched case insensitively, and only the short prompt carries them
  // because only a closed question has an answer you can assert on.
  {
    name: 'short',
    text: 'In one sentence, what is photosynthesis?',
    expect: ['photosynth', 'plant', 'light'],
  },
  {
    name: 'medium',
    text:
      'You are interviewing a candidate for a backend engineering role. They have just described '
      + 'a caching layer they built. Ask one follow up question that would tell you whether they '
      + 'understand cache invalidation, and explain in two sentences why you chose it.',
  },
  {
    name: 'long',
    text:
      'Read the following and summarise the argument in three bullet points. '
      + 'On device inference moves the model to the user rather than the user to the model. '
      + 'That changes the privacy story, because nothing typed has to leave the machine, and it '
      + 'changes the cost story, because there is no per token bill. It also changes the '
      + 'engineering story: memory bandwidth becomes the budget, the weights must be quantized '
      + 'to fit, and the first token is paid for by prefill rather than by a network round trip. '
      + 'The hardest constraint is not speed but download size, since a user will abandon a page '
      + 'long before a model finishes arriving. Every design choice below follows from that.',
  },
];

const MAX_NEW_TOKENS = 64;

interface PromptResult {
  name: string;
  /** null when the prompt carries no canary. */
  coherent: boolean | null;
  ttftMs: number;
  tokens: number;
  decodeTokPerSec: number;
  totalMs: number;
  text: string;
}

declare const __BUILD_ID__: string;
declare const __BUILD_AT__: string;

interface Result {
  /** Which build produced this. A pasted result with no build is a stale page. */
  build: { id: string; at: string };
  /** FAILED means do not quote any number below it. See `failures`. */
  verdict: 'ok' | 'FAILED';
  failures: string[];
  measuredAt: string;
  userAgent: string;
  modelId: string;
  adapter: unknown;
  device: unknown | null;
  loadSeconds: number;
  maxNewTokens: number;
  reducePolicy: string | null;
  /** What the GPU says about itself. A lost device makes every number above meaningless. */
  gpu: { lostReason: string | null; errors: readonly string[] };
  /** Bytes the loader says it delivered. A load that "succeeded" without moving bytes did not. */
  load: { totalBytes: number; cachedBytes: number; fetchedBytes: number } | null;
  smoke: SmokeResult | null;
  allocation: AllocResult | null;
  prompts: PromptResult[];
}

let result: Result | null = null;
let smoke: SmokeResult | null = null;
let alloc: AllocResult | null = null;

/**
 * A run is only reportable if it produced text and moved at a physically possible rate. An iPhone
 * once reported 3705 tok/s on empty output because the dispatches were silently not running; a
 * page that prints that number is worse than a page that prints nothing.
 */
const PLAUSIBLE_MAX_TOK_PER_SEC = 500;

function validate(rows: PromptResult[]): string[] {
  const failures: string[] = [];
  for (const r of rows) {
    if (r.text.trim() === '') {
      failures.push(`prompt "${r.name}" produced no text, so the engine generated nothing`);
    }
    if (r.decodeTokPerSec > PLAUSIBLE_MAX_TOK_PER_SEC) {
      failures.push(`prompt "${r.name}" reported ${r.decodeTokPerSec} tok/s, above the `
        + `${PLAUSIBLE_MAX_TOK_PER_SEC} tok/s plausibility ceiling, so the work did not run`);
    }
    if (r.decodeTokPerSec <= 0) failures.push(`prompt "${r.name}" reported no decode rate`);
    if (r.coherent === false) {
      failures.push(`prompt "${r.name}" produced text that mentions none of its expected words, `
        + 'so the engine is emitting fluent nonsense rather than an answer');
    }
  }
  return failures;
}

// ------------------------------------------------------------------ stage 1, the device profile

async function profile(): Promise<Record<string, unknown> | null> {
  const out = $('profileOut');
  if (!navigator.gpu) {
    out.innerHTML = '<p class="bad">WebGPU is not available in this browser. '
      + 'The engine cannot run here.</p>';
    return null;
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    out.innerHTML = '<p class="bad">No WebGPU adapter. The browser has the API but no usable GPU.</p>';
    return null;
  }
  const features = [...adapter.features].sort();
  const limits: Record<string, number> = {};
  for (const key of [
    'maxComputeWorkgroupStorageSize',
    'maxComputeInvocationsPerWorkgroup',
    'maxStorageBufferBindingSize',
    'maxBufferSize',
  ]) {
    const v = (adapter.limits as unknown as Record<string, number | undefined>)[key];
    if (typeof v === 'number') limits[key] = v;
  }
  const info = (adapter as unknown as { info?: Record<string, unknown> }).info ?? {};
  const sub = adapter as unknown as { subgroupMinSize?: number; subgroupMaxSize?: number };

  const need = ['shader-f16'];
  const missing = need.filter((f) => !adapter.features.has(f as GPUFeatureName));
  const canRun = missing.length === 0;

  const rows: [string, string][] = [
    ['vendor', String(info['vendor'] ?? 'not reported')],
    ['architecture', String(info['architecture'] ?? 'not reported')],
    ['device', String(info['device'] ?? 'not reported')],
    ['description', String(info['description'] ?? 'not reported')],
    ['subgroup size range', sub.subgroupMinSize != null
      ? `${sub.subgroupMinSize} to ${sub.subgroupMaxSize}` : 'not reported'],
    ['shader-f16', adapter.features.has('shader-f16' as GPUFeatureName) ? 'yes' : 'NO, required'],
    ['subgroups', adapter.features.has('subgroups' as GPUFeatureName) ? 'yes' : 'no'],
    ['timestamp-query', adapter.features.has('timestamp-query' as GPUFeatureName) ? 'yes' : 'no'],
    ['all features', features.join(', ') || 'none'],
  ];
  for (const [k, v] of Object.entries(limits)) rows.push([k, String(v)]);

  out.innerHTML = `<div class="scroll"><table>${
    rows.map(([k, v]) => `<tr><th>${k}</th><td class="n">${escapeHtml(v)}</td></tr>`).join('')
  }</table></div>` + (canRun
    ? ''
    : `<p class="bad" style="margin-bottom:0">Missing required feature: ${missing.join(', ')}. `
      + 'The engine will not run here.</p>');

  if (!canRun) return { info, features, limits };

  // The cheap probes. These are why a failure is diagnosable without a 2 GB download.
  out.insertAdjacentHTML('beforeend', '<p id="probing">Running GPU smoke test and allocation ladder...</p>');
  let device: GPUDevice | null = null;
  try {
    device = await adapter.requestDevice({ requiredFeatures: ['shader-f16' as GPUFeatureName] });
  } catch (err) {
    $('probing').outerHTML = `<p class="bad">The adapter refused a device: ${escapeHtml(String(err))}</p>`;
    return { info, features, limits };
  }
  smoke = await smokeTest(device);
  alloc = await allocationLadder(device);
  device.destroy();

  const memOk = alloc.fitsTotal && alloc.fitsSingle;
  $('probing').outerHTML = `<div class="scroll"><table>
    <tr><th>compute correctness</th><td class="n ${smoke.computeOk ? 'ok' : 'bad'}">${
      smoke.computeOk ? 'PASS' : 'FAIL'} ${escapeHtml(smoke.computeDetail)}</td></tr>
    <tr><th>GPU errors</th><td class="n">${smoke.errors.length ? escapeHtml(smoke.errors.join(' | ')) : 'none'}</td></tr>
    <tr><th>largest single buffer</th><td class="n">${alloc.largestSingleMiB} MiB</td></tr>
    <tr><th>total granted</th><td class="n ${memOk ? 'ok' : 'bad'}">${alloc.totalMiB} MiB of about ${alloc.neededTotalMiB} MiB needed</td></tr>
  </table></div><p class="${smoke.computeOk && memOk ? 'ok' : 'bad'}" style="margin-bottom:0">${
    smoke.computeOk && memOk
      ? 'This device can run the engine.'
      : escapeHtml(!smoke.computeOk
          ? 'This device miscompiles or cannot run the compute path. The benchmark would produce garbage.'
          : alloc.note + '. The benchmark will not fit and is disabled.')}</p>`;

  if (smoke.computeOk && memOk) ($('bench') as HTMLButtonElement).disabled = false;
  return { info, features, limits, subgroupMinSize: sub.subgroupMinSize, subgroupMaxSize: sub.subgroupMaxSize };
}

// ---------------------------------------------------------------------- stage 2, the benchmark

async function bench(adapterInfo: Record<string, unknown> | null): Promise<void> {
  const out = $('benchOut');
  const btn = $('bench') as HTMLButtonElement;
  btn.disabled = true;
  out.innerHTML = '<p>Loading weights. First run downloads about 2 GB.</p>'
    + '<progress id="pg" max="1" value="0"></progress><pre id="pgt"></pre>';
  const pg = $('pg') as HTMLProgressElement;
  const pgt = $('pgt');

  const t0 = performance.now();
  let engine: Gemma4Mobile;
  try {
    engine = await Gemma4Mobile.load(null, {
      onProgress: (p: Gemma4Progress) => {
        if (typeof p.fraction === 'number') pg.value = p.fraction;
        const mb = typeof p.loaded === 'number' && typeof p.total === 'number' && p.total > 0
          ? `${(p.loaded / 1e6).toFixed(0)} of ${(p.total / 1e6).toFixed(0)} MB`
          : '';
        pgt.textContent = [p.status, p.message ?? '', mb, p.fromCache ? '(from cache)' : '']
          .filter(Boolean).join('  ');
      },
    });
  } catch (err) {
    // A refused load is the most informative result this page can produce, so it must be
    // pasteable. Before this, the guard fired and the page handed back nothing to copy.
    result = {
      build: { id: __BUILD_ID__, at: __BUILD_AT__ },
      verdict: 'FAILED',
      failures: [`the model refused to load: ${String(err)}`],
      measuredAt: new Date().toISOString(),
      userAgent: navigator.userAgent,
      modelId: DEFAULT_MODEL_ID,
      adapter: adapterInfo,
      device: null,
      loadSeconds: Math.round(((performance.now() - t0) / 1000) * 10) / 10,
      maxNewTokens: MAX_NEW_TOKENS,
      reducePolicy: null,
      gpu: { lostReason: null, errors: [] },
      load: null,
      smoke,
      allocation: alloc,
      prompts: [],
    };
    out.innerHTML = `<p class="bad"><strong>The model refused to load.</strong> This is a real `
      + 'result, not a crash: press Copy result JSON and send it.</p>'
      + `<pre>${escapeHtml(String(err))}</pre>`;
    ($('copy') as HTMLButtonElement).disabled = false;
    btn.disabled = false;
    return;
  }
  const loadSeconds = (performance.now() - t0) / 1000;

  const results: PromptResult[] = [];
  out.innerHTML = `<p>Loaded in ${loadSeconds.toFixed(1)} s. Running ${PROMPTS.length} prompts, `
    + `${MAX_NEW_TOKENS} new tokens each.</p><div id="live"></div>`;
  const live = $('live');

  const hardFailures: string[] = [];
  for (const prompt of PROMPTS) {
    // Independent prompts, so the KV cache from the last one must not survive into this one.
    engine.reset();
    const messages: Gemma4Message[] = [{ role: 'user', content: prompt.text }];
    const start = performance.now();
    let firstAt = 0;
    let tokens = 0;
    let text = '';
    try {
      for await (const chunk of engine.generate(messages, { maxNewTokens: MAX_NEW_TOKENS })) {
        if (tokens === 0) firstAt = performance.now();
        tokens += 1;
        text = chunk.text;
      }
    } catch (err) {
      hardFailures.push(`prompt "${prompt.name}" threw: ${String(err)}`);
    }
    const end = performance.now();
    const ttftMs = firstAt - start;
    // Decode rate excludes the first token, which is prefill, so this is the steady state.
    const decodeTokPerSec = tokens > 1 ? ((tokens - 1) / ((end - firstAt) / 1000)) : 0;
    const lower = text.toLowerCase();
    const coherent = prompt.expect ? prompt.expect.some((w) => lower.includes(w)) : null;
    results.push({
      name: prompt.name,
      coherent,
      ttftMs: Math.round(ttftMs),
      tokens,
      decodeTokPerSec: Math.round(decodeTokPerSec * 100) / 100,
      totalMs: Math.round(end - start),
      text,
    });
    live.innerHTML = renderTable(results);
  }

  const gpu = engine.deviceErrors();
  const receipt = engine.loadReceipt();
  const loadBytes = receipt
    ? {
      totalBytes: receipt.totalBytes,
      cachedBytes: receipt.cachedBytes,
      fetchedBytes: receipt.fetchedBytes,
    }
    : null;
  const failures = [...hardFailures, ...validate(results)];
  if (gpu.lostReason) {
    failures.unshift(`the GPU device was lost (${gpu.lostReason}). Every dispatch after that is a `
      + 'no op that returns zeros, which is why the rates above are impossible and the text empty.');
  }
  for (const e of gpu.errors) failures.push(`uncaptured GPU error: ${e}`);
  if (loadBytes && loadBytes.cachedBytes + loadBytes.fetchedBytes < loadBytes.totalBytes) {
    failures.push(`the loader delivered ${loadBytes.cachedBytes + loadBytes.fetchedBytes} of `
      + `${loadBytes.totalBytes} bytes, so the weights are incomplete`);
  }
  result = {
    build: { id: __BUILD_ID__, at: __BUILD_AT__ },
    verdict: failures.length === 0 ? 'ok' : 'FAILED',
    failures,
    measuredAt: new Date().toISOString(),
    userAgent: navigator.userAgent,
    modelId: DEFAULT_MODEL_ID,
    adapter: adapterInfo,
    device: engine.deviceInfo(),
    loadSeconds: Math.round(loadSeconds * 10) / 10,
    maxNewTokens: MAX_NEW_TOKENS,
    reducePolicy: engine.reducePolicy(),
    gpu,
    load: loadBytes,
    smoke,
    allocation: alloc,
    prompts: results,
  };
  const banner = failures.length === 0
    ? '<p class="ok"><strong>Run valid.</strong> These numbers are reportable.</p>'
    : '<p class="bad"><strong>RUN FAILED. Do not quote these numbers.</strong></p><ul class="bad">'
      + failures.map((f) => `<li>${escapeHtml(f)}</li>`).join('') + '</ul>';
  live.innerHTML = banner + renderTable(results)
    + '<h2 style="margin-top:1.2rem">Output, so you can see it is coherent</h2>'
    + results.map((r) => `<pre><strong>${r.name}</strong>\n${escapeHtml(r.text)}</pre>`).join('<hr style="border:0;border-top:1px solid var(--line);margin:.8rem 0">');
  ($('copy') as HTMLButtonElement).disabled = false;
  btn.disabled = false;
}

function renderTable(rows: PromptResult[]): string {
  return '<div class="scroll"><table><tr><th>prompt</th><th>TTFT ms</th>'
    + '<th>decode tok/s</th><th>total ms</th></tr>'
    + rows.map((r) => `<tr><td>${r.name}</td><td class="n">${r.ttftMs}</td>`
      + `<td class="n">${r.decodeTokPerSec.toFixed(2)}</td><td class="n">${r.totalMs}</td></tr>`).join('')
    + '</table></div>';
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c] ?? c));
}

let adapterInfo: Record<string, unknown> | null = null;
$('profile').addEventListener('click', () => { void profile().then((p) => { adapterInfo = p; }); });
$('bench').addEventListener('click', () => { void bench(adapterInfo); });
$('copy').addEventListener('click', () => {
  if (result) void navigator.clipboard.writeText(JSON.stringify(result, null, 2));
});

// The stamp, visible so a screenshot carries it as well as the JSON.
{
  const el = document.createElement('div');
  el.style.cssText = 'color:var(--muted);font-size:12px;margin-top:.6rem';
  el.textContent = `build ${__BUILD_ID__}, ${__BUILD_AT__}`;
  document.querySelector('footer')?.appendChild(el);
}
