// SPDX-License-Identifier: Apache-2.0
//
// The bench page's driver. Two stages, deliberately separate: a profile that any device can run
// with no download, and a benchmark that needs the weights. Every number this page reports is
// measured here in the page, and the prompts are fixed so two devices can be compared.

import { Gemma4Mobile, DEFAULT_MODEL_ID } from '../src/index';
import type { Gemma4Message, Gemma4Progress } from '../src/index';

const $ = (id: string): HTMLElement => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing #${id}`);
  return el;
};

/** Fixed so two devices are comparable. Short, medium and long prefill, one generation length. */
const PROMPTS: { name: string; text: string }[] = [
  { name: 'short', text: 'In one sentence, what is photosynthesis?' },
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
  ttftMs: number;
  tokens: number;
  decodeTokPerSec: number;
  totalMs: number;
  text: string;
}

interface Result {
  measuredAt: string;
  userAgent: string;
  modelId: string;
  adapter: unknown;
  device: unknown;
  loadSeconds: number;
  maxNewTokens: number;
  prompts: PromptResult[];
}

let result: Result | null = null;

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
    ? '<p class="ok" style="margin-bottom:0">This device can run the engine.</p>'
    : `<p class="bad" style="margin-bottom:0">Missing required feature: ${missing.join(', ')}. `
      + 'The engine will not run here.</p>');

  if (canRun) ($('bench') as HTMLButtonElement).disabled = false;
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
    out.innerHTML = `<p class="bad">Load failed: ${escapeHtml(String(err))}</p>`;
    btn.disabled = false;
    return;
  }
  const loadSeconds = (performance.now() - t0) / 1000;

  const results: PromptResult[] = [];
  out.innerHTML = `<p>Loaded in ${loadSeconds.toFixed(1)} s. Running ${PROMPTS.length} prompts, `
    + `${MAX_NEW_TOKENS} new tokens each.</p><div id="live"></div>`;
  const live = $('live');

  for (const prompt of PROMPTS) {
    const messages: Gemma4Message[] = [{ role: 'user', content: prompt.text }];
    const start = performance.now();
    let firstAt = 0;
    let tokens = 0;
    let text = '';
    for await (const chunk of engine.generate(messages, { maxNewTokens: MAX_NEW_TOKENS })) {
      if (tokens === 0) firstAt = performance.now();
      tokens += 1;
      text = chunk.text;
    }
    const end = performance.now();
    const ttftMs = firstAt - start;
    // Decode rate excludes the first token, which is prefill, so this is the steady state.
    const decodeTokPerSec = tokens > 1 ? ((tokens - 1) / ((end - firstAt) / 1000)) : 0;
    results.push({
      name: prompt.name,
      ttftMs: Math.round(ttftMs),
      tokens,
      decodeTokPerSec: Math.round(decodeTokPerSec * 100) / 100,
      totalMs: Math.round(end - start),
      text,
    });
    live.innerHTML = renderTable(results);
  }

  result = {
    measuredAt: new Date().toISOString(),
    userAgent: navigator.userAgent,
    modelId: DEFAULT_MODEL_ID,
    adapter: adapterInfo,
    device: engine.deviceInfo(),
    loadSeconds: Math.round(loadSeconds * 10) / 10,
    maxNewTokens: MAX_NEW_TOKENS,
    prompts: results,
  };
  live.innerHTML = renderTable(results)
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
