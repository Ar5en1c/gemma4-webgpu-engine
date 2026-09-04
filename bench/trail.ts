// SPDX-License-Identifier: Apache-2.0
//
// A breadcrumb trail that survives the tab dying.
//
// WHY THIS EXISTS. On iOS the failure mode is not an exception. Safari kills the WebContent
// process and shows "cannot open this page", sometimes reloading first, and everything the page
// knew goes with it: no console, no thrown error, no result object, nothing to copy. A device
// report that says "it crashed" is not a bug report, and asking someone to reproduce a five minute
// download to learn one more fact is not a diagnostic loop.
//
// So each step writes one line to localStorage before it starts the work. localStorage is
// synchronous and persisted, so the line is on disk before the step that might not return. The
// next load reads the trail back, sees that it never reached the end, and shows what the last
// completed step was.
//
// The trail is deliberately small and dull: a phase name, a millisecond timestamp, and a few
// numbers. It records what the page did, never what anybody typed.

const KEY = 'g4.trail.v1';
/** Enough to localize any crash, small enough that the write stays cheap. */
const MAX_ENTRIES = 240;

export interface Crumb {
  /** Milliseconds since the trail was started, so the gaps are readable at a glance. */
  ms: number;
  phase: string;
  [k: string]: unknown;
}

let started = 0;
let cache: Crumb[] = [];

/** localStorage throws in some privacy modes, and a diagnostic that breaks the page is worse than none. */
function write(entries: Crumb[]): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(entries));
  } catch {
    /* no trail on this device, which is itself survivable */
  }
}

/** Read whatever the last run left behind, before this run overwrites it. */
export function readTrail(): Crumb[] {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    return Array.isArray(parsed) ? (parsed as Crumb[]) : [];
  } catch {
    return [];
  }
}

export function clearTrail(): void {
  try {
    localStorage.removeItem(KEY);
  } catch {
    /* nothing to clear */
  }
}

/** Begin a fresh trail. Anything the previous run left has to be read before this. */
export function startTrail(context: Record<string, unknown>): void {
  started = Date.now();
  cache = [{ ms: 0, phase: 'start', ...context }];
  write(cache);
}

/**
 * Record one step, synchronously, BEFORE the work it names.
 *
 * A crumb written after a step tells you the step finished, which is the half you would have
 * learned anyway from the next crumb. Written before, the last crumb in a killed trail names the
 * thing that killed it.
 */
export function crumb(phase: string, data?: Record<string, unknown>): void {
  if (started === 0) startTrail({});
  cache.push({ ms: Date.now() - started, phase, ...(data ?? {}) });
  // Keep the beginning, which carries the device, and the end, which carries the crash.
  if (cache.length > MAX_ENTRIES) cache.splice(8, cache.length - MAX_ENTRIES);
  write(cache);
}

/** The run reached its own end, so the trail is not evidence of anything. */
export function endTrail(): void {
  crumb('complete');
}

/** True when a trail exists and never reached `complete`, which is the case worth showing. */
export function trailIsUnfinished(entries: readonly Crumb[]): boolean {
  return entries.length > 0 && entries[entries.length - 1]?.phase !== 'complete';
}
