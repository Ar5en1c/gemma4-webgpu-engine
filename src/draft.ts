// Written from ./kernels/qgemvWide.ts (GEMV_WIDE_COLS and the round 7 cost ledger it records),
// ./kv.ts (the truncate primitive a rejected draft rewinds with), and the prompt lookup decoding
// idea as described in its own public write ups. No vendored bundle, no extracted kernel and no
// third party engine source was read.
//
// SPDX-License-Identifier: Apache-2.0
//
// THE DRAFTER: guessing the next token for free.
//
// WHY THIS IS THE LEVER AND NOT KERNEL TUNING. A decode token reads 879 MB of weights to produce
// one token, and measurement says the GPU is busy 100% of the 26.3 ms that takes on this M1
// (.engine-ref/perf/ours-r4-final.json timing). So decode is bandwidth bound, the weight stream is
// the whole cost, and the only way to make a token cheaper is to make one weight stream produce
// more than one token. That is exactly what kernels/qgemvWide.ts was built for: one weight read,
// GEMV_WIDE_COLS activation columns, measured at 1.23x the one column cost for two columns. A
// verify pass therefore pays for itself whenever the drafter is right more than about a quarter of
// the time.
//
// WHY LOOKUP AND NOT A DRAFT MODEL. A second model is a second set of weights, and weights are the
// thing we cannot afford: the resident set is already 2.1 GB and the whole mobile problem this week
// was that number. Lookup drafting has no weights at all. It proposes the token that followed the
// last time this same short suffix appeared, which is right startlingly often on the work this
// engine actually does: an interview transcript quotes the question, a code answer repeats
// identifiers and keywords, a summary reuses the source's phrasing.
//
// WHAT IT COSTS WHEN IT IS WRONG. Nothing but the 1.23x, and it is never wrong in a way that
// changes the answer. The verify pass computes the logits the plain step would have computed, bit
// for bit, because the wide kernel is column for column the one column kernel (qgemvWide.ts
// header). A rejected draft is a KV truncate, which kv.ts records as a counter change with no GPU
// work. So speculation here cannot alter a single generated id; it can only change how long they
// take. That is the property the gate below is written against.

/** The sequence a drafter reads. Prompt tokens first, then everything generated so far. */
export interface DraftContext {
  readonly tokens: readonly number[];
}

export interface Drafter {
  readonly name: string;
  /**
   * Propose up to `want` continuation tokens after the end of `tokens`, or an empty array when
   * this drafter has nothing to say. Never throws; a drafter with no opinion is the normal case.
   */
  propose(tokens: readonly number[], want: number): number[];
}

/**
 * The longest suffix this drafter will try to match, and the shortest.
 *
 * Longer matches are likelier to be right and rarer. Trying long first and walking down is the
 * standard shape and it costs nothing here because the search is a backwards scan over an array of
 * numbers, which for a few thousand tokens is far below the 26 ms a decode token takes.
 */
export const LOOKUP_MAX_NGRAM = 8;
export const LOOKUP_MIN_NGRAM = 2;

/**
 * Prompt lookup drafting.
 *
 * Take the last n tokens of the sequence, find the most recent EARLIER occurrence of that same run,
 * and propose whatever followed it. Try the longest n first so a confident long match beats a
 * coincidental short one, and stop at the first n that matches anywhere.
 *
 * The scan runs backwards from the most recent occurrence, because in a transcript the nearest
 * repeat is the relevant one: the model is usually continuing the thing it just started, not the
 * thing it said four paragraphs ago.
 */
export function createLookupDrafter(
  maxNgram: number = LOOKUP_MAX_NGRAM,
  minNgram: number = LOOKUP_MIN_NGRAM,
): Drafter {
  const hi = Math.max(1, Math.trunc(maxNgram));
  const lo = Math.max(1, Math.min(hi, Math.trunc(minNgram)));
  return {
    name: `lookup-${lo}-${hi}`,
    propose(tokens, want) {
      const wanted = Math.max(0, Math.trunc(want));
      if (wanted === 0 || tokens.length < lo + 1) return [];
      for (let n = Math.min(hi, tokens.length - 1); n >= lo; n -= 1) {
        const from = tokens.length - n;
        // Start one before the suffix itself, so the suffix never matches its own position, and
        // walk backwards to the nearest earlier occurrence.
        for (let start = from - 1; start >= 0; start -= 1) {
          let same = true;
          for (let i = 0; i < n; i += 1) {
            if (tokens[start + i] !== tokens[from + i]) { same = false; break; }
          }
          if (!same) continue;
          const out: number[] = [];
          for (let i = 0; i < wanted; i += 1) {
            const next = tokens[start + n + i];
            // Stop at the end of what we have rather than padding with anything invented.
            if (next === undefined) break;
            out.push(next);
          }
          if (out.length > 0) return out;
        }
      }
      return [];
    },
  };
}

/** A drafter that never proposes, so the speculative loop degrades to the plain one. */
export function createNullDrafter(): Drafter {
  return { name: 'none', propose: () => [] };
}

/**
 * What a speculative run actually did, so a claim about it is a measurement rather than a hope.
 *
 * `acceptedDrafts / proposedDrafts` is the acceptance rate the 1.23x has to beat, and
 * `tokensPerPass` is the number that decides whether any of this was worth doing.
 */
export interface SpeculationStats {
  /** Verify passes run, each one weight stream. */
  passes: number;
  /**
   * Passes that ran a verify rather than a plain decode. This is the cost field: the wide kernel
   * is compiled at a fixed GEMV_WIDE_COLS and priced by that, not by how many of its columns
   * carried a real token, so every verify pass costs the same whatever the draft length was.
   */
  proposedDrafts: number;
  /** Draft tokens offered across every proposing pass. */
  draftedTokens: number;
  /** Draft tokens the model agreed with, each one a token the weight stream did not have to be re-read for. */
  acceptedDrafts: number;
  /** Tokens emitted across every pass. */
  tokens: number;
}

export function emptySpeculationStats(): SpeculationStats {
  return { passes: 0, proposedDrafts: 0, draftedTokens: 0, acceptedDrafts: 0, tokens: 0 };
}

/**
 * Tokens per verify pass. One means speculation bought nothing and the 1.23x was a pure loss.
 *
 * The break even against the round 7 ledger's 1.23x for two columns is a rate of about 0.23, so a
 * value below roughly 1.23 here means the drafter should be turned off for this workload rather
 * than tuned.
 */
export function tokensPerPass(stats: SpeculationStats): number {
  return stats.passes === 0 ? 0 : stats.tokens / stats.passes;
}

/**
 * Accepted draft tokens over offered draft tokens, or 0 when nothing was ever offered. This is the
 * number the break even is stated against: at two columns the wide kernel costs 1.23 weight
 * streams to emit up to 2 tokens, so the pass pays for itself above about 0.23.
 */
export function acceptanceRate(stats: SpeculationStats): number {
  return stats.draftedTokens === 0 ? 0 : stats.acceptedDrafts / stats.draftedTokens;
}

/**
 * The speedup this run earned against a plain decode, given the wide kernel's measured cost.
 *
 * `wideCost` is the ledger's figure for the column count in use: 1.23 for two columns on the M1
 * (kernels/qgemvWide.ts), 1.39 on a 5070 (lab-results/5070-gemv-sweep-sep04.json). Passes that
 * proposed nothing ran the one column kernel and cost 1.0.
 *
 * `streamFraction` is the share of a plain decode pass that is the weight stream, and it is the
 * parameter that stops this arithmetic from being an M1 fact dressed up as a general one. The wide
 * kernel makes the WEIGHT STREAM more expensive and leaves the rest of the pass alone: the same
 * norms, the same attention, the same gathers, the same argmax, the same per dispatch cost, all of
 * it amortised over more tokens instead of repeated. So a proposing pass costs
 * `(1 - streamFraction) + streamFraction * wideCost`, not `wideCost`.
 *
 * Why this matters rather than being a refinement. On the M1 the stream is about 58 percent of a
 * token and on a 5070 it is 29 percent, because that machine streams an order of magnitude faster
 * while the fixed costs barely move. Charging the whole pass at `wideCost`, which is what a
 * `streamFraction` of 1 does, says two columns need 39 percent acceptance to pay on a 5070. Charging
 * it honestly says 11 percent. The default is 1 so that an uninstrumented caller gets the
 * pessimistic answer rather than a flattering one.
 */
export function estimatedSpeedup(
  stats: SpeculationStats,
  wideCost: number,
  streamFraction = 1,
): number {
  if (stats.passes === 0) return 1;
  const f = Math.max(0, Math.min(1, streamFraction));
  const wide = stats.proposedDrafts;
  const narrow = stats.passes - wide;
  const cost = narrow * 1 + wide * ((1 - f) + f * wideCost);
  return cost === 0 ? 1 : stats.tokens / cost;
}

/** The break even acceptance rate: below this a proposing pass costs more than it saves. */
export function breakEvenAcceptance(wideCost: number, streamFraction = 1): number {
  const f = Math.max(0, Math.min(1, streamFraction));
  return (1 - f) + f * wideCost - 1;
}

/**
 * THE SPECULATIVE LOOP.
 *
 * It is `runGreedyLoop` (plan.ts) with one substitution: where that loop runs one decode step per
 * token, this one runs one pass per group of tokens, and a pass is a verify when the drafter had
 * something to say and a plain decode when it did not. Everything else is held identical on
 * purpose, because the gate that matters is that the ids come out the same: the stop set is
 * checked before the yield at every site, a stop token is never yielded, the abort is consulted
 * once per pass, and the first token still comes from the prefill and runs no pass at all.
 *
 * WHY A DECODE AND NOT A ONE COLUMN VERIFY when the drafter is silent. The wide pipeline is
 * compiled at a fixed column count and named `-m${GEMV_WIDE_COLS}` (execute.ts), so a verify
 * carrying one real token still streams the weights through the wide kernel and still pays the
 * wide kernel's price. Falling back to `decode` is what keeps a silent drafter free, and it is why
 * `estimatedSpeedup` charges 1.0 to the passes that did not propose.
 *
 * THE ACCEPTANCE RULE. `verify([prev, ...draft], position)` answers with the model's own argmax
 * after each of those inputs, so `answer[j]` is the token that follows `answer[j - 1]` exactly when
 * `draft[j - 1]` was right. Walk `m` up while `answer[j] === draft[j]`; the pass then emits
 * `answer[0..m]`, which is `m + 1` tokens, and `answer[0]` alone is the plain greedy token for this
 * position whatever the drafter said. A wrong draft costs the pass and nothing else.
 *
 * WHAT THIS LOOP GIVES UP, WHICH THE LEDGER DOES NOT CHARGE IT FOR. The plain loop's fast path is
 * `decodeAhead`, which submits the step for position p + 1 before the token at p has been read
 * back, so the host's encoding overlaps the GPU's previous run (plan.ts). A verify pass cannot do
 * that: the draft for the next pass is a function of the tokens this one produced, so the host has
 * to see them first. Speculation therefore starts one pipeline bubble down against the loop it has
 * to beat, and `estimatedSpeedup` does not know about that bubble; it is arithmetic over the wide
 * kernel's price alone. TREAT IT AS THE CEILING, NOT THE READING. The reading is wall clock tokens
 * per second with the switch on against the same prompt with it off, which is what the bench does.
 *
 * WHAT THE HOST DOES WITH THE REJECTED ROWS. Nothing. The pass wrote KV rows for every input it
 * carried, accepted or not, but the transcript only takes the accepted ones, so the rows past its
 * length are what `runGreedyLoop`'s discarded lookahead step already leaves behind: written, past
 * the end, and treated as absent by the next prefill's rewind. `kv.ts`'s truncate is therefore not
 * needed on this path; the length is right because nothing wrong was ever appended.
 */
export async function* runSpeculativeLoop(options: {
  firstToken: number;
  maxNewTokens: number;
  eosIds: ReadonlySet<number>;
  startPosition: number;
  /** The prompt, which is the drafter's corpus. Generated tokens are appended as they are emitted. */
  history: readonly number[];
  /** The compiled column count of the wide kernel. A draft is at most this minus one. */
  maxColumns: number;
  isAborted: () => boolean;
  drafter: Drafter;
  decode: (prevToken: number, position: number) => Promise<number>;
  verify: (tokens: readonly number[], position: number) => Promise<number[]>;
  /** Fires for each token whose KV row is written and kept, in transcript order. */
  onAdvance?: (token: number, position: number) => void;
  /** Filled in as the run goes, so a caller can read acceptance without waiting for the end. */
  stats?: SpeculationStats;
}): AsyncGenerator<{ token: number; index: number }, void, void> {
  const budget = Math.max(0, Math.floor(options.maxNewTokens));
  if (budget === 0 || options.isAborted()) return;
  if (options.eosIds.has(options.firstToken)) return;
  yield { token: options.firstToken, index: 0 };

  const stats = options.stats ?? emptySpeculationStats();
  // The prefill's token is not a pass, so it is not counted in `tokens`; tokensPerPass stays a
  // statement about the loop and not about the prompt.
  const history = options.history.slice();
  history.push(options.firstToken);

  // One column carries `prev`, so the draft gets the rest. The drafter's own n-gram limits are a
  // separate thing entirely: they bound how far back it looks, not how far forward it guesses.
  const maxDraft = Math.max(0, Math.floor(options.maxColumns) - 1);
  let prev = options.firstToken;
  let position = options.startPosition;
  let index = 1;

  while (index < budget) {
    if (options.isAborted()) return;
    // A pass emits one token plus one per accepted draft, so the draft is capped by what is left
    // of the budget. Without this an M wide pass could overshoot maxNewTokens.
    const want = Math.min(maxDraft, budget - index - 1);
    const draft = want > 0 ? options.drafter.propose(history, want) : [];
    stats.passes += 1;

    let answer: number[];
    if (draft.length > 0) {
      stats.proposedDrafts += 1;
      stats.draftedTokens += draft.length;
      answer = await options.verify([prev, ...draft], position);
      // An aborted verify answers empty. Anything else short is a contract break, not a state to
      // paper over, so it is reported rather than truncated into a plausible looking reply.
      if (answer.length === 0) return;
      if (answer.length !== draft.length + 1) {
        throw new Error(
          `gemma4 engine: a verify pass over ${draft.length + 1} positions answered ${answer.length}`,
        );
      }
    } else {
      answer = [await options.decode(prev, position)];
    }

    let matched = 0;
    while (matched < draft.length && answer[matched] === draft[matched]) matched += 1;
    stats.acceptedDrafts += matched;

    // `prev`'s row is written and kept whatever the drafter said, so the transcript takes it before
    // any stop check: a run that ends here should still be able to reuse this prefix next turn.
    options.onAdvance?.(prev, position);
    for (let j = 0; j <= matched; j += 1) {
      const token = answer[j]!;
      if (options.eosIds.has(token)) return;
      yield { token, index };
      index += 1;
      stats.tokens += 1;
      history.push(token);
      // Every token before the last one this pass emitted was also an input the pass ran, so its
      // row is written; the last one is the next pass's input and is appended by that pass.
      if (j < matched) options.onAdvance?.(token, position + 1 + j);
    }
    prev = answer[matched]!;
    position += 1 + matched;
  }
}
