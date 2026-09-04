// Written from the tokenizer.json, tokenizer_config.json and chat_template.jinja shipped with
// google/gemma-4-E2B-it-qat-mobile-transformers, the Hugging Face tokenizers JSON format, the
// reference data contract and section 4 of docs/ENGINE-PLAN.md. No inference engine source, no
// vendored bundle and no extracted kernel was read while writing this file.
//
// SPDX-License-Identifier: Apache-2.0
//
// What the checkpoint's files actually specify, read off them rather than recalled:
//
//   model          BPE, byte_fallback true, fuse_unk true, ignore_merges false, no dropout,
//                  no continuing_subword_prefix, no end_of_word_suffix. 262,144 entries,
//                  514,906 merges. There is no tokenizer.model, so nothing here parses a
//                  SentencePiece proto: the JSON is the whole specification.
//   normalizer     Replace, the ASCII space to U+2581, the metaspace character.
//   pre_tokenizer  Split on the ASCII space, MergedWithPrevious. Note that the normalizer has
//                  already turned every space into U+2581 by the time this runs, so on this
//                  checkpoint the split finds nothing and BPE sees the whole segment as one
//                  word. It is implemented anyway, from the file, rather than assumed inert.
//   post_processor TemplateProcessing whose special_tokens map is empty, so the tokenizer adds
//                  no BOS of its own. The chat template emits <bos> as text and the added token
//                  matcher turns it back into id 2. A tokenizer that helpfully added a second
//                  BOS would hand the model a prompt it has never seen.
//   decoder        Replace U+2581 back to a space, then ByteFallback, then Fuse.
//
// Ids that matter, again read from the files: pad 0, eos 1, bos 2, unk 3, and the turn markers
// <|turn> 105 and <turn|> 106. Gemma 4 does not use the <start_of_turn> markup of Gemma 2 and 3,
// so prompt code ported from a Gemma 3 project frames every turn wrongly while still producing
// fluent looking text. That is what the chat parity gate in scripts/engine-check.mjs catches.
//
// Built rather than borrowed, which is a deviation from the default ENGINE-PLAN section 4 landed
// on and so is argued rather than assumed. The plan's own reasoning for reaching for
// @huggingface/transformers was that it bundles a Jinja engine and that tokenization is not where
// this project's edge is. Both halves turned out cheaper than expected here. The template's tool
// calling, tool response and content part branches are unreachable from our message contract,
// which is an array of {role, content} with three roles and string content, so what is left to
// render is a leading system turn, a turn per message, one continuation rule and a generation
// prompt: the function at the bottom of this file, not a Jinja interpreter. And the BPE itself is
// the standard four components, so the whole file is about 900 lines against a dependency that
// would sit in the worker chunk next to a 2 GB weight load. The gate is what makes the choice
// safe rather than brave: every id this file produces is compared against the reference tokenizer,
// on the three probes, on eight message shapes, on the reference lane's three transformers
// prompts, and on the app's own five prompt cases. If any of that ever goes red and cannot be
// fixed quickly, the plan's option 1 is still sitting there.
//
// One thing lives here that the reference tokenizer has no equivalent of: Gemma4DecodeStream, the
// incremental decoder the generate loop needs for its `delta`. Decoding a token at a time is the
// obvious implementation and it is wrong, because one Devanagari character is three byte fallback
// tokens and one emoji is four.

/** A chat turn. Structurally the `Gemma4Message` of src/engine/llm/gemma4-engine.d.ts. */
export interface Gemma4Message {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

/** The parts of tokenizer.json this implementation reads. Everything else is ignored. */
export interface TokenizerJson {
  added_tokens?: AddedTokenJson[];
  normalizer?: unknown;
  pre_tokenizer?: unknown;
  post_processor?: unknown;
  decoder?: unknown;
  model: {
    type?: string;
    vocab: Record<string, number>;
    merges: Array<string | [string, string]>;
    unk_token?: string | null;
    dropout?: number | null;
    fuse_unk?: boolean;
    byte_fallback?: boolean;
    ignore_merges?: boolean;
    continuing_subword_prefix?: string | null;
    end_of_word_suffix?: string | null;
  };
}

export interface AddedTokenJson {
  id: number;
  content: string;
  special?: boolean;
  normalized?: boolean;
  lstrip?: boolean;
  rstrip?: boolean;
  single_word?: boolean;
}

/** The parts of tokenizer_config.json this implementation reads. */
export interface TokenizerConfigJson {
  bos_token?: string;
  eos_token?: string;
  eot_token?: string;
  pad_token?: string;
  unk_token?: string;
  [key: string]: unknown;
}

export interface DecodeOptions {
  /** Drop the special ids rather than rendering their text. Default false, as the reference is. */
  skipSpecialTokens?: boolean;
}

export interface ChatOptions {
  /** Append the `<|turn>model\n` opener the model completes into. Default true. */
  addGenerationPrompt?: boolean;
  /**
   * Emit the `<|think|>` marker at the top of the first system turn, which is how this template
   * opens the thinking channel. Default false: v1 is a two sentence interviewer, not a reasoner.
   */
  enableThinking?: boolean;
}

export interface TokenizerLoadOptions {
  fetch?: typeof fetch;
  signal?: AbortSignal;
  /** Called with the byte counts of each file as it lands, for the `tokenizer` progress event. */
  onFile?: (name: string, bytes: number) => void;
}

/** Diagnostics the parity check asserts on, so a silently degraded build cannot pass. */
export interface TokenizerStats {
  vocabSize: number;
  mergeCount: number;
  /** Merge rows whose pair or result is missing from the vocabulary. Expected to be zero. */
  droppedMerges: number;
  addedTokenCount: number;
  byteTokenCount: number;
}

const METASPACE = '▁';

/** Python's `str.strip()` set, which is what Jinja's `| trim` filter uses. */
const PY_SPACE = '[\\t\\n\\v\\f\\r \\x1c\\x1d\\x1e\\x1f\\x85\\u00a0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000]';
const PY_TRIM = new RegExp(`^${PY_SPACE}+|${PY_SPACE}+$`, 'gu');

/**
 * Jinja's `| trim`, which is Python's `str.strip()`. Not JavaScript's `String.trim()`: Python
 * strips U+001C to U+001F and U+0085 and does not strip U+FEFF, and JavaScript is the other way
 * round on both counts. The difference only shows on pasted text, which is exactly the text a
 * candidate pastes into an interview.
 */
export function jinjaTrim(value: string): string {
  return value.replace(PY_TRIM, '');
}

/**
 * The template's `strip_thinking` macro: drop every `<|channel>...<channel|>` span from an
 * assistant turn before it is replayed, then trim. Written from the macro's own text.
 */
export function stripThinking(text: string): string {
  let result = '';
  for (const part of text.split('<channel|>')) {
    const cut = part.indexOf('<|channel>');
    result += cut === -1 ? part : part.slice(0, cut);
  }
  return jinjaTrim(result);
}

interface AddedToken {
  id: number;
  content: string;
}

interface PendingMerge {
  rank: number;
  pos: number;
  newId: number;
}

/**
 * A binary heap ordered the way the reference BPE orders its merge queue: lowest rank first, and
 * on a rank tie the leftmost position first. The order is the algorithm, not an optimisation, so
 * it is spelled out here rather than left to a sort's stability.
 */
class MergeHeap {
  private readonly items: PendingMerge[] = [];

  get size(): number {
    return this.items.length;
  }

  push(item: PendingMerge): void {
    const items = this.items;
    items.push(item);
    let child = items.length - 1;
    while (child > 0) {
      const parent = (child - 1) >> 1;
      if (!before(items[child]!, items[parent]!)) break;
      const swap = items[parent]!;
      items[parent] = items[child]!;
      items[child] = swap;
      child = parent;
    }
  }

  pop(): PendingMerge | undefined {
    const items = this.items;
    const top = items[0];
    if (top === undefined) return undefined;
    const last = items.pop()!;
    if (items.length === 0) return top;
    items[0] = last;
    let parent = 0;
    for (;;) {
      const left = parent * 2 + 1;
      const right = left + 1;
      let best = parent;
      if (left < items.length && before(items[left]!, items[best]!)) best = left;
      if (right < items.length && before(items[right]!, items[best]!)) best = right;
      if (best === parent) break;
      const swap = items[best]!;
      items[best] = items[parent]!;
      items[parent] = swap;
      parent = best;
    }
    return top;
  }
}

function before(a: PendingMerge, b: PendingMerge): boolean {
  return a.rank !== b.rank ? a.rank < b.rank : a.pos < b.pos;
}

function fail(message: string): never {
  throw new Error(`gemma4 tokenizer: ${message}`);
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' ? (value as Record<string, unknown>) : {};
}

/** A `{ "String": "x" }` pattern from the tokenizers JSON, which is the only form here. */
function stringPattern(value: unknown, where: string): string {
  const pattern = asRecord(asRecord(value).pattern);
  const literal = pattern['String'];
  if (typeof literal !== 'string') {
    fail(`${where} is not a literal string pattern, and no other form is implemented`);
  }
  return literal as string;
}

export class Gemma4Tokenizer {
  /** id to token text. Dense over the vocabulary. */
  private readonly tokens: string[];
  private readonly vocab: Map<string, number>;
  /** `left * vocabSize + right` to merge rank. The rank is the row index in `merges`. */
  private readonly mergeRank: Map<number, number>;
  /** Merge rank to the id of the joined pair. */
  private readonly mergeResult: Int32Array;
  private readonly addedByFirstChar: Map<string, AddedToken[]>;
  private readonly specialIds: Set<number>;
  /** byte value to the id of its `<0xNN>` token. */
  private readonly byteToId: Int32Array;
  /** id of a `<0xNN>` token to its byte value, for the ByteFallback decoder. */
  private readonly idToByte: Map<number, number>;
  private readonly normalizeFrom: string;
  private readonly normalizeTo: string;
  private readonly splitOn: string;
  private readonly decodeReplaceFrom: string;
  private readonly decodeReplaceTo: string;
  private readonly unkId: number | null;
  private readonly fuseUnk: boolean;
  private readonly byteFallback: boolean;
  private readonly ignoreMerges: boolean;
  private readonly config: TokenizerConfigJson;
  private readonly droppedMerges: number;

  readonly bosTokenId: number;
  readonly eosTokenId: number;
  readonly padTokenId: number;
  readonly unkTokenId: number;
  /** `<turn|>`, which ends a model turn and is the second member of config.json's eos set. */
  readonly eotTokenId: number;

  constructor(json: TokenizerJson, config: TokenizerConfigJson = {}) {
    const model = json.model;
    if (model?.type !== undefined && model.type !== 'BPE') {
      fail(`model type ${String(model.type)} is not BPE, which is the only model implemented`);
    }
    if (model.dropout !== null && model.dropout !== undefined && model.dropout !== 0) {
      fail('BPE dropout is set, which would make tokenization non deterministic');
    }
    if (model.continuing_subword_prefix) fail('continuing_subword_prefix is not implemented');
    if (model.end_of_word_suffix) fail('end_of_word_suffix is not implemented');

    this.config = config;
    this.fuseUnk = model.fuse_unk === true;
    this.byteFallback = model.byte_fallback === true;
    this.ignoreMerges = model.ignore_merges === true;

    // Normalizer. Replace of one literal string by another is the whole pipeline on this file.
    const normalizer = asRecord(json.normalizer);
    if (normalizer['type'] !== 'Replace') {
      fail(`normalizer ${String(normalizer['type'])} is not implemented`);
    }
    this.normalizeFrom = stringPattern(normalizer, 'normalizer');
    this.normalizeTo = typeof normalizer['content'] === 'string' ? (normalizer['content'] as string) : METASPACE;

    // Pre tokenizer. Split with MergedWithPrevious keeps each delimiter on the piece before it.
    const pre = asRecord(json.pre_tokenizer);
    if (pre['type'] !== 'Split') fail(`pre tokenizer ${String(pre['type'])} is not implemented`);
    if (pre['behavior'] !== 'MergedWithPrevious') {
      fail(`split behaviour ${String(pre['behavior'])} is not implemented`);
    }
    if (pre['invert'] === true) fail('inverted split is not implemented');
    this.splitOn = stringPattern(pre, 'pre tokenizer');

    // Post processor. The template's special token map is empty, so nothing is added around the
    // sequence. If a future checkpoint fills it in, refuse rather than silently drop a BOS.
    const post = asRecord(json.post_processor);
    if (post['type'] !== undefined) {
      if (post['type'] !== 'TemplateProcessing') {
        fail(`post processor ${String(post['type'])} is not implemented`);
      }
      const specials = asRecord(post['special_tokens']);
      if (Object.keys(specials).length > 0) {
        fail('post processor carries special tokens, which this checkpoint does not');
      }
    }

    // Decoder. Replace, then ByteFallback, then Fuse, in that order.
    const decoder = asRecord(json.decoder);
    const chain = Array.isArray(decoder['decoders']) ? (decoder['decoders'] as unknown[]) : [];
    if (decoder['type'] !== 'Sequence' || chain.length !== 3) {
      fail('decoder is not the three step Replace, ByteFallback, Fuse sequence');
    }
    const [replaceStep, byteStep, fuseStep] = chain.map((step) => asRecord(step));
    if (replaceStep!['type'] !== 'Replace' || byteStep!['type'] !== 'ByteFallback' || fuseStep!['type'] !== 'Fuse') {
      fail('decoder steps are not Replace, ByteFallback, Fuse');
    }
    this.decodeReplaceFrom = stringPattern(replaceStep, 'decoder replace');
    this.decodeReplaceTo = typeof replaceStep!['content'] === 'string' ? (replaceStep!['content'] as string) : ' ';

    // Vocabulary.
    const vocab = new Map<string, number>();
    let maxId = -1;
    for (const key of Object.keys(model.vocab)) {
      const id = model.vocab[key]!;
      vocab.set(key, id);
      if (id > maxId) maxId = id;
    }
    const tokens = new Array<string>(maxId + 1).fill('');
    for (const [key, id] of vocab) tokens[id] = key;
    this.vocab = vocab;
    this.tokens = tokens;

    // Byte tokens. `<0xNN>` with upper case hex on this checkpoint, both cases accepted.
    this.byteToId = new Int32Array(256).fill(-1);
    this.idToByte = new Map<number, number>();
    for (let b = 0; b < 256; b += 1) {
      const upper = `<0x${b.toString(16).toUpperCase().padStart(2, '0')}>`;
      const lower = `<0x${b.toString(16).toLowerCase().padStart(2, '0')}>`;
      const id = vocab.get(upper) ?? vocab.get(lower);
      if (id !== undefined) {
        this.byteToId[b] = id;
        this.idToByte.set(id, b);
      }
    }
    if (this.byteFallback && this.idToByte.size !== 256) {
      fail(`byte_fallback is on but only ${this.idToByte.size} of 256 byte tokens exist`);
    }

    // Merges. The row index is the rank, which is the whole priority order of the algorithm.
    const merges = model.merges;
    this.mergeRank = new Map<number, number>();
    this.mergeResult = new Int32Array(merges.length);
    const stride = tokens.length;
    let dropped = 0;
    for (let rank = 0; rank < merges.length; rank += 1) {
      const row = merges[rank]!;
      let left: string;
      let right: string;
      if (typeof row === 'string') {
        const cut = row.indexOf(' ');
        if (cut === -1) {
          dropped += 1;
          continue;
        }
        left = row.slice(0, cut);
        right = row.slice(cut + 1);
      } else {
        left = row[0];
        right = row[1];
      }
      const leftId = vocab.get(left);
      const rightId = vocab.get(right);
      const joined = vocab.get(left + right);
      if (leftId === undefined || rightId === undefined || joined === undefined) {
        dropped += 1;
        continue;
      }
      const key = leftId * stride + rightId;
      // A later row for the same pair wins, which is what building a map in file order gives and
      // is what the reference does. There are no duplicate pairs in this checkpoint's list, and
      // the parity check asserts that by comparing ids rather than by trusting the claim.
      this.mergeRank.set(key, rank);
      this.mergeResult[rank] = joined;
    }
    this.droppedMerges = dropped;

    // Added tokens. All of this checkpoint's are matched against the raw text, before the
    // normalizer runs, and none of them strip surrounding whitespace or need a word boundary.
    this.addedByFirstChar = new Map<string, AddedToken[]>();
    this.specialIds = new Set<number>();
    for (const added of json.added_tokens ?? []) {
      if (added.normalized === true) {
        fail(`added token ${added.content} is normalized, which is not implemented`);
      }
      if (added.lstrip || added.rstrip || added.single_word) {
        fail(`added token ${added.content} carries strip or word rules that are not implemented`);
      }
      if (added.content.length === 0) continue;
      const first = added.content[0]!;
      const bucket = this.addedByFirstChar.get(first) ?? [];
      bucket.push({ id: added.id, content: added.content });
      this.addedByFirstChar.set(first, bucket);
      if (added.special !== false) this.specialIds.add(added.id);
    }
    // Longest first, so the match is leftmost longest the way the reference matcher is: without
    // this, `<|image>` would swallow the front of `<|image|>`.
    for (const bucket of this.addedByFirstChar.values()) {
      bucket.sort((a, b) => b.content.length - a.content.length);
    }

    const unkToken = model.unk_token ?? (typeof config.unk_token === 'string' ? config.unk_token : null);
    this.unkId = unkToken === null ? null : vocab.get(unkToken) ?? null;

    this.bosTokenId = this.requireToken(config.bos_token ?? '<bos>');
    this.eosTokenId = this.requireToken(config.eos_token ?? '<eos>');
    this.padTokenId = this.requireToken(config.pad_token ?? '<pad>');
    this.unkTokenId = this.requireToken(config.unk_token ?? '<unk>');
    this.eotTokenId = this.requireToken(config.eot_token ?? '<turn|>');
  }

  private requireToken(token: string): number {
    const id = this.vocab.get(token);
    if (id === undefined) fail(`the vocabulary has no ${token}`);
    return id;
  }

  get vocabSize(): number {
    return this.tokens.length;
  }

  stats(): TokenizerStats {
    return {
      vocabSize: this.tokens.length,
      mergeCount: this.mergeResult.length,
      droppedMerges: this.droppedMerges,
      addedTokenCount: [...this.addedByFirstChar.values()].reduce((n, b) => n + b.length, 0),
      byteTokenCount: this.idToByte.size,
    };
  }

  tokenToId(token: string): number | undefined {
    return this.vocab.get(token);
  }

  idToToken(id: number): string | undefined {
    return this.tokens[id];
  }

  isSpecial(id: number): boolean {
    return this.specialIds.has(id);
  }

  /**
   * The stop set the tokenizer files themselves name: `<eos>` and `<turn|>`, which is also the
   * `eos_token_id` of `config.json`.
   *
   * Read this before wiring the decode loop. `generation_config.json` on this checkpoint carries a
   * third id, 50, which is `<|tool_response>`. Our chat template never emits it and a text only v1
   * has nothing to answer it with, so a model that produces it has started a tool call into a void
   * and should stop rather than narrate one. The engine's stop set is therefore the union of this
   * pair with whatever `generation_config.json` lists, taken at load time from the file rather
   * than hardcoded here, since this class reads only the two tokenizer files.
   */
  get defaultStopTokenIds(): number[] {
    return [this.eosTokenId, this.eotTokenId];
  }

  /**
   * Encode text to ids. No BOS and no turn markers are added: this checkpoint's post processor
   * is empty and the chat template is what frames a prompt. Added token text inside `text` is
   * recognised, which is what the reference tokenizer does, so a caller that concatenates
   * untrusted text into a prompt is the caller that has to think about it.
   */
  encode(text: string): number[] {
    const ids: number[] = [];
    if (text.length === 0) return ids;
    let pending = '';
    let i = 0;
    while (i < text.length) {
      const bucket = this.addedByFirstChar.get(text[i]!);
      let matched: AddedToken | undefined;
      if (bucket !== undefined) {
        for (const candidate of bucket) {
          if (text.startsWith(candidate.content, i)) {
            matched = candidate;
            break;
          }
        }
      }
      if (matched === undefined) {
        pending += text[i];
        i += 1;
        continue;
      }
      if (pending.length > 0) {
        this.encodeSegment(pending, ids);
        pending = '';
      }
      ids.push(matched.id);
      i += matched.content.length;
    }
    if (pending.length > 0) this.encodeSegment(pending, ids);
    return ids;
  }

  /** One run of ordinary text: normalize, pre tokenize, then BPE each piece. */
  private encodeSegment(segment: string, out: number[]): void {
    const normalized = segment.split(this.normalizeFrom).join(this.normalizeTo);
    for (const piece of this.splitMergedWithPrevious(normalized)) {
      if (piece.length > 0) this.bpe(piece, out);
    }
  }

  /**
   * Split on a literal delimiter, keeping each delimiter attached to the piece before it. On this
   * checkpoint the delimiter is the ASCII space and the normalizer has already removed every one
   * of them, so this returns its input, but the file says Split and so this splits.
   */
  private splitMergedWithPrevious(text: string): string[] {
    const delimiter = this.splitOn;
    if (delimiter.length === 0) return [text];
    const pieces: string[] = [];
    let start = 0;
    for (;;) {
      const hit = text.indexOf(delimiter, start);
      if (hit === -1) break;
      pieces.push(text.slice(start, hit + delimiter.length));
      start = hit + delimiter.length;
    }
    pieces.push(text.slice(start));
    return pieces;
  }

  /**
   * Byte pair encoding of one piece. The initial symbols are Unicode code points, not UTF-16
   * units and not bytes: a code point missing from the vocabulary is replaced by its UTF-8 bytes
   * as `<0xNN>` tokens, which then take part in merges like any other symbol. Iterating by UTF-16
   * unit instead would split every emoji into two lone surrogates and quietly produce a different
   * id sequence for exactly the text a candidate's name is most likely to contain.
   */
  private bpe(piece: string, out: number[]): void {
    if (this.ignoreMerges) {
      const whole = this.vocab.get(piece);
      if (whole !== undefined) {
        out.push(whole);
        return;
      }
    }

    const ids: number[] = [];
    let lastWasUnk = false;
    for (const ch of piece) {
      const known = this.vocab.get(ch);
      if (known !== undefined) {
        ids.push(known);
        lastWasUnk = false;
        continue;
      }
      let handled = false;
      if (this.byteFallback) {
        const bytes = utf8Bytes(ch);
        const mapped: number[] = [];
        for (const b of bytes) {
          const id = this.byteToId[b]!;
          if (id < 0) {
            mapped.length = 0;
            break;
          }
          mapped.push(id);
        }
        if (mapped.length > 0) {
          for (const id of mapped) ids.push(id);
          lastWasUnk = false;
          handled = true;
        }
      }
      if (handled) continue;
      if (this.unkId === null) continue;
      if (this.fuseUnk && lastWasUnk) continue;
      ids.push(this.unkId);
      lastWasUnk = true;
    }

    const n = ids.length;
    if (n === 0) return;
    if (n === 1) {
      out.push(ids[0]!);
      return;
    }

    // A doubly linked list over the symbols, so a merge is a pointer update and every queued
    // position stays valid until the symbol under it is consumed.
    const prev = new Int32Array(n);
    const next = new Int32Array(n);
    const alive = new Uint8Array(n).fill(1);
    for (let i = 0; i < n; i += 1) {
      prev[i] = i - 1;
      next[i] = i === n - 1 ? -1 : i + 1;
    }

    const stride = this.tokens.length;
    const heap = new MergeHeap();
    for (let i = 0; i + 1 < n; i += 1) {
      const rank = this.mergeRank.get(ids[i]! * stride + ids[i + 1]!);
      if (rank !== undefined) heap.push({ rank, pos: i, newId: this.mergeResult[rank]! });
    }

    while (heap.size > 0) {
      const top = heap.pop()!;
      const pos = top.pos;
      if (alive[pos] === 0) continue;
      const right = next[pos]!;
      if (right === -1) continue;
      // The queue holds positions, so an entry can be stale: what matters is whether the pair
      // sitting there now is still the pair that was queued, judged by the id it merges to.
      const rank = this.mergeRank.get(ids[pos]! * stride + ids[right]!);
      if (rank === undefined || this.mergeResult[rank] !== top.newId) continue;

      ids[pos] = top.newId;
      alive[right] = 0;
      const after = next[right]!;
      next[pos] = after;
      if (after !== -1) prev[after] = pos;

      const behind = prev[pos]!;
      if (behind !== -1) {
        const leftRank = this.mergeRank.get(ids[behind]! * stride + ids[pos]!);
        if (leftRank !== undefined) {
          heap.push({ rank: leftRank, pos: behind, newId: this.mergeResult[leftRank]! });
        }
      }
      if (after !== -1) {
        const rightRank = this.mergeRank.get(ids[pos]! * stride + ids[after]!);
        if (rightRank !== undefined) {
          heap.push({ rank: rightRank, pos, newId: this.mergeResult[rightRank]! });
        }
      }
    }

    for (let i = 0; i !== -1; i = next[i]!) out.push(ids[i]!);
  }

  /**
   * Decode ids back to text: metaspace back to a space, then the byte fallback run, then fuse.
   * An invalid UTF-8 run decodes to one replacement character per byte, which is what the
   * reference does and is worth keeping, because it makes a truncated multi byte character
   * visible instead of silently shortening the string.
   *
   * One property to know before filing it as a bug: a literal U+2581 in the input comes back as
   * a space, because the normalizer maps a space to U+2581 on the way in and the decoder maps it
   * back on the way out. The reference tokenizer is lossy on that character in exactly the same
   * way, verified against it, so this matches rather than diverges.
   */
  decode(ids: Iterable<number>, options: DecodeOptions = {}): string {
    const skipSpecial = options.skipSpecialTokens === true;
    let out = '';
    let bytes: number[] = [];
    const flush = (): void => {
      if (bytes.length === 0) return;
      out += decodeUtf8(bytes);
      bytes = [];
    };
    for (const id of ids) {
      if (skipSpecial && this.specialIds.has(id)) continue;
      const byte = this.idToByte.get(id);
      if (byte !== undefined) {
        bytes.push(byte);
        continue;
      }
      const text = this.tokenText(id);
      if (text === '') continue;
      flush();
      out += text;
    }
    flush();
    return out;
  }

  /** The byte behind a `<0xNN>` token. Undefined for every other id, including a missing one. */
  byteForId(id: number): number | undefined {
    return this.idToByte.get(id);
  }

  /**
   * The visible text of one non byte token, with the metaspace character already turned back into
   * a space. Empty for a byte token, since a byte only becomes text once its run is decoded, and
   * empty for an id the vocabulary does not hold.
   */
  tokenText(id: number): string {
    const token = this.tokens[id];
    if (token === undefined || token === '') return '';
    if (this.idToByte.has(id)) return '';
    return token.split(this.decodeReplaceFrom).join(this.decodeReplaceTo);
  }

  /**
   * A decoder for the generate loop, which needs the text of each token as it arrives rather than
   * the text of the whole sequence at the end. Calling `decode` on one id at a time is the obvious
   * thing and it is wrong: one Devanagari character is three byte fallback tokens and one emoji is
   * four, so a per token decode emits three or four replacement characters where a name should be.
   * The stream holds an unfinished byte run back until it completes.
   */
  decodeStream(options: DecodeOptions = {}): Gemma4DecodeStream {
    return new Gemma4DecodeStream(this, options);
  }

  /**
   * Render the chat template to text. This implements the branches our message contract can
   * reach, which is roles system, user and assistant with string content, and refuses the rest
   * rather than guessing: no tools, no tool responses, no content part arrays, no reasoning
   * channel replay. The rendered shape, read off chat_template.jinja:
   *
   *   <bos><|turn>system\n{system}<turn|>\n<|turn>user\n{user}<turn|>\n<|turn>model\n
   *
   * Two details that are easy to lose. A system message is only folded into the leading system
   * turn when it is the first message; anywhere else it renders as its own turn. And consecutive
   * assistant messages continue one model turn rather than opening a second, so their contents
   * are concatenated with no separator at all.
   */
  renderChat(messages: Gemma4Message[], options: ChatOptions = {}): string {
    const addGenerationPrompt = options.addGenerationPrompt !== false;
    const enableThinking = options.enableThinking === true;
    const bos = this.config.bos_token ?? '<bos>';

    for (const message of messages) {
      if (typeof message?.content !== 'string') {
        fail('a message content is not a string, and content part arrays are not implemented');
      }
      if (message.role !== 'system' && message.role !== 'user' && message.role !== 'assistant') {
        fail(`message role ${String(message.role)} is not implemented`);
      }
    }

    let out = bos;
    let rest = messages;
    const leadingSystem = messages.length > 0 && messages[0]!.role === 'system';
    if (enableThinking || leadingSystem) {
      out += '<|turn>system\n';
      if (enableThinking) out += '<|think|>\n';
      if (leadingSystem) {
        out += jinjaTrim(messages[0]!.content);
        rest = messages.slice(1);
      }
      out += '<turn|>\n';
    }

    let previousRole: string | null = null;
    for (let i = 0; i < rest.length; i += 1) {
      const message = rest[i]!;
      const role = message.role === 'assistant' ? 'model' : message.role;
      const continuesFromPrevious = role === 'model' && previousRole === 'assistant';
      if (!continuesFromPrevious) out += `<|turn>${role}\n`;
      out += role === 'model' ? stripThinking(message.content) : jinjaTrim(message.content);
      const nextRole = i + 1 < rest.length ? rest[i + 1]!.role : null;
      const continuesIntoNext = role === 'model' && nextRole === 'assistant';
      if (!continuesIntoNext) out += '<turn|>\n';
      previousRole = message.role;
    }

    if (addGenerationPrompt) out += '<|turn>model\n';
    return out;
  }

  /**
   * The `encodePrompt` of the engine contract: messages in, prompt ids out. `client.ts` uses the
   * length as the prompt budget against `PROMPT_TOKEN_BUDGET`, so a tokenizer that is right about
   * the text and wrong about the count silently changes what gets trimmed out of a transcript.
   */
  encodePrompt(messages: Gemma4Message[], options: ChatOptions = {}): number[] {
    return this.encode(this.renderChat(messages, options));
  }

  /** Parse the two files. `tokenizerJson` is the 32 MB one, so it is passed already parsed. */
  static fromJson(tokenizerJson: TokenizerJson, tokenizerConfig?: TokenizerConfigJson): Gemma4Tokenizer {
    return new Gemma4Tokenizer(tokenizerJson, tokenizerConfig ?? {});
  }

  /**
   * Fetch and build from a model root, the same base URL the weight loader resolves. There is no
   * tokenizer.model in this checkpoint, so these two files are the whole tokenizer. tokenizer.json
   * is about 32 MB, which is inside the download consent gate rather than a free extra.
   *
   * Measured cost, on the M1 through node: parsing the JSON holds about 143 MB and building the
   * maps takes about 1.5 seconds, after which the parsed JSON is dropped and the tokenizer itself
   * retains about 37 MB. Load the tokenizer before the weights rather than alongside them, so
   * that transient peak does not land on top of a 2 GB fetch.
   */
  static async load(modelRoot: string, options: TokenizerLoadOptions = {}): Promise<Gemma4Tokenizer> {
    // Bound to the global object. An unbound global fetch called through a binding throws
    // "Illegal invocation" in a browser and works in Node, which is the worst of both.
    const doFetch = options.fetch ?? ((input: RequestInfo | URL, init?: RequestInit) => fetch(input, init));
    const base = modelRoot.endsWith('/') ? modelRoot : `${modelRoot}/`;
    const read = async (name: string): Promise<string> => {
      const response = await doFetch(`${base}${name}`, { signal: options.signal });
      if (!response.ok) fail(`${name} fetch failed with ${response.status}`);
      const text = await response.text();
      options.onFile?.(name, text.length);
      return text;
    };
    const [tokenizerText, configText] = await Promise.all([
      read('tokenizer.json'),
      read('tokenizer_config.json').catch(() => '{}'),
    ]);
    return new Gemma4Tokenizer(
      JSON.parse(tokenizerText) as TokenizerJson,
      JSON.parse(configText) as TokenizerConfigJson,
    );
  }
}

/**
 * Incremental decode for the generate loop. `push` returns the text this token added, which is the
 * `delta` of the engine contract, and `text` is the cumulative string the contract yields next to
 * it. A byte fallback run is held until it forms whole code points, so a token that carries the
 * middle of an emoji produces no delta and the token that completes it produces the whole
 * character.
 *
 * The one place this differs from `decode` over the same ids: a byte run that is valid and then
 * turns invalid emits the valid part as text and a replacement character per undecodable byte
 * after it, where `decode` sees the run as one blob and replaces all of it. Every id sequence that
 * decodes to valid UTF-8, which is every sequence the round trip gate covers, gives the identical
 * string either way.
 */
export class Gemma4DecodeStream {
  private readonly tokenizer: Gemma4Tokenizer;
  private readonly skipSpecial: boolean;
  /** The bytes of a byte fallback run that has not yet completed a code point. At most 3. */
  private readonly pending: number[] = [];
  private cumulative = '';

  constructor(tokenizer: Gemma4Tokenizer, options: DecodeOptions = {}) {
    this.tokenizer = tokenizer;
    this.skipSpecial = options.skipSpecialTokens === true;
  }

  /** Everything pushed so far, as text. The `text` field of a generate chunk. */
  get text(): string {
    return this.cumulative;
  }

  /** True while a byte run is incomplete, so a caller can tell an empty delta from a finished one. */
  get holdingBytes(): boolean {
    return this.pending.length > 0;
  }

  /** Feed one token id. Returns the text it added, which is often the empty string. */
  push(id: number): string {
    if (this.skipSpecial && this.tokenizer.isSpecial(id)) return '';
    const byte = this.tokenizer.byteForId(id);
    if (byte === undefined) {
      const text = this.tokenizer.tokenText(id);
      if (text === '' && this.pending.length === 0) return '';
      const delta = this.drain() + text;
      this.cumulative += text;
      return delta;
    }
    this.pending.push(byte);
    const whole = tryDecodeUtf8(this.pending);
    if (whole === null) {
      // Four bytes is the longest UTF-8 sequence there is, so a run this long that still will not
      // decode is not going to start. Give up on it rather than buffering a broken stream forever.
      if (this.pending.length < 4) return '';
      return this.drain();
    }
    this.pending.length = 0;
    this.cumulative += whole;
    return whole;
  }

  /** Feed several ids and return their combined delta. */
  pushAll(ids: Iterable<number>): string {
    let delta = '';
    for (const id of ids) delta += this.push(id);
    return delta;
  }

  /** Finish the stream, releasing any incomplete byte run as replacement characters. */
  end(): string {
    return this.drain();
  }

  private drain(): string {
    if (this.pending.length === 0) return '';
    const text = decodeUtf8(this.pending);
    this.pending.length = 0;
    this.cumulative += text;
    return text;
  }
}

/** UTF-8 bytes of one code point. Written out rather than allocating a TextEncoder per symbol. */
function utf8Bytes(ch: string): number[] {
  const code = ch.codePointAt(0)!;
  if (code < 0x80) return [code];
  if (code < 0x800) return [0xc0 | (code >> 6), 0x80 | (code & 0x3f)];
  if (code < 0x10000) {
    return [0xe0 | (code >> 12), 0x80 | ((code >> 6) & 0x3f), 0x80 | (code & 0x3f)];
  }
  return [
    0xf0 | (code >> 18),
    0x80 | ((code >> 12) & 0x3f),
    0x80 | ((code >> 6) & 0x3f),
    0x80 | (code & 0x3f),
  ];
}

const strictUtf8 = new TextDecoder('utf-8', { fatal: true });

function decodeUtf8(bytes: number[]): string {
  try {
    return strictUtf8.decode(new Uint8Array(bytes));
  } catch {
    return '�'.repeat(bytes.length);
  }
}

/** Strict decode, or null when the bytes are not yet, or never will be, valid UTF-8. */
function tryDecodeUtf8(bytes: number[]): string | null {
  try {
    return strictUtf8.decode(new Uint8Array(bytes));
  } catch {
    return null;
  }
}
