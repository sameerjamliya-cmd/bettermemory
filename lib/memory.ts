import { randomUUID } from "crypto";
import { QdrantVectorStore } from "./providers/qdrant-store";
import { OpenAILLMClient } from "./providers/openai-llm";
import type { ExtractedFact, LLMClient, MemoryFilter, VectorStoreClient } from "./providers/types";
import type {
  AddInput,
  AddResult,
  DeleteAllResult,
  DeleteResult,
  GetResult,
  HistoryResult,
  MemoryRecord,
  MemoryType,
  ProceduralMemory,
  RelatedMemory,
  Scope,
  SearchResult,
  SkippedFact,
  StoredFact,
  StoredMemory,
} from "./types";
import {
  EXCLUDE_PROCEDURAL,
  pointMatchesScope,
  scopeFilter,
  validateScope,
  type ValidatedScope,
} from "./scope";
import { buildExtractionPrompt, labelMemories, type LabeledMemory } from "./prompts/build";
import { HYBRID_CANDIDATE_POOL, computeBM25Scores, fuseScores, tokenize } from "./scoring";
import { hashContent } from "./dedup";
import {
  appendEvent,
  eventsForMemories,
  hasHistory,
  predecessorOf,
  successorsOf,
  type MemoryEvent,
} from "./events";
import { logEvent } from "./logger";

export type { MemoryEvent, MemoryEventType } from "./events";
export type {
  AddInput, AddResult, DeleteAllResult, DeleteResult, GetResult, HistoryResult,
  MemoryRecord, MemoryType, ProceduralMemory, RelatedMemory, Scope, SearchResult,
  SkippedFact, StoredFact, StoredMemory,
} from "./types";

// The concrete providers are chosen once, here. Everything below this line talks
// only to the two interfaces. Swapping vendors means constructing a different
// implementation — not editing any orchestration function.
let store: VectorStoreClient = new QdrantVectorStore();
let llm: LLMClient = new OpenAILLMClient();

/** Replace the providers. Intended for tests that need an in-memory store; there
 *  is deliberately no config file or env switch behind this. */
export function setProviders(providers: { vectorStore?: VectorStoreClient; llm?: LLMClient }): void {
  if (providers.vectorStore) store = providers.vectorStore;
  if (providers.llm) llm = providers.llm;
}

const SCROLL_PAGE_SIZE = 100;
// Counts at which add() surfaces a one-off scale notice. Crossing a value fires
// once; ordinary calls either side of it stay silent.
const SCALE_NOTICE_THRESHOLDS = [100, 500, 1000];

async function ensureCollection() {
  await store.ensureReady();
}

async function embed(text: string): Promise<number[]> {
  return llm.embed(text);
}

interface RelatedMemoryInternal extends RelatedMemory {
  id: string;
}

async function findRelated(text: string, scope: ValidatedScope): Promise<RelatedMemoryInternal[]> {
  const vector = await embed(text);
  // Procedural memories are instructions, not source material for extraction.
  const points = await store.query(
    { ...scopeFilter(scope), mustNot: [EXCLUDE_PROCEDURAL] },
    vector,
    5
  );
  return points.map((point) => ({
    id: String(point.id),
    content: String(point.payload?.content ?? ""),
    source: String(point.payload?.source ?? ""),
    score: point.score,
    extractedAt: String(point.payload?.extractedAt ?? ""),
  }));
}

async function findSupersededIds(
  ids: string[],
  scope: ValidatedScope
): Promise<Set<string>> {
  if (ids.length === 0) return new Set();
  const base = scopeFilter(scope);
  const superseded = new Set<string>();
  const filter: MemoryFilter = {
    must: [...(base.must ?? []), { key: "supersededMemoryId", anyOf: ids }],
  };
  for await (const page of store.scroll(filter, SCROLL_PAGE_SIZE, {
    payloadFields: ["supersededMemoryId"],
  })) {
    for (const point of page) {
      const target = point.payload?.supersededMemoryId;
      if (typeof target === "string") superseded.add(target);
    }
  }
  return superseded;
}

async function extractFacts(
  text: string,
  labeled: LabeledMemory[],
  customInstructions?: string,
  observedAt?: string,
  currentDate?: string,
  imageDataUrl?: string
): Promise<ExtractedFact[]> {
  const prompt = buildExtractionPrompt(text, labeled, customInstructions, observedAt, currentDate);
  return llm.extractFacts(prompt, imageDataUrl ? { imageDataUrl } : undefined);
}

function buildNotice(countBefore: number, countAfter: number): string | null {
  if (countBefore === 0 && countAfter > 0) {
    return (
      "This is your first memory for this scope. Facts you add will be extracted, " +
      "deduplicated, and retrievable via search()."
    );
  }

  const crossed = SCALE_NOTICE_THRESHOLDS.filter((t) => countBefore < t && countAfter >= t);
  if (crossed.length > 0) {
    const threshold = Math.max(...crossed);
    return (
      `This scope now has ${threshold}+ memories. Consider using getAll() periodically ` +
      "to review what is active, or deleteAll() to reset if this is test data."
    );
  }

  return null;
}

function toImageDataUrl(imageBase64: string): string {
  const trimmed = imageBase64.trim();
  if (trimmed.startsWith("data:")) return trimmed;
  const mime = trimmed.startsWith("/9j/")
    ? "image/jpeg"
    : trimmed.startsWith("R0lGOD")
      ? "image/gif"
      : trimmed.startsWith("UklGR")
        ? "image/webp"
        : "image/png";
  return `data:${mime};base64,${trimmed}`;
}

export async function add(
  input: string | AddInput,
  scope: Scope,
  options?: { extract?: boolean; customInstructions?: string; observedAt?: string }
): Promise<AddResult> {
  // Accepts a bare string (every existing caller) or an object that may carry an
  // image. Normalised here so the rest of add() is unchanged either way.
  const normalized: AddInput = typeof input === "string" ? { text: input } : input ?? {};
  const trimmedText = (normalized.text ?? "").trim();
  const imageBase64 = (normalized.imageBase64 ?? "").trim();

  // Reject empty input before spending an embedding and an LLM call on it.
  if (trimmedText === "" && imageBase64 === "") {
    throw new Error('Invalid input: provide "text", "imageBase64", or both.');
  }
  const imageDataUrl = imageBase64 === "" ? undefined : toImageDataUrl(imageBase64);

  // Relative time references ("last week") resolve against observedAt, which
  // defaults to now — so live callers see no change, and a backfill can say when
  // the message was actually observed.
  const currentDate = new Date().toISOString();
  let observedAt = currentDate;
  if (options?.observedAt !== undefined) {
    const parsed = new Date(options.observedAt);
    if (Number.isNaN(parsed.getTime())) {
      throw new Error('Invalid input: "observedAt" must be a parseable ISO date string.');
    }
    observedAt = parsed.toISOString();
  }

  const validatedScope = validateScope(scope);
  await ensureCollection();
  const extract = options?.extract ?? true;

  // findRelated needs a query embedding, and an image-only add has no text to
  // embed. Such adds therefore run without related-memory context, which means no
  // dedup and no supersede linking for them — see the README note.
  // Counted once, before any write. countAfter is derived from stored.length
  // rather than issuing a second count. Superseded points are included and
  // procedural ones excluded, matching what this scope's facts cost to store.
  const countBefore = await store.count({
    ...scopeFilter(validatedScope),
    mustNot: [EXCLUDE_PROCEDURAL],
  });

  const related =
    extract && trimmedText !== "" ? await findRelated(trimmedText, validatedScope) : [];
  const labeled = labelMemories(related);
  const labelToMemory = new Map(labeled.map(({ label, memory }) => [label, memory]));

  const extracted = extract
    ? await extractFacts(
        trimmedText === "" ? "(no accompanying text — extract from the attached image)" : trimmedText,
        labeled,
        options?.customInstructions,
        observedAt,
        currentDate,
        imageDataUrl
      )
    : [{ content: trimmedText, supersedes: null, skip: false, skipReason: null }];
  const extractedAt = new Date().toISOString();

  // The base64 is deliberately not stored; the marker records that a fact came
  // from an image so provenance is not silently lost.
  const sourceText =
    imageDataUrl === undefined
      ? trimmedText
      : trimmedText === ""
        ? "[image]"
        : `[image] ${trimmedText}`;

  const stored: StoredFact[] = [];
  const skipped: SkippedFact[] = [];

  const dedupMap = new Map<string, { id: string; content: string }>();
  for (const { memory } of labeled) {
    dedupMap.set(hashContent(memory.content), { id: memory.id, content: memory.content });
  }

  for (const fact of extracted) {
    if (fact.skip) {
      const skipLabel = fact.skipReason?.match(/duplicate_of:\s*(\S+)/)?.[1];
      const skipTarget = skipLabel ? labelToMemory.get(skipLabel) : undefined;
      if (skipTarget) {
        skipped.push({
          content: fact.content,
          reason: "semantic_dedup",
          matchedExisting: { id: skipTarget.id, content: skipTarget.content },
        });
        continue;
      }
      // skipReason didn't resolve to a known memory — fall through and store it rather than lose the fact
    }

    // supersedes is now either a real label or null (see blankToNull). A label
    // that still fails to resolve is a model error, not a formatting artefact:
    // store the fact unlinked rather than lose it.
    const target = fact.supersedes !== null ? labelToMemory.get(fact.supersedes) : undefined;

    // Identical content is never a supersede — it carries no new information —
    // so the hash check runs even when the model asked to supersede that exact
    // memory. Under link-based supersession, honouring such a request would keep
    // both the old point and a byte-identical new one.
    const hash = hashContent(fact.content);
    const match = dedupMap.get(hash);

    if (match) {
      skipped.push({ content: fact.content, reason: "hash_dedup", matchedExisting: match });
      continue;
    }

    // Link-based supersession: the old point stays in the collection and is
    // filtered out at read time, so the history is recoverable rather than gone.
    const supersededMemoryId = target ? target.id : null;

    const vector = await embed(fact.content);
    const id = randomUUID();
    await store.upsert([
      {
        id,
        vector,
        payload: {
          content: fact.content,
          source: sourceText,
          extractedAt,
          observedAt,
          type: "fact",
          supersededMemoryId,
          userId: validatedScope.userId,
          agentId: validatedScope.agentId,
          runId: validatedScope.runId,
        },
      },
    ]);
    // Append-only history. A supersede is one event on the NEW memory carrying
    // both sides of the change, so the old content survives the old point.
    appendEvent(
      target
        ? {
            memoryId: id,
            event: "SUPERSEDE",
            oldContent: target.content,
            newContent: fact.content,
            supersedesMemoryId: target.id,
            scope: validatedScope,
          }
        : { memoryId: id, event: "ADD", oldContent: null, newContent: fact.content, scope: validatedScope }
    );

    stored.push({ id, content: fact.content, source: sourceText, extractedAt, observedAt, supersededMemoryId });
    dedupMap.set(hash, { id, content: fact.content });
  }

  const result: AddResult = {
    relatedMemories: labeled.map(({ label, memory }) => ({ ...memory, label })),
    stored,
    skipped,
    notice: buildNotice(countBefore, countBefore + stored.length),
  };
  await logEvent({
    call: "add",
    // The image itself is omitted from the log — it would be megabytes of base64.
    input: { text: trimmedText, hasImage: imageDataUrl !== undefined, scope, options },
    output: result,
  });

  return result;
}

export async function search(
  query: string,
  scope: Scope,
  options?: { includeSuperseded?: boolean }
): Promise<SearchResult[]> {
  const validatedScope = validateScope(scope);
  await ensureCollection();
  const includeSuperseded = options?.includeSuperseded ?? false;

  const vector = await embed(query);
  const points = await store.query(
    { ...scopeFilter(validatedScope), mustNot: [EXCLUDE_PROCEDURAL] },
    vector,
    HYBRID_CANDIDATE_POOL
  );

  const retrieved = points.map((point) => ({
    id: String(point.id),
    content: String(point.payload?.content ?? ""),
    source: String(point.payload?.source ?? ""),
    extractedAt: String(point.payload?.extractedAt ?? ""),
    supersededMemoryId:
      typeof point.payload?.supersededMemoryId === "string"
        ? point.payload.supersededMemoryId
        : null,
    cosineScore: point.score,
  }));

  // Second pass: drop candidates that a newer memory supersedes. Done before
  // BM25 so the superseded points do not skew the IDF of the surviving set.
  const supersededIds = await findSupersededIds(
    retrieved.map((c) => c.id),
    validatedScope
  );
  const candidates = includeSuperseded
    ? retrieved
    : retrieved.filter((c) => !supersededIds.has(c.id));

  const queryTokens = tokenize(query);
  const bm25Scores = computeBM25Scores(
    queryTokens,
    candidates.map((c) => ({ id: c.id, tokens: tokenize(c.content) }))
  );

  const results: SearchResult[] = candidates.map((c) => {
    const { bm25RawScore, bm25Normalized, combinedScore } = fuseScores(
      c.cosineScore,
      bm25Scores.get(c.id) ?? null
    );

    return {
      id: c.id,
      content: c.content,
      source: c.source,
      extractedAt: c.extractedAt,
      supersededMemoryId: c.supersededMemoryId,
      superseded: supersededIds.has(c.id),
      cosineScore: c.cosineScore,
      bm25RawScore,
      bm25Normalized,
      combinedScore,
    };
  });

  results.sort((a, b) => b.combinedScore - a.combinedScore);
  const top5 = results.slice(0, 5);

  await logEvent({ call: "search", input: { query, scope, options }, output: top5 });

  return top5;
}

// Unranked listing of a scope. No query embedding, no cosine, no BM25 — this is
// a listing, not a search, so the only ordering that means anything is time.
export async function getAll(
  scope: Scope,
  options?: { includeSuperseded?: boolean }
): Promise<MemoryRecord[]> {
  const validatedScope = validateScope(scope);
  await ensureCollection();
  const includeSuperseded = options?.includeSuperseded ?? false;

  const retrieved: MemoryRecord[] = [];
  for await (const page of store.scroll(
    { ...scopeFilter(validatedScope), mustNot: [EXCLUDE_PROCEDURAL] },
    SCROLL_PAGE_SIZE
  )) {
    for (const point of page) {
      retrieved.push({
        id: String(point.id),
        content: String(point.payload?.content ?? ""),
        source: String(point.payload?.source ?? ""),
        extractedAt: String(point.payload?.extractedAt ?? ""),
        supersededMemoryId:
          typeof point.payload?.supersededMemoryId === "string"
            ? point.payload.supersededMemoryId
            : null,
        superseded: false,
      });
    }
  }

  // Same supersession check search() uses, not a second implementation of it.
  const supersededIds = await findSupersededIds(
    retrieved.map((r) => r.id),
    validatedScope
  );

  const withFlags = retrieved.map((r) => ({ ...r, superseded: supersededIds.has(r.id) }));
  const results = includeSuperseded ? withFlags : withFlags.filter((r) => !r.superseded);

  results.sort((a, b) => b.extractedAt.localeCompare(a.extractedAt));

  await logEvent({ call: "getAll", input: { scope, options }, output: results });

  return results;
}

export async function addProcedural(
  instruction: string,
  scope: Scope
): Promise<ProceduralMemory> {
  const trimmed = instruction.trim();
  if (trimmed === "") throw new Error('Invalid input: "instruction" must not be empty.');

  const validatedScope = validateScope(scope);
  await ensureCollection();

  const createdAt = new Date().toISOString();
  const vector = await embed(trimmed);
  const id = randomUUID();
  await store.upsert([
    {
      id,
      vector,
      payload: {
        content: trimmed,
        source: trimmed,
        extractedAt: createdAt,
        type: "procedural",
        supersededMemoryId: null,
        userId: validatedScope.userId,
        agentId: validatedScope.agentId,
        runId: validatedScope.runId,
      },
    },
  ]);

  appendEvent({ memoryId: id, event: "ADD", oldContent: null, newContent: trimmed, scope: validatedScope });

  const result: ProceduralMemory = { id, content: trimmed, createdAt, type: "procedural" };
  await logEvent({ call: "addProcedural", input: { instruction, scope }, output: result });
  return result;
}

// Every procedural memory in a scope, newest first. Unranked by design: these
// are instructions to be followed, not candidates to be matched against a query,
// so there is nothing to score them against.
export async function getProcedural(scope: Scope): Promise<ProceduralMemory[]> {
  const validatedScope = validateScope(scope);
  await ensureCollection();

  const results: ProceduralMemory[] = [];
  const filter: MemoryFilter = {
    must: [...(scopeFilter(validatedScope).must ?? []), EXCLUDE_PROCEDURAL],
  };
  for await (const page of store.scroll(filter, SCROLL_PAGE_SIZE)) {
    for (const point of page) {
      results.push({
        id: String(point.id),
        content: String(point.payload?.content ?? ""),
        createdAt: String(point.payload?.extractedAt ?? ""),
        type: "procedural",
      });
    }
  }

  results.sort((a, b) => b.createdAt.localeCompare(a.createdAt));

  await logEvent({ call: "getProcedural", input: { scope }, output: results });
  return results;
}

function toStoredMemory(id: string, payload: Record<string, unknown>): StoredMemory {
  const rawType = payload.type;
  return {
    id,
    content: String(payload.content ?? ""),
    source: String(payload.source ?? ""),
    extractedAt: String(payload.extractedAt ?? ""),
    observedAt: typeof payload.observedAt === "string" ? payload.observedAt : null,
    supersededMemoryId:
      typeof payload.supersededMemoryId === "string" ? payload.supersededMemoryId : null,
    superseded: false,
    // Points written before `type` existed are facts.
    type: rawType === "procedural" ? "procedural" : "fact",
  };
}

// Fetch one memory by id. Scope-checked exactly as deleteMemory is, and returns
// a status rather than throwing so a miss and a refusal are handled the same way.
// The refusal deliberately carries no content: it must not become a way to read
// another scope's data by guessing ids.
export async function get(id: string, scope: Scope): Promise<GetResult> {
  const validatedScope = validateScope(scope);
  await ensureCollection();

  const [point] = await store.retrieve([id]);
  if (!point) return { status: "not_found", id };
  if (!pointMatchesScope(point.payload, validatedScope)) return { status: "scope_mismatch", id };

  const memory = toStoredMemory(String(point.id), point.payload ?? {});
  const successors = await findSupersededIds([memory.id], validatedScope);
  memory.superseded = successors.has(memory.id);

  const result: GetResult = { status: "found", memory };
  await logEvent({ call: "get", input: { id, scope }, output: result });
  return result;
}

// History is read from the append-only event log, never reconstructed from live
// memory fields. That is the whole point: a memory can be deleted from the store
// and its lineage still resolves, including the content it used to hold.
export async function history(id: string, scope: Scope): Promise<HistoryResult> {
  const validatedScope = validateScope(scope);

  if (!hasHistory(id, validatedScope)) {
    // No events in this scope. An unknown id and one belonging to another scope
    // are deliberately indistinguishable, so history() cannot be used to probe
    // for ids outside the caller's scope.
    const miss: HistoryResult = { status: "not_found", id };
    await logEvent({ call: "history", input: { id, scope }, output: miss });
    return miss;
  }

  // Walk the chain both ways using the supersede links recorded in the log,
  // then collect every event for every memory in it.
  const chain = new Set<string>([id]);

  let cursor: string | null = predecessorOf(id, validatedScope);
  while (cursor && !chain.has(cursor)) {
    chain.add(cursor);
    cursor = predecessorOf(cursor, validatedScope);
  }

  let frontier = successorsOf(id, validatedScope);
  while (frontier.length > 0) {
    const next: string[] = [];
    for (const memoryId of frontier) {
      if (chain.has(memoryId)) continue;
      chain.add(memoryId);
      next.push(...successorsOf(memoryId, validatedScope));
    }
    frontier = next;
  }

  const events = eventsForMemories([...chain], validatedScope);
  const result: HistoryResult = { status: "found", id, events };
  await logEvent({ call: "history", input: { id, scope }, output: result });
  return result;
}

// Deletes everything the scope covers, using the same hierarchical semantics as
// deleteMemory: a broad {userId} scope removes narrower agent/run-scoped points
// beneath it, and never reaches another userId.
//
// includeProcedural defaults to true, matching "delete every memory in this
// scope". Pass false to keep standing instructions while clearing learned facts
// — that is the only behaviour that would make reset() meaningfully different
// from this function.
export async function deleteAll(
  scope: Scope,
  options?: { includeProcedural?: boolean }
): Promise<DeleteAllResult> {
  const validatedScope = validateScope(scope);
  await ensureCollection();
  const includeProcedural = options?.includeProcedural ?? true;

  const filter: MemoryFilter = includeProcedural
    ? scopeFilter(validatedScope)
    : { ...scopeFilter(validatedScope), mustNot: [EXCLUDE_PROCEDURAL] };

  // Read the points first so each removal can be logged individually. A bulk
  // delete-by-filter alone would be one fewer round-trip but would erase what
  // was removed, which is exactly what the event log exists to prevent.
  const doomed: { id: string; content: string }[] = [];
  for await (const page of store.scroll(filter, SCROLL_PAGE_SIZE)) {
    for (const point of page) {
      doomed.push({ id: point.id, content: String(point.payload?.content ?? "") });
    }
  }

  if (doomed.length > 0) await store.deleteByFilter(filter);
  for (const d of doomed) {
    appendEvent({
      memoryId: d.id,
      event: "DELETE",
      oldContent: d.content,
      newContent: null,
      scope: validatedScope,
    });
  }

  const count = doomed.length;
  const result: DeleteAllResult = { deleted: count, includedProcedural: includeProcedural };
  await logEvent({ call: "deleteAll", input: { scope, options }, output: result });
  return result;
}

// Currently identical to deleteAll(scope) — see the note there. Kept as its own
// entry point because "reset this scope to empty" is a different intent from
// "delete these memories", and because it is the natural place to hang any
// future teardown that deleteAll should not do.
export async function reset(scope: Scope): Promise<DeleteAllResult> {
  const result = await deleteAll(scope, { includeProcedural: true });
  await logEvent({ call: "reset", input: { scope }, output: result });
  return result;
}

// Manual removal of one memory. Declared as deleteMemory because `delete` is a
// reserved word and cannot be a function declaration name; it is also exported
// under the name `delete` below, for `import * as memory; memory.delete(...)`.
//
// Dangling references: any memory in this scope whose supersededMemoryId points
// at the deleted point has that link nulled out first. Leaving the links would
// not crash anything — findSupersededIds only matches ids it was asked about, and
// a deleted id is never asked about — but a reference to a memory that no longer
// exists is unresolvable, and it would keep the referring fact looking like an
// update to something nobody can inspect.
export async function deleteMemory(id: string, scope: Scope): Promise<DeleteResult> {
  const validatedScope = validateScope(scope);
  await ensureCollection();

  const [point] = await store.retrieve([id]);

  if (!point) {
    const result: DeleteResult = { status: "not_found", id };
    await logEvent({ call: "delete", input: { id, scope }, output: result });
    return result;
  }

  // Scope enforcement: the caller may only delete what their own scope covers.
  // (Shared with get() and history() via pointMatchesScope.)
  // Deliberately does not return the content of a point outside the caller's
  // scope — a refusal should not become a read primitive for another scope.
  const payload = point.payload ?? {};
  const matchesScope = pointMatchesScope(payload, validatedScope);

  if (!matchesScope) {
    const result: DeleteResult = { status: "scope_mismatch", id };
    await logEvent({ call: "delete", input: { id, scope }, output: result });
    return result;
  }

  // Clear inbound supersede links before removing the point, so no window exists
  // in which a live memory references an id that is already gone.
  const referring: string[] = [];
  const referringFilter: MemoryFilter = {
    must: [
      ...(scopeFilter(validatedScope).must ?? []),
      { key: "supersededMemoryId", equals: id },
    ],
  };
  for await (const page of store.scroll(referringFilter, SCROLL_PAGE_SIZE, { payloadFields: [] })) {
    for (const p of page) referring.push(p.id);
  }

  if (referring.length > 0) {
    await store.setPayload(referring, { supersededMemoryId: null });
  }

  await store.deleteByIds([id]);

  // The point is gone from the live store; its history is not.
  appendEvent({
    memoryId: id,
    event: "DELETE",
    oldContent: String(payload.content ?? ""),
    newContent: null,
    scope: validatedScope,
  });

  const result: DeleteResult = {
    status: "deleted",
    id,
    content: String(payload.content ?? ""),
    clearedSupersedeLinks: referring,
  };
  await logEvent({ call: "delete", input: { id, scope }, output: result });
  return result;
}

export { deleteMemory as delete };

