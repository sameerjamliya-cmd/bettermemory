import { randomUUID, createHash } from "crypto";
import { QdrantVectorStore } from "./providers/qdrant-store";
import { OpenAILLMClient } from "./providers/openai-llm";
import type {
  ExtractedFact,
  FieldCondition,
  LLMClient,
  MemoryFilter,
  VectorStoreClient,
} from "./providers/types";
import { logEvent } from "./logger";

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

const BM25_MIDPOINT = 5;
const BM25_STEEPNESS = 0.5;
const BM25_K1 = 1.5;
const BM25_B = 0.75;
const HYBRID_CANDIDATE_POOL = 20;
const SCROLL_PAGE_SIZE = 100;
// Counts at which add() surfaces a one-off scale notice. Crossing a value fires
// once; ordinary calls either side of it stay silent.
const SCALE_NOTICE_THRESHOLDS = [100, 500, 1000];

export interface Scope {
  userId: string;
  agentId?: string;
  runId?: string;
}

interface ValidatedScope {
  userId: string;
  agentId: string | null;
  runId: string | null;
}

function validateId(field: string, value: string | undefined, required: boolean): string | null {
  if (value === undefined || value === null) {
    if (required) throw new Error(`Invalid scope: "${field}" is required.`);
    return null;
  }
  const trimmed = value.trim();
  if (trimmed === "") {
    throw new Error(`Invalid scope: "${field}" must not be empty.`);
  }
  if (/\s/.test(trimmed)) {
    throw new Error(`Invalid scope: "${field}" must not contain whitespace.`);
  }
  return trimmed;
}

function validateScope(scope: Scope): ValidatedScope {
  const userId = validateId("userId", scope.userId, true) as string;
  const agentId = validateId("agentId", scope.agentId, false);
  const runId = validateId("runId", scope.runId, false);
  return { userId, agentId, runId };
}

export type MemoryType = "fact" | "procedural";

// Points written before `type` existed have no such field. Filtering with
// must_not on the procedural value (rather than must type == "fact") keeps
// those legacy points visible, since a missing field cannot match.
const EXCLUDE_PROCEDURAL: FieldCondition = { key: "type", equals: "procedural" };

function scopeFilter(scope: ValidatedScope): MemoryFilter {
  const must: FieldCondition[] = [{ key: "userId", equals: scope.userId }];
  if (scope.agentId !== null) must.push({ key: "agentId", equals: scope.agentId });
  if (scope.runId !== null) must.push({ key: "runId", equals: scope.runId });
  return { must };
}

// Hierarchical, matching read semantics: userId must match exactly, but an
// unspecified agentId/runId in the caller's scope is a wildcard, exactly as
// scopeFilter() treats it for search()/getAll(). A caller can therefore act on
// anything it can see — and nothing it cannot. Cross-userId remains strict.
function pointMatchesScope(
  payload: Record<string, unknown> | null | undefined,
  scope: ValidatedScope
): boolean {
  const p = payload ?? {};
  const userId = typeof p.userId === "string" ? p.userId : null;
  const agentId = typeof p.agentId === "string" ? p.agentId : null;
  const runId = typeof p.runId === "string" ? p.runId : null;
  return (
    userId === scope.userId &&
    (scope.agentId === null || agentId === scope.agentId) &&
    (scope.runId === null || runId === scope.runId)
  );
}

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

function normalizeBM25(rawScore: number, midpoint: number, steepness: number): number {
  return 1 / (1 + Math.exp(-steepness * (rawScore - midpoint)));
}

function tokenize(text: string): string[] {
  return text.toLowerCase().match(/[a-z0-9]+/g) ?? [];
}

function computeBM25Scores(
  queryTokens: string[],
  docs: { id: string; tokens: string[] }[]
): Map<string, number> {
  const scores = new Map<string, number>();
  const N = docs.length;
  if (N === 0 || queryTokens.length === 0) {
    for (const doc of docs) scores.set(doc.id, 0);
    return scores;
  }

  const avgdl = docs.reduce((sum, d) => sum + d.tokens.length, 0) / N;

  const df = new Map<string, number>();
  for (const term of new Set(queryTokens)) {
    let count = 0;
    for (const doc of docs) {
      if (doc.tokens.includes(term)) count++;
    }
    df.set(term, count);
  }

  for (const doc of docs) {
    let score = 0;
    const dl = doc.tokens.length;
    for (const term of queryTokens) {
      const f = doc.tokens.filter((t) => t === term).length;
      if (f === 0) continue;
      const n = df.get(term) ?? 0;
      const idf = Math.log((N - n + 0.5) / (n + 0.5) + 1);
      score += (idf * (f * (BM25_K1 + 1))) / (f + BM25_K1 * (1 - BM25_B + BM25_B * (dl / avgdl)));
    }
    scores.set(doc.id, score);
  }

  return scores;
}

// Returns the subset of `ids` that some newer point in the same scope claims to
// supersede. One paginated scroll for the whole candidate set, not one query per
// candidate. Pagination rather than a limit derived from ids.length: several
// points can independently supersede the same memory (branching), so the number
// of matching points has no upper bound in terms of the number of ids asked
// about. This is a filtered lookup, not a collection scan, so the pages are
// bounded by how much supersession actually exists in the scope.
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

interface LabeledMemory {
  label: string;
  memory: RelatedMemoryInternal;
}

function labelMemories(memories: RelatedMemoryInternal[]): LabeledMemory[] {
  return memories.map((memory, i) => ({ label: String(i), memory }));
}

function buildExtractionPrompt(
  text: string,
  labeled: LabeledMemory[],
  customInstructions?: string,
  observedAt?: string,
  currentDate?: string
): string {
  // Placed after every rule and example, so a per-call instruction supplements
  // the core contract rather than replacing it, and immediately before the
  // output-format line so that stays last. The guard sentence is not optional:
  // anti-hallucination, dedup and coreference are structural guarantees, and a
  // caller-supplied string must not be able to switch them off.
  const additionalInstructions = customInstructions?.trim()
    ? `
## Additional Instructions For This Call
${customInstructions.trim()}

Additional instructions may refine what counts as significant or how facts are phrased, but do not override the rules above regarding hallucination, dedup, or coreference.
`
    : "";

  // Both dates are always supplied by add(); they are optional here only so the
  // prompt builder stays usable in isolation.
  const observation = observedAt ?? new Date().toISOString();
  const now = currentDate ?? new Date().toISOString();

  const existingMemoriesBlock = labeled.length
    ? JSON.stringify(
        labeled.map(({ label, memory }) => ({ id: label, text: memory.content }))
      )
    : "[]";

  return `You will be given a new message and a list of existing memories that may be related, each labeled with a short id.

Existing Memories:
${existingMemoriesBlock}

Observation Date (when the New Message was observed): ${observation}
Current Date (now): ${now}

New Message:
${text}

Extract any standalone facts from the new message as a list of objects: { "content": "...", "supersedes": "<id>" | null, "skip": true | false, "skipReason": "duplicate_of: <id>" | null }.

Note on the examples below: values in square brackets ([N], [CITY_NEW], [PERSON], ...) are placeholders standing in for real values. They illustrate the shape of correct output only. Never copy a bracketed placeholder, or any other literal value from an example, into a fact's "content" — the wording of every fact you extract must come from the New Message itself. This does not restrict "supersedes" or "skipReason", which reference the Existing Memories by their id and must still be set whenever the rules above call for it.

Rules:
1. If the new message describes a change from an old state to a new one (e.g. "used to be X, now Y" or "switched from X to Y"), extract ONLY the current/final state as one clear, self-contained fact.
2. For each fact you extract, check the Existing Memories list and decide "supersedes":
- Set "supersedes" to an existing memory's id when this fact updates or changes that memory's value. Otherwise set "supersedes" to null. Only set it when you're confident it's the same underlying fact with a new value — not just a related topic.
- A fact with no new information beyond what an existing memory already states is NOT a supersede — it's a duplicate, skip it entirely.
- Only set "supersedes" if the new fact is a genuine, complete replacement — at least as specific as the memory it supersedes. If the new message is too vague to match that specificity, do not set "supersedes"; store it as a new, unlinked memory instead.

Example:
Existing Memory: [{"id": "0", "text": "My bench press PR is [N]kg."}]
New Message: "finally cracked past that bench plateau today"
WRONG: [{"content": "I cracked past my bench press plateau today.", "supersedes": "0", "skip": false, "skipReason": null}]  ← buries the only fact with a real number
CORRECT: [{"content": "Broke past previous bench press plateau, exact new PR not stated.", "supersedes": null, "skip": false, "skipReason": null}]  ← old [N]kg fact stays visible, vague fact stored alongside it

When a message describes a change to an ongoing state — a location, job, role, relationship, living situation, possession, or habit — even if phrased as an event ("moved to", "switched to", "started", "got a new", "began") rather than a state, rewrite the extracted fact in the same current-state form as the existing memory it updates, so the connection is unambiguous.

Example 1 — location:
Existing Memory: [{"id": "0", "text": "My current city is [CITY_OLD]."}]
New Message: "I moved to [CITY_NEW]."
WRONG: [{"content": "I moved to [CITY_NEW].", "supersedes": null}]
CORRECT: [{"content": "My current city is [CITY_NEW].", "supersedes": "0"}]

Example 2 — job/role:
Existing Memory: [{"id": "0", "text": "I work at a design studio as a product designer."}]
New Message: "I now work at a fintech company as an engineer."
WRONG: [{"content": "I now work at a fintech company as an engineer.", "supersedes": null}]
CORRECT: [{"content": "I work at a fintech company as an engineer.", "supersedes": "0"}]

Example 3 — living situation:
Existing Memory: [{"id": "0", "text": "I live with two roommates."}]
New Message: "I just moved into my own place."
WRONG: [{"content": "I just moved into my own place.", "supersedes": null}]
CORRECT: [{"content": "I live alone now.", "supersedes": "0"}]

Example 4 — possession:
Existing Memory: [{"id": "0", "text": "I drive a [CAR_OLD]."}]
New Message: "Got a new car last week, a [CAR_NEW]."
WRONG: [{"content": "Got a new car last week, a [CAR_NEW].", "supersedes": null}]
CORRECT: [{"content": "I drive a [CAR_NEW].", "supersedes": "0"}]

If a fact you would extract is semantically equivalent to an existing memory shown above, with no new information, still include it in the output, but set "skip": true and "skipReason": "duplicate_of: <id>" (the id of the existing memory it duplicates). This is different from "supersedes": use "supersedes" when the new fact updates or changes an old value; use "skip": true when the new fact is just a restatement of information that's already fully captured, with nothing new to store. When "skip" is true, "supersedes" must be null and "skipReason" must be set; when "skip" is false, "skipReason" must be null.

3. Extract each distinct fact as exactly one sentence. Do not split one fact into multiple entries, and do not merge two distinct facts into one.
4. If there's nothing worth remembering, return an empty list.

CRITICAL — Extract ONLY from the New Message. Never extract facts based on:
- Your own knowledge, assumptions, or inferences about the user
- The Existing Memories shown above (they are context for deciding "supersedes", not source material for new facts)
- What seems plausible, likely, or a reasonable guess

If a fact is not explicitly and literally stated in the New Message text, do not extract it — even if it seems related to or implied by the conversation history.

Before extracting each fact, verify: "Is this sentence, or something equivalent to it, actually present in the New Message?" If not, discard it.

Example — do not extract from context:
Existing Memories: [{"id": "0", "text": "My favorite language for backend work used to be Python, but I've switched to TypeScript."}]
New Message: "I go to the gym 5 days a week."
WRONG Output: [{"content": "I go to the gym 5 days a week.", "supersedes": null, "skip": false, "skipReason": null}, {"content": "My favorite language is TypeScript.", "supersedes": null, "skip": false, "skipReason": null}]  ← TypeScript fact was NOT in this New Message, it leaked in from Existing Memories
CORRECT Output: [{"content": "I go to the gym 5 days a week.", "supersedes": null, "skip": false, "skipReason": null}]

CRITICAL — Resolve every relative time reference in the New Message ("last week", "yesterday", "this morning", "a couple of months ago") against the Observation Date, never against the Current Date. The two are usually the same, but when they differ the Observation Date is the moment the message describes and is authoritative. State the resolved date in the fact where it matters, so the fact stays true when read later.

Resolving a relative reference into a concrete date this way is expected and is NOT an invented value — the Observation Date is given to you above for exactly this purpose. It is the ONLY value that may enter a fact from outside the New Message, and it may ONLY be used to fill in a time reference. It must not influence any other decision: it never affects whether two facts describe the same underlying thing, and it never justifies a "supersedes" link.

Example:
Observation Date: 2026-03-14T09:00:00.000Z
Current Date: 2026-09-06T12:00:00.000Z
New Message: "Hit a new deadlift PR last week."
WRONG: [{"content": "Hit a new deadlift PR in the week of 2026-08-31.", "supersedes": null}]  ← resolved against the Current Date
CORRECT: [{"content": "Hit a new deadlift PR in the week of 2026-03-09.", "supersedes": null}]  ← resolved against the Observation Date

CRITICAL — Do not resolve pronouns, ambiguous references, or implied subjects using content from Existing Memories. If the New Message contains a pronoun ("his", "her", "their", "it") or an ambiguous reference whose subject is not explicitly stated in the New Message itself, extract the fact with the reference intact, exactly as written — do not substitute in a name or detail pulled from Existing Memories, even if it seems like an obvious, helpful resolution.

Example:
Existing Memory: [{"id": "0", "text": "[PERSON]'s favorite color is [COLOR]."}]
New Message: "Saw someone wearing his signature color today, didn't say hi."
WRONG: [{"content": "Saw someone wearing [PERSON]'s favorite color today, didn't say hi.", "supersedes": null}]  ← "[PERSON]" was never stated in the New Message, only inferred from Existing Memories
CORRECT: [{"content": "Saw someone wearing his signature color today, didn't say hi.", "supersedes": null}]  ← reference left exactly as stated, unresolved

Example — do not invent unstated values:
New Message: "broke through a plateau today, finally past [N] on the lift"
WRONG Output: [{"content": "My deadlift PR is [M]kg.", "supersedes": "0", "skip": false, "skipReason": null}]  ← [M] was never stated anywhere in the New Message
CORRECT Output: [{"content": "Broke past [N]kg on deadlift, exact new PR not stated.", "supersedes": "0", "skip": false, "skipReason": null}]

Example 1 — supersession:
Existing Memories: [{"id": "0", "text": "My squat PR is [N]kg."}]
New Message: "Hit a new squat PR today, [M]kg."
Output: [{"content": "My squat PR is [M]kg.", "supersedes": "0", "skip": false, "skipReason": null}]

Example 2 — no relation:
Existing Memories: [{"id": "0", "text": "I work at a healthtech startup."}]
New Message: "I'm learning Rust on the side."
Output: [{"content": "I am learning Rust.", "supersedes": null, "skip": false, "skipReason": null}]

Example — skip a restatement:
Existing Memories: [{"id": "0", "text": "I usually train at the gym in the evening."}]
New Message: "I usually train at the gym in the evening"
Output: [{"content": "I usually train at the gym in the evening.", "supersedes": null, "skip": true, "skipReason": "duplicate_of: 0"}]

Example — skip a paraphrased duplicate (not just verbatim):
Existing Memories: [{"id": "0", "text": "I usually train at the gym in the evening."}]
New Message: "I like working out in the evenings at the gym"
WRONG: [{"content": "I like working out in the evenings at the gym.", "supersedes": "0"}]  ← no new information was added, this is not an update, it's the same fact reworded
CORRECT: [{"content": "I like working out in the evenings at the gym.", "supersedes": null, "skip": true, "skipReason": "duplicate_of: 0"}]

${additionalInstructions}
Return a JSON list of objects.`;
}

function normalizeForHash(text: string): string {
  return text.toLowerCase().trim().replace(/\s+/g, " ");
}

function hashContent(text: string): string {
  return createHash("md5").update(normalizeForHash(text)).digest("hex");
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

export interface RelatedMemory {
  content: string;
  source: string;
  score: number;
  extractedAt: string;
}

export interface MemoryRecord {
  id: string;
  content: string;
  source: string;
  extractedAt: string;
  supersededMemoryId: string | null;
  superseded: boolean;
}

export interface SearchResult {
  id: string;
  content: string;
  source: string;
  extractedAt: string;
  // What this fact itself superseded, and whether a newer fact has superseded it.
  supersededMemoryId: string | null;
  superseded: boolean;
  cosineScore: number;
  bm25RawScore: number | null;
  bm25Normalized: number | null;
  combinedScore: number;
}

export interface StoredFact {
  id: string;
  content: string;
  source: string;
  extractedAt: string;
  // When the message was observed. Defaults to write time, so it equals
  // extractedAt for live use and differs only when a caller backfills.
  observedAt: string;
  // The id of the memory this fact supersedes. The superseded point is left in
  // the collection; search() hides it by default instead of deleting it.
  supersededMemoryId: string | null;
}

export interface SkippedFact {
  content: string;
  reason: "semantic_dedup" | "hash_dedup";
  matchedExisting: { id: string; content: string };
}

export interface AddResult {
  relatedMemories: (RelatedMemory & { label: string })[];
  stored: StoredFact[];
  skipped: SkippedFact[];
  // Informational only, and null on the overwhelming majority of calls. Computed
  // locally from the scope's own count — nothing is recorded or sent anywhere.
  notice: string | null;
}

// One notice per call at most, first-run taking precedence over scale, mirroring
// an if/elif chain rather than accumulating messages.
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

export interface AddInput {
  text?: string;
  // Raw base64, or a full data: URL. Never stored — only sent to the extraction
  // call — because a base64 image in a Qdrant payload would dwarf the fact.
  imageBase64?: string;
}

// PNG and JPEG cover what this path is for; the prefix check avoids asking the
// caller for a mime type they may not have. An unrecognised payload is passed as
// PNG rather than rejected, since the API will reject a genuinely bad image with
// a clearer error than we could produce here.
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
    const bm25RawScore = bm25Scores.get(c.id) ?? null;
    const bm25Normalized =
      bm25RawScore !== null ? normalizeBM25(bm25RawScore, BM25_MIDPOINT, BM25_STEEPNESS) : null;

    let total = c.cosineScore;
    let signals = 1;
    if (bm25RawScore !== null) {
      total += bm25Normalized as number;
      signals += 1;
    }
    const combinedScore = total / signals;

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

export interface ProceduralMemory {
  id: string;
  content: string;
  createdAt: string;
  type: "procedural";
}

// Procedural memory: an instruction the agent should follow, not a fact about
// the user. Stored verbatim — no extraction call, no dedup, no supersession —
// because rewriting an instruction risks changing what it tells the agent to do,
// and two similar instructions are not necessarily redundant. Kept in the same
// collection, separated by the `type` payload field and filtered out of
// search()/getAll()/the extraction context at the Qdrant query level.
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

export interface StoredMemory {
  id: string;
  content: string;
  source: string;
  extractedAt: string;
  observedAt: string | null;
  supersededMemoryId: string | null;
  superseded: boolean;
  type: MemoryType;
}

export type GetResult =
  | { status: "found"; memory: StoredMemory }
  | { status: "not_found"; id: string }
  | { status: "scope_mismatch"; id: string };

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

export interface HistoryResult {
  status: "found" | "not_found" | "scope_mismatch";
  id: string;
  // Oldest first: what this memory replaced, and what that replaced, and so on.
  ancestors?: StoredMemory[];
  memory?: StoredMemory;
  // Everything that superseded this memory, directly or transitively. More than
  // one direct successor is possible (branching supersession).
  descendants?: StoredMemory[];
}

async function loadInScope(
  id: string,
  scope: ValidatedScope
): Promise<StoredMemory | null> {
  const [point] = await store.retrieve([id]);
  if (!point || !pointMatchesScope(point.payload, scope)) return null;
  return toStoredMemory(String(point.id), point.payload ?? {});
}

async function directSuccessors(id: string, scope: ValidatedScope): Promise<StoredMemory[]> {
  const out: StoredMemory[] = [];
  const filter: MemoryFilter = {
    must: [...(scopeFilter(scope).must ?? []), { key: "supersededMemoryId", equals: id }],
  };
  for await (const page of store.scroll(filter, SCROLL_PAGE_SIZE)) {
    for (const p of page) out.push(toStoredMemory(p.id, p.payload));
  }
  return out;
}

// The lifecycle of one memory, reconstructed from the supersededMemoryId links
// rather than a separate event log: walk backwards through what it replaced and
// forwards through what replaced it. No new storage is needed because the links
// are never deleted — that is the whole point of link-based supersession.
export async function history(id: string, scope: Scope): Promise<HistoryResult> {
  const validatedScope = validateScope(scope);
  await ensureCollection();

  const [point] = await store.retrieve([id]);
  if (!point) return { status: "not_found", id };
  if (!pointMatchesScope(point.payload, validatedScope)) return { status: "scope_mismatch", id };

  const memory = toStoredMemory(String(point.id), point.payload ?? {});
  const seen = new Set<string>([memory.id]);

  // Backwards: what this replaced, and what that replaced.
  const ancestors: StoredMemory[] = [];
  let cursor: string | null = memory.supersededMemoryId;
  while (cursor && !seen.has(cursor)) {
    seen.add(cursor);
    const ancestor: StoredMemory | null = await loadInScope(cursor, validatedScope);
    if (!ancestor) break;
    ancestor.superseded = true;
    ancestors.unshift(ancestor);
    cursor = ancestor.supersededMemoryId;
  }

  // Forwards: everything that superseded this, breadth-first so branching is
  // captured rather than only the first path.
  const descendants: StoredMemory[] = [];
  let frontier = await directSuccessors(memory.id, validatedScope);
  while (frontier.length > 0) {
    const next: StoredMemory[] = [];
    for (const node of frontier) {
      if (seen.has(node.id)) continue;
      seen.add(node.id);
      descendants.push(node);
      next.push(...(await directSuccessors(node.id, validatedScope)));
    }
    frontier = next;
  }
  for (const d of descendants) {
    d.superseded = descendants.some((other) => other.supersededMemoryId === d.id);
  }
  memory.superseded = descendants.some((d) => d.supersededMemoryId === memory.id);

  const result: HistoryResult = { status: "found", id, ancestors, memory, descendants };
  await logEvent({ call: "history", input: { id, scope }, output: result });
  return result;
}

export interface DeleteAllResult {
  deleted: number;
  includedProcedural: boolean;
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

  const count = await store.count(filter);
  if (count > 0) await store.deleteByFilter(filter);

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

export type DeleteResult =
  | { status: "deleted"; id: string; content: string; clearedSupersedeLinks: string[] }
  | { status: "not_found"; id: string }
  | { status: "scope_mismatch"; id: string };

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

