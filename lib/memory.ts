import { randomUUID, createHash } from "crypto";
import {
  openai,
  qdrant,
  COLLECTION_NAME,
  EMBEDDING_MODEL,
  EMBEDDING_DIMENSIONS,
  EXTRACTION_MODEL,
} from "./clients";
import { logEvent } from "./logger";

const BM25_MIDPOINT = 5;
const BM25_STEEPNESS = 0.5;
const BM25_K1 = 1.5;
const BM25_B = 0.75;
const HYBRID_CANDIDATE_POOL = 20;
const SCROLL_PAGE_SIZE = 100;

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

function scopeFilter(scope: ValidatedScope) {
  const must: { key: string; match: { value: string } }[] = [
    { key: "userId", match: { value: scope.userId } },
  ];
  if (scope.agentId !== null) must.push({ key: "agentId", match: { value: scope.agentId } });
  if (scope.runId !== null) must.push({ key: "runId", match: { value: scope.runId } });
  return { must };
}

let collectionReady = false;

async function ensureCollection() {
  if (collectionReady) return;
  const { exists } = await qdrant.collectionExists(COLLECTION_NAME);
  if (!exists) {
    await qdrant.createCollection(COLLECTION_NAME, {
      vectors: { size: EMBEDDING_DIMENSIONS, distance: "Cosine" },
    });
  }
  collectionReady = true;
}

async function embed(text: string): Promise<number[]> {
  const res = await openai.embeddings.create({
    model: EMBEDDING_MODEL,
    input: text,
  });
  return res.data[0].embedding;
}

interface RelatedMemoryInternal extends RelatedMemory {
  id: string;
}

async function findRelated(text: string, scope: ValidatedScope): Promise<RelatedMemoryInternal[]> {
  const vector = await embed(text);
  const res = await qdrant.query(COLLECTION_NAME, {
    query: vector,
    filter: scopeFilter(scope),
    limit: 5,
    with_payload: true,
  });
  return res.points.map((point) => ({
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
  let offset: string | number | undefined | null = undefined;
  do {
    const page = await qdrant.scroll(COLLECTION_NAME, {
      filter: {
        must: [...base.must, { key: "supersededMemoryId", match: { any: ids } }],
      },
      limit: SCROLL_PAGE_SIZE,
      offset: offset ?? undefined,
      with_payload: ["supersededMemoryId"],
      with_vector: false,
    });
    for (const point of page.points) {
      const target = point.payload?.supersededMemoryId;
      if (typeof target === "string") superseded.add(target);
    }
    offset = page.next_page_offset as typeof offset;
  } while (offset !== null && offset !== undefined);
  return superseded;
}

interface LabeledMemory {
  label: string;
  memory: RelatedMemoryInternal;
}

function labelMemories(memories: RelatedMemoryInternal[]): LabeledMemory[] {
  return memories.map((memory, i) => ({ label: String(i), memory }));
}

function buildExtractionPrompt(text: string, labeled: LabeledMemory[]): string {
  const existingMemoriesBlock = labeled.length
    ? JSON.stringify(
        labeled.map(({ label, memory }) => ({ id: label, text: memory.content }))
      )
    : "[]";

  return `You will be given a new message and a list of existing memories that may be related, each labeled with a short id.

Existing Memories:
${existingMemoriesBlock}

New Message:
${text}

Extract any standalone facts from the new message as a list of objects: { "content": "...", "supersedes": "<id>" | null, "skip": true | false, "skipReason": "duplicate_of: <id>" | null }.

Rules:
1. If the new message describes a change from an old state to a new one (e.g. "used to be X, now Y" or "switched from X to Y"), extract ONLY the current/final state as one clear, self-contained fact.
2. For each fact you extract, check the Existing Memories list and decide "supersedes":
- Set "supersedes" to an existing memory's id when this fact updates or changes that memory's value. Otherwise set "supersedes" to null. Only set it when you're confident it's the same underlying fact with a new value — not just a related topic.
- A fact with no new information beyond what an existing memory already states is NOT a supersede — it's a duplicate, skip it entirely.
- Only set "supersedes" if the new fact is a genuine, complete replacement — at least as specific as the memory it supersedes. If the new message is too vague to match that specificity, do not set "supersedes"; store it as a new, unlinked memory instead.

Example:
Existing Memory: [{"id": "0", "text": "My bench press PR is 80kg."}]
New Message: "finally cracked past that bench plateau today"
WRONG: [{"content": "I cracked past my bench press plateau today.", "supersedes": "0", "skip": false, "skipReason": null}]  ← buries the only fact with a real number
CORRECT: [{"content": "Broke past previous bench press plateau, exact new PR not stated.", "supersedes": null, "skip": false, "skipReason": null}]  ← old 80kg fact stays visible, vague fact stored alongside it

When a message describes a change to an ongoing state — a location, job, role, relationship, living situation, possession, or habit — even if phrased as an event ("moved to", "switched to", "started", "got a new", "began") rather than a state, rewrite the extracted fact in the same current-state form as the existing memory it updates, so the connection is unambiguous.

Example 1 — location:
Existing Memory: [{"id": "0", "text": "My current city is Chennai."}]
New Message: "I moved to Bangalore."
WRONG: [{"content": "I moved to Bangalore.", "supersedes": null}]
CORRECT: [{"content": "My current city is Bangalore.", "supersedes": "0"}]

Example 2 — job/role:
Existing Memory: [{"id": "0", "text": "I work at a design studio as a product designer."}]
New Message: "I started at a fintech company as an engineer."
WRONG: [{"content": "I started at a fintech company as an engineer.", "supersedes": null}]
CORRECT: [{"content": "I work at a fintech company as an engineer.", "supersedes": "0"}]

Example 3 — living situation:
Existing Memory: [{"id": "0", "text": "I live with two roommates."}]
New Message: "I just moved into my own place."
WRONG: [{"content": "I just moved into my own place.", "supersedes": null}]
CORRECT: [{"content": "I live alone now.", "supersedes": "0"}]

Example 4 — possession:
Existing Memory: [{"id": "0", "text": "I drive a Honda Civic."}]
New Message: "Got a new car last week, a Tesla Model 3."
WRONG: [{"content": "Got a new car last week, a Tesla Model 3.", "supersedes": null}]
CORRECT: [{"content": "I drive a Tesla Model 3.", "supersedes": "0"}]

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

CRITICAL — Do not resolve pronouns, ambiguous references, or implied subjects using content from Existing Memories. If the New Message contains a pronoun ("his", "her", "their", "it") or an ambiguous reference whose subject is not explicitly stated in the New Message itself, extract the fact with the reference intact, exactly as written — do not substitute in a name or detail pulled from Existing Memories, even if it seems like an obvious, helpful resolution.

Example:
Existing Memory: [{"id": "0", "text": "Sameer's favorite color is green."}]
New Message: "Saw someone wearing his signature color today, didn't say hi."
WRONG: [{"content": "Saw someone wearing Sameer's favorite color today, didn't say hi.", "supersedes": null}]  ← "Sameer" was never stated in the New Message, only inferred from Existing Memories
CORRECT: [{"content": "Saw someone wearing his signature color today, didn't say hi.", "supersedes": null}]  ← reference left exactly as stated, unresolved

Example — do not invent unstated values:
New Message: "broke through a plateau today, finally past 145 on the lift"
WRONG Output: [{"content": "My deadlift PR is 150kg.", "supersedes": "0", "skip": false, "skipReason": null}]  ← 150 was never stated
CORRECT Output: [{"content": "Broke past 145kg on deadlift, exact new PR not stated.", "supersedes": "0", "skip": false, "skipReason": null}]

Example 1 — supersession:
Existing Memories: [{"id": "0", "text": "My squat PR is 100kg."}]
New Message: "Hit a new squat PR today, 110kg."
Output: [{"content": "My squat PR is 110kg.", "supersedes": "0", "skip": false, "skipReason": null}]

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

Return a JSON list of objects.`;
}

interface ExtractedFact {
  content: string;
  supersedes: string | null;
  skip: boolean;
  skipReason: string | null;
}

function blankToNull(value: string | null | undefined): string | null {
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  return trimmed === "" ? null : trimmed;
}

function normalizeForHash(text: string): string {
  return text.toLowerCase().trim().replace(/\s+/g, " ");
}

function hashContent(text: string): string {
  return createHash("md5").update(normalizeForHash(text)).digest("hex");
}

async function extractFacts(text: string, labeled: LabeledMemory[]): Promise<ExtractedFact[]> {
  const res = await openai.chat.completions.create({
    model: EXTRACTION_MODEL,
    messages: [
      {
        role: "user",
        content: buildExtractionPrompt(text, labeled),
      },
    ],
    tools: [
      {
        type: "function",
        function: {
          name: "return_facts",
          description: "Return the extracted facts.",
          parameters: {
            type: "object",
            properties: {
              facts: {
                type: "array",
                items: {
                  type: "object",
                  properties: {
                    content: { type: "string" },
                    supersedes: { type: ["string", "null"] },
                    skip: { type: "boolean" },
                    skipReason: { type: ["string", "null"] },
                  },
                  required: ["content", "supersedes", "skip", "skipReason"],
                },
              },
            },
            required: ["facts"],
          },
        },
      },
    ],
    tool_choice: { type: "function", function: { name: "return_facts" } },
  });

  const call = res.choices[0].message.tool_calls?.[0];
  if (!call) return [];
  const args = JSON.parse(call.function.arguments);
  if (!Array.isArray(args.facts)) return [];

  // The schema allows "string | null", and the model sometimes says "" where it
  // means null. An empty label would otherwise reach labelToMemory.get(""),
  // miss, and silently degrade a supersede into an unlinked new fact — the same
  // input producing different stored shapes with no error. Normalise once, here,
  // so downstream code only ever sees a real label or null.
  return (args.facts as ExtractedFact[]).map((fact) => ({
    ...fact,
    supersedes: blankToNull(fact.supersedes),
    skipReason: blankToNull(fact.skipReason),
  }));
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
}

export async function add(
  text: string,
  scope: Scope,
  options?: { extract?: boolean }
): Promise<AddResult> {
  // Reject empty input before spending an embedding and an LLM call on it.
  const trimmedText = text.trim();
  if (trimmedText === "") throw new Error('Invalid input: "text" must not be empty.');

  const validatedScope = validateScope(scope);
  await ensureCollection();
  const extract = options?.extract ?? true;

  const related = extract ? await findRelated(trimmedText, validatedScope) : [];
  const labeled = labelMemories(related);
  const labelToMemory = new Map(labeled.map(({ label, memory }) => [label, memory]));

  const extracted = extract
    ? await extractFacts(trimmedText, labeled)
    : [{ content: trimmedText, supersedes: null, skip: false, skipReason: null }];
  const extractedAt = new Date().toISOString();

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
    await qdrant.upsert(COLLECTION_NAME, {
      points: [
        {
          id,
          vector,
          payload: {
            content: fact.content,
            source: trimmedText,
            extractedAt,
            supersededMemoryId,
            userId: validatedScope.userId,
            agentId: validatedScope.agentId,
            runId: validatedScope.runId,
          },
        },
      ],
    });
    stored.push({ id, content: fact.content, source: trimmedText, extractedAt, supersededMemoryId });
    dedupMap.set(hash, { id, content: fact.content });
  }

  const result: AddResult = {
    relatedMemories: labeled.map(({ label, memory }) => ({ ...memory, label })),
    stored,
    skipped,
  };
  await logEvent({ call: "add", input: { text, scope, options }, output: result });

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
  const res = await qdrant.query(COLLECTION_NAME, {
    query: vector,
    filter: scopeFilter(validatedScope),
    limit: HYBRID_CANDIDATE_POOL,
    with_payload: true,
  });

  const retrieved = res.points.map((point) => ({
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
  let offset: string | number | undefined | null = undefined;
  do {
    const page = await qdrant.scroll(COLLECTION_NAME, {
      filter: scopeFilter(validatedScope),
      limit: SCROLL_PAGE_SIZE,
      offset: offset ?? undefined,
      with_payload: true,
      with_vector: false,
    });
    for (const point of page.points) {
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
    offset = page.next_page_offset as typeof offset;
  } while (offset !== null && offset !== undefined);

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

  const [point] = await qdrant.retrieve(COLLECTION_NAME, {
    ids: [id],
    with_payload: true,
    with_vector: false,
  });

  if (!point) {
    const result: DeleteResult = { status: "not_found", id };
    await logEvent({ call: "delete", input: { id, scope }, output: result });
    return result;
  }

  // Scope enforcement: the caller may only delete what their own scope covers.
  // Deliberately does not return the content of a point outside the caller's
  // scope — a refusal should not become a read primitive for another scope.
  const payload = point.payload ?? {};
  const pointScope = {
    userId: typeof payload.userId === "string" ? payload.userId : null,
    agentId: typeof payload.agentId === "string" ? payload.agentId : null,
    runId: typeof payload.runId === "string" ? payload.runId : null,
  };
  // Hierarchical, matching read semantics: userId must match exactly, but an
  // unspecified agentId/runId in the caller's scope is a wildcard, exactly as
  // scopeFilter() treats it for search()/getAll(). A caller can therefore delete
  // anything it can see — and nothing it cannot. Cross-userId remains strict.
  const matchesScope =
    pointScope.userId === validatedScope.userId &&
    (validatedScope.agentId === null || pointScope.agentId === validatedScope.agentId) &&
    (validatedScope.runId === null || pointScope.runId === validatedScope.runId);

  if (!matchesScope) {
    const result: DeleteResult = { status: "scope_mismatch", id };
    await logEvent({ call: "delete", input: { id, scope }, output: result });
    return result;
  }

  // Clear inbound supersede links before removing the point, so no window exists
  // in which a live memory references an id that is already gone.
  const referring: string[] = [];
  let offset: string | number | undefined | null = undefined;
  do {
    const page = await qdrant.scroll(COLLECTION_NAME, {
      filter: {
        must: [
          ...scopeFilter(validatedScope).must,
          { key: "supersededMemoryId", match: { value: id } },
        ],
      },
      limit: SCROLL_PAGE_SIZE,
      offset: offset ?? undefined,
      with_payload: false,
      with_vector: false,
    });
    for (const p of page.points) referring.push(String(p.id));
    offset = page.next_page_offset as typeof offset;
  } while (offset !== null && offset !== undefined);

  if (referring.length > 0) {
    await qdrant.setPayload(COLLECTION_NAME, {
      payload: { supersededMemoryId: null },
      points: referring,
      wait: true,
    });
  }

  await qdrant.delete(COLLECTION_NAME, { points: [id], wait: true });

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

