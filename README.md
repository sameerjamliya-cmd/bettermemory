# Personal Memory Layer

A memory layer for AI agents, built from scratch to understand how systems
like mem0 actually work — not a production system, a learning project
where every feature exists because a real failure justified it.

Stack: Next.js/TypeScript, Qdrant, OpenAI (embeddings + extraction).

## Philosophy

Most of this wasn't designed upfront. It started as the dumbest possible
pipeline — raw extraction, no dedup, no conflict handling — and every
addition below was built only after that pipeline broke in an observable,
reproducible way. Where a design choice diverges from mem0 (which was read
directly from source throughout this project, not just its docs), that's
noted.

## Phase 1 — Extraction and Conflict Resolution

Started bare: one LLM call, no rules. Immediately broke — a deadlift PR
updated from 140kg to 145kg and both facts sat in the store, nearly tied
in cosine similarity (0.7300 vs 0.7275). Pure semantic search cannot tell
you what's currently true, only what's textually similar to a query.

Built in response:
- Context-aware extraction — existing related memories are retrieved first
  and passed into the extraction call as labeled ids (`"0"`, `"1"`, …), so
  the model can reason about what it already knows
- Two-layer dedup — semantic (LLM judgment against existing memories) +
  a hash-based exact-match backstop over normalized content
- Explicit `skip: true` reporting so semantic dedup is auditable, not a
  silent omission: every skip surfaces in `AddResult.skipped` with a
  `reason` of `semantic_dedup` or `hash_dedup` and the memory it matched
- Anti-hallucination rules, added after the model invented a deadlift
  number ("past 145" became "150kg") and separately inflated a tentative
  claim ("learning Rust") into a strong one ("favorite language")
- A vague-update safety rail — an update only overwrites an existing fact
  if it's at least as specific; otherwise it's stored separately, so a
  vague follow-up can never destroy a specific fact

## Phase 2 — Retrieval and Scoping

- Scoping (`userId`/`agentId`/`runId`) enforced at the Qdrant query level,
  not as an application-side filter — a scoped search cannot see another
  scope's data even indirectly. Scope ids are validated and normalized at
  the entry point of every public function, and the internal helpers accept
  a distinct `ValidatedScope` type, so unvalidated input cannot reach a
  query by construction
- BM25 keyword search added alongside cosine similarity, sigmoid-normalized
  (midpoint 5, steepness 0.5) and fused via an adaptive divisor — the score
  is the mean of however many signals were actually available, so a fact
  with no keyword overlap is not penalized against one that has it
  (inspired by mem0's `score_and_rank`, reasoned through independently
  before comparing against theirs)
- Entity linking was deliberately tested, not assumed necessary: three
  topically unrelated facts sharing only a name ("Sameer") were all
  correctly retrieved by a query about that name — BM25's exact-keyword
  matching did the job entity linking exists for in mem0, so it wasn't built

## Phase 3 — Conflict Resolution Redesign, and Store Management

Originally, updates deleted the old fact outright (`replaces`). That made
every skip-vs-update judgment call destructive — and testing showed the
model would sometimes misclassify a reworded duplicate as an update.
Rather than keep fighting that judgment call with prompt engineering, the
architecture changed: updates now link to and hide the old fact
(`supersedes`) instead of deleting it. A wrong classification now costs
nothing destructive — this mirrors mem0's actual `linked_memory_ids`
design, but adds the automatic read-time filtering mem0 itself is
missing (its `expiration_date` field exists but nothing sets it
automatically).

Concretely: the extraction model returns `supersedes: "<label>" | null`, and
a superseding fact is stored with a `supersededMemoryId` payload field
pointing at the memory it replaces. The old point stays in the collection.
`search()` runs a second pass that excludes any candidate another point
claims to supersede, and takes an optional `includeSuperseded` flag
(default `false`) to look at the history deliberately — mirroring the
visibility mem0 keeps via `show_expired`.

Also added:
- `getAll(scope, options?)` — full, unranked listing of a scope's memories,
  sorted newest-first by `extractedAt`, paginated to handle collections
  larger than one Qdrant scroll page (100 points)
- `deleteMemory(id, scope)` — manual removal, scope-enforced, with dangling
  `supersededMemoryId` references cleaned up on delete. Also exported as
  `delete` (`import * as memory` → `memory.delete(...)`), since `delete` is
  a reserved word and cannot be a function declaration name. Returns a
  discriminated result (`deleted` / `not_found` / `scope_mismatch`) rather
  than throwing; a cross-scope refusal deliberately returns only the id, so
  the error path can't be used to read another scope's content
- Fixed a branching-supersession edge case: the supersession lookup capped
  its scroll at the number of ids it was asked about, which assumed each id
  had at most one superseder. When several memories independently supersede
  the same original, the matches can outnumber the ids and a stale fact
  could leak into results. Now paginated instead. In practice this was only
  reachable through `search()` on a scope larger than the 20-point candidate
  pool — `getAll()` was structurally immune, since its id set is the whole
  scope

## Testing

`npm run test:suite` runs an 8-case regression suite against a live Qdrant
and the real extraction model. Each run namespaces its scopes under a
timestamped prefix and deletes exactly what it wrote, so repeated runs never
pollute each other or touch real data.

The cases cover: hash dedup, paraphrase handling, supersession on a genuine
update, the vague-update safety rail, branching supersession, scope
isolation, anti-hallucination, and BM25's contribution to ranking.

Two of these are deliberately written to tolerate model nondeterminism. The
paraphrase case passes whether the model skips the restatement or treats it
as an update, but fails if the fact is lost or duplicated — because under
link-don't-delete, both classifications are safe and only loss is a bug.

`npm run test:memory` is an interactive console for manual exploration:
add sentences one per line, then `list` / `list all` to see the scope,
`delete <id>` to remove a memory, and a search query at the end.

## Known limitations (deliberately deferred, not overlooked)

- No temporal/observation-date grounding — every fact is implicitly "as
  of when `add()` was called." Fine for live use, would break on
  backfilling historical data (the same gap found in mem0's own
  open-source version)
- The model occasionally misclassifies a pure reword (no new information)
  as an update rather than a duplicate — confirmed not fixable via prompt
  engineering alone after three attempts; accepted because the
  link-don't-delete architecture makes the consequence mild
- No `history()`/audit trail per memory
- `findRelated()` (the context passed to extraction) does not filter
  superseded facts, so the model can see stale versions when deciding what
  to supersede
- `search()` does not validate its query string the way `add()` validates
  its input text

## Setup

Requires Node 18+ and Docker.

```bash
# 1. Start Qdrant (stores data in a named Docker volume, qdrant_storage)
docker compose up -d

# 2. Install dependencies
npm install

# 3. Configure credentials
cp .env.example .env.local
# then set OPENAI_API_KEY in .env.local

# 4. Verify the whole pipeline against live Qdrant + OpenAI
npm run test:suite
```

`.env.local` is gitignored and never committed.

| Command | What it does |
| --- | --- |
| `npm run test:suite` | 8-case regression suite, self-cleaning |
| `npm run test:memory` | Interactive console (add / list / delete / search) |
| `npm run dev` | Next.js dev server |

Configuration lives in [`lib/clients.ts`](lib/clients.ts): collection
`memory_facts`, embeddings `text-embedding-3-small` (1536 dims), extraction
`gpt-4o-mini`. Every `add`/`search`/`getAll`/`delete` call is appended to
`logs/memory.log.jsonl` with its full input and output, which is how most of
the failures above were diagnosed after the fact.
