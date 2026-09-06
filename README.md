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

`npm run test:suite` runs an 11-case regression suite against a live Qdrant
and the real extraction model. Each run namespaces its scopes under a
timestamped prefix and deletes exactly what it wrote, so repeated runs never
pollute each other or touch real data.

The cases cover: hash dedup, paraphrase handling, supersession on a genuine
update, the vague-update safety rail, branching supersession, scope
isolation, scope hierarchy (a broad scope reads and deletes narrow-scoped
points while cross-user deletion stays refused), chained supersession
(A→B→C leaves only C visible), deleting a successor to un-hide its original,
anti-hallucination, and BM25's contribution to ranking.

The paraphrase case is deliberately written to tolerate model
nondeterminism: it passes whether the model skips the restatement or treats
it as an update, but fails if the fact is lost or duplicated — because under
link-don't-delete, both classifications are safe and only loss is a bug.

`npm run test:extraction` is a separate, non-gating measurement of
extraction quality on scenarios where the model's judgement genuinely
varies between runs. It reports pass RATES over N attempts (default 6,
override with `ATTEMPTS`) and always exits 0. These live outside the suite
on purpose: at the measured rate for the harder case, any pass/fail
threshold fails more often than it succeeds, which would make the suite a
coin flip rather than a contract. A rate that drops across several runs is
the signal to investigate.

Current baseline on `gpt-4o-mini`, one message updating two facts at once:

| Scenario | Rate |
| --- | --- |
| `3b-explicit` — clear departure signal ("relocated to", "switched to") | 6/6 |
| `3b-ambiguous` — no departure signal ("stayed in", "also started at") | 0/6 |
| `3c` — dimensions with no worked example (relationship + habit) | ~50% |

The first two were previously one blended number, and splitting them is what
made the result interpretable. The dominant factor in whether an update links
turned out not to be the dimension or how well the prompt demonstrates it, but
whether the message signals that the **old fact stopped holding**. Measured in
isolation against the same stored job fact, "I started at a logistics company"
links 0/5, while "I switched to", "I left X and started at Y" and "I now work
at" each link 5/5 — and the same holds in reverse for location, where
"I relocated to Hyderabad" links 5/5 but "I stayed in Hyderabad this week"
links 0/5.

So `3b-ambiguous` scoring 0 is not a defect. "I also started at a logistics
company" does not assert the previous job ended — people hold two jobs — and
superseding there would hide a fact the user never retracted. A *high* rate on
that row would be the worse outcome. `3b-explicit` is the real capability
measure and should stay high.

These figures replace an earlier, higher baseline (~85% / ~40-65%) that was
partly measuring prompt contamination rather than reasoning. Several prompt
examples used the same concrete values as the test inputs — the
anti-hallucination example contained `145kg`, which is the exact number the
anti-hallucination test asserts on, and the vague-update, location, possession
and coreference examples reused `80kg`, Chennai/Bangalore, Honda/Tesla and a
person's name that the tests also used. Those examples now use bracketed
placeholders (`[N]kg`, `[CITY_NEW]`, `[PERSON]`), so a value echoed from an
example is immediately visible in output instead of looking plausible. The
numbers moved when the contamination was removed; the earlier ones were not a
like-for-like better result.

`npm run test:memory` is an interactive console for manual exploration:
add sentences one per line, then `list` / `list all` to see the scope,
`delete <id>` to remove a memory, and a search query at the end.

## Procedural memory

Facts describe the user; procedural memories are instructions the agent should
follow. They live in the same collection, separated by a `type` payload field,
and travel a deliberately minimal path:

- `addProcedural(instruction, scope)` stores the text **verbatim** — no
  extraction call, no dedup, no supersession. Rewriting an instruction risks
  changing what it tells the agent to do, and two similar instructions are not
  necessarily redundant, so neither mechanism is a safe default here.
- `getProcedural(scope)` returns all of them, newest first, unranked and
  paginated. There is no query to score them against: instructions are to be
  followed, not matched.
- `search()`, `getAll()` and the extraction context inside `add()` all exclude
  `type: "procedural"` at the Qdrant query level, so instructions never appear
  as facts and never become source material for extraction.
- `deleteMemory()` needed no changes and works on them unmodified, with the
  same scope enforcement.

Points written before `type` existed carry no such field. The exclusion filter
uses `must_not` on the procedural value rather than requiring `type: "fact"`,
so legacy points stay visible — a missing field cannot match.

## API and dashboard

Three thin Next.js routes wrap the library, plus a view-only page at
`/dashboard` for inspecting a scope and running searches:

| Route | Body / params |
| --- | --- |
| `POST /api/memory/add` | `{ text, scope, options? }` (or `imageBase64`) |
| `POST /api/memory/search` | `{ query, scope }` |
| `GET /api/memory/all` | `?userId=&agentId=&runId=&includeSuperseded=` |

The routes duplicate no validation: scope and input checks come from
`add()`/`search()`/`getAll()` themselves, and a thrown `Invalid ...` becomes a
400.

> **⚠️ There is no authentication of any kind on these routes.**
> Anyone who can reach the port can read, write and enumerate every scope, and
> `userId` is the only thing separating one user's memories from another's — it
> is an identifier, not a credential. Guessing or supplying another `userId` is
> enough to read that scope in full. Do not bind this to a public interface, put
> it behind a reverse proxy, or deploy it anywhere reachable from a network you
> do not control. It is a localhost debugging tool with the same trust model as
> the console script, and making it safe to expose would mean real authentication,
> per-user authorisation on every scope parameter, and rate limiting — none of
> which exist.

## Known limitations (deliberately deferred, not overlooked)

Consolidated here rather than scattered across commits. Numbers are measured,
not estimated — reproduce them with `npm run test:extraction`.

**Extraction quality (stochastic, model-dependent)**

Measured on `gpt-4o-mini`, one message updating two facts at once:

| Scenario | Rate |
| --- | --- |
| `3b-explicit` — clear departure signal ("relocated to", "switched to") | 6/6 |
| `3b-ambiguous` — no departure signal ("stayed in", "also started at") | 0/6 |
| `3c` — dimensions with no worked example (relationship + habit) | 3/6 to 6/6 across runs |

- **`3b-ambiguous` scoring 0 is the defensible outcome, not a bug.** "I also
  started at a logistics company" does not assert the previous job ended, so
  superseding would hide a fact the user never retracted. A *high* rate on that
  row would be worse.
- **The job-role exclusivity rule is a tracked candidate, deliberately not
  implemented.** Stating in the prompt that mentioning a new employer/home/partner
  without indicating the previous one ended is not a supersede would make the
  behaviour deliberate rather than emergent, at the risk of over-suppressing
  legitimate updates. It needs measuring before adoption.
- **`3c` varies run to run.** Gains from worked examples are per-dimension rather
  than cumulative, so an example-per-dimension arms race scales badly. A stronger
  extraction model closes much of the gap at roughly 10x per-token cost.
- **Pure rewords** are sometimes classified as updates rather than duplicates.
  Three prompt revisions failed to fix it; accepted because link-don't-delete
  makes the consequence mild (id churn, no data loss).

**Retrieval**

- **No associative retrieval.** Every memory is scored independently against the
  query, so a fact reachable only *through* another fact is unreachable. Confirmed
  by ablation: two linked facts sharing no vocabulary score bit-for-bit
  identically whether or not the other is present. A mechanism
  (`associatedMemoryIds` + an opt-in `expandAssociations`) is designed and
  deliberately not built — the diagnostics justified it, the cost/benefit was not
  yet worth it.
- **No memory strength or importance weighting.** Ranking is similarity only.
  Measured: asked "anything significant lately", a new job ranked *below*
  restocking a fridge and sorting a sock drawer, with the whole field spanning
  ~0.02 of combined score. Nothing encodes that some memories matter more.
- **No reconsolidation.** Memories are never revised, merged, or strengthened by
  being retrieved or re-encountered — retrieval is read-only, and a fact restated
  ten times is identical to one stated once.
- **`findRelated()` does not filter superseded facts**, so the extraction model
  can see stale versions when deciding what to supersede.
- **`search()` does not validate its query string** the way `add()` validates its
  input text.

**Infrastructure**

- **The API has no authentication** — see the warning above. Localhost only.
- **The provider seam has exactly one real implementation of each interface.**
  `VectorStoreClient` and `LLMClient` exist and are proven by an in-memory mock
  (`add()` → `search()` round-trip with no network), but the only production
  implementations are Qdrant and OpenAI. The abstraction is real; the portability
  is unproven against a second vendor, and a second vendor would likely surface
  assumptions the mock does not.
- **Temporal grounding is approximate.** `observedAt` resolves relative references
  ("last week") against the observation date rather than now, but resolution
  succeeds roughly two times in three and produces week-granular dates.

**A correction worth recording**

Feature #3 (usage notices) was originally understood as simple UX tips, by
analogy with a `notices.py` in mem0. That reading was wrong: mem0's is a
**commercial feature-gating system** — it decides what to advertise and gate
based on account state. What is implemented here is a deliberately simpler,
honest reimplementation of the *idea* of a proactive notice (first-run and
scale-threshold messages computed locally, returned in `AddResult.notice`, with
no telemetry and nothing leaving the system). It is not a port of theirs, and
should not be described as parity with it.

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
| `npm run test:suite` | 11-case regression suite, self-cleaning |
| `npm run test:e2e` | One session exercising every feature in sequence (26 checks) |
| `npm run test:providers` | Provider seam against in-memory mocks — no network, no API key |
| `npm run test:extraction` | Extraction-quality rates (non-gating, always exits 0) |
| `npm run test:memory` | Interactive console (add / list / delete / search) |
| `npm run dev` | Next.js dev server |

Configuration lives in [`lib/clients.ts`](lib/clients.ts): collection
`memory_facts`, embeddings `text-embedding-3-small` (1536 dims), extraction
`gpt-4o-mini`. Every `add`/`search`/`getAll`/`delete` call is appended to
`logs/memory.log.jsonl` with its full input and output, which is how most of
the failures above were diagnosed after the fact.
