# bettermemory

A memory layer for AI agents, built from scratch — not to ship a product, but to understand how systems like mem0 actually work by reading their source, testing every design decision, and fixing what broke.

Stack: Next.js/TypeScript, Qdrant, OpenAI (embeddings + extraction, vision).

## The approach

Every feature here exists because a real, reproducible failure justified it — not because a reference system has it. Where mem0 or Graphiti (Zep's temporal graph engine) were used as reference points, their actual source was read directly, not their docs or marketing — and in several cases, that surfaced real gaps in *their* systems, or proved a feature they have isn't needed here at all.

This shows up in the numbers: stochastic behaviors are tracked as measured rates with real sample sizes, not pass/fail claims. When an early test read 6/6, it got re-measured before being trusted — one turned out to be partly grading a model on an answer sitting in its own prompt.

## What was learned from mem0 (and what wasn't copied)

Read directly from `mem0ai/mem0`'s source, not its docs:

- **Two-layer dedup** (semantic LLM judgment + hash backstop) — adopted, with full observability added (mem0's own dedup is a silent omission you can't audit)
- **Link, don't delete, on update** — mem0's `linked_memory_ids` design was adopted deliberately after finding their actual `_add_to_vector_store` still supports outright deletion by default. This project's version also closes a real gap mem0 has: `expiration_date` exists in their schema but nothing in their code ever sets it — this project's `supersedes` mechanism does both halves, linking *and* automatically hiding at read time
- **BM25 + cosine fusion** — mem0's adaptive-divisor formula and query-length-tuned sigmoid parameters were read from `scoring.py` and tested directly against this implementation's own BM25. The tuning table didn't transfer: this system's raw BM25 scores don't scale with query length the way mem0's do (a 2-term query scored 3.22 raw where an 18-term query scored 3.74 — barely more), so adopting their constants would have suppressed keyword matching for no reason. Checked, not needed.
- **Entity linking, graph memory** — tested with adversarial scenarios designed to isolate real association from lexical coincidence (four rounds of test design, each round finding a hidden flaw in the last). Verdict: not needed for the cases tested — BM25 and embedding proximity already handle them. The one genuine gap found is recorded under limitations below.
- **`notices.py`** — inspired an early "usage notices" feature before actually reading the source, which turned out to be commercial feature-gating and telemetry, not UX tips. Corrected: this project's notices are a simpler, honest reimplementation, not a port
- **`storage.py`** — reading mem0's real `history()` (a true append-only SQLite log) revealed this project's first `history()` implementation had a real gap: it was reconstructed from live data and would break if a root memory in a chain was ever deleted. Rebuilt as an independent event log that survives deletion of what it references.

## What was learned from Graphiti (Zep's temporal graph engine)

Read `getzep/graphiti`'s source specifically to address an unresolved bug in this project's own supersede logic. The key finding: Graphiti splits the *decision* into two separate steps — an LLM identifies candidate relationships (semantic judgment), then **pure deterministic date comparison** decides whether to actually invalidate, with the model never touching that half.

This system now uses the same split:

- **Timestamp gate** — a fact can no longer be silently overwritten by something chronologically older just because it was *written* second (`observedAt`-aware, catches out-of-order backfills). The link is still recorded for lineage, flagged `outOfOrder`, but it never hides the newer fact.
- **Attribute-key gate** — a fact can no longer be superseded by something the model *thinks* is related but is actually a different underlying attribute (fixed a real, reproducible bug: "bench PR" was hiding "squat PR" 7% of the time, purely on semantic over-matching, unrelated to timing)

## The bug that mattered

A wrong-supersede defect — the only bug in this project that could **silently hide true data** — was found, reproduced at a measured 7/100 rate, and traced to its exact cause: the extraction model treating two different lift types as "the same fact, updated," because both matched the shape `"[exercise] PR is [n]kg"`.

Getting there took discipline. An earlier n=50 study appeared to blame an unrelated prompt addition (0/50 vs 3/50); re-run at n=150 per arm, the effect *reversed* (11/150 vs 2/150, p = 0.02) — a direction flip under identical methodology, which is the signature of an uncontrolled variable, not a cause. A same-length neutral filler block scoring 0/50 had already ruled out prompt length. The suspected culprit was cleared, and the real one only surfaced once concrete failure examples were captured rather than aggregate rates.

The fix (a deterministic `attributeKey` gate, Graphiti-inspired) was built, and its first version introduced the exact regression that was flagged as a risk before building it — a legitimate job-role update got silently blocked because the model split one life dimension into two attribute names (`current_job_title` vs `current_employer`), leaving two contradictory job facts live. Caught by the regression suite, fixed by asking for the coarsest attribute grouping that still separates genuinely different facts, and reverified: the bug closed to 0/100 without breaking any of the four known-good supersession cases.

## Features

- Context-aware extraction with anti-hallucination and coreference-resolution guards, each added after observing the model do the specific thing they now prevent
- Two-layer dedup, fully observable
- Link-based supersession with deterministic timestamp and attribute gates
- Hierarchical scoping (`userId`/`agentId`/`runId`), enforced at the database query level throughout — including a caught asymmetry where memories were readable but not deletable through the same scope
- Hybrid BM25 + cosine retrieval
- Procedural memory (standing instructions, separate from fact retrieval)
- Custom per-call extraction instructions, with adversarial testing confirming they can't bypass the structural safety rules
- Temporal grounding (`observedAt`) for backfilling historical data
- Vision (image → extracted facts)
- Append-only event log (SQLite via Node's built-in `node:sqlite`), so `history()` survives deletion of what it describes
- Provider-agnostic seam for the vector store and LLM, proven by an in-memory mock that runs `add()` → `search()` with no network
- Minimal REST API + dashboard, with API-key auth
- mem0 method parity: `add`, `search`, `get`, `getAll`, update (via supersede), `delete`, `deleteAll`, `reset`, `history` — `chat()` excluded, since it's an unimplemented stub in mem0 itself

Tests: `npm run test:suite` (11 deterministic cases), `test:e2e` (27 checks across every feature in one session), `test:providers` (mock providers, no network), `test:extraction` (stochastic rates, non-gating).

## Known limitations — stated honestly, not omitted

- **Multi-dimension update linking is stochastic**, not solved: 6/6 on demonstrated patterns with an explicit departure signal (e.g. "switched to"), 0/6 by design on ambiguous ones with no such signal (correct behavior — the model rightly declines to infer an ending that wasn't stated), ~6/6 on undemonstrated dimensions after prompt tuning. Real, measured numbers — not the early inflated 6/6 that turned out to be partly measuring prompt contamination.
- **A job-role exclusivity rule** was identified as a real fix but never built — tracked as an open candidate requiring its own dedicated test.
- **The `attributeKey` gate only protects memories written after it existed.** A candidate with no stored key is treated as unknown rather than mismatched, so the link is allowed through — a deliberate choice, since treating "missing" as "different" would have silently broken every update against pre-existing data.
- **Retrieval has no associative expansion.** Confirmed by ablation, not assumed: two linked facts sharing no vocabulary score bit-for-bit identically whether or not the other is present, so a fact reachable *only* through another fact is unreachable. A mechanism was designed and deliberately not built.
- **No usage-based decay or reinforcement** (a memory's rank never adapts to how often it's retrieved) — scoped, deliberately deferred. Measured consequence: asked "anything significant lately", a new job ranked *below* restocking a fridge.
- **No reconsolidation** (retrieval never mutates a memory, unlike human recall) — flagged from the start as too structurally risky, since every safety guarantee in this project assumes reads are non-destructive.
- **Provider-agnostic seam exists but is proven against only one real implementation** (Qdrant + OpenAI) — the abstraction is real (verified via a mock provider), not battle-tested against a second vendor.
- **The API key gates who may call the API; it does not isolate one user from another.** `userId` is an identifier, not a credential: any caller holding the key can address any scope. Multi-tenant use would need per-user credentials and authorisation on every scope parameter.

## Setup

See [QUICKSTART.md](./QUICKSTART.md).
