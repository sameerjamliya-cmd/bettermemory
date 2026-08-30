import { add, search, getAll, type Scope } from "../lib/memory";
import { qdrant, COLLECTION_NAME } from "../lib/clients";
import { randomUUID } from "crypto";

// Every scope this run creates is namespaced under one prefix, so the suite can
// delete exactly what it wrote at the end and never touch real user data.
const RUN_PREFIX = `test-suite-run-${Date.now()}`;

const usedUserIds = new Set<string>();

function testScope(caseName: string): Scope {
  const userId = `${RUN_PREFIX}-${caseName}`;
  usedUserIds.add(userId);
  return { userId };
}

async function cleanup() {
  if (usedUserIds.size === 0) return;
  await qdrant.delete(COLLECTION_NAME, {
    filter: {
      should: [...usedUserIds].map((userId) => ({ key: "userId", match: { value: userId } })),
    },
    wait: true,
  });
}

interface TestResult {
  name: string;
  passed: boolean;
  message?: string;
  actual?: unknown;
  expected?: unknown;
}

const results: TestResult[] = [];

async function runTest(name: string, fn: () => Promise<void>) {
  try {
    await fn();
    results.push({ name, passed: true });
  } catch (err) {
    if (err instanceof AssertionFailure) {
      results.push({ name, passed: false, message: err.message, actual: err.actual, expected: err.expected });
    } else {
      results.push({ name, passed: false, message: err instanceof Error ? err.stack ?? err.message : String(err) });
    }
  }
}

class AssertionFailure extends Error {
  actual: unknown;
  expected: unknown;
  constructor(message: string, actual: unknown, expected: unknown) {
    super(message);
    this.actual = actual;
    this.expected = expected;
  }
}

function assertEqual(message: string, actual: unknown, expected: unknown) {
  if (actual !== expected) {
    throw new AssertionFailure(message, actual, expected);
  }
}

function assertTrue(message: string, condition: boolean, actual?: unknown, expected?: unknown) {
  if (!condition) {
    throw new AssertionFailure(message, actual, expected);
  }
}

async function testHashDedup() {
  const scope = testScope("case1-hash-dedup");
  const sentence = "I always drink coffee before my morning workout.";

  await add(sentence, scope);
  const second = await add(sentence, scope);

  assertTrue(
    "second call should skip exactly one fact",
    second.skipped.length === 1,
    second.skipped,
    "array of length 1"
  );
  assertTrue(
    'skip reason should be "hash_dedup" or "semantic_dedup"',
    second.skipped[0]?.reason === "hash_dedup" || second.skipped[0]?.reason === "semantic_dedup",
    second.skipped[0]?.reason,
    "hash_dedup or semantic_dedup"
  );
  assertTrue("stored should be empty", second.stored.length === 0, second.stored, "[]");
}

// A paraphrase of an existing fact is a judgement call the extraction model does
// not make reliably: it may recognise the restatement (skip) or read it as an
// update (supersede). Under link-not-delete both are acceptable — the original
// fact survives either way — so this case accepts either and fails only if the
// paraphrase is stored as an unrelated new fact, or if anything is lost.
async function testSemanticDedupDivergentPhrasing() {
  const scope = testScope("case2-semantic-dedup");

  const first = await add("My favorite snack is trail mix.", scope);
  const firstId = first.stored[0]?.id;
  assertTrue("first call should store one fact", !!firstId, first.stored, "one stored fact with an id");

  const second = await add("I love eating trail mix as a snack.", scope);

  const caughtAsDuplicate =
    second.skipped.length === 1 && second.skipped[0]?.reason === "semantic_dedup";
  const treatedAsUpdate =
    second.stored.length === 1 && second.stored[0]?.supersededMemoryId === firstId;

  assertTrue(
    "paraphrase should be either skipped as a duplicate or stored as a supersede linked to the original",
    caughtAsDuplicate || treatedAsUpdate,
    {
      skipped: second.skipped,
      stored: second.stored.map((f) => ({ content: f.content, supersededMemoryId: f.supersededMemoryId })),
      originalId: firstId,
    },
    "skipped[0].reason === 'semantic_dedup', or stored[0].supersededMemoryId === the original id"
  );

  // Either way, nothing may be deleted: the original point must still exist.
  const retrieved = await qdrant.retrieve(COLLECTION_NAME, { ids: [firstId as string], with_payload: true });
  assertTrue(
    "the original trail-mix point must still exist regardless of which branch was taken",
    retrieved.length === 1,
    retrieved.map((p) => p.payload?.content),
    `one point with id ${firstId}`
  );

  // And the fact must still be reachable: exactly one live trail-mix memory.
  const live = await search("what is my favorite snack", scope);
  const trailMix = live.filter((r) => r.content.toLowerCase().includes("trail mix"));
  assertTrue(
    "default search should surface exactly one live trail-mix fact",
    trailMix.length === 1,
    live.map((r) => ({ content: r.content, superseded: r.superseded })),
    "exactly one trail-mix result"
  );
}

async function testSupersedesOnGenuineUpdate() {
  const scope = testScope("case3-supersedes");

  const first = await add("My deadlift PR is 140kg.", scope);
  const firstId = first.stored[0]?.id;
  assertTrue("first call should store one fact", !!firstId, first.stored, "one stored fact with an id");

  const second = await add("Just hit a new deadlift PR, 145kg this time.", scope);

  assertTrue(
    "second call should store one fact",
    second.stored.length === 1,
    second.stored,
    "array of length 1"
  );
  assertEqual(
    "stored[0].supersededMemoryId should match first memory's id",
    second.stored[0]?.supersededMemoryId,
    firstId
  );

  // The point of link-based supersession: the old fact is preserved, not deleted.
  const retrieved = await qdrant.retrieve(COLLECTION_NAME, { ids: [firstId as string], with_payload: true });
  assertTrue(
    "the superseded 140kg point should still exist in Qdrant",
    retrieved.length === 1 && String(retrieved[0]?.payload?.content ?? "").includes("140kg"),
    retrieved.map((p) => p.payload?.content),
    "one point whose content contains '140kg'"
  );

  // ...but it must not compete in default retrieval.
  const defaultResults = await search("what is my deadlift pr", scope);
  const defaultContents = defaultResults.map((r) => r.content);
  assertTrue(
    "default search should return the 145kg fact",
    defaultContents.some((c) => c.includes("145kg")),
    defaultContents,
    "an entry containing '145kg'"
  );
  assertTrue(
    "default search should NOT return the superseded 140kg fact",
    !defaultContents.some((c) => c.includes("140kg")),
    defaultContents,
    "no entry containing '140kg'"
  );

  // Opting in brings the history back.
  const withSuperseded = await search("what is my deadlift pr", scope, { includeSuperseded: true });
  const allContents = withSuperseded.map((r) => r.content);
  assertTrue(
    "includeSuperseded search should surface the 140kg fact again",
    allContents.some((c) => c.includes("140kg")),
    allContents,
    "an entry containing '140kg'"
  );
  assertTrue(
    "the 140kg result should be flagged superseded: true",
    withSuperseded.find((r) => r.content.includes("140kg"))?.superseded === true,
    withSuperseded.map((r) => ({ content: r.content, superseded: r.superseded })),
    "superseded: true on the 140kg entry"
  );
}

async function testVagueUpdateSafetyRail() {
  const scope = testScope("case4-vague-update");

  await add("My bench press PR is 80kg.", scope);
  const second = await add("finally cracked past that bench plateau today", scope);

  assertTrue(
    "second call's facts should all have supersededMemoryId: null",
    second.stored.every((f) => f.supersededMemoryId === null),
    second.stored.map((f) => f.supersededMemoryId),
    "all null"
  );

  const searchResults = await search("bench press", scope);
  const contents = searchResults.map((r) => r.content);
  assertTrue(
    "search should still surface the original 80kg fact",
    contents.some((c) => c.includes("80kg")),
    contents,
    "an entry containing '80kg'"
  );
  assertTrue(
    "search should still surface the new vague fact",
    contents.some((c) => c.toLowerCase().includes("plateau")),
    contents,
    "an entry containing 'plateau'"
  );
}

// Branching supersession: two points independently claiming to supersede the
// same original. findSupersededIds must still detect the original, which is why
// it paginates rather than capping its scroll at the number of ids it asked about.
async function testBranchingSupersession() {
  const scope = testScope("case8-branching");

  const first = await add("My current city is Chennai.", scope);
  const chennaiId = first.stored[0]?.id;
  assertTrue("setup should store the Chennai fact", !!chennaiId, first.stored, "one stored fact");

  await add("Moved to Bangalore recently.", scope);
  await add("Actually now I'm in Bangalore, for work.", scope);

  // The extractor tends to produce a chain (each fact superseding the previous),
  // so force a genuine second superseder of the ORIGINAL to guarantee branching.
  // Reuses an existing vector so no embedding call is needed.
  const [donor] = await qdrant.retrieve(COLLECTION_NAME, {
    ids: [chennaiId as string],
    with_vector: true,
  });
  await qdrant.upsert(COLLECTION_NAME, {
    wait: true,
    points: [
      {
        id: randomUUID(),
        vector: donor.vector as number[],
        payload: {
          content: "I live in Bangalore these days.",
          source: "synthetic branching superseder",
          extractedAt: new Date().toISOString(),
          supersededMemoryId: chennaiId,
          userId: (scope as { userId: string }).userId,
          agentId: null,
          runId: null,
        },
      },
    ],
  });

  const superseders = await qdrant.scroll(COLLECTION_NAME, {
    filter: {
      must: [
        { key: "userId", match: { value: (scope as { userId: string }).userId } },
        { key: "supersededMemoryId", match: { value: chennaiId as string } },
      ],
    },
    limit: 20,
    with_payload: false,
    with_vector: false,
  });
  assertTrue(
    "the scenario must actually branch: 2+ points superseding the Chennai fact",
    superseders.points.length >= 2,
    superseders.points.length,
    "at least 2"
  );

  const searched = await search("what city do I live in", scope);
  assertTrue(
    "search should not leak the superseded Chennai fact",
    !searched.some((r) => r.content.includes("Chennai")),
    searched.map((r) => ({ content: r.content, superseded: r.superseded })),
    "no result containing 'Chennai'"
  );

  const listed = await getAll(scope);
  assertTrue(
    "getAll should not leak the superseded Chennai fact",
    !listed.some((m) => m.content.includes("Chennai")),
    listed.map((m) => ({ content: m.content, superseded: m.superseded })),
    "no entry containing 'Chennai'"
  );

  // Preserved, not deleted — and visible when asked for.
  const listedAll = await getAll(scope, { includeSuperseded: true });
  const chennai = listedAll.find((m) => m.content.includes("Chennai"));
  assertTrue(
    "includeSuperseded listing should surface the Chennai fact flagged superseded",
    chennai?.superseded === true,
    listedAll.map((m) => ({ content: m.content, superseded: m.superseded })),
    "Chennai entry with superseded: true"
  );
}

async function testScopingIsolation() {
  const scopeA = testScope("case5-testA");
  const scopeB = testScope("case5-testB");
  const query = "I have a pet iguana named Steve.";

  await add(query, scopeA);
  const resultsB = await search(query, scopeB);

  assertTrue(
    "userB should see zero results for a fact only stored under userA",
    resultsB.length === 0,
    resultsB,
    "[]"
  );
}

async function testNoHallucinatedValues() {
  const scope = testScope("case6-no-hallucination");

  await add("My deadlift PR is 145kg.", scope);
  const second = await add("broke through a plateau today, finally past 145 on the lift", scope);

  const numbersFound = second.stored
    .flatMap((f) => f.content.match(/\d+/g) ?? [])
    .filter((n) => n !== "145");

  assertTrue(
    `stored content should not contain any number other than 145${
      numbersFound.length ? ` — invented number(s): ${numbersFound.join(", ")}` : ""
    }`,
    numbersFound.length === 0,
    { inventedNumbers: numbersFound, storedContent: second.stored.map((f) => f.content) },
    "no numbers other than 145"
  );
}

async function testBM25ContributesOnExactTermQuery() {
  const scope = testScope("case7-bm25");

  await add("My squat PR is 90kg.", scope);
  await add("My bench PR is 90kg.", scope);

  const searchResults = await search("what's my squat PR", scope);

  const squat = searchResults.find((r) => r.content.toLowerCase().includes("squat"));
  const bench = searchResults.find((r) => r.content.toLowerCase().includes("bench"));

  assertTrue(
    "both the squat and bench facts should be stored and searchable",
    !!squat && !!bench,
    searchResults.map((r) => r.content),
    "results containing both a squat fact and a bench fact"
  );

  assertTrue(
    "squat fact should outrank bench fact on combinedScore for an exact-term squat query",
    (squat as NonNullable<typeof squat>).combinedScore > (bench as NonNullable<typeof bench>).combinedScore,
    searchResults.map((r) => ({
      content: r.content,
      cosineScore: r.cosineScore,
      bm25RawScore: r.bm25RawScore,
      combinedScore: r.combinedScore,
    })),
    "squat.combinedScore > bench.combinedScore"
  );
}

async function main() {
  try {
    await runTest("Hash dedup: identical sentence added twice", testHashDedup);
    await runTest(
      "Paraphrase: skipped as duplicate or linked as supersede, never lost",
      testSemanticDedupDivergentPhrasing
    );
    await runTest(
      "Supersedes: genuine update links to and hides the old fact",
      testSupersedesOnGenuineUpdate
    );
    await runTest("Vague-update safety rail: no replace on incomplete info", testVagueUpdateSafetyRail);
    await runTest(
      "Branching supersession: two superseders, original still hidden",
      testBranchingSupersession
    );
    await runTest("Scoping isolation: userB cannot see userA's memory", testScopingIsolation);
    await runTest("No hallucinated values: vague number not invented", testNoHallucinatedValues);
    await runTest("BM25 contributes: exact term outranks sibling fact", testBM25ContributesOnExactTermQuery);
  } finally {
    try {
      await cleanup();
      console.log(`\nCleaned up ${usedUserIds.size} test scope(s) under "${RUN_PREFIX}".`);
    } catch (err) {
      console.error(`\nWARNING: cleanup failed for prefix "${RUN_PREFIX}":`, err);
    }
  }

  console.log("\n=== Regression Suite Results ===\n");
  let failCount = 0;
  for (const r of results) {
    if (r.passed) {
      console.log(`PASS  ${r.name}`);
    } else {
      failCount++;
      console.log(`FAIL  ${r.name}`);
      if (r.message) console.log(`      ${r.message}`);
      if ("actual" in r || "expected" in r) {
        console.log(`      actual:   ${JSON.stringify(r.actual)}`);
        console.log(`      expected: ${JSON.stringify(r.expected)}`);
      }
    }
  }

  console.log(`\n${results.length - failCount}/${results.length} passed.\n`);
  process.exitCode = failCount > 0 ? 1 : 0;
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
