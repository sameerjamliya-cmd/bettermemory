// Extraction-quality measurement, deliberately NOT part of the pass/fail suite.
//
// These scenarios depend on a judgement the extraction model makes, and it does
// not make it the same way every run. Gating on them turns the suite into a coin
// flip: at the measured ~39% per-attempt rate for the undemonstrated-dimension
// case, a "2 of 3 attempts" gate fails roughly 70% of the time — worse than the
// single-shot flake it was meant to replace. So this script reports RATES over a
// fixed number of attempts instead, giving a number to track across prompt and
// model changes rather than a green light that has to be tuned to the model's mood.
//
// Note on the numbers: several prompt examples once reused the same concrete
// values as these scenarios and as the regression suite (145kg, 80kg,
// Chennai/Bangalore), so earlier, higher rates were partly measuring recall of
// the prompt rather than reasoning. Those examples now use bracketed
// placeholders, and these rates are the corrected, like-for-like baseline.
//
// Also note that 3b's job clause ("also started at a logistics company") does
// not actually assert that the previous job ended, so a portion of its failure
// rate is the model declining a supersede it was never licensed to make.
//
// Run: npm run test:extraction
import { add, getAll, type Scope } from "../lib/memory";
import { qdrant, COLLECTION_NAME, EXTRACTION_MODEL } from "../lib/clients";

const ATTEMPTS = Number(process.env.ATTEMPTS ?? 6);
const RUN_PREFIX = `test-extraction-run-${Date.now()}`;
const usedUserIds = new Set<string>();

function testScope(name: string): Scope {
  const userId = `${RUN_PREFIX}-${name}`;
  usedUserIds.add(userId);
  return { userId };
}

async function cleanup() {
  if (usedUserIds.size === 0) return;
  await qdrant.delete(COLLECTION_NAME, {
    filter: { should: [...usedUserIds].map((userId) => ({ key: "userId", match: { value: userId } })) },
    wait: true,
  });
}

interface Scenario {
  name: string;
  slug: string;
  factA: string;
  factB: string;
  update: string;
  staleA: RegExp;
  staleB: RegExp;
}

const SCENARIOS: Scenario[] = [
  {
    name: "3b — dimensions demonstrated in the prompt (location + job)",
    slug: "3b",
    factA: "My current city is Pune.",
    factB: "I work as a designer at an edtech company.",
    update: "I relocated to Hyderabad and also started at a logistics company.",
    staleA: /Pune/,
    staleB: /edtech/,
  },
  {
    name: "3c — dimensions NOT demonstrated in the prompt (relationship + habit)",
    slug: "3c",
    factA: "I have been single for about a year.",
    factB: "I train at the gym on weekday mornings.",
    update: "I started seeing someone new, and I've also switched my workouts to the evenings.",
    staleA: /single/i,
    staleB: /morning/i,
  },
];

interface Attempt {
  passed: boolean;
  aLinked: boolean;
  bLinked: boolean;
  staleLive: boolean;
  stored: string[];
}

async function attempt(s: Scenario, i: number): Promise<Attempt> {
  const scope = testScope(`${s.slug}-try${i}`);
  const a = (await add(s.factA, scope)).stored[0];
  const b = (await add(s.factB, scope)).stored[0];
  const r = await add(s.update, scope);

  const aLinked = r.stored.filter((f) => f.supersededMemoryId === a?.id).length === 1;
  const bLinked = r.stored.filter((f) => f.supersededMemoryId === b?.id).length === 1;
  const live = (await getAll(scope)).map((m) => m.content);
  const staleLive = live.some((c) => s.staleA.test(c)) || live.some((c) => s.staleB.test(c));

  return {
    passed: aLinked && bLinked && !staleLive,
    aLinked,
    bLinked,
    staleLive,
    stored: r.stored.map((f) => `${f.content}${f.supersededMemoryId ? " [linked]" : " [unlinked]"}`),
  };
}

async function main() {
  console.log(`\nExtraction quality — EXTRACTION_MODEL=${EXTRACTION_MODEL}, ${ATTEMPTS} attempts per scenario\n`);
  const summary: { name: string; passes: number }[] = [];

  try {
    for (const s of SCENARIOS) {
      console.log(`${s.name}`);
      const results: Attempt[] = [];
      for (let i = 1; i <= ATTEMPTS; i++) {
        const a = await attempt(s, i);
        results.push(a);
        console.log(`  attempt ${i}: ${a.passed ? "pass" : "fail"}${a.passed ? "" : `  (A linked=${a.aLinked}, B linked=${a.bLinked}, stale live=${a.staleLive})`}`);
        if (!a.passed) for (const line of a.stored) console.log(`      ${line}`);
      }
      const passes = results.filter((r) => r.passed).length;
      summary.push({ name: s.name, passes });
      console.log(`  --> ${passes}/${ATTEMPTS} (${Math.round((passes / ATTEMPTS) * 100)}%)\n`);
    }
  } finally {
    try {
      await cleanup();
      console.log(`Cleaned up ${usedUserIds.size} test scope(s) under "${RUN_PREFIX}".`);
    } catch (err) {
      console.error(`WARNING: cleanup failed for prefix "${RUN_PREFIX}":`, err);
    }
  }

  console.log("\n=== Extraction Quality Rates ===\n");
  for (const s of summary) {
    console.log(`  ${s.passes}/${ATTEMPTS}  ${s.name}`);
  }
  console.log("\nThese are rates, not assertions — this script always exits 0.");
  console.log("A drop across several runs means extraction quality regressed.\n");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
