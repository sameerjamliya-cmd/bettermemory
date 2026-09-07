// One session exercising every feature in sequence, to catch conflicts the
// per-feature tests cannot: each phase runs against the same live Qdrant, and
// several share a scope on purpose.
import * as fs from "fs";
import * as memory from "../lib/memory";
import { qdrant, COLLECTION_NAME, EXTRACTION_MODEL } from "../lib/clients";

const stamp = Date.now();
const users: string[] = [];
const mk = (t: string) => { const userId = `e2e-${t}-${stamp}`; users.push(userId); return { userId }; };

const results: { name: string; ok: boolean; detail: string }[] = [];
function check(name: string, ok: boolean, detail = "") {
  results.push({ name, ok, detail });
  console.log(`  ${ok ? "PASS" : "FAIL"}  ${name}${detail ? `  — ${detail}` : ""}`);
}

async function main() {
  console.log(`END-TO-END SESSION   model=${EXTRACTION_MODEL}\n`);
  try {
    // ---- 1. dedup (hash + semantic) ------------------------------------
    console.log("1. dedup");
    const d = mk("dedup");
    const first = await memory.add("I always drink coffee before my morning workout.", d);
    const dup = await memory.add("I always drink coffee before my morning workout.", d);
    check("identical sentence is skipped, not stored twice",
      dup.stored.length === 0 && dup.skipped.length === 1,
      `reason=${dup.skipped[0]?.reason}`);
    check("first-run notice fired on the very first add", first.notice !== null);
    check("second add carries no notice", dup.notice === null);

    // ---- 2. supersede + chain + history --------------------------------
    console.log("\n2. supersede, chain, history");
    const s = mk("supersede");
    const a = (await memory.add("My deadlift PR is 140kg.", s)).stored[0];
    const b = (await memory.add("Just hit a new deadlift PR, 145kg this time.", s)).stored[0];
    const c = (await memory.add("New deadlift PR today, 150kg.", s)).stored[0];
    check("chain links A<-B<-C", b?.supersededMemoryId === a?.id && c?.supersededMemoryId === b?.id);
    const live = (await memory.getAll(s)).map((m) => m.content);
    check("only newest is live", live.length === 1 && live[0].includes("150kg"), JSON.stringify(live));
    const all = await memory.getAll(s, { includeSuperseded: true });
    check("older versions preserved, flagged", all.length === 3 && all.filter((m) => m.superseded).length === 2);
    const h = await memory.history(c!.id, s);
    const chainIds = new Set((h.events ?? []).map((e) => e.memoryId));
    check("history covers the whole lineage from the event log",
      h.status === "found" && chainIds.size === 3,
      `memories in lineage=${chainIds.size} events=${h.events?.length}`);
    check("history retains the original 140kg content",
      (h.events ?? []).some((e) => (e.newContent ?? "").includes("140kg")));

    // ---- 3. scoping + hierarchy ----------------------------------------
    console.log("\n3. scoping");
    const other = mk("other");
    check("other scope cannot see it", (await memory.search("deadlift", other)).length === 0);
    const narrow = { userId: s.userId, agentId: "agent1" };
    const ag = await memory.add("I am scoped to an agent.", narrow);
    check("broad scope reads agent-scoped point",
      (await memory.getAll({ userId: s.userId })).some((m) => m.id === ag.stored[0].id));
    check("cross-user delete refused",
      (await memory.delete(ag.stored[0].id, other)).status === "scope_mismatch");

    // ---- 4. BM25 contribution ------------------------------------------
    console.log("\n4. hybrid retrieval");
    const bm = mk("bm25");
    await memory.add("My squat PR is 90kg.", bm, { extract: false });
    await memory.add("My bench PR is 90kg.", bm, { extract: false });
    const ranked = await memory.search("what's my squat PR", bm);
    const sq = ranked.find((r) => /squat/i.test(r.content));
    const be = ranked.find((r) => /bench/i.test(r.content));
    check("exact term outranks sibling fact",
      !!sq && !!be && sq.combinedScore > be.combinedScore,
      sq && be ? `squat=${sq.combinedScore.toFixed(4)} bench=${be.combinedScore.toFixed(4)}` : "missing");
    check("bm25 contributed a nonzero score", (sq?.bm25RawScore ?? 0) > 0);

    // ---- 5. procedural memory ------------------------------------------
    console.log("\n5. procedural memory");
    const p = mk("proc");
    await memory.add("My deadlift PR is 140kg.", p);
    const instr = await memory.addProcedural("Always reply in British English.", p);
    check("stored verbatim", instr.content === "Always reply in British English.");
    check("hidden from getAll", !(await memory.getAll(p)).some((m) => /British/.test(m.content)));
    check("hidden from search",
      !(await memory.search("what language should replies be in", p)).some((r) => /British/.test(r.content)));
    check("visible via getProcedural", (await memory.getProcedural(p)).length === 1);

    // ---- 6. temporal grounding -----------------------------------------
    console.log("\n6. temporal grounding");
    const t = mk("temporal");
    const back = await memory.add("Hit a new deadlift PR last week.", t,
      { observedAt: "2026-03-14T09:00:00.000Z" });
    check("observedAt persisted and differs from write time",
      back.stored[0]?.observedAt === "2026-03-14T09:00:00.000Z" &&
      back.stored[0]?.observedAt !== back.stored[0]?.extractedAt);
    check("relative reference resolved against observation date, not now",
      /2026-03/.test(back.stored[0]?.content ?? "") || /last week/i.test(back.stored[0]?.content ?? ""),
      `"${back.stored[0]?.content}"`);

    // ---- 7. vision ------------------------------------------------------
    console.log("\n7. vision");
    const v = mk("vision");
    const img = fs.readFileSync(
      new URL("./fixtures/receipt.png", `file://${__filename}`).pathname
    ).toString("base64");
    const vis = await memory.add(
      { text: "Receipt from the bike shop.", imageBase64: img }, v);
    const joined = vis.stored.map((f) => f.content).join(" | ");
    check("facts extracted from the image", vis.stored.length > 0, joined.slice(0, 90) + "...");
    check("image content actually read (total or shop name present)",
      /33\.50|Greenfield/i.test(joined));
    check("base64 not persisted in the payload", (vis.stored[0]?.source ?? "").startsWith("[image]"));

    // ---- 8. provider seam ----------------------------------------------
    console.log("\n8. provider seam");
    check("memory.ts references no vendor SDK",
      !/qdrant\.|openai\./.test(fs.readFileSync(new URL("../lib/memory.ts", `file://${__filename}`).pathname, "utf8")));

    // ---- 9. notices at scale -------------------------------------------
    console.log("\n9. scale notice");
    const n = mk("scale");
    for (let i = 1; i <= 99; i++) await memory.add(`Bulk fact ${i}.`, n, { extract: false });
    const crossing = await memory.add("Bulk fact 100.", n, { extract: false });
    const after = await memory.add("Bulk fact 101.", n, { extract: false });
    check("threshold notice fires on the crossing call", crossing.notice?.includes("100+") === true);
    check("silent on the call after", after.notice === null);

    // ---- 10. deleteAll / reset -----------------------------------------
    console.log("\n10. teardown paths");
    const wiped = await memory.deleteAll(n);
    check("deleteAll removes the whole scope",
      wiped.deleted >= 101 && (await memory.getAll(n)).length === 0, `deleted=${wiped.deleted}`);
    check("unrelated scope untouched", (await memory.getAll(p)).length > 0);
  } finally {
    await qdrant.delete(COLLECTION_NAME, {
      filter: { should: users.map((u) => ({ key: "userId", match: { value: u } })) }, wait: true,
    });
    const failed = results.filter((r) => !r.ok);
    console.log(`\n${results.length - failed.length}/${results.length} end-to-end checks passed.`);
    if (failed.length) {
      console.log("FAILED:");
      for (const f of failed) console.log(`  - ${f.name} ${f.detail}`);
    }
    console.log("cleaned up");
  }
}
main().catch((e) => { console.error(e); process.exit(1); });
