// A vector store that is a plain array, and an LLM that never calls a network.
// If add() -> search() works against these with no change to orchestration code,
// the seam is genuine. Imports nothing from Qdrant or OpenAI.
import {
  add, search, getAll, get, history, deleteMemory, deleteAll,
  addProcedural, getProcedural, setProviders,
} from "../lib/memory";
import type {
  ExtractedFact, ExtractOptions, LLMClient, MemoryFilter, MemoryPoint,
  ScoredPoint, ScrollOptions, VectorStoreClient,
} from "../lib/providers/types";

class InMemoryVectorStore implements VectorStoreClient {
  points = new Map<string, MemoryPoint>();
  calls: string[] = [];

  async ensureReady() { this.calls.push("ensureReady"); }

  async upsert(points: MemoryPoint[]) {
    this.calls.push(`upsert(${points.length})`);
    for (const p of points) this.points.set(p.id, { ...p, payload: { ...p.payload } });
  }

  private matches(p: MemoryPoint, f: MemoryFilter): boolean {
    const ok = (c: any) => {
      const v = p.payload[c.key];
      return "anyOf" in c ? typeof v === "string" && c.anyOf.includes(v) : v === c.equals;
    };
    if ((f.must ?? []).some((c) => !ok(c))) return false;
    if ((f.mustNot ?? []).some((c) => ok(c))) return false;
    return true;
  }

  // Cosine similarity, so ranking is meaningful rather than arbitrary.
  private cosine(a: number[], b: number[]) {
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
    return na && nb ? dot / (Math.sqrt(na) * Math.sqrt(nb)) : 0;
  }

  async query(filter: MemoryFilter, vector: number[], topK: number): Promise<ScoredPoint[]> {
    this.calls.push(`query(topK=${topK})`);
    return [...this.points.values()]
      .filter((p) => this.matches(p, filter))
      .map((p) => ({ ...p, score: this.cosine(vector, p.vector ?? []) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);
  }

  async *scroll(filter: MemoryFilter, pageSize: number, _o?: ScrollOptions) {
    this.calls.push(`scroll(pageSize=${pageSize})`);
    const all = [...this.points.values()].filter((p) => this.matches(p, filter));
    for (let i = 0; i < all.length; i += pageSize) yield all.slice(i, i + pageSize);
  }

  async retrieve(ids: string[]) {
    this.calls.push(`retrieve(${ids.length})`);
    return ids.map((id) => this.points.get(id)).filter(Boolean) as MemoryPoint[];
  }

  async count(filter: MemoryFilter) {
    return [...this.points.values()].filter((p) => this.matches(p, filter)).length;
  }

  async setPayload(ids: string[], payload: Record<string, unknown>) {
    for (const id of ids) {
      const p = this.points.get(id);
      if (p) p.payload = { ...p.payload, ...payload };
    }
  }

  async deleteByIds(ids: string[]) { for (const id of ids) this.points.delete(id); }

  async deleteByFilter(filter: MemoryFilter) {
    for (const [id, p] of [...this.points]) if (this.matches(p, filter)) this.points.delete(id);
  }
}

// Deterministic bag-of-words embedding — no network, no API key.
class FakeLLMClient implements LLMClient {
  extractCalls: { prompt: string; hasImage: boolean }[] = [];

  async embed(text: string): Promise<number[]> {
    const v = new Array(64).fill(0);
    for (const w of text.toLowerCase().match(/[a-z0-9]+/g) ?? []) {
      let h = 0;
      for (const ch of w) h = (h * 31 + ch.charCodeAt(0)) % 64;
      v[h] += 1;
    }
    return v;
  }

  async extractFacts(prompt: string, options?: ExtractOptions): Promise<ExtractedFact[]> {
    this.extractCalls.push({ prompt, hasImage: !!options?.imageDataUrl });
    // Echo back the New Message as one fact, with no supersede/skip logic.
    const m = prompt.match(/New Message:\n(.*)\n/);
    const content = (m?.[1] ?? "unknown").trim();
    return [{ content, supersedes: null, skip: false, skipReason: null }];
  }
}

const store = new InMemoryVectorStore();
const llm = new FakeLLMClient();
setProviders({ vectorStore: store, llm });

const scope = { userId: "mock-user" };

async function main() {
  console.log("=== add() -> search() round-trip against in-memory providers ===\n");

  const a = await add("My squat PR is 140kg.", scope);
  const b = await add("I have a pet iguana named Steve.", scope);
  console.log(`  add #1 stored: ${JSON.stringify(a.stored.map((f) => f.content))}`);
  console.log(`  add #2 stored: ${JSON.stringify(b.stored.map((f) => f.content))}`);
  console.log(`  points held in the fake store: ${store.points.size}`);

  const results = await search("squat PR", scope);
  console.log(`\n  search("squat PR") ->`);
  for (const r of results) console.log(`    "${r.content}"  cosine=${r.cosineScore.toFixed(4)} combined=${r.combinedScore.toFixed(4)}`);

  console.log(`\n  getAll -> ${JSON.stringify((await getAll(scope)).map((m) => m.content))}`);

  const one = await get(a.stored[0].id, scope);
  console.log(`  get(id) -> ${one.status}: "${one.status === "found" ? one.memory.content : ""}"`);

  const h = await history(a.stored[0].id, scope);
  console.log(`  history(id) -> status=${h.status} events=${h.events?.length ?? 0}`);

  await addProcedural("Always reply in British English.", scope);
  console.log(`  getProcedural -> ${JSON.stringify((await getProcedural(scope)).map((m) => m.content))}`);
  console.log(`  getAll still excludes procedural -> ${JSON.stringify((await getAll(scope)).map((m) => m.content))}`);

  const del = await deleteMemory(b.stored[0].id, scope);
  console.log(`  deleteMemory -> ${del.status}; remaining facts: ${(await getAll(scope)).length}`);

  const cross = await deleteMemory(a.stored[0].id, { userId: "someone-else" });
  console.log(`  cross-scope deleteMemory -> ${cross.status}`);

  const wiped = await deleteAll(scope);
  console.log(`  deleteAll -> ${JSON.stringify(wiped)}; store now holds ${store.points.size} points`);

  console.log(`\n  fake LLM extract calls: ${llm.extractCalls.length} (image used: ${llm.extractCalls.some((c) => c.hasImage)})`);
  console.log(`  distinct store methods exercised: ${[...new Set(store.calls.map((c) => c.split("(")[0]))].join(", ")}`);
  console.log("\n  no Qdrant, no OpenAI, no network, no API key.");
}
main().catch((e) => { console.error(e); process.exit(1); });
