// BM25 scoring and the cosine+BM25 fusion. Pure functions — no I/O, no provider
// access — so the ranking maths can be reasoned about and tested on its own.

const BM25_MIDPOINT = 5;
const BM25_STEEPNESS = 0.5;
const BM25_K1 = 1.5;
const BM25_B = 0.75;

export const HYBRID_CANDIDATE_POOL = 20;

export function normalizeBM25(rawScore: number, midpoint: number, steepness: number): number {
  return 1 / (1 + Math.exp(-steepness * (rawScore - midpoint)));
}

export function tokenize(text: string): string[] {
  return text.toLowerCase().match(/[a-z0-9]+/g) ?? [];
}

export function computeBM25Scores(
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

export interface FusedScore {
  bm25RawScore: number | null;
  bm25Normalized: number | null;
  combinedScore: number;
}

// The combined score is the mean of however many signals were available, so a
// fact with no keyword overlap is not penalised against one that has it.
export function fuseScores(cosineScore: number, bm25RawScore: number | null): FusedScore {
  const bm25Normalized =
    bm25RawScore !== null ? normalizeBM25(bm25RawScore, BM25_MIDPOINT, BM25_STEEPNESS) : null;

  let total = cosineScore;
  let signals = 1;
  if (bm25RawScore !== null) {
    total += bm25Normalized as number;
    signals += 1;
  }
  return { bm25RawScore, bm25Normalized, combinedScore: total / signals };
}
