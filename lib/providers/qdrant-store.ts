// The only place in the codebase that knows Qdrant's request shapes.
import { qdrant, COLLECTION_NAME, EMBEDDING_DIMENSIONS } from "../clients";
import type {
  FieldCondition,
  MemoryFilter,
  MemoryPoint,
  ScoredPoint,
  ScrollOptions,
  VectorStoreClient,
} from "./types";

type QdrantCondition = { key: string; match: { value: string } | { any: string[] } };

function toQdrantCondition(c: FieldCondition): QdrantCondition {
  return "anyOf" in c
    ? { key: c.key, match: { any: c.anyOf } }
    : { key: c.key, match: { value: c.equals } };
}

function toQdrantFilter(filter: MemoryFilter) {
  const out: { must?: QdrantCondition[]; must_not?: QdrantCondition[] } = {};
  if (filter.must?.length) out.must = filter.must.map(toQdrantCondition);
  if (filter.mustNot?.length) out.must_not = filter.mustNot.map(toQdrantCondition);
  return out;
}

function toMemoryPoint(point: {
  id: string | number;
  vector?: unknown;
  payload?: Record<string, unknown> | null;
}): MemoryPoint {
  const p: MemoryPoint = { id: String(point.id), payload: point.payload ?? {} };
  if (Array.isArray(point.vector)) p.vector = point.vector as number[];
  return p;
}

export class QdrantVectorStore implements VectorStoreClient {
  private ready = false;

  async ensureReady(): Promise<void> {
    if (this.ready) return;
    const { exists } = await qdrant.collectionExists(COLLECTION_NAME);
    if (!exists) {
      await qdrant.createCollection(COLLECTION_NAME, {
        vectors: { size: EMBEDDING_DIMENSIONS, distance: "Cosine" },
      });
    }
    this.ready = true;
  }

  async upsert(points: MemoryPoint[]): Promise<void> {
    if (points.length === 0) return;
    await qdrant.upsert(COLLECTION_NAME, {
      wait: true,
      points: points.map((p) => ({ id: p.id, vector: p.vector as number[], payload: p.payload })),
    });
  }

  async query(filter: MemoryFilter, vector: number[], topK: number): Promise<ScoredPoint[]> {
    const res = await qdrant.query(COLLECTION_NAME, {
      query: vector,
      filter: toQdrantFilter(filter),
      limit: topK,
      with_payload: true,
    });
    return res.points.map((p) => ({ ...toMemoryPoint(p), score: p.score }));
  }

  async *scroll(
    filter: MemoryFilter,
    pageSize: number,
    options?: ScrollOptions
  ): AsyncIterable<MemoryPoint[]> {
    // Cursor handling lives here so callers never see a vendor's offset token.
    let offset: string | number | undefined | null = undefined;
    do {
      const page = await qdrant.scroll(COLLECTION_NAME, {
        filter: toQdrantFilter(filter),
        limit: pageSize,
        offset: offset ?? undefined,
        with_payload: options?.payloadFields ?? true,
        with_vector: options?.withVector ?? false,
      });
      yield page.points.map(toMemoryPoint);
      offset = page.next_page_offset as typeof offset;
    } while (offset !== null && offset !== undefined);
  }

  async retrieve(ids: string[], options?: { withVector?: boolean }): Promise<MemoryPoint[]> {
    if (ids.length === 0) return [];
    const points = await qdrant.retrieve(COLLECTION_NAME, {
      ids,
      with_payload: true,
      with_vector: options?.withVector ?? false,
    });
    return points.map(toMemoryPoint);
  }

  async count(filter: MemoryFilter): Promise<number> {
    const { count } = await qdrant.count(COLLECTION_NAME, {
      filter: toQdrantFilter(filter),
      exact: true,
    });
    return count;
  }

  async setPayload(ids: string[], payload: Record<string, unknown>): Promise<void> {
    if (ids.length === 0) return;
    await qdrant.setPayload(COLLECTION_NAME, { payload, points: ids, wait: true });
  }

  async deleteByIds(ids: string[]): Promise<void> {
    if (ids.length === 0) return;
    await qdrant.delete(COLLECTION_NAME, { points: ids, wait: true });
  }

  async deleteByFilter(filter: MemoryFilter): Promise<void> {
    await qdrant.delete(COLLECTION_NAME, { filter: toQdrantFilter(filter), wait: true });
  }
}
