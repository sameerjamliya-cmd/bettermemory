// The seam between orchestration logic (lib/memory.ts) and the SDKs that talk to
// a vector database and a model provider. memory.ts references only these types;
// anything vendor-specific lives behind an implementation of one of them.

/** A single field condition. Deliberately smaller than any one vendor's filter
 *  DSL — it covers exactly what the orchestration layer actually needs. */
export type FieldCondition =
  | { key: string; equals: string | boolean }
  | { key: string; anyOf: string[] };

export interface MemoryFilter {
  must?: FieldCondition[];
  mustNot?: FieldCondition[];
}

export interface MemoryPoint {
  id: string;
  /** Omitted unless the caller asked for it — vectors are large. */
  vector?: number[];
  payload: Record<string, unknown>;
}

export interface ScoredPoint extends MemoryPoint {
  /** Similarity of this point to the query vector. Higher is closer. */
  score: number;
}

export interface ScrollOptions {
  withVector?: boolean;
  /** Restrict the payload to these fields, when the caller needs only a few. */
  payloadFields?: string[];
}

export interface VectorStoreClient {
  /** Create the collection if it does not exist. Safe to call repeatedly. */
  ensureReady(): Promise<void>;
  upsert(points: MemoryPoint[]): Promise<void>;
  query(filter: MemoryFilter, vector: number[], topK: number): Promise<ScoredPoint[]>;
  /** Pages through every match. The implementation owns cursor handling. */
  scroll(filter: MemoryFilter, pageSize: number, options?: ScrollOptions): AsyncIterable<MemoryPoint[]>;
  retrieve(ids: string[], options?: { withVector?: boolean }): Promise<MemoryPoint[]>;
  count(filter: MemoryFilter): Promise<number>;
  /** Merge these fields into the payload of each id, leaving other fields. */
  setPayload(ids: string[], payload: Record<string, unknown>): Promise<void>;
  deleteByIds(ids: string[]): Promise<void>;
  deleteByFilter(filter: MemoryFilter): Promise<void>;
}

export interface ExtractedFact {
  content: string;
  supersedes: string | null;
  skip: boolean;
  skipReason: string | null;
}

export interface ExtractOptions {
  /** A `data:` URL. When present the call is multimodal: the image and the
   *  prompt are one message, not two calls. */
  imageDataUrl?: string;
}

export interface LLMClient {
  embed(text: string): Promise<number[]>;
  /** Runs the extraction prompt and returns the facts, already normalised. */
  extractFacts(prompt: string, options?: ExtractOptions): Promise<ExtractedFact[]>;
}
