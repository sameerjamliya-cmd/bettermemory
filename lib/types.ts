// Shared shapes used across the memory modules. Types only — no logic, so any
// module can import from here without pulling in behaviour.

export interface Scope {
  userId: string;
  agentId?: string;
  runId?: string;
}

export interface AddInput {
  text?: string;
  // Raw base64, or a full data: URL. Never stored — only sent to the extraction
  // call — because a base64 image in a Qdrant payload would dwarf the fact.
  imageBase64?: string;
}

export interface RelatedMemory {
  content: string;
  source: string;
  score: number;
  extractedAt: string;
}

export interface MemoryRecord {
  id: string;
  content: string;
  source: string;
  extractedAt: string;
  supersededMemoryId: string | null;
  superseded: boolean;
  /** This memory supersedes something chronologically newer than itself. */
  outOfOrder: boolean;
}

export interface SearchResult {
  id: string;
  content: string;
  source: string;
  extractedAt: string;
  // What this fact itself superseded, and whether a newer fact has superseded it.
  supersededMemoryId: string | null;
  superseded: boolean;
  /** This memory supersedes something chronologically newer than itself. */
  outOfOrder: boolean;
  cosineScore: number;
  bm25RawScore: number | null;
  bm25Normalized: number | null;
  combinedScore: number;
}

export interface StoredFact {
  id: string;
  content: string;
  source: string;
  extractedAt: string;
  // When the message was observed. Defaults to write time, so it equals
  // extractedAt for live use and differs only when a caller backfills.
  observedAt: string;
  // The id of the memory this fact supersedes. The superseded point is left in
  // the collection; search() hides it by default instead of deleting it.
  supersededMemoryId: string | null;
  /** What this fact is about, per extraction. Gates supersession: a candidate
   *  link between two different attributeKeys is discarded. */
  attributeKey: string | null;
  // True when the supersede link points at a CHRONOLOGICALLY NEWER memory — a
  // backfill that arrived late. The link is kept for lineage, but an
  // out-of-order fact never hides the fact it claims to replace.
  outOfOrder: boolean;
}

export interface SkippedFact {
  content: string;
  reason: "semantic_dedup" | "hash_dedup";
  matchedExisting: { id: string; content: string };
}

export interface AddResult {
  relatedMemories: (RelatedMemory & { label: string })[];
  stored: StoredFact[];
  skipped: SkippedFact[];
  // Informational only, and null on the overwhelming majority of calls. Computed
  // locally from the scope's own count — nothing is recorded or sent anywhere.
  notice: string | null;
}

export interface ProceduralMemory {
  id: string;
  content: string;
  createdAt: string;
  type: "procedural";
}

export interface StoredMemory {
  id: string;
  content: string;
  source: string;
  extractedAt: string;
  observedAt: string | null;
  supersededMemoryId: string | null;
  superseded: boolean;
  type: MemoryType;
}

export interface HistoryResult {
  status: "found" | "not_found";
  id: string;
  /** Every event for every memory in this lineage, oldest first. Present when
   *  status is "found". Read from the append-only log, so entries survive the
   *  deletion of the memories they describe. */
  events?: import("./events").MemoryEvent[];
}

export interface DeleteAllResult {
  deleted: number;
  includedProcedural: boolean;
}

export type MemoryType = "fact" | "procedural";

export type GetResult =
  | { status: "found"; memory: StoredMemory }
  | { status: "not_found"; id: string }
  | { status: "scope_mismatch"; id: string };

export type DeleteResult =
  | { status: "deleted"; id: string; content: string; clearedSupersedeLinks: string[] }
  | { status: "not_found"; id: string }
  | { status: "scope_mismatch"; id: string };
