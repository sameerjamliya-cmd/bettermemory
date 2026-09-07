// Append-only event log for memory lifecycle.
//
// Why SQLite (via node:sqlite) rather than a second Qdrant collection or a JSONL
// file:
//   - Qdrant requires a vector per point. Events have nothing to embed, so a
//     second collection would mean storing dummy vectors and paying a network
//     round-trip per write, to get worse query ergonomics than a local index.
//   - The existing logs/*.jsonl logger is append-only but unindexed: answering
//     "every event for this memoryId" would mean scanning the whole file, and it
//     has no notion of scope or types.
//   - node:sqlite ships with Node 22+, so this adds a real indexed store with NO
//     new dependency and no native build step. It also matches what mem0 itself
//     uses for its history table.
//
// The one cost is that node:sqlite is still flagged experimental, which is why
// the module is isolated behind this file — swapping it for better-sqlite3 later
// would touch nothing outside these functions.
//
// APPEND-ONLY IS THE POINT: this file contains INSERT and SELECT statements and
// nothing else. There is deliberately no update or delete path, so removing a
// memory from the live store can never remove its history.
import { DatabaseSync } from "node:sqlite";
import { randomUUID } from "crypto";
import path from "path";
import type { Scope } from "./types";
import type { ValidatedScope } from "./scope";

export type MemoryEventType = "ADD" | "SUPERSEDE" | "DELETE";

export interface MemoryEvent {
  id: string;
  memoryId: string;
  event: MemoryEventType;
  oldContent: string | null;
  newContent: string | null;
  /** Set on SUPERSEDE: the memory this one replaced. Not in mem0's schema — it
   *  is what lets history() walk a chain whose earlier links have been deleted,
   *  since content alone cannot identify a predecessor. */
  supersedesMemoryId: string | null;
  createdAt: string;
  scope: Scope;
}

const DB_PATH = path.join(process.cwd(), "logs", "memory-events.db");

let db: DatabaseSync | null = null;

function connect(): DatabaseSync {
  if (db) return db;
  db = new DatabaseSync(DB_PATH);
  db.exec(`
    CREATE TABLE IF NOT EXISTS memory_events (
      id                 TEXT PRIMARY KEY,
      memoryId           TEXT NOT NULL,
      event              TEXT NOT NULL,
      oldContent         TEXT,
      newContent         TEXT,
      supersedesMemoryId TEXT,
      createdAt          TEXT NOT NULL,
      userId             TEXT NOT NULL,
      agentId            TEXT,
      runId              TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_events_memory ON memory_events (memoryId);
    CREATE INDEX IF NOT EXISTS idx_events_supersedes ON memory_events (supersedesMemoryId);
    CREATE INDEX IF NOT EXISTS idx_events_user ON memory_events (userId);
  `);
  return db;
}

export interface AppendEventInput {
  memoryId: string;
  event: MemoryEventType;
  oldContent?: string | null;
  newContent?: string | null;
  supersedesMemoryId?: string | null;
  scope: ValidatedScope;
}

/** The only write path. INSERT only — never UPDATE, never DELETE. */
export function appendEvent(input: AppendEventInput): MemoryEvent {
  const row = {
    id: randomUUID(),
    memoryId: input.memoryId,
    event: input.event,
    oldContent: input.oldContent ?? null,
    newContent: input.newContent ?? null,
    supersedesMemoryId: input.supersedesMemoryId ?? null,
    createdAt: new Date().toISOString(),
    userId: input.scope.userId,
    agentId: input.scope.agentId,
    runId: input.scope.runId,
  };
  connect()
    .prepare(
      `INSERT INTO memory_events
         (id, memoryId, event, oldContent, newContent, supersedesMemoryId, createdAt, userId, agentId, runId)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
    )
    .run(
      row.id, row.memoryId, row.event, row.oldContent, row.newContent,
      row.supersedesMemoryId, row.createdAt, row.userId, row.agentId, row.runId
    );
  return toEvent(row);
}

function toEvent(r: Record<string, unknown>): MemoryEvent {
  const scope: Scope = { userId: String(r.userId) };
  if (typeof r.agentId === "string") scope.agentId = r.agentId;
  if (typeof r.runId === "string") scope.runId = r.runId;
  return {
    id: String(r.id),
    memoryId: String(r.memoryId),
    event: String(r.event) as MemoryEventType,
    oldContent: r.oldContent === null ? null : String(r.oldContent),
    newContent: r.newContent === null ? null : String(r.newContent),
    supersedesMemoryId: r.supersedesMemoryId === null ? null : String(r.supersedesMemoryId),
    createdAt: String(r.createdAt),
    scope,
  };
}

// Scope matching mirrors pointMatchesScope(): userId exact, an unspecified
// agentId/runId acting as a wildcard, so a caller sees exactly the history of
// what it could have read.
function scopeClause(scope: ValidatedScope): { sql: string; params: (string | null)[] } {
  const parts = ["userId = ?"];
  const params: (string | null)[] = [scope.userId];
  if (scope.agentId !== null) { parts.push("agentId = ?"); params.push(scope.agentId); }
  if (scope.runId !== null) { parts.push("runId = ?"); params.push(scope.runId); }
  return { sql: parts.join(" AND "), params };
}

/** Every event for these memory ids, in scope, oldest first. */
export function eventsForMemories(memoryIds: string[], scope: ValidatedScope): MemoryEvent[] {
  if (memoryIds.length === 0) return [];
  const { sql, params } = scopeClause(scope);
  const placeholders = memoryIds.map(() => "?").join(", ");
  const rows = connect()
    .prepare(
      `SELECT * FROM memory_events
        WHERE ${sql} AND memoryId IN (${placeholders})
        ORDER BY createdAt ASC, rowid ASC`
    )
    .all(...params, ...memoryIds) as Record<string, unknown>[];
  return rows.map(toEvent);
}

/** The memory this one superseded, if any — read from the log, not live data. */
export function predecessorOf(memoryId: string, scope: ValidatedScope): string | null {
  const { sql, params } = scopeClause(scope);
  const row = connect()
    .prepare(
      `SELECT supersedesMemoryId FROM memory_events
        WHERE ${sql} AND memoryId = ? AND supersedesMemoryId IS NOT NULL
        ORDER BY createdAt ASC LIMIT 1`
    )
    .get(...params, memoryId) as Record<string, unknown> | undefined;
  return row?.supersedesMemoryId ? String(row.supersedesMemoryId) : null;
}

/** Memories that superseded this one. Several are possible (branching). */
export function successorsOf(memoryId: string, scope: ValidatedScope): string[] {
  const { sql, params } = scopeClause(scope);
  const rows = connect()
    .prepare(
      `SELECT DISTINCT memoryId FROM memory_events
        WHERE ${sql} AND supersedesMemoryId = ?`
    )
    .all(...params, memoryId) as Record<string, unknown>[];
  return rows.map((r) => String(r.memoryId));
}

/** Whether this scope has any history for this id at all. */
export function hasHistory(memoryId: string, scope: ValidatedScope): boolean {
  const { sql, params } = scopeClause(scope);
  const row = connect()
    .prepare(`SELECT 1 AS present FROM memory_events WHERE ${sql} AND memoryId = ? LIMIT 1`)
    .get(...params, memoryId);
  return row !== undefined;
}

/** Test helper: closes the handle so a temp database file can be removed. */
export function closeEventLog(): void {
  db?.close();
  db = null;
}
