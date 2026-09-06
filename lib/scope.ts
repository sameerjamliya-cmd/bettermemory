// Scope validation and the filters derived from it. Every public entry point
// runs validateScope() before touching the store, and the internal helpers take
// ValidatedScope rather than Scope so unvalidated input cannot reach a query.
import type { FieldCondition, MemoryFilter } from "./providers/types";
import type { Scope } from "./types";

export interface ValidatedScope {
  userId: string;
  agentId: string | null;
  runId: string | null;
}

export function validateId(field: string, value: string | undefined, required: boolean): string | null {
  if (value === undefined || value === null) {
    if (required) throw new Error(`Invalid scope: "${field}" is required.`);
    return null;
  }
  const trimmed = value.trim();
  if (trimmed === "") {
    throw new Error(`Invalid scope: "${field}" must not be empty.`);
  }
  if (/\s/.test(trimmed)) {
    throw new Error(`Invalid scope: "${field}" must not contain whitespace.`);
  }
  return trimmed;
}

export function validateScope(scope: Scope): ValidatedScope {
  const userId = validateId("userId", scope.userId, true) as string;
  const agentId = validateId("agentId", scope.agentId, false);
  const runId = validateId("runId", scope.runId, false);
  return { userId, agentId, runId };
}

export const EXCLUDE_PROCEDURAL: FieldCondition = { key: "type", equals: "procedural" };

export function scopeFilter(scope: ValidatedScope): MemoryFilter {
  const must: FieldCondition[] = [{ key: "userId", equals: scope.userId }];
  if (scope.agentId !== null) must.push({ key: "agentId", equals: scope.agentId });
  if (scope.runId !== null) must.push({ key: "runId", equals: scope.runId });
  return { must };
}

// Hierarchical, matching read semantics: userId must match exactly, but an
// unspecified agentId/runId in the caller's scope is a wildcard, exactly as
// scopeFilter() treats it for search()/getAll(). A caller can therefore act on
// anything it can see — and nothing it cannot. Cross-userId remains strict.
export function pointMatchesScope(
  payload: Record<string, unknown> | null | undefined,
  scope: ValidatedScope
): boolean {
  const p = payload ?? {};
  const userId = typeof p.userId === "string" ? p.userId : null;
  const agentId = typeof p.agentId === "string" ? p.agentId : null;
  const runId = typeof p.runId === "string" ? p.runId : null;
  return (
    userId === scope.userId &&
    (scope.agentId === null || agentId === scope.agentId) &&
    (scope.runId === null || runId === scope.runId)
  );
}
