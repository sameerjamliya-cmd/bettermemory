"use server";

// The dashboard talks to the memory library through these server actions rather
// than fetching /api/memory/* from the browser.
//
// The alternative — having the client fetch the API — would mean shipping
// API_KEY to the browser, where it is readable by anyone who opens devtools, or
// proxying through a route that re-authenticates the server to itself over HTTP
// for no security gain. These actions run in the same process as the library, so
// the key never leaves the server and there is no extra hop.
//
// Consequence worth knowing: the dashboard therefore does not exercise the API's
// bearer-token check. That path is covered separately by the curl tests.
import { getAll, search } from "../../lib/memory";
import type { MemoryRecord, Scope, SearchResult } from "../../lib/types";

export interface ScopeInput {
  userId: string;
  agentId?: string;
  runId?: string;
}

function toScope(input: ScopeInput): Scope {
  const scope: Scope = { userId: input.userId };
  if (input.agentId) scope.agentId = input.agentId;
  if (input.runId) scope.runId = input.runId;
  return scope;
}

export async function loadMemoriesAction(
  input: ScopeInput,
  includeSuperseded: boolean
): Promise<{ ok: true; memories: MemoryRecord[] } | { ok: false; error: string }> {
  try {
    return { ok: true, memories: await getAll(toScope(input), { includeSuperseded }) };
  } catch (err) {
    return { ok: false, error: err instanceof Error ? err.message : String(err) };
  }
}

export async function searchAction(
  input: ScopeInput,
  query: string
): Promise<{ ok: true; results: SearchResult[] } | { ok: false; error: string }> {
  try {
    return { ok: true, results: await search(query, toScope(input)) };
  } catch (err) {
    return { ok: false, error: err instanceof Error ? err.message : String(err) };
  }
}
