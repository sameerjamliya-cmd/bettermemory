// Requires Authorization: Bearer <API_KEY>. The key gates who may call the API
// at all; it does NOT add per-user isolation — a caller holding the key may
// address any scope, so `userId` remains an identifier, not a credential.
// See "API and dashboard" in the README.

import { NextResponse } from "next/server";
import { requireApiKey } from "../../../../lib/api-auth";
import { getAll } from "../../../../lib/memory";

export async function GET(request: Request) {
  // Before any parsing, scope validation or memory logic.
  const unauthorized = requireApiKey(request);
  if (unauthorized) return unauthorized;

  const params = new URL(request.url).searchParams;
  const userId = params.get("userId");
  if (userId === null) {
    return NextResponse.json({ error: 'Missing "userId" query parameter.' }, { status: 400 });
  }

  // Only pass the optional ids when present: an explicit undefined and an absent
  // key mean the same thing to validateScope, but omitting keeps intent clear.
  const scope: { userId: string; agentId?: string; runId?: string } = { userId };
  const agentId = params.get("agentId");
  const runId = params.get("runId");
  if (agentId !== null) scope.agentId = agentId;
  if (runId !== null) scope.runId = runId;

  const includeSuperseded = params.get("includeSuperseded") === "true";

  try {
    const memories = await getAll(scope, { includeSuperseded });
    return NextResponse.json(memories);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    const isValidation = message.startsWith("Invalid ");
    return NextResponse.json({ error: message }, { status: isValidation ? 400 : 500 });
  }
}
