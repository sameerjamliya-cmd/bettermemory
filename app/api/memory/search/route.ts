// Requires Authorization: Bearer <API_KEY>. The key gates who may call the API
// at all; it does NOT add per-user isolation — a caller holding the key may
// address any scope, so `userId` remains an identifier, not a credential.
// See "API and dashboard" in the README.

import { NextResponse } from "next/server";
import { requireApiKey } from "../../../../lib/api-auth";
import { search } from "../../../../lib/memory";

export async function POST(request: Request) {
  // Before any parsing, scope validation or memory logic.
  const unauthorized = requireApiKey(request);
  if (unauthorized) return unauthorized;

  let body: { query?: string; scope?: unknown; options?: unknown };
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Request body must be valid JSON." }, { status: 400 });
  }

  if (typeof body?.query !== "string") {
    return NextResponse.json({ error: 'Missing "query".' }, { status: 400 });
  }
  if (body?.scope === undefined) {
    return NextResponse.json({ error: 'Missing "scope".' }, { status: 400 });
  }

  try {
    const results = await search(body.query, body.scope as never, (body.options ?? undefined) as never);
    return NextResponse.json(results);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    const isValidation = message.startsWith("Invalid ");
    return NextResponse.json({ error: message }, { status: isValidation ? 400 : 500 });
  }
}
