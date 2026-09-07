// Requires Authorization: Bearer <API_KEY>. The key gates who may call the API
// at all; it does NOT add per-user isolation — a caller holding the key may
// address any scope, so `userId` remains an identifier, not a credential.
// See "API and dashboard" in the README.

import { NextResponse } from "next/server";
import { requireApiKey } from "../../../../lib/api-auth";
import { add } from "../../../../lib/memory";

// Thin wrapper: scope and input validation already live inside add(), so the
// route only translates a thrown validation error into a 400.
export async function POST(request: Request) {
  // Before any parsing, scope validation or memory logic.
  const unauthorized = requireApiKey(request);
  if (unauthorized) return unauthorized;

  let body: { text?: string; imageBase64?: string; scope?: unknown; options?: unknown };
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Request body must be valid JSON." }, { status: 400 });
  }

  if (body?.scope === undefined) {
    return NextResponse.json({ error: 'Missing "scope".' }, { status: 400 });
  }

  try {
    const input =
      body.imageBase64 !== undefined
        ? { text: body.text, imageBase64: body.imageBase64 }
        : (body.text ?? "");
    const result = await add(
      input as never,
      body.scope as never,
      (body.options ?? undefined) as never
    );
    return NextResponse.json(result);
  } catch (err) {
    // add() throws on invalid scope or empty input; anything else is a 500.
    const message = err instanceof Error ? err.message : String(err);
    const isValidation = message.startsWith("Invalid ");
    return NextResponse.json({ error: message }, { status: isValidation ? 400 : 500 });
  }
}
