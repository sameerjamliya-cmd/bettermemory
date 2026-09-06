// ⚠️  NO AUTHENTICATION. These routes are a localhost debugging surface with the
// same trust model as the console script: `userId` is an identifier, not a
// credential, so anyone who can reach this port can read and write every scope
// by supplying its userId. Do not bind to a public interface or deploy this.
// See "API and dashboard" in the README.

import { NextResponse } from "next/server";
import { add } from "../../../../lib/memory";

// Thin wrapper: scope and input validation already live inside add(), so the
// route only translates a thrown validation error into a 400.
export async function POST(request: Request) {
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
