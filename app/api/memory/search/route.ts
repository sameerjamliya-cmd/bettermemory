// ⚠️  NO AUTHENTICATION. These routes are a localhost debugging surface with the
// same trust model as the console script: `userId` is an identifier, not a
// credential, so anyone who can reach this port can read and write every scope
// by supplying its userId. Do not bind to a public interface or deploy this.
// See "API and dashboard" in the README.

import { NextResponse } from "next/server";
import { search } from "../../../../lib/memory";

export async function POST(request: Request) {
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
