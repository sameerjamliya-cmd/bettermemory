// Bearer-token check for the /api/memory/* routes.
//
// This is a gate on who may call the API at all. It is NOT per-user
// authorisation: a caller holding the key may address any scope, exactly as
// before. `userId` remains an identifier, not a credential.
import { NextResponse } from "next/server";

/** Constant-time comparison, so a wrong key cannot be recovered by timing. */
function safeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}

/**
 * Returns a 401 response when the request is not authorised, or null when it is.
 * Call this before any body parsing, scope validation or memory logic.
 */
export function requireApiKey(request: Request): NextResponse | null {
  const expected = process.env.API_KEY;

  // Fail closed. An unset key must not mean "open to everyone" — that is how a
  // deployment ends up unauthenticated by accident.
  if (!expected || expected.trim() === "") {
    return NextResponse.json(
      { error: "Server misconfigured: API_KEY is not set." },
      { status: 500 }
    );
  }

  const header = request.headers.get("authorization") ?? "";
  const match = /^Bearer\s+(.+)$/i.exec(header.trim());
  if (!match || !safeEqual(match[1], expected)) {
    // Deliberately identical for a missing and a wrong key: distinguishing them
    // tells an attacker whether the scheme is right.
    return NextResponse.json(
      { error: "Unauthorized. Send Authorization: Bearer <API_KEY>." },
      { status: 401 }
    );
  }

  return null;
}
