// Exact-match dedup. Content is normalised before hashing so trivial whitespace
// or casing differences do not create a second copy of the same fact.
import { createHash } from "crypto";

export function normalizeForHash(text: string): string {
  return text.toLowerCase().trim().replace(/\s+/g, " ");
}

export function hashContent(text: string): string {
  return createHash("md5").update(normalizeForHash(text)).digest("hex");
}
