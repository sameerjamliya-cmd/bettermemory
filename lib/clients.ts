import OpenAI from "openai";
import { QdrantClient } from "@qdrant/js-client-rest";

// Both SDK clients are constructed lazily, on first property access.
//
// Eager construction meant that merely importing lib/memory.ts built an OpenAI
// client, which throws when OPENAI_API_KEY is unset — so a caller that supplies
// its own providers still needed credentials for vendors it never uses. The
// Proxy keeps the original `openai.x` / `qdrant.y` call shape at every existing
// call site while deferring the constructor until something is actually called.
function lazy<T extends object>(create: () => T): T {
  let instance: T | null = null;
  const resolve = (): T => (instance ??= create());
  return new Proxy({} as T, {
    get(_target, prop, receiver) {
      const value = Reflect.get(resolve() as object, prop, receiver);
      return typeof value === "function" ? value.bind(resolve()) : value;
    },
    has: (_t, prop) => prop in (resolve() as object),
    getPrototypeOf: () => Object.getPrototypeOf(resolve()),
  });
}

export const openai = lazy(() => new OpenAI({ apiKey: process.env.OPENAI_API_KEY }));

export const qdrant = lazy(
  () => new QdrantClient({ url: process.env.QDRANT_URL ?? "http://localhost:6333" })
);

export const COLLECTION_NAME = "memory_facts";
export const EMBEDDING_MODEL = "text-embedding-3-small";
export const EMBEDDING_DIMENSIONS = 1536;
// Overridable so a stronger model can be measured against the default without
// changing it. Default stays gpt-4o-mini until a comparison justifies otherwise.
export const EXTRACTION_MODEL = process.env.EXTRACTION_MODEL ?? "gpt-4o-mini";
