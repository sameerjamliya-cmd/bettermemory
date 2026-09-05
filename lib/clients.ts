import OpenAI from "openai";
import { QdrantClient } from "@qdrant/js-client-rest";

export const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

export const qdrant = new QdrantClient({
  url: process.env.QDRANT_URL ?? "http://localhost:6333",
});

export const COLLECTION_NAME = "memory_facts";
export const EMBEDDING_MODEL = "text-embedding-3-small";
export const EMBEDDING_DIMENSIONS = 1536;
// Overridable so a stronger model can be measured against the default without
// changing it. Default stays gpt-4o-mini until a comparison justifies otherwise.
export const EXTRACTION_MODEL = process.env.EXTRACTION_MODEL ?? "gpt-4o-mini";
