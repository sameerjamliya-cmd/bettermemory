// The only place in the codebase that knows OpenAI's request/response shapes.
import { openai, EMBEDDING_MODEL, EXTRACTION_MODEL } from "../clients";
import type { ExtractedFact, ExtractOptions, LLMClient } from "./types";

function blankToNull(value: string | null | undefined): string | null {
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  return trimmed === "" ? null : trimmed;
}

export class OpenAILLMClient implements LLMClient {
  async embed(text: string): Promise<number[]> {
    const res = await openai.embeddings.create({
      model: EMBEDDING_MODEL,
      input: text,
    });
    return res.data[0].embedding;
  }

  async extractFacts(prompt: string, options?: ExtractOptions): Promise<ExtractedFact[]> {
    const imageDataUrl = options?.imageDataUrl;

    // One multimodal message rather than two calls: the image and any accompanying
    // text are a single observation, and the model must be able to read them
    // together (e.g. "this is the receipt from Tuesday" only makes sense jointly).
    const content = imageDataUrl
      ? ([
          { type: "text", text: prompt },
          { type: "image_url", image_url: { url: imageDataUrl } },
        ] as const)
      : prompt;

    const res = await openai.chat.completions.create({
      model: EXTRACTION_MODEL,
      messages: [{ role: "user", content: content as never }],
      tools: [
        {
          type: "function",
          function: {
            name: "return_facts",
            description: "Return the extracted facts.",
            parameters: {
              type: "object",
              properties: {
                facts: {
                  type: "array",
                  items: {
                    type: "object",
                    properties: {
                      content: { type: "string" },
                      attributeKey: { type: "string" },
                      supersedes: { type: ["string", "null"] },
                      skip: { type: "boolean" },
                      skipReason: { type: ["string", "null"] },
                    },
                    required: ["content", "attributeKey", "supersedes", "skip", "skipReason"],
                  },
                },
              },
              required: ["facts"],
            },
          },
        },
      ],
      tool_choice: { type: "function", function: { name: "return_facts" } },
    });

    const call = res.choices[0].message.tool_calls?.[0];
    if (!call) return [];
    const args = JSON.parse(call.function.arguments);
    if (!Array.isArray(args.facts)) return [];

    // The schema allows "string | null", and the model sometimes says "" where it
    // means null. An empty label would otherwise reach labelToMemory.get(""),
    // miss, and silently degrade a supersede into an unlinked new fact — the same
    // input producing different stored shapes with no error. Normalise once, here,
    // so downstream code only ever sees a real label or null.
    return (args.facts as ExtractedFact[]).map((fact) => ({
      ...fact,
      supersedes: blankToNull(fact.supersedes),
      skipReason: blankToNull(fact.skipReason),
    }));
  }
}
