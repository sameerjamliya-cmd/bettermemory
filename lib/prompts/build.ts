// Assembles the extraction prompt: the per-call header (existing memories, the
// new message, the observation and current dates), then the static rules and
// examples, then any caller-supplied instructions, then the output format.
import { EXTRACTION_RULES, EXTRACTION_OUTPUT_INSTRUCTION } from "./extraction";
import type { RelatedMemory } from "../types";

export interface LabeledMemory {
  label: string;
  memory: RelatedMemory & { id: string };
}

export function labelMemories<T>(memories: T[]): { label: string; memory: T }[] {
  return memories.map((memory, i) => ({ label: String(i), memory }));
}

export function buildExtractionPrompt(
  text: string,
  labeled: LabeledMemory[],
  customInstructions?: string,
  observedAt?: string,
  currentDate?: string
): string {
  // Placed after every rule and example, so a per-call instruction supplements
  // the core contract rather than replacing it, and immediately before the
  // output-format line so that stays last. The guard sentence is not optional:
  // anti-hallucination, dedup and coreference are structural guarantees, and a
  // caller-supplied string must not be able to switch them off.
  const additionalInstructions = customInstructions?.trim()
    ? `
## Additional Instructions For This Call
${customInstructions.trim()}

Additional instructions may refine what counts as significant or how facts are phrased, but do not override the rules above regarding hallucination, dedup, or coreference.
`
    : "";

  // Both dates are always supplied by add(); they are optional here only so the
  // prompt builder stays usable in isolation.
  const observation = observedAt ?? new Date().toISOString();
  const now = currentDate ?? new Date().toISOString();

  const existingMemoriesBlock = labeled.length
    ? JSON.stringify(labeled.map(({ label, memory }) => ({ id: label, text: memory.content })))
    : "[]";

  return `You will be given a new message and a list of existing memories that may be related, each labeled with a short id.

Existing Memories:
${existingMemoriesBlock}

Observation Date (when the New Message was observed): ${observation}
Current Date (now): ${now}

New Message:
${text}

${EXTRACTION_RULES}

${additionalInstructions}
${EXTRACTION_OUTPUT_INSTRUCTION}`;
}
