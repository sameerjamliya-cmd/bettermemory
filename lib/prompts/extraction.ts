// The extraction contract: every rule and worked example the model is given.
// Kept apart from the assembly logic so the wording can be reviewed and diffed
// without wading through string plumbing. Values in square brackets are
// deliberate placeholders — see the note at the top of the rules.

/** Rules and examples, verbatim. Contains no interpolation. */
export const EXTRACTION_RULES = String.raw`Extract any standalone facts from the new message as a list of objects: { "content": "...", "supersedes": "<id>" | null, "skip": true | false, "skipReason": "duplicate_of: <id>" | null }.

Note on the examples below: values in square brackets ([N], [CITY_NEW], [PERSON], ...) are placeholders standing in for real values. They illustrate the shape of correct output only. Never copy a bracketed placeholder, or any other literal value from an example, into a fact's "content" — the wording of every fact you extract must come from the New Message itself. This does not restrict "supersedes" or "skipReason", which reference the Existing Memories by their id and must still be set whenever the rules above call for it.

Rules:
1. If the new message describes a change from an old state to a new one (e.g. "used to be X, now Y" or "switched from X to Y"), extract ONLY the current/final state as one clear, self-contained fact.
2. For each fact you extract, check the Existing Memories list and decide "supersedes":
- Set "supersedes" to an existing memory's id when this fact updates or changes that memory's value. Otherwise set "supersedes" to null. Only set it when you're confident it's the same underlying fact with a new value — not just a related topic.
- A fact with no new information beyond what an existing memory already states is NOT a supersede — it's a duplicate, skip it entirely.
- Only set "supersedes" if the new fact is a genuine, complete replacement — at least as specific as the memory it supersedes. If the new message is too vague to match that specificity, do not set "supersedes"; store it as a new, unlinked memory instead.

Example:
Existing Memory: [{"id": "0", "text": "My bench press PR is [N]kg."}]
New Message: "finally cracked past that bench plateau today"
WRONG: [{"content": "I cracked past my bench press plateau today.", "supersedes": "0", "skip": false, "skipReason": null}]  ← buries the only fact with a real number
CORRECT: [{"content": "Broke past previous bench press plateau, exact new PR not stated.", "supersedes": null, "skip": false, "skipReason": null}]  ← old [N]kg fact stays visible, vague fact stored alongside it

When a message describes a change to an ongoing state — a location, job, role, relationship, living situation, possession, or habit — even if phrased as an event ("moved to", "switched to", "started", "got a new", "began") rather than a state, rewrite the extracted fact in the same current-state form as the existing memory it updates, so the connection is unambiguous.

Example 1 — location:
Existing Memory: [{"id": "0", "text": "My current city is [CITY_OLD]."}]
New Message: "I moved to [CITY_NEW]."
WRONG: [{"content": "I moved to [CITY_NEW].", "supersedes": null}]
CORRECT: [{"content": "My current city is [CITY_NEW].", "supersedes": "0"}]

Example 2 — job/role:
Existing Memory: [{"id": "0", "text": "I work at a design studio as a product designer."}]
New Message: "I now work at a fintech company as an engineer."
WRONG: [{"content": "I now work at a fintech company as an engineer.", "supersedes": null}]
CORRECT: [{"content": "I work at a fintech company as an engineer.", "supersedes": "0"}]

Example 3 — living situation:
Existing Memory: [{"id": "0", "text": "I live with two roommates."}]
New Message: "I just moved into my own place."
WRONG: [{"content": "I just moved into my own place.", "supersedes": null}]
CORRECT: [{"content": "I live alone now.", "supersedes": "0"}]

Example 4 — possession:
Existing Memory: [{"id": "0", "text": "I drive a [CAR_OLD]."}]
New Message: "Got a new car last week, a [CAR_NEW]."
WRONG: [{"content": "Got a new car last week, a [CAR_NEW].", "supersedes": null}]
CORRECT: [{"content": "I drive a [CAR_NEW].", "supersedes": "0"}]

If a fact you would extract is semantically equivalent to an existing memory shown above, with no new information, still include it in the output, but set "skip": true and "skipReason": "duplicate_of: <id>" (the id of the existing memory it duplicates). This is different from "supersedes": use "supersedes" when the new fact updates or changes an old value; use "skip": true when the new fact is just a restatement of information that's already fully captured, with nothing new to store. When "skip" is true, "supersedes" must be null and "skipReason" must be set; when "skip" is false, "skipReason" must be null.

3. Extract each distinct fact as exactly one sentence. Do not split one fact into multiple entries, and do not merge two distinct facts into one.
4. If there's nothing worth remembering, return an empty list.

CRITICAL — Extract ONLY from the New Message. Never extract facts based on:
- Your own knowledge, assumptions, or inferences about the user
- The Existing Memories shown above (they are context for deciding "supersedes", not source material for new facts)
- What seems plausible, likely, or a reasonable guess

If a fact is not explicitly and literally stated in the New Message text, do not extract it — even if it seems related to or implied by the conversation history.

Before extracting each fact, verify: "Is this sentence, or something equivalent to it, actually present in the New Message?" If not, discard it.

Example — do not extract from context:
Existing Memories: [{"id": "0", "text": "My favorite language for backend work used to be Python, but I've switched to TypeScript."}]
New Message: "I go to the gym 5 days a week."
WRONG Output: [{"content": "I go to the gym 5 days a week.", "supersedes": null, "skip": false, "skipReason": null}, {"content": "My favorite language is TypeScript.", "supersedes": null, "skip": false, "skipReason": null}]  ← TypeScript fact was NOT in this New Message, it leaked in from Existing Memories
CORRECT Output: [{"content": "I go to the gym 5 days a week.", "supersedes": null, "skip": false, "skipReason": null}]

CRITICAL — Resolve every relative time reference in the New Message ("last week", "yesterday", "this morning", "a couple of months ago") against the Observation Date, never against the Current Date. The two are usually the same, but when they differ the Observation Date is the moment the message describes and is authoritative. State the resolved date in the fact where it matters, so the fact stays true when read later.

Resolving a relative reference into a concrete date this way is expected and is NOT an invented value — the Observation Date is given to you above for exactly this purpose. It is the ONLY value that may enter a fact from outside the New Message, and it may ONLY be used to fill in a time reference. It must not influence any other decision: it never affects whether two facts describe the same underlying thing, and it never justifies a "supersedes" link.

Example:
Observation Date: 2026-03-14T09:00:00.000Z
Current Date: 2026-09-06T12:00:00.000Z
New Message: "Hit a new deadlift PR last week."
WRONG: [{"content": "Hit a new deadlift PR in the week of 2026-08-31.", "supersedes": null}]  ← resolved against the Current Date
CORRECT: [{"content": "Hit a new deadlift PR in the week of 2026-03-09.", "supersedes": null}]  ← resolved against the Observation Date

CRITICAL — Do not resolve pronouns, ambiguous references, or implied subjects using content from Existing Memories. If the New Message contains a pronoun ("his", "her", "their", "it") or an ambiguous reference whose subject is not explicitly stated in the New Message itself, extract the fact with the reference intact, exactly as written — do not substitute in a name or detail pulled from Existing Memories, even if it seems like an obvious, helpful resolution.

Example:
Existing Memory: [{"id": "0", "text": "[PERSON]'s favorite color is [COLOR]."}]
New Message: "Saw someone wearing his signature color today, didn't say hi."
WRONG: [{"content": "Saw someone wearing [PERSON]'s favorite color today, didn't say hi.", "supersedes": null}]  ← "[PERSON]" was never stated in the New Message, only inferred from Existing Memories
CORRECT: [{"content": "Saw someone wearing his signature color today, didn't say hi.", "supersedes": null}]  ← reference left exactly as stated, unresolved

Example — do not invent unstated values:
New Message: "broke through a plateau today, finally past [N] on the lift"
WRONG Output: [{"content": "My deadlift PR is [M]kg.", "supersedes": "0", "skip": false, "skipReason": null}]  ← [M] was never stated anywhere in the New Message
CORRECT Output: [{"content": "Broke past [N]kg on deadlift, exact new PR not stated.", "supersedes": "0", "skip": false, "skipReason": null}]

Example 1 — supersession:
Existing Memories: [{"id": "0", "text": "My squat PR is [N]kg."}]
New Message: "Hit a new squat PR today, [M]kg."
Output: [{"content": "My squat PR is [M]kg.", "supersedes": "0", "skip": false, "skipReason": null}]

Example 2 — no relation:
Existing Memories: [{"id": "0", "text": "I work at a healthtech startup."}]
New Message: "I'm learning Rust on the side."
Output: [{"content": "I am learning Rust.", "supersedes": null, "skip": false, "skipReason": null}]

Example — skip a restatement:
Existing Memories: [{"id": "0", "text": "I usually train at the gym in the evening."}]
New Message: "I usually train at the gym in the evening"
Output: [{"content": "I usually train at the gym in the evening.", "supersedes": null, "skip": true, "skipReason": "duplicate_of: 0"}]

Example — skip a paraphrased duplicate (not just verbatim):
Existing Memories: [{"id": "0", "text": "I usually train at the gym in the evening."}]
New Message: "I like working out in the evenings at the gym"
WRONG: [{"content": "I like working out in the evenings at the gym.", "supersedes": "0"}]  ← no new information was added, this is not an update, it's the same fact reworded
CORRECT: [{"content": "I like working out in the evenings at the gym.", "supersedes": null, "skip": true, "skipReason": "duplicate_of: 0"}]`;

/** The output-format line, kept last in the assembled prompt. */
export const EXTRACTION_OUTPUT_INSTRUCTION = String.raw`Return a JSON list of objects.`;
