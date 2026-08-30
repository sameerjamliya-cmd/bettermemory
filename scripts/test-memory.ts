import * as readline from "readline/promises";
import { stdin, stdout } from "process";
import { add, search, getAll, deleteMemory } from "../lib/memory";

async function main() {
  const rl = readline.createInterface({ input: stdin, output: stdout });

  const userIdInput = await rl.question("userId (blank for \"sameer\")> ");
  const userId = userIdInput.trim() === "" ? "sameer" : userIdInput.trim();
  const scope = { userId };

  console.log(`\nScope: ${JSON.stringify(scope)}\n`);
  console.log("Enter sentences to add one at a time. Empty line to stop adding.");
  console.log('Commands: "list" (live memories), "list all" (including superseded),');
  console.log('          "delete <id>" (remove one memory from this scope).\n');

  while (true) {
    const line = await rl.question("add> ");
    if (line.trim() === "") break;

    const deleteMatch = line.trim().match(/^delete\s+(\S+)$/i);
    if (deleteMatch) {
      const result = await deleteMemory(deleteMatch[1], scope);
      if (result.status === "deleted") {
        console.log(`\ndeleted ${result.id}: "${result.content}"`);
        if (result.clearedSupersedeLinks.length > 0) {
          console.log(`  cleared supersede link on: ${result.clearedSupersedeLinks.join(", ")}`);
        }
      } else if (result.status === "not_found") {
        console.log(`\nnot found: ${result.id}`);
      } else {
        console.log(`\nrefused: ${result.id} is not in ${JSON.stringify(scope)}`);
      }
      console.log("");
      continue;
    }

    const command = line.trim().toLowerCase();
    if (command === "list" || command === "list all") {
      const includeSuperseded = command === "list all";
      const memories = await getAll(scope, { includeSuperseded });
      console.log(
        `\n${memories.length} memor${memories.length === 1 ? "y" : "ies"} in ${JSON.stringify(scope)}` +
          `${includeSuperseded ? " (including superseded)" : " (superseded hidden)"}, newest first:`
      );
      for (const m of memories) {
        const mark = m.superseded ? " [superseded]" : "";
        const link = m.supersededMemoryId ? ` (supersedes ${m.supersededMemoryId})` : "";
        console.log(`  ${m.extractedAt}  ${m.id}${mark}`);
        console.log(`    "${m.content}"${link}`);
      }
      console.log("");
      continue;
    }

    const result = await add(line, scope);
    console.log(JSON.stringify(result, null, 2));
    for (const fact of result.stored) {
      if (fact.supersededMemoryId !== null) {
        console.log(`  ↳ supersedes ${fact.supersededMemoryId} (old fact kept, hidden from default search)`);
      }
    }
    console.log("");
  }

  const query = await rl.question("\nsearch query> ");
  if (query.trim() !== "") {
    const results = await search(query, scope);
    console.log(JSON.stringify(results, null, 2));

    // Show what the default view hid, so supersession is visible while testing.
    const all = await search(query, scope, { includeSuperseded: true });
    const hidden = all.filter((r) => r.superseded);
    console.log(`\nDefault search returned ${results.length} result(s); ${hidden.length} superseded fact(s) hidden:`);
    for (const r of hidden) {
      console.log(`  - "${r.content}" (id ${r.id})`);
    }
  }

  rl.close();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
