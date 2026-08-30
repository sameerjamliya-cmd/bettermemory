import { appendFile } from "fs/promises";
import path from "path";

const LOG_PATH = path.join(process.cwd(), "logs", "memory.log.jsonl");

export async function logEvent(event: Record<string, unknown>) {
  const line = JSON.stringify({ timestamp: new Date().toISOString(), ...event });
  await appendFile(LOG_PATH, line + "\n");
}
