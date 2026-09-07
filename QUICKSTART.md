# Quickstart

From a fresh clone to seeing the memory layer work, in about five minutes.
Requires Node 18+, Docker, and an OpenAI API key.

## 1. Install and configure

```bash
npm install
docker compose up -d                 # Qdrant on :6333
cp .env.example .env.local
```

Open `.env.local` and set `OPENAI_API_KEY`, plus an `API_KEY` for the REST API:

```bash
openssl rand -hex 32     # paste the result as API_KEY
```

Check Qdrant is up:

```bash
curl http://localhost:6333/healthz    # -> healthz check passed
```

## 2. See it work (30 seconds)

```bash
npm run test:memory
```

At the prompts, paste these one at a time:

```
sameer
My current city is Chennai.
I relocated to Bangalore.
list
list all
remember Always reply in British English.
instructions
```

Then press Enter on an empty line and type a search query:

```
where do I live
```

What you should see: the second sentence **supersedes** the first, so `list`
shows only Bangalore while `list all` still shows Chennai marked
`[superseded]` — nothing was deleted. The procedural memory appears under
`instructions` but never in `list` or in the search results.

## 3. Run the test suites

```bash
npm run test:suite         # 11 deterministic cases, self-cleaning
npm run test:extraction    # extraction-quality RATES, non-gating, always exits 0
```

`test:suite` should be 11/11. `test:extraction` reports pass rates rather than
pass/fail, because those scenarios depend on a model judgement that varies
between runs — see "Known limitations" in the README.

## 4. Browse it in a browser

```bash
npm run dev
```

Open <http://localhost:3000/dashboard>, enter `sameer` as the userId, and click
**Load memories**. Tick *include superseded* to see the history.

> The dashboard reads the key server-side, so nothing secret reaches the browser.
> Calling the API directly needs `Authorization: Bearer $API_KEY`. The key gates
> who may call the API; it does not isolate one `userId` from another.

## Using it from code

```ts
import { add, search, getAll, addProcedural, deleteMemory } from "./lib/memory";

const scope = { userId: "sameer" };

await add("My deadlift PR is 140kg.", scope);
await add("Just hit a new deadlift PR, 145kg this time.", scope);  // supersedes

await search("what is my deadlift pr", scope);            // -> 145kg only
await getAll(scope, { includeSuperseded: true });         // -> both, 140 flagged

await add({ text: "The receipt from today", imageBase64 }, scope);  // vision
await add("Hit a new PR last week.", scope, { observedAt: "2026-03-14T09:00:00Z" });
await addProcedural("Always reply in British English.", scope);
```

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| `OPENAI_API_KEY ... is missing` | `.env.local` not created, or key not set |
| `ECONNREFUSED ... 6333` | Qdrant not running — `docker compose up -d` |
| `Port 3000 is in use` | Next picks the next free port; read the URL it prints |
| `test:extraction` rates look low | Expected — those cases are stochastic by design |
