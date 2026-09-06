"use client";

import { useState } from "react";

interface MemoryRow {
  id: string;
  content: string;
  extractedAt: string;
  superseded: boolean;
  supersededMemoryId: string | null;
}

interface SearchRow extends MemoryRow {
  cosineScore: number;
  bm25RawScore: number | null;
  combinedScore: number;
}

const box: React.CSSProperties = {
  border: "1px solid #ccc",
  borderRadius: 4,
  padding: 12,
  marginBottom: 16,
};

const mono: React.CSSProperties = { fontFamily: "monospace", fontSize: 12, color: "#666" };

export default function Dashboard() {
  const [userId, setUserId] = useState("");
  const [agentId, setAgentId] = useState("");
  const [runId, setRunId] = useState("");
  const [includeSuperseded, setIncludeSuperseded] = useState(false);

  const [memories, setMemories] = useState<MemoryRow[] | null>(null);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const scope = () => {
    const s: Record<string, string> = { userId: userId.trim() };
    if (agentId.trim()) s.agentId = agentId.trim();
    if (runId.trim()) s.runId = runId.trim();
    return s;
  };

  async function loadMemories(e?: React.FormEvent) {
    e?.preventDefault();
    setError(null);
    setBusy(true);
    try {
      const params = new URLSearchParams(scope());
      if (includeSuperseded) params.set("includeSuperseded", "true");
      const res = await fetch(`/api/memory/all?${params.toString()}`);
      const data = await res.json();
      if (!res.ok) throw new Error(data.error ?? `HTTP ${res.status}`);
      setMemories(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setMemories(null);
    } finally {
      setBusy(false);
    }
  }

  async function runSearch(e?: React.FormEvent) {
    e?.preventDefault();
    setError(null);
    setBusy(true);
    try {
      const res = await fetch("/api/memory/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, scope: scope() }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error ?? `HTTP ${res.status}`);
      setResults(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setResults(null);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main style={{ maxWidth: 820, margin: "40px auto", padding: "0 16px", fontFamily: "system-ui, sans-serif" }}>
      <h1 style={{ fontSize: 22 }}>Memory dashboard</h1>
      <p style={{ color: "#666", fontSize: 14 }}>
        View-only. Enter a scope to inspect its memories and run searches against it.
      </p>

      <form onSubmit={loadMemories} style={box}>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center" }}>
          <input value={userId} onChange={(e) => setUserId(e.target.value)} placeholder="userId (required)" style={{ padding: 6, flex: "1 1 200px" }} />
          <input value={agentId} onChange={(e) => setAgentId(e.target.value)} placeholder="agentId (optional)" style={{ padding: 6, flex: "1 1 140px" }} />
          <input value={runId} onChange={(e) => setRunId(e.target.value)} placeholder="runId (optional)" style={{ padding: 6, flex: "1 1 140px" }} />
          <button type="submit" disabled={busy || !userId.trim()} style={{ padding: "6px 14px" }}>
            Load memories
          </button>
        </div>
        <label style={{ display: "block", marginTop: 8, fontSize: 13 }}>
          <input type="checkbox" checked={includeSuperseded} onChange={(e) => setIncludeSuperseded(e.target.checked)} />{" "}
          include superseded
        </label>
      </form>

      {error && (
        <div style={{ ...box, borderColor: "#c00", color: "#c00" }}>
          <strong>Error:</strong> {error}
        </div>
      )}

      {memories && (
        <section style={box}>
          <h2 style={{ fontSize: 16 }}>
            {memories.length} memor{memories.length === 1 ? "y" : "ies"}
          </h2>
          {memories.length === 0 && <p style={{ color: "#666" }}>Nothing stored for this scope.</p>}
          <ul style={{ listStyle: "none", padding: 0 }}>
            {memories.map((m) => (
              <li key={m.id} style={{ borderTop: "1px solid #eee", padding: "8px 0" }}>
                <div>
                  {m.content}{" "}
                  {m.superseded && (
                    <span style={{ color: "#a60", fontSize: 12 }}>[superseded]</span>
                  )}
                </div>
                <div style={mono}>
                  {m.extractedAt} · {m.id}
                  {m.supersededMemoryId ? ` · supersedes ${m.supersededMemoryId}` : ""}
                </div>
              </li>
            ))}
          </ul>
        </section>
      )}

      <form onSubmit={runSearch} style={box}>
        <div style={{ display: "flex", gap: 8 }}>
          <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="search this scope" style={{ padding: 6, flex: 1 }} />
          <button type="submit" disabled={busy || !userId.trim() || !query.trim()} style={{ padding: "6px 14px" }}>
            Search
          </button>
        </div>
      </form>

      {results && (
        <section style={box}>
          <h2 style={{ fontSize: 16 }}>{results.length} result{results.length === 1 ? "" : "s"}</h2>
          <ul style={{ listStyle: "none", padding: 0 }}>
            {results.map((r) => (
              <li key={r.id} style={{ borderTop: "1px solid #eee", padding: "8px 0" }}>
                <div>{r.content}</div>
                <div style={mono}>
                  combined {r.combinedScore.toFixed(4)} · cosine {r.cosineScore.toFixed(4)} · bm25{" "}
                  {r.bm25RawScore === null ? "n/a" : r.bm25RawScore.toFixed(4)}
                </div>
              </li>
            ))}
          </ul>
        </section>
      )}
    </main>
  );
}
