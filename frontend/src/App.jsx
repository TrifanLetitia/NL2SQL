import { useState, useEffect, useRef } from "react";

const API = "http://127.0.0.1:5000";

const MODELS = [
  { id: "template",    label: "Template-based",    color: "#f97316"},
  { id: "grammar",     label: "Grammar PCFG",       color: "#8b5cf6"},
  { id: "seq2seq",     label: "Seq2Seq LSTM",       color: "#06b6d4"},
  { id: "transformer", label: "Transformer",        color: "#10b981"},
  { id: "mt5",         label: "mT5 Fine-tuned",     color: "#ec4899"},
  { id: "prompt",      label: "Prompt Engineering", color: "#f59e0b"},
];

const TECHNIQUES = [
  { id: "few_shot",  label: "Few-shot"  }
];

const SCHEMA = {
  pacienti:    ["id","nume","prenume","varsta","sex","oras","telefon"],
  medici:      ["id","nume","prenume","specialitate","experienta_ani","oras"],
  consultatii: ["id","pacient_id","medic_id","data","diagnostic","cost","durata_minute"],
};

const EXAMPLES = [
  "câți medici sunt din sibiu ?",
  "medici din cluj-napoca",
  "pacienți cu vârsta peste 50 ani",
  "toate consultațiile pacientului ionescu",
  "care este costul total al consultațiilor ?",
  "care este experiența medie a medicilor ?",
  "medici cardiologi din bucurești",
  "câte consultații au fost în 2023 ?",
  "arată toți pacienții",
  "medici de neurologie din timișoara",
];

const C = {
  bg: "linear-gradient(135deg, #f8fafc 0%, #eef2ff 45%, #fdf4ff 100%)",
  surface: "rgba(255, 255, 255, 0.88)",
  card: "rgba(255, 255, 255, 0.92)",
  cardHover: "rgba(248, 250, 252, 0.98)",
  border: "rgba(148, 163, 184, 0.22)",
  text: "#0f172a",
  muted: "#64748b",
  dim: "#475569",
  accent: "#6366f1",
  accent2: "#ec4899",
  danger: "#ef4444",
  success: "#10b981",
};

function tokenizeSQL(sql) {
  if (sql == null) return [];

  if (typeof sql !== "string") {
    if (typeof sql === "object" && sql.sql) {
      sql = sql.sql;
    } else if (typeof sql === "object" && sql.error) {
      sql = sql.error;
    } else {
      sql = JSON.stringify(sql);
    }
  }

  if (!sql.trim()) return [];

  const KW = new Set([
    "SELECT", "FROM", "WHERE", "JOIN", "ON", "AND", "OR", "AS",
    "ORDER", "BY", "GROUP", "HAVING", "LIMIT", "LIKE", "IN",
    "NOT", "IS", "NULL", "COUNT", "SUM", "AVG", "MAX", "MIN",
    "DISTINCT", "LEFT", "RIGHT", "INNER", "OUTER", "ASC", "DESC"
  ]);

  return sql
    .split(/(\s+|[,();*=<>'`])/g)
    .filter(Boolean)
    .map((tok, i) => {
      const trimmed = tok.trim();
      const u = trimmed.toUpperCase();

      if (KW.has(u)) {
        return <span key={i} style={{ color: "#2563eb", fontWeight: 700 }}>{tok}</span>;
      }

      if (/^'[^']*'$/.test(trimmed)) {
        return <span key={i} style={{ color: "#059669" }}>{tok}</span>;
      }

      if (/^\d+(\.\d)?$/.test(trimmed)) {
        return <span key={i} style={{ color: "#d97706" }}>{tok}</span>;
      }

      if (/^[a-z_]+\.[a-z_]+$/i.test(trimmed)) {
        return <span key={i} style={{ color: "#7c3aed" }}>{tok}</span>;
      }

      if (trimmed === ";") {
        return <span key={i} style={{ color: "#64748b" }}>{tok}</span>;
      }

      return <span key={i} style={{ color: "#1e293b" }}>{tok}</span>;
    });
}

function ModelCard({ model, result, loading, technique, onTechChange }) {
  const [copied, setCopied] = useState(false);

  const copy = () => {
    if (!result?.sql) return;
    navigator.clipboard.writeText(result.sql);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const isPrompt  = model.id === "prompt";
  const hasResult = result?.sql && !result?.error;
  const hasError  = result?.error;

  return (
    <div style={{
      background: C.card,
      border: `1px solid ${loading ? model.color + "60" : hasResult ? model.color + "40" : hasError ? "#ef444430" : C.border}`,
      borderRadius: 18,
      overflow: "hidden",
      transition: "all 0.3s",
      boxShadow: hasResult ? `0 18px 42px ${model.color}14` : "0 12px 30px rgba(148, 163, 184, 0.16)",
      backdropFilter: "blur(18px)",
    }}>
      {/* Header */}
      <div style={{
        padding: "10px 16px",
        borderBottom: `1px solid ${C.border}`,
        display: "flex", alignItems: "center", gap: 10,
        background: `linear-gradient(135deg, ${model.color}18, rgba(15, 23, 42, 0.35))`,
      }}>
        <span style={{ fontSize: 14 }}>{model.icon}</span>
        <div>
          <span style={{ fontSize: 12, color: model.color, fontWeight: 600 }}>{model.label}</span>
          <span style={{ fontSize: 10, color: C.muted, marginLeft: 8 }}>{model.desc}</span>
        </div>

        {}
        {isPrompt && (
          <div style={{ display: "flex", gap: 4, marginLeft: 8 }}>
            {TECHNIQUES.map(t => (
              <button
                key={t.id}
                onClick={() => onTechChange(t.id)}
                style={{
                  background: technique === t.id ? `${model.color}25` : "transparent",
                  border: `1px solid ${technique === t.id ? model.color : C.border}`,
                  borderRadius: 999, padding: "2px 8px",
                  color: technique === t.id ? model.color : C.muted,
                  fontSize: 9, fontFamily: "inherit", cursor: "pointer",
                  transition: "all 0.15s",
                }}>
                {t.label}
              </button>
            ))}
          </div>
        )}

        <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 10 }}>
          {hasResult && (
            <span style={{ fontSize: 10, color: C.muted }}>{result.time_ms}ms</span>
          )}
          {}
          <div style={{
            width: 7, height: 7, borderRadius: "50%",
            background: loading ? model.color : hasResult ? model.color : hasError ? C.danger : C.border,
            boxShadow: loading ? `0 0 10px ${model.color}` : hasResult ? `0 0 6px ${model.color}80` : "none",
            animation: loading ? "pulse 1s infinite" : "none",
            transition: "all 0.3s",
          }} />
          {hasResult && (
            <button onClick={copy} style={{
              background: "transparent", border: "none",
              cursor: "pointer", fontSize: 10,
              color: copied ? C.success : C.muted, fontFamily: "inherit",
            }}>
              {copied ? "✓" : "copy"}
            </button>
          )}
        </div>
      </div>

      {}
      <div style={{ padding: "14px 16px", minHeight: 64 }}>
        {loading && (
          <div style={{ display: "flex", alignItems: "center", gap: 8, color: C.muted, fontSize: 11 }}>
            <div style={{
              width: 13, height: 13,
              border: `2px solid ${C.border}`, borderTopColor: model.color,
              borderRadius: "50%", animation: "spin 0.7s linear infinite",
            }} />
            {model.id === "prompt" ? "Apel API Gemini..." : "Generare SQL..."}
          </div>
        )}
        {!loading && hasError && (
          <span style={{ fontSize: 11, color: C.danger }}>✗ {result.error}</span>
        )}
        {!loading && hasResult && (
          <code style={{ fontSize: 12, lineHeight: 2.0, wordBreak: "break-word", display: "block" }}>
            {tokenizeSQL(result.sql)}
          </code>
        )}
        {!loading && !result && (
          <span style={{ fontSize: 11, color: C.muted, fontStyle: "italic" }}>
            {model.id === "prompt" ? `Tehnica: ${technique}` : "—"}
          </span>
        )}
      </div>
    </div>
  );
}

export default function NL2SQLApp() {
  const [query,       setQuery]       = useState("");
  const [results,     setResults]     = useState({});
  const [loading,     setLoading]     = useState({});
  const [serverOk,    setServerOk]    = useState(null);
  const [modelStatus, setModelStatus] = useState({});
  const [activeTable, setActiveTable] = useState("pacienti");
  const [history,     setHistory]     = useState([]);
  const [technique,   setTechnique]   = useState("few_shot");
  const inputRef = useRef(null);

  useEffect(() => {
    fetch(`${API}/health`)
      .then(r => r.json())
      .then(d => { setServerOk(true); setModelStatus(d.models || {}); })
      .catch(() => setServerOk(false));
    inputRef.current?.focus();
  }, []);

  const runModel = async (modelId, question) => {
    setLoading(l => ({ ...l, [modelId]: true }));
    setResults(r => ({ ...r, [modelId]: null }));
    try {
      const body = { question, model: modelId };
      if (modelId === "prompt") body.technique = technique;

      const res  = await fetch(`${API}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      setResults(r => ({ ...r, [modelId]: data }));
    } catch {
      setResults(r => ({ ...r, [modelId]: { error: "Server indisponibil" } }));
    } finally {
      setLoading(l => ({ ...l, [modelId]: false }));
    }
  };

  const handleQuery = async (q = query) => {
    const trimmed = q.trim();
    if (!trimmed) return;
    setQuery(trimmed);
    setResults({});
    setHistory(h => [{ q: trimmed, ts: Date.now() }, ...h.slice(0, 9)]);
    MODELS.forEach(m => runModel(m.id, trimmed));
  };

  const handleExample = (ex) => { setQuery(ex); handleQuery(ex); };
  const anyLoading    = Object.values(loading).some(Boolean);

  return (
    <div style={{ minHeight: "100vh", background: C.bg, color: C.text, fontFamily: "'JetBrains Mono','Fira Code',monospace" }}>
      <style>{`
        @keyframes spin  { to { transform: rotate(360deg); } }
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.35} }
        * { box-sizing: border-box; }
        textarea:focus { outline: none; }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-thumb { background: rgba(34, 211, 238, 0.45); border-radius: 999px; }
      `}</style>

      {}
      <div style={{ borderBottom: `1px solid ${C.border}`, padding: "13px 28px", display: "flex", alignItems: "center", gap: 14, background: C.surface }}>
        <div style={{ width: 9, height: 9, borderRadius: "50%", background: C.accent, boxShadow: `0 0 18px ${C.accent}` }} />
        <span style={{ fontSize: 11, letterSpacing: 4, color: C.muted, textTransform: "uppercase" }}>NL → SQL</span>
        <span style={{ color: C.border }}>|</span>
        <span style={{ fontSize: 11, color: C.dim }}>Sistem medical</span>
        <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 8 }}>
          <div style={{ width: 6, height: 6, borderRadius: "50%", background: serverOk === null ? C.muted : serverOk ? C.success : C.danger }} />
          <span style={{ fontSize: 10, color: C.muted }}>
            {serverOk === null ? "verificare..." : serverOk ? "server activ" : "server offline"}
          </span>
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "230px 1fr 200px", minHeight: "calc(100vh - 49px)" }}>

        {}
        <div style={{ borderRight: `1px solid ${C.border}`, background: C.surface, display: "flex", flexDirection: "column", overflowY: "auto" }}>
          {}
          <div style={{ padding: 18, borderBottom: `1px solid ${C.border}` }}>
            <div style={{ fontSize: 9, letterSpacing: 3, color: C.muted, textTransform: "uppercase", marginBottom: 12 }}>Schema BD</div>
            {Object.entries(SCHEMA).map(([tbl, cols]) => (
              <div key={tbl} style={{ marginBottom: 5 }}>
                <button onClick={() => setActiveTable(tbl)} style={{
                  width: "100%", textAlign: "left",
                  background: activeTable === tbl ? "linear-gradient(135deg, rgba(34, 211, 238, 0.14), rgba(167, 139, 250, 0.14))" : "transparent",
                  border: `1px solid ${activeTable === tbl ? C.accent : C.border}`,
                  borderRadius: 12, padding: "6px 10px", cursor: "pointer",
                  color: activeTable === tbl ? C.text : C.dim,
                  fontSize: 10, fontFamily: "inherit", transition: "all 0.15s",
                }}>▸ {tbl}</button>
                {activeTable === tbl && (
                  <div style={{ paddingLeft: 8, marginTop: 2 }}>
                    {cols.map(col => (
                      <div key={col} style={{ fontSize: 9, padding: "2px 6px",
                        color: col === "id" ? "#fbbf24" : col.endsWith("_id") ? "#60a5fa70" : C.muted }}>
                        <span style={{ color: C.border }}>│ </span>{col}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>

          {}
          

          {}
          <div style={{ padding: 18, flex: 1 }}>
            <div style={{ fontSize: 9, letterSpacing: 3, color: C.muted, textTransform: "uppercase", marginBottom: 10 }}>Istoric</div>
            {history.length === 0 && <div style={{ fontSize: 9, color: C.muted }}>Nicio interogare încă</div>}
            {history.map((h, i) => (
              <div key={h.ts} onClick={() => handleExample(h.q)} style={{
                fontSize: 9, color: i === 0 ? C.dim : C.muted,
                padding: "5px 8px", borderRadius: 4, cursor: "pointer",
                border: `1px solid ${i === 0 ? C.border : "transparent"}`,
                marginBottom: 3, transition: "all 0.15s", lineHeight: 1.5,
              }}
                onMouseEnter={e => e.currentTarget.style.color = C.dim}
                onMouseLeave={e => e.currentTarget.style.color = i === 0 ? C.dim : C.muted}>
                {h.q}
              </div>
            ))}
          </div>
        </div>

        {}
        <div style={{ padding: 24, overflowY: "auto" }}>

          {}
          <div style={{ marginBottom: 18 }}>
            <div style={{ fontSize: 9, letterSpacing: 3, color: C.muted, textTransform: "uppercase", marginBottom: 10 }}>
              Întrebare în limbaj natural
            </div>
            <div style={{ position: "relative" }}>
              <textarea
                ref={inputRef}
                value={query}
                onChange={e => setQuery(e.target.value)}
                onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleQuery(); } }}
                placeholder="ex: câți medici cardiologi sunt din cluj-napoca ?"
                rows={2}
                style={{
                  width: "100%", background: C.card,
                  border: `1px solid ${C.border}`, borderRadius: 18,
                  padding: "13px 160px 13px 16px", color: C.text,
                  fontSize: 14, fontFamily: "inherit", resize: "none",
                  transition: "border-color 0.15s",
                }}
                onFocus={e => e.target.style.borderColor = C.accent}
                onBlur={e => e.target.style.borderColor = C.border}
              />
              <button
                onClick={() => handleQuery()}
                disabled={anyLoading || !query.trim()}
                style={{
                  position: "absolute", right: 12, top: "50%", transform: "translateY(-50%)",
                  background: anyLoading || !query.trim() ? C.border : "linear-gradient(135deg, #22d3ee, #8b5cf6)",
                  border: "none", borderRadius: 14, padding: "9px 18px",
                  color: "#fff", fontSize: 11, fontFamily: "inherit",
                  cursor: anyLoading || !query.trim() ? "not-allowed" : "pointer",
                  letterSpacing: 1, transition: "background 0.15s", whiteSpace: "nowrap",
                }}>
                {anyLoading ? "..." : "RULEAZĂ →"}
              </button>
            </div>
          </div>

          {}
          <div style={{ marginBottom: 22 }}>
            <div style={{ fontSize: 9, letterSpacing: 3, color: C.muted, textTransform: "uppercase", marginBottom: 10 }}>Exemple rapide</div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
              {EXAMPLES.map(ex => (
                <button key={ex} onClick={() => handleExample(ex)} style={{
                  background: "transparent", border: `1px solid ${C.border}`,
                  borderRadius: 999, padding: "4px 10px", color: C.muted,
                  fontSize: 10, fontFamily: "inherit", cursor: "pointer", transition: "all 0.15s",
                }}
                  onMouseEnter={e => { e.target.style.borderColor = C.accent; e.target.style.color = C.text; e.target.style.background = "rgba(34, 211, 238, 0.08)"; }}
                  onMouseLeave={e => { e.target.style.borderColor = C.border; e.target.style.color = C.muted; e.target.style.background = "transparent"; }}>
                  {ex}
                </button>
              ))}
            </div>
          </div>

          {}
          <div>
            <div style={{ fontSize: 9, letterSpacing: 3, color: C.muted, textTransform: "uppercase", marginBottom: 12 }}>
              Rezultate comparative — {MODELS.length} metode
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
              {MODELS.map(m => (
                <ModelCard
                  key={m.id}
                  model={m}
                  result={results[m.id]}
                  loading={!!loading[m.id]}
                  technique={technique}
                  onTechChange={setTechnique}
                />
              ))}
            </div>
          </div>
        </div>

        {}
        <div style={{ borderLeft: `1px solid ${C.border}`, padding: 18, background: C.surface, overflowY: "auto" }}>

          {}
          <div style={{ marginBottom: 24, paddingTop: 16, borderTop: `1px solid ${C.border}` }}>
            <div style={{ fontSize: 9, letterSpacing: 3, color: C.muted, textTransform: "uppercase", marginBottom: 12 }}>Metode</div>
            {MODELS.map(m => (
              <div key={m.id} style={{ marginBottom: 12 }}>
                <div style={{ display: "flex", gap: 7, alignItems: "center", marginBottom: 2 }}>
                  <span style={{ fontSize: 11 }}>{m.icon}</span>
                  <span style={{ fontSize: 10, color: m.color }}>{m.label}</span>
                </div>
                <div style={{ fontSize: 9, color: C.muted, paddingLeft: 20 }}>{m.desc}</div>
                {m.id === "prompt" && (
                  <div style={{ fontSize: 9, color: C.muted, paddingLeft: 20, marginTop: 2 }}>
                    → -few-shot
                  </div>
                )}
              </div>
            ))}
          </div>

          {}
          <div style={{ paddingTop: 16, borderTop: `1px solid ${C.border}` }}>
            <div style={{ fontSize: 9, letterSpacing: 3, color: C.muted, textTransform: "uppercase", marginBottom: 10 }}>Utilizare</div>
            <div style={{ fontSize: 9, color: C.muted, lineHeight: 2.0 }}>
              <div>① Scrie o întrebare</div>
              <div>② Apasă <span style={{ color: C.accent }}>RULEAZĂ</span></div>
              <div>③ Compară cele 6 metode</div>
              <div style={{ marginTop: 8, color: C.border }}>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}