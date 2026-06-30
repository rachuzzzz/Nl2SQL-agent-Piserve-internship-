/* ============================================================
   DebugPanel — right rail: pipeline trace, SQL, reasoning,
   performance metrics, session details.
   Exports to window: DebugPanel, SqlBlock
   ============================================================ */

// --- naive SQL syntax highlighter ---
const SQL_KW = /\b(SELECT|FROM|WHERE|AND|OR|NOT|NULL|IS|IN|AS|JOIN|LEFT|RIGHT|INNER|ON|GROUP|ORDER|BY|LIMIT|HAVING|DISTINCT|COUNT|AVG|SUM|MIN|MAX|ROUND|DATE_TRUNC|TO_CHAR|FILTER|INTERVAL|CURRENT_DATE|BETWEEN|DESC|ASC|TRUE|FALSE)\b/gi;

function highlightSql(sql) {
  const out = [];
  let key = 0;
  sql.split("\n").forEach((line, li) => {
    const parts = [];
    let last = 0;
    // strings
    const tokenRe = /'[^']*'|--.*$|\b\d+(\.\d+)?\b|[a-zA-Z_][a-zA-Z0-9_.]*/g;
    let m;
    while ((m = tokenRe.exec(line)) !== null) {
      if (m.index > last) parts.push(line.slice(last, m.index));
      const tok = m[0];
      let cls = null;
      if (/^'/.test(tok)) cls = "s";
      else if (/^--/.test(tok)) cls = "c";
      else if (/^\d/.test(tok)) cls = "n";
      else if (SQL_KW.test(tok)) cls = "k";
      SQL_KW.lastIndex = 0;
      if (cls) parts.push(<span key={key++} className={"sql-" + cls}>{tok}</span>);
      else parts.push(tok);
      last = m.index + tok.length;
    }
    if (last < line.length) parts.push(line.slice(last));
    out.push(<div key={li} className="sql-line">{parts.length ? parts : "\u00A0"}</div>);
  });
  return out;
}

function SqlBlock({ sql, compact }) {
  const [copied, setCopied] = React.useState(false);
  const copy = () => {
    navigator.clipboard && navigator.clipboard.writeText(sql);
    setCopied(true);
    setTimeout(() => setCopied(false), 1400);
  };
  return (
    <div style={{ position: "relative", borderRadius: 13, overflow: "hidden",
      background: "var(--code-bg)", border: "1px solid var(--rail-border)" }}>
      <button onClick={copy} title="Copy SQL" style={{
        position: "absolute", top: 8, right: 8, zIndex: 2,
        display: "inline-flex", alignItems: "center", gap: 5,
        background: copied ? "var(--ok)" : "rgba(255,255,255,.08)",
        color: "#fff", border: "1px solid rgba(255,255,255,.14)",
        borderRadius: 8, padding: "4px 8px", fontSize: 11, fontWeight: 600,
      }}>
        <Icon name={copied ? "check" : "copy"} size={12} color="#fff" />
        {copied ? "Copied" : "Copy"}
      </button>
      <pre className="mono" style={{
        margin: 0, padding: compact ? "12px 14px" : "14px 16px", fontSize: 12, lineHeight: 1.65,
        color: "var(--code-text)", overflowX: "auto",
      }}>{highlightSql(sql)}</pre>
    </div>
  );
}

function Section({ title, dot, right, children, defaultOpen = true, mono = true }) {
  const [open, setOpen] = React.useState(defaultOpen);
  return (
    <div style={{ borderTop: "1px solid var(--border)" }}>
      <button onClick={() => setOpen((o) => !o)} style={{
        width: "100%", display: "flex", alignItems: "center", gap: 8,
        background: "none", border: "none", padding: "13px 16px", color: "var(--text)",
      }}>
        {dot && <span style={{ width: 7, height: 7, borderRadius: "50%", background: dot }} />}
        <span className={mono ? "mono" : ""} style={{ fontSize: 11, fontWeight: 700,
          letterSpacing: ".11em", textTransform: "uppercase", color: "var(--text-2)" }}>{title}</span>
        <span style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 8 }}>
          {right}
          <Icon name={open ? "chevUp" : "chevDown"} size={14} color="var(--text-3)" />
        </span>
      </button>
      {open && <div style={{ padding: "0 16px 16px" }}>{children}</div>}
    </div>
  );
}

const STATUS_COLOR = { ok: "var(--ok)", warn: "var(--warn)", err: "var(--err)" };

function MetricTile({ label, value, tone }) {
  return (
    <div style={{ background: "var(--surface-2)", border: "1px solid var(--border)",
      borderRadius: 12, padding: "11px 13px" }}>
      <div className="mono" style={{ fontSize: 9.5, fontWeight: 700, letterSpacing: ".09em",
        color: "var(--text-3)", textTransform: "uppercase", marginBottom: 5 }}>{label}</div>
      <div className="mono" style={{ fontSize: 19, fontWeight: 700,
        color: tone === "err" ? "var(--err)" : tone === "accent" ? "var(--accent)" : "var(--text)" }}>{value}</div>
    </div>
  );
}

function DebugPanel({ ex, session }) {
  const d = ex.debug;
  const totalMs = d.trace.reduce((a, s) => a + s.ms, 0);
  return (
    <div style={{ height: "100%", display: "flex", flexDirection: "column",
      background: "var(--surface)", borderLeft: "1px solid var(--border)" }}>
      {/* header */}
      <div style={{ display: "flex", alignItems: "center", gap: 9, padding: "14px 16px",
        borderBottom: "1px solid var(--border)" }}>
        <span style={{ position: "relative", display: "inline-flex" }}>
          <span style={{ width: 9, height: 9, borderRadius: "50%", background: "var(--err)" }} />
          <span style={{ position: "absolute", inset: 0, borderRadius: "50%", background: "var(--err)",
            animation: "om-pulse 1.4s ease-in-out infinite" }} />
        </span>
        <span className="mono" style={{ fontSize: 12, fontWeight: 700, letterSpacing: ".14em" }}>DEBUG</span>
        <span className="mono" style={{ marginLeft: "auto", fontSize: 11, color: "var(--text-3)" }}>
          {d.calls} {d.calls === 1 ? "call" : "calls"} · {fmtTok(d.tokens)} tok
        </span>
      </div>

      <div style={{ flex: 1, overflowY: "auto" }}>
        {/* banner */}
        <div style={{ padding: "14px 16px 4px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 9, padding: "10px 13px", borderRadius: 11,
            background: d.banner.tone === "ok" ? "color-mix(in srgb, var(--ok) 12%, transparent)"
              : "color-mix(in srgb, var(--warn) 14%, transparent)",
            border: `1px solid ${d.banner.tone === "ok" ? "color-mix(in srgb, var(--ok) 32%, transparent)" : "color-mix(in srgb, var(--warn) 38%, transparent)"}` }}>
            <Icon name={d.banner.tone === "ok" ? "check" : "warn"} size={15}
                  color={d.banner.tone === "ok" ? "var(--ok)" : "var(--warn)"} />
            <span className="mono" style={{ fontSize: 11.5, fontWeight: 600,
              color: d.banner.tone === "ok" ? "var(--ok)" : "color-mix(in srgb, var(--warn) 75%, var(--text))" }}>
              {d.banner.text}
            </span>
          </div>
        </div>

        {/* pipeline trace */}
        <Section title="Pipeline Trace" dot="var(--accent)" right={
          <span className="mono" style={{ fontSize: 10.5, color: "var(--text-3)" }}>{(totalMs/1000).toFixed(1)}s</span>
        }>
          <div style={{ position: "relative", paddingLeft: 6 }}>
            {d.trace.map((s, i) => (
              <div key={i} style={{ display: "flex", gap: 11, paddingBottom: i === d.trace.length - 1 ? 0 : 13, position: "relative",
                animation: `om-rise .4s var(--ease) ${i * 0.06}s both` }}>
                {i < d.trace.length - 1 && (
                  <span style={{ position: "absolute", left: 6, top: 16, bottom: -2, width: 2,
                    background: "var(--border-2)" }} />
                )}
                <span style={{ flex: "0 0 auto", width: 14, height: 14, marginTop: 2, borderRadius: "50%",
                  background: STATUS_COLOR[s.status], display: "grid", placeItems: "center", zIndex: 1,
                  boxShadow: "0 0 0 3px var(--surface)" }}>
                  {s.status === "err"
                    ? <span style={{ width: 5, height: 5, background: "#fff", borderRadius: 1 }} />
                    : <Icon name="check" size={9} color="#fff" stroke={3} />}
                </span>
                <div style={{ minWidth: 0, flex: 1 }}>
                  <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
                    <span style={{ fontSize: 13, fontWeight: 600, color: "var(--text)" }}>{s.step}</span>
                    <span className="mono" style={{ marginLeft: "auto", fontSize: 10.5,
                      color: s.status === "err" ? "var(--err)" : "var(--text-3)" }}>{s.ms}ms</span>
                  </div>
                  <div style={{ display: "flex", alignItems: "center", gap: 7, marginTop: 3 }}>
                    <span className="mono" style={{ fontSize: 10, padding: "1px 6px", borderRadius: 5,
                      background: "var(--surface-3)", color: "var(--text-2)" }}>{s.model}</span>
                    {s.note && <span style={{ fontSize: 11, color: s.status === "err" ? "var(--err)" : "var(--text-3)",
                      overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{s.note}</span>}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </Section>

        {/* generated SQL */}
        <Section title="Generated SQL" dot="var(--viz-teal)">
          <SqlBlock sql={ex.sql} />
        </Section>

        {/* reasoning */}
        <Section title="Model Reasoning" dot="var(--viz-violet)">
          <div style={{ fontSize: 13, lineHeight: 1.62, fontStyle: "italic", color: "var(--text-2)",
            borderLeft: "2.5px solid var(--accent-soft-border)", paddingLeft: 12 }}>
            “{ex.reasoning}”
          </div>
        </Section>

        {/* performance */}
        <Section title="Performance" dot="var(--ok)">
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 9 }}>
            <MetricTile label="Response Time" value={d.responseTime} tone="accent" />
            <MetricTile label="LLM Calls" value={d.llmCalls} />
            <MetricTile label="Prompt Tokens" value={d.prompt.toLocaleString()} />
            <MetricTile label="Completion" value={d.completion} />
            <MetricTile label="SQL Errors" value={d.sqlErrors} tone={d.sqlErrors ? "err" : undefined} />
            <MetricTile label="Total Tokens" value={d.tokens.toLocaleString()} />
          </div>
        </Section>

        {/* session details */}
        <Section title="Session Details" dot="var(--text-3)" defaultOpen={false}>
          <div style={{ display: "flex", flexDirection: "column", gap: 0,
            border: "1px solid var(--border)", borderRadius: 12, overflow: "hidden" }}>
            {[
              ["Session ID", session.id, true],
              ["User", session.user],
              ["Connected DB", session.db, true],
              ["Model", session.model],
              ["Region", session.region],
              ["Started", session.started],
              ["Queries this session", session.queries],
            ].map(([k, v, mono], i) => (
              <div key={k} style={{ display: "flex", alignItems: "center", gap: 10, padding: "9px 12px",
                borderTop: i ? "1px solid var(--border)" : "none", background: i % 2 ? "var(--surface-2)" : "transparent" }}>
                <span style={{ fontSize: 12, color: "var(--text-3)", fontWeight: 500 }}>{k}</span>
                <span className={mono ? "mono" : ""} style={{ marginLeft: "auto", fontSize: 12,
                  fontWeight: 600, color: "var(--text)", textAlign: "right" }}>{v}</span>
              </div>
            ))}
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 7, marginTop: 11 }}>
            <span style={{ width: 7, height: 7, borderRadius: "50%", background: "var(--ok)",
              boxShadow: "0 0 0 3px color-mix(in srgb, var(--ok) 22%, transparent)" }} />
            <span style={{ fontSize: 11.5, color: "var(--text-2)", fontWeight: 600 }}>
              Live · pipeline healthy
            </span>
          </div>
        </Section>
      </div>
    </div>
  );
}

function fmtTok(n) { return n >= 1000 ? (n / 1000).toFixed(n % 1000 < 100 ? 1 : 1).replace(/\.0$/, "") + "k" : "" + n; }

Object.assign(window, { DebugPanel, SqlBlock });
