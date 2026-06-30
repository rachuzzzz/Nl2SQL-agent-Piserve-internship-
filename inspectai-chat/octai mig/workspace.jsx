/* ============================================================
   Workspace — AI Chat: conversation flow, KPI cards, chart
   panel, results table, query input, thinking sequence.
   Exports to window: Workspace
   ============================================================ */

const TONE = {
  blue:  { c: "var(--accent)",     bg: "var(--accent-soft)" },
  amber: { c: "var(--viz-amber)",  bg: "color-mix(in srgb, var(--viz-amber) 16%, transparent)" },
  teal:  { c: "var(--viz-teal)",   bg: "color-mix(in srgb, var(--viz-teal) 16%, transparent)" },
  coral: { c: "var(--viz-coral)",  bg: "color-mix(in srgb, var(--viz-coral) 14%, transparent)" },
};

// Count-up that settles on the final value (animates the numbers inside any
// string: "85.52", "81.35–91.64", "−41%", "4 days"; leaves words intact).
function useCountUp(value, delay = 0) {
  const [out, setOut] = React.useState(value);
  React.useEffect(() => {
    const reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const animOff = document.documentElement.getAttribute("data-anim") === "off";
    const nums = String(value).match(/-?\d[\d,]*\.?\d*/g);
    if (reduce || animOff || !nums) { setOut(value); return; }
    let raf, start;
    const dur = 900;
    const targets = nums.map((n) => parseFloat(n.replace(/,/g, "")));
    const decimals = nums.map((n) => (n.includes(".") ? n.split(".")[1].length : 0));
    const tick = (ts) => {
      if (start == null) start = ts + delay;
      const k = Math.max(0, Math.min(1, (ts - start) / dur));
      const e = 1 - Math.pow(1 - k, 3);
      let idx = 0;
      const s = String(value).replace(/-?\d[\d,]*\.?\d*/g, () => {
        const v = targets[idx] * e;
        const d = decimals[idx];
        idx++;
        return (d ? v.toFixed(d) : Math.round(v).toLocaleString());
      });
      setOut(s);
      if (k < 1) raf = requestAnimationFrame(tick);
      else setOut(value);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [value]);
  return out;
}

function KpiCard({ kpi, index = 0 }) {
  const t = TONE[kpi.tone] || TONE.blue;
  const shown = useCountUp(kpi.value, index * 90);
  const [hov, setHov] = React.useState(false);
  return (
    <div onMouseEnter={() => setHov(true)} onMouseLeave={() => setHov(false)}
      style={{ flex: "1 1 0", minWidth: 150, background: "var(--surface)", border: "1px solid var(--border)",
      borderRadius: 16, padding: "16px 18px", boxShadow: hov ? "var(--shadow-md)" : "var(--shadow-sm)",
      transform: hov ? "translateY(-3px)" : "none", transition: "transform .2s var(--ease), box-shadow .2s var(--ease)",
      animation: `om-rise .5s var(--ease) ${index * 0.07}s both` }}>
      <div style={{ display: "flex", alignItems: "center", gap: 11, marginBottom: 12 }}>
        <span style={{ width: 34, height: 34, borderRadius: 10, background: t.bg, display: "grid",
          placeItems: "center", transform: hov ? "scale(1.08) rotate(-4deg)" : "none", transition: "transform .25s var(--ease)" }}>
          <Icon name={kpi.icon} size={17} color={t.c} />
        </span>
        <span className="mono" style={{ fontSize: 10, fontWeight: 700, letterSpacing: ".08em",
          color: "var(--text-3)", textTransform: "uppercase", lineHeight: 1.3 }}>{kpi.label}</span>
      </div>
      <div className="mono" style={{ fontSize: 27, fontWeight: 700, color: "var(--text)", letterSpacing: "-.01em" }}>
        {shown}
      </div>
      <span style={{ display: "inline-block", marginTop: 9, fontSize: 10.5, fontWeight: 700,
        letterSpacing: ".05em", color: t.c, background: t.bg, padding: "3px 9px", borderRadius: 999, whiteSpace: "nowrap" }}>
        {kpi.tag}
      </span>
    </div>
  );
}

const CHART_TYPES = [
  { id: "line", label: "Line", icon: "axes" },
  { id: "area", label: "Area", icon: "report" },
  { id: "bar",  label: "Bar",  icon: "dashboard" },
];

function ZoomBtn({ expanded, onToggle }) {
  return (
    <button onClick={onToggle} title={expanded ? "Restore split view" : "Zoom — expand & push the other panel down"}
      style={{ width: 32, height: 32, borderRadius: 9, display: "grid", placeItems: "center",
        border: `1px solid ${expanded ? "var(--accent)" : "var(--border)"}`,
        background: expanded ? "var(--accent-soft)" : "var(--surface)", transition: "all .15s var(--ease)" }}>
      <Icon name={expanded ? "shrink" : "expand"} size={15} color={expanded ? "var(--accent)" : "var(--text-2)"} />
    </button>
  );
}

function ChartPanel({ ex, expanded, onToggle }) {
  const [type, setType] = React.useState(ex.chart.type);
  const [nonce, setNonce] = React.useState(0);
  React.useEffect(() => { setType(ex.chart.type); }, [ex.id]);
  const height = expanded ? 420 : 248;
  return (
    <div style={{ width: "100%", height: "100%", background: "var(--surface)", border: "1px solid var(--border)",
      borderRadius: 16, padding: 16, boxShadow: expanded ? "var(--shadow-md)" : "var(--shadow-sm)",
      transition: "box-shadow .35s var(--ease)" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14, flexWrap: "wrap" }}>
        <span style={{ width: 30, height: 30, borderRadius: 9, background: "var(--accent-soft)",
          display: "grid", placeItems: "center" }}>
          <Icon name="sparkle" size={16} color="var(--accent)" />
        </span>
        <div style={{ display: "flex", gap: 4, background: "var(--surface-2)", borderRadius: 999, padding: 3 }}>
          {CHART_TYPES.map((ct) => (
            <button key={ct.id} onClick={() => setType(ct.id)} style={{
              display: "inline-flex", alignItems: "center", gap: 6, border: "none", borderRadius: 999,
              padding: "6px 13px", fontSize: 12.5, fontWeight: 600,
              background: type === ct.id ? "var(--surface)" : "transparent",
              color: type === ct.id ? "var(--accent)" : "var(--text-2)",
              boxShadow: type === ct.id ? "var(--shadow-sm)" : "none", transition: "all .15s var(--ease)" }}>
              <Icon name={ct.icon} size={13} />{ct.label}
              {ex.chart.best === ct.id && (
                <span style={{ fontSize: 9, fontWeight: 700, letterSpacing: ".05em", color: "var(--viz-teal)",
                  background: "color-mix(in srgb, var(--viz-teal) 16%, transparent)", padding: "1px 5px",
                  borderRadius: 4, marginLeft: 2 }}>BEST</span>
              )}
            </button>
          ))}
        </div>
        <div style={{ marginLeft: "auto", display: "flex", gap: 4 }}>
          <button onClick={() => setNonce((n) => n + 1)} title="Replay" style={{ width: 32, height: 32, borderRadius: 9,
            border: "1px solid var(--border)", background: "var(--surface)", display: "grid", placeItems: "center" }}>
            <Icon name="refresh" size={15} color="var(--text-2)" />
          </button>
          <ZoomBtn expanded={expanded} onToggle={onToggle} />
        </div>
      </div>
      <div key={nonce}>
        <OptimaXChart type={type} x={ex.chart.x} series={ex.chart.series} height={height} />
      </div>
    </div>
  );
}

function ResultsTable({ table, expanded, onToggle }) {
  return (
    <div style={{ width: "100%", height: "100%", background: "var(--surface)", border: "1px solid var(--border)",
      borderRadius: 16, overflow: "hidden", boxShadow: expanded ? "var(--shadow-md)" : "var(--shadow-sm)",
      display: "flex", flexDirection: "column", transition: "box-shadow .35s var(--ease)" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "12px 16px",
        borderBottom: "1px solid var(--border)" }}>
        <span style={{ width: 30, height: 30, borderRadius: 9, background: "var(--accent-soft)",
          display: "grid", placeItems: "center" }}>
          <Icon name="table" size={15} color="var(--accent)" />
        </span>
        <span style={{ fontSize: 13.5, fontWeight: 700, color: "var(--text)" }}>Result set</span>
        <span className="mono" style={{ fontSize: 11, color: "var(--text-3)" }}>{table.cols.length} cols</span>
        <div style={{ marginLeft: "auto" }}><ZoomBtn expanded={expanded} onToggle={onToggle} /></div>
      </div>
      <div style={{ overflow: "auto", flex: 1, maxHeight: expanded ? "none" : 322 }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
          <thead>
            <tr>
              {table.cols.map((c) => (
                <th key={c} style={{ textAlign: "left", padding: "13px 16px", whiteSpace: "nowrap",
                  fontSize: 10.5, fontWeight: 700, letterSpacing: ".07em", color: "var(--accent)",
                  textTransform: "uppercase", borderBottom: "1.5px solid var(--border-2)",
                  background: "var(--surface-2)", position: "sticky", top: 0, zIndex: 1 }} className="mono">{c}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {table.rows.map((r, ri) => (
              <tr key={ri} style={{ background: ri % 2 ? "var(--surface-2)" : "transparent",
                animation: `om-rise .45s var(--ease) ${Math.min(ri * 0.05, 0.5)}s both` }}>
                {r.map((cell, ci) => (
                  <td key={ci} className={ci === 0 ? "mono" : ""} style={{ padding: "11px 16px", whiteSpace: "nowrap",
                    color: ci === 0 ? "var(--text-3)" : "var(--text)", fontWeight: ci === 0 ? 500 : 600,
                    borderBottom: "1px solid var(--border)", fontSize: ci === 0 ? 11.5 : 13 }}>{cell}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "10px 16px",
        borderTop: "1px solid var(--border)" }}>
        <span style={{ fontSize: 12, color: "var(--text-3)" }}>
          1–{table.rows.length} of <b style={{ color: "var(--text)" }}>{table.total}</b> rows
        </span>
        <button className="pill subtle" style={{ marginLeft: "auto", padding: "6px 13px", fontSize: 12 }}>
          <Icon name="download" size={13} /> Export CSV
        </button>
      </div>
    </div>
  );
}

function AiResult({ ex }) {
  // focus: null = split view · 'chart' / 'table' = that panel zoomed full-width on top
  const [focus, setFocus] = React.useState(null);
  React.useEffect(() => { setFocus(null); }, [ex.id]);
  const toggle = (k) => setFocus((f) => (f === k ? null : k));
  const any = focus !== null;
  const cell = (key) => ({
    flexGrow: 1, flexShrink: 1,
    flexBasis: any ? "100%" : "calc(50% - 8px)",
    order: focus === key ? 0 : any ? 2 : 1,
    minWidth: 288,
  });
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <div style={{ fontSize: 15, lineHeight: 1.6, color: "var(--text)", maxWidth: 760, fontWeight: 500 }}>
        {ex.summary}
      </div>
      <div style={{ display: "flex", gap: 14, flexWrap: "wrap" }}>
        {ex.kpis.map((k, i) => <KpiCard key={i} kpi={k} index={i} />)}
      </div>
      <div style={{ display: "flex", gap: 16, flexWrap: "wrap", alignItems: "stretch" }}>
        <div key={"chart-" + (focus || "split")} style={{ ...cell("chart"), animation: "om-rise .42s var(--ease)" }}>
          <ChartPanel ex={ex} expanded={focus === "chart"} onToggle={() => toggle("chart")} />
        </div>
        <div key={"table-" + (focus || "split")} style={{ ...cell("table"), animation: "om-rise .42s var(--ease)" }}>
          <ResultsTable table={ex.table} expanded={focus === "table"} onToggle={() => toggle("table")} />
        </div>
      </div>
    </div>
  );
}

function Avatar({ kind }) {
  if (kind === "user") {
    return <div style={{ width: 34, height: 34, borderRadius: 11, flex: "0 0 auto", background: "var(--rail)",
      color: "var(--rail-text)", display: "grid", placeItems: "center", fontSize: 12, fontWeight: 700 }} className="mono">QA</div>;
  }
  return <div style={{ width: 34, height: 34, borderRadius: 11, flex: "0 0 auto", background: "var(--accent-soft)",
    border: "1px solid var(--accent-soft-border)", display: "grid", placeItems: "center" }}>
    <OptimaXMark size={20} mode="static" stroke={8} glow={false} />
  </div>;
}

function UserMsg({ text, time }) {
  return (
    <div style={{ display: "flex", justifyContent: "flex-end", gap: 12, animation: "om-rise .3s var(--ease)" }}>
      <div style={{ textAlign: "right", maxWidth: 620 }}>
        <div style={{ display: "inline-block", background: "var(--rail)", color: "var(--rail-text)",
          padding: "13px 18px", borderRadius: "18px 18px 4px 18px", fontSize: 15, fontWeight: 600, lineHeight: 1.45 }}>
          {text}
        </div>
        <div style={{ fontSize: 11, color: "var(--text-3)", marginTop: 6, marginRight: 4 }} className="mono">{time}</div>
      </div>
      <Avatar kind="user" />
    </div>
  );
}

const STAGE_LABELS = ["Parsing your question", "Resolving schema", "Generating SQL", "Validating & executing", "Summarizing results"];

function Workspace({ session, convoSeed, onActiveExChange }) {
  const D = window.OPTIMAX_DATA;
  const [messages, setMessages] = React.useState([]);
  const [thinking, setThinking] = React.useState(null); // {q, stage}
  const scrollRef = React.useRef(null);
  const timers = React.useRef([]);

  // load a conversation seed as a finished exchange
  React.useEffect(() => {
    clearTimers();
    setThinking(null);
    if (convoSeed) {
      const ex = D.EX[convoSeed.exId];
      setMessages([{ role: "user", text: ex.q, time: timeNow() }, { role: "ai", ex }]);
      onActiveExChange && onActiveExChange(ex);
    } else {
      setMessages([]);
      onActiveExChange && onActiveExChange(null);
    }
  }, [convoSeed]);

  React.useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, [messages, thinking]);

  function clearTimers() { timers.current.forEach(clearTimeout); timers.current = []; }

  function runQuery(q) {
    const exId = matchExample(q);
    const ex = exId ? D.EX[exId] : synthExample(q, D);
    setMessages((m) => [...m, { role: "user", text: q, time: timeNow() }]);
    setThinking({ q, stage: 0 });
    const errStep = ex.debug.sqlErrors > 0 ? 3 : -1;
    const stageDur = [700, 650, 1100, errStep === 3 ? 1300 : 800, 700];
    let acc = 0;
    STAGE_LABELS.forEach((_, i) => {
      acc += stageDur[i];
      timers.current.push(setTimeout(() => setThinking((t) => t && { ...t, stage: i + 1 }), acc));
    });
    timers.current.push(setTimeout(() => {
      setThinking(null);
      setMessages((m) => [...m, { role: "ai", ex }]);
      onActiveExChange && onActiveExChange(ex);
    }, acc + 350));
  }

  const empty = messages.length === 0 && !thinking;

  return (
    <div style={{ position: "relative", height: "100%", display: "flex", flexDirection: "column" }}>
      <BgMark />
      <div ref={scrollRef} style={{ flex: 1, overflowY: "auto", position: "relative", zIndex: 1 }}>
        <div style={{ maxWidth: 1180, margin: "0 auto", padding: "28px 32px 8px",
          display: "flex", flexDirection: "column", gap: 26 }}>
          {empty && <EmptyState onPick={runQuery} D={D} />}
          {messages.map((m, i) =>
            m.role === "user"
              ? <UserMsg key={i} text={m.text} time={m.time} />
              : <div key={i} style={{ display: "flex", gap: 12, alignItems: "flex-start" }}>
                  <Avatar kind="ai" />
                  <div style={{ flex: 1, minWidth: 0 }}><AiResult ex={m.ex} /></div>
                </div>
          )}
          {thinking && (
            <InlineThinking stages={STAGE_LABELS} activeIndex={thinking.stage} />
          )}
        </div>
      </div>
      <QueryBar onSubmit={runQuery} disabled={!!thinking} hasMessages={!empty} D={D} />
    </div>
  );
}

function EmptyState({ onPick, D }) {
  return (
    <div style={{ padding: "52px 0 20px", textAlign: "center", animation: "om-rise .4s var(--ease)" }}>
      <div style={{ display: "inline-grid", placeItems: "center", marginBottom: 22 }}>
        <OptimaXMark size={68} mode="draw" stroke={7} />
      </div>
      <h1 style={{ margin: "0 0 10px", fontSize: 34, fontWeight: 800, letterSpacing: "-.02em",
        color: "var(--text)" }}>Ask your data anything.</h1>
      <p style={{ margin: "0 auto 30px", maxWidth: 520, fontSize: 16, color: "var(--text-2)", lineHeight: 1.55 }}>
        OctaAI turns plain English into validated SQL across your inspections, audits,
        permits & EHS workflows — with every step traced.
      </p>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 10, justifyContent: "center", maxWidth: 720, margin: "0 auto" }}>
        {D.SUGGESTIONS.map((s, i) => (
          <button key={s.ex} onClick={() => onPick(s.q)} style={{ display: "inline-flex", alignItems: "center", gap: 9,
            background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 999, padding: "10px 16px",
            fontSize: 13.5, fontWeight: 600, color: "var(--text)", boxShadow: "var(--shadow-sm)",
            transition: "all .15s var(--ease)", animation: `om-pop .5s var(--ease) ${0.15 + i * 0.07}s both` }}
            onMouseEnter={(e) => { e.currentTarget.style.borderColor = "var(--accent)"; e.currentTarget.style.color = "var(--accent)"; e.currentTarget.style.transform = "translateY(-2px)"; }}
            onMouseLeave={(e) => { e.currentTarget.style.borderColor = "var(--border)"; e.currentTarget.style.color = "var(--text)"; e.currentTarget.style.transform = "none"; }}>
            <Icon name={s.icon} size={15} color="var(--accent)" />{s.q}
          </button>
        ))}
      </div>
    </div>
  );
}

function QueryBar({ onSubmit, disabled, hasMessages, D }) {
  const [val, setVal] = React.useState("");
  const taRef = React.useRef(null);
  function submit() {
    const q = val.trim();
    if (!q || disabled) return;
    onSubmit(q);
    setVal("");
    if (taRef.current) taRef.current.style.height = "auto";
  }
  function onKey(e) {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); }
  }
  function grow(e) {
    setVal(e.target.value);
    e.target.style.height = "auto";
    e.target.style.height = Math.min(140, e.target.scrollHeight) + "px";
  }
  return (
    <div style={{ position: "relative", zIndex: 2, padding: "12px 32px 22px",
      background: "linear-gradient(to top, var(--bg) 62%, transparent)" }}>
      <div style={{ maxWidth: 1180, margin: "0 auto" }}>
        {hasMessages && (
          <div style={{ display: "flex", gap: 8, marginBottom: 10, flexWrap: "wrap" }}>
            {D.SUGGESTIONS.slice(0, 3).map((s) => (
              <button key={s.ex} onClick={() => onSubmit(s.q)} disabled={disabled} style={{ display: "inline-flex",
                alignItems: "center", gap: 6, background: "var(--surface-2)", border: "1px solid var(--border)",
                borderRadius: 999, padding: "6px 12px", fontSize: 12, fontWeight: 600, color: "var(--text-2)" }}>
                <Icon name={s.icon} size={12} color="var(--accent)" />{s.q}
              </button>
            ))}
          </div>
        )}
        <div style={{ display: "flex", alignItems: "flex-end", gap: 10, background: "var(--surface)",
          border: "1.5px solid var(--border-2)", borderRadius: 22, padding: "8px 8px 8px 20px",
          boxShadow: "var(--shadow-md)" }}>
          <textarea ref={taRef} value={val} onChange={grow} onKeyDown={onKey} rows={1}
            placeholder="Ask a question or type / for commands (/chart, /export, /sql…)"
            style={{ flex: 1, border: "none", outline: "none", resize: "none", background: "transparent",
              color: "var(--text)", fontSize: 15, lineHeight: 1.5, padding: "9px 0", maxHeight: 140,
              fontFamily: "var(--font-sans)" }} />
          <button title="Voice" style={{ width: 40, height: 40, borderRadius: "50%", border: "1px solid var(--border)",
            background: "var(--surface-2)", display: "grid", placeItems: "center", flex: "0 0 auto" }}>
            <Icon name="mic" size={17} color="var(--text-2)" />
          </button>
          <button onClick={submit} disabled={disabled || !val.trim()} title="Send" style={{ width: 40, height: 40,
            borderRadius: "50%", border: "none", flex: "0 0 auto", display: "grid", placeItems: "center",
            background: val.trim() && !disabled ? "var(--accent)" : "var(--surface-3)",
            boxShadow: val.trim() && !disabled ? "0 4px 14px color-mix(in srgb, var(--accent) 38%, transparent)" : "none",
            transition: "all .15s var(--ease)", cursor: val.trim() && !disabled ? "pointer" : "default" }}>
            <Icon name="send" size={17} color={val.trim() && !disabled ? "#fff" : "var(--text-3)"} />
          </button>
        </div>
        <div style={{ textAlign: "center", marginTop: 9, fontSize: 11, color: "var(--text-3)" }}>
          OctaAI validates every query before execution · qwen3.5:9b + qwen2.5-coder
        </div>
      </div>
    </div>
  );
}

// ---- query matching + synthesis ----
function matchExample(q) {
  const D = window.OPTIMAX_DATA;
  const t = q.toLowerCase();
  const has = (...ws) => ws.some((w) => t.includes(w));
  if (has("average", "avg") && has("score", "inspection")) return "avg_score";
  if (has("corrective", "capa", "action")) return "open_capa";
  if (has("permit")) return "permits";
  if (has("incident", "safety", "severity")) return "incidents";
  if (has("inspector", "auditor", "top")) return "top_inspectors";
  // exact match to a suggestion
  const exact = D.SUGGESTIONS.find((s) => s.q.toLowerCase() === t);
  return exact ? exact.ex : null;
}

function synthExample(q, D) {
  // deterministic believable fallback for free-text questions
  const table = /audit/i.test(q) ? "audit" : /permit/i.test(q) ? "permit"
    : /incident|safety/i.test(q) ? "incident" : /finding/i.test(q) ? "finding" : "inspection_report";
  const cats = ["US Gulf", "NW Europe", "APAC", "MEA", "LATAM"];
  const data = cats.map((_, i) => Math.round(60 + Math.sin(i * 1.7) * 22 + (5 - i) * 6));
  const total = data.reduce((a, b) => a + b, 0);
  return {
    id: "synth", q,
    summary: `Here's what OctaAI found for “${q}”. Results are grouped by region across the ${table.replace("_", " ")} table — ${cats[0]} leads with ${data[0]} records out of ${total} total.`,
    sql: `SELECT
  s.region,
  COUNT(*)                       AS records,
  ROUND(AVG(t.inspection_score)
        ::numeric, 1)            AS avg_value
FROM ${table} t
JOIN site s ON s.id = t.site_id
WHERE t.active = true
GROUP BY s.region
ORDER BY records DESC;`,
    reasoning: `Parsed the request as an aggregation over ${table}. Joined to site for regional grouping, counted rows and averaged the primary metric, and ordered by volume. Schema retriever matched ${table} with high confidence.`,
    kpis: [
      { label: "TOTAL RECORDS", value: "" + total, tag: "ALL REGIONS", icon: "rows", tone: "blue" },
      { label: "TOP REGION", value: cats[0], tag: "" + data[0], icon: "site", tone: "teal" },
      { label: "REGIONS", value: "" + cats.length, tag: "GROUPED", icon: "range", tone: "amber" },
    ],
    chart: { type: "bar", best: "bar", x: cats, series: [{ name: "Records", color: "var(--viz-blue)", data }] },
    table: { cols: ["REGION", "RECORDS", "AVG VALUE"], rows: cats.map((c, i) => [c, "" + data[i], (70 + i * 2.3).toFixed(1)]), total: cats.length },
    debug: { calls: 1, tokens: 14820, prompt: 14510, completion: 310, responseTime: "3.74s",
      llmCalls: 1, sqlErrors: 0, status: "ok",
      banner: { tone: "ok", text: "Executed cleanly on first attempt" },
      trace: [
        { step: "Parse intent", model: "qwen3.5:9b", ms: 360, status: "ok" },
        { step: "Resolve schema", model: "retriever", ms: 210, status: "ok", note: `matched ${table}` },
        { step: "Generate SQL", model: "qwen2.5-coder", ms: 2380, status: "ok" },
        { step: "Validate SQL", model: "validator", ms: 80, status: "ok" },
        { step: "Execute", model: "postgres", ms: 330, status: "ok", note: `${cats.length} rows · 15ms` },
        { step: "Summarize", model: "qwen3.5:9b", ms: 480, status: "ok" },
      ] },
  };
}

function timeNow() {
  return new Date().toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
}

Object.assign(window, { Workspace, useCountUp });
