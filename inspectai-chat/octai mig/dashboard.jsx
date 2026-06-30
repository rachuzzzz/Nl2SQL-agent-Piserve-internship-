/* ============================================================
   DashboardView · SavedView · SchemaView · ConfigView
   Exports to window: DashboardView, SavedView, SchemaView, ConfigView
   ============================================================ */

function PageHead({ title, sub, action }) {
  return (
    <div style={{ display: "flex", alignItems: "flex-end", gap: 16, marginBottom: 26, flexWrap: "wrap" }}>
      <div>
        <h1 style={{ margin: "0 0 5px", fontSize: 30, fontWeight: 800, letterSpacing: "-.02em", color: "var(--text)" }}>{title}</h1>
        <p style={{ margin: 0, fontSize: 15, color: "var(--text-2)" }}>{sub}</p>
      </div>
      {action && <div style={{ marginLeft: "auto" }}>{action}</div>}
    </div>
  );
}

function Scroll({ children }) {
  return (
    <div style={{ height: "100%", overflowY: "auto", position: "relative" }}>
      <BgMark />
      <div style={{ maxWidth: 1180, margin: "0 auto", padding: "30px 32px 60px", position: "relative", zIndex: 1 }}>
        {children}
      </div>
    </div>
  );
}

function DashKpiTile({ k, index }) {
  const shown = (window.useCountUp ? window.useCountUp(k.value, index * 90) : k.value);
  const [hov, setHov] = React.useState(false);
  return (
    <div onMouseEnter={() => setHov(true)} onMouseLeave={() => setHov(false)}
      style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 18,
      padding: "20px 22px", boxShadow: hov ? "var(--shadow-md)" : "var(--shadow-sm)",
      transform: hov ? "translateY(-3px)" : "none", transition: "transform .2s var(--ease), box-shadow .2s var(--ease)",
      animation: `om-rise .5s var(--ease) ${index * 0.07}s both` }}>
      <div style={{ display: "flex", alignItems: "center", gap: 11, marginBottom: 14 }}>
        <span style={{ width: 36, height: 36, borderRadius: 11, background: "var(--accent-soft)",
          display: "grid", placeItems: "center", transform: hov ? "scale(1.08) rotate(-4deg)" : "none",
          transition: "transform .25s var(--ease)" }}><Icon name={k.icon} size={18} color="var(--accent)" /></span>
        <span className="mono" style={{ fontSize: 10, fontWeight: 700, letterSpacing: ".07em",
          color: "var(--text-3)", textTransform: "uppercase", lineHeight: 1.3 }}>{k.label}</span>
      </div>
      <div className="mono" style={{ fontSize: 32, fontWeight: 700, letterSpacing: "-.02em", color: "var(--text)" }}>{shown}</div>
      <div style={{ display: "flex", alignItems: "center", gap: 5, marginTop: 8 }}>
        <Icon name={k.up ? "arrowDown" : "arrowUp"} size={13} color={k.up ? "var(--ok)" : "var(--err)"} />
        <span style={{ fontSize: 12.5, fontWeight: 700, color: k.up ? "var(--ok)" : "var(--err)" }}>{k.delta}</span>
        <span style={{ fontSize: 12, color: "var(--text-3)" }}>vs last month</span>
      </div>
    </div>
  );
}

function DashboardView({ onRun }) {
  const D = window.OPTIMAX_DATA;
  return (
    <Scroll>
      <PageHead title="Operations overview" sub="Live EHS posture across all sites · updated 2 min ago"
        action={<button className="pill" onClick={() => onRun("avg_score")}><Icon name="sparkle" size={15} color="#fff" />Ask OctaAI</button>} />

      {/* KPI tiles */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 16, marginBottom: 22 }}>
        {D.DASH_KPIS.map((k, i) => <DashKpiTile key={i} k={k} index={i} />)}
      </div>

      {/* chart cards */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(380px, 1fr))", gap: 16, marginBottom: 22 }}>
        {[["avg_score", "Inspection score trend"], ["open_capa", "Open corrective actions by site"]].map(([id, title]) => {
          const ex = D.EX[id];
          return (
            <div key={id} style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 18,
              padding: 20, boxShadow: "var(--shadow-sm)" }}>
              <div style={{ display: "flex", alignItems: "center", marginBottom: 14 }}>
                <h3 style={{ margin: 0, fontSize: 16, fontWeight: 700, color: "var(--text)" }}>{title}</h3>
                <button onClick={() => onRun(id)} className="pill subtle" style={{ marginLeft: "auto", padding: "5px 12px", fontSize: 12 }}>
                  <Icon name="play" size={11} color="var(--accent)" />Open
                </button>
              </div>
              <OptimaXChart type={ex.chart.type} x={ex.chart.x} series={ex.chart.series} height={220} />
            </div>
          );
        })}
      </div>

      {/* recent queries */}
      <div style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 18, overflow: "hidden",
        boxShadow: "var(--shadow-sm)" }}>
        <div style={{ padding: "16px 22px", borderBottom: "1px solid var(--border)" }}>
          <h3 style={{ margin: 0, fontSize: 16, fontWeight: 700 }}>Recent queries</h3>
        </div>
        {D.SAVED.slice(0, 5).map((s, i) => (
          <button key={i} onClick={() => onRun(s.ex)} style={{ width: "100%", display: "flex", alignItems: "center",
            gap: 14, padding: "14px 22px", background: "transparent", border: "none",
            borderTop: i ? "1px solid var(--border)" : "none", textAlign: "left" }}
            onMouseEnter={(e) => e.currentTarget.style.background = "var(--surface-2)"}
            onMouseLeave={(e) => e.currentTarget.style.background = "transparent"}>
            <span style={{ width: 32, height: 32, borderRadius: 9, background: "var(--accent-soft)", display: "grid",
              placeItems: "center", flex: "0 0 auto" }}><Icon name="chat" size={15} color="var(--accent)" /></span>
            <span style={{ fontSize: 14, fontWeight: 600, color: "var(--text)", flex: 1, minWidth: 0,
              overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{D.EX[s.ex].q}</span>
            <span style={{ fontSize: 11.5, fontWeight: 600, color: "var(--text-2)", background: "var(--surface-3)",
              padding: "3px 9px", borderRadius: 999 }}>{s.tag}</span>
            <Icon name="chevR" size={15} color="var(--text-3)" />
          </button>
        ))}
      </div>
    </Scroll>
  );
}

function SavedView({ onRun }) {
  const D = window.OPTIMAX_DATA;
  return (
    <Scroll>
      <PageHead title="Saved queries" sub={`${D.SAVED.length} pinned questions · reusable across your team`} />
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(320px, 1fr))", gap: 16 }}>
        {D.SAVED.map((s, i) => {
          const ex = D.EX[s.ex];
          const previewSeries = [{ name: "", color: s.color, data: ex.chart.series[0].data }];
          return (
            <div key={i} style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 18,
              overflow: "hidden", boxShadow: "var(--shadow-sm)", display: "flex", flexDirection: "column",
              transition: "all .16s var(--ease)" }}
              onMouseEnter={(e) => { e.currentTarget.style.boxShadow = "var(--shadow-md)"; e.currentTarget.style.transform = "translateY(-2px)"; }}
              onMouseLeave={(e) => { e.currentTarget.style.boxShadow = "var(--shadow-sm)"; e.currentTarget.style.transform = "none"; }}>
              <div style={{ padding: "16px 18px 4px" }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 9 }}>
                  <Icon name="bookmark" size={15} color={s.color} />
                  <span style={{ fontSize: 11, fontWeight: 700, color: "var(--text-2)", background: "var(--surface-3)",
                    padding: "2px 9px", borderRadius: 999 }}>{s.tag}</span>
                  <span className="mono" style={{ marginLeft: "auto", fontSize: 11, color: "var(--text-3)" }}>{s.runs} runs</span>
                </div>
                <h3 style={{ margin: "0 0 2px", fontSize: 16, fontWeight: 700, color: "var(--text)" }}>{s.name}</h3>
              </div>
              <div style={{ padding: "4px 10px 6px", pointerEvents: "none" }}>
                <OptimaXChart type={s.chart} x={ex.chart.x} series={previewSeries} height={110} />
              </div>
              <div style={{ marginTop: "auto", padding: "12px 18px", borderTop: "1px solid var(--border)", display: "flex", gap: 8 }}>
                <button className="pill" onClick={() => onRun(s.ex)} style={{ flex: 1, justifyContent: "center", padding: "9px 14px", fontSize: 13 }}>
                  <Icon name="play" size={13} color="#fff" />Run query
                </button>
                <button className="pill subtle" style={{ padding: "9px 12px" }}><Icon name="copy" size={14} color="var(--text-2)" /></button>
              </div>
            </div>
          );
        })}
      </div>
    </Scroll>
  );
}

function SchemaView() {
  const D = window.OPTIMAX_DATA;
  const [q, setQ] = React.useState("");
  const [open, setOpen] = React.useState(D.SCHEMA[0].name);
  const filtered = D.SCHEMA.filter((t) => t.name.includes(q.toLowerCase()) || t.desc.toLowerCase().includes(q.toLowerCase()));
  return (
    <Scroll>
      <PageHead title="Schema" sub="8 tables · the grounding OctaAI uses to generate SQL"
        action={
          <div style={{ display: "flex", alignItems: "center", gap: 8, background: "var(--surface)", border: "1px solid var(--border)",
            borderRadius: 999, padding: "8px 14px", boxShadow: "var(--shadow-sm)" }}>
            <Icon name="search" size={16} color="var(--text-3)" />
            <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search tables…"
              style={{ border: "none", outline: "none", background: "transparent", color: "var(--text)", fontSize: 14, width: 160 }} />
          </div>
        } />
      <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        {filtered.map((t) => {
          const isOpen = open === t.name;
          return (
            <div key={t.name} style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 16,
              overflow: "hidden", boxShadow: "var(--shadow-sm)" }}>
              <button onClick={() => setOpen(isOpen ? "" : t.name)} style={{ width: "100%", display: "flex",
                alignItems: "center", gap: 13, padding: "16px 20px", background: "transparent", border: "none", textAlign: "left" }}>
                <span style={{ width: 38, height: 38, borderRadius: 11, background: "var(--accent-soft)", display: "grid",
                  placeItems: "center", flex: "0 0 auto" }}><Icon name="db" size={19} color="var(--accent)" /></span>
                <div style={{ minWidth: 0 }}>
                  <div className="mono" style={{ fontSize: 14.5, fontWeight: 700, color: "var(--text)" }}>{t.name}</div>
                  <div style={{ fontSize: 12.5, color: "var(--text-2)" }}>{t.desc}</div>
                </div>
                <span className="mono" style={{ marginLeft: "auto", fontSize: 12, color: "var(--text-3)" }}>{t.rows} rows</span>
                <Icon name={isOpen ? "chevUp" : "chevDown"} size={16} color="var(--text-3)" />
              </button>
              {isOpen && (
                <div style={{ padding: "4px 20px 18px", display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))", gap: 8 }}>
                  {t.cols.map(([name, type]) => (
                    <div key={name} style={{ display: "flex", alignItems: "center", gap: 9, padding: "8px 12px",
                      background: "var(--surface-2)", borderRadius: 10, border: "1px solid var(--border)" }}>
                      <span className="mono" style={{ fontSize: 12.5, fontWeight: 600, color: "var(--text)" }}>{name}</span>
                      <span className="mono" style={{ marginLeft: "auto", fontSize: 11, color: "var(--accent)",
                        background: "var(--accent-soft)", padding: "2px 7px", borderRadius: 6 }}>{type}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </Scroll>
  );
}

Object.assign(window, { DashboardView, SavedView, SchemaView });
