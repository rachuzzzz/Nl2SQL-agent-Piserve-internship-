/* ============================================================
   App shell — header nav, conversation rail, tab routing,
   debug panel, theme + boot splash + tweaks.
   ============================================================ */

const TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
  "dark": false,
  "accent": "#1B43FF",
  "anim": true,
  "showDebug": true
}/*EDITMODE-END*/;

const NAV = [
  { id: "chat", label: "AI Chat", icon: "chat" },
  { id: "dashboard", label: "Dashboard", icon: "dashboard" },
  { id: "saved", label: "Saved Queries", icon: "bookmark" },
  { id: "schema", label: "Schema", icon: "db" },
];

const SESSION = {
  id: "sx_9f3a-21c7", user: "qa@piserve.io", db: "ehs_prod · postgres 16",
  model: "qwen3.5:9b + qwen2.5-coder", region: "eu-west-1",
  started: "2:14 PM", queries: 7,
};

function applyTheme(t) {
  const r = document.documentElement;
  r.setAttribute("data-theme", t.dark ? "dark" : "light");
  r.setAttribute("data-anim", t.anim ? "on" : "off");
  const a = t.accent;
  r.style.setProperty("--accent", a);
  r.style.setProperty("--blue-600", a);
  r.style.setProperty("--blue-400", `color-mix(in srgb, ${a} 62%, white)`);
  r.style.setProperty("--blue-700", `color-mix(in srgb, ${a} 82%, black)`);
  r.style.setProperty("--blue-800", `color-mix(in srgb, ${a} 64%, black)`);
  r.style.setProperty("--accent-soft", `color-mix(in srgb, ${a} ${t.dark ? "20%" : "9%"}, transparent)`);
  r.style.setProperty("--accent-soft-border", `color-mix(in srgb, ${a} 26%, transparent)`);
}

function Logo() {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 11 }}>
      <div style={{ width: 38, height: 38, borderRadius: 12, background: "rgba(255,255,255,.06)",
        border: "1px solid var(--rail-border)", display: "grid", placeItems: "center" }}>
        <OptimaXMark size={24} mode="static" stroke={8} glow={false} />
      </div>
      <div style={{ lineHeight: 1.05 }}>
        <div style={{ fontSize: 19, fontWeight: 800, color: "var(--rail-text)", letterSpacing: "-.01em" }}>OctaAI</div>
      </div>
    </div>
  );
}

function Header({ tab, setTab, t, setTweak }) {
  return (
    <header style={{ display: "flex", alignItems: "center", gap: 20, height: 64, padding: "0 22px",
      background: "var(--rail)", borderBottom: "1px solid var(--rail-border)", flex: "0 0 auto", zIndex: 30 }}>
      <Logo />
      <nav style={{ display: "flex", gap: 4, marginLeft: 18 }}>
        {NAV.map((n) => {
          const active = tab === n.id;
          return (
            <button key={n.id} onClick={() => setTab(n.id)} style={{ display: "inline-flex", alignItems: "center", gap: 8,
              background: active ? "rgba(255,255,255,.08)" : "transparent", border: "none", borderRadius: 10,
              padding: "9px 15px", fontSize: 14, fontWeight: 600,
              color: active ? "var(--rail-text)" : "var(--rail-text-2)", position: "relative", transition: "all .15s var(--ease)" }}>
              <Icon name={n.icon} size={16} color={active ? "var(--blue-400)" : "var(--rail-text-2)"} />{n.label}
            </button>
          );
        })}
      </nav>

      <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 12 }}>
        <div className="mono" style={{ display: "flex", alignItems: "center", gap: 7, fontSize: 11.5, fontWeight: 600,
          color: "var(--rail-text-2)", background: "rgba(255,255,255,.05)", border: "1px solid var(--rail-border)",
          borderRadius: 999, padding: "6px 13px" }}>
          <span style={{ width: 6, height: 6, borderRadius: "50%", background: "var(--blue-400)" }} />
          {SESSION.model}
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12.5, fontWeight: 700, color: "#3ED598" }}>
          <span style={{ position: "relative", display: "inline-flex" }}>
            <span style={{ width: 8, height: 8, borderRadius: "50%", background: "#3ED598" }} />
            <span style={{ position: "absolute", inset: 0, borderRadius: "50%", background: "#3ED598", animation: "om-pulse 1.6s ease-in-out infinite" }} />
          </span>Live
        </div>
        <button onClick={() => setTweak("dark", !t.dark)} title="Toggle theme" style={{ width: 38, height: 38,
          borderRadius: 11, border: "1px solid var(--rail-border)", background: "rgba(255,255,255,.05)",
          display: "grid", placeItems: "center" }}>
          <Icon name={t.dark ? "sun" : "moon"} size={18} color="var(--rail-text)" />
        </button>
      </div>
    </header>
  );
}

function ConvRail({ convoSeed, setConvoSeed, collapsed, setCollapsed }) {
  const D = window.OPTIMAX_DATA;
  if (collapsed) {
    return (
      <div style={{ width: 52, flex: "0 0 auto", background: "var(--rail-2)", borderRight: "1px solid var(--rail-border)",
        display: "flex", flexDirection: "column", alignItems: "center", paddingTop: 16, gap: 10 }}>
        <button onClick={() => setCollapsed(false)} title="Expand" style={{ width: 34, height: 34, borderRadius: 9,
          border: "1px solid var(--rail-border)", background: "rgba(255,255,255,.05)", display: "grid", placeItems: "center" }}>
          <Icon name="chevR" size={16} color="var(--rail-text-2)" />
        </button>
        <button onClick={() => setConvoSeed(null)} title="New" style={{ width: 34, height: 34, borderRadius: 9,
          border: "none", background: "var(--accent)", display: "grid", placeItems: "center" }}>
          <Icon name="plus" size={16} color="#fff" />
        </button>
      </div>
    );
  }
  const Item = ({ c }) => {
    const active = convoSeed && convoSeed.id === c.id;
    return (
      <button onClick={() => setConvoSeed({ id: c.id, exId: c.ex })} style={{ width: "100%", display: "flex",
        alignItems: "center", gap: 9, padding: "10px 12px", borderRadius: 10, border: "none", textAlign: "left",
        background: active ? "rgba(255,255,255,.09)" : "transparent",
        color: active ? "var(--rail-text)" : "var(--rail-text-2)", transition: "all .14s var(--ease)" }}
        onMouseEnter={(e) => { if (!active) e.currentTarget.style.background = "rgba(255,255,255,.05)"; }}
        onMouseLeave={(e) => { if (!active) e.currentTarget.style.background = "transparent"; }}>
        <Icon name="chat" size={15} color={active ? "var(--blue-400)" : "var(--rail-text-2)"} />
        <span style={{ fontSize: 13, fontWeight: 600, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{c.title}</span>
      </button>
    );
  };
  const Group = ({ label, items }) => (
    <div style={{ marginBottom: 8 }}>
      <div className="mono" style={{ fontSize: 10, fontWeight: 700, letterSpacing: ".12em", color: "var(--rail-text-2)",
        opacity: .7, padding: "8px 12px 6px", textTransform: "uppercase" }}>{label}</div>
      <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>{items.map((c) => <Item key={c.id} c={c} />)}</div>
    </div>
  );
  return (
    <div style={{ width: 250, flex: "0 0 auto", background: "var(--rail-2)", borderRight: "1px solid var(--rail-border)",
      display: "flex", flexDirection: "column" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "16px 14px 12px" }}>
        <button onClick={() => setCollapsed(true)} title="Collapse" style={{ width: 30, height: 30, borderRadius: 8,
          border: "1px solid var(--rail-border)", background: "rgba(255,255,255,.05)", display: "grid", placeItems: "center", flex: "0 0 auto" }}>
          <Icon name="chevL" size={15} color="var(--rail-text-2)" />
        </button>
        <span className="mono" style={{ fontSize: 11, fontWeight: 700, letterSpacing: ".1em", color: "var(--rail-text)" }}>CONVERSATIONS</span>
        <button onClick={() => setConvoSeed(null)} style={{ marginLeft: "auto", display: "inline-flex", alignItems: "center", gap: 5,
          border: "none", borderRadius: 999, background: "var(--accent)", color: "#fff", fontSize: 12, fontWeight: 700, padding: "6px 11px" }}>
          <Icon name="plus" size={13} color="#fff" />New
        </button>
      </div>
      <div style={{ flex: 1, overflowY: "auto", padding: "0 10px 14px" }}>
        <Group label="Today" items={D.CONVERSATIONS.today} />
        <Group label="Earlier" items={D.CONVERSATIONS.earlier} />
      </div>
    </div>
  );
}

function DebugRail({ activeEx, open, setOpen }) {
  if (!open) {
    return (
      <div style={{ width: 48, flex: "0 0 auto", borderLeft: "1px solid var(--border)", background: "var(--surface)",
        display: "flex", flexDirection: "column", alignItems: "center", paddingTop: 16, gap: 12 }}>
        <button onClick={() => setOpen(true)} title="Open debug" style={{ width: 32, height: 32, borderRadius: 9,
          border: "1px solid var(--border)", background: "var(--surface-2)", display: "grid", placeItems: "center" }}>
          <Icon name="chevL" size={16} color="var(--text-2)" />
        </button>
        <div style={{ writingMode: "vertical-rl", transform: "rotate(180deg)", fontSize: 11, fontWeight: 700,
          letterSpacing: ".18em", color: "var(--text-3)" }} className="mono">DEBUG</div>
        <span style={{ width: 8, height: 8, borderRadius: "50%", background: "var(--err)", animation: "om-pulse 1.4s ease-in-out infinite" }} />
      </div>
    );
  }
  return (
    <div style={{ width: 372, flex: "0 0 auto", position: "relative" }}>
      <button onClick={() => setOpen(false)} title="Collapse debug" style={{ position: "absolute", left: -13, top: 16, zIndex: 5,
        width: 26, height: 26, borderRadius: "50%", border: "1px solid var(--border)", background: "var(--surface)",
        display: "grid", placeItems: "center", boxShadow: "var(--shadow-sm)" }}>
        <Icon name="chevR" size={14} color="var(--text-2)" />
      </button>
      {activeEx
        ? <DebugPanel ex={activeEx} session={SESSION} />
        : <div style={{ height: "100%", background: "var(--surface)", borderLeft: "1px solid var(--border)",
            display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 14, padding: 28, textAlign: "center" }}>
            <OptimaXMark size={48} mode="static" stroke={7} />
            <p style={{ margin: 0, fontSize: 14, color: "var(--text-2)", maxWidth: 220, lineHeight: 1.5 }}>
              Run a query to see the full pipeline trace, generated SQL and performance.
            </p>
          </div>}
    </div>
  );
}

function BootSplash({ done }) {
  const [phase, setPhase] = React.useState("draw");
  const [step, setStep] = React.useState(0);
  const STATUS = ["Initializing workspace", "Connecting to ehs_prod", "Loading schema · 8 tables", "Warming up models", "Ready"];
  React.useEffect(() => {
    const a = setTimeout(() => setPhase("spin"), 900);
    const ints = STATUS.map((_, i) => setTimeout(() => setStep(i), (i / STATUS.length) * 1900));
    return () => { clearTimeout(a); ints.forEach(clearTimeout); };
  }, []);
  return (
    <div style={{ position: "fixed", inset: 0, zIndex: 200, display: "grid", placeItems: "center",
      background: "radial-gradient(120% 90% at 50% 35%, var(--bg-grad-a), var(--bg-grad-b))",
      opacity: done ? 0 : 1, pointerEvents: done ? "none" : "auto",
      transform: done ? "scale(1.04)" : "scale(1)",
      transition: "opacity .55s var(--ease), transform .55s var(--ease)" }}>
      {/* soft brand blooms */}
      <div aria-hidden="true" style={{ position: "absolute", width: 520, height: 520, borderRadius: "50%",
        background: "radial-gradient(circle, color-mix(in srgb, var(--accent) 16%, transparent), transparent 68%)",
        filter: "blur(46px)", animation: "om-pulse 3s ease-in-out infinite" }} />
      <div style={{ position: "relative", textAlign: "center" }}>
        <div style={{ display: "inline-grid", placeItems: "center", marginBottom: 26,
          filter: "drop-shadow(0 12px 30px color-mix(in srgb, var(--accent) 32%, transparent))" }}>
          <OptimaXMark size={96} mode={phase} stroke={7} />
        </div>
        <div style={{ fontSize: 26, fontWeight: 800, letterSpacing: "-.02em", color: "var(--text)" }}>
          Octa<span style={{ color: "var(--accent)" }}>AI</span>
        </div>
        <div className="mono" style={{ fontSize: 11, fontWeight: 600, letterSpacing: ".22em",
          color: "var(--text-3)", textTransform: "uppercase", marginTop: 7 }}>NL → SQL Intelligence</div>

        {/* progress bar */}
        <div style={{ width: 240, height: 4, borderRadius: 99, background: "var(--surface-3)",
          margin: "30px auto 14px", overflow: "hidden" }}>
          <div style={{ height: "100%", borderRadius: 99, background: "var(--accent)",
            width: done ? "100%" : "8%", animation: done ? "none" : "om-boot-fill 2s var(--ease) forwards",
            boxShadow: "0 0 12px color-mix(in srgb, var(--accent) 60%, transparent)" }} />
        </div>
        <div className="mono" style={{ fontSize: 12, color: "var(--text-2)", fontWeight: 600, height: 16 }}>
          {STATUS[step]}{step < STATUS.length - 1 ? "…" : " ✓"}
        </div>
      </div>
    </div>
  );
}

function App() {
  const [t, setTweak] = useTweaks(TWEAK_DEFAULTS);
  const [tab, setTab] = React.useState("chat");
  const D = window.OPTIMAX_DATA;
  const firstConv = D.CONVERSATIONS.today[0];
  const [convoSeed, setConvoSeed] = React.useState({ id: firstConv.id, exId: firstConv.ex });
  const [activeEx, setActiveEx] = React.useState(null);
  const [railCollapsed, setRailCollapsed] = React.useState(false);
  const [debugOpen, setDebugOpen] = React.useState(true);

  React.useEffect(() => { applyTheme(t); }, [t.dark, t.accent, t.anim]);

  function runInChat(exId) {
    setTab("chat");
    const conv = [...D.CONVERSATIONS.today, ...D.CONVERSATIONS.earlier].find((c) => c.ex === exId);
    setConvoSeed({ id: conv ? conv.id : "run-" + exId, exId });
  }

  const showDebug = t.showDebug !== false;

  return (
    <div style={{ height: "100vh", display: "flex", flexDirection: "column", background: "var(--bg)" }}>
      <Header tab={tab} setTab={setTab} t={t} setTweak={setTweak} />
      <div style={{ flex: 1, display: "flex", minHeight: 0 }}>
        {tab === "chat" && <ConvRail convoSeed={convoSeed} setConvoSeed={setConvoSeed} collapsed={railCollapsed} setCollapsed={setRailCollapsed} />}
        <main style={{ flex: 1, minWidth: 0, display: "flex" }}>
          <div style={{ flex: 1, minWidth: 0,
            background: "radial-gradient(120% 100% at 50% 0%, var(--bg-grad-a), var(--bg-grad-b))" }}>
            {tab === "chat" && <Workspace session={SESSION} convoSeed={convoSeed} onActiveExChange={setActiveEx} />}
            {tab === "dashboard" && <DashboardView onRun={runInChat} />}
            {tab === "saved" && <SavedView onRun={runInChat} />}
            {tab === "schema" && <SchemaView />}
          </div>
          {tab === "chat" && showDebug && <DebugRail activeEx={activeEx} open={debugOpen} setOpen={setDebugOpen} />}
        </main>
      </div>

      <TweaksPanel>
        <TweakSection label="Appearance" />
        <TweakRadio label="Theme" value={t.dark ? "dark" : "light"} options={["light", "dark"]}
          onChange={(v) => setTweak("dark", v === "dark")} />
        <TweakColor label="Accent" value={t.accent}
          options={["#1B43FF", "#1E6FFF", "#4338CA", "#6D28D9"]}
          onChange={(v) => setTweak("accent", v)} />
        <TweakSection label="Workspace" />
        <TweakToggle label="Debug panel" value={t.showDebug !== false} onChange={(v) => setTweak("showDebug", v)} />
        <TweakToggle label="Motion & animations" value={t.anim} onChange={(v) => setTweak("anim", v)} />
      </TweaksPanel>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
