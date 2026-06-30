/* ============================================================
   OptimaX animated mark + glassmorphism thinking loader.
   Exports to window: OptimaXMark, ThinkingCard, BgMark
   ============================================================ */

// Piserve "p" mark — rounded stem, circular bowl, rising flourish.
// viewBox 0 0 100 100, p centered. Recolors via gradient; animates
// draw-on (boot) and orbit (thinking) while staying on-brand.
const OM_STEM = "M34 36 V87";
const OM_FLOURISH = "M67 31 L85 12";
const OM_BOWL = { cx: 52, cy: 42, r: 18 };
// One continuous track tracing the whole p: up from the bottom of the stem,
// a FULL loop around the bowl, then up and out along the rising flourish.
const OM_WORM = "M34 87 L34 42 A18 18 0 1 1 70 42 A18 18 0 1 1 34 42 A18 18 0 0 1 67 31 L85 12";

function OptimaXMark({ size = 40, mode = "static", stroke = 7, glow = true }) {
  // mode: "static" | "draw" | "spin"
  const uid = React.useMemo(() => "om" + Math.random().toString(36).slice(2, 8), []);
  const sw = stroke * 1.85;
  const draw = (delay, dur) =>
    mode === "draw"
      ? { strokeDasharray: 1, strokeDashoffset: 1, animation: `om-draw ${dur}s var(--ease) ${delay}s forwards` }
      : {};
  const grad = `url(#${uid}g)`;
  const fil = glow ? `url(#${uid}f)` : undefined;

  return (
    <svg width={size} height={size} viewBox="0 0 100 100" fill="none"
         style={{ overflow: "visible", display: "block" }}>
      <defs>
        <linearGradient id={uid + "g"} x1="20" y1="90" x2="86" y2="10" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="var(--blue-700)" />
          <stop offset="0.55" stopColor="var(--blue-600)" />
          <stop offset="1" stopColor="var(--blue-400)" />
        </linearGradient>
        {glow && (
          <filter id={uid + "f"} x="-60%" y="-60%" width="220%" height="220%">
            <feGaussianBlur stdDeviation="3" result="b" />
            <feMerge>
              <feMergeNode in="b" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        )}
      </defs>

      <g filter={fil} style={mode === "spin" ? { opacity: 0.22 } : {}}>
        {/* stem */}
        <path d={OM_STEM} stroke={grad} strokeWidth={sw} strokeLinecap="round"
              pathLength="1" style={draw(0, 0.4)} />
        {/* bowl */}
        <circle cx={OM_BOWL.cx} cy={OM_BOWL.cy} r={OM_BOWL.r} stroke={grad} strokeWidth={sw}
                fill="none" pathLength="1" style={draw(0.22, 0.5)} />
        {/* flourish */}
        <path d={OM_FLOURISH} stroke={grad} strokeWidth={sw} strokeLinecap="round"
              pathLength="1" style={draw(0.66, 0.3)} />
      </g>

      {/* a bright "worm/train" segment running along the p's track */}
      {mode === "spin" && (
        <path d={OM_WORM} fill="none" stroke={grad} strokeWidth={sw}
          strokeLinecap="round" pathLength="100"
          style={{ strokeDasharray: "26 74", animation: "om-worm-run 1.3s linear infinite" }} />
      )}
    </svg>
  );
}

// Faint giant watermark behind the chat canvas (glassmorphism vibe).
function BgMark() {
  return (
    <div aria-hidden="true" style={{
      position: "absolute", inset: 0, overflow: "hidden",
      pointerEvents: "none", zIndex: 0,
    }}>
      <div style={{
        position: "absolute", right: "-6%", top: "8%",
        width: 560, height: 560, opacity: 0.05,
        filter: "blur(0.5px)", animation: "om-float 9s ease-in-out infinite",
      }}>
        <OptimaXMark size={560} mode="static" stroke={5} glow={false} />
      </div>
      {/* soft color blooms */}
      <div style={{
        position: "absolute", left: "-10%", bottom: "-12%",
        width: 460, height: 460, borderRadius: "50%",
        background: "radial-gradient(circle, color-mix(in srgb, var(--accent) 12%, transparent), transparent 70%)",
        filter: "blur(40px)",
      }} />
    </div>
  );
}

// Glassmorphism "thinking" card with live pipeline checklist.
function ThinkingCard({ stages, activeIndex, label }) {
  return (
    <div style={{ display: "flex", gap: 14, alignItems: "flex-start", animation: "om-rise .35s var(--ease)" }}>
      <div style={{ flex: "0 0 auto", paddingTop: 2 }}>
        <div style={{
          width: 44, height: 44, borderRadius: 13,
          display: "grid", placeItems: "center",
          background: "var(--accent-soft)", border: "1px solid var(--accent-soft-border)",
        }}>
          <OptimaXMark size={26} mode="spin" stroke={7} />
        </div>
      </div>

      <div style={{
        position: "relative", overflow: "hidden",
        background: "var(--glass-bg)",
        WebkitBackdropFilter: "blur(16px) saturate(150%)",
        backdropFilter: "blur(16px) saturate(150%)",
        border: "1px solid var(--glass-border)",
        borderRadius: 18, padding: "16px 20px", minWidth: 320, maxWidth: 460,
        boxShadow: "var(--shadow-md)",
      }}>
        {/* faint mark inside the glass */}
        <div style={{ position: "absolute", right: -28, top: -28, opacity: 0.08, pointerEvents: "none" }}>
          <OptimaXMark size={150} mode="static" stroke={5} glow={false} />
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 9, marginBottom: 13, position: "relative" }}>
          <span className="mono" style={{ fontSize: 11, fontWeight: 700, letterSpacing: ".12em",
            color: "var(--accent)", textTransform: "uppercase" }}>
            {label || "Generating SQL"}
          </span>
          <span style={{ display: "inline-flex", gap: 4 }}>
            {[0, 1, 2].map((i) => (
              <span key={i} style={{
                width: 5, height: 5, borderRadius: "50%", background: "var(--accent)",
                animation: `om-pulse 1s ease-in-out ${i * 0.18}s infinite`,
              }} />
            ))}
          </span>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 9, position: "relative" }}>
          {stages.map((s, i) => {
            const done = i < activeIndex;
            const active = i === activeIndex;
            return (
              <div key={s} style={{
                display: "flex", alignItems: "center", gap: 10,
                opacity: done || active ? 1 : 0.42,
                transition: "opacity .3s var(--ease)",
              }}>
                <span style={{
                  width: 16, height: 16, borderRadius: "50%", flex: "0 0 auto",
                  display: "grid", placeItems: "center",
                  background: done ? "var(--accent)" : active ? "transparent" : "var(--surface-3)",
                  border: active ? "2px solid var(--accent)" : "none",
                }}>
                  {done && (
                    <svg width="9" height="9" viewBox="0 0 12 12" fill="none">
                      <path d="M2.5 6.2L5 8.6L9.5 3.6" stroke="#fff" strokeWidth="2"
                            strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                  )}
                  {active && (
                    <span style={{ width: 6, height: 6, borderRadius: "50%", background: "var(--accent)",
                      animation: "om-pulse .8s ease-in-out infinite" }} />
                  )}
                </span>
                <span style={{ fontSize: 13.5, fontWeight: active ? 700 : 500,
                  color: active ? "var(--text)" : "var(--text-2)" }}>{s}</span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

// Simple one-line inline loader: small spinning piserve mark + cycling text.
function InlineThinking({ stages, activeIndex }) {
  const text = stages[Math.min(activeIndex, stages.length - 1)] || "Thinking";
  return (
    <div style={{
      display: "flex", alignItems: "center", gap: 10,
      animation: "om-rise .3s var(--ease)", paddingLeft: 2,
    }}>
      <OptimaXMark size={20} mode="spin" stroke={7} glow={false} />
      <span key={text} style={{
        fontSize: 13.5, fontWeight: 500, color: "var(--text-2)",
        animation: "om-text-in .35s var(--ease)",
      }}>
        {text}
        <span style={{ display: "inline-flex", gap: 3, marginLeft: 6, verticalAlign: "middle" }}>
          {[0, 1, 2].map((i) => (
            <span key={i} style={{
              width: 3.5, height: 3.5, borderRadius: "50%", background: "var(--accent)",
              display: "inline-block",
              animation: `om-pulse 1s ease-in-out ${i * 0.18}s infinite`,
            }} />
          ))}
        </span>
      </span>
    </div>
  );
}

Object.assign(window, { OptimaXMark, BgMark, ThinkingCard, InlineThinking });
