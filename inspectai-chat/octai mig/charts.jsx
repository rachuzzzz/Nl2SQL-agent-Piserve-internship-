/* ============================================================
   OptimaXChart — lightweight responsive SVG charts.
   Exports to window: OptimaXChart
   props: { type:'line'|'area'|'bar', x:[], series:[{name,color,data}], height }
   ============================================================ */

function OptimaXChart({ type = "line", x = [], series = [], height = 240 }) {
  const wrapRef = React.useRef(null);
  const [w, setW] = React.useState(680);
  React.useEffect(() => {
    if (!wrapRef.current) return;
    const ro = new ResizeObserver((es) => {
      for (const e of es) setW(Math.max(280, e.contentRect.width));
    });
    ro.observe(wrapRef.current);
    return () => ro.disconnect();
  }, []);

  const padL = 40, padR = 14, padT = 14, padB = 28;
  const innerW = w - padL - padR;
  const innerH = height - padT - padB;

  const allVals = series.flatMap((s) => s.data);
  const rawMax = Math.max(...allVals, 1);
  const niceMax = niceCeil(rawMax);
  const yTicks = 4;

  const xStep = innerW / Math.max(1, x.length - (type === "bar" ? 0 : 1));
  const xPos = (i) =>
    type === "bar" ? padL + xStep * i + xStep / 2 : padL + xStep * i;
  const yPos = (v) => padT + innerH - (v / niceMax) * innerH;

  const [hover, setHover] = React.useState(null);

  // entrance: animate progress 0→1 (re-runs when data or type changes)
  const sig = type + "|" + series.map((s) => s.data.join(",")).join("|");
  const [p, setP] = React.useState(1);
  React.useEffect(() => {
    let raf, start;
    const reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const animOff = document.documentElement.getAttribute("data-anim") === "off";
    if (reduce || animOff) { setP(1); return; }
    setP(0);
    const dur = 760;
    const tick = (t) => {
      if (start == null) start = t;
      const k = Math.min(1, (t - start) / dur);
      setP(1 - Math.pow(1 - k, 3));
      if (k < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [sig]);
  const segs = Math.max(1, x.length - 1);
  const barP = (i) => { const span = 0.62, st = (1 - span) / Math.max(1, x.length - 1); return Math.max(0, Math.min(1, (p - i * st) / span)); };

  return (
    <div ref={wrapRef} style={{ width: "100%", position: "relative" }}>
      <svg width="100%" height={height} viewBox={`0 0 ${w} ${height}`} style={{ display: "block", overflow: "visible" }}>
        <defs>
          {series.map((s, si) => (
            <linearGradient key={si} id={`fill${si}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0" stopColor={s.color} stopOpacity="0.30" />
              <stop offset="1" stopColor={s.color} stopOpacity="0.02" />
            </linearGradient>
          ))}
        </defs>

        {/* gridlines + y labels */}
        {Array.from({ length: yTicks + 1 }).map((_, i) => {
          const val = (niceMax / yTicks) * i;
          const y = yPos(val);
          return (
            <g key={i}>
              <line x1={padL} y1={y} x2={w - padR} y2={y}
                    stroke="var(--border)" strokeWidth="1" strokeDasharray={i === 0 ? "0" : "3 4"} />
              <text x={padL - 8} y={y + 3.5} textAnchor="end"
                    fontSize="10.5" fill="var(--text-3)" fontFamily="var(--font-mono)">
                {fmtNum(val)}
              </text>
            </g>
          );
        })}

        {/* x labels */}
        {x.map((lbl, i) => (
          <text key={i} x={xPos(i)} y={height - 9} textAnchor="middle"
                fontSize="11" fill="var(--text-3)" fontWeight="600">{lbl}</text>
        ))}

        {/* series */}
        {(type === "line" || type === "area") &&
          series.map((s, si) => {
            const baseY = padT + innerH;
            const pts = s.data.map((v, i) => [xPos(i), baseY + (yPos(v) - baseY) * p]);
            const d = smoothPath(pts);
            const ptP = (i) => Math.max(0, Math.min(1, (p - (i / Math.max(1, s.data.length)) * 0.5) / 0.5));
            return (
              <g key={si}>
                {type === "area" && (
                  <path d={`${d} L ${pts[pts.length - 1][0]} ${padT + innerH} L ${pts[0][0]} ${padT + innerH} Z`}
                        fill={`url(#fill${si})`} opacity={p} />
                )}
                <path d={d} fill="none" stroke={s.color} strokeWidth="2.6"
                      strokeLinecap="round" strokeLinejoin="round" />
                {pts.map(([px, py], i) => (
                  <circle key={i} cx={px} cy={py} r={(hover && hover.i === i ? 4.5 : 3) * ptP(i)}
                          fill="var(--surface)" stroke={s.color} strokeWidth="2.4" opacity={ptP(i)}
                          onMouseEnter={() => setHover({ i, si })}
                          onMouseLeave={() => setHover(null)}
                          style={{ cursor: "pointer", transition: "r .12s" }} />
                ))}
              </g>
            );
          })}

        {type === "bar" &&
          (() => {
            const n = series.length;
            const groupW = xStep * 0.62;
            const barW = groupW / n;
            return x.map((_, i) =>
              series.map((s, si) => {
                const bh = (s.data[i] / niceMax) * innerH * barP(i);
                const bx = xPos(i) - groupW / 2 + barW * si;
                const by = padT + innerH - bh;
                return (
                  <rect key={si + "-" + i} x={bx + 1} y={by} width={Math.max(2, barW - 2)} height={bh}
                        rx="3" fill={s.color}
                        onMouseEnter={() => setHover({ i, si })}
                        onMouseLeave={() => setHover(null)}
                        style={{ cursor: "pointer", opacity: hover && hover.i !== i ? 0.55 : 1, transition: "opacity .12s" }}>
                    <title>{`${x[i]} · ${s.name}: ${s.data[i]}`}</title>
                  </rect>
                );
              })
            );
          })()}

        {/* hover guide for line/area */}
        {hover && type !== "bar" && (
          <line x1={xPos(hover.i)} y1={padT} x2={xPos(hover.i)} y2={padT + innerH}
                stroke="var(--accent)" strokeWidth="1" strokeDasharray="3 3" opacity="0.5" />
        )}
      </svg>

      {/* tooltip */}
      {hover && (
        <div style={{
          position: "absolute", left: `${(xPos(hover.i) / w) * 100}%`, top: 0,
          pointerEvents: "none",
          background: "var(--rail)", color: "var(--rail-text)",
          padding: "7px 10px", borderRadius: 9, fontSize: 11.5, fontWeight: 600,
          whiteSpace: "nowrap", boxShadow: "var(--shadow-md)", marginLeft: -40,
        }}>
          <div style={{ opacity: .7, marginBottom: 3 }}>{x[hover.i]}</div>
          {series.map((s, si) => (
            <div key={si} style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <span style={{ width: 7, height: 7, borderRadius: "50%", background: s.color }} />
              <span style={{ opacity: .85 }}>{s.name}</span>
              <span className="mono" style={{ marginLeft: "auto", fontWeight: 700 }}>{s.data[hover.i]}</span>
            </div>
          ))}
        </div>
      )}

      {/* legend */}
      <div style={{ display: "flex", gap: 18, justifyContent: "center", marginTop: 6, flexWrap: "wrap" }}>
        {series.map((s, si) => (
          <div key={si} style={{ display: "flex", alignItems: "center", gap: 7 }}>
            <span style={{ width: 9, height: 9, borderRadius: "50%", background: s.color }} />
            <span style={{ fontSize: 12.5, fontWeight: 600, color: "var(--text-2)" }}>{s.name}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ---- helpers ----
function smoothPath(pts) {
  if (pts.length < 2) return pts.length ? `M ${pts[0][0]} ${pts[0][1]}` : "";
  let d = `M ${pts[0][0]} ${pts[0][1]}`;
  for (let i = 0; i < pts.length - 1; i++) {
    const [x0, y0] = pts[Math.max(0, i - 1)];
    const [x1, y1] = pts[i];
    const [x2, y2] = pts[i + 1];
    const [x3, y3] = pts[Math.min(pts.length - 1, i + 2)];
    const t = 0.18;
    const c1x = x1 + (x2 - x0) * t, c1y = y1 + (y2 - y0) * t;
    const c2x = x2 - (x3 - x1) * t, c2y = y2 - (y3 - y1) * t;
    d += ` C ${c1x} ${c1y} ${c2x} ${c2y} ${x2} ${y2}`;
  }
  return d;
}
function niceCeil(v) {
  if (v <= 0) return 1;
  const mag = Math.pow(10, Math.floor(Math.log10(v)));
  const n = v / mag;
  const step = n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10;
  return step * mag;
}
function fmtNum(v) {
  if (v >= 1000) return (v / 1000).toFixed(v % 1000 === 0 ? 0 : 1) + "k";
  return Number.isInteger(v) ? "" + v : v.toFixed(0);
}

Object.assign(window, { OptimaXChart });
