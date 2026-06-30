/* ============================================================
   Icon — compact stroke icon set. <Icon name="..." size={18} />
   Exports to window: Icon
   ============================================================ */
const OM_ICONS = {
  chat:    "M4 5h16v10H8l-4 4z",
  dashboard: "M4 4h7v7H4zM13 4h7v4h-7zM13 11h7v9h-7zM4 13h7v7H4z",
  list:    "M4 6h16M4 12h16M4 18h10",
  report:  "M7 3h7l5 5v13H7zM14 3v5h5M10 13h6M10 17h6",
  config:  "M12 9a3 3 0 100 6 3 3 0 000-6zM12 2v3M12 19v3M4.2 4.2l2.1 2.1M17.7 17.7l2.1 2.1M2 12h3M19 12h3M4.2 19.8l2.1-2.1M17.7 6.3l2.1-2.1",
  star:    "M12 4l2.2 4.8L19 9.4l-3.7 3.4 1 5L12 15.6 7.7 17.8l1-5L5 9.4l4.8-.6z",
  range:   "M12 4v16M12 4l-4 4M12 4l4 4M12 20l-4-4M12 20l4-4",
  rows:    "M4 6h16M4 12h16M4 18h16",
  warn:    "M12 3l9 16H3zM12 9v5M12 17h.01",
  site:    "M4 20V9l8-5 8 5v11M9 20v-6h6v6",
  permit:  "M6 3h9l4 4v14H6zM14 3v5h5M9 13l2 2 4-4",
  incident:"M12 2a7 7 0 00-7 7c0 5 7 13 7 13s7-8 7-13a7 7 0 00-7-7zM12 6v4M12 13h.01",
  send:    "M4 12l16-7-7 16-2-7z",
  mic:     "M12 3a3 3 0 00-3 3v5a3 3 0 006 0V6a3 3 0 00-3-3zM5 11a7 7 0 0014 0M12 18v3",
  copy:    "M9 9h10v10H9zM5 15V5h10",
  plus:    "M12 5v14M5 12h14",
  chevL:   "M15 6l-6 6 6 6",
  chevR:   "M9 6l6 6-6 6",
  chevDown:"M6 9l6 6 6-6",
  chevUp:  "M6 15l6-6 6 6",
  search:  "M11 4a7 7 0 100 14 7 7 0 000-14zM20 20l-4-4",
  bolt:    "M13 3L4 14h6l-1 7 9-11h-6z",
  expand:  "M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5",
  shrink:  "M9 4v5H4M15 4v5h5M9 20v-5H4M15 20v-5h5",
  refresh: "M20 11a8 8 0 10-2 6M20 5v6h-6",
  zoomIn:  "M11 4a7 7 0 100 14 7 7 0 000-14zM20 20l-4-4M11 8v6M8 11h6",
  zoomOut: "M11 4a7 7 0 100 14 7 7 0 000-14zM20 20l-4-4M8 11h6",
  axes:    "M5 3v16h16M5 15l4-5 3 3 5-7",
  download:"M12 4v10M12 14l-4-4M12 14l4-4M5 18h14",
  play:    "M7 4l12 8-12 8z",
  clock:   "M12 4a8 8 0 100 16 8 8 0 000-16zM12 8v4l3 2",
  db:      "M12 4c-4 0-7 1.1-7 2.5S8 9 12 9s7-1.1 7-2.5S16 4 12 4zM5 6.5v11C5 19 8 20 12 20s7-1 7-2.5v-11M5 12c0 1.4 3 2.5 7 2.5s7-1.1 7-2.5",
  check:   "M5 12l4.5 4.5L19 7",
  bookmark:"M6 4h12v16l-6-4-6 4z",
  sun:     "M12 8a4 4 0 100 8 4 4 0 000-8zM12 2v2M12 20v2M4 12H2M22 12h-2M5 5l1.5 1.5M17.5 17.5L19 19M5 19l1.5-1.5M17.5 6.5L19 5",
  moon:    "M20 14a8 8 0 01-10-10 8 8 0 1010 10z",
  sparkle: "M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8zM18 15l.8 2.2L21 18l-2.2.8L18 21l-.8-2.2L15 18l2.2-.8z",
  table:   "M4 5h16v14H4zM4 10h16M10 5v14",
  filter:  "M4 5h16l-6 7v6l-4 2v-8z",
  arrowUp: "M12 19V5M12 5l-5 5M12 5l5 5",
  arrowDown:"M12 5v14M12 19l-5-5M12 19l5-5",
};

function Icon({ name, size = 18, stroke = 2, color = "currentColor", style = {}, fill = "none" }) {
  const d = OM_ICONS[name];
  if (!d) return null;
  const filled = name === "star" || name === "play";
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill={filled ? color : "none"}
         stroke={filled ? "none" : color} strokeWidth={stroke}
         strokeLinecap="round" strokeLinejoin="round" style={{ flex: "0 0 auto", ...style }}>
      <path d={d} />
    </svg>
  );
}
Object.assign(window, { Icon });
