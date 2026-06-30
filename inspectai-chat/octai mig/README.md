# Handoff: OctaAI — NL→SQL Intelligence (full product)

## Overview
**OctaAI** (a Piserve product) is a natural-language-to-SQL workspace for EHS data
(inspections, audits, permits, findings, corrective actions, incidents). A user asks a
question in plain English; OctaAI parses it, grounds on the schema, generates and
validates SQL, executes it, and returns a summary + KPIs + best-fit chart + results table,
with a full **debug/pipeline trace** of every step.

This handoff documents the **entire prototype**: a marketing **landing page**, a branded
**loading/splash screen**, and the full **workspace app** (AI chat, dashboard, saved
queries, schema browser, debug panel), all sharing one design-token system.

## About the Design Files
The files in this bundle are **design references created in HTML/CSS/JS + React (via
in-browser Babel)** — prototypes showing intended look, motion, and behavior. **They are
not production code to copy verbatim.** Your task is to **recreate these designs inside the
existing Angular frontend**, using its established conventions: Angular components &
templates, the project's SCSS/theme system, Angular Router for navigation between
screens, `HttpClient`/services for real data (the prototype uses mock data in `data.js`),
and `@angular/animations` or CSS for motion. Treat the HTML/JSX as the spec for layout,
tokens, copy, structure, and motion — then implement idiomatically in Angular.

Note on the React prototype structure: each `*.jsx` is a logical "module" that maps cleanly
to one or more **Angular components**. The mapping is called out per file below.

## Fidelity
**High-fidelity (hifi).** Final colors, typography, spacing, radii, shadows, and motion are
specified. Recreate pixel-accurately, mapping raw values onto the codebase's existing
SCSS variables / design tokens where equivalents exist (add the new brand tokens listed
under **Design Tokens** where they don't).

---

## Information Architecture / Routes
Three top-level HTML entry points in the prototype → suggested Angular routes:

1. **Landing page** — `OptimaX Landing.html` → `/` (or `/home`). Marketing page; CTA buttons
   link to the workspace.
2. **Loading / splash screen** — `OctaAI Loading.html` → `/loading` (transient; auto-advances
   to the workspace). *Optional* — only if you want a dedicated boot screen.
3. **Workspace app** — `OptimaX.html` → `/app` (or `/workspace`). The product itself, an
   SPA shell with an in-app tab router (see Workspace Shell). Tabs: **AI Chat**,
   **Dashboard**, **Saved Queries**, **Schema**.

> The prototype also has an *in-app* React boot splash (`BootSplash` in `app.jsx`) that was
> **removed** in the latest version in favor of the standalone `OctaAI Loading.html`. Don't
> reintroduce two splashes — pick one (recommended: the standalone loading screen routes
> into the workspace once the app is genuinely ready).

---

## SCREEN 1 — Landing Page (`OptimaX Landing.html`)
A standard scrolling marketing page, `max-width: 1200px` content wrapper (`.wrap`,
`padding: 0 32px`). Sections top→bottom:

1. **Sticky nav** (`nav.top`, height 72, blurred translucent bg
   `color-mix(in srgb, var(--bg) 82%, transparent)` + `backdrop-filter: blur(14px)`,
   bottom border `--border`):
   - Brand: 34×34 SVG "p" mark (Piserve mark — stem + bowl + flourish, see Brand Marks) in
     `--accent`, wordmark **OctaAI** (21px/800/-.02em).
   - Links: "How it works", "Features", "Product" (`--text-2`, 15px/600, hover `--accent`).
   - Right: **"Launch workspace"** pill button (→ workspace) with arrow icon.
2. **Hero** (`.hero`, `padding: 96px 0 64px`):
   - Eyebrow chip "NL → SQL Intelligence" + muted note "for inspections · audits · permits · EHS".
   - H1 `clamp(44px,7vw,88px)`/800/-.035em/line-height .98, max-width 16ch, `text-wrap:
     balance`: **"Ask your data _anything._"** ("anything." in `--accent`).
   - Lead paragraph `clamp(18px,2.2vw,22px)`, `--text-2`, max-width 56ch.
   - CTA row: primary pill "Launch workspace" + ghost pill "See it in action".
   - **Decorative "squiggles":** absolutely-positioned SVG arcs (`.squiggle`, stroke
     `--accent`, `stroke-width: 7`, round caps) scattered behind the hero — Piserve's
     playful arc motif. Reused in the CTA band.
3. **Product preview** (`#preview`): a faux **browser window** (`.preview-shell`, radius 20,
   `--shadow-lg`) with a dark title bar (3 traffic-light dots + a mono URL pill
   "octaai.piserve.io / workspace"). Inside, an **iframe of the live workspace**
   (`OptimaX.html`) scaled to fit the clip (`aspect-ratio: 1440/820`; JS `fitFrame()` sets
   `transform: scale(clipWidth/1440)`).
   - In Angular, replace the iframe-of-self with either a real embedded
     `<app-workspace>` preview or a static screenshot — an iframe of the whole app is a
     prototype shortcut.
4. **"How it works"** (`#how`): eyebrow + H2, then a **3-column step grid** (`.steps`).
   Each step: mono number ("01"/"02"/"03"), a thin progress `.bar` (track `--border-2`
   with a 38%-wide `--accent` fill), H3, body. Content:
   - 01 "Ask in plain English" · 02 "OctaAI generates & validates" · 03 "Answer, charted & traced".
5. **"Built for trust" features** (`#features`): eyebrow + H2 "Every answer is
   explainable.", then a **2×2 feature card grid** (`.features`, gap 18). Each card
   (`.feat`, radius 20, `--shadow-sm`, hover lifts -3px + `--shadow-md`): 46×46 accent-soft
   icon tile + H3 + body. Cards:
   - "Live pipeline trace", "Generated SQL, in the open", "Dashboards & saved queries",
     "Self-healing queries".
6. **Stats band** (`.stats`): dark (`--rail`) rounded-24 panel, 4-col grid of big mono
   numbers (accent digit highlighted via `.n span`) + labels: **48k** inspection reports ·
   **<4s** median question-to-answer · **100%** queries validated pre-run · **8** grounded
   EHS tables.
7. **CTA band** (`.cta-band`, centered, 96px vertical, squiggles): H2 "Stop writing SQL.
   Start asking questions." + sub + "Launch workspace" pill.
8. **Footer**: brand mark + wordmark, "A Piserve product · NL → SQL Intelligence".
- **Responsive:** below 860px, `.steps/.features/.stats` collapse to 1 column and nav links hide.

---

## SCREEN 2 — Loading / Splash (`OctaAI Loading.html`)
Full-screen branded boot screen; fades out and routes to the workspace when loading
completes. Two-column composition on a fixed **1500 × 940** design canvas, centered and
uniformly scaled to fit (`scale = min(vw/1500, vh/940) * 0.98`).

> In Angular prefer a naturally responsive flex/grid layout over literally porting the
> transform-scale trick, unless you need pixel-locked proportions.

**Canvas:** 2-col grid `1.02fr 1.18fr`, `border-radius: 28px`, 1px `--border`,
`overflow: hidden`, `--shadow-lg`, bg `radial-gradient(130% 120% at 72% 30%, #FFFFFF, #F4F6FC 62%)`.
Three decorative corner dot-grids.

**Left column** (`padding: 72px`, vertical flex):
- **Logo lockup:** 46×46 **octagon** mark (gradient stroke `--blue-700`→`--blue-400`,
  stroke-width 15 in viewBox 100, round joins) + wordmark **Octa**+**AI** (30px/800/-.02em,
  both `--text`). Octagon points: `86.9,65.3 65.3,86.9 34.7,86.9 13.1,65.3 13.1,34.7 34.7,13.1 65.3,13.1 86.9,34.7`.
  - NOTE: the loading screen uses an **octagon** "Octa" mark; the rest of the app and the
    landing page use the Piserve **"p"** mark. Pick one brand mark and apply consistently
    (confirm with design which is canonical).
- **Headline** `56px/800/1.1/-.035em`, max-width 13ch: "Turning Questions into Insights
  _with AI_" ("with AI" in `--accent`). Subtitle `20px/500` `--text-2`: "Ask in plain
  English. Get accurate SQL."
- **Feature row** (gap 56): three centered items, each a 58×58 accent-soft icon tile (radius
  16, 1px `--accent-soft-border`, accent icon) + label (14.5/600 `--text-2`): "Natural
  Language" (chat icon), "SQL Generation" (database icon), "Instant Results" (lightning icon).
- **Loading footer** (pinned bottom): 64×64 **robot mascot** SVG (rounded-rect white head,
  `--blue-200` stroke, dark `#141A36` face, cyan `#7FE3FF` eyes, antenna w/ accent dot)
  with **bob** (translateY 0↔-5px, 3s) + eye **blink** (scaleY 1↔.12, 3.4s); plus a status
  line (15/600 `--text-2`, fades in on each change) and a 7px progress bar (track
  `--surface-3`, fill gradient `--blue-700`→`--blue-400` with accent glow, width 4%→100%).

**Right column** — animated OctaAI logo, "audio-synthesizer" style (600×600 stage, layered
back→front):
1. **Glow halo** — 380px blurred (34px) radial of 45%-accent; `haloPulse` scale .9↔1.1 /
   opacity .5↔.85, 2.4s.
2. **Radial equalizer** (`.eq`) — **76 thin bars** placed radially (`rotate(i/76*360deg)
   translateY(-172px)`, origin bottom-center), each width 5 / radius 3, gradient transparent→
   `--blue-400`→`--accent`. **Driven by JS `requestAnimationFrame`:** per frame, `t=now/1000`,
   for bar i: `v=0.5+0.5*sin(t*2.1+i*0.46)`, `w=0.5+0.5*sin(t*3.6-i*0.83)`,
   `height=13+v*30+w*20`px, `opacity=0.34+0.5*v`. Two overlaid sine waves create a flowing
   equalizer ripple. (Angular: `*ngFor` the 76 bars; run the rAF loop in `ngAfterViewInit`,
   cancel in `ngOnDestroy`.)
3. **Morphing blob** — 340px, radial-gradient `--blue-400`→`--blue-600`→`--blue-700`, organic
   animated `border-radius` (`blobMorph` 7s, 3 keyframes) + `blobThrob` scale 1↔1.035 (1.35s);
   outer `0 34px 90px` accent shadow + inset `0 -22px 64px` `--blue-700`.
4. **Rings + tail** (SVG, viewBox 600): solid arc ring r232 (`#131A33`, sw 4.5,
   `dasharray 1276 182`, rotated 128°) spinning 26s; dotted ring r210 (`dasharray 0.5 13.5`)
   spinning reverse 19s; an 8-dot tapering "comet" tail (radii 7→2.2, alternating
   `#131A33`/`#1B43FF`) with staggered `tailPulse` opacity .3↔1 (delays 0…0.84s).

**Behavior:** single timer over `DURATION` (prototype 7600 ms) drives 5 stage messages
("Preparing your workspace…", "Connecting to ehs_prod…", "Generating SQL…", "Validating
the query…", "Almost ready…") and the progress bar; at completion fade canvas
`opacity→0` (`.5s`) then navigate. **In Angular replace `window.location.href` with
`Router.navigate`, and ideally drive the transition off a real "app ready" signal with
`DURATION` as a fallback floor.** Reduced-motion: disable looping decorative animations
(and render equalizer bars static); keep progress/status functional.

---

## SCREEN 3 — Workspace App (`OptimaX.html` + the `*.jsx` files)

### Shell & layout (`app.jsx` → `AppComponent` + layout components)
- Full-viewport flex column: **Header** (fixed 64px) above a flex row of **[left rail] ·
  [main] · [right rail]**.
- **In-app tab routing:** `tab` state switches the main panel between `chat` / `dashboard`
  / `saved` / `schema`. In Angular use child routes or a router-outlet; the left
  conversation rail and right debug rail only show on the **chat** tab.
- **Theme:** `applyTheme()` sets `data-theme` (light/dark) and `data-anim` (on/off) on the
  root and overrides accent-derived CSS vars from a single accent hex (light/dark mixes).
  Replicate with an Angular theme service writing CSS custom properties / a `[data-theme]`
  attribute.
- **Session object** (mocked): id, user, db ("ehs_prod · postgres 16"), model
  ("qwen3.5:9b + qwen2.5-coder"), region, started, queries — shown in header chip + debug
  panel. Replace with real session/service data.

**Header** (`--rail` dark navy bar):
- Brand lockup (38×38 rounded tile holding the mark + wordmark "OctaAI", `--rail-text`).
- Nav buttons (AI Chat, Dashboard, Saved Queries, Schema) — active state
  `rgba(255,255,255,.08)` bg, accent icon; inactive `--rail-text-2`.
- Right cluster: model chip (mono, dot + model name), animated green **"Live"** indicator
  (pulsing dot), theme toggle (sun/moon).

**Left Conversation Rail** (`ConvRail`, 250px, `--rail-2`, collapsible to 52px):
- Header "CONVERSATIONS" + collapse + **New** (accent pill). Grouped lists "Today" /
  "Earlier" of conversation items (chat icon + title; active = `rgba(255,255,255,.09)`).
  Collapsed state shows expand + new icon buttons only.

**Right Debug Rail** (`DebugRail` → `DebugPanel`, 372px, collapsible to 48px) — see Debug Panel.

### AI Chat / Workspace (`workspace.jsx` → multiple components)
The core screen. Scrollable message column (`max-width: 1180px`, centered) over a faint
brand **watermark background** (`BgMark`: giant 5%-opacity mark drifting + a soft accent
bloom), with a fixed **query bar** at the bottom.

- **Empty state** (`EmptyState`): centered 68px animated mark (draw-on), H1 "Ask your data
  anything." (34/800), descriptive paragraph, and a wrap of **suggestion pills** (from
  `D.SUGGESTIONS`) that pop in staggered and on hover turn accent + lift.
- **User message** (`UserMsg`): right-aligned dark (`--rail`) bubble, radius
  `18px 18px 4px 18px`, 15/600, with a mono timestamp + "QA" avatar.
- **AI message**: AI avatar (accent-soft tile w/ mark) + an **`AiResult`** block:
  - **Summary** paragraph (15/1.6, max-width 760).
  - **KPI cards** row (`KpiCard`): flexible cards (min 150), each an icon tile (toned by
    `kpi.tone`: blue/amber/teal/coral), mono uppercase label, big mono value, and a toned
    tag pill. Values **count up** on mount via `useCountUp` (rAF, ~900ms cubic ease-out;
    animates numbers inside strings, respects reduced-motion / anim-off). Hover lifts -3px.
  - **Two side-by-side panels** (`ChartPanel` + `ResultsTable`), each ~50% width, that can
    **zoom**: clicking a panel's zoom button makes it full-width on top and pushes the other
    below (`focus` state, flex order/basis swap).
    - **ChartPanel**: a segmented control (Line / Area / Bar) with a "BEST" tag on the
      recommended type, a replay button (re-triggers entrance), the zoom button, and the
      chart. 248px tall (420 expanded).
    - **ResultsTable**: header (table icon + "Result set" + col count + zoom), sticky
      accent-uppercase mono column headers, zebra rows that rise-in staggered, mono first
      column, and a footer "1–N of TOTAL rows" + "Export CSV" pill.
- **Query bar** (`QueryBar`): auto-growing textarea (max 140px) in a rounded-22 surface with
  1.5px `--border-2` + `--shadow-md`; mic button + circular **send** button (accent when
  there's text, disabled grey otherwise, accent glow). Enter submits (Shift+Enter
  newlines). When a conversation is active, 3 quick suggestion chips appear above. Caption:
  "OctaAI validates every query before execution · qwen3.5:9b + qwen2.5-coder".
- **Thinking sequence**: on submit, an **`InlineThinking`** one-line loader shows — a small
  spinning "p" mark (the **worm** animation, see Brand Marks) + a status label that cycles
  through 5 stage labels ("Parsing your question", "Resolving schema", "Generating SQL",
  "Validating & executing", "Summarizing results") + 3 pulsing dots. Stage timing is a
  scripted sequence (`stageDur = [700,650,1100,800,700]`ms); then the AI result is appended.
  (An older richer `ThinkingCard` glass component also exists in `loader.jsx` but the
  **inline** loader is the current one — see the chat flow.)
  - In production, drive these stages off real backend pipeline events, not fixed timers.

> Query→example matching (`matchExample` / `synthExample`) is **prototype mock logic**:
> it keyword-matches the question to a canned example, else synthesizes a believable
> fallback result. **Replace entirely with your real NL→SQL backend.**

### Dashboard (`dashboard.jsx` → `DashboardView`)
Scrolling page (shared `PageHead` title/subtitle/action + `Scroll` wrapper w/ watermark):
- **KPI tiles** (`DashKpiTile`): responsive grid `minmax(220px,1fr)`, radius 18, count-up
  values, delta row with up/down arrow colored `--ok`/`--err` "vs last month".
- **Chart cards**: 2-col grid (`minmax(380px,1fr)`) — "Inspection score trend" and "Open
  corrective actions by site", each with an "Open" pill that runs that query in chat.
- **Recent queries** list: card with rows (icon + question + tag pill + chevron) that open
  the query in chat.
- Page action: **"Ask OctaAI"** pill.

### Saved Queries (`dashboard.jsx` → `SavedView`)
Card grid (`minmax(320px,1fr)`): each card has a bookmark icon (per-card color), tag pill,
run count, name, a **mini chart preview** (pointer-events off), and a footer with "Run
query" pill + copy button. Hover lifts the card.

### Schema (`dashboard.jsx` → `SchemaView`)
Search field in the page action; an accordion list of 8 tables. Each row: db-icon tile,
mono table name, description, mono row-count; expands to a grid of column chips (mono name +
accent type badge). Data from `D.SCHEMA` (8 EHS tables — see `data.js`).

### Debug Panel (`debugpanel.jsx` → `DebugPanel`)
Right rail shown on the chat tab when a result is active. Header (pulsing red "DEBUG" dot +
call/token count). Collapsible **`Section`s**:
- **Banner**: ok/warn tinted status strip ("Executed cleanly on first attempt", etc.).
- **Pipeline Trace**: vertical timeline of steps (parse → resolve schema → generate →
  validate → execute → summarize), each with a status dot (`--ok`/`--warn`/`--err`), step
  name, ms timing, a mono model badge, and optional note; rows rise-in staggered.
- **Generated SQL** (`SqlBlock`): dark code block (`--code-bg`) with a **naive SQL syntax
  highlighter** (keywords/strings/numbers/comments colored — see the `.sql-*` classes in
  `OptimaX.html`) and a copy button.
- **Model Reasoning**: italic quote with an accent left border.
- **Performance**: 2-col grid of `MetricTile`s (response time, LLM calls, prompt tokens,
  completion, SQL errors, total tokens).
- **Session Details**: collapsible key/value table + "Live · pipeline healthy" indicator.

### Charts (`charts.jsx` → `OptimaXChart`)
A lightweight **dependency-free responsive SVG chart** supporting `line` / `area` / `bar`.
Props `{ type, x[], series[{name,color,data}], height }`. Features: ResizeObserver-based
responsive width; "nice" y-axis ceiling + 4 gridlines with mono labels; smooth
Catmull-Rom-ish line path (`smoothPath`); area gradient fill; grouped bars; hover
tooltip (dark `--rail` card) + hover guide line; legend; **entrance animation** (progress
0→1 cubic over ~760ms, bars stagger), respecting reduced-motion / anim-off.
- In Angular you may keep this as a standalone SVG component, or swap for the codebase's
  existing chart library — but match the visual style (smooth lines, soft gradient area
  fills, rounded bars, mono axis labels, dark tooltip) and the data-viz palette.

### Icons (`icons.jsx` → icon component/set)
A compact inline **stroke icon set** (`<Icon name size color stroke />`), single-path SVGs
in a 24-viewBox, `stroke-width` default 2, round caps/joins. Names used across the app:
chat, dashboard, list, report, config, star, range, rows, warn, site, permit, incident,
send, mic, copy, db, table, download, search, bookmark, play, refresh, sparkle, axes,
expand, shrink, plus, check, moon, sun, chev{L,R,Up,Down}, arrow{Up,Down}, etc.
- **Map to the codebase's existing icon set** if one exists; otherwise port these paths.

### Data (`data.js` → backend + models)
`window.OPTIMAX_DATA` holds all mock content: `SCHEMA` (8 tables w/ columns), `EX` (canned
query examples — each with `q, sql, reasoning, summary, kpis[], chart{type,best,x,series[]},
table{cols,rows,total}, debug{...trace[]}`), `SUGGESTIONS`, `CONVERSATIONS{today,earlier}`,
`SAVED`, `DASH_KPIS`. **This is all mock data** — replace with real API responses; use the
shapes here as a guide to the view-model each component expects.

---

## Animations (summary)
From `theme.css` keyframes + component-level CSS:
- `om-spin` rotate 360° · `om-pulse` opacity .55↔1 · `om-float` translateY ±8 ·
  `om-rise` slide up 12px · `om-fade-rise` fade+slide · `om-pop` scale bounce ·
  `om-draw` stroke draw-on · `om-worm-run` the "p" mark worm (dashoffset travel) ·
  `om-text-in` status text fade · `om-boot-fill` progress fill.
- Brand **mark** (`OptimaXMark` in `loader.jsx`) has 3 modes: `static`, `draw` (strokes
  draw on), `spin` (a bright "worm" segment travels the full "p" path — up the stem, a full
  loop around the bowl, then out along the flourish; `OM_WORM` path, `dasharray 26 74`,
  1.3s linear). Used as the inline thinking loader.
- Loading-screen specifics (halo/blob/equalizer/rings/tail) documented in Screen 2.
- Everything respects `prefers-reduced-motion` and an in-app `data-anim="off"` switch.

## Design Tokens (`theme.css`)
**Brand blue ramp:** `--blue-50 #EEF1FF · 100 #E0E6FF · 200 #C2CCFF · 300 #94A5FF ·
400 #5C73FF · 500 #2F4CFF · 600 #1B43FF (primary) · 700 #1535D6 · 800 #1129A8`.

**Data-viz palette** (oklch, shared chroma): blue `oklch(.55 .22 264)`, coral
`oklch(.68 .18 27)`, teal `oklch(.70 .13 178)`, amber `oklch(.79 .15 78)`, violet
`oklch(.60 .20 300)`, pink `oklch(.68 .18 350)`. **Status:** ok `oklch(.66 .16 158)`,
warn `oklch(.78 .16 75)`, err `oklch(.62 .21 25)`.

**Light theme:** bg `#FBFBFD` (grad `#FFFFFF`→`#F4F6FC`); surface `#FFFFFF` / `#F7F8FC` /
`#EFF1F8`; border `#E7E9F2` / `#DDE0EE`; text `#0B0D16` / `#565E76` / `#8A91A8`; accent
`#1B43FF`; accent-soft `#EEF1FF` (border `#E0E6FF`); rails (dark navy chrome) `#0C1024` /
`#141A36` (border `#232a4d`, text `#EEF1FF` / `#9aa3c8`); code-bg `#0C1024`.

**Dark theme:** bg `#080B16` (grad `#0B0F1F`→`#070A14`); surface `#11162A`/`#161D36`/`#1D2647`;
border `#232C4C`; text `#F2F4FC`/`#A4ADCF`/`#6F789B`; accent `#4D6BFF`; rails `#070A16`/`#0E1430`.

**Radii:** xs 8 · sm 12 · md 16 · lg 22 · pill 999 (cards mostly 16–18; canvas 28).
**Shadows (light):** sm `0 1px 2px rgba(13,18,40,.05), 0 1px 3px rgba(13,18,40,.04)` ·
md `0 4px 16px rgba(13,18,40,.07), 0 1px 4px rgba(13,18,40,.04)` ·
lg `0 18px 50px rgba(13,18,40,.13), 0 4px 14px rgba(13,18,40,.06)`.
**Type:** sans **Plus Jakarta Sans** (400–800); mono **JetBrains Mono**.
**Easing:** `--ease: cubic-bezier(.22,.61,.36,1)`.
**Glass** (thinking card): bg `rgba(255,255,255,.62)`, `backdrop-filter: blur(16px) saturate(150%)`.
**Pill button** classes: `.pill` (accent), `.pill.ghost`, `.pill.subtle` — see `theme.css`.

## Brand Marks (two exist — pick the canonical one)
- **Piserve "p" mark** (used in workspace + landing + footer): rounded **stem** `M34 36 V87`,
  **bowl** circle `cx52 cy42 r18`, rising **flourish** `M67 31 L85 12`; gradient stroke
  `--blue-700`→`--blue-600`→`--blue-400`. Full source + 3 animation modes in `loader.jsx`
  (`OptimaXMark`).
- **Octagon "Octa" mark** (used only on the loading screen): regular octagon outline,
  gradient stroke (points listed in Screen 2).
- Confirm with design which is the real OctaAI logo and use it consistently.

## Assets
**No external/raster assets** — everything is inline SVG + CSS. Fonts via Google Fonts
(Plus Jakarta Sans, JetBrains Mono). Use the app's existing icon set + font loading where
possible. (The extra Google Font `<link>`s in `OptimaX.html` head — Oswald, Abril Fatface,
etc. — are leftover from font experiments and not used by the design; ignore them.)

## Files in this handoff
Prototype source (design reference):
- `OptimaX Landing.html` — landing page (Screen 1)
- `OctaAI Loading.html` — loading/splash screen (Screen 2)
- `OptimaX.html` — workspace entry (loads the scripts below; Screen 3)
- `app.jsx` — shell: header, rails, tab routing, theme, tweaks
- `workspace.jsx` — AI chat: messages, KPIs, chart/table panels, query bar, thinking
- `dashboard.jsx` — DashboardView · SavedView · SchemaView
- `charts.jsx` — `OptimaXChart` SVG chart component
- `debugpanel.jsx` — debug rail: trace, SQL highlighter, metrics, session
- `loader.jsx` — brand mark (3 modes) + background watermark + thinking loaders
- `icons.jsx` — stroke icon set
- `tweaks-panel.jsx` — in-prototype tweak controls (NOT product UI; ignore for the port)
- `data.js` — all mock data (replace with real API)
- `theme.css` — the full design-token system (light + dark) + utility classes + keyframes

> `tweaks-panel.jsx` is a prototyping aid (lets a reviewer toggle theme/accent/motion) — it
> is **not** part of the product and should not be ported.
