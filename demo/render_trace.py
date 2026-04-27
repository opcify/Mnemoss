# ruff: noqa: E501
# ^ the _JS and _CSS module-level strings are inline frontend assets —
#   they naturally run longer than 100 chars per line. Re-flowing them
#   hurts readability; this file is 80% embedded browser code.
"""Render a :class:`~demo.types.Trace` as a self-contained HTML page.

Output is a single ``.html`` file: vanilla JS, inline CSS, embedded
trace JSON. No external requests. Safe to commit to the repo, safe
to embed in the blog post via ``<iframe>`` or direct file open.

Player behavior:

- Auto-plays on load; play/pause/restart controls available.
- Left pane: chat transcript, lines appear at each event's ``t``.
- Right pane: memory list, populated on ``observe`` events.
- On ``recall`` events: the target query is highlighted, and a
  stacked activation bar renders for the top hit (if
  ``breakdown`` is present — Mnemoss supplies it, StaticFileBackend
  doesn't).
- Speed toggle (1x / 2x / 4x) for quick replay.

This is intentionally not a framework-heavy UI. ~300 lines of hand-
rolled JS beats dragging in React/Svelte for something that renders
~40 events.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

from demo.types import Trace

# ─── inline assets ─────────────────────────────────────────────────

_CSS = """
:root {
  --bg: #0f1115;
  --fg: #e6e9ef;
  --fg-dim: #9aa3b2;
  --panel: #151821;
  --panel-border: #242938;
  --user: #3b82f6;
  --assistant: #10b981;
  --accent: #f59e0b;
  --breakdown-base: #4f46e5;
  --breakdown-spread: #06b6d4;
  --breakdown-match: #10b981;
  --breakdown-noise: #6b7280;
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 24px;
  background: var(--bg); color: var(--fg);
  min-height: 100vh;
}
h1 { font-size: 20px; margin: 0 0 4px 0; font-weight: 600; }
.subtitle { color: var(--fg-dim); font-size: 13px; margin-bottom: 20px; }

.controls {
  display: flex; gap: 8px; align-items: center;
  margin-bottom: 16px; flex-wrap: wrap;
}
.controls button {
  background: var(--panel); color: var(--fg);
  border: 1px solid var(--panel-border);
  padding: 6px 14px; border-radius: 6px;
  cursor: pointer; font-size: 13px;
  font-family: inherit;
}
.controls button:hover { background: var(--panel-border); }
.controls button.active { border-color: var(--accent); color: var(--accent); }
.progress {
  flex: 1; height: 6px; background: var(--panel);
  border-radius: 3px; overflow: hidden; min-width: 120px;
}
.progress-bar {
  height: 100%; background: var(--accent);
  width: 0%; transition: width 120ms linear;
}
.progress-label {
  color: var(--fg-dim); font-size: 12px;
  font-variant-numeric: tabular-nums; min-width: 80px;
}

.stage {
  display: grid;
  grid-template-columns: 1fr 380px;
  gap: 16px;
  min-height: 500px;
}
@media (max-width: 760px) { .stage { grid-template-columns: 1fr; } }

.panel {
  background: var(--panel);
  border: 1px solid var(--panel-border);
  border-radius: 8px;
  padding: 16px;
  overflow-y: auto;
  max-height: 70vh;
}
.panel h2 {
  font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em;
  color: var(--fg-dim); margin: 0 0 12px 0; font-weight: 600;
}

.chat-row {
  margin-bottom: 12px; opacity: 0;
  transition: opacity 220ms ease-in;
}
.chat-row.visible { opacity: 1; }
.chat-role {
  font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em;
  color: var(--fg-dim); margin-bottom: 2px;
}
.chat-role.user { color: var(--user); }
.chat-role.assistant { color: var(--assistant); }
.chat-bubble { font-size: 14px; line-height: 1.5; }

.memory-row {
  border-top: 1px solid var(--panel-border);
  padding: 10px 0; font-size: 13px;
  opacity: 0; transition: opacity 220ms;
}
.memory-row.visible { opacity: 1; }
.memory-row:first-child { border-top: none; padding-top: 0; }
.memory-role {
  font-size: 11px; color: var(--fg-dim);
  text-transform: uppercase; letter-spacing: 0.06em;
}
.memory-text { color: var(--fg); margin-top: 2px; }
.memory-row.recalled { background: rgba(245, 158, 11, 0.08); padding-left: 8px; padding-right: 8px; border-radius: 4px; }

.recall-panel {
  background: rgba(245, 158, 11, 0.08);
  border: 1px solid rgba(245, 158, 11, 0.3);
  padding: 10px 12px; border-radius: 6px;
  margin-bottom: 12px;
  opacity: 0; transition: opacity 220ms;
}
.recall-panel.visible { opacity: 1; }
.recall-panel .label {
  font-size: 11px; color: var(--accent);
  text-transform: uppercase; letter-spacing: 0.08em;
  margin-bottom: 4px;
}
.recall-panel .query { font-size: 13px; color: var(--fg); margin-bottom: 8px; }
.recall-panel .hit { font-size: 12px; color: var(--fg-dim); margin-bottom: 4px; display: flex; gap: 8px; align-items: center; }
.recall-panel .hit-rank { color: var(--accent); font-weight: 600; min-width: 20px; }
.recall-panel .hit-score { color: var(--fg-dim); font-variant-numeric: tabular-nums; }

.breakdown {
  display: flex; height: 8px; border-radius: 4px; overflow: hidden;
  margin-top: 6px;
}
.breakdown-seg {
  height: 100%;
}
.breakdown-legend {
  display: flex; gap: 10px; font-size: 10px; color: var(--fg-dim);
  margin-top: 4px; flex-wrap: wrap;
}
.breakdown-legend .swatch {
  display: inline-block; width: 8px; height: 8px; border-radius: 2px;
  margin-right: 4px; vertical-align: middle;
}

footer {
  margin-top: 20px; color: var(--fg-dim); font-size: 11px;
  text-align: center;
}
"""


_JS = """
(function() {
  const el = document.getElementById('trace-data');
  const trace = JSON.parse(el.textContent);
  const events = trace.events;
  const total_t = events.length ? events[events.length - 1].t + 1 : 1;

  const chatPane = document.getElementById('chat');
  const memPane = document.getElementById('memory');
  const progBar = document.getElementById('progress-bar');
  const progLabel = document.getElementById('progress-label');
  const btnPlay = document.getElementById('play');
  const btnRestart = document.getElementById('restart');
  const speedButtons = [...document.querySelectorAll('[data-speed]')];

  let playing = false;
  let speed = 1;
  let nextIdx = 0;
  let simulatedTime = 0;
  let lastWall = 0;
  let rafId = null;
  const memoryIds = new Set();

  function reset() {
    nextIdx = 0;
    simulatedTime = 0;
    chatPane.innerHTML = '';
    memPane.innerHTML = '';
    memoryIds.clear();
    progBar.style.width = '0%';
    progLabel.textContent = `0.0s / ${total_t.toFixed(1)}s`;
  }

  function renderMemoryRow(ev) {
    if (memoryIds.has(ev.memory_id)) return;
    memoryIds.add(ev.memory_id);
    const row = document.createElement('div');
    row.className = 'memory-row';
    row.dataset.memoryId = ev.memory_id;
    row.innerHTML = `
      <div class="memory-role">${ev.role || 'memory'}</div>
      <div class="memory-text"></div>
    `;
    row.querySelector('.memory-text').textContent = ev.content || '';
    memPane.appendChild(row);
    requestAnimationFrame(() => row.classList.add('visible'));
  }

  function renderChatRow(role, content) {
    const row = document.createElement('div');
    row.className = 'chat-row';
    row.innerHTML = `
      <div class="chat-role ${role}">${role}</div>
      <div class="chat-bubble"></div>
    `;
    row.querySelector('.chat-bubble').textContent = content;
    chatPane.appendChild(row);
    requestAnimationFrame(() => row.classList.add('visible'));
    chatPane.scrollTop = chatPane.scrollHeight;
  }

  function renderRecall(ev) {
    const top = ev.hits && ev.hits.length ? ev.hits[0] : null;
    const panel = document.createElement('div');
    panel.className = 'recall-panel';
    let breakdownHtml = '';
    if (top && top.breakdown) {
      const b = top.breakdown;
      // Keys from ActivationBreakdown.to_dict(): base_level, spreading,
      // matching, noise, total, idx_priority, w_f, w_s, query_bias.
      const base = Math.max(0, b.base_level || 0);
      const spread = Math.max(0, b.spreading || 0);
      const match = Math.max(0, b.matching || 0);
      const noise = Math.abs(b.noise || 0);
      const sum = base + spread + match + noise || 1;
      const pct = (x) => (100 * x / sum).toFixed(1);
      breakdownHtml = `
        <div class="breakdown">
          <div class="breakdown-seg" style="width:${pct(base)}%;background:var(--breakdown-base)"></div>
          <div class="breakdown-seg" style="width:${pct(spread)}%;background:var(--breakdown-spread)"></div>
          <div class="breakdown-seg" style="width:${pct(match)}%;background:var(--breakdown-match)"></div>
          <div class="breakdown-seg" style="width:${pct(noise)}%;background:var(--breakdown-noise)"></div>
        </div>
        <div class="breakdown-legend">
          <span><span class="swatch" style="background:var(--breakdown-base)"></span>B_i ${base.toFixed(2)}</span>
          <span><span class="swatch" style="background:var(--breakdown-spread)"></span>spread ${spread.toFixed(2)}</span>
          <span><span class="swatch" style="background:var(--breakdown-match)"></span>match ${match.toFixed(2)}</span>
          <span><span class="swatch" style="background:var(--breakdown-noise)"></span>noise ${noise.toFixed(2)}</span>
        </div>
      `;
    }
    const hitsHtml = (ev.hits || []).slice(0, 3).map(h => {
      const s = (h.score == null) ? '-' : Number(h.score).toFixed(3);
      return `<div class="hit"><span class="hit-rank">#${h.rank}</span><span class="hit-score">A=${s}</span></div>`;
    }).join('');
    panel.innerHTML = `
      <div class="label">recall</div>
      <div class="query"></div>
      ${hitsHtml}
      ${breakdownHtml}
    `;
    panel.querySelector('.query').textContent = ev.query || '';
    chatPane.appendChild(panel);
    requestAnimationFrame(() => panel.classList.add('visible'));

    // Highlight the recalled memories in the right pane briefly.
    (ev.hits || []).forEach(h => {
      const m = memPane.querySelector(`[data-memory-id="${h.memory_id}"]`);
      if (m) {
        m.classList.add('recalled');
        setTimeout(() => m.classList.remove('recalled'), 2400 / speed);
      }
    });
    chatPane.scrollTop = chatPane.scrollHeight;
  }

  function processEvent(ev) {
    if (ev.kind === 'observe') {
      // Observe populates the memory list (right). Chat transcript
      // is driven by agent_response events + user observe.
      if (ev.role === 'user') {
        renderChatRow('user', ev.content || '');
      }
      renderMemoryRow(ev);
    } else if (ev.kind === 'agent_response') {
      renderChatRow('assistant', ev.content || '');
    } else if (ev.kind === 'recall') {
      renderRecall(ev);
    }
  }

  function tick(wallNow) {
    if (!playing) { rafId = null; return; }
    const dt = (wallNow - lastWall) / 1000;
    lastWall = wallNow;
    simulatedTime += dt * speed;

    while (nextIdx < events.length && events[nextIdx].t <= simulatedTime) {
      processEvent(events[nextIdx]);
      nextIdx += 1;
    }

    const ratio = Math.min(1, simulatedTime / total_t);
    progBar.style.width = (ratio * 100).toFixed(2) + '%';
    progLabel.textContent = `${simulatedTime.toFixed(1)}s / ${total_t.toFixed(1)}s`;

    if (nextIdx >= events.length && simulatedTime >= total_t) {
      playing = false;
      btnPlay.textContent = 'Replay';
      return;
    }
    rafId = requestAnimationFrame(tick);
  }

  function start() {
    if (nextIdx >= events.length) reset();
    playing = true;
    btnPlay.textContent = 'Pause';
    lastWall = performance.now();
    rafId = requestAnimationFrame(tick);
  }
  function pause() {
    playing = false;
    btnPlay.textContent = 'Play';
    if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
  }

  btnPlay.addEventListener('click', () => playing ? pause() : start());
  btnRestart.addEventListener('click', () => { pause(); reset(); start(); });
  speedButtons.forEach(b => {
    b.addEventListener('click', () => {
      speed = parseFloat(b.dataset.speed);
      speedButtons.forEach(x => x.classList.toggle('active', x === b));
    });
  });
  // Default speed button "1x" active.
  const def = speedButtons.find(b => b.dataset.speed === '1');
  if (def) def.classList.add('active');

  // Auto-start after a tiny delay so the layout settles.
  reset();
  setTimeout(start, 150);
})();
"""


def render_html(trace: Trace, *, title: str = "Mnemoss simulation") -> str:
    """Produce a single-file HTML string embedding ``trace``.

    The trace JSON is embedded in a ``<script id="trace-data">`` tag
    rather than serialized into JS directly so escaping is handled
    naturally by ``</script>``-safe encoding.
    """

    safe_title = html.escape(title)
    scen = trace.scenario
    subtitle = (
        f"backend: {html.escape(trace.backend)} · llm: {html.escape(trace.llm)} · "
        f"{len(trace.events)} events · {trace.duration_seconds:.2f}s recording"
    )

    # Encode trace JSON with </script> protection. Browsers parse
    # <script> greedily, so we escape the sentinel.
    trace_json = json.dumps(trace.to_dict(), ensure_ascii=False, separators=(",", ":"))
    trace_json = trace_json.replace("</", "<\\/")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{safe_title}</title>
<style>{_CSS}</style>
</head>
<body>
<h1>{safe_title}</h1>
<div class="subtitle">{subtitle}</div>
<div class="subtitle">{html.escape(scen.description)}</div>

<div class="controls">
  <button id="play">Play</button>
  <button id="restart">Restart</button>
  <button data-speed="1">1x</button>
  <button data-speed="2">2x</button>
  <button data-speed="4">4x</button>
  <div class="progress"><div id="progress-bar" class="progress-bar"></div></div>
  <div id="progress-label" class="progress-label">0.0s / 0.0s</div>
</div>

<div class="stage">
  <div id="chat" class="panel"><h2>conversation</h2></div>
  <div id="memory" class="panel"><h2>memory</h2></div>
</div>

<footer>
  Mnemoss · <a href="https://github.com/opcify/mnemoss" style="color:var(--fg-dim)">github.com/opcify/mnemoss</a>
</footer>

<script id="trace-data" type="application/json">{trace_json}</script>
<script>{_JS}</script>
</body>
</html>
"""


# ─── CLI ───────────────────────────────────────────────────────────


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render a simulation trace JSON as a self-contained HTML page."
    )
    p.add_argument("trace", type=Path, help="Trace JSON path (from demo.simulate).")
    p.add_argument("--out", type=Path, required=True, help="Output HTML path.")
    p.add_argument("--title", default="Mnemoss simulation")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    data = json.loads(args.trace.read_text())
    trace = Trace.from_dict(data)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_html(trace, title=args.title))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
