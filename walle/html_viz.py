"""Interactive Web UI visualizer for mission runs.

Generates a standalone, zero-dependency HTML5 application with an interactive
time-scrubber, probability heatmap, path playback, telemetry gauges, and
agent reasoning inspector.
"""
import json
from pathlib import Path
import numpy as np

from .trace import _default

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>WALL-E & EVE Mission Visualizer</title>
<style>
  :root {
    --bg-dark: #090d16;
    --card-bg: #111827;
    --card-border: #1f293d;
    --text-primary: #f8fafc;
    --text-muted: #94a3b8;
    --accent-blue: #3b82f6;
    --accent-green: #10b981;
    --accent-amber: #f59e0b;
    --accent-orange: #f97316;
    --accent-purple: #a855f7;
    --accent-red: #ef4444;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background-color: var(--bg-dark);
    color: var(--text-primary);
    display: flex;
    flex-direction: column;
    height: 100vh;
    overflow: hidden;
  }

  /* Header */
  header {
    background-color: var(--card-bg);
    border-bottom: 1px solid var(--card-border);
    padding: 10px 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-shrink: 0;
  }
  .title-group { display: flex; align-items: center; gap: 12px; }
  .title-group h1 { font-size: 1.15rem; font-weight: 700; letter-spacing: -0.02em; }
  .badge {
    padding: 3px 8px;
    border-radius: 9999px;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  .badge-found { background: #064e3b; color: #34d399; border: 1px solid #059669; }
  .badge-active { background: #1e3a8a; color: #60a5fa; border: 1px solid #2563eb; }
  .badge-other { background: #7f1d1d; color: #f87171; border: 1px solid #dc2626; }

  /* Main Container */
  .main-layout {
    display: flex;
    flex: 1;
    overflow: hidden;
  }

  /* Left/Center: Viewport & Controls */
  .viewport-panel {
    flex: 1.25;
    display: flex;
    flex-direction: column;
    padding: 16px;
    gap: 12px;
    overflow: hidden;
  }
  .canvas-wrapper {
    flex: 1;
    background: #060911;
    border: 1px solid var(--card-border);
    border-radius: 8px;
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
  }
  canvas#gridCanvas {
    box-shadow: 0 4px 20px rgba(0,0,0,0.5);
    border-radius: 4px;
    cursor: crosshair;
  }
  .tooltip {
    position: absolute;
    background: rgba(15, 23, 42, 0.95);
    border: 1px solid #334155;
    color: #f1f5f9;
    padding: 6px 10px;
    border-radius: 6px;
    font-size: 0.75rem;
    pointer-events: none;
    display: none;
    z-index: 100;
    box-shadow: 0 4px 12px rgba(0,0,0,0.6);
  }

  /* Controls */
  .control-bar {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 8px;
    padding: 10px 14px;
    display: flex;
    align-items: center;
    gap: 12px;
    flex-shrink: 0;
  }
  .btn {
    background: #1e293b;
    border: 1px solid #334155;
    color: #f8fafc;
    padding: 6px 12px;
    border-radius: 6px;
    font-size: 0.8rem;
    font-weight: 600;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    transition: all 0.15s ease;
  }
  .btn:hover { background: #334155; border-color: #475569; }
  .btn-primary { background: var(--accent-blue); border-color: #2563eb; }
  .btn-primary:hover { background: #2563eb; }
  .btn:active { transform: scale(0.97); }

  .scrubber-container {
    flex: 1;
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .scrubber-container input[type=range] {
    flex: 1;
    accent-color: var(--accent-blue);
    cursor: pointer;
  }
  .step-label {
    font-family: monospace;
    font-size: 0.85rem;
    min-width: 90px;
    color: var(--text-muted);
  }

  /* Right: Telemetry & Reasoning */
  .telemetry-panel {
    flex: 1;
    min-width: 360px;
    max-width: 480px;
    background: var(--card-bg);
    border-left: 1px solid var(--card-border);
    display: flex;
    flex-direction: column;
    overflow-y: auto;
  }
  .tab-bar {
    display: flex;
    border-bottom: 1px solid var(--card-border);
    background: #0d131f;
  }
  .tab-btn {
    flex: 1;
    padding: 10px;
    background: transparent;
    border: none;
    color: var(--text-muted);
    font-size: 0.8rem;
    font-weight: 600;
    cursor: pointer;
    border-bottom: 2px solid transparent;
  }
  .tab-btn.active {
    color: var(--text-primary);
    border-bottom-color: var(--accent-blue);
    background: var(--card-bg);
  }

  .panel-section {
    padding: 14px 16px;
    border-bottom: 1px solid var(--card-border);
  }
  .section-title {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    font-weight: 700;
    margin-bottom: 10px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  /* Metrics Grid */
  .metrics-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
    margin-bottom: 12px;
  }
  .metric-card {
    background: #0a0f1d;
    border: 1px solid var(--card-border);
    border-radius: 6px;
    padding: 8px 10px;
  }
  .metric-label { font-size: 0.68rem; color: var(--text-muted); }
  .metric-value { font-size: 1.05rem; font-weight: 700; font-family: monospace; margin-top: 2px; }

  /* Reasoning Box */
  .reasoning-card {
    background: #0f172a;
    border-left: 4px solid var(--accent-blue);
    border-radius: 4px;
    padding: 10px 12px;
    font-size: 0.82rem;
    line-height: 1.45;
    margin-bottom: 10px;
  }
  .strategy-tag {
    display: inline-block;
    padding: 2px 7px;
    border-radius: 4px;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    margin-bottom: 6px;
  }

  /* Candidates Table */
  .candidates-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.72rem;
    font-family: monospace;
  }
  .candidates-table th {
    text-align: left;
    color: var(--text-muted);
    padding: 4px 6px;
    border-bottom: 1px solid var(--card-border);
  }
  .candidates-table td {
    padding: 5px 6px;
    border-bottom: 1px solid #172033;
  }

  /* Status Bars */
  .status-bars { display: flex; flex-direction: column; gap: 6px; }
  .status-bar-row { display: flex; align-items: center; gap: 8px; font-size: 0.75rem; }
  .status-bar-name { width: 65px; color: var(--text-muted); }
  .status-bar-track { flex: 1; height: 6px; background: #1e293b; border-radius: 3px; overflow: hidden; }
  .status-bar-fill { height: 100%; border-radius: 3px; transition: width 0.2s ease; }

  /* Timeline Canvas */
  .timeline-container {
    height: 90px;
    background: #090e1a;
    border: 1px solid var(--card-border);
    border-radius: 6px;
    position: relative;
    cursor: pointer;
  }
  canvas#timelineCanvas { width: 100%; height: 100%; display: block; }

  /* Legend */
  .legend-bar {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    font-size: 0.72rem;
    color: var(--text-muted);
    padding: 4px 0;
  }
  .legend-item { display: flex; align-items: center; gap: 5px; }
  .legend-dot { width: 8px; height: 8px; border-radius: 50%; }

  .toggle-switch {
    display: flex;
    align-items: center;
    gap: 6px;
    cursor: pointer;
    font-size: 0.75rem;
    color: var(--text-muted);
  }
  .toggle-switch input { cursor: pointer; }
</style>
</head>
<body>

<header>
  <div class="title-group">
    <span style="font-size: 1.3rem;">🤖</span>
    <div>
      <h1 id="missionTitle">WALL-E & EVE Mission</h1>
      <div style="font-size: 0.72rem; color: var(--text-muted);" id="missionMeta">Loading...</div>
    </div>
    <span id="outcomeBadge" class="badge">--</span>
  </div>
  <div style="display: flex; gap: 10px; align-items: center;">
    <label class="toggle-switch">
      <input type="checkbox" id="evalToggle"> Show Ground Truth (Evaluation)
    </label>
    <label class="toggle-switch">
      <input type="checkbox" id="heatmapToggle" checked> Heatmap
    </label>
    <button class="btn" id="openReportBtn">📄 Report</button>
  </div>
</header>

<div class="main-layout">
  <!-- Viewport -->
  <div class="viewport-panel">
    <div class="canvas-wrapper">
      <canvas id="gridCanvas"></canvas>
      <div id="cellTooltip" class="tooltip"></div>
    </div>

    <!-- Scrubber & Controls -->
    <div class="control-bar">
      <button class="btn" id="btnFirst" title="First step">⏮</button>
      <button class="btn" id="btnPrev" title="Previous step">◀</button>
      <button class="btn btn-primary" id="btnPlay" title="Play / Pause">▶ Play</button>
      <button class="btn" id="btnNext" title="Next step">▶</button>
      <button class="btn" id="btnLast" title="Last step">⏭</button>
      
      <div class="scrubber-container">
        <input type="range" id="stepScrubber" min="0" max="1" value="0">
        <span class="step-label" id="stepCounter">Step 0 / 0</span>
      </div>

      <select id="speedSelect" class="btn" style="padding: 5px 8px;">
        <option value="1">1x Speed</option>
        <option value="2" selected>2x Speed</option>
        <option value="5">5x Speed</option>
        <option value="10">10x Speed</option>
      </select>
    </div>

    <!-- Legend -->
    <div class="legend-bar">
      <div class="legend-item"><div class="legend-dot" style="background: #d4a017;"></div> Station</div>
      <div class="legend-item"><div class="legend-dot" style="background: #a855f7;"></div> Clue</div>
      <div class="legend-item"><div class="legend-dot" style="background: #10b981;"></div> EVE (Ground Truth)</div>
      <div class="legend-item"><div class="legend-dot" style="background: #ef4444;"></div> Decoy (Ground Truth)</div>
      <div class="legend-item"><div class="legend-dot" style="background: #3b82f6;"></div> Localize</div>
      <div class="legend-item"><div class="legend-dot" style="background: #10b981;"></div> Approach</div>
      <div class="legend-item"><div class="legend-dot" style="background: #f97316;"></div> Return</div>
      <div class="legend-item"><div class="legend-dot" style="background: #eab308;"></div> Recharge</div>
    </div>
  </div>

  <!-- Telemetry HUD -->
  <div class="telemetry-panel">
    <!-- Top Gauges -->
    <div class="panel-section">
      <div class="section-title">Telemetry & Resources</div>
      <div class="metrics-grid">
        <div class="metric-card">
          <div class="metric-label">BATTERY</div>
          <div class="metric-value" id="valBattery" style="color: var(--accent-orange);">--</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">TIME LEFT</div>
          <div class="metric-value" id="valTime">--</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">ENTROPY</div>
          <div class="metric-value" id="valEntropy" style="color: var(--accent-blue);">--</div>
        </div>
      </div>

      <!-- Inferred Condition -->
      <div class="status-bars">
        <div class="status-bar-row">
          <span class="status-bar-name">Active</span>
          <div class="status-bar-track"><div id="barActive" class="status-bar-fill" style="background: var(--accent-green); width: 0%;"></div></div>
          <span id="pctActive" style="width: 38px; text-align: right; font-family: monospace;">0%</span>
        </div>
        <div class="status-bar-row">
          <span class="status-bar-name">Damaged</span>
          <div class="status-bar-track"><div id="barDamaged" class="status-bar-fill" style="background: var(--accent-amber); width: 0%;"></div></div>
          <span id="pctDamaged" style="width: 38px; text-align: right; font-family: monospace;">0%</span>
        </div>
        <div class="status-bar-row">
          <span class="status-bar-name">Dormant</span>
          <div class="status-bar-track"><div id="barDormant" class="status-bar-fill" style="background: var(--accent-red); width: 0%;"></div></div>
          <span id="pctDormant" style="width: 38px; text-align: right; font-family: monospace;">0%</span>
        </div>
      </div>
    </div>

    <!-- Agent Decision -->
    <div class="panel-section">
      <div class="section-title">
        <span>Agent Decision & Reasoning</span>
        <span id="currentStrategyTag" class="strategy-tag" style="background: #1e293b;">--</span>
      </div>
      <div class="reasoning-card" id="reasoningText">
        Agent reasoning will appear here as you scrub through steps.
      </div>
      <div style="font-size: 0.75rem; margin-bottom: 6px;">
        <strong>Action Executed:</strong> <span id="actionExecuted" style="font-family: monospace; color: var(--accent-blue);">--</span>
      </div>
      <div style="font-size: 0.75rem;">
        <strong>Observation Event:</strong> <span id="obsEvent" style="font-family: monospace; color: var(--accent-amber);">--</span>
      </div>
    </div>

    <!-- Candidate Alternatives Weighed -->
    <div class="panel-section" style="flex: 1;">
      <div class="section-title">Top Alternatives Weighed</div>
      <div style="overflow-x: auto;">
        <table class="candidates-table">
          <thead>
            <tr><th>Action</th><th>Target</th><th>Gain</th><th>Steps</th><th>Score</th></tr>
          </thead>
          <tbody id="candidatesTableBody">
            <tr><td colspan="5" style="color: var(--text-muted); text-align: center;">No candidates logged</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- Timeline Chart -->
    <div class="panel-section">
      <div class="section-title">Mission Timeline (Click to Seek)</div>
      <div class="timeline-container">
        <canvas id="timelineCanvas"></canvas>
      </div>
    </div>
  </div>
</div>

<!-- Embedded Mission Data -->
<script id="mission-data" type="application/json">
DATA_PAYLOAD_PLACEHOLDER
</script>

<script>
(function() {
  const rawElement = document.getElementById('mission-data');
  let missionData = null;
  try {
    missionData = JSON.parse(rawElement.textContent);
  } catch(e) {
    console.error("Failed to parse embedded mission data", e);
  }

  if (!missionData || !missionData.trace) {
    document.body.innerHTML = '<div style="padding: 40px; text-align: center;"><h2>No mission data embedded.</h2></div>';
    return;
  }

  const { report, trace, grid_width: W, grid_height: H, stations, snapshots, path, strategies, entropy, battery, ground_truth } = missionData;
  const totalSteps = trace.length;
  let currentStep = 0;
  let isPlaying = false;
  let playTimer = null;

  // DOM Elements
  const canvas = document.getElementById('gridCanvas');
  const ctx = canvas.getContext('2d');
  const timelineCanvas = document.getElementById('timelineCanvas');
  const tctx = timelineCanvas.getContext('2d');
  const scrubber = document.getElementById('stepScrubber');
  const stepCounter = document.getElementById('stepCounter');
  const btnPlay = document.getElementById('btnPlay');
  const btnPrev = document.getElementById('btnPrev');
  const btnNext = document.getElementById('btnNext');
  const btnFirst = document.getElementById('btnFirst');
  const btnLast = document.getElementById('btnLast');
  const speedSelect = document.getElementById('speedSelect');
  const evalToggle = document.getElementById('evalToggle');
  const heatmapToggle = document.getElementById('heatmapToggle');
  const tooltip = document.getElementById('cellTooltip');

  // Colors
  const STRAT_COLORS = {
    "localize": "#3b82f6",
    "search-clues": "#a855f7",
    "approach-eve": "#10b981",
    "return": "#f97316",
    "recharge": "#eab308",
    "finish": "#64748b",
    "random": "#94a3b8"
  };

  // Init Header
  document.getElementById('missionTitle').textContent = `Mission: ${report.mission || 'Run'}`;
  document.getElementById('missionMeta').textContent = `Agent: ${report.agent} | Seed: ${report.seed} | Grid: ${W}x${H} | Total Steps: ${report.resources.steps}`;
  const badge = document.getElementById('outcomeBadge');
  badge.textContent = report.outcome;
  badge.className = 'badge ' + (report.outcome === 'found' ? 'badge-found' : (report.outcome === 'active' ? 'badge-active' : 'badge-other'));

  scrubber.max = totalSteps;

  // Resize canvas
  function resizeCanvas() {
    const wrapper = canvas.parentElement;
    const size = Math.min(wrapper.clientWidth - 40, wrapper.clientHeight - 40, 680);
    canvas.width = size;
    canvas.height = size;
    renderGrid();
  }
  window.addEventListener('resize', () => { resizeCanvas(); renderTimeline(); });

  // Grid Cell Calculations
  function getCellSize() { return canvas.width / W; }

  // Hover Tooltip
  canvas.addEventListener('mousemove', (e) => {
    const rect = canvas.getBoundingClientRect();
    const cellSize = getCellSize();
    const x = Math.floor((e.clientX - rect.left) / cellSize);
    const y = Math.floor((e.clientY - rect.top) / cellSize);
    if (x >= 0 && x < W && y >= 0 && y < H) {
      const snap = snapshots[Math.min(currentStep, snapshots.length - 1)];
      const prob = snap ? (snap[y * W + x] * 100).toFixed(2) : '0';
      const isStation = stations.some(s => s[0] === x && s[1] === y);
      const isWall = ground_truth && ground_truth.walls && ground_truth.walls.some(w => w[0] === x && w[1] === y);
      tooltip.style.display = 'block';
      tooltip.style.left = (e.clientX - canvas.parentElement.getBoundingClientRect().left + 15) + 'px';
      tooltip.style.top = (e.clientY - canvas.parentElement.getBoundingClientRect().top + 15) + 'px';
      tooltip.innerHTML = `<strong>Cell (${x}, ${y})</strong><br>P(EVE): <b>${prob}%</b>${isStation ? '<br>⚡ Station' : ''}${isWall && evalToggle.checked ? '<br>🧱 Wall' : ''}`;
    } else {
      tooltip.style.display = 'none';
    }
  });
  canvas.addEventListener('mouseleave', () => { tooltip.style.display = 'none'; });

  // Render Grid
  function renderGrid() {
    const cs = getCellSize();
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const snap = snapshots[Math.min(currentStep, snapshots.length - 1)] || [];
    const showEval = evalToggle.checked;
    const showHeatmap = heatmapToggle.checked;

    // 1. Draw Background Cells & Heatmap
    let maxP = 0.0001;
    for (let i = 0; i < snap.length; i++) if (snap[i] > maxP) maxP = snap[i];

    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const idx = y * W + x;
        const p = snap[idx] || 0;
        
        ctx.fillStyle = '#0a0f1d';
        ctx.fillRect(x * cs, y * cs, cs, cs);

        if (showHeatmap && p > 0.001) {
          const intensity = Math.min(1, (p / maxP));
          ctx.fillStyle = `rgba(239, 68, 68, ${intensity * 0.75})`;
          ctx.fillRect(x * cs, y * cs, cs, cs);
        }

        // Cell border
        ctx.strokeStyle = '#151f32';
        ctx.lineWidth = 0.5;
        ctx.strokeRect(x * cs, y * cs, cs, cs);
      }
    }

    // 2. Discovered Walls or True Walls
    if (showEval && ground_truth && ground_truth.walls) {
      ctx.fillStyle = '#1e293b';
      ground_truth.walls.forEach(([wx, wy]) => {
        ctx.fillRect(wx * cs, wy * cs, cs, cs);
        ctx.strokeStyle = '#334155';
        ctx.lineWidth = 1;
        ctx.strokeRect(wx * cs, wy * cs, cs, cs);
      });
    }

    // 3. Charging Stations
    stations.forEach(([sx, sy]) => {
      ctx.fillStyle = '#d4a017';
      ctx.fillRect(sx * cs + cs*0.2, sy * cs + cs*0.2, cs*0.6, cs*0.6);
      ctx.strokeStyle = '#fef08a';
      ctx.lineWidth = 1.5;
      ctx.strokeRect(sx * cs + cs*0.2, sy * cs + cs*0.2, cs*0.6, cs*0.6);
      // Lightning symbol
      ctx.fillStyle = '#000';
      ctx.font = `bold ${Math.max(10, cs*0.45)}px sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('⚡', sx * cs + cs*0.5, sy * cs + cs*0.5);
    });

    // 4. Clues Discovered up to currentStep
    const rec = trace[Math.max(0, currentStep - 1)];
    const cluesSeen = [];
    for (let i = 0; i < currentStep; i++) {
      const o = trace[i].observation;
      if (o.clues) {
        o.clues.forEach(c => {
          if (!cluesSeen.some(x => x.id === c.id)) cluesSeen.push(c);
        });
      }
    }
    cluesSeen.forEach(c => {
      const [cx, cy] = c.pos;
      ctx.save();
      ctx.translate(cx * cs + cs/2, cy * cs + cs/2);
      ctx.fillStyle = '#a855f7';
      ctx.beginPath();
      ctx.moveTo(0, -cs*0.35);
      ctx.lineTo(cs*0.35, 0);
      ctx.lineTo(0, cs*0.35);
      ctx.lineTo(-cs*0.35, 0);
      ctx.closePath();
      ctx.fill();
      ctx.restore();

      // Line pointing to clue target
      if (c.target) {
        ctx.strokeStyle = 'rgba(168, 85, 247, 0.4)';
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(cx * cs + cs/2, cy * cs + cs/2);
        ctx.lineTo(c.target[0] * cs + cs/2, c.target[1] * cs + cs/2);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    });

    // 5. Evaluation Overlay: True EVE & Decoys
    if (showEval && ground_truth) {
      if (ground_truth.decoys) {
        ground_truth.decoys.forEach(([dx, dy]) => {
          ctx.strokeStyle = '#ef4444';
          ctx.lineWidth = 2.5;
          ctx.beginPath();
          ctx.moveTo(dx * cs + cs*0.2, dy * cs + cs*0.2);
          ctx.lineTo(dx * cs + cs*0.8, dy * cs + cs*0.8);
          ctx.moveTo(dx * cs + cs*0.8, dy * cs + cs*0.2);
          ctx.lineTo(dx * cs + cs*0.2, dy * cs + cs*0.8);
          ctx.stroke();
        });
      }
      if (ground_truth.eve_pos) {
        const [ex, ey] = ground_truth.eve_pos;
        ctx.fillStyle = '#10b981';
        ctx.font = `bold ${cs*0.75}px sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('★', ex * cs + cs*0.5, ey * cs + cs*0.52);
      }
    }

    // 6. Path Trail up to currentStep
    const pts = path.slice(0, currentStep + 1);
    if (pts.length > 1) {
      for (let i = 0; i < pts.length - 1; i++) {
        const strat = strategies[i] || 'localize';
        ctx.strokeStyle = STRAT_COLORS[strat] || '#94a3b8';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(pts[i][0] * cs + cs/2, pts[i][1] * cs + cs/2);
        ctx.lineTo(pts[i+1][0] * cs + cs/2, pts[i+1][1] * cs + cs/2);
        ctx.stroke();
      }
    }

    // 7. WALL-E Robot Marker
    if (pts.length > 0) {
      const cur = pts[pts.length - 1];
      ctx.fillStyle = '#38bdf8';
      ctx.beginPath();
      ctx.arc(cur[0] * cs + cs/2, cur[1] * cs + cs/2, cs*0.38, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.stroke();

      // Eyes/Icon
      ctx.fillStyle = '#0f172a';
      ctx.beginPath();
      ctx.arc(cur[0] * cs + cs*0.4, cur[1] * cs + cs*0.45, cs*0.08, 0, Math.PI * 2);
      ctx.arc(cur[0] * cs + cs*0.6, cur[1] * cs + cs*0.45, cs*0.08, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // Update HUD
  function updateHUD() {
    scrubber.value = currentStep;
    stepCounter.textContent = `Step ${currentStep} / ${totalSteps}`;

    if (currentStep === 0) {
      document.getElementById('valBattery').textContent = battery[0].toFixed(1);
      document.getElementById('valTime').textContent = report.resources.time_budget;
      document.getElementById('valEntropy').textContent = entropy[0].toFixed(2);
      document.getElementById('reasoningText').textContent = "Initial state. Waiting for first action decision.";
      document.getElementById('actionExecuted').textContent = "None (Briefing)";
      document.getElementById('obsEvent').textContent = "start";
      document.getElementById('currentStrategyTag').textContent = "START";
      document.getElementById('currentStrategyTag').style.background = "#334155";
      document.getElementById('candidatesTableBody').innerHTML = '<tr><td colspan="5" style="color: var(--text-muted); text-align: center;">Mission briefing phase</td></tr>';
      return;
    }

    const rec = trace[currentStep - 1];
    document.getElementById('valBattery').textContent = rec.before.battery.toFixed(1);
    document.getElementById('valTime').textContent = rec.before.time_left.toFixed(0);
    document.getElementById('valEntropy').textContent = rec.belief.entropy.toFixed(2);

    // Condition meters
    if (rec.belief && rec.belief.status_p) {
      const sp = rec.belief.status_p;
      const a = Math.round((sp.active || 0) * 100);
      const d = Math.round((sp.damaged || 0) * 100);
      const dm = Math.round((sp.dormant || 0) * 100);
      document.getElementById('barActive').style.width = a + '%';
      document.getElementById('pctActive').textContent = a + '%';
      document.getElementById('barDamaged').style.width = d + '%';
      document.getElementById('pctDamaged').textContent = d + '%';
      document.getElementById('barDormant').style.width = dm + '%';
      document.getElementById('pctDormant').textContent = dm + '%';
    }

    // Reasoning & Strategy
    const strat = rec.strategy || 'localize';
    const tag = document.getElementById('currentStrategyTag');
    tag.textContent = strat;
    tag.style.background = STRAT_COLORS[strat] || '#334155';

    document.getElementById('reasoningText').textContent = rec.reason || 'No reason logged.';
    
    // Action format
    let actStr = rec.action.type;
    if (rec.action.dir) actStr += ` (${rec.action.dir})`;
    document.getElementById('actionExecuted').textContent = actStr;
    document.getElementById('obsEvent').textContent = rec.observation.event || 'ok';

    // Candidates Table
    const tbody = document.getElementById('candidatesTableBody');
    if (rec.candidates && rec.candidates.length > 0) {
      tbody.innerHTML = rec.candidates.map(c => `
        <tr>
          <td><span style="color:${STRAT_COLORS[c.kind]||'#fff'}">${c.kind}</span></td>
          <td>(${c.cell[0]},${c.cell[1]})</td>
          <td>${c.gain}</td>
          <td>${c.steps}</td>
          <td><b>${c.score}</b></td>
        </tr>
      `).join('');
    } else {
      tbody.innerHTML = '<tr><td colspan="5" style="color: var(--text-muted); text-align: center;">Direct action (no candidate list)</td></tr>';
    }
  }

  // Render Timeline
  function renderTimeline() {
    const w = timelineCanvas.parentElement.clientWidth;
    const h = timelineCanvas.parentElement.clientHeight;
    timelineCanvas.width = w;
    timelineCanvas.height = h;

    tctx.clearRect(0, 0, w, h);
    if (totalSteps === 0) return;

    const maxBat = Math.max(...battery, 100);
    const maxEnt = Math.max(...entropy, 1);

    // Shading per strategy
    for (let i = 0; i < totalSteps; i++) {
      const x1 = (i / totalSteps) * w;
      const x2 = ((i + 1) / totalSteps) * w;
      const strat = strategies[i] || 'localize';
      tctx.fillStyle = (STRAT_COLORS[strat] || '#334155') + '22';
      tctx.fillRect(x1, 0, x2 - x1, h);
    }

    // Battery curve (orange)
    tctx.strokeStyle = '#f97316';
    tctx.lineWidth = 1.5;
    tctx.beginPath();
    battery.forEach((b, i) => {
      const x = (i / totalSteps) * w;
      const y = h - (b / maxBat) * (h - 10) - 5;
      if (i === 0) tctx.moveTo(x, y); else tctx.lineTo(x, y);
    });
    tctx.stroke();

    // Entropy curve (blue)
    tctx.strokeStyle = '#3b82f6';
    tctx.lineWidth = 1.5;
    tctx.beginPath();
    entropy.forEach((e, i) => {
      const x = (i / totalSteps) * w;
      const y = h - (e / maxEnt) * (h - 10) - 5;
      if (i === 0) tctx.moveTo(x, y); else tctx.lineTo(x, y);
    });
    tctx.stroke();

    // Current Step Playhead
    const curX = (currentStep / totalSteps) * w;
    tctx.strokeStyle = '#ffffff';
    tctx.lineWidth = 2;
    tctx.beginPath();
    tctx.moveTo(curX, 0);
    tctx.lineTo(curX, h);
    tctx.stroke();
  }

  // Timeline Click Seek
  timelineCanvas.parentElement.addEventListener('click', (e) => {
    const rect = timelineCanvas.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    currentStep = Math.round(ratio * totalSteps);
    onStepChange();
  });

  function onStepChange() {
    renderGrid();
    updateHUD();
    renderTimeline();
  }

  scrubber.addEventListener('input', (e) => {
    currentStep = parseInt(e.target.value, 10);
    onStepChange();
  });

  // Playback Loop
  function play() {
    if (currentStep >= totalSteps) currentStep = 0;
    isPlaying = true;
    btnPlay.textContent = '⏸ Pause';
    btnPlay.className = 'btn btn-primary';
    
    const interval = 1000 / (4 * parseFloat(speedSelect.value));
    playTimer = setInterval(() => {
      if (currentStep < totalSteps) {
        currentStep++;
        onStepChange();
      } else {
        pause();
      }
    }, interval);
  }

  function pause() {
    isPlaying = false;
    btnPlay.textContent = '▶ Play';
    btnPlay.className = 'btn btn-primary';
    if (playTimer) clearInterval(playTimer);
  }

  btnPlay.addEventListener('click', () => isPlaying ? pause() : play());
  btnPrev.addEventListener('click', () => { if (currentStep > 0) { currentStep--; onStepChange(); } });
  btnNext.addEventListener('click', () => { if (currentStep < totalSteps) { currentStep++; onStepChange(); } });
  btnFirst.addEventListener('click', () => { currentStep = 0; onStepChange(); });
  btnLast.addEventListener('click', () => { currentStep = totalSteps; onStepChange(); });
  speedSelect.addEventListener('change', () => { if (isPlaying) { pause(); play(); } });
  evalToggle.addEventListener('change', () => renderGrid());
  heatmapToggle.addEventListener('change', () => renderGrid());

  // Keyboard Navigation
  window.addEventListener('keydown', (e) => {
    if (e.code === 'Space') { e.preventDefault(); isPlaying ? pause() : play(); }
    else if (e.code === 'ArrowLeft') { if (currentStep > 0) { currentStep--; onStepChange(); } }
    else if (e.code === 'ArrowRight') { if (currentStep < totalSteps) { currentStep++; onStepChange(); } }
  });

  // Open Report Modal
  document.getElementById('openReportBtn').addEventListener('click', () => {
    const win = window.open('', '_blank');
    win.document.write(`
      <html><head><title>Mission Report: ${report.mission}</title>
      <style>
        body { font-family: monospace; background: #090d16; color: #f8fafc; padding: 24px; line-height: 1.5; }
        pre { background: #111827; padding: 16px; border-radius: 8px; border: 1px solid #1f293d; overflow: auto; }
      </style></head>
      <body>
        <h1>Mission Report: ${report.mission}</h1>
        <p><b>Outcome:</b> ${report.outcome} | <b>Location:</b> ${JSON.stringify(report.inferred_location)} | <b>Status:</b> ${JSON.stringify(report.inferred_status)}</p>
        <h2>Recommended Next Step</h2>
        <p style="background: #1e293b; padding: 12px; border-left: 4px solid #3b82f6;">${report.recommended_next_step}</p>
        <h2>Full JSON Report</h2>
        <pre>${JSON.stringify(report, null, 2)}</pre>
      </body></html>
    `);
  });

  // Initial Render
  resizeCanvas();
  onStepChange();
})();
</script>
</body>
</html>
"""


def build_payload(res, rep) -> dict:
    truth = res.env.ground_truth()
    walls_list = [[int(x), int(y)] for y, x in zip(*truth["walls"].nonzero())]

    # Convert snapshots: round to 4 decimals to keep JSON compact
    snaps = []
    for s in res.snapshots:
        snaps.append([round(float(v), 4) for v in s.ravel()])

    return {
        "report": rep,
        "trace": res.trace.records,
        "grid_width": res.agent.W,
        "grid_height": res.agent.H,
        "stations": [list(s) for s in res.agent.stations],
        "snapshots": snaps,
        "path": [list(p) for p in res.path],
        "strategies": res.strategies,
        "entropy": [round(float(e), 4) for e in res.entropy],
        "battery": [round(float(b), 2) for b in res.battery],
        "ground_truth": {
            "eve_pos": list(truth["eve_pos"]),
            "eve_status": truth["eve_status"],
            "decoys": [list(d) for d in truth["decoys"]],
            "walls": walls_list,
        },
    }


def save_html(res, rep, path: Path):
    payload = build_payload(res, rep)
    json_str = json.dumps(payload, default=_default)
    html = HTML_TEMPLATE.replace("DATA_PAYLOAD_PLACEHOLDER", json_str)
    path.write_text(html, encoding="utf-8")
    return path
