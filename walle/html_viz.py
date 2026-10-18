"""Interactive Web UI visualizer for mission runs.

Generates a standalone, zero-dependency HTML5 application with Story Mode
milestones, mission switching, interactive probability heatmap, path playback,
telemetry gauges, and agent reasoning inspector.
"""
import json
from pathlib import Path

from .trace import _default

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>WALL-E & EVE Mission Visualizer</title>
<style>
  :root {
    --bg-dark: #080c16;
    --card-bg: #0f172a;
    --card-border: #1e293b;
    --text-primary: #f8fafc;
    --text-muted: #94a3b8;
    --accent-blue: #38bdf8;
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
    padding: 10px 18px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-shrink: 0;
    gap: 16px;
  }
  .title-group { display: flex; align-items: center; gap: 12px; }
  .title-group h1 { font-size: 1.15rem; font-weight: 700; letter-spacing: -0.02em; }
  
  .badge {
    padding: 3px 8px;
    border-radius: 9999px;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  .badge-found { background: #064e3b; color: #34d399; border: 1px solid #059669; }
  .badge-active { background: #1e3a8a; color: #60a5fa; border: 1px solid #2563eb; }
  .badge-other { background: #7f1d1d; color: #f87171; border: 1px solid #dc2626; }

  /* Story Mode Ribbon */
  .story-ribbon {
    background: #090e1d;
    border-bottom: 1px solid var(--card-border);
    padding: 7px 18px;
    display: flex;
    align-items: center;
    gap: 12px;
    overflow-x: auto;
    flex-shrink: 0;
  }
  .story-ribbon-title {
    font-size: 0.72rem;
    font-weight: 800;
    color: var(--accent-blue);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    display: flex;
    align-items: center;
    gap: 5px;
    white-space: nowrap;
  }
  .milestone-btn {
    background: #131c31;
    border: 1px solid #23314d;
    color: var(--text-muted);
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 0.74rem;
    font-weight: 600;
    cursor: pointer;
    white-space: nowrap;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    transition: all 0.15s ease;
  }
  .milestone-btn:hover {
    background: #1e293b;
    border-color: var(--accent-blue);
    color: var(--text-primary);
  }
  .milestone-btn.active {
    background: rgba(56, 189, 248, 0.15);
    border-color: var(--accent-blue);
    color: var(--accent-blue);
    box-shadow: 0 0 10px rgba(56, 189, 248, 0.2);
  }

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
    padding: 14px;
    gap: 10px;
    overflow: hidden;
  }
  .canvas-wrapper {
    flex: 1;
    background: #050811;
    border: 1px solid var(--card-border);
    border-radius: 8px;
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
  }
  canvas#gridCanvas {
    box-shadow: 0 4px 20px rgba(0,0,0,0.6);
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
    padding: 8px 12px;
    display: flex;
    align-items: center;
    gap: 10px;
    flex-shrink: 0;
  }
  .btn {
    background: #1e293b;
    border: 1px solid #334155;
    color: #f8fafc;
    padding: 6px 12px;
    border-radius: 6px;
    font-size: 0.78rem;
    font-weight: 600;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    transition: all 0.15s ease;
  }
  .btn:hover { background: #334155; border-color: #475569; }
  .btn-primary { background: var(--accent-blue); border-color: #0284c7; color: #0f172a; font-weight: 700; }
  .btn-primary:hover { background: #38bdf8; }
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
    min-width: 95px;
    color: var(--text-muted);
  }

  /* Right: Telemetry & Reasoning */
  .telemetry-panel {
    flex: 1;
    min-width: 380px;
    max-width: 490px;
    background: var(--card-bg);
    border-left: 1px solid var(--card-border);
    display: flex;
    flex-direction: column;
    overflow-y: auto;
  }

  .panel-section {
    padding: 12px 16px;
    border-bottom: 1px solid var(--card-border);
  }
  .section-title {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-muted);
    font-weight: 700;
    margin-bottom: 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  /* Metrics Grid */
  .metrics-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
    margin-bottom: 10px;
  }
  .metric-card {
    background: #0a0f1d;
    border: 1px solid var(--card-border);
    border-radius: 6px;
    padding: 7px 9px;
  }
  .metric-label { font-size: 0.66rem; color: var(--text-muted); }
  .metric-value { font-size: 1.05rem; font-weight: 700; font-family: monospace; margin-top: 2px; }

  /* Safety Alert Banner */
  .safety-alert {
    display: none;
    background: rgba(249, 115, 22, 0.12);
    border: 1px solid #f97316;
    color: #fdba74;
    padding: 8px 10px;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-bottom: 10px;
    align-items: center;
    gap: 8px;
  }

  /* Reasoning Box */
  .reasoning-card {
    background: #090e1d;
    border-left: 4px solid var(--accent-blue);
    border-radius: 4px;
    padding: 9px 11px;
    font-size: 0.81rem;
    line-height: 1.45;
    margin-bottom: 8px;
  }
  .strategy-tag {
    display: inline-block;
    padding: 2px 7px;
    border-radius: 4px;
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    margin-bottom: 4px;
  }

  /* Compass Bearing Radar Widget */
  .compass-widget {
    display: flex;
    align-items: center;
    gap: 12px;
    background: #0a0f1d;
    border: 1px solid var(--card-border);
    border-radius: 6px;
    padding: 8px 12px;
    margin-bottom: 8px;
  }
  .compass-dial {
    width: 44px;
    height: 44px;
    position: relative;
    border: 1px solid #334155;
    border-radius: 50%;
    background: radial-gradient(circle, #0f172a 40%, #060911 100%);
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .compass-needle {
    width: 2px;
    height: 18px;
    background: var(--accent-blue);
    position: absolute;
    transform-origin: bottom center;
    top: 4px;
    border-radius: 1px;
    transition: transform 0.2s ease;
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
    padding: 4px 6px;
    border-bottom: 1px solid #141c2e;
  }

  /* Strategy Progress Breakdown */
  .strat-bar {
    display: flex;
    height: 6px;
    border-radius: 3px;
    overflow: hidden;
    margin-top: 6px;
    background: #1e293b;
  }
  .strat-segment { height: 100%; transition: width 0.3s ease; }

  /* Status Bars */
  .status-bars { display: flex; flex-direction: column; gap: 5px; }
  .status-bar-row { display: flex; align-items: center; gap: 8px; font-size: 0.74rem; }
  .status-bar-name { width: 65px; color: var(--text-muted); }
  .status-bar-track { flex: 1; height: 6px; background: #1e293b; border-radius: 3px; overflow: hidden; }
  .status-bar-fill { height: 100%; border-radius: 3px; transition: width 0.2s ease; }

  /* Timeline Canvas */
  .timeline-container {
    height: 80px;
    background: #080c16;
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
    padding: 2px 0;
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
    <span style="font-size: 1.35rem;">🤖</span>
    <div>
      <h1 id="missionTitle">WALL-E & EVE: Directive-EVE</h1>
      <div style="font-size: 0.72rem; color: var(--text-muted);" id="missionMeta">Autonomous Bayesian Search & Rescue</div>
    </div>
    <span id="outcomeBadge" class="badge">--</span>
  </div>

  <div style="display: flex; gap: 12px; align-items: center;">
    <!-- Mission Switcher -->
    <select id="missionSelect" class="btn" style="padding: 5px 10px; font-weight: 700; background: #131c31; border-color: #2b3b5e; color: var(--accent-blue);">
      <option value="current">Current Run</option>
      <option value="easy">Mission Easy (13 steps)</option>
      <option value="decoys">Mission Decoys (113 steps)</option>
      <option value="upload">📂 Load Custom Run...</option>
    </select>
    <input type="file" id="fileUploadInput" multiple style="display: none;">

    <label class="toggle-switch">
      <input type="checkbox" id="evalToggle"> Show Ground Truth
    </label>
    <label class="toggle-switch">
      <input type="checkbox" id="heatmapToggle" checked> Heatmap
    </label>
    <button class="btn" id="openReportBtn">📄 Report</button>
  </div>
</header>

<!-- Story Mode / Milestones Ribbon -->
<div class="story-ribbon">
  <div class="story-ribbon-title">
    <span>🎬 STORY HIGHLIGHTS:</span>
  </div>
  <div id="milestonesContainer" style="display: flex; gap: 8px; align-items: center;">
    <!-- Populated dynamically -->
  </div>
  <button class="btn" id="btnAutoTour" style="margin-left: auto; font-size: 0.7rem; padding: 3px 8px; background: rgba(56, 189, 248, 0.12); color: var(--accent-blue); border-color: var(--accent-blue);">
    ▶ Auto-Tour
  </button>
</div>

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

      <select id="speedSelect" class="btn" style="padding: 4px 6px;">
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
      <div class="legend-item"><div class="legend-dot" style="background: #38bdf8;"></div> Localize</div>
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

      <!-- Safety Alert Banner -->
      <div id="safetyBanner" class="safety-alert">
        <span>⚠️</span>
        <div><strong>CONSERVATIVE SAFETY OVERRIDE ACTIVE</strong><br><span style="font-size: 0.68rem; opacity: 0.9;">Battery reserve critical; returning home along confirmed free cells.</span></div>
      </div>

      <div class="reasoning-card" id="reasoningText">
        Agent reasoning will appear here as you scrub through steps.
      </div>

      <!-- Bearing Radar Compass -->
      <div class="compass-widget" id="compassWidget">
        <div class="compass-dial">
          <div class="compass-needle" id="compassNeedle"></div>
          <div style="font-size: 0.55rem; color: #64748b; position: absolute; top: 1px;">N</div>
        </div>
        <div>
          <div style="font-size: 0.68rem; color: var(--text-muted);">LAST RADIO SCAN BEARING</div>
          <div id="compassLabel" style="font-family: monospace; font-size: 0.85rem; font-weight: 700; color: var(--accent-blue);">--</div>
        </div>
      </div>

      <div style="display: flex; justify-content: space-between; font-size: 0.74rem;">
        <div><strong>Action:</strong> <span id="actionExecuted" style="font-family: monospace; color: var(--accent-blue);">--</span></div>
        <div><strong>Observation:</strong> <span id="obsEvent" style="font-family: monospace; color: var(--accent-amber);">--</span></div>
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
      <div class="section-title">
        <span>Mission Timeline (Click to Seek)</span>
        <span id="timelineStats" style="font-family: monospace; font-size: 0.68rem; color: var(--text-muted);">0 / 0</span>
      </div>
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
  let currentMission = null;
  try {
    currentMission = JSON.parse(rawElement.textContent);
  } catch(e) {
    console.error("Failed to parse embedded mission data", e);
  }

  if (!currentMission || !currentMission.trace) {
    document.body.innerHTML = '<div style="padding: 40px; text-align: center;"><h2>No mission data embedded.</h2></div>';
    return;
  }

  let { report, trace, grid_width: W, grid_height: H, stations, snapshots, path, strategies, entropy, battery, ground_truth } = currentMission;
  let totalSteps = trace.length;
  let currentStep = 0;
  let isPlaying = false;
  let playTimer = null;
  let isAutoTour = false;

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
  const milestonesContainer = document.getElementById('milestonesContainer');
  const missionSelect = document.getElementById('missionSelect');
  const fileUploadInput = document.getElementById('fileUploadInput');
  const safetyBanner = document.getElementById('safetyBanner');
  const compassNeedle = document.getElementById('compassNeedle');
  const compassLabel = document.getElementById('compassLabel');
  const btnAutoTour = document.getElementById('btnAutoTour');

  // Colors
  const STRAT_COLORS = {
    "localize": "#38bdf8",
    "search-clues": "#a855f7",
    "approach-eve": "#10b981",
    "return": "#f97316",
    "recharge": "#eab308",
    "finish": "#64748b",
    "random": "#94a3b8"
  };

  // Sync Header & Selector
  function initHeader() {
    document.getElementById('missionTitle').textContent = `Mission: ${report.mission || 'Run'}`;
    document.getElementById('missionMeta').textContent = `Agent: ${report.agent} | Seed: ${report.seed} | Grid: ${W}x${H} | Total Steps: ${report.resources.steps}`;
    const badge = document.getElementById('outcomeBadge');
    badge.textContent = report.outcome;
    badge.className = 'badge ' + (report.outcome === 'found' ? 'badge-found' : (report.outcome === 'active' ? 'badge-active' : 'badge-other'));
    scrubber.max = totalSteps;

    // Set select value
    if (report.mission && report.mission.includes('easy')) {
      missionSelect.value = 'easy';
    } else if (report.mission && report.mission.includes('decoy')) {
      missionSelect.value = 'decoys';
    } else {
      missionSelect.value = 'current';
    }
  }

  // Compute Milestones for Story Mode
  function buildMilestones() {
    milestonesContainer.innerHTML = '';
    const milestones = [];

    // Milestone 1: Launch
    milestones.push({ step: 1, title: 'Launch', icon: '🚀', desc: 'Initial scan & departure' });

    let hasDecoy = false;
    let hasReturn = false;
    let hasRecharge = false;

    for (let i = 0; i < trace.length; i++) {
      const rec = trace[i];
      const s = rec.strategy;
      const stepN = rec.step;

      // Decoy Encounter (First Empty Inspect)
      if (!hasDecoy && rec.action.type === 'INSPECT' && !rec.observation.found) {
        milestones.push({ step: stepN, title: 'Decoy Peak', icon: '⚠️', desc: 'Empty inspection; unmasking decoy' });
        hasDecoy = true;
      }

      // Safety Return
      if (!hasReturn && s === 'return') {
        milestones.push({ step: stepN, title: 'Safety Return', icon: '🔋', desc: 'Battery guard engaged; turning back' });
        hasReturn = true;
      }

      // Station Recharge
      if (!hasRecharge && s === 'recharge') {
        milestones.push({ step: stepN, title: 'Dock & Recharge', icon: '⚡', desc: 'Station recharge cycle' });
        hasRecharge = true;
      }

      // EVE Found
      if (rec.observation.found) {
        milestones.push({ step: stepN, title: 'EVE Rescued!', icon: '🎯', desc: 'Target location and condition confirmed' });
        break;
      }
    }

    if (!report.found && trace.length > 1) {
      milestones.push({ step: trace.length, title: 'Mission Complete', icon: '🏁', desc: 'Search concluded' });
    }

    milestones.forEach((m) => {
      const btn = document.createElement('button');
      btn.className = 'milestone-btn';
      btn.innerHTML = `<span>${m.icon}</span> <span>${m.title} (Step ${m.step})</span>`;
      btn.title = m.desc;
      btn.addEventListener('click', () => {
        pause();
        currentStep = m.step;
        onStepChange();
      });
      milestonesContainer.appendChild(btn);
    });
  }

  // Highlight Active Milestone
  function updateMilestonesActive() {
    const btns = milestonesContainer.querySelectorAll('.milestone-btn');
    btns.forEach(btn => btn.classList.remove('active'));
    // Find closest milestone
    let bestIdx = -1;
    let minDiff = 999999;
    btns.forEach((btn, idx) => {
      const match = btn.textContent.match(/Step (\\d+)/);
      if (match) {
        const s = parseInt(match[1], 10);
        const diff = Math.abs(currentStep - s);
        if (diff < minDiff && diff <= 3) {
          minDiff = diff;
          bestIdx = idx;
        }
      }
    });
    if (bestIdx >= 0) btns[bestIdx].classList.add('active');
  }

  // Resize canvas
  function resizeCanvas() {
    const wrapper = canvas.parentElement;
    const size = Math.min(wrapper.clientWidth - 40, wrapper.clientHeight - 40, 680);
    canvas.width = size;
    canvas.height = size;
    renderGrid();
  }
  window.addEventListener('resize', () => { resizeCanvas(); renderTimeline(); });

  // Grid Cell Size
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
          ctx.fillStyle = `rgba(239, 68, 68, ${intensity * 0.78})`;
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
      ctx.fillRect(sx * cs + cs*0.18, sy * cs + cs*0.18, cs*0.64, cs*0.64);
      ctx.strokeStyle = '#fef08a';
      ctx.lineWidth = 1.5;
      ctx.strokeRect(sx * cs + cs*0.18, sy * cs + cs*0.18, cs*0.64, cs*0.64);
      // Lightning symbol
      ctx.fillStyle = '#000';
      ctx.font = `bold ${Math.max(10, cs*0.45)}px sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('⚡', sx * cs + cs*0.5, sy * cs + cs*0.5);
    });

    // 4. Clues Discovered up to currentStep
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
        ctx.strokeStyle = 'rgba(168, 85, 247, 0.45)';
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
      document.getElementById('reasoningText').textContent = "Initial state. Mission briefing loaded; awaiting first decision.";
      document.getElementById('actionExecuted').textContent = "None (Briefing)";
      document.getElementById('obsEvent').textContent = "start";
      document.getElementById('currentStrategyTag').textContent = "START";
      document.getElementById('currentStrategyTag').style.background = "#334155";
      document.getElementById('candidatesTableBody').innerHTML = '<tr><td colspan="5" style="color: var(--text-muted); text-align: center;">Mission briefing phase</td></tr>';
      safetyBanner.style.display = 'none';
      compassLabel.textContent = "Standby";
      compassNeedle.style.transform = 'rotate(0deg)';
      updateMilestonesActive();
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

    // Strategy & Safety Alert
    const strat = rec.strategy || 'localize';
    const tag = document.getElementById('currentStrategyTag');
    tag.textContent = strat;
    tag.style.background = STRAT_COLORS[strat] || '#334155';

    if (strat === 'return' || strat === 'recharge') {
      safetyBanner.style.display = 'flex';
    } else {
      safetyBanner.style.display = 'none';
    }

    document.getElementById('reasoningText').textContent = rec.reason || 'No reason logged.';
    
    // Action & Observation
    let actStr = rec.action.type;
    if (rec.action.dir) actStr += ` (${rec.action.dir})`;
    document.getElementById('actionExecuted').textContent = actStr;
    document.getElementById('obsEvent').textContent = rec.observation.event || 'ok';

    // Compass Needle
    if (rec.action.type === 'SCAN_SIGNAL' && rec.observation.reading) {
      const deg = rec.observation.reading.bearing_deg;
      compassNeedle.style.transform = `rotate(${deg}deg)`;
      compassLabel.textContent = `${deg.toFixed(0)}° (Strength: ${rec.observation.reading.strength.toFixed(2)})`;
    } else if (rec.observation.event === 'no_signal') {
      compassLabel.textContent = "Signal Dropout (null)";
    }

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
      tbody.innerHTML = '<tr><td colspan="5" style="color: var(--text-muted); text-align: center;">Direct action / homing guard active</td></tr>';
    }

    updateMilestonesActive();
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
    tctx.strokeStyle = '#38bdf8';
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

  // Auto-Tour Mode
  btnAutoTour.addEventListener('click', () => {
    pause();
    const btns = Array.from(milestonesContainer.querySelectorAll('.milestone-btn'));
    if (btns.length === 0) return;

    let idx = 0;
    btnAutoTour.textContent = '⏹ Stop Tour';

    function tourStep() {
      if (idx >= btns.length) {
        btnAutoTour.textContent = '▶ Auto-Tour';
        return;
      }
      btns[idx].click();
      idx++;
      playTimer = setTimeout(tourStep, 3200);
    }
    tourStep();
  });

  btnPlay.addEventListener('click', () => isPlaying ? pause() : play());
  btnPrev.addEventListener('click', () => { if (currentStep > 0) { currentStep--; onStepChange(); } });
  btnNext.addEventListener('click', () => { if (currentStep < totalSteps) { currentStep++; onStepChange(); } });
  btnFirst.addEventListener('click', () => { currentStep = 0; onStepChange(); });
  btnLast.addEventListener('click', () => { currentStep = totalSteps; onStepChange(); });
  speedSelect.addEventListener('change', () => { if (isPlaying) { pause(); play(); } });
  evalToggle.addEventListener('change', () => renderGrid());
  heatmapToggle.addEventListener('change', () => renderGrid());

  // Mission Switcher Dropdown
  missionSelect.addEventListener('change', (e) => {
    const val = e.target.value;
    if (val === 'easy') {
      window.location.href = 'easy.html';
    } else if (val === 'decoys') {
      window.location.href = 'visualizer.html';
    } else if (val === 'upload') {
      fileUploadInput.click();
    }
  });

  // Custom File Loader (Drag & Drop or File Picker)
  fileUploadInput.addEventListener('change', (e) => {
    const files = Array.from(e.target.files);
    let repFile = files.find(f => f.name.endsWith('.json') && !f.name.includes('trace'));
    let traceFile = files.find(f => f.name.endsWith('.jsonl') || f.name.includes('trace'));

    if (repFile && traceFile) {
      Promise.all([repFile.text(), traceFile.text()]).then(([repText, traceText]) => {
        try {
          const parsedRep = JSON.parse(repText);
          const parsedTrace = traceText.splitlines ? traceText.splitlines() : traceText.split('\\n');
          const records = parsedTrace.filter(l => l.trim()).map(l => JSON.parse(l));

          // Swap active mission
          report = parsedRep;
          trace = records;
          totalSteps = trace.length;
          currentStep = 0;
          path = [records[0].before.pos, ...records.map(r => r.observation.pos)];
          strategies = records.map(r => r.strategy);
          entropy = records.map(r => r.belief.entropy);
          battery = records.map(r => r.before.battery);

          initHeader();
          buildMilestones();
          resizeCanvas();
          onStepChange();
        } catch(err) {
          alert("Error parsing uploaded run files: " + err);
        }
      });
    } else {
      alert("Please select both a report.json and trace.jsonl file.");
    }
  });

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
        body { font-family: monospace; background: #080c16; color: #f8fafc; padding: 24px; line-height: 1.5; }
        pre { background: #0f172a; padding: 16px; border-radius: 8px; border: 1px solid #1e293b; overflow: auto; }
      </style></head>
      <body>
        <h1>Mission Report: ${report.mission}</h1>
        <p><b>Outcome:</b> ${report.outcome} | <b>Location:</b> ${JSON.stringify(report.inferred_location)} | <b>Status:</b> ${JSON.stringify(report.inferred_status)}</p>
        <h2>Recommended Next Step</h2>
        <p style="background: #1e293b; padding: 12px; border-left: 4px solid #38bdf8;">${report.recommended_next_step}</p>
        <h2>Full JSON Report</h2>
        <pre>${JSON.stringify(report, null, 2)}</pre>
      </body></html>
    `);
  });

  // Initial Boot
  initHeader();
  buildMilestones();
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
