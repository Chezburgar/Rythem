"use strict";

/* =========================================================
   Rhythm Funkin' — auto-charting FNF-style rhythm game.
   Analysis: spectral flux onset detection + BPM estimate,
   soft-snapped to the beat grid, mapped to 4 lanes by pitch.
   ========================================================= */

// ---------- shared state ----------
const S = {
  audioCtx: null,
  buffer: null,        // decoded AudioBuffer
  songName: "",
  analysis: null,      // { onsets:[{t,strength,centroid}], bpm, duration }
  difficulty: "normal",
  offsetMs: Number(localStorage.getItem("rf_offset") || 0),
};

const LANE_COLORS = ["#c24b99", "#00e5e5", "#12fa05", "#f9393f"]; // L D U R
const LANE_ROT = [-Math.PI / 2, Math.PI, 0, Math.PI / 2];         // arrow rotation
const KEY_LANES = {
  ArrowLeft: 0, ArrowDown: 1, ArrowUp: 2, ArrowRight: 3,
  KeyD: 0, KeyF: 1, KeyJ: 2, KeyK: 3,
};

const DIFFS = {
  easy:   { minGap: 0.30, keepFrac: 0.55, doubles: false, speed: 420, label: "Easy" },
  normal: { minGap: 0.17, keepFrac: 0.80, doubles: false, speed: 500, label: "Normal" },
  hard:   { minGap: 0.11, keepFrac: 1.00, doubles: true,  speed: 590, label: "Hard" },
};

// ---------- screens ----------
const screens = {};
for (const el of document.querySelectorAll(".screen")) screens[el.id.replace("screen-", "")] = el;
function showScreen(name) {
  for (const k in screens) screens[k].classList.toggle("active", k === name);
}

// ---------- upload ----------
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
document.getElementById("browse-btn").addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => { if (fileInput.files[0]) loadFile(fileInput.files[0]); });

["dragenter", "dragover"].forEach(ev => dropzone.addEventListener(ev, e => {
  e.preventDefault(); dropzone.classList.add("dragover");
}));
["dragleave", "drop"].forEach(ev => dropzone.addEventListener(ev, e => {
  e.preventDefault(); dropzone.classList.remove("dragover");
}));
dropzone.addEventListener("drop", e => {
  const f = e.dataTransfer.files && e.dataTransfer.files[0];
  if (f) loadFile(f);
});

async function loadFile(file) {
  showScreen("analyzing");
  setAnalyzeProgress(0, "Decoding audio…");
  try {
    if (!S.audioCtx) S.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    if (S.audioCtx.state === "suspended") await S.audioCtx.resume();
    const arr = await file.arrayBuffer();
    S.buffer = await S.audioCtx.decodeAudioData(arr);
    S.songName = file.name.replace(/\.[^.]+$/, "");
    S.analysis = await analyzeBuffer(S.buffer);
    document.getElementById("song-title").textContent = S.songName;
    document.getElementById("song-meta").textContent =
      `${formatTime(S.buffer.duration)} · ~${Math.round(S.analysis.bpm)} BPM · ${S.analysis.onsets.length} beats detected`;
    showScreen("ready");
  } catch (err) {
    console.error(err);
    alert("Couldn't read that file as audio. Try an mp3, ogg, or wav.");
    showScreen("upload");
  } finally {
    fileInput.value = "";
  }
}

function setAnalyzeProgress(frac, status) {
  document.getElementById("analyze-bar").style.width = (frac * 100).toFixed(1) + "%";
  if (status) document.getElementById("analyze-status").textContent = status;
}

function formatTime(sec) {
  const m = Math.floor(sec / 60), s = Math.floor(sec % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

// ---------- FFT ----------
function makeFFT(n) {
  const levels = Math.round(Math.log2(n));
  const cosT = new Float32Array(n / 2), sinT = new Float32Array(n / 2);
  for (let i = 0; i < n / 2; i++) {
    cosT[i] = Math.cos(2 * Math.PI * i / n);
    sinT[i] = Math.sin(2 * Math.PI * i / n);
  }
  const rev = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    let r = 0;
    for (let j = 0; j < levels; j++) r = (r << 1) | ((i >>> j) & 1);
    rev[i] = r;
  }
  return function fft(re, im) {
    for (let i = 0; i < n; i++) {
      const j = rev[i];
      if (j > i) {
        let t = re[i]; re[i] = re[j]; re[j] = t;
        t = im[i]; im[i] = im[j]; im[j] = t;
      }
    }
    for (let size = 2; size <= n; size *= 2) {
      const half = size / 2, step = n / size;
      for (let i = 0; i < n; i += size) {
        for (let j = i, k = 0; j < i + half; j++, k += step) {
          const tre = re[j + half] * cosT[k] + im[j + half] * sinT[k];
          const tim = im[j + half] * cosT[k] - re[j + half] * sinT[k];
          re[j + half] = re[j] - tre;
          im[j + half] = im[j] - tim;
          re[j] += tre;
          im[j] += tim;
        }
      }
    }
  };
}

// ---------- audio analysis ----------
async function analyzeBuffer(buffer) {
  const sr = buffer.sampleRate;
  const WIN = 1024, HOP = 512;

  // mono mixdown
  const mono = new Float32Array(buffer.length);
  for (let c = 0; c < buffer.numberOfChannels; c++) {
    const ch = buffer.getChannelData(c);
    for (let i = 0; i < ch.length; i++) mono[i] += ch[i] / buffer.numberOfChannels;
  }

  const nFrames = Math.max(0, Math.floor((mono.length - WIN) / HOP));
  const fft = makeFFT(WIN);
  const hann = new Float32Array(WIN);
  for (let i = 0; i < WIN; i++) hann[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (WIN - 1));

  const re = new Float32Array(WIN), im = new Float32Array(WIN);
  const nBins = WIN / 2;
  let prevMag = new Float32Array(nBins);
  let curMag = new Float32Array(nBins);
  const flux = new Float32Array(nFrames);
  const centroids = new Float32Array(nFrames);

  setAnalyzeProgress(0.02, "Scanning for beats…");
  for (let f = 0; f < nFrames; f++) {
    const off = f * HOP;
    for (let i = 0; i < WIN; i++) { re[i] = mono[off + i] * hann[i]; im[i] = 0; }
    fft(re, im);
    let fl = 0, magSum = 0, magWeighted = 0;
    for (let b = 1; b < nBins; b++) {
      const m = Math.sqrt(re[b] * re[b] + im[b] * im[b]);
      curMag[b] = m;
      const d = m - prevMag[b];
      if (d > 0) fl += d;
      magSum += m;
      magWeighted += m * b;
    }
    flux[f] = fl;
    centroids[f] = magSum > 1e-6 ? magWeighted / magSum : 0;
    const tmp = prevMag; prevMag = curMag; curMag = tmp;

    if ((f & 511) === 0) {
      setAnalyzeProgress(0.02 + 0.85 * (f / nFrames));
      await new Promise(r => setTimeout(r, 0));
    }
  }

  setAnalyzeProgress(0.9, "Finding the groove…");
  const frameDur = HOP / sr;

  // onset peak-picking with adaptive threshold
  let globalMean = 0;
  for (let i = 0; i < nFrames; i++) globalMean += flux[i];
  globalMean /= Math.max(1, nFrames);

  const onsets = [];
  const W = 16; // local window (~185ms each side)
  for (let i = 3; i < nFrames - 3; i++) {
    const v = flux[i];
    if (v <= flux[i - 1] || v < flux[i + 1] || v <= flux[i - 2] || v < flux[i + 2]) continue;
    let lo = Math.max(0, i - W), hi = Math.min(nFrames - 1, i + W), sum = 0;
    for (let j = lo; j <= hi; j++) sum += flux[j];
    const localMean = sum / (hi - lo + 1);
    if (v > localMean * 1.45 + globalMean * 0.25) {
      onsets.push({ t: i * frameDur, strength: v / (localMean + 1e-6), centroid: centroids[i] });
    }
  }

  // BPM via autocorrelation of the flux curve
  const bpmInfo = estimateBPM(flux, frameDur, globalMean);

  // soft-snap onsets to the eighth-note grid (only if very close — keeps sync safe)
  if (bpmInfo.bpm > 0) {
    const grid = (60 / bpmInfo.bpm) / 2; // eighth note
    const phase = bpmInfo.phase;
    for (const o of onsets) {
      const snapped = phase + Math.round((o.t - phase) / grid) * grid;
      if (Math.abs(snapped - o.t) < 0.035) o.t = snapped;
    }
  }

  setAnalyzeProgress(1, "Done!");
  return { onsets, bpm: bpmInfo.bpm || 120, duration: buffer.duration };
}

function estimateBPM(flux, frameDur, mean) {
  const n = flux.length;
  if (n < 200) return { bpm: 120, phase: 0 };
  const x = new Float32Array(n);
  for (let i = 0; i < n; i++) x[i] = flux[i] - mean;

  const minLag = Math.max(2, Math.floor((60 / 200) / frameDur)); // 200 BPM
  const maxLag = Math.min(n - 1, Math.ceil((60 / 60) / frameDur)); // 60 BPM
  let bestLag = 0, bestScore = -Infinity;
  for (let lag = minLag; lag <= maxLag; lag++) {
    let s = 0;
    for (let i = 0; i + lag < n; i++) s += x[i] * x[i + lag];
    s /= (n - lag);
    // gentle preference for typical song tempos (~90–160)
    const bpm = 60 / (lag * frameDur);
    const pref = 1 - 0.25 * Math.abs(Math.log2(bpm / 120));
    s *= Math.max(0.5, pref);
    if (s > bestScore) { bestScore = s; bestLag = lag; }
  }
  if (!bestLag) return { bpm: 120, phase: 0 };
  const period = bestLag * frameDur;
  const bpm = 60 / period;

  // beat phase: offset that catches the most flux energy on the grid
  const steps = 24;
  let bestPhase = 0, bestE = -1;
  for (let s = 0; s < steps; s++) {
    const ph = (s / steps) * period;
    let e = 0;
    for (let t = ph; t < n * frameDur; t += period) {
      const idx = Math.round(t / frameDur);
      if (idx >= 0 && idx < n) e += flux[idx];
    }
    if (e > bestE) { bestE = e; bestPhase = ph; }
  }
  return { bpm, phase: bestPhase };
}

// ---------- chart building ----------
function buildChart(analysis, diffName) {
  const diff = DIFFS[diffName];
  let onsets = analysis.onsets.filter(o => o.t > 0.4 && o.t < analysis.duration - 0.3);

  // keep strongest fraction
  if (diff.keepFrac < 1 && onsets.length > 8) {
    const sorted = [...onsets].sort((a, b) => b.strength - a.strength);
    const cutoff = sorted[Math.floor(sorted.length * diff.keepFrac) - 1].strength;
    onsets = onsets.filter(o => o.strength >= cutoff);
  }

  // enforce minimum gap (keep the stronger of two crowded onsets)
  onsets.sort((a, b) => a.t - b.t);
  const spaced = [];
  for (const o of onsets) {
    const last = spaced[spaced.length - 1];
    if (last && o.t - last.t < diff.minGap) {
      if (o.strength > last.strength) spaced[spaced.length - 1] = o;
    } else spaced.push(o);
  }

  // lane assignment by pitch quartile (spectral centroid), with anti-repeat
  const cents = [...spaced].map(o => o.centroid).sort((a, b) => a - b);
  const q = i => cents[Math.min(cents.length - 1, Math.floor(cents.length * i))] || 0;
  const q1 = q(0.25), q2 = q(0.5), q3 = q(0.75);

  const notes = [];
  let lastLane = -1, repeatCount = 0;
  const strongCut = [...spaced].map(o => o.strength).sort((a, b) => b - a)[Math.floor(spaced.length * 0.12)] || Infinity;

  for (const o of spaced) {
    let lane = o.centroid < q1 ? 0 : o.centroid < q2 ? 1 : o.centroid < q3 ? 2 : 3;
    if (lane === lastLane) {
      repeatCount++;
      if (repeatCount >= 2) { lane = (lane + 1 + Math.floor(Math.random() * 3)) % 4; repeatCount = 0; }
    } else repeatCount = 0;
    lastLane = lane;
    notes.push({ t: o.t, lane, hit: false, missed: false, judged: false });

    // doubles on the strongest hits (hard only)
    if (diff.doubles && o.strength >= strongCut) {
      const other = (lane + 2) % 4;
      notes.push({ t: o.t, lane: other, hit: false, missed: false, judged: false });
    }
  }
  notes.sort((a, b) => a.t - b.t || a.lane - b.lane);
  return notes;
}

// ---------- difficulty buttons ----------
for (const btn of document.querySelectorAll(".btn.diff")) {
  btn.addEventListener("click", () => startGame(btn.dataset.diff));
}
document.getElementById("back-btn").addEventListener("click", () => showScreen("upload"));

// ---------- game ----------
const canvas = document.getElementById("game-canvas");
const ctx2d = canvas.getContext("2d");

const JUDGE = [
  { name: "SICK!", window: 0.045, score: 350, acc: 1,    health: +1.5, color: "#12fa05" },
  { name: "GOOD",  window: 0.090, score: 200, acc: 0.75, health: +0.8, color: "#00e5e5" },
  { name: "BAD",   window: 0.140, score: 50,  acc: 0.35, health: -1.0, color: "#e8b23a" },
];
const MISS_WINDOW = 0.16;

const G = {
  running: false,
  paused: false,
  notes: [],
  ptr: 0,
  source: null,
  startTime: 0,      // audioCtx time when song position 0 plays
  pausedAt: 0,
  speed: 500,
  score: 0,
  combo: 0,
  maxCombo: 0,
  health: 50,
  counts: { sick: 0, good: 0, bad: 0, miss: 0 },
  accSum: 0,
  judgedCount: 0,
  keysDown: [false, false, false, false],
  receptorFlash: [0, 0, 0, 0],   // time of last press
  hitFlash: [0, 0, 0, 0],        // time of last successful hit
  hitColor: ["", "", "", ""],
  judgement: { text: "", color: "", time: -10 },
  comboPop: -10,
  songEnded: false,
  countdownEnd: 0,
  rafId: 0,
};

function startGame(diffName) {
  S.difficulty = diffName;
  const diff = DIFFS[diffName];

  G.notes = buildChart(S.analysis, diffName);
  G.ptr = 0;
  G.speed = diff.speed;
  G.score = 0; G.combo = 0; G.maxCombo = 0;
  G.health = 50;
  G.counts = { sick: 0, good: 0, bad: 0, miss: 0 };
  G.accSum = 0; G.judgedCount = 0;
  G.judgement = { text: "", color: "", time: -10 };
  G.songEnded = false;
  G.paused = false;
  G.running = true;

  document.getElementById("hud-song").textContent = S.songName;
  document.getElementById("hud-diff").textContent = diff.label;
  document.getElementById("pause-overlay").classList.add("hidden");
  const slider = document.getElementById("offset-slider");
  slider.value = S.offsetMs;
  document.getElementById("offset-value").textContent = S.offsetMs;

  showScreen("game");
  resizeCanvas();

  const LEAD_IN = 3; // countdown seconds
  const now = S.audioCtx.currentTime;
  G.startTime = now + LEAD_IN;
  G.countdownEnd = performance.now() / 1000 + LEAD_IN;

  G.source = S.audioCtx.createBufferSource();
  G.source.buffer = S.buffer;
  G.source.connect(S.audioCtx.destination);
  G.source.onended = () => { if (G.running && !G.paused) G.songEnded = true; };
  G.source.start(G.startTime);

  cancelAnimationFrame(G.rafId);
  G.rafId = requestAnimationFrame(gameLoop);
}

function songTime() {
  return S.audioCtx.currentTime - G.startTime + S.offsetMs / 1000;
}

function stopAudio() {
  if (G.source) {
    G.source.onended = null;
    try { G.source.stop(); } catch (e) { /* not started / already stopped */ }
    G.source = null;
  }
}

function endGame(result) {
  G.running = false;
  cancelAnimationFrame(G.rafId);
  stopAudio();
  if (S.audioCtx.state === "suspended") S.audioCtx.resume();
  if (result === "clear") showResults();
  else if (result === "dead") showScreen("gameover");
}

// ---------- input ----------
window.addEventListener("keydown", e => {
  if (e.code === "Escape" && G.running) { togglePause(); e.preventDefault(); return; }
  const lane = KEY_LANES[e.code];
  if (lane === undefined || e.repeat) return;
  if (!G.running || G.paused) return;
  e.preventDefault();
  G.keysDown[lane] = true;
  G.receptorFlash[lane] = performance.now() / 1000;
  tryHit(lane);
});
window.addEventListener("keyup", e => {
  const lane = KEY_LANES[e.code];
  if (lane !== undefined) G.keysDown[lane] = false;
});

function tryHit(lane) {
  const t = songTime();
  let best = null, bestDt = Infinity;
  for (let i = G.ptr; i < G.notes.length; i++) {
    const n = G.notes[i];
    if (n.t - t > MISS_WINDOW) break;
    if (n.judged || n.lane !== lane) continue;
    const dt = Math.abs(n.t - t);
    if (dt < bestDt) { bestDt = dt; best = n; }
  }
  if (!best || bestDt > JUDGE[JUDGE.length - 1].window) return; // ghost tap, no penalty

  best.hit = true;
  best.judged = true;
  const j = JUDGE.find(j => bestDt <= j.window);
  G.score += j.score;
  G.health = Math.min(100, Math.max(0, G.health + j.health));
  G.accSum += j.acc;
  G.judgedCount++;
  if (j.name === "SICK!") G.counts.sick++;
  else if (j.name === "GOOD") G.counts.good++;
  else G.counts.bad++;
  if (j.acc >= 0.35) { G.combo++; G.maxCombo = Math.max(G.maxCombo, G.combo); }
  G.judgement = { text: j.name, color: j.color, time: performance.now() / 1000 };
  G.comboPop = performance.now() / 1000;
  G.hitFlash[lane] = performance.now() / 1000;
  G.hitColor[lane] = j.color;
}

function registerMiss(note) {
  note.missed = true;
  note.judged = true;
  G.counts.miss++;
  G.judgedCount++;
  G.combo = 0;
  G.score = Math.max(0, G.score - 10);
  G.health = Math.max(0, G.health - 6);
  G.judgement = { text: "MISS", color: "#f9393f", time: performance.now() / 1000 };
}

// ---------- pause ----------
function togglePause() {
  if (!G.running) return;
  G.paused = !G.paused;
  document.getElementById("pause-overlay").classList.toggle("hidden", !G.paused);
  if (G.paused) S.audioCtx.suspend();
  else S.audioCtx.resume();
}
document.getElementById("resume-btn").addEventListener("click", togglePause);
document.getElementById("pause-restart-btn").addEventListener("click", () => {
  document.getElementById("pause-overlay").classList.add("hidden");
  G.paused = false;
  S.audioCtx.resume();
  stopAudio();
  startGame(S.difficulty);
});
document.getElementById("quit-btn").addEventListener("click", () => {
  G.paused = false;
  S.audioCtx.resume();
  endGame("quit");
  showScreen("ready");
});

const offsetSlider = document.getElementById("offset-slider");
offsetSlider.addEventListener("input", () => {
  S.offsetMs = Number(offsetSlider.value);
  document.getElementById("offset-value").textContent = S.offsetMs;
  localStorage.setItem("rf_offset", S.offsetMs);
});

// ---------- rendering ----------
function resizeCanvas() {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.floor(canvas.clientWidth * dpr);
  canvas.height = Math.floor(canvas.clientHeight * dpr);
  ctx2d.setTransform(dpr, 0, 0, dpr, 0, 0);
}
window.addEventListener("resize", () => { if (G.running) resizeCanvas(); });

function drawArrow(x, y, size, rot, style) {
  // style: {fill, stroke, lineWidth, glow}
  const c = ctx2d;
  c.save();
  c.translate(x, y);
  c.rotate(rot);
  const s = size / 2;
  c.beginPath();
  c.moveTo(0, -s);            // tip (pointing up pre-rotation)
  c.lineTo(s, 0);
  c.lineTo(s * 0.45, 0);
  c.lineTo(s * 0.45, s);
  c.lineTo(-s * 0.45, s);
  c.lineTo(-s * 0.45, 0);
  c.lineTo(-s, 0);
  c.closePath();
  if (style.glow) { c.shadowColor = style.glow; c.shadowBlur = 18; }
  if (style.fill) { c.fillStyle = style.fill; c.fill(); }
  if (style.stroke) { c.strokeStyle = style.stroke; c.lineWidth = style.lineWidth || 3; c.stroke(); }
  c.restore();
}

function gameLoop() {
  if (!G.running) return;
  G.rafId = requestAnimationFrame(gameLoop);
  if (G.paused) return;

  const t = songTime();
  const w = canvas.clientWidth, h = canvas.clientHeight;
  const nowSec = performance.now() / 1000;

  // ----- update -----
  for (let i = G.ptr; i < G.notes.length; i++) {
    const n = G.notes[i];
    if (n.judged) continue;
    if (t - n.t > MISS_WINDOW) registerMiss(n);
    else break;
  }
  while (G.ptr < G.notes.length && G.notes[G.ptr].judged) G.ptr++;

  if (G.health <= 0) { endGame("dead"); return; }
  if (G.songEnded || t > S.buffer.duration + 1) { endGame("clear"); return; }

  // ----- draw -----
  const c = ctx2d;
  c.clearRect(0, 0, w, h);

  // beat pulse background
  const beatPeriod = 60 / S.analysis.bpm;
  const beatPhase = t >= 0 ? (t % beatPeriod) / beatPeriod : 1;
  const pulse = Math.max(0, 1 - beatPhase * 4);
  c.fillStyle = `rgba(194, 75, 153, ${(0.05 * pulse).toFixed(3)})`;
  c.fillRect(0, 0, w, h);

  const laneW = Math.min(110, w / 6);
  const noteSize = laneW * 0.82;
  const fieldW = laneW * 4;
  const fieldX = (w - fieldW) / 2;
  const receptorY = Math.max(90, h * 0.14);

  // lane guides
  c.fillStyle = "rgba(0,0,0,0.35)";
  c.fillRect(fieldX - 10, 0, fieldW + 20, h);
  c.strokeStyle = "rgba(255,255,255,0.06)";
  c.lineWidth = 1;
  for (let i = 1; i < 4; i++) {
    c.beginPath();
    c.moveTo(fieldX + i * laneW, 0);
    c.lineTo(fieldX + i * laneW, h);
    c.stroke();
  }

  // receptors
  for (let lane = 0; lane < 4; lane++) {
    const x = fieldX + lane * laneW + laneW / 2;
    const pressAge = nowSec - G.receptorFlash[lane];
    const hitAge = nowSec - G.hitFlash[lane];
    const pressed = G.keysDown[lane];
    const scale = pressed ? 0.9 : 1;

    if (hitAge < 0.18) {
      // hit ring
      const p = hitAge / 0.18;
      c.strokeStyle = G.hitColor[lane];
      c.globalAlpha = 1 - p;
      c.lineWidth = 4;
      c.beginPath();
      c.arc(x, receptorY, noteSize * (0.55 + p * 0.6), 0, Math.PI * 2);
      c.stroke();
      c.globalAlpha = 1;
    }
    drawArrow(x, receptorY, noteSize * scale, LANE_ROT[lane], {
      fill: pressed ? "rgba(255,255,255,0.55)" : "rgba(255,255,255,0.10)",
      stroke: hitAge < 0.15 ? G.hitColor[lane] : "rgba(255,255,255,0.55)",
      lineWidth: 3.5,
      glow: hitAge < 0.15 ? G.hitColor[lane] : null,
    });
  }

  // notes (rising toward receptors)
  for (let i = G.ptr; i < G.notes.length; i++) {
    const n = G.notes[i];
    const y = receptorY + (n.t - t) * G.speed;
    if (y > h + noteSize) break;
    if (n.judged || y < -noteSize) continue;
    const x = fieldX + n.lane * laneW + laneW / 2;
    drawArrow(x, y, noteSize, LANE_ROT[n.lane], {
      fill: LANE_COLORS[n.lane],
      stroke: "rgba(255,255,255,0.85)",
      lineWidth: 2.5,
      glow: LANE_COLORS[n.lane],
    });
  }

  // judgement popup
  const jAge = nowSec - G.judgement.time;
  if (jAge < 0.5) {
    const p = jAge / 0.5;
    c.globalAlpha = 1 - p;
    c.font = `900 ${Math.round(38 - p * 6)}px "Segoe UI", sans-serif`;
    c.textAlign = "center";
    c.fillStyle = G.judgement.color;
    c.shadowColor = "#000"; c.shadowBlur = 8;
    c.fillText(G.judgement.text, w / 2, h * 0.44 - p * 20);
    c.shadowBlur = 0;
    c.globalAlpha = 1;
  }

  // combo
  if (G.combo >= 3) {
    const pop = Math.max(0, 1 - (nowSec - G.comboPop) * 6);
    c.font = `900 ${Math.round(30 + pop * 8)}px "Segoe UI", sans-serif`;
    c.textAlign = "center";
    c.fillStyle = "rgba(255,255,255,0.9)";
    c.shadowColor = "#000"; c.shadowBlur = 6;
    c.fillText(`${G.combo} combo`, w / 2, h * 0.52);
    c.shadowBlur = 0;
  }

  // countdown
  if (t < 0) {
    const remaining = -t;
    const count = Math.ceil(remaining);
    const frac = count - remaining; // 0→1 within each second
    c.globalAlpha = Math.min(1, 1.6 - frac);
    c.font = `900 ${Math.round(90 + frac * 30)}px "Segoe UI", sans-serif`;
    c.textAlign = "center";
    c.fillStyle = "#fff";
    c.shadowColor = "#c24b99"; c.shadowBlur = 30;
    c.fillText(count > 0 ? String(count) : "GO!", w / 2, h / 2);
    c.shadowBlur = 0;
    c.globalAlpha = 1;
  } else if (t < 0.5) {
    c.globalAlpha = 1 - t * 2;
    c.font = `900 100px "Segoe UI", sans-serif`;
    c.textAlign = "center";
    c.fillStyle = "#12fa05";
    c.fillText("GO!", w / 2, h / 2);
    c.globalAlpha = 1;
  }

  // ----- HUD -----
  document.getElementById("hud-score").textContent = `Score: ${G.score}`;
  const acc = G.judgedCount ? (G.accSum / G.judgedCount * 100) : null;
  document.getElementById("hud-acc").textContent = `Accuracy: ${acc === null ? "—" : acc.toFixed(1) + "%"}`;
  document.getElementById("health-bar").style.width = G.health + "%";
  document.getElementById("song-progress").style.width =
    Math.min(100, Math.max(0, t / S.buffer.duration * 100)) + "%";
}

// ---------- results ----------
function showResults() {
  const acc = G.judgedCount ? G.accSum / G.judgedCount * 100 : 0;
  const grade = acc >= 95 ? "S" : acc >= 90 ? "A" : acc >= 80 ? "B" : acc >= 65 ? "C" : "D";
  document.getElementById("grade-badge").textContent = grade;
  document.getElementById("results-heading").textContent =
    grade === "S" ? "PERFECT PERFORMANCE!" : grade === "D" ? "SONG SURVIVED…" : "SONG CLEAR!";
  document.getElementById("results-stats").innerHTML = `
    <span>Score</span><b>${G.score.toLocaleString()}</b>
    <span>Accuracy</span><b>${acc.toFixed(1)}%</b>
    <span>Max combo</span><b>${G.maxCombo}</b>
    <span class="stat-sick">Sick</span><b class="stat-sick">${G.counts.sick}</b>
    <span class="stat-good">Good</span><b class="stat-good">${G.counts.good}</b>
    <span class="stat-bad">Bad</span><b class="stat-bad">${G.counts.bad}</b>
    <span class="stat-miss">Miss</span><b class="stat-miss">${G.counts.miss}</b>`;
  showScreen("results");
}

document.getElementById("results-retry-btn").addEventListener("click", () => startGame(S.difficulty));
document.getElementById("results-menu-btn").addEventListener("click", () => showScreen("upload"));
document.getElementById("gameover-retry-btn").addEventListener("click", () => startGame(S.difficulty));
document.getElementById("gameover-menu-btn").addEventListener("click", () => showScreen("upload"));

showScreen("upload");
