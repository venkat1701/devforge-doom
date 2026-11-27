"use strict";
// "ep" charts: rolling mean over the last 100 FREEDOOM training episodes (stochastic policy; scenario maps excluded)
const CHARTS = [
 ["Exit rate (Freedoom, train)", "ep", "exit"], ["New areas / episode (Freedoom)", "ep", "cells"],
 ["Revisit ratio (Freedoom; 1 = stuck)", "ep", "revisit_ratio"], ["Doors opened (Freedoom)", "ep", "doors_opened"],
 ["Keys picked (Freedoom)", "ep", "keys_picked"], ["Kills (Freedoom)", "ep", "kills"], ["Deaths (Freedoom)", "ep", "dead"],
 ["Native return (Freedoom)", "ep", "native_return"], ["Shaped return (Freedoom)", "ep", "return"],
 ["Entropy", "up", "entropy"], ["Approx KL", "up", "approx_kl"], ["Clip fraction", "up", "clip_fraction"],
 ["Value loss", "up", "vloss"], ["Explained variance", "up", "explained_variance"], ["Grad norm", "up", "grad_norm"],
 ["Learning rate", "up", "lr"], ["SPS", "up", "sps"]];
const $ = id => document.getElementById(id);
let updates = [], lastSeq = 0, eps = [];
const charts = CHARTS.map(([t]) => { const d = document.createElement("div"); d.className = "chart";
  d.innerHTML = `<h3>${t}</h3><canvas></canvas>`; $("charts").appendChild(d); return d.querySelector("canvas"); });
const fmt = (v, d = 2) => v == null ? "n/a" : Number(v).toFixed(d);
const dur = s => { if (s == null) return "n/a"; s = Math.round(s); return `${Math.floor(s / 3600)}h ${Math.floor(s % 3600 / 60)}m ${s % 60}s`; };
function roll(series, key, w) { // rolling mean of last w non-null; null where source null
  return series.map((r, i) => { if (r[key] == null) return null; let s = 0, n = 0;
    for (let j = Math.max(0, i - w + 1); j <= i; j++) if (series[j][key] != null) { s += series[j][key]; n++; } return s / n; }); }
function draw(cv, xs, ys, color) {
  const dpr = devicePixelRatio || 1, W = cv.clientWidth, H = cv.clientHeight; cv.width = W * dpr; cv.height = H * dpr;
  const c = cv.getContext("2d"); c.scale(dpr, dpr); c.clearRect(0, 0, W, H);
  const pts = ys.map((y, i) => [xs[i], y]).filter(p => p[1] != null && p[0] != null);
  if (!pts.length) { c.fillStyle = "#888"; c.fillText("no data", 8, 16); return; }
  const x0 = Math.min(...pts.map(p => p[0])), x1 = Math.max(...pts.map(p => p[0])) || 1;
  let y0 = Math.min(...pts.map(p => p[1])), y1 = Math.max(...pts.map(p => p[1])); if (y0 === y1) { y0 -= 1; y1 += 1; }
  const px = x => 34 + (x - x0) / ((x1 - x0) || 1) * (W - 40), py = y => H - 14 - (y - y0) / (y1 - y0) * (H - 24);
  c.fillStyle = "#888"; c.font = "10px sans-serif"; c.fillText(y1.toPrecision(3), 0, 10); c.fillText(y0.toPrecision(3), 0, H - 14);
  c.fillText(x1.toExponential(2), W - 50, H - 2);
  c.strokeStyle = color; c.lineWidth = 1.5; c.beginPath(); let pen = false; // nulls break the line (gaps)
  xs.forEach((x, i) => { const y = ys[i]; if (y == null || x == null) { pen = false; return; }
    pen ? c.lineTo(px(x), py(y)) : c.moveTo(px(x), py(y)); pen = true; }); c.stroke();
}
function render() {
  const acc = getComputedStyle(document.documentElement).getPropertyValue("--acc") || "#4fc3f7";
  CHARTS.forEach(([t, src, key], i) => {
    if (src === "ep") { const fd = eps.filter(e => e.freedoom); const ys = roll(fd, key, 100); draw(charts[i], fd.map(e => e.global_step), ys, acc); }
    else draw(charts[i], updates.map(u => u.global_step), updates.map(u => u[key]), acc);
  });
}
async function j(url) { const r = await fetch(url, { cache: "no-store" }); return { ok: r.ok, status: r.status, body: await r.json().catch(() => null) }; }
async function tick() {
  try {
    const st = (await j("/api/status")).body;
    if (st) {
      $("stale").classList.toggle("hidden", !st.stale);
      $("bar").style.width = Math.min(100, st.percent_done || 0) + "%";
      const rows = [["Steps", `${st.steps.toLocaleString()} / ${st.horizon.toLocaleString()} (${fmt(st.percent_done)}%)`],
        ["Updates", st.updates], ["SPS (last " + st.sps_recent_window + " updates)", fmt(st.sps_recent, 1)], ["SPS (wall)", fmt(st.sps_wall, 1)],
        ["ETA", dur(st.eta_seconds)], ["Elapsed", dur(st.elapsed_seconds)],
        ["Spend so far (estimated)", "$" + fmt(st.spend_estimated)], ["Projected total (estimated)", st.spend_projected_total_estimated == null ? "n/a" : "$" + fmt(st.spend_projected_total_estimated)],
        ["Last checkpoint", st.last_checkpoint_step == null ? "none" : `${st.last_checkpoint_step.toLocaleString()} (${dur(st.last_checkpoint_age_seconds)} ago)`],
        ["Heartbeat age", dur(st.heartbeat_age_seconds) + (st.stale ? "  STALE" : "")],
        ["GPU", st.gpu ? st.gpu.map(g => `${g.util_percent}% ${g.mem_used_mib}/${g.mem_total_mib} MiB`).join("; ") : "n/a"],
        ["Disk free", st.disk_free_bytes == null ? "n/a" : fmt(st.disk_free_bytes / 1e9, 1) + " GB"]];
      $("stats").innerHTML = rows.map(([a, b]) => `<tr><td>${a}</td><td>${b}</td></tr>`).join("");
    }
    const m = (await j("/api/metrics?after=" + lastSeq)).body;
    if (m) { if (m.gap) updates = []; updates = updates.concat(m.updates).slice(-2000); if (m.updates.length) lastSeq = m.last_seq; }
    const e = (await j("/api/episodes")).body; if (e) { eps = e.series;
      const sh = e.shaping_last_500; $("shaping").innerHTML = sh ? "<tr><th>component</th><th>mean per episode</th></tr>" +
        Object.entries(sh).map(([k, v]) => `<tr><td>${k}</td><td>${fmt(v, 3)}</td></tr>`).join("") +
        `<tr><td>native (wrapper)</td><td>${fmt(e.last_500.native_return, 3)}</td></tr>` : "<tr><td>no shaping in this run</td></tr>"; }
    const ev = (await j("/api/evals")).body;
    if (ev) { const cols = ["exit", "keys", "doors_opened", "locked_opened", "cells", "revisit_ratio", "kills", "dead", "proxy_score", "ms_mean"];
      $("evals").innerHTML = ev.evals.length ? `<tr><th>checkpoint / suite</th>${cols.map(c => `<th>${c}</th>`).join("")}</tr>` +
        ev.evals.map(r => `<tr><td>${r.name} (${r.episodes} eps)</td>${cols.map(c => `<td>${fmt(r[c], 2)}</td>`).join("")}</tr>`).join("")
        : "<tr><td>no evaluations yet</td></tr>"; }
    const t = await j("/api/frame/train/meta");
    if (t.ok) { $("train-img").src = "/api/frame/train?t=" + Date.now(); $("train-img").classList.remove("hidden");
      const b = t.body; $("train-frame").className = "meta";
      $("train-frame").textContent = `${b.stale ? "STALE " : ""}worker ${b.worker} | ${b.map} | episode step ${b.episode_step} | global ${b.global_step} | update ${b.update} | return ${fmt(b.episode_return, 2)} | ${dur(b.frame_age_seconds)} ago`; }
    else { $("train-img").classList.add("hidden"); $("train-frame").className = "unavail";
      $("train-frame").textContent = t.status === 503 && t.body ? "NOT AVAILABLE: " + t.body.reason : "unexpected response " + t.status; }
    const sm = await j("/api/frame/spectator/meta");
    if (sm.ok) { $("spec").src = "/api/frame/spectator?t=" + Date.now(); const b = sm.body;
      $("spec-meta").textContent = `checkpoint ${b.checkpoint_steps} (${dur(b.checkpoint_age_seconds)} old)${b.old ? " OLD: newer failed verification" : ""} | ${b.map} | ep ${b.episode} | tic ${b.tic} | sha ${String(b.manifest_sha256).slice(0, 12)}`; }
    else $("spec-meta").textContent = "no spectator frame yet";
    render();
  } catch (err) { $("stale").classList.remove("hidden"); }
}
tick(); setInterval(tick, 2000);
