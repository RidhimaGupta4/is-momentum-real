// Intensity of chances under the main model (chances, full season), from the exported coefficients, plus the
// design terms needed for delta-method intervals. Shared by the page and by tests/test_model.py (run with node),
// which checks every mode and the intervals against the Python model.
"use strict";
const B = DATA.beta;
const WIN = [[0, 120, "w0_2"], [120, 300, "w2_5"], [300, 600, "w5_10"], [600, 1200, "w10_20"]];
const DEAD_AFTER_GOAL = 56;   // seconds: median goal-to-kick-off time in the data; no chances are possible

function addWindows(terms, prefix, ages) {   // count events of the given ages (s) into their window terms
  for (const a of ages) {
    const w = WIN.find(([lo, hi]) => a > lo && a <= hi);
    if (w) terms[`${prefix}_${w[2]}`] = (terms[`${prefix}_${w[2]}`] || 0) + 1;
  }
}

// events: [{team: "home"|"away", kind: "chance"|"goal", half: 1|2, t: seconds into the half}]
// A goal is also a chance (as for goals from shots in the data). opts: {attHome, attAway (log ratios), mode}
// mode "full":    the main model.
// mode "restart": only the first two minutes after events: events up to 2 minutes old get the 0-2 minute terms
//                 (restarts: dead ball, kick-offs); older events count only in the earlier-in-match terms.
// mode "off":     restarts and momentum switched off: every past event counts only in the earlier-in-match terms
//                 (as if it had happened more than 20 minutes earlier) -- the contrast used in the analysis.
// mode "score":   only the score changes (goals move the game state; dead time after goals); no history terms.
// mode "none":    no events at all.
// Returns null during dead time after a goal (no chance is possible), otherwise {terms: {coefficient: value}}.
function designTerms(side, half, t, events, opts) {
  const mode = opts.mode || "full";
  const evs = mode === "none" ? [] : events;
  const other = side === "home" ? "away" : "home";
  for (const e of evs) {
    if (e.kind === "goal" && e.half === half && t > e.t && t < e.t + DEAD_AFTER_GOAL) return null;
  }
  const before = e => e.half < half || (e.half === half && e.t < t);
  const goalsFor = evs.filter(e => e.kind === "goal" && e.team === side && before(e)).length;
  const goalsAg = evs.filter(e => e.kind === "goal" && e.team === other && before(e)).length;
  const gs = Math.max(-2, Math.min(2, goalsFor - goalsAg));
  const bin = Math.min(Math.floor(t / 900), 3);
  const terms = {intercept: 1};
  if (!(half === 1 && bin === 0)) terms[`clock_p${half}_${bin}`] = 1;
  if (gs !== 0) terms[`gs_${gs > 0 ? "+" : ""}${gs}`] = 1;
  if (side === "home") terms.home = 1;
  terms.att = side === "home" ? opts.attHome : opts.attAway;
  const counts = (team, keep, kind) => evs.filter(e => e.team === team && keep(e) && (kind === "any" || e.kind === kind))
    .map(e => t - e.t);
  const early = (team, keep) => evs.filter(e => e.team === team && before(e) && !keep(e)).length;
  const addHistory = keep => {                   // events kept in windows; all other past events count as earlier
    addWindows(terms, "own", counts(side, keep, "any"));
    addWindows(terms, "opp", counts(other, keep, "any"));
    addWindows(terms, "gf", counts(side, keep, "goal"));
    addWindows(terms, "ga", counts(other, keep, "goal"));
    terms.own_early = early(side, keep);
    terms.opp_early = early(other, keep);
  };
  if (mode === "full") addHistory(e => e.half === half && e.t < t && t - e.t <= 1200);
  else if (mode === "restart") addHistory(e => e.half === half && e.t < t && t - e.t <= 120);
  else if (mode === "off") addHistory(() => false);
  return {terms};
}

function intensity(side, half, t, events, opts) {
  const d = designTerms(side, half, t, events, opts);
  if (d === null) return 0;
  let eta = 0;
  for (const [k, v] of Object.entries(d.terms)) eta += B[k] * v;
  return Math.exp(eta);
}

// Expected chances for one side over the whole match (two 47-minute halves, 10-second steps) and the gradient of that
// total with respect to the coefficients (names in DATA.cov_names), for delta-method intervals.
const HALF_SECONDS = 47 * 60, STEP_SECONDS = 10;
function matchTotal(side, events, opts) {
  const names = DATA.cov_names, idx = Object.fromEntries(names.map((n, i) => [n, i]));
  let total = 0;
  const grad = new Array(names.length).fill(0);
  for (const half of [1, 2]) {
    for (let t = 0; t < HALF_SECONDS; t += STEP_SECONDS) {
      const d = designTerms(side, half, t, events, opts);
      if (d === null) continue;
      let eta = 0;
      for (const [n, v] of Object.entries(d.terms)) eta += B[n] * v;
      const r = Math.exp(eta);
      total += r * STEP_SECONDS;
      for (const [n, v] of Object.entries(d.terms)) grad[idx[n]] += r * STEP_SECONDS * v;
    }
  }
  return {total, grad};
}
function seOf(g) {      // delta method: sqrt(g' V g), V = cluster-robust covariance of the coefficients
  let v = 0;
  for (let i = 0; i < g.length; i++) {
    if (!g[i]) continue;
    for (let j = 0; j < g.length; j++) v += g[i] * DATA.cov[i][j] * g[j];
  }
  return Math.sqrt(Math.max(v, 0));
}
function difference(a, b) {     // a, b: results of matchTotal (b may be null): estimate and 95% interval of a - b
  const g = a.grad.map((x, i) => x - (b ? b.grad[i] : 0));
  const est = a.total - (b ? b.total : 0), se = seOf(g);
  return {est, se, lo: est - 1.96 * se, hi: est + 1.96 * se};
}

if (typeof module !== "undefined") module.exports = {intensity, designTerms, matchTotal, difference, DEAD_AFTER_GOAL};
