"""Time segments with constant covariates, for the log-linear self-exciting point process (DESIGN.md).

For one event kind (chance, open_chance, box_entry or shot), each team x period window is cut at every point where
a covariate can change: clock-bin edges, goals, dead-time edges, player-count changes, and every own / opponent
event s together with s + 2, 5, 10, 20 minutes. Covariates are evaluated at the segment midpoint from history
strictly before it; an event at time t is counted in the segment (a, b] with a < t <= b, so its intensity uses only
the past (left-continuous, predictable intensity). Dead time after goals is removed from exposure.

Usage: python src/segments.py [kind ...]   -> data/segments_<kind>.parquet
"""
import sys

import numpy as np
import pandas as pd

from build_events import ROOT

WIN = np.array([0.0, 120.0, 300.0, 600.0, 1200.0])        # history windows (s): (0,2], (2,5], (5,10], (10,20] min
WNAMES = ["w0_2", "w2_5", "w5_10", "w10_20"]
SHRINK_MATCHES = 5.0


def load():
    d = ROOT / "data"
    return {k: pd.read_parquet(d / f"{k}.parquet") for k in ("matches", "windows", "events", "dead", "onpitch")}


def window_counts(m, s):
    """For midpoints m and event times s (same period): counts of s in each history window, and > 20 min ago."""
    if len(s) == 0:
        return np.zeros((len(m), len(WNAMES))), np.zeros(len(m))
    diff = m[:, None] - s[None, :]
    out = np.stack([((diff > WIN[k]) & (diff <= WIN[k + 1])).sum(1) for k in range(4)], axis=1)
    return out.astype(float), (diff > WIN[-1]).sum(1).astype(float)


def _shrunk(a, lbar, tid, col):
    """Per-90 rate from earlier matches, shrunk towards the league average with SHRINK_MATCHES matches' weight."""
    if tid not in a.index:
        return lbar
    x = a.loc[tid]
    return (x[col] + SHRINK_MATCHES * lbar) / (x.sec / 5400.0 + SHRINK_MATCHES)


def prematch_strength(D, kind):
    """Shrunken events created / conceded per 90 from matches strictly before each match's date (same league)."""
    M, W, E, dead = D["matches"], D["windows"], D["events"], D["dead"]
    expo = W.groupby(["match_id", "team_id"])["T"].sum()
    dd = dead.groupby("match_id").apply(lambda g: (g.end - g.start).sum(), include_groups=False)
    expo = expo - expo.index.get_level_values(0).map(dd).fillna(0.0).values
    cnt = E[E.kind == kind].groupby(["match_id", "team_id"]).size()
    tm = expo.rename("sec").reset_index()
    tm["n"] = [int(cnt.get((a, b), 0)) for a, b in zip(tm.match_id, tm.team_id, strict=True)]
    tm = tm.merge(M[["match_id", "league", "date", "home_id", "away_id"]], on="match_id")
    tm["opp_id"] = np.where(tm.team_id == tm.home_id, tm.away_id, tm.home_id)
    conc = tm[["match_id", "team_id", "n"]].rename(columns={"team_id": "opp_id", "n": "n_against"})
    tm = tm.merge(conc, on=["match_id", "opp_id"])
    rows = []
    for (_lg, date), g in tm.groupby(["league", "date"]):
        past = tm[(tm.league == _lg) & (tm.date < date)]           # strictly earlier dates only
        if len(past) == 0:
            for r in g.itertuples():
                rows.append(dict(match_id=r.match_id, team_id=r.team_id, att=0.0, dfn=0.0, n_prior=0))
            continue
        lbar = past.n.sum() / (past.sec.sum() / 5400.0)            # league events per 90 (= conceded per 90)
        a = past.groupby("team_id").agg(n=("n", "sum"), na=("n_against", "sum"), sec=("sec", "sum"),
                                        k=("match_id", "nunique"))
        for r in g.itertuples():
            att = _shrunk(a, lbar, r.team_id, "n")
            dfn = _shrunk(a, lbar, r.opp_id, "na")
            rows.append(dict(match_id=r.match_id, team_id=r.team_id, att=np.log(att / lbar), dfn=np.log(dfn / lbar),
                             n_prior=int(a.k.get(r.team_id, 0))))
    return pd.DataFrame(rows)


def build(kind, D=None):
    D = D or load()
    M, W, E, dead, onp = D["matches"], D["windows"], D["events"], D["dead"], D["onpitch"]
    strength = prematch_strength(D, kind).set_index(["match_id", "team_id"])
    ev = E[E.kind == kind]
    goals = E[E.kind == "goal"]
    by_match = {k: g for k, g in ev.groupby("match_id")}
    goals_m = {k: g for k, g in goals.groupby("match_id")}
    dead_m = {k: g for k, g in dead.groupby("match_id")}
    on_m = {k: g for k, g in onp.groupby("match_id")}
    out = []
    empty = E.iloc[:0]
    for w in W.itertuples():
        mid, p, tid, oid, T = w.match_id, w.period, w.team_id, w.opp_id, w.T
        e = by_match.get(mid, empty)
        g = goals_m.get(mid, empty)
        dm = dead_m.get(mid, dead.iloc[:0])
        om = on_m[mid]
        own_all, opp_all = e[e.team_id == tid], e[e.team_id == oid]
        own = own_all[own_all.period == p].t.values
        opp = opp_all[opp_all.period == p].t.values
        own_prev = float((own_all.period < p).sum())
        opp_prev = float((opp_all.period < p).sum())
        gf = g[(g.team_id == tid) & (g.period == p)].t.values
        ga = g[(g.team_id == oid) & (g.period == p)].t.values
        gf_before = float(((g.team_id == tid) & (g.period < p)).sum())
        ga_before = float(((g.team_id == oid) & (g.period < p)).sum())
        dd = dm[dm.period == p]
        op = om[om.period == p]
        cps = [0.0, T, *[x for x in (900.0, 1800.0, 2700.0) if x < T], *dd.start.values, *dd.end.values,
               *op.start.values, *op.end.values]
        for s in np.r_[own, opp, gf, ga]:
            cps.extend(s + WIN)
        c = np.unique(np.clip(np.array(cps, float), 0.0, T))
        a, b = c[:-1], c[1:]
        keep = b - a > 1e-9
        a, b = a[keep], b[keep]
        mmid = (a + b) / 2
        in_dead = np.zeros(len(mmid), bool)
        for d in dd.itertuples():
            in_dead |= (mmid > d.start) & (mmid < d.end)
        own_w, own_early = window_counts(mmid, own)
        opp_w, opp_early = window_counts(mmid, opp)
        gf_w, _ = window_counts(mmid, gf)
        ga_w, _ = window_counts(mmid, ga)
        gs = (gf_before + (gf[None, :] < mmid[:, None]).sum(1)) - (ga_before + (ga[None, :] < mmid[:, None]).sum(1))
        n_own = ((op.team_id.values[None, :] == tid) & (op.start.values[None, :] < mmid[:, None])
                 & (op.end.values[None, :] > mmid[:, None])).sum(1)
        n_opp = ((op.team_id.values[None, :] == oid) & (op.start.values[None, :] < mmid[:, None])
                 & (op.end.values[None, :] > mmid[:, None])).sum(1)
        y = np.zeros(len(a))
        if len(own):
            idx = np.searchsorted(b, own, side="left")          # a[idx] < t <= b[idx]
            assert (idx < len(b)).all() and not in_dead[idx].any()
            np.add.at(y, idx, 1.0)
        st = strength.loc[(mid, tid)]
        seg = pd.DataFrame(dict(match_id=mid, team_id=tid, period=p, a=a, b=b, dur=b - a, y=y,
                                clock=np.minimum((mmid // 900).astype(int), 3), gs=np.clip(gs, -2, 2),
                                adv=np.clip(n_own - n_opp, -1, 1), home=w.home, att=st.att, dfn=st.dfn,
                                own_early=own_early + own_prev, opp_early=opp_early + opp_prev))
        for k, nm in enumerate(WNAMES):
            seg["own_" + nm], seg["opp_" + nm] = own_w[:, k], opp_w[:, k]
            seg["gf_" + nm], seg["ga_" + nm] = gf_w[:, k], ga_w[:, k]
        out.append(seg[~in_dead])
    S = pd.concat(out, ignore_index=True)
    S = S.merge(M[["match_id", "league", "half", "date"]], on="match_id")
    n_events = int((E.kind == kind).sum())
    assert int(S.y.sum()) == n_events, (S.y.sum(), n_events)   # every event assigned to exactly one segment
    return S


def main(kinds):
    D = load()
    for kind in kinds:
        S = build(kind, D)
        S.to_parquet(ROOT / "data" / f"segments_{kind}.parquet")
        print(kind, "segments", len(S), "events", int(S.y.sum()), "exposure (h)", round(S.dur.sum() / 3600, 1))


if __name__ == "__main__":
    main(sys.argv[1:] or ["chance", "open_chance", "box_entry", "shot"])
