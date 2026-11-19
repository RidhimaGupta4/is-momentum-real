"""Exploratory analyses, decided after the primary results were seen (DESIGN.md). Writes results/explore_*.csv.

1. Momentum contrasts: each later window (2-5, 5-10, 10-20 min) against the earlier-in-match term, for own and
   opponent chances, in the main (M2) and fixed-effects (FE) models, for every event type.
2. Out-of-sample decomposition (fit H1, score H2): M0 -> + earlier-in-match counts -> + 0-2 min terms
   -> + 2-20 min terms (= M1). Shows how much of the predictive gain is the short dip and how much is momentum.
3. Model-free: hazard of the next chance (either team) by seconds since a chance, relative to the average rate;
   split by which team has the next chance. Gaps that contain a goal (from any shot of the attack, or an own goal)
   are excluded, because play then restarts with a kick-off after about a minute of dead time.
4. Standard errors of the primary estimands with clusters = matches (as pre-registered), teams, and two-way
   (team and opponent; Cameron, Gelbach and Miller), because a team's residual style can persist across matches.
"""
import numpy as np
import pandas as pd
from scipy import stats

import model
from analysis import KINDS, RES, SEED, load
from build_events import ROOT


def contrasts():
    rows = []
    for kind in KINDS:
        S = load(kind)
        for spec in ("M2", "FE"):
            r = model.fit(S, spec)
            for side in ("own", "opp"):
                for w, lab in (("w0_2", "0-2"), ("w2_5", "2-5"), ("w5_10", "5-10"), ("w10_20", "10-20")):
                    est, se = model.combo(r, {f"{side}_{w}": 1, f"{side}_early": -1})
                    rows.append(dict(kind=kind, spec=spec, side=side, window=lab, log_rr=est, se=se,
                                     rate_ratio=np.exp(est), rr_lo=np.exp(est - 1.96 * se),
                                     rr_hi=np.exp(est + 1.96 * se), p=2 * stats.norm.sf(abs(est / se))))
    return pd.DataFrame(rows)


def oos_decomposition(rng):
    S = load("chance")
    h1, h2 = S[S.half == "H1"].reset_index(drop=True), S[S.half == "H2"].reset_index(drop=True)
    steps = ["M0", "X_early", "X_early_02", "M1"]
    pm = pd.DataFrame({sp: model.pp_loglik(h2, model.fit(h1, sp)) for sp in steps})
    rows = []
    for a, b, lab in (("X_early", "M0", "earlier-in-match counts"), ("X_early_02", "X_early", "0-2 min terms"),
                      ("M1", "X_early_02", "2-20 min terms (momentum windows)")):
        d = (pm[a] - pm[b]).values
        boots = [d[rng.integers(0, len(d), len(d))].sum() for _ in range(4000)]
        rows.append(dict(step=lab, ll_gain_total=d.sum(), ci_lo=np.percentile(boots, 2.5),
                         ci_hi=np.percentile(boots, 97.5), per_match=d.mean()))
    return pd.DataFrame(rows)


def post_chance_hazard(bin_s=10, max_s=300):
    E = pd.read_parquet(ROOT / "data" / "events.parquet")
    W = pd.read_parquet(ROOT / "data" / "windows.parquet")
    D = pd.read_parquet(ROOT / "data" / "dead.parquet")
    ch = E[E.kind == "chance"].sort_values(["match_id", "period", "t"])
    goals_at = {k: g.t.values for k, g in E[E.kind == "goal"].groupby(["match_id", "period"])}
    T = W.groupby(["match_id", "period"])["T"].max()
    dead_total = (D.end - D.start).sum()
    overall = len(ch) / (T.sum() - dead_total)                 # chances (either team) per second of play
    edges = np.arange(0, max_s + bin_s, bin_s)
    events_same, events_opp, exposure = np.zeros(len(edges) - 1), np.zeros(len(edges) - 1), np.zeros(len(edges) - 1)
    for (mid, p), g in ch.groupby(["match_id", "period"]):
        t, team = g.t.values, g.team_id.values
        gl = goals_at.get((mid, p), np.array([]))
        for i in range(len(t)):
            end = t[i + 1] if i + 1 < len(t) else T[(mid, p)]
            if ((gl >= t[i]) & (gl < end)).any():
                continue                 # a goal in the gap (any shot of the attack, or an own goal): kick-off restart

            gap = end - t[i]
            exposure += np.clip(gap - edges[:-1], 0, bin_s)
            if i + 1 < len(t) and gap < max_s:
                k = int(gap // bin_s)
                (events_same if team[i + 1] == team[i] else events_opp)[k] += 1
    return pd.DataFrame(dict(sec_lo=edges[:-1], sec_hi=edges[1:], exposure_s=exposure, next_same=events_same,
                             next_opp=events_opp,
                             rel_hazard_any=(events_same + events_opp) / exposure / overall,
                             rel_hazard_same=events_same / exposure / (overall / 2),
                             rel_hazard_opp=events_opp / exposure / (overall / 2)))


def clustering():
    from analysis import estimands
    S = load("chance")
    W = pd.read_parquet(ROOT / "data" / "windows.parquet")[["match_id", "period", "team_id", "opp_id"]]
    opp = S[["match_id", "period", "team_id"]].merge(W, on=["match_id", "period", "team_id"], how="left").opp_id.values
    r = model.fit(S, "M2")
    X, names = model.design(S, "M2")
    lam = np.exp(X @ r["beta"].values + np.log(S.dur.values))
    score = X * (S.y.values - lam)[:, None]
    Hinv = np.linalg.inv((X * lam[:, None]).T @ X)

    def vcov(keys):
        g = pd.factorize(keys)[0]
        G = g.max() + 1
        Sc = np.zeros((G, X.shape[1]))
        np.add.at(Sc, g, score)
        return Hinv @ (Sc.T @ Sc) @ Hinv * G / (G - 1)

    team = S.league.values + "_" + S.team_id.astype(str).values
    oppk = S.league.values + "_" + opp.astype(str)
    V = {"match": vcov(S.match_id.values), "team": vcov(team),
         "two-way team x opponent": vcov(team) + vcov(oppk) - vcov(team + "|" + oppk)}
    rows = []
    for name, w in estimands().items():
        c = pd.Series(0.0, index=names)
        for k, v in w.items():
            c[k] += v
        e = float(c @ r["beta"])
        for lab, Vm in V.items():
            se = float(np.sqrt(max(c.values @ Vm @ c.values, 0.0)))
            rows.append(dict(estimand=name, clusters=lab, rate_ratio=np.exp(e), se=se, rr_lo=np.exp(e - 1.96 * se),
                             rr_hi=np.exp(e + 1.96 * se)))
    return pd.DataFrame(rows)


def main():
    rng = np.random.default_rng(SEED + 1)
    C = contrasts()
    C.to_csv(RES / "explore_momentum_contrasts.csv", index=False)
    print(C[C.kind == "chance"].round(4).to_string(index=False))
    dec = oos_decomposition(rng)
    dec.to_csv(RES / "explore_oos_decomposition.csv", index=False)
    print(dec.round(3).to_string(index=False))
    H = post_chance_hazard()
    H.to_csv(RES / "explore_post_chance_hazard.csv", index=False)
    print(H.head(15).round(3).to_string(index=False))
    K = clustering()
    K.to_csv(RES / "explore_clustering.csv", index=False)
    print(K.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
