"""All pre-registered analyses (DESIGN.md). Writes results/*.csv.

1. Full-season fits (M0, M1, M2 = main, NOHET, FE) for chances, open-play chances, box entries, and shots
   (positive control); all coefficients with cluster-robust SEs.
2. The eight primary estimands on the main model for chances, Holm-corrected; the same estimands under FE and
   NOHET, and for the sensitivity event types; main model per league.
3. Out-of-sample: M0 / M1 / M2 fitted on first halves (H1), exact point-process log-likelihood on second halves
   (H2); per-match gains with 95% intervals from resampling H2 matches.
4. Time-rescaling goodness of fit on H2 for M0 and M2 (censoring-aware; see DESIGN.md deviations).
5. Observed vs expected chances after goals on H2, expectations from the H1-fitted M1 (no goal terms).
"""
import numpy as np
import pandas as pd
from scipy import stats

import model
from build_events import ROOT

RES = ROOT / "results"
KINDS = ["chance", "open_chance", "box_entry", "shot"]
SEED = 20160815


def load(kind):
    return pd.read_parquet(ROOT / "data" / f"segments_{kind}.parquet")


def estimands():
    e = {"self_excitation_0_2_vs_earlier": {"own_w0_2": 1, "own_early": -1},
         "cross_excitation_0_2_vs_earlier": {"opp_w0_2": 1, "opp_early": -1}}
    for w, lab in (("w0_2", "0-2"), ("w2_5", "2-5"), ("w5_10", "5-10")):
        e[f"scorer_after_scoring_{lab}"] = {f"own_{w}": 1, f"gf_{w}": 1}
    for w, lab in (("w0_2", "0-2"), ("w2_5", "2-5"), ("w5_10", "5-10")):
        e[f"conceder_after_conceding_{lab}"] = {f"opp_{w}": 1, f"ga_{w}": 1}
    return e


def estimand_table(res, kind, spec, scope="all"):
    rows = []
    for name, w in estimands().items():
        if not all(k in res["beta"].index for k in w):
            continue
        est, se = model.combo(res, w)
        z = est / se
        rows.append(dict(kind=kind, spec=spec, scope=scope, estimand=name, log_rr=est, se=se,
                         rate_ratio=np.exp(est), rr_lo=np.exp(est - 1.96 * se), rr_hi=np.exp(est + 1.96 * se),
                         p=2 * stats.norm.sf(abs(z))))
    return pd.DataFrame(rows)


def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def coef_table(res, kind, spec, scope="all"):
    se = np.sqrt(np.diag(res["cov"].values))
    b = res["beta"]
    return pd.DataFrame(dict(kind=kind, spec=spec, scope=scope, term=b.index, coef=b.values, se=se,
                             rate_ratio=np.exp(b.values), rr_lo=np.exp(b.values - 1.96 * se),
                             rr_hi=np.exp(b.values + 1.96 * se)))


def time_rescaling(S, res):
    """Time-rescaling check (corrected; see DESIGN.md, deviations). In each team x period window the compensator
    Lambda maps event times to a unit-rate Poisson process on [0, Lambda(T)] if the model is right. Returns
    (a) the rescaled gaps between consecutive events, with the last gap of each window right-censored at
        Lambda(T) (dropping it would bias the gaps short);
    (b) u = Lambda(t_i) / Lambda(T), which are iid Uniform(0, 1) given the number of events (exact test of the
        shape of the intensity within windows)."""
    lam = model.predict_rate(S, res)
    S = S.assign(L=lam * S.dur.values)
    gaps, cens, u = [], [], []
    for _, g in S.groupby(["match_id", "team_id", "period"], sort=False):
        cum = np.cumsum(g.L.values)
        total = cum[-1]
        idx = np.repeat(np.arange(len(g)), g.y.values.astype(int))
        at = cum[idx]
        if len(at):
            gaps.extend(np.diff(np.r_[0.0, at]))
            cens.append(total - at[-1])
            u.extend(at / total)
        else:
            cens.append(total)
    return np.array(gaps), np.array(cens), np.array(u)


def km_vs_exponential(gaps, cens, grid):
    """Kaplan-Meier survival of rescaled gaps (with right-censoring) on a grid, and exp(-t) for comparison."""
    t = np.r_[gaps, cens]
    d = np.r_[np.ones(len(gaps)), np.zeros(len(cens))]
    o = np.argsort(t, kind="mergesort")
    t, d = t[o], d[o]
    at_risk = len(t) - np.arange(len(t))
    surv = np.cumprod(1 - d / at_risk)
    km = np.array([surv[np.searchsorted(t, x, side="right") - 1] if x >= t[0] else 1.0 for x in grid])
    return km, np.exp(-grid)


def main():
    rng = np.random.default_rng(SEED)
    coefs, ests, fits = [], [], []
    for kind in KINDS:
        S = load(kind)
        for spec in ("M0", "M1", "M2", "NOHET", "FE"):
            r = model.fit(S, spec)
            coefs.append(coef_table(r, kind, spec))
            if kind == "chance" and spec == "M2":      # covariance for the page's delta-method intervals
                r["cov"].to_csv(RES / "cov_chance_M2.csv")
            fits.append(dict(kind=kind, spec=spec, ll=r["ll"], n_events=r["n_events"], n_matches=r["n_clusters"],
                             n_params=len(r["beta"])))
            if spec in ("M2", "NOHET", "FE"):
                ests.append(estimand_table(r, kind, spec))
        print(kind, "fitted", flush=True)
    E = pd.concat(ests, ignore_index=True)
    prim = (E.kind == "chance") & (E.spec == "M2")
    E["p_holm"] = np.nan
    E.loc[prim, "p_holm"] = holm(E.loc[prim, "p"].values)
    # per league (main model, chances)
    S = load("chance")
    for lg, g in S.groupby("league"):
        r = model.fit(g.reset_index(drop=True), "M2")
        E = pd.concat([E, estimand_table(r, "chance", "M2", scope=lg)], ignore_index=True)
        coefs.append(coef_table(r, "chance", "M2", scope=lg))
    pd.concat(coefs, ignore_index=True).to_csv(RES / "coefficients.csv", index=False)
    E.to_csv(RES / "estimands.csv", index=False)
    pd.DataFrame(fits).to_csv(RES / "fits.csv", index=False)
    print(E[(E.kind == "chance") & (E.spec == "M2") & (E.scope == "all")].round(4).to_string(index=False))

    # ------------------------------------------------------------------ out of sample: H1 -> H2
    h1, h2 = S[S.half == "H1"].reset_index(drop=True), S[S.half == "H2"].reset_index(drop=True)
    oos, per_match = [], {}
    fitted = {}
    for spec in ("M0", "M1", "M2"):
        r = model.fit(h1, spec)
        fitted[spec] = r
        per_match[spec] = model.pp_loglik(h2, r)
    pm = pd.DataFrame(per_match)
    n = len(pm)
    for a, b in (("M1", "M0"), ("M2", "M1"), ("M2", "M0")):
        d = (pm[a] - pm[b]).values
        boots = [d[rng.integers(0, n, n)].sum() for _ in range(4000)]
        oos.append(dict(comparison=f"{a} vs {b}", h2_matches=n, h2_events=int(h2.y.sum()), ll_gain_total=d.sum(),
                        ci_lo=np.percentile(boots, 2.5), ci_hi=np.percentile(boots, 97.5),
                        ll_gain_per_match=d.mean(), share_matches_improved=float(np.mean(d > 0))))
    pd.DataFrame(oos).to_csv(RES / "out_of_sample.csv", index=False)
    print(pd.DataFrame(oos).round(3).to_string(index=False))

    # ------------------------------------------------------------------ time-rescaling goodness of fit on H2
    gof, curves = [], {"t": np.linspace(0, 4, 81)}
    for spec in ("M0", "M2"):
        gaps, cens, u = time_rescaling(h2, fitted[spec])
        km, ex = km_vs_exponential(gaps, cens, curves["t"])
        curves[f"km_{spec}"] = km
        curves["exp"] = ex
        gof.append(dict(spec=spec, n_events=len(gaps), n_windows=len(cens),
                        km_max_abs_diff=float(np.max(np.abs(km - ex))),
                        uniform_ks=float(stats.kstest(u, "uniform").statistic),
                        uniform_ks_p=float(stats.kstest(u, "uniform").pvalue)))
    pd.DataFrame(gof).to_csv(RES / "time_rescaling.csv", index=False)
    pd.DataFrame(curves).to_csv(RES / "time_rescaling_km.csv", index=False)
    print(pd.DataFrame(gof).round(4).to_string(index=False))

    # ------------------------------------------------------------------ observed vs expected after goals, H2
    lam = model.predict_rate(h2, fitted["M1"])
    h2 = h2.assign(exp=lam * h2.dur.values)
    rows = []
    for side, prefix in (("scorer", "gf"), ("conceder", "ga")):
        for w, lab in zip(model.WIN, ("0-2", "2-5", "5-10", "10-20"), strict=True):
            m = h2[f"{prefix}_{w}"].values > 0
            per = h2[m].groupby("match_id").agg(o=("y", "sum"), e=("exp", "sum"))
            o, e = per.o.values, per.e.values
            boots = []
            for _ in range(4000):
                k = rng.integers(0, len(per), len(per))
                boots.append(o[k].sum() / e[k].sum())
            rows.append(dict(side=side, window=lab, observed=o.sum(), expected=e.sum(), o_over_e=o.sum() / e.sum(),
                             ci_lo=np.percentile(boots, 2.5), ci_hi=np.percentile(boots, 97.5),
                             exposure_min=h2[m].dur.sum() / 60, matches=len(per)))
    pd.DataFrame(rows).to_csv(RES / "after_goal_observed_vs_expected.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
