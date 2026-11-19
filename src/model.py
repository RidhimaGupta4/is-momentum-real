"""Log-linear self-exciting point process fitted as an exact piecewise-exponential Poisson likelihood.

Model specs (DESIGN.md):
  M0       baseline only: period x clock bin, game state, player advantage, home, pre-match strength
  M1       M0 + own / opponent chances in (0,2], (2,5], (5,10], (10,20] min and earlier in the match
  M2       M1 + goals scored / conceded in the same windows                      (= main model)
  NOHET    M2 without the earlier-in-match counts                                (biased towards momentum)
  FE       M2 with one intercept per team-match instead of home / strength terms (biased against momentum),
           fitted by the exact conditional (multinomial) likelihood, which removes the intercepts.
Standard errors: cluster-robust sandwich, clusters = matches.
"""
import numpy as np
import pandas as pd

WIN = ["w0_2", "w2_5", "w5_10", "w10_20"]
HIST = [f"own_{w}" for w in WIN] + [f"opp_{w}" for w in WIN]
EARLY = ["own_early", "opp_early"]
GOALS = [f"gf_{w}" for w in WIN] + [f"ga_{w}" for w in WIN]
SPECS = {
    "M0": dict(base=True, hist=False, early=False, goals=False, fe=False),
    "M1": dict(base=True, hist=True, early=True, goals=False, fe=False),
    "M2": dict(base=True, hist=True, early=True, goals=True, fe=False),
    "NOHET": dict(base=True, hist=True, early=False, goals=True, fe=False),
    "FE": dict(base=False, hist=True, early=True, goals=True, fe=True),
    # exploratory decomposition (DESIGN.md, exploratory analyses)
    "X_early": dict(base=True, hist=False, early=True, goals=False, fe=False),
    "X_early_02": dict(base=True, hist=True, early=True, goals=False, fe=False, windows=["w0_2"]),
}


def design(S, spec):
    sp = SPECS[spec]
    cols, names = [], []

    def add(v, n):
        cols.append(np.asarray(v, float))
        names.append(n)

    if not sp["fe"]:
        add(np.ones(len(S)), "intercept")
    for p in (1, 2):
        for c in range(4):
            if (p, c) != (1, 0):
                add((S.period.values == p) & (S.clock.values == c), f"clock_p{p}_{c}")
    for g in (-2, -1, 1, 2):
        add(S.gs.values == g, f"gs_{g:+d}")
    for a in (-1, 1):
        add(S.adv.values == a, f"adv_{a:+d}")
    if sp["base"] and not sp["fe"]:
        for c in ("home", "att", "dfn"):
            add(S[c].values, c)
    if sp["hist"]:
        wins = sp.get("windows", WIN)        # exploratory specs may use a subset of the history windows
        for c in HIST:
            if c.split("_", 1)[1] in wins:
                add(S[c].values, c)
    if sp["early"]:
        for c in EARLY:
            add(S[c].values, c)
    if sp["goals"]:
        for c in GOALS:
            add(S[c].values, c)
    return np.column_stack(cols), names


def _groups(keys):
    _, inv = np.unique(keys, return_inverse=True)
    return inv


def fit(S, spec, max_iter=50, tol=1e-10):
    """Newton-Raphson. Returns coefficients, names, cluster-robust covariance, log-likelihood (point process)."""
    X, names = design(S, spec)
    y, off = S.y.values, np.log(S.dur.values)
    fe = SPECS[spec]["fe"]
    clusters = _groups(S.match_id.values)
    if fe:
        grp = _groups(S.match_id.astype(str).values + "_" + S.team_id.astype(str).values)
        Yg = np.bincount(grp, weights=y)
        keep = Yg[grp] > 0                       # team-matches without events carry no information
        X, y, off, grp, clusters = X[keep], y[keep], off[keep], _groups(grp[keep]), clusters[keep]
        Yg = np.bincount(grp, weights=y)
    beta = np.zeros(X.shape[1])
    if not fe:
        beta[0] = np.log(y.sum() / np.exp(off).sum())
    for _ in range(max_iter):
        eta = X @ beta + off
        if fe:
            mu = np.exp(eta - eta.max())
            Mg = np.bincount(grp, weights=mu)
            p = mu / Mg[grp]
            r = y - Yg[grp] * p
            grad = X.T @ r
            xbar = np.column_stack([np.bincount(grp, weights=p * X[:, j]) for j in range(X.shape[1])])
            H = (X * (Yg[grp] * p)[:, None]).T @ X - (xbar * Yg[:, None]).T @ xbar
        else:
            mu = np.exp(eta)
            r = y - mu
            grad = X.T @ r
            H = (X * mu[:, None]).T @ X
        step = np.linalg.solve(H, grad)
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    else:
        raise RuntimeError(f"{spec}: Newton did not converge")
    eta = X @ beta + off
    if fe:
        mu = np.exp(eta - eta.max())
        Mg = np.bincount(grp, weights=mu)
        p = mu / Mg[grp]
        r = y - Yg[grp] * p
        ll = float(np.sum(y * np.log(np.where(y > 0, p, 1.0))))   # conditional log-likelihood
    else:
        mu = np.exp(eta)
        r = y - mu
        ll = float(np.sum(y * eta - mu) - np.sum(y * off))       # point-process log-likelihood
    score = X * r[:, None]
    G = clusters.max() + 1
    Sc = np.zeros((G, X.shape[1]))
    np.add.at(Sc, clusters, score)
    Hinv = np.linalg.inv(H)
    V = Hinv @ (Sc.T @ Sc) @ Hinv * G / (G - 1)
    return dict(beta=pd.Series(beta, index=names), cov=pd.DataFrame(V, index=names, columns=names), ll=ll,
                n_events=float(y.sum()), n_clusters=int(G), spec=spec)


def predict_rate(S, res):
    """Intensity (per second) in each segment under a fitted non-FE model."""
    X, names = design(S, res["spec"])
    assert names == list(res["beta"].index)
    return np.exp(X @ res["beta"].values)


def pp_loglik(S, res):
    """Exact point-process log-likelihood of segments S under a fitted non-FE model, per match."""
    lam = predict_rate(S, res)
    contrib = S.y.values * np.log(lam) - lam * S.dur.values
    return pd.Series(contrib).groupby(S.match_id.values).sum()


def combo(res, weights):
    """Linear combination sum_k w_k beta_k with cluster-robust standard error."""
    c = pd.Series(0.0, index=res["beta"].index)
    for k, v in weights.items():
        c[k] += v
    est = float(c @ res["beta"])
    se = float(np.sqrt(c @ res["cov"] @ c))
    return est, se
