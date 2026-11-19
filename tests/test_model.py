"""Model correctness (against statsmodels and an explicit dummy-variable fit), the H1 -> H2 separation, and parity
between the interactive page's JavaScript intensity and the Python model."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DATA = ROOT / "data"
needs = pytest.mark.skipif(not (DATA / "segments_chance.parquet").exists(), reason="derived data not built")


@pytest.fixture(scope="module")
def S():
    return pd.read_parquet(DATA / "segments_chance.parquet")


@needs
def test_solver_matches_statsmodels(S):
    sm = pytest.importorskip("statsmodels.api")
    import model
    s = S[S.match_id.isin(np.sort(S.match_id.unique())[:300])].reset_index(drop=True)
    r = model.fit(s, "M2")
    X, _ = model.design(s, "M2")
    g = sm.GLM(s.y.values, X, family=sm.families.Poisson(), offset=np.log(s.dur.values)).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(s.match_id)[0]})
    np.testing.assert_allclose(r["beta"].values, g.params, atol=1e-9)
    np.testing.assert_allclose(np.sqrt(np.diag(r["cov"].values)), g.bse, rtol=1e-3)


@needs
def test_fixed_effects_equal_explicit_dummies(S):
    sm = pytest.importorskip("statsmodels.api")
    import model
    s = S[S.match_id.isin(np.sort(S.match_id.unique())[:80])].reset_index(drop=True)
    r = model.fit(s, "FE")
    X, _ = model.design(s, "FE")
    key = s.match_id.astype(str) + "_" + s.team_id.astype(str)
    keep = (s.groupby(key).y.transform("sum") > 0).values
    D = pd.get_dummies(key[keep]).values.astype(float)
    g = sm.GLM(s.y.values[keep], np.column_stack([X[keep], D]), family=sm.families.Poisson(),
               offset=np.log(s.dur.values[keep])).fit()
    np.testing.assert_allclose(r["beta"].values, g.params[:X.shape[1]], atol=1e-8)


@needs
def test_out_of_sample_split_is_by_whole_match_and_time(S):
    h1, h2 = S[S.half == "H1"], S[S.half == "H2"]
    assert set(h1.match_id).isdisjoint(set(h2.match_id))
    for lg in S.league.unique():
        assert h1[h1.league == lg].date.max() < h2[h2.league == lg].date.min()
    src = (ROOT / "src" / "analysis.py").read_text()
    assert 'r = model.fit(h1, spec)' in src and 'model.pp_loglik(h2, r)' in src


@needs
@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_explorer_intensity_equals_python_model():
    """explorer/rate.js must equal the Python model for the same match history in every mode the page uses:
    full model; restarts only (events up to 2 min old in the 0-2 min terms, older ones in the earlier-in-match
    terms); momentum and restarts off (all past events in the earlier-in-match terms); score only; no events."""
    import model
    C = pd.read_csv(ROOT / "results" / "coefficients.csv")
    c = C[(C.kind == "chance") & (C.spec == "M2") & (C.scope == "all")]
    res = dict(beta=pd.Series(c.coef.values, index=c.term.values), spec="M2")
    events = [dict(team="home", kind="chance", half=1, t=300.0), dict(team="home", kind="goal", half=1, t=1500.0),
              dict(team="away", kind="chance", half=1, t=2000.0), dict(team="away", kind="goal", half=2, t=400.0),
              dict(team="away", kind="chance", half=2, t=700.0), dict(team="home", kind="chance", half=2, t=1500.0)]
    queries = [("home", 1, 200.0), ("away", 1, 1520.0), ("home", 1, 1600.0), ("away", 1, 1700.0),
               ("away", 1, 2050.0), ("away", 1, 2750.0), ("home", 2, 100.0), ("home", 2, 500.0), ("away", 2, 800.0),
               ("away", 2, 1650.0), ("home", 2, 2800.0)]
    modes = ["full", "restart", "off", "score", "none"]
    att = {"home": np.log(1.2), "away": np.log(0.9)}
    js = (ROOT / "explorer" / "data.js").read_text() + (ROOT / "explorer" / "rate.js").read_text()
    js += f"\nconst ev = {json.dumps(events)}; const q = {json.dumps(queries)}; const modes = {json.dumps(modes)};\n"
    js += f"const o = md => ({{attHome: {att['home']}, attAway: {att['away']}, mode: md}});\n"
    js += "console.log(JSON.stringify(modes.map(md => q.map(x => intensity(x[0], x[1], x[2], ev, o(md))))));"
    out = dict(zip(modes, json.loads(subprocess.run(["node", "-e", js], capture_output=True, text=True,
                                                    check=True).stdout), strict=True))
    edges = [0, 120, 300, 600, 1200]
    n_dead = 0
    for qi, (side, half, t) in enumerate(queries):
        other = "away" if side == "home" else "home"
        for mode in modes:
            evs = [] if mode == "none" else events
            before = [e for e in evs if e["half"] < half or (e["half"] == half and e["t"] < t)]
            dead = any(e["kind"] == "goal" and e["half"] == half and e["t"] < t < e["t"] + 56 for e in evs)
            r_js = out[mode][qi]
            if dead:
                assert r_js == 0
                n_dead += 1
                continue
            gf = sum(e["kind"] == "goal" and e["team"] == side for e in before)
            ga = sum(e["kind"] == "goal" and e["team"] == other for e in before)
            row = dict(period=half, clock=min(int(t // 900), 3), gs=int(np.clip(gf - ga, -2, 2)), adv=0,
                       home=int(side == "home"), att=att[side], dfn=0.0, y=0.0, dur=1.0, match_id=1,
                       own_early=0, opp_early=0)
            for w in model.WIN:
                row[f"own_{w}"] = row[f"opp_{w}"] = row[f"gf_{w}"] = row[f"ga_{w}"] = 0
            limit = {"full": 1200, "restart": 120, "off": 0, "score": None, "none": None}[mode]
            if limit is not None:
                in_win = [e for e in before if e["half"] == half and t - e["t"] <= limit] if limit else []
                for team, pre in ((side, "own"), (other, "opp")):
                    row[f"{pre}_early"] = sum(e["team"] == team for e in before) - \
                        sum(e["team"] == team for e in in_win)
                    for k, w in enumerate(model.WIN):
                        row[f"{pre}_{w}"] = sum(e["team"] == team and edges[k] < t - e["t"] <= edges[k + 1]
                                                for e in in_win)
                for team, pre in ((side, "gf"), (other, "ga")):
                    for k, w in enumerate(model.WIN):
                        row[f"{pre}_{w}"] = sum(e["team"] == team and e["kind"] == "goal"
                                                and edges[k] < t - e["t"] <= edges[k + 1] for e in in_win)
            p_py = float(model.predict_rate(pd.DataFrame([row]), res)[0])
            assert abs(r_js - p_py) / p_py < 1e-5, (mode, side, half, t, r_js, p_py)
    assert n_dead >= 1                      # the dead-time branch was exercised


def _row(model, events, side, half, t, mode, att):
    """Python-side covariates for the page's modes (independent of explorer/rate.js)."""
    edges = [0, 120, 300, 600, 1200]
    other = "away" if side == "home" else "home"
    evs = [] if mode == "none" else events
    before = [e for e in evs if e["half"] < half or (e["half"] == half and e["t"] < t)]
    if any(e["kind"] == "goal" and e["half"] == half and e["t"] < t < e["t"] + 56 for e in evs):
        return None
    gf = sum(e["kind"] == "goal" and e["team"] == side for e in before)
    ga = sum(e["kind"] == "goal" and e["team"] == other for e in before)
    row = dict(period=half, clock=min(int(t // 900), 3), gs=int(np.clip(gf - ga, -2, 2)), adv=0,
               home=int(side == "home"), att=att[side], dfn=0.0, y=0.0, dur=1.0, match_id=1, own_early=0, opp_early=0)
    for w in model.WIN:
        row[f"own_{w}"] = row[f"opp_{w}"] = row[f"gf_{w}"] = row[f"ga_{w}"] = 0
    limit = {"full": 1200, "restart": 120, "off": 0, "score": None, "none": None}[mode]
    if limit is not None:
        in_win = [e for e in before if e["half"] == half and t - e["t"] <= limit] if limit else []
        for team, pre in ((side, "own"), (other, "opp")):
            row[f"{pre}_early"] = sum(e["team"] == team for e in before) - sum(e["team"] == team for e in in_win)
            for k, w in enumerate(model.WIN):
                row[f"{pre}_{w}"] = sum(e["team"] == team and edges[k] < t - e["t"] <= edges[k + 1] for e in in_win)
        for team, pre in ((side, "gf"), (other, "ga")):
            for k, w in enumerate(model.WIN):
                row[f"{pre}_{w}"] = sum(e["team"] == team and e["kind"] == "goal"
                                        and edges[k] < t - e["t"] <= edges[k + 1] for e in in_win)
    return row


@needs
@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_explorer_totals_and_intervals_equal_python_delta_method():
    """Match totals and the 95% intervals shown on the page (delta method with the cluster-robust covariance) must
    equal an independent Python computation."""
    import model
    C = pd.read_csv(ROOT / "results" / "coefficients.csv")
    c = C[(C.kind == "chance") & (C.spec == "M2") & (C.scope == "all")]
    res = dict(beta=pd.Series(c.coef.values, index=c.term.values), spec="M2")
    V = pd.read_csv(ROOT / "results" / "cov_chance_M2.csv", index_col=0)
    events = [dict(team="home", kind="chance", half=1, t=660.0), dict(team="home", kind="goal", half=1, t=1440.0),
              dict(team="away", kind="chance", half=1, t=1980.0), dict(team="away", kind="goal", half=2, t=1500.0)]
    att = {"home": np.log(1.1), "away": np.log(0.95)}
    pairs = [("full", None), ("score", "none"), ("full", "restart"), ("restart", "off")]
    js = (ROOT / "explorer" / "data.js").read_text() + (ROOT / "explorer" / "rate.js").read_text()
    js += f"\nconst ev = {json.dumps(events)}; const pairs = {json.dumps(pairs)};\n"
    js += f"const o = md => ({{attHome: {att['home']}, attAway: {att['away']}, mode: md}});\n"
    js += ("console.log(JSON.stringify(['home', 'away'].map(sd => pairs.map(([a, b]) => "
           "difference(matchTotal(sd, ev, o(a)), b ? matchTotal(sd, ev, o(b)) : null)))));")
    out = json.loads(subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True).stdout)

    def py_total(side, mode):
        rows = [r for half in (1, 2) for t in np.arange(0, 47 * 60, 10.0)
                if (r := _row(model, events, side, half, float(t), mode, att)) is not None]
        S = pd.DataFrame(rows)
        X, names = model.design(S, "M2")
        lam = np.exp(X @ res["beta"][names].values)
        return float((lam * 10).sum()), pd.Series((X * (lam * 10)[:, None]).sum(0), index=names)

    for si, side in enumerate(("home", "away")):
        for pi, (a, b) in enumerate(pairs):
            ta, ga = py_total(side, a)
            tb, gb = py_total(side, b) if b else (0.0, ga * 0)
            g = (ga - gb)[V.index]
            se = float(np.sqrt(g.values @ V.values @ g.values))
            js_r = out[si][pi]
            assert abs(js_r["est"] - (ta - tb)) < 1e-6 * max(1, abs(ta)), (side, a, b, js_r, ta - tb)
            assert abs(js_r["se"] - se) < 1e-6 * max(1e-3, se) + 1e-9, (side, a, b, js_r["se"], se)
