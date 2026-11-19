"""Brute-force checks of the segment table: covariates recomputed from raw events must match, and nothing may use
information from at or after the moment being predicted."""
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
def D():
    return {k: pd.read_parquet(DATA / f"{k}.parquet") for k in ("matches", "windows", "events", "dead", "onpitch")}


@pytest.fixture(scope="module")
def S():
    return pd.read_parquet(DATA / "segments_chance.parquet")


def brute(D, mid, tid, p, t):
    """Covariates for team tid at time t (period p) from the raw tables, using only what happened before t."""
    E, OP = D["events"], D["onpitch"]
    M = D["matches"].set_index("match_id").loc[mid]
    oid = M.away_id if tid == M.home_id else M.home_id
    ch = E[(E.match_id == mid) & (E.kind == "chance")]
    gl = E[(E.match_id == mid) & (E.kind == "goal")]

    def before(df, team):
        return df[(df.team_id == team) & ((df.period < p) | ((df.period == p) & (df.t < t)))]

    def windows(df, team):
        x = df[(df.team_id == team) & (df.period == p)]
        d = t - x.t.values
        edges = [0, 120, 300, 600, 1200]
        return [int(((d > edges[k]) & (d <= edges[k + 1])).sum()) for k in range(4)]

    own_b, opp_b = before(ch, tid), before(ch, oid)
    own_w, opp_w = windows(ch, tid), windows(ch, oid)
    early_own = len(own_b) - sum(own_w)
    early_opp = len(opp_b) - sum(opp_w)
    gs = len(before(gl, tid)) - len(before(gl, oid))
    op = OP[(OP.match_id == mid) & (OP.period == p) & (OP.start < t) & (OP.end >= t)]
    adv = int((op.team_id == tid).sum()) - int((op.team_id == oid).sum())
    return dict(own=own_w, opp=opp_w, gf=windows(gl, tid), ga=windows(gl, oid), own_early=early_own,
                opp_early=early_opp, gs=int(np.clip(gs, -2, 2)), adv=int(np.clip(adv, -1, 1)),
                clock=int(min(t // 900, 3)))


@needs
def test_covariates_at_events_match_brute_force(D, S):
    rng = np.random.default_rng(1)
    ev = D["events"][D["events"].kind == "chance"].reset_index(drop=True)
    pick = ev.iloc[rng.choice(len(ev), 400, replace=False)]
    for e in pick.itertuples():
        seg = S[(S.match_id == e.match_id) & (S.team_id == e.team_id) & (S.period == e.period)
                & (S.a < e.t) & (S.b >= e.t)]
        assert len(seg) == 1, (e.match_id, e.t)
        r = seg.iloc[0]
        assert r.y >= 1 and abs(r.b - e.t) < 1e-6        # the event closes its segment
        b = brute(D, e.match_id, e.team_id, e.period, e.t)
        assert [r[f"own_{w}"] for w in ("w0_2", "w2_5", "w5_10", "w10_20")] == b["own"], e
        assert [r[f"opp_{w}"] for w in ("w0_2", "w2_5", "w5_10", "w10_20")] == b["opp"], e
        assert [r[f"gf_{w}"] for w in ("w0_2", "w2_5", "w5_10", "w10_20")] == b["gf"], e
        assert [r[f"ga_{w}"] for w in ("w0_2", "w2_5", "w5_10", "w10_20")] == b["ga"], e
        assert (r.own_early, r.opp_early, r.gs, r.adv, r.clock) == (b["own_early"], b["opp_early"], b["gs"],
                                                                    b["adv"], b["clock"]), e


@needs
def test_every_event_in_one_segment_and_exposure(D, S):
    assert int(S.y.sum()) == int((D["events"].kind == "chance").sum())
    W = D["windows"]
    dead = (D["dead"].end - D["dead"].start).sum()
    assert abs(S.dur.sum() - (W["T"].sum() - 2 * dead)) < 1.0      # both teams lose the dead time


@needs
def test_prematch_strength_uses_strictly_earlier_dates(D):
    from segments import prematch_strength
    M = D["matches"]
    st = prematch_strength(D, "chance").merge(M[["match_id", "league", "date"]], on="match_id")
    first = M.groupby("league").date.min()
    on_first = st[st.date == st.league.map(first)]
    assert (on_first.att == 0).all() and (on_first.dfn == 0).all() and (on_first.n_prior == 0).all()
    # changing every outcome on or after a date must not change strengths for matches on that date
    E = D["events"].copy()
    cut_date = sorted(M.date.unique())[150]
    later = M.match_id[M.date >= cut_date]
    E2 = E[~(E.match_id.isin(later) & (E.kind == "chance"))]
    st2 = prematch_strength(dict(D, events=E2), "chance").merge(M[["match_id", "date"]], on="match_id")
    a = st[st.date == cut_date].sort_values(["match_id", "team_id"])
    b = st2[st2.date == cut_date].sort_values(["match_id", "team_id"])
    np.testing.assert_allclose(a.att.values, b.att.values)
    np.testing.assert_allclose(a.dfn.values, b.dfn.values)


@needs
def test_halves_strictly_ordered(D):
    M = D["matches"]
    for lg, g in M.groupby("league"):
        assert g[g.half == "H1"].date.max() < g[g.half == "H2"].date.min(), lg


@needs
def test_goals_equal_scores(D):
    g = D["events"][D["events"].kind == "goal"].groupby(["match_id", "team_id"]).size()
    for m in D["matches"].itertuples():
        assert g.get((m.match_id, m.home_id), 0) == m.home_score
        assert g.get((m.match_id, m.away_id), 0) == m.away_score


@needs
def test_no_events_inside_dead_time(D):
    k = D["events"].merge(D["dead"], on=["match_id", "period"])
    assert not ((k.t > k.start) & (k.t < k.end)).any()
