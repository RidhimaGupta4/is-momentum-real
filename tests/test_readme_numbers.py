"""Every headline number in README.md is recomputed from results/ and must appear in the README as written."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
README = " ".join((ROOT / "README.md").read_text().split())


def has(s):
    s = " ".join(s.split())
    assert s in README, f"README does not contain {s!r}"


def f3(x):
    return f"{x:.3f}"


def est(kind="chance", spec="M2", scope="all"):
    E = pd.read_csv(RES / "estimands.csv")
    return E[(E.kind == kind) & (E.spec == spec) & (E.scope == scope)].set_index("estimand")


def test_effects_in_chances():
    E = pd.read_csv(RES / "effects_in_chances.csv").set_index("effect")
    t = E.loc["trailing by one goal -> extra chances over 30 min"]
    has(f"**+{t.extra_chances:.2f} extra chances over 30 minutes** (95% CI {t.ci_lo:.2f} to {t.ci_hi:.2f})")
    lead = E.loc["leading by one goal -> extra chances over 30 min"]
    has(f"leading by one: {lead.extra_chances:.2f} ({lead.ci_lo:.2f} to {lead.ci_hi:.2f})".replace("-", "−"))
    m = E.loc["own chance -> own extra chances, 2-20 min (momentum, vs earlier-in-match level)"]
    has(f"**+{m.extra_chances:.3f} extra chances** for the same team over the next 2–20 minutes "
        f"({m.ci_lo:.3f} to {m.ci_hi:.3f})")
    assert round(1 / m.extra_chances) == 21
    assert E.loc["own chance -> own extra chances, 0-2 min (restart dip)"].extra_chances < 0


def test_primary_estimands():
    e = est()
    for w in ("2-5", "5-10"):
        r = e.loc[f"scorer_after_scoring_{w}"]
        has(f"{f3(r.rate_ratio)} ({f3(r.rr_lo)} to {f3(r.rr_hi)})")
    for w in ("0-2", "2-5", "5-10"):
        r = e.loc[f"conceder_after_conceding_{w}"]
        has(f"{f3(r.rate_ratio)} ({f3(r.rr_lo)} to {f3(r.rr_hi)})")
    assert e.rr_hi[[f"conceder_after_conceding_{w}" for w in ("0-2", "2-5", "5-10")]].max() <= 1.0305
    has(f"rate ratio {f3(e.loc['self_excitation_0_2_vs_earlier'].rate_ratio)} for the same team and "
        f"{f3(e.loc['cross_excitation_0_2_vs_earlier'].rate_ratio)} for the opponent")
    p = e.p_holm
    assert p["scorer_after_scoring_0-2"] < 0.001 and p["conceder_after_conceding_0-2"] < 0.001
    later = [k for k in p.index if k.startswith(("scorer", "conceder")) and not k.endswith("0-2")]
    assert p[later].min() >= 0.135 - 5e-4 and f"{p[later].min():.3f}" == "0.135"
    fe = est(spec="FE")
    row = (f"| Scoring team, fixed effects | {f3(fe.loc['scorer_after_scoring_0-2'].rate_ratio)} | "
           f"{f3(fe.loc['scorer_after_scoring_2-5'].rate_ratio)} | "
           f"{f3(fe.loc['scorer_after_scoring_5-10'].rate_ratio)} |")
    has(row)
    row = (f"| Conceding team, fixed effects | {f3(fe.loc['conceder_after_conceding_0-2'].rate_ratio)} | "
           f"{f3(fe.loc['conceder_after_conceding_2-5'].rate_ratio)} | "
           f"{f3(fe.loc['conceder_after_conceding_5-10'].rate_ratio)} |")
    has(row)


def test_momentum_contrasts():
    C = pd.read_csv(RES / "explore_momentum_contrasts.csv")
    c = C[(C.kind == "chance") & (C.side == "own")]
    m = c[c.spec == "M2"].set_index("window")
    f = c[c.spec == "FE"].set_index("window")
    has(f"rate ratio {f3(m.rate_ratio['2-5'])} ({f3(m.rr_lo['2-5'])} to {f3(m.rr_hi['2-5'])}) at 2–5 minutes, "
        f"{f3(m.rate_ratio['5-10'])} at 5–10 and {f3(m.rate_ratio['10-20'])} at 10–20")
    has(f"gives {f3(f.rate_ratio['2-5'])}, {f3(f.rate_ratio['5-10'])} and {f3(f.rate_ratio['10-20'])}")
    o = C[(C.kind == "chance") & (C.side == "opp") & (C.spec == "M2")].set_index("window")
    has(f"({f3(o.rate_ratio['2-5'])} at 2–5 minutes)")
    op = C[(C.kind == "open_chance") & (C.side == "own") & (C.spec == "M2")].set_index("window")
    has(f"open-play chances only ({f3(op.rate_ratio['2-5'])} at 2–5 minutes)")
    sh = C[(C.kind == "shot") & (C.side == "own") & (C.spec == "M2")].set_index("window")
    has(f"(rate ratio {f3(sh.rate_ratio['0-2'])} relative to the earlier-in-match level)")


def test_out_of_sample_and_fit():
    oos = pd.read_csv(RES / "out_of_sample.csv").set_index("comparison")
    a, g = oos.loc["M1 vs M0"], oos.loc["M2 vs M1"]
    has(f"improves the log-likelihood by {a.ll_gain_total:.0f} ({a.ci_lo:.0f} to {a.ci_hi:.0f}) and goal terms by a "
        f"further {g.ll_gain_total:.0f} ({g.ci_lo:.0f} to {g.ci_hi:.0f})")
    assert int(a.h2_matches) == 760
    D = pd.read_csv(RES / "explore_oos_decomposition.csv")
    v = D.ll_gain_total.values
    has(f"{v[0]:.0f} from the earlier-in-match counts, {v[1]:.0f} from the restart dip, {v[2]:.0f} "
        f"({D.ci_lo[2]:.0f} to {D.ci_hi[2]:.0f}) from the momentum")
    T = pd.read_csv(RES / "time_rescaling.csv").set_index("spec")
    has(f"(largest difference {T.km_max_abs_diff['M2']:.3f} for the main model, {T.km_max_abs_diff['M0']:.3f} for "
        f"the baseline)")
    has(f"(KS {T.uniform_ks['M2']:.4f}, p = {T.uniform_ks_p['M2']:.4f})")
    assert int(T.n_events["M2"]) == 17464
    A = pd.read_csv(RES / "after_goal_observed_vs_expected.csv").set_index(["side", "window"])
    r = A.loc[("conceder", "0-2")]
    has(f"(observed / expected {r.o_over_e:.2f}, {r.ci_lo:.2f} to {r.ci_hi:.2f})")


def test_hazard_and_counts():
    H = pd.read_csv(RES / "explore_post_chance_hazard.csv")
    has(f"{H.rel_hazard_any.iloc[0]:.3f} times the average rate")
    E = pd.read_parquet(ROOT / "data" / "events.parquet") if (ROOT / "data" / "events.parquet").exists() else None
    if E is not None:
        k = E.kind.value_counts()
        for kind, txt in (("chance", "34,991"), ("open_chance", "21,938"), ("box_entry", "43,455"),
                          ("shot", "37,888"), ("goal", "3,997")):
            assert f"{k[kind]:,}" == txt
            has(txt)
        D = pd.read_parquet(ROOT / "data" / "dead.parquet")
        assert round(float(np.median(D.end - D.start))) == 56


def test_per_league_claims():
    E = pd.read_csv(RES / "estimands.csv")
    e = E[(E.kind == "chance") & (E.spec == "M2") & (E.scope != "all")]
    for est_ in ("self_excitation_0_2_vs_earlier", "cross_excitation_0_2_vs_earlier", "scorer_after_scoring_0-2"):
        assert (e[e.estimand == est_].rr_hi < 1).all()
    c = e[e.estimand == "conceder_after_conceding_0-2"].set_index("scope")
    assert (c.rate_ratio < 1).all()
    for lg in ("Ligue 1", "Serie A"):
        assert c.rr_hi[lg] > 1
        has(f"{lg} ({f3(c.rate_ratio[lg])}, {f3(c.rr_lo[lg])} to {f3(c.rr_hi[lg])})")
    for lg in ("La Liga", "Premier League"):
        assert c.rr_hi[lg] < 1


def test_clustering_claim():
    K = pd.read_csv(RES / "explore_clustering.csv")
    base = K[K.clusters == "match"].set_index("estimand")
    worst = 0.0
    for lab in ("team", "two-way team x opponent"):
        k = K[K.clusters == lab].set_index("estimand")
        worst = max(worst, float((k.se / base.se).max()))
        same_side = np.sign(np.log(k.rr_lo)) * np.sign(np.log(k.rr_hi)) == \
            np.sign(np.log(base.rr_lo)) * np.sign(np.log(base.rr_hi))
        assert same_side.all()                      # whether a CI includes 1 never changes
    assert worst <= 1.125
    has("changes the standard errors of the eight primary estimates by at most 12% and no conclusion")
    c = K[K.estimand.str.startswith("conceder")]
    assert c.rr_hi.max() < 1.035                    # "more than about 3% ruled out"
    has("An increase of more than about 3% is ruled out in every window.")


def test_no_heterogeneity_claim():
    m, n = est(spec="M2"), est(spec="NOHET")
    common = [k for k in n.index if k in m.index]
    assert len(common) == 6
    assert (m.rate_ratio[common] - n.rate_ratio[common]).abs().max() < 0.0075
    has(f"(for example {f3(n.rate_ratio['conceder_after_conceding_2-5'])} against "
        f"{f3(m.rate_ratio['conceder_after_conceding_2-5'])} for the conceding team at 2–5 minutes)")


def test_page_hard_coded_percentages():
    """The page's notes quote the score effects in words; they must match the fitted coefficients."""
    page = (ROOT / "explorer" / "template.html").read_text()
    C = pd.read_csv(RES / "coefficients.csv")
    c = C[(C.kind == "chance") & (C.spec == "M2") & (C.scope == "all")].set_index("term").coef
    up = round(100 * (np.exp(c["gs_-1"]) - 1))
    down = round(100 * (1 - np.exp(c["gs_+1"])))
    assert f"a trailing team creates about {up}% more chances" in " ".join(page.split())
    assert f"a leading team about {down}% fewer" in " ".join(page.split())
    assert "const DEAD_AFTER_GOAL = 56;" in (ROOT / "explorer" / "rate.js").read_text()


def test_after_goal_table_main_rows():
    e = est()

    def cell(k):
        r = e.loc[k]
        return f"{f3(r.rate_ratio)} ({f3(r.rr_lo)}–{f3(r.rr_hi)})"

    has("| Scoring team, main model | " + " | ".join(cell(f"scorer_after_scoring_{w}") for w in ("0-2", "2-5", "5-10"))
        + " |")
    has("| Conceding team, main model | "
        + " | ".join(cell(f"conceder_after_conceding_{w}") for w in ("0-2", "2-5", "5-10")) + " |")
