"""Effects in expected chances (main model M2, full season, chances), with delta-method 95% intervals.

At the average chance rate r (chances per team per minute of play), the extra chances over the windows after an
event are sum_w (exp(c_w) - 1) * r * length_w, where c_w is the relevant log rate ratio in window w. Momentum
contrasts (exploratory) use c_w = own_w - own_early; goal effects use the totals (chance term + goal term) for the
scorer and the conceder; the score effect is gs_-1 (trailing by one) or gs_+1 (leading by one) over 30 minutes.
Writes results/effects_in_chances.csv.
"""
import numpy as np
import pandas as pd

import model
from analysis import RES, load

LEN = {"w0_2": 2.0, "w2_5": 3.0, "w5_10": 5.0, "w10_20": 10.0}


def extra_chances(res, terms_by_window, r):
    b, V = res["beta"], res["cov"]
    total, grad = 0.0, pd.Series(0.0, index=b.index)
    for w, terms in terms_by_window.items():
        c = sum(sgn * b[t] for t, sgn in terms.items())
        total += (np.exp(c) - 1) * r * LEN.get(w, w if isinstance(w, float) else 0)
        for t, sgn in terms.items():
            grad[t] += sgn * np.exp(c) * r * LEN.get(w, w if isinstance(w, float) else 0)
    se = float(np.sqrt(grad @ V @ grad))
    return total, se


def main():
    S = load("chance")
    r = S.y.sum() / (S.dur.sum() / 60.0)                # chances per team per minute of play
    res = model.fit(S, "M2")
    rows = []
    sets = {
        "own chance -> own extra chances, 2-20 min (momentum, vs earlier-in-match level)":
            {w: {f"own_{w}": 1, "own_early": -1} for w in ("w2_5", "w5_10", "w10_20")},
        "own chance -> own extra chances, 0-2 min (restart dip)": {"w0_2": {"own_w0_2": 1, "own_early": -1}},
        "scorer after scoring -> extra chances, 0-20 min": {w: {f"own_{w}": 1, f"gf_{w}": 1} for w in LEN},
        "conceder after conceding -> extra chances, 0-20 min": {w: {f"opp_{w}": 1, f"ga_{w}": 1} for w in LEN},
        "trailing by one goal -> extra chances over 30 min": {30.0: {"gs_-1": 1}},
        "leading by one goal -> extra chances over 30 min": {30.0: {"gs_+1": 1}},
    }
    for name, tw in sets.items():
        est, se = extra_chances(res, tw, r)
        rows.append(dict(effect=name, extra_chances=est, ci_lo=est - 1.96 * se, ci_hi=est + 1.96 * se,
                         avg_rate_per_min=r))
    out = pd.DataFrame(rows)
    out.to_csv(RES / "effects_in_chances.csv", index=False)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
