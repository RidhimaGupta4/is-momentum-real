"""Export the main model's coefficients (chances, full season) and headline numbers to explorer/data.js.
Only fitted coefficients and summary numbers are exported; no StatsBomb event data.
Usage: python explorer/export.py"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def main():
    C = pd.read_csv(RES / "coefficients.csv")
    c = C[(C.kind == "chance") & (C.spec == "M2") & (C.scope == "all")].set_index("term")
    E = pd.read_csv(RES / "effects_in_chances.csv").set_index("effect")
    V = pd.read_csv(RES / "cov_chance_M2.csv", index_col=0)
    assert list(V.index) == list(c.index) == list(V.columns)
    data = dict(beta={k: round(float(v), 9) for k, v in c.coef.items()},
                cov_names=list(V.index), cov=[[float(f"{x:.6e}") for x in row] for row in V.values],
                effects={k: dict(est=round(float(r.extra_chances), 4), lo=round(float(r.ci_lo), 4),
                                 hi=round(float(r.ci_hi), 4)) for k, r in E.iterrows()},
                avg_rate=round(float(E.avg_rate_per_min.iloc[0]), 5))
    out = ROOT / "explorer" / "data.js"
    out.write_text("const DATA = " + json.dumps(data, separators=(",", ":")) + ";\n")
    print("wrote", out, f"{out.stat().st_size} bytes")


if __name__ == "__main__":
    main()
