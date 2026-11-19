"""All README figures. Every figure carries the StatsBomb logo and a data credit (StatsBomb user agreement).
Exploratory results are labelled as such on the figure."""
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from build_events import ROOT

FIG, RES = ROOT / "figures", ROOT / "results"
# dark "floodlit" theme, matching the interactive page (dark-mode steps of the validated categorical palette)
SURF, INK, INK2, MUTED, LINE = "#1a1a19", "#f0efec", "#c3c2b7", "#9a998f", "#34342f"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"
LOGO = plt.imread(ROOT / "assets" / "statsbomb_logo.png")
WLAB = ["0-2", "2-5", "5-10", "10-20"]
WIN = ["w0_2", "w2_5", "w5_10", "w10_20"]

plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.edgecolor": LINE, "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 11,
                     "axes.titlelocation": "left", "axes.axisbelow": True})


def credit(fig):
    w, h = fig.get_size_inches()
    lw = 1.25 / w
    lh = lw * w / h * LOGO.shape[0] / LOGO.shape[1]
    pad = 0.006
    chip = fig.add_axes([0.995 - lw - pad, 0.995 - lh - pad * w / h, lw + 2 * pad, lh + 2 * pad * w / h])
    chip.set_facecolor("#fcfcfb")
    chip.set_xticks([])
    chip.set_yticks([])
    for sp in chip.spines.values():
        sp.set_visible(False)
    ax = fig.add_axes([0.995 - lw, 0.995 - lh, lw, lh])
    ax.imshow(LOGO)
    ax.axis("off")
    fig.text(0.995 - lw - 0.02, 0.995 - lh / 2, "Data: StatsBomb Open Data, 2015/16", ha="right", va="center",
             fontsize=7.5, color=MUTED)


def save(fig, name):
    credit(fig)
    fig.savefig(FIG / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def explore_tag(ax):
    ax.text(1.0, 1.01, "EXPLORATORY", transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
            color=ORANGE, fontweight="bold")


# ----------------------------------------------------------------------------- fig 1: what drives chances
def fig_effects():
    E = pd.read_csv(RES / "effects_in_chances.csv")
    lab = {
        "leading by one goal -> extra chances over 30 min": "Leading by one goal (30 min)",
        "trailing by one goal -> extra chances over 30 min": "Trailing by one goal (30 min)",
        "own chance -> own extra chances, 2-20 min (momentum, vs earlier-in-match level)":
            "A chance: momentum, 2-20 min later*",
        "own chance -> own extra chances, 0-2 min (restart dip)": "A chance: first 2 min (restart)*",
        "scorer after scoring -> extra chances, 0-20 min": "Just scored (next 20 min)",
        "conceder after conceding -> extra chances, 0-20 min": "Just conceded (next 20 min)",
    }
    E = E.set_index("effect").loc[list(lab)]
    fig, ax = plt.subplots(figsize=(9.5, 4.3))
    y = np.arange(len(E))[::-1]
    cols = [BLUE if v > 0 else ORANGE for v in E.extra_chances]
    ax.barh(y, E.extra_chances, color=cols, height=0.55)
    ax.errorbar(E.extra_chances, y, xerr=[E.extra_chances - E.ci_lo, E.ci_hi - E.extra_chances], fmt="none",
                ecolor=INK, elinewidth=1, capsize=3)
    for yi, v, hi, lo in zip(y, E.extra_chances, E.ci_hi, E.ci_lo, strict=True):
        ax.text((hi if v > 0 else lo) + (0.02 if v > 0 else -0.02), yi, f"{v:+.2f}", va="center",
                ha="left" if v > 0 else "right", fontsize=9, color=INK2)
    ax.set_yticks(y, [lab[k] for k in E.index])
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlim(-0.8, 0.8)
    ax.set_xlabel("Extra chances for the team, compared with the same situation without the event")
    ax.set_title("The score changes a match far more than momentum does")
    ax.text(0, -0.2, "Main model, all 1,517 matches; 95% intervals (cluster-robust, delta method). "
            "* exploratory contrast against the team's level earlier in the match.", transform=ax.transAxes,
            fontsize=7.8, color=MUTED)
    ax.grid(axis="x", color=LINE, lw=0.5)
    save(fig, "fig1_what_moves_chances.png")


# ----------------------------------------------------------------------------- fig 2: seconds after a chance
def fig_hazard():
    H = pd.read_csv(RES / "explore_post_chance_hazard.csv")
    mid = (H.sec_lo + H.sec_hi) / 2
    fig, ax = plt.subplots(figsize=(9.5, 4.1))
    ax.axvspan(0, 25, color=LINE, alpha=0.9, lw=0)
    ax.text(12.5, 1.55, "ball dead,\nrestart", ha="center", fontsize=8, color=INK2)
    ax.plot(mid, H.rel_hazard_same, color=BLUE, lw=2, marker="o", ms=3.5, label="Same team's next chance")
    ax.plot(mid, H.rel_hazard_opp, color=ORANGE, lw=2, marker="o", ms=3.5, label="Opponent's next chance")
    ax.axhline(1, color=INK2, lw=0.8, ls="--")
    ax.set_xlim(0, 300)
    ax.set_ylim(0, 1.7)
    ax.set_xlabel("Seconds since a chance (gaps containing a goal excluded)")
    ax.set_ylabel("Rate of the next chance\nrelative to the average")
    ax.set_title("What happens in the seconds after a chance (model-free)")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.grid(axis="y", color=LINE, lw=0.5)
    explore_tag(ax)
    ax.text(0, -0.24, "Not adjusted for team strength or score: the level after ~40 s partly reflects busy matches "
            "being busy. The models handle that.", transform=ax.transAxes, fontsize=7.8, color=MUTED)
    save(fig, "fig2_seconds_after_a_chance.png")


# ----------------------------------------------------------------------------- fig 3: momentum profile
def fig_profile():
    C = pd.read_csv(RES / "explore_momentum_contrasts.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True, gridspec_kw=dict(wspace=0.08))
    for ax, side, title in ((axes[0], "own", "After the team's own chance"),
                            (axes[1], "opp", "After the opponent's chance")):
        x = np.arange(4)
        for k, (spec, c, name) in enumerate((("M2", BLUE, "Main model"),
                                             ("FE", ORANGE, "Team-match fixed effects"))):
            s = C[(C.kind == "chance") & (C.spec == spec) & (C.side == side)].set_index("window").loc[WLAB]
            xx = x + (k - 0.5) * 0.18
            ax.errorbar(xx, s.rate_ratio, yerr=[s.rate_ratio - s.rr_lo, s.rr_hi - s.rate_ratio], fmt="o", color=c,
                        ms=6, capsize=3, lw=1.3, label=name)
        ax.axhline(1, color=INK2, lw=0.8, ls="--")
        ax.set_xticks(x, [w + " min" for w in WLAB])
        ax.set_xlabel("Time since the chance")
        ax.set_title(title)
        ax.grid(axis="y", color=LINE, lw=0.5)
        explore_tag(ax)
    axes[0].set_ylabel("Chance rate vs the team's level\nearlier in the match (rate ratio)")
    axes[0].legend(frameon=False, fontsize=8.5, loc="lower right")
    axes[0].set_ylim(0.6, 1.15)
    fig.text(0.06, -0.06, "Each point: effect of one extra chance in that window minus the effect of one chance "
             "earlier in the match. The two models err in opposite directions; where they agree, the result is robust.",
             fontsize=7.8, color=MUTED)
    save(fig, "fig3_momentum_profile.png")


# ----------------------------------------------------------------------------- fig 4: after goals
def fig_goals():
    E = pd.read_csv(RES / "estimands.csv")
    OE = pd.read_csv(RES / "after_goal_observed_vs_expected.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), gridspec_kw=dict(wspace=0.3))
    ax = axes[0]
    x = np.arange(3)
    for k, (who, c, name) in enumerate((("scorer_after_scoring", BLUE, "Scoring team"),
                                        ("conceder_after_conceding", ORANGE, "Conceding team"))):
        for j, (spec, mk) in enumerate((("M2", "o"), ("FE", "s"))):
            s = E[(E.kind == "chance") & (E.spec == spec) & (E.scope == "all")
                  & E.estimand.str.startswith(who)].set_index("estimand")
            s = s.loc[[f"{who}_{w}" for w in ("0-2", "2-5", "5-10")]]
            xx = x + (k * 2 + j - 1.5) * 0.12
            ax.errorbar(xx, s.rate_ratio, yerr=[s.rate_ratio - s.rr_lo, s.rr_hi - s.rate_ratio], fmt=mk, color=c,
                        mfc=c if spec == "M2" else SURF, ms=6, capsize=3, lw=1.2,
                        label=f"{name}, {'main' if spec == 'M2' else 'fixed effects'}")
    ax.axhline(1, color=INK2, lw=0.8, ls="--")
    ax.set_xticks(x, ["0-2 min", "2-5 min", "5-10 min"])
    ax.set_xlabel("Time since the goal (dead time before the kick-off removed)")
    ax.set_ylabel("Chance rate vs the new score situation")
    ax.set_title("After a goal (pre-registered estimands)")
    ax.legend(frameon=False, fontsize=7.6, loc="lower right")
    ax.grid(axis="y", color=LINE, lw=0.5)
    ax = axes[1]
    x = np.arange(4)
    for k, (side, c, name) in enumerate((("scorer", BLUE, "Scoring team"), ("conceder", ORANGE, "Conceding team"))):
        s = OE[OE.side == side].set_index("window").loc[WLAB]
        xx = x + (k - 0.5) * 0.2
        ax.errorbar(xx, s.o_over_e, yerr=[s.o_over_e - s.ci_lo, s.ci_hi - s.o_over_e], fmt="o", color=c, ms=6,
                    capsize=3, lw=1.3, label=name)
    ax.axhline(1, color=INK2, lw=0.8, ls="--")
    ax.set_xticks(x, [w + " min" for w in WLAB])
    ax.set_xlabel("Time since the goal")
    ax.set_ylabel("Observed / expected chances")
    ax.set_title("Second half of the season, model fitted on the first")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.grid(axis="y", color=LINE, lw=0.5)
    ax.text(0, -0.24, "Expected: model with score, clock, strength and chance history but no goal terms.",
            transform=ax.transAxes, fontsize=7.8, color=MUTED)
    save(fig, "fig4_after_goals.png")


# ----------------------------------------------------------------------------- fig 5: out of sample
def fig_oos():
    D = pd.read_csv(RES / "explore_oos_decomposition.csv")
    OS = pd.read_csv(RES / "out_of_sample.csv").set_index("comparison")
    K = pd.read_csv(RES / "time_rescaling_km.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), gridspec_kw=dict(wspace=0.35, width_ratios=[1.2, 1]))
    ax = axes[0]
    labels = ["Chance counts earlier\nin the match", "Chances 0-2 min ago\n(restart dip)",
              "Chances 2-20 min ago\n(momentum)", "Goals 0-20 min ago"]
    g = OS.loc["M2 vs M1"]
    vals = list(D.ll_gain_total) + [g.ll_gain_total]
    lo = list(D.ci_lo) + [g.ci_lo]
    hi = list(D.ci_hi) + [g.ci_hi]
    y = np.arange(4)[::-1]
    ax.barh(y, vals, color=[MUTED, ORANGE, BLUE, AQUA], height=0.55)
    ax.errorbar(vals, y, xerr=[np.array(vals) - lo, np.array(hi) - vals], fmt="none", ecolor=INK, capsize=3, lw=1)
    for yi, v, h in zip(y, vals, hi, strict=True):
        ax.text(h + 4, yi, f"{v:+.0f}", va="center", fontsize=9, color=INK2)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Gain in log-likelihood on 760 unseen matches")
    ax.set_title("What improves predictions of the second half?")
    ax.text(0, -0.24, "Terms added in order, each fitted on the first half only. Bars 1-3 exploratory split of the "
            "pre-registered chance-history gain; 95% CIs by resampling matches.", transform=ax.transAxes,
            fontsize=7.6, color=MUTED)
    ax.grid(axis="x", color=LINE, lw=0.5)
    ax = axes[1]
    ax.plot(K.t, K.exp, color=INK, lw=1.2, ls="--", label="Perfect model: Exp(1)")
    ax.plot(K.t, K.km_M0, color=MUTED, lw=2, label="Baseline only")
    ax.plot(K.t, K.km_M2, color=BLUE, lw=2, label="Main model")
    ax.set_yscale("log")
    ax.set_ylim(0.015, 1.05)
    ax.set_xlabel("Rescaled time between chances")
    ax.set_ylabel("Share of gaps longer than this")
    ax.set_title("Goodness of fit on unseen matches")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.grid(color=LINE, lw=0.5)
    save(fig, "fig5_out_of_sample.png")


# ----------------------------------------------------------------------------- fig 6: robustness
def fig_robust():
    C = pd.read_csv(RES / "explore_momentum_contrasts.csv")
    kinds = [("chance", "Chances (shot-ending possessions)"), ("open_chance", "Open-play chances"),
             ("box_entry", "Box entries"), ("shot", "Individual shots (positive control)")]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), sharey=True, gridspec_kw=dict(wspace=0.08))
    for ax, w, title in ((axes[0], "0-2", "0-2 min after the team's own chance"),
                         (axes[1], "2-5", "2-5 min after the team's own chance")):
        y = np.arange(len(kinds))[::-1]
        for k, (spec, c, name) in enumerate((("M2", BLUE, "Main model"), ("FE", ORANGE, "Fixed effects"))):
            s = C[(C.spec == spec) & (C.side == "own") & (C.window == w)].set_index("kind").loc[[k_ for k_, _ in kinds]]
            yy = y + (0.5 - k) * 0.22
            ax.errorbar(s.rate_ratio, yy, xerr=[s.rate_ratio - s.rr_lo, s.rr_hi - s.rate_ratio], fmt="o", color=c,
                        ms=6, capsize=3, lw=1.3, label=name)
        ax.axvline(1, color=INK2, lw=0.8, ls="--")
        ax.set_yticks(y, [lab for _, lab in kinds])
        ax.set_xlabel("Rate ratio vs the team's level earlier in the match")
        ax.set_title(title)
        ax.grid(axis="x", color=LINE, lw=0.5)
        explore_tag(ax)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    save(fig, "fig6_robustness.png")


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    fig_effects()
    fig_hazard()
    fig_profile()
    fig_goals()
    fig_oos()
    fig_robust()
