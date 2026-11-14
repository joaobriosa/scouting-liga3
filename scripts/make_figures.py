"""Four PNG charts (reports/figures/), English labels.

    python scripts/make_figures.py

a) rise rate by quartile of % of team minutes;
b) predicted probability by age for a player with 80% of the minutes
   (final model c), averaged over the training players: the age effect at
   equal minutes);
c) rise rate: B teams vs. the rest;
d) precision in the top 20 of the test (2023/24): c) vs. age + minutes vs.
   minutes only (from data/processed/test_predictions.parquet, reading 3).

Rise = subiu_individual (>= 450 league minutes in Liga 2 / I Liga or a league
with Opta rating >= Liga Portugal 2 at another club, in s+1/s+2), labelled seasons
2021/22-2023/24, rows with the label observed.
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scouting import config  # noqa: E402
from scouting.model import TARGET, precision_at_k  # noqa: E402

sys.path.insert(0, str(config.PROJECT_ROOT / "scripts"))
from rank_season import LABELLED, train_final  # noqa: E402

OUT = config.PROJECT_ROOT / "reports" / "figures"
SOURCE = "Data: zerozero.pt; league strength: Opta Power Rankings"
FOCUS, MUTED = "#2a78d6", "#b4b2ab"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 20, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
})


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def figure(title: str, subtitle: str):
    fig, ax = plt.subplots(figsize=(10.8, 10.8), dpi=100)
    fig.subplots_adjust(left=0.11, right=0.95, top=0.77, bottom=0.17)
    fig.text(0.05, 0.94, title, fontsize=32, weight="bold", color=INK, va="top")
    fig.text(0.05, 0.875, subtitle, fontsize=19, color=INK2, va="top", linespacing=1.4)
    fig.text(0.05, 0.03, SOURCE, fontsize=15, color=INK2)
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=8)
    return fig, ax


def bars(ax, labels, values, colors, notes=None):
    x = np.arange(len(values))
    ax.bar(x, values, width=0.6, color=colors, edgecolor=SURFACE, linewidth=2)
    for i, v in enumerate(values):
        ax.text(i, v + 0.012, pct(v), ha="center", va="bottom", fontsize=24, weight="bold", color=INK)
        if notes:
            ax.text(i, -0.06, notes[i], ha="center", va="top", fontsize=15, color=INK2,
                    transform=ax.get_xaxis_transform())
    ax.set_xticks(x, labels)
    ax.set_ylim(0, max(values) * 1.22)
    ax.yaxis.set_major_formatter(lambda v, _: pct(v))


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, facecolor=SURFACE)
    plt.close(fig)
    print(OUT / name)


def main() -> int:
    df = pd.read_parquet(config.PROCESSED_DIR / "features.parquet")
    model, C, train = train_final(df)  # non-B, labelled seasons, label observed
    n_seasons = f"{LABELLED[0]} to {LABELLED[-1]}"

    # a) rise rate by quartile of % of minutes
    q = pd.qcut(train["pct_team_minutes"], 4)
    rate = train.groupby(q, observed=True)[TARGET].mean()
    labels = [f"Q{i + 1}\n{max(iv.left, 0):.0f}–{iv.right:.0f}%" for i, iv in enumerate(rate.index)]
    fig, ax = figure("Playing time is the first signal",
                     f"Step-up rate by quartile of share of team minutes\n"
                     f"Liga 3, non-reserve teams, {n_seasons} (n = {len(train)})")
    bars(ax, labels, rate.to_numpy(), [MUTED, MUTED, MUTED, FOCUS])
    save(fig, "a_step_up_by_minutes_quartile.png")

    # b) partial dependence on age with 80% of the minutes
    lo, hi = np.percentile(train["age"], [5, 95])
    ages = np.arange(np.floor(lo), np.ceil(hi) + 0.01, 0.25)
    probs = []
    for a in ages:
        x = train.assign(age=a, pct_team_minutes=80.0)
        probs.append(model.predict_proba(x)[:, 1].mean())
    probs = np.array(probs)
    fig, ax = figure("At equal minutes, age matters",
                     "Predicted probability of stepping up, player with 80% of the minutes\n"
                     f"Final model (logistic), averaged over {len(train)} training players")
    ax.plot(ages, probs, color=FOCUS, linewidth=3)
    for a in (20, 24, 28):
        if ages[0] <= a <= ages[-1]:
            p = probs[np.argmin(abs(ages - a))]
            ax.plot(a, p, "o", ms=11, color=FOCUS, mec=SURFACE, mew=2)
            ax.annotate(f"age {a}: {pct(p)}", (a, p), xytext=(12, 12), textcoords="offset points",
                        fontsize=20, weight="bold", color=INK)
    ax.set_xlabel("Age", fontsize=20)
    ax.set_ylim(0, probs.max() * 1.25)
    ax.yaxis.set_major_formatter(lambda v, _: pct(v))
    save(fig, "b_probability_by_age.png")

    # c) B teams vs the rest (labelled seasons, label observed)
    lab = df[df["season"].isin(LABELLED) & df["target_observed_individual"].astype(bool)].copy()
    lab[TARGET] = lab[TARGET].astype(int)
    g = lab.groupby(lab["is_b_team"].astype(bool))[TARGET].agg(["mean", "size"])
    fig, ax = figure("Reserve-team players step up 4x more",
                     f"Step-up rate (another club, >= 450 min in a league >= Liga Portugal 2)\n"
                     f"Liga 3, {n_seasons}")
    bars(ax, ["Reserve (B) teams", "Other teams"], [g.loc[True, "mean"], g.loc[False, "mean"]],
         [FOCUS, MUTED], notes=[f"n = {g.loc[True, 'size']}", f"n = {g.loc[False, 'size']}"])
    save(fig, "c_step_up_reserve_teams.png")

    # d) precision in the top 20 of the test season
    t = pd.read_parquet(config.PROCESSED_DIR / "test_predictions.parquet")
    t = t[t["y_main"].notna()]
    vals = [precision_at_k(t["y_main"], t[c], 20) for c in ("p_a", "p_b", "p_c")]
    fig, ax = figure("1 in 2 of the model's top 20 stepped up",
                     "Precision among the 20 highest-scored players, test season 2023/24\n"
                     f"(non-reserve teams, n = {len(t)}, {int(t['y_main'].sum())} stepped up)")
    bars(ax, ["Minutes\nonly", "Age +\nminutes", "Full\nmodel"], vals, [MUTED, MUTED, FOCUS])
    save(fig, "d_precision_top20_test.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
