"""Cross-validation of the main model on the training set only (the test
season 2023/24 is not touched). GroupKFold by player, 5 folds.

    python scripts/feature_signal.py   # writes data/processed/features.parquet
    python scripts/train_cv.py         # reports/cv_report.md

Models: a) % of minutes only; b) age + age^2 + % of minutes; c) L2 logistic
regression with all final features, C chosen by an inner GroupKFold inside
each outer training fold; d) LightGBM with the same features (fixed, small
trees, no tuning).
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scouting import config
from scouting.target import load_opta, threshold_suffix
from scouting.model import (
    TARGET,
    with_target_variant,
    FoldFeatures,
    group_folds,
    metrics,
    precision_at_k,
    training_population,
)

warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

C_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
ALL = None  # every column FoldFeatures produces
NO_POSITION = ["age_c", "age_c2", "pct_team_minutes", "team_points_per_game_phase1", "rel_gd90",
               "goals_pct_in_position", "low_minutes", "prev_profissional_ou_estrangeiro",
               "prev_liga3_ou_pt_senior", "prev_formacao_ou_sem_registo"]


def logit(columns, C=1.0):
    return make_pipeline(FoldFeatures(columns), StandardScaler(),
                         LogisticRegression(C=C, max_iter=5000))


def lgbm():
    return make_pipeline(FoldFeatures(ALL), LGBMClassifier(
        n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=20,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=42, verbose=-1))


def choose_C(X, y, groups, columns):
    """C with the best mean PR-AUC over an inner GroupKFold."""
    best, best_score = None, -1.0
    for C in C_GRID:
        scores = []
        for tr, va in group_folds(groups).split(X, y, groups):
            m = logit(columns, C).fit(X.iloc[tr], y.iloc[tr])
            scores.append(average_precision_score(y.iloc[va], m.predict_proba(X.iloc[va])[:, 1]))
        if np.mean(scores) > best_score:
            best, best_score = C, float(np.mean(scores))
    return best


def oof(make_model, X, y, groups, tune_columns="none"):
    """Out-of-fold predictions; per-fold PR-AUC; chosen C per fold."""
    pred = np.zeros(len(X))
    fold_pr, chosen = [], []
    for tr, va in group_folds(groups).split(X, y, groups):
        if tune_columns != "none":
            C = choose_C(X.iloc[tr], y.iloc[tr], groups.iloc[tr], tune_columns)
            chosen.append(C)
            model = logit(tune_columns, C)
        else:
            model = make_model()
        model.fit(X.iloc[tr], y.iloc[tr])
        pred[va] = model.predict_proba(X.iloc[va])[:, 1]
        fold_pr.append(average_precision_score(y.iloc[va], pred[va]))
    return pred, fold_pr, chosen


def md(t: pd.DataFrame) -> str:
    rows = [list(map(str, t.columns)), ["---"] * len(t.columns)] + t.astype(str).values.tolist()
    return chr(10).join("| " + " | ".join(r) + " |" for r in rows)


def summary_row(name, y, pred, fold_pr, chosen=None):
    m = metrics(y, pred, 50)
    return {"modelo": name, "n": len(y), "positivos": int(y.sum()),
            "PR-AUC": round(m["pr_auc"], 3),
            "precisão top 20": round(precision_at_k(y, pred, 20), 3),
            "PR-AUC por fold (média ± dp)": f"{np.mean(fold_pr):.3f} ± {np.std(fold_pr):.3f}",
            "ROC-AUC": round(m["roc_auc"], 3), "precisão top 50": round(m["prec_top50"], 3),
            "C escolhido por fold": ", ".join(str(c) for c in chosen) if chosen else ""}


def run(df: pd.DataFrame):
    X, y, groups = df, df[TARGET], df["player_source_id"]
    preds, rows = {}, []
    specs = [
        ("a) % de minutos", lambda: logit(["pct_team_minutes"], C=1e6), "none"),
        ("b) idade + idade² + % de minutos", lambda: logit(["age_c", "age_c2", "pct_team_minutes"], C=1e6), "none"),
        ("c) logística L2, todas as features", None, ALL),
        ("d) LightGBM, mesmas features", lgbm, "none"),
    ]
    for name, mk, tune in specs:
        pred, fold_pr, chosen = oof(mk, X, y, groups, tune if tune is not None else ALL)
        preds[name] = pred
        rows.append(summary_row(name, y, pred, fold_pr, chosen))
    return preds, pd.DataFrame(rows)


def per_position(df, preds, k=20):
    rows = []
    for name, pred in preds.items():
        for pos in ["GK", "DF", "MF", "FW", "unknown"]:
            mask = (df["position_model"] == pos).to_numpy()
            y = df.loc[mask, TARGET].to_numpy()
            if y.sum() == 0:
                rows.append({"modelo": name[:2], "posição": pos, "n": int(mask.sum()), "positivos": 0,
                             "taxa base": 0.0, "PR-AUC": "—", "precisão top 20": "—"})
                continue
            rows.append({"modelo": name[:2], "posição": pos, "n": int(mask.sum()), "positivos": int(y.sum()),
                         "taxa base": round(y.mean(), 3),
                         "PR-AUC": round(average_precision_score(y, pred[mask]), 3),
                         "precisão top 20": round(precision_at_k(y, pred[mask], min(k, mask.sum())), 3)})
    return pd.DataFrame(rows)


def coefficients(df):
    X, y, groups = df, df[TARGET], df["player_source_id"]
    C = choose_C(X, y, groups, ALL)
    m = logit(ALL, C).fit(X, y)
    names = m[0].transform(X.head(1)).columns
    coef = pd.DataFrame({"feature": names, "coef_padronizado": m[-1].coef_[0].round(3)})
    coef["odds ratio por 1 dp"] = np.exp(coef["coef_padronizado"]).round(2)
    return C, coef.sort_values("coef_padronizado", key=abs, ascending=False)


def main() -> int:
    df = pd.read_parquet(config.PROCESSED_DIR / "features.parquet")
    pop = training_population(df)
    lines = ["# Validação cruzada (só treino)", "",
             f"População: equipas não-B, 2021/22 + 2022/23, sem jogadores que aparecem em 2023/24, "
             f"alvo `{TARGET}` observado. n = {len(pop)}, positivos = {int(pop[TARGET].sum())} "
             f"({100 * pop[TARGET].mean():.1f}%). GroupKFold por jogador, 5 folds. "
             "PR-AUC, ROC-AUC e precisão no top 50 calculadas sobre as previsões fora-do-fold juntas.", ""]
    preds, table = run(pop)
    lines += ["## Modelos", "", md(table), ""]
    lines += ["## Por posição (previsões fora-do-fold)", "", md(per_position(pop, preds)), ""]
    C, coef = coefficients(pop)
    lines += [f"## Coeficientes padronizados do modelo c) (C = {C}, ajustado em todo o treino)", "",
              md(coef), ""]

    # Sensitivity analyses (owner, 2026-11-14): the config's sensitivity thresholds,
    # and without the player-seasons that have no future data.
    for label, sfx, drop in (*[(f"limiar Opta {t:.2f}".replace(".", ","), "_" + threshold_suffix(t), False)
                               for t in load_opta()["meta"]["sensitivity_thresholds"]],
                             ("sem os jogador-épocas sem dados futuros", "", True)):
        sens = training_population(with_target_variant(df, sfx))
        if drop:  # after the test-player exclusion, so no test player leaks in
            sens = sens[sens["has_future_data"].astype(bool)].reset_index(drop=True)
        _, t2 = run(sens)
        lines += [f"## Sensibilidade: {label}", "", md(t2), ""]

    # Sensitivity: c) without the position dummies (2021/22 starters take the
    # profile line because those match sheets have no tactical pitch).
    pred, fold_pr, chosen = oof(None, pop, pop[TARGET], pop["player_source_id"], NO_POSITION)
    t3 = pd.DataFrame([summary_row("c) sem dummies de posição", pop[TARGET], pred, fold_pr, chosen)])
    lines += ["## Sensibilidade: modelo c) sem dummies de posição", "", md(t3), ""]

    out = config.PROJECT_ROOT / "reports" / "cv_report.md"
    out.write_text(chr(10).join(lines), encoding="utf-8")
    print(chr(10).join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
