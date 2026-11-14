"""Extra cross-validation checks (training set only):
1. model c) with vs without the position dummies, fold by fold (same folds);
2. calibration of model c) out-of-fold: mean predicted probability per decile
   vs observed rate.

    python scripts/cv_checks.py   # reports/cv_checks.md
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from scouting import config
from scouting.model import TARGET, training_population

sys.path.insert(0, str(config.PROJECT_ROOT / "scripts"))
from train_cv import ALL, NO_POSITION, md, oof  # noqa: E402


def calibration_table(y, pred, n_bins=10) -> pd.DataFrame:
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(pred)})
    d["decil"] = pd.qcut(d["p"].rank(method="first"), n_bins, labels=range(1, n_bins + 1))
    t = d.groupby("decil", observed=True).agg(n=("y", "size"), prob_media=("p", "mean"),
                                               taxa_real=("y", "mean"), positivos=("y", "sum"))
    t["prob_media"] = t["prob_media"].round(3)
    t["taxa_real"] = t["taxa_real"].round(3)
    return t.reset_index()


def main() -> int:
    pop = training_population(pd.read_parquet(config.PROCESSED_DIR / "features.parquet"))
    X, y, g = pop, pop[TARGET], pop["player_source_id"]
    p_all, pr_all, c_all = oof(None, X, y, g, ALL)
    p_nop, pr_nop, c_nop = oof(None, X, y, g, NO_POSITION)
    folds = pd.DataFrame({"fold": range(1, 6),
                          "PR-AUC com posição": np.round(pr_all, 3), "C com": c_all,
                          "PR-AUC sem posição": np.round(pr_nop, 3), "C sem": c_nop})
    folds["diferença"] = (folds["PR-AUC com posição"] - folds["PR-AUC sem posição"]).round(3)
    wins = int((folds["diferença"] > 0).sum())
    lines = ["# Verificações extra da validação cruzada", "",
             "## Modelo c) com vs sem dummies de posição (mesmos folds)", "", md(folds), "",
             f"A versão com posição ganha em {wins} de 5 folds.", "",
             "## Calibração do modelo c) (previsões fora-do-fold, decis da probabilidade)", "",
             md(calibration_table(y, p_all)), ""]
    out = config.PROJECT_ROOT / "reports" / "cv_checks.md"
    out.write_text(chr(10).join(lines), encoding="utf-8")
    print(chr(10).join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
