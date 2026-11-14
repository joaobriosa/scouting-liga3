"""Evaluation on the test season (2023/24). The test is now CLOSED: this was
the third and last reading (owner, 2026-11-14). The script refuses to run if
reports/test_report.md already exists.

Final model: c) L2 logistic regression, C chosen by the same inner GroupKFold
procedure on the whole training set, fitted on the whole training set. a) and
b) are fitted the same way for comparison. Sensitivity analyses (Opta
sensitivity thresholds, without player-seasons with no future data) refit the
models on the training labels of that variant.

    python scripts/evaluate_test.py
"""

from __future__ import annotations

import sys

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from scouting import config
from scouting.model import TARGET, TEST_SEASON, precision_at_k, training_population, with_target_variant
from scouting.target import league_minutes, load_opta, rows_needing_minutes, threshold_suffix

sys.path.insert(0, str(config.PROJECT_ROOT / "scripts"))
from cv_checks import calibration_table  # noqa: E402
from train_cv import ALL, choose_C, logit, md  # noqa: E402

MODELS = {
    "a) % de minutos": (["pct_team_minutes"], 1e6),
    "b) idade + idade² + % de minutos": (["age_c", "age_c2", "pct_team_minutes"], 1e6),
    "c) logística L2": (ALL, None),  # C chosen on the training set
}
REPORT = config.PROJECT_ROOT / "reports" / "test_report.md"
PREVIOUS = [  # (label, predictions file, C used)
    ("leitura 1: grupos A/B/C, épocas em ano civil com erro, minutos sem filtro de clube",
     config.PROCESSED_DIR / "test_predictions_leitura1.parquet", 3.0),
    ("leitura 2: grupos A/B/C, ano civil corrigido, minutos sem filtro de clube",
     config.PROCESSED_DIR / "test_predictions_leitura2.parquet", 1.0),
]
READINGS_NOTE = """## Três leituras do teste

O teste foi lido três vezes, sempre com o **mesmo modelo e o mesmo procedimento**
(c: regressão logística L2, mesmas features, C escolhido só no treino pelo mesmo
GroupKFold interno). Nenhuma feature, hiperparâmetro ou modelo foi alterado com
base no teste. As leituras 2 e 3 devem-se a **erros nas etiquetas** (e, na 3, à
nova definição de destino aprovada pelo dono do projeto), não a ajustes do modelo.

1. **Leitura 1** (`reports/test_report_leitura1.md`): etiquetas com grupos A/B/C.
   Erro encontrado ao rever o top 20: as linhas de carreira com época em ano civil
   ("2025": Brasil, EUA, Escandinávia, ...) não eram associadas a s+1/s+2.
2. **Leitura 2** (`reports/test_report_leitura2.md`): épocas em ano civil corrigidas
   (Y -> Y/Y+1; linhas de outros países com rótulo de ano -> época do `epoca_id`).
   Erro encontrado depois, ao analisar o grupo B: a página de jogos do zerozero
   ignora o filtro de clube, por isso os minutos de liga de cada destino somavam
   todos os clubes da época (incluindo a Liga 3) e ligas sub-23 e taças.
3. **Leitura 3** (este relatório, a última; o teste fica fechado): minutos de liga
   por clube a partir da lista de jogos, sem Liga Next Gen, taças e competições
   continentais; e a regra de destino passa a ser o Opta Power Rankings (liga com
   rating médio >= Liga Portugal 2; snapshot de 13/11/2026), em vez dos
   grupos A/B/C. As três correções entraram juntas.
"""


def test_frame(df: pd.DataFrame) -> pd.DataFrame:
    """2023/24, non-B; all rows (each target variant picks its own observed rows)."""
    return df[(df["season"] == TEST_SEASON) & ~df["is_b_team"].astype(bool)].copy().reset_index(drop=True)


def labels(t: pd.DataFrame) -> pd.Series:
    return t[TARGET].astype(float).where(t["target_observed_individual"].astype(bool))


def metric_row(name, y, p):
    return {"modelo": name, "n": len(y), "positivos": int(y.sum()),
            "PR-AUC": round(average_precision_score(y, p), 3),
            "ROC-AUC": round(roc_auc_score(y, p), 3),
            "precisão top 20": round(precision_at_k(y, p, 20), 3),
            "precisão top 50": round(precision_at_k(y, p, 50), 3)}


def per_position(t, ycol, pcol):
    rows = []
    for pos in ["GK", "DF", "MF", "FW", "unknown"]:
        s = t[(t["position_model"] == pos) & t[ycol].notna()]
        y, p = s[ycol].to_numpy(), s[pcol].to_numpy()
        row = {"posição": pos, "n": len(s), "positivos": int(y.sum()), "taxa base": round(y.mean(), 3)}
        if 0 < y.sum() < len(y):
            row |= {"PR-AUC": round(average_precision_score(y, p), 3),
                    "ROC-AUC": round(roc_auc_score(y, p), 3),
                    "precisão top 20": round(precision_at_k(y, p, min(20, len(y))), 3)}
        rows.append(row)
    return pd.DataFrame(rows).fillna("—")


def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    final_C = choose_C(train, train[TARGET], train["player_source_id"], ALL)
    out = test.copy()
    for name, (cols, C) in MODELS.items():
        model = logit(cols, final_C if C is None else C).fit(train, train[TARGET])
        out[f"p_{name[:1]}"] = model.predict_proba(test)[:, 1]
    return out, final_C


def destinations(t: pd.DataFrame, threshold: float) -> pd.Series:
    """Where each player went: the counting club (Liga 2 / I Liga / foreign
    league with Opta rating >= threshold) with most league minutes at a club
    other than his Liga 3 team (s+1/s+2); otherwise his s+1 club."""
    dest = pd.read_parquet(config.INTERIM_DIR / "destination_rows.parquet")
    dest = dest[dest["season"] == TEST_SEASON]
    need = rows_needing_minutes(dest)
    need = need[need["dest"].eq("PT") | need["opta_rating"].ge(threshold)].copy()
    need["mins"] = [league_minutes(r) or 0 for r in need.itertuples()]
    ps = pd.read_parquet(config.PROCESSED_DIR / "player_season_minutes.parquet")
    l3 = ps[ps["season"] == TEST_SEASON].groupby("player_source_id")["team_id"].agg(set)
    need = need[[r.club_source_id not in l3.get(r.player_source_id, set()) for r in need.itertuples()]]
    need["label"] = [
        f"{r.club} ({r.future}, "
        + ("Liga 2" if r.pt_tier == 2 else "I Liga" if r.pt_tier == 1 else f"{r.country} {r.league}, Opta {r.opta_rating:.1f}")
        + f", {r.mins} min)" for r in need.itertuples()]
    best = need.sort_values("mins", ascending=False).drop_duplicates("player_source_id")
    lab = best.set_index("player_source_id")["label"]
    s1 = dest[dest["k"] == 1].sort_values("matches", ascending=False).drop_duplicates("player_source_id")
    s1_lab = s1.set_index("player_source_id").apply(lambda r: f"s+1: {r['club']} ({r['country']})", axis=1)
    return t["player_source_id"].map(lab).fillna(t["player_source_id"].map(s1_lab)).fillna("sem dados futuros")


def main() -> int:
    if REPORT.exists() and "--force" not in sys.argv:
        print(f"{REPORT} already exists: the test set is closed.")
        return 1
    opta = load_opta()
    df = pd.read_parquet(config.PROCESSED_DIR / "features.parquet")
    target = pd.read_parquet(config.PROCESSED_DIR / "target.parquet")
    train = training_population(df)
    test, final_C = fit_predict(train, test_frame(df))
    test["y_main"] = labels(test)
    main_t = test[test["y_main"].notna()]

    rows = []
    for n in MODELS:
        for label, path, c in PREVIOUS:
            if path.exists():
                r = pd.read_parquet(path)
                r = r[r["y_main"].notna()]
                rows.append({"leitura": f"{label} (C = {c})"} | metric_row(n, r["y_main"], r[f"p_{n[:1]}"]))
        rows.append({"leitura": f"leitura 3: Opta, ano civil corrigido, minutos por clube (C = {final_C})"}
                    | metric_row(n, main_t["y_main"], main_t[f"p_{n[:1]}"]))

    lines = ["# Avaliação no teste (2023/24) — leitura 3 (final)", "", READINGS_NOTE, "",
             "## As três leituras lado a lado", "", md(pd.DataFrame(rows)), "",
             f"Modelo c): C = {final_C} (escolhido no treino pelo mesmo procedimento). "
             f"Treino: n = {len(train)}, positivos = {int(train[TARGET].sum())}. Teste: equipas não-B de "
             f"2023/24 com `{TARGET}` observado, n = {len(main_t)}, positivos = {int(main_t['y_main'].sum())} "
             f"({100 * main_t['y_main'].mean():.1f}%). Limiar Opta = {opta['meta']['threshold']} "
             f"(snapshot {opta['meta']['snapshot_date']}).", "",
             "## Métricas", "",
             md(pd.DataFrame([metric_row(n, main_t["y_main"], main_t[f"p_{n[:1]}"]) for n in MODELS])), "",
             "## Por posição (modelo c)", "", md(per_position(test, "y_main", "p_c")), "",
             "## Calibração do modelo c) no teste (decis)", "",
             md(calibration_table(main_t["y_main"], main_t["p_c"])), ""]

    # Top 20 of the evaluated population.
    ps = pd.read_parquet(config.PROCESSED_DIR / "player_season_minutes.parquet")
    clubs = ps[ps["season"] == TEST_SEASON].groupby("player_source_id")["team"].agg(" / ".join)
    top = main_t.sort_values("p_c", ascending=False).head(20).copy()
    top["clube"] = top["player_source_id"].map(clubs)
    top["destino"] = destinations(top, opta["meta"]["threshold"]).to_numpy()
    names = target.set_index(["player_source_id", "season"])["player_name"]
    top["jogador"] = [names.get((p, TEST_SEASON)) for p in top["player_source_id"]]
    top_t = pd.DataFrame({
        "#": range(1, 21), "jogador": top["jogador"], "idade": top["age"].round(1), "clube": top["clube"],
        "posição": top["position_model"], "% minutos": top["pct_team_minutes"],
        "prob": top["p_c"].round(3), "subiu": top["y_main"].astype(int), "destino": top["destino"]})
    lines += ["## Top 20 de 2023/24 (modelo c)", "", md(top_t), ""]

    # Sensitivity analyses: models refitted on the training labels of the variant.
    for label, sfx, drop in (*[(f"limiar Opta {t:.2f}".replace(".", ","), "_" + threshold_suffix(t), False)
                               for t in load_opta()["meta"]["sensitivity_thresholds"]],
                             ("sem os jogador-épocas sem dados futuros", "", True)):
        v = with_target_variant(df, sfx)
        tr = training_population(v)
        te = test_frame(v)
        if drop:
            tr = tr[tr["has_future_data"].astype(bool)].reset_index(drop=True)
            te = te[te["has_future_data"].astype(bool)].reset_index(drop=True)
        te, c_v = fit_predict(tr, te)
        te["y"] = labels(te)
        te = te[te["y"].notna()]
        lines += [f"## Sensibilidade: {label} (C = {c_v}; treino n = {len(tr)}, positivos = {int(tr[TARGET].sum())})",
                  "", md(pd.DataFrame([metric_row(n, te["y"], te[f"p_{n[:1]}"]) for n in MODELS])), ""]

    test.to_parquet(config.PROCESSED_DIR / "test_predictions.parquet", index=False)
    REPORT.write_text(chr(10).join(lines), encoding="utf-8")
    print(chr(10).join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
