"""Final model c) trained on every labelled season, applied to an unlabelled
season (2025/26) to produce a scouting ranking.

    python scripts/rank_season.py                 # reports/ranking_2025_26.md
    python scripts/rank_season.py --destinations  # + 2026/27 destinations of the top 30 (descriptive)

Training: non-B player-seasons of 2021/22-2023/24 with subiu_individual
observed (test season closed; no player is held out now). C chosen by the same
inner GroupKFold procedure. The ranking shows the percentile of the score
within the ranked season (non-B), not the probability.

The 2026/27 destinations section is descriptive only: it is computed after
the ranking and never feeds back into the model.
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from scouting import config
from scouting.http import PageError
from scouting.model import LABELLED_SEASONS, TARGET
from scouting.parsing.zerozero import in_main_squad, parse_edition_teams, parse_team_season
from scouting.report import write_main
from scouting.sources import zerozero as zz
from scouting.target import load_opta, opta_rating

sys.path.insert(0, str(config.PROJECT_ROOT / "scripts"))
from train_cv import ALL, choose_C, logit, md  # noqa: E402

LABELLED = LABELLED_SEASONS
RANK_SEASON = "2025/26"
NEXT_SEASON = "2026/27"
NEXT_SEASON_ID = 156  # zerozero epoca_id of 2026/27
PT_EDITIONS_NEXT = {1: 218294, 2: 218295}  # I Liga and Liga 2 2026/27
TOP_N = 30
POSITIONS = ["GK", "DF", "MF", "FW"]
GROUPS = ["subiu individualmente (outro clube, nível >= Liga 2)",
          "subiu com a própria equipa", "não subiu (ainda)"]
log = logging.getLogger("rank")


def train_final(df: pd.DataFrame):
    train = df[df["season"].isin(LABELLED) & ~df["is_b_team"].astype(bool)
               & df["target_observed_individual"].astype(bool)].copy().reset_index(drop=True)
    train[TARGET] = train[TARGET].astype(int)
    C = choose_C(train, train[TARGET], train["player_source_id"], ALL)
    return logit(ALL, C).fit(train, train[TARGET]), C, train


def ranked(df: pd.DataFrame, model) -> pd.DataFrame:
    ps = pd.read_parquet(config.PROCESSED_DIR / "player_season_minutes.parquet")
    ps = ps[ps["season"] == RANK_SEASON]
    r = df[(df["season"] == RANK_SEASON) & ~df["is_b_team"].astype(bool)].copy().reset_index(drop=True)
    r["score"] = model.predict_proba(r)[:, 1]
    r["percentil"] = (100 * r["score"].rank(pct=True, method="max")).round(1)
    names = ps.groupby("player_source_id")["player_name"].first()
    # Main club: most minutes in the season (players who changed club list both).
    main = ps.sort_values("minutes", ascending=False).drop_duplicates("player_source_id")
    r["jogador"] = r["player_source_id"].map(names)
    r["clube"] = r["player_source_id"].map(ps.groupby("player_source_id")["team"].agg(" / ".join))
    r["clube_principal"] = r["player_source_id"].map(main.set_index("player_source_id")["team"])
    r["amarelos"] = r["player_source_id"].map(ps.groupby("player_source_id")["yellow_cards"].sum())
    return r.sort_values("score", ascending=False).reset_index(drop=True)


def view(r: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "jogador": r["jogador"], "idade": r["age"].round(1), "clube": r["clube"],
        "posição": r["position_model"], "% minutos": r["pct_team_minutes"],
        "percentil": r["percentil"]})


def stats(r: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "equipa": r["clube_principal"], "jogador": r["jogador"], "idade": r["age"].round(1),
        "posição": r["position_model"], "jogos": r["matches_played"], "titular": r["starts"],
        "minutos": r["minutes"], "% minutos": r["pct_team_minutes"], "golos": r["goals"],
        "golos/90": r["goals_per90"].astype(float).round(2), "amarelos": r["amarelos"],
        "dif. golos/90 rel.": r["rel_gd90"].astype(float).round(2), "percentil": r["percentil"]})


def destinations(top: pd.DataFrame, fetch: bool) -> pd.DataFrame:
    """Club and league of each top player in 2026/27 (descriptive)."""
    career = pd.read_parquet(config.INTERIM_DIR / "player_career.parquet")
    nxt = career[(career["season"] == NEXT_SEASON) & career["player_source_id"].isin(top["player_source_id"])]
    nxt = nxt[~nxt["squad"].fillna("").str.contains(r"(?:Jun\.|S1\d|S2\d)")]
    nxt = nxt.sort_values("matches", ascending=False, na_position="last").drop_duplicates("player_source_id")
    opta = load_opta()
    pt = {}
    client = zz.make_client() if fetch else None
    try:
        for tier, eid in PT_EDITIONS_NEXT.items():
            path = config.RAW_DIR / "zerozero" / "edition" / f"{eid}.html"
            if not path.exists() and client is not None:
                zz.fetch_edition(client, eid)
            if path.exists():
                for t in parse_edition_teams(path.read_text(encoding="utf-8")):
                    pt[t["team_id"]] = "I Liga" if tier == 1 else "Liga 2"
        rows = []
        for r in nxt.itertuples():
            league, rating, counts = None, None, False
            if r.country == "PT":
                league = pt.get(r.club_source_id, "abaixo da Liga 2 (PT)") if pt else "PT (divisão por confirmar)"
                counts = r.club_source_id in pt
            else:
                # Calendar-year countries: the 2026 season has epoca_id 155, not 156.
                sid = int(r.season_source_id) if pd.notna(r.season_source_id) else NEXT_SEASON_ID
                path = config.RAW_DIR / "zerozero" / "team" / f"{int(r.club_source_id)}_{sid}.html"
                if not path.exists() and client is not None:
                    try:
                        zz.fetch_team_season(client, int(r.club_source_id), sid)
                    except PageError as exc:
                        log.warning("Team page unavailable for %s: %s", r.club, exc)
                if path.exists():
                    page = parse_team_season(path.read_text(encoding="utf-8"))
                    league = page["league"]["competition"] if page["league"] else None
                rating = opta_rating(r.country, league, r.club_source_id, opta)
                if rating is None and isinstance(league, str):
                    # Descriptive-only ratings (config [descriptive]); never used by the target.
                    desc = opta.get("descriptive", {})
                    rating = desc.get("ratings", {}).get(desc.get("map", {}).get(f"{r.country}|{league}"))
                counts = rating is not None and rating >= opta["meta"]["threshold"]
                # Only the main-team squad counts (not reserves / U23), as in destinations_next.py.
                if counts and not in_main_squad(path.read_text(encoding="utf-8"), r.player_source_id):
                    counts, league = False, f"{league} (fora do plantel principal)"
            rows.append({"player_source_id": r.player_source_id, "club_source_id": r.club_source_id,
                         "clube_2026_27": r.club, "país": r.country,
                         "liga_2026_27": league or "—", "rating_opta": rating, "jogos": r.matches,
                         "nível >= Liga 2": "sim" if counts else "não"})
        if client is not None:
            log.info("Network requests made: %d", client.requests_made)
    finally:
        if client is not None:
            client.close()
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--destinations", action="store_true", help="add the 2026/27 destinations of the top 30")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    df = pd.read_parquet(config.PROCESSED_DIR / "features.parquet")
    model, C, train = train_final(df)
    r = ranked(df, model)
    top = r.head(TOP_N)
    lines = [f"# Ranking Liga 3 {RANK_SEASON}", "",
             f"Modelo final c) (regressão logística L2, C = {C}, escolhido no treino pelo mesmo "
             f"procedimento), treinado em {', '.join(LABELLED)} (equipas não-B, `{TARGET}` observado; "
             f"n = {len(train)}, positivos = {int(train[TARGET].sum())}). Aplicado a {len(r)} "
             f"jogador-épocas de equipas não-B de {RANK_SEASON}. **Percentil** = posição do score "
             f"dentro de {RANK_SEASON} (100 = score mais alto); não é uma probabilidade.", "",
             f"## Top {TOP_N} geral", "", md(view(top).assign(**{"#": range(1, TOP_N + 1)})
                                             [["#", "jogador", "idade", "clube", "posição", "% minutos", "percentil"]]), ""]
    lines += ["## Top 5 por posição", ""]
    for pos in POSITIONS:
        lines += [f"### {pos}", "", md(view(r[r["position_model"] == pos].head(5))), ""]
    best = r.drop_duplicates("clube_principal").sort_values("clube_principal")
    lines += ["## Jogador com score mais alto de cada equipa", "",
              md(stats(best)), "",
              "Jogos / titular / minutos / golos / amarelos: toda a época na Liga 3 (inclui a 2.ª fase). "
              "Golos/90: vazio abaixo de 450 minutos. Dif. golos/90 rel.: diferença de golos por 90 da "
              "equipa com o jogador em campo menos a da equipa no total (positivo = equipa melhor com ele).", ""]
    if args.destinations:
        d = destinations(top, fetch=True)
        dd = top[["player_source_id", "jogador", "clube", "percentil"]].merge(d, on="player_source_id", how="left")
        dd["clube_2026_27"] = dd["clube_2026_27"].fillna("sem registo em 2026/27")
        ps = pd.read_parquet(config.PROCESSED_DIR / "player_season_minutes.parquet")
        l3 = ps[ps["season"] == RANK_SEASON].groupby("player_source_id")["team_id"].agg(set)
        counts = dd["nível >= Liga 2"].eq("sim")
        own = [c in l3.get(p, set()) for p, c in zip(dd["player_source_id"], dd["club_source_id"])]
        dd["grupo"] = [GROUPS[0] if k and not o else GROUPS[1] if k else GROUPS[2] for k, o in zip(counts, own)]
        dd["rating_opta"] = dd["rating_opta"].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
        n = dd["grupo"].value_counts()
        lines += [f"## Onde estão os {TOP_N} do top em {NEXT_SEASON} (só descritivo)", "",
                  "Carreira em cache (perfis de novembro de 2026; época em curso). Nível >= Liga 2 = clube "
                  "da I Liga ou da Liga 2, ou liga estrangeira com rating Opta >= Liga Portugal 2. Não entra no modelo "
                  "nem o altera. Ratings de ligas fora do snapshot principal "
                  "estão em `[descriptive]` no config e só são usados nesta secção.", "",
                  "| grupo | jogadores |", "| --- | --- |"]
        lines += [f"| {g} | {int(n.get(g, 0))} |" for g in GROUPS] + [""]
        cols = ["jogador", "clube", "percentil", "clube_2026_27", "país", "liga_2026_27", "rating_opta", "jogos"]
        for g in GROUPS:
            sub = dd[dd["grupo"] == g]
            lines += [f"### {g} ({len(sub)})", "", md(sub[cols]) if len(sub) else "Nenhum.", ""]
    # Keeps the "Destinos 2026/27" section written by destinations_next.py.
    write_main(config.PROJECT_ROOT / "reports" / "ranking_2025_26.md", chr(10).join(lines))
    print(chr(10).join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
