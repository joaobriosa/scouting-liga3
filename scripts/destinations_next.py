"""Where every Liga 3 2025/26 player is playing in 2026/27 (partial,
descriptive only). Never used to change the model, the features or the
threshold.

    python scripts/destinations_next.py            # cache only
    python scripts/destinations_next.py --fetch    # + missing pages (see below)

Destination of each player: his senior 2026/27 career row with most matches
(cached profiles). Categories: I Liga / Liga 2 / Liga 3 (2026/27 edition
pages) / Campeonato de Portugal ou abaixo (other PT clubs) / foreign with Opta
rating >= Liga Portugal 2 / foreign below / no club or no record.

Requests (--fetch): the two Liga 3 2026/27 series pages, and the 2026/27 team
page of foreign clubs in countries where some league reaches the threshold.
Countries with no league rated >= the threshold in config/opta_leagues.toml
(countries_reaching_threshold, computed at run time) are "foreign below"
whatever the club, with no request.

Writes the section "Destinos 2026/27 (resultado parcial)" at the end of
reports/ranking_2025_26.md (replacing it if present; rank_season.py keeps
it when it rewrites the ranking) and reports/figures/e_step_up_by_decile_2026_27.png.
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from scouting import config
from scouting.http import PageError
from scouting.model import precision_at_k
from scouting.parsing.zerozero import in_main_squad, parse_edition_teams, parse_team_season
from scouting.sources import zerozero as zz
from scouting.report import DESTINATIONS_SECTION, write_section
from scouting.target import load_opta, opta_rating

sys.path.insert(0, str(config.PROJECT_ROOT / "scripts"))
from make_figures import FOCUS, MUTED, bars, figure, save  # noqa: E402
from rank_season import NEXT_SEASON, NEXT_SEASON_ID, PT_EDITIONS_NEXT, RANK_SEASON, ranked, train_final  # noqa: E402
from train_cv import logit, md  # noqa: E402

LIGA3_SERIES_NEXT = [219287, 219288]  # Liga 3 2026/27 Série A / Série B
CATS = ["I Liga", "Liga 2", "Liga 3", "Campeonato de Portugal ou abaixo",
        "estrangeiro, rating >= Liga 2", "estrangeiro, abaixo", "sem clube ou sem registo"]
UP = {"I Liga", "Liga 2", "estrangeiro, rating >= Liga 2"}
SECTION = DESTINATIONS_SECTION
REPORT = config.PROJECT_ROOT / "reports" / "ranking_2025_26.md"
log = logging.getLogger("dest")


def pt_tiers(client) -> dict:
    tiers = {}
    for name, eids in (("Liga 3", LIGA3_SERIES_NEXT), ("Liga 2", [PT_EDITIONS_NEXT[2]]),
                       ("I Liga", [PT_EDITIONS_NEXT[1]])):  # higher tier wins
        for eid in eids:
            path = config.RAW_DIR / "zerozero" / "edition" / f"{eid}.html"
            if not path.exists() and client is not None:
                zz.fetch_edition(client, eid)
            if not path.exists():
                raise SystemExit(f"Edition {eid} not cached: run with --fetch")
            teams = parse_edition_teams(path.read_text(encoding="utf-8"))
            if not teams:
                raise SystemExit(f"Edition {eid}: no teams parsed")
            tiers |= {t["team_id"]: name for t in teams}
    return tiers


def countries_reaching_threshold(opta: dict) -> set[str]:
    """Countries with at least one mapped league rated >= the threshold, in the
    main tables or in [descriptive]. Every other country counts as below."""
    threshold = opta["meta"]["threshold"]
    out = set()
    for table in (opta, opta.get("descriptive", {})):
        ratings = table.get("ratings", {})
        for key, name in table.get("map", {}).items():
            rating = ratings.get(name)
            if rating is not None and rating >= threshold:
                out.add(key.split("|")[0])
    return out


def foreign_league(r, client, opta):
    if r.country not in countries_reaching_threshold(opta):
        return None, None, "país sem liga >= Liga 2"
    # Calendar-year countries: the 2026 season has epoca_id 155, not 156.
    sid = int(r.season_source_id) if pd.notna(r.season_source_id) else NEXT_SEASON_ID
    path = config.RAW_DIR / "zerozero" / "team" / f"{int(r.club_source_id)}_{sid}.html"
    if not path.exists() and client is not None:
        try:
            zz.fetch_team_season(client, int(r.club_source_id), sid)
        except PageError as exc:
            log.warning("Team page unavailable for %s: %s", r.club, exc)
    if not path.exists():
        return None, None, "página em falta"
    page = parse_team_season(path.read_text(encoding="utf-8"))
    league = page["league"]["competition"] if page["league"] else None
    if not page["competitions"]:
        return None, None, "página sem competições"
    rating = opta_rating(r.country, league, r.club_source_id, opta)
    if rating is None and isinstance(league, str):
        desc = opta.get("descriptive", {})
        rating = desc.get("ratings", {}).get(desc.get("map", {}).get(f"{r.country}|{league}"))
    if rating is not None and rating >= opta["meta"]["threshold"] \
            and not in_main_squad(path.read_text(encoding="utf-8"), r.player_source_id):
        return league, rating, "fora do plantel principal"
    return league, rating, "página de equipa"


def classify(fetch: bool) -> pd.DataFrame:
    ps = pd.read_parquet(config.PROCESSED_DIR / "player_season_minutes.parquet")
    ps = ps[ps["season"] == RANK_SEASON]
    players = ps.groupby("player_source_id").agg(
        jogador=("player_name", "first"), is_b_team=("is_b_team", "max"),
        l3_teams=("team_id", lambda s: set(s)))
    career = pd.read_parquet(config.INTERIM_DIR / "player_career.parquet")
    nxt = career[(career["season"] == NEXT_SEASON) & career["player_source_id"].isin(players.index)]
    nxt = nxt[~nxt["squad"].fillna("").str.contains(r"(?:Jun\.|S1\d|S2\d)")]
    nxt = nxt.sort_values("matches", ascending=False, na_position="last").drop_duplicates("player_source_id")
    opta = load_opta()
    threshold = opta["meta"]["threshold"]
    client = zz.make_client() if fetch else None
    rows = []
    try:
        tiers = pt_tiers(client)
        for r in nxt.itertuples():
            league, rating, src = None, None, "edições PT 2026/27"
            if r.country == "PT":
                cat = tiers.get(r.club_source_id, "Campeonato de Portugal ou abaixo")
                league = cat
            else:
                league, rating, src = foreign_league(r, client, opta)
                cat = "estrangeiro, rating >= Liga 2" if rating is not None and rating >= threshold \
                    and src != "fora do plantel principal" else "estrangeiro, abaixo"
            rows.append({"player_source_id": r.player_source_id, "clube_2026_27": r.club,
                         "club_2026_id": r.club_source_id, "país": r.country, "liga_2026_27": league,
                         "rating_opta": rating, "fonte_liga": src, "jogos_2026_27": r.matches,
                         "equipa_b_2026_27": r.squad == "B", "destino": cat})
        if client is not None:
            log.info("Network requests made: %d", client.requests_made)
    finally:
        if client is not None:
            client.close()
    d = players.reset_index().merge(pd.DataFrame(rows), on="player_source_id", how="left")
    d["destino"] = d["destino"].fillna("sem clube ou sem registo")
    d["mesmo_clube"] = [pd.notna(c) and c in t for c, t in zip(d["club_2026_id"], d["l3_teams"])]
    up = d["destino"].isin(UP)
    d["subiu_com_propria_equipa"] = up & d["mesmo_clube"]
    d["subiu_individual"] = up & ~d["mesmo_clube"]
    return d


def distribution(d: pd.DataFrame) -> pd.DataFrame:
    n = d["destino"].value_counts()
    t = pd.DataFrame({"destino": CATS, "jogadores": [int(n.get(c, 0)) for c in CATS]})
    t["%"] = (100 * t["jogadores"] / len(d)).round(1)
    t["mesmo clube"] = [int(d.loc[d["destino"] == c, "mesmo_clube"].sum()) for c in CATS]
    t["destino equipa B"] = [int(d.loc[d["destino"] == c, "equipa_b_2026_27"].fillna(False).astype(bool).sum())
                            for c in CATS]
    return pd.concat([t, pd.DataFrame([{"destino": "total", "jogadores": len(d), "%": 100.0,
                                        "mesmo clube": int(d["mesmo_clube"].sum()),
                                        "destino equipa B": int(t["destino equipa B"].sum())}])])


def rate_by(e: pd.DataFrame, col: str, order=None) -> pd.DataFrame:
    g = e.groupby(col, observed=True)["y"].agg(["size", "sum", "mean"]).reset_index()
    g.columns = [col, "jogadores", "subiram", "% subiu"]
    g["subiram"] = g["subiram"].astype(int)
    g["% subiu"] = (100 * g["% subiu"]).round(1)
    if order is not None:
        g = g.set_index(col).reindex(order).dropna(how="all").reset_index()
    return g


def label(r) -> str:
    if r.destino == "sem clube ou sem registo":
        return "sem registo"
    extra = f", Opta {r.rating_opta:.1f}" if pd.notna(r.rating_opta) else ""
    lg = r.liga_2026_27 if isinstance(r.liga_2026_27, str) else "liga ?"
    return f"{r.clube_2026_27} ({r.país}, {lg}{extra})"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    d = classify(args.fetch)
    missing = d[d["fonte_liga"].isin(["página em falta", "página sem competições"])]
    if len(missing):
        log.warning("%d foreign destinations without a team page", len(missing))
    unmapped = d[d["país"].ne("PT") & d["fonte_liga"].eq("página de equipa") & d["rating_opta"].isna()]
    feats = pd.read_parquet(config.PROCESSED_DIR / "features.parquet")
    model, C, train = train_final(feats)
    r = ranked(feats, model)  # non-B 2025/26, sorted by score
    for name, cols in (("p_a", ["pct_team_minutes"]), ("p_b", ["age_c", "age_c2", "pct_team_minutes"])):
        r[name] = logit(cols, 1e6).fit(train, train["subiu_individual"]).predict_proba(r)[:, 1]
    e = r.drop(columns=["subiu_individual"]).merge(d.drop(columns=["jogador", "is_b_team"]), on="player_source_id", how="left")
    # Same convention as the target: a rise with the own team is not observed individually.
    e_obs = e[~e["subiu_com_propria_equipa"]].copy()
    e_obs["y"] = e_obs["subiu_individual"].astype(int)
    e_obs["decil"] = pd.qcut(e_obs["score"].rank(method="first"), 10, labels=range(1, 11))
    e_obs["faixa etária"] = pd.cut(e_obs["age"], [0, 20.999, 22.999, 24.999, 27.999, 99],
                                   labels=["<= 20", "21-22", "23-24", "25-27", ">= 28"])

    dec = rate_by(e_obs, "decil")
    prec = []
    for name, col in (("c) modelo final", "score"), ("b) idade + minutos", "p_b"), ("a) só minutos", "p_a")):
        prec.append({"modelo": name, "precisão top 20": round(precision_at_k(e_obs["y"], e_obs[col], 20), 2),
                     "precisão top 50": round(precision_at_k(e_obs["y"], e_obs[col], 50), 2)})
    e_obs["destino_txt"] = [label(x) for x in e_obs.itertuples()]
    view = lambda x: pd.DataFrame({  # noqa: E731
        "jogador": x["jogador"], "idade": x["age"].round(1), "clube 2025/26": x["clube"],
        "posição": x["position_model"], "% minutos": x["pct_team_minutes"], "percentil": x["percentil"],
        "2026/27": x["destino_txt"]})
    high_miss = e_obs[e_obs["y"] == 0].sort_values("score", ascending=False).head(10)
    low_hit = e_obs[e_obs["y"] == 1].sort_values("score").head(10)

    b = d[d["is_b_team"].astype(bool)]
    nb = d[~d["is_b_team"].astype(bool)]
    n_up = int(e_obs["y"].sum())
    lines = [SECTION, "",
             "Só descritivo: não muda o modelo, as features, as etiquetas nem o limiar.", "",
             "**Limitações**", "",
             "- Estar num clube não quer dizer jogar: ainda não há 450 minutos para verificar.",
             "- A época 2026/27 vai a meio; as transferências de janeiro ainda não estão incluídas.",
             "- Os perfis foram descarregados a 13 e 14/11/2026.", "",
             "**Método.** Clube de 2026/27 = linha sénior da carreira com mais jogos. Liga 3 / Liga 2 / "
             "I Liga pelas páginas das edições 2026/27; outro clube português = Campeonato de Portugal "
             "ou abaixo. Estrangeiro: liga da página de equipa 2026/27 (época de 2026 nos países de ano "
             "civil) e rating Opta (snapshot de 13/11/2026; `[descriptive]` para ligas novas: Dinamarca e "
             "Suécia, league-meta.json de 14/11/2026); países sem nenhuma liga >= Liga 2 "
             "no config contam como abaixo sem pedido. Um destino "
             "estrangeiro >= Liga 2 só conta se o jogador estiver no plantel da equipa principal na página "
             "do clube (não reservas nem sub-23); clubes sem competições na página contam como abaixo. **Subiu "
             "individualmente** = I Liga, Liga 2 ou estrangeiro >= Liga 2 num clube diferente do(s) "
             "de 2025/26; quem subiu com a própria equipa fica fora das taxas (b-e), como no alvo.", "",
             f"### a) Distribuição dos destinos: equipas não-B ({len(nb)} jogadores)", "",
             md(distribution(nb)), "",
             f"Subiram com a própria equipa: {int(nb['subiu_com_propria_equipa'].sum())}. "
             f"Subiram individualmente: {int(nb['subiu_individual'].sum())}.", "",
             f"### Equipas B ({len(b)} jogadores), só a distribuição", "", md(distribution(b)), "",
             f"### b) % que já subiu individualmente por decil do score (n = {len(e_obs)}, {n_up} subiram)", "",
             "Decil 10 = scores mais altos.", "", md(dec), "",
             "### c) Por faixa etária, posição e equipa", "",
             md(rate_by(e_obs, "faixa etária")), "",
             md(rate_by(e_obs, "position_model", ["GK", "DF", "MF", "FW", "unknown"]).rename(
                 columns={"position_model": "posição"})), "",
             md(rate_by(e_obs, "clube_principal").sort_values("% subiu", ascending=False).rename(
                 columns={"clube_principal": "equipa"})), "",
             "### d) Precisão com este resultado parcial", "",
             "Mesma população; modelos a) e b) treinados nos mesmos dados do modelo final.", "",
             md(pd.DataFrame(prec)), "",
             "### e) Surpresas", "",
             "Score alto que (ainda) não subiu:", "", md(view(high_miss)), "",
             "Score baixo que subiu:", "", md(view(low_hit)), ""]
    if len(unmapped):
        lines += ["Ligas estrangeiras sem rating no config (contam como abaixo): "
                  + "; ".join(sorted({f"{x.país} {x.liga_2026_27}" for x in unmapped.itertuples()})), ""]
    if len(missing):
        lines += [f"Destinos estrangeiros sem competições na página do clube (contam como abaixo): {len(missing)}.", ""]

    write_section(REPORT, "\n".join(lines), SECTION)
    print("\n".join(lines))

    fig, ax = figure("The top decile stands out",
                     f"Share of 2025/26 Liga 3 players already at a level >= Liga Portugal 2\n"
                     f"at another club in {NEXT_SEASON}, by score decile (partial result)")
    bars(ax, [str(i) for i in range(1, 11)], dec["% subiu"].to_numpy() / 100, [MUTED] * 9 + [FOCUS])
    for t in ax.texts:
        t.set_fontsize(17)
    ax.set_xlabel("Score decile (10 = highest)", fontsize=20)
    save(fig, "e_step_up_by_decile_2026_27.png")
    d.drop(columns="l3_teams").to_parquet(config.PROCESSED_DIR / "destinations_2026_27.parquet", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
