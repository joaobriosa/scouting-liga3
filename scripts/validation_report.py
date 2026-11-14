"""Write a short validation report (Markdown) from data/interim/.

    python scripts/validation_report.py            # random sample of 3 matches
    python scripts/validation_report.py --seed 42  # reproducible sample

Output: reports/validation_report.md (report text is in Portuguese).
"""

from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime

import pandas as pd

from scouting import config, validation

REPORT_PATH = config.PROJECT_ROOT / "reports" / "validation_report.md"

CHECK_LABELS = {
    "team_minutes": "Minutos por equipa = 11 × duração (90, ou 120 com prolongamento) − minutos perdidos por expulsões",
    "player_minutes_range": "Nenhum jogador com < 0 minutos ou mais do que a duração do jogo",
    "minutes_unknown": "Sem minutos de entrada/saída/expulsão desconhecidos (n/d)",
    "starters": "11 titulares por equipa e jogo",
    "goals_vs_score": "Golos em goal_events = resultado final (autogolos à equipa beneficiada)",
    "calendar_format": "Calendário: cada equipa com 2×(n.º equipas − 1) jogos por fase",
    "matches_per_team": "Jogos por equipa recolhidos = calendário (no âmbito recolhido)",
    "missing_matches": "Nenhum jogo do âmbito em falta",
    "calendar_results": "Resultados do calendário no âmbito são todos marcadores normais",
    "unknown_matches": "Todos os jogos recolhidos existem no calendário",
    "duplicate_match_ids": "Nenhum match_id repetido (dentro ou entre épocas)",
    "parse_failures": "Todas as fichas recolhidas foram lidas sem erro",
    "unknown_events": "Nenhum evento (ícone) por classificar nas fichas",
    "players_missing": "Todos os jogadores têm linha em players",
    "birth_date": "Todos os jogadores têm data de nascimento",
}


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(vazio)_"
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        cells = ["" if pd.isna(v) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def fmt_minute(minute, added) -> str:
    if pd.isna(minute):
        return ""
    return f"{int(minute)}+{int(added)}'" if not pd.isna(added) else f"{int(minute)}'"


def match_detail(mid: int, t: dict[str, pd.DataFrame]) -> str:
    m = t["matches"].set_index("match_id").loc[mid]
    goals = t["goal_events"][t["goal_events"]["match_id"] == mid].sort_values(["minute", "added_time_minute"])
    apps = t["appearances"][t["appearances"]["match_id"] == mid]
    url = m["source_url"]
    out = [
        f"### {m['home_team']} {m['home_goals']}-{m['away_goals']} {m['away_team']}",
        f"{m['edition_name']}, {m['round']}, {m['date']:%Y-%m-%d} · match_id {mid} · [ficha na fonte]({url})",
        "",
        "**Golos:** " + (", ".join(
            f"{fmt_minute(g.minute, g.added_time_minute)} {g.scorer_name} ({g.scoring_team})"
            + (" [pen.]" if g.is_penalty else "") + (" [autogolo]" if g.is_own_goal else "")
            for g in goals.itertuples()
        ) or "nenhum"),
        "",
    ]
    for team in (m["home_team"], m["away_team"]):
        a = apps[apps["team"] == team].copy()
        a = a.sort_values(["started", "unused_sub", "minute_in"], ascending=[False, True, True])
        a["entrou"] = [fmt_minute(i, ad) if not s and not u else "" for i, ad, s, u in
                       zip(a["minute_in"], a["minute_in_added"], a["started"], a["unused_sub"])]
        a["saiu"] = [fmt_minute(o, ad) if not u and (o != 90 or not pd.isna(ad)) else "" for o, ad, u in
                     zip(a["minute_out"], a["minute_out_added"], a["unused_sub"])]
        a["estado"] = a.apply(lambda r: "titular" if r.started else ("não utilizado" if r.unused_sub else "suplente"), axis=1)
        a["cap."] = a["is_captain"].map({True: "C", False: ""}).fillna("?")
        a["cartões"] = a.apply(lambda r: "A" * int(r.yellow_cards) + (f" V{int(r.red_card_minute)}'" if r.red_card and not pd.isna(r.red_card_minute) else (" V" if r.red_card else "")), axis=1)
        view = a[["shirt_number", "player_name", "cap.", "position", "estado", "entrou", "saiu",
                  "minutes_played", "goals", "cartões"]].rename(columns={
            "shirt_number": "n.º", "player_name": "jogador", "position": "pos.",
            "minutes_played": "min", "goals": "golos"})
        out += [f"**{team}** (total {int(a['minutes_played'].sum())} min)", "", md_table(view), ""]
    return "\n".join(out)


def build_report(seed: int) -> str:
    t = validation.load_tables()
    results = validation.run_all(t)
    raw_results = validation.run_all(t, use_exceptions=False)
    exceptions = validation.load_exceptions()
    m, a, g, p = t["matches"], t["appearances"], t["goal_events"], t["players"]
    scope = validation.collected_scope(m, t["calendar"])

    lines = [
        "# Relatório de validação",
        "",
        f"Gerado em {datetime.now():%Y-%m-%d %H:%M} · fonte: zerozero.pt · épocas {', '.join(sorted(m['season'].unique()))}",
        "",
        "## Âmbito",
        "",
        f"- Jogos recolhidos: **{len(m)}** de {len(scope)} no âmbito "
        f"(calendário completo da época: {len(t['calendar'])} jogos).",
        "",
        md_table(m.groupby(["season", "stage", "series"], as_index=False)
                 .agg(jornadas=("round", "nunique"), jogos=("match_id", "size"))),
        "",
        "## Números",
        "",
        f"- Linhas em `appearances`: {len(a)} "
        f"(titulares {int(a['started'].sum())}, suplentes utilizados "
        f"{int((~a['started'] & ~a['unused_sub']).sum())}, não utilizados {int(a['unused_sub'].sum())})",
        f"- Jogadores únicos: **{a['player_source_id'].nunique()}** "
        f"(sem data de nascimento: {int(p['birth_date'].isna().sum())})",
        f"- Golos: {len(g)} (penáltis {int(g['is_penalty'].sum())}, autogolos {int(g['is_own_goal'].sum())}, "
        f"com assistência {int(g['assist_source_id'].notna().sum())})",
        f"- Cartões: {int(a['yellow_cards'].sum())} amarelos, {int(a['red_card'].sum())} vermelhos "
        f"({int(a['red_card_off_pitch'].sum())} a jogadores fora de campo: banco ou já substituídos; "
        f"{int(a['red_card_second_yellow'].sum())} por duplo amarelo)",
        f"- Capitão: indicado em {int(a.groupby(['match_id','team'])['is_captain'].apply(lambda s: s.notna().all()).sum())} "
        f"de {a.groupby(['match_id','team']).ngroups} equipas-jogo",
        "",
        "## Verificações",
        "",
        "| verificação | resultado | problemas | explicados (exceções) |",
        "|---|---|---|---|",
    ]
    for name, label in CHECK_LABELS.items():
        n, n_raw = len(results[name]), len(raw_results[name])
        status = "✅" if n == 0 else "❌"
        lines.append(f"| {label} | {status} | {n} | {n_raw - n} |")
    lines.append("")

    problems = {k: v for k, v in results.items() if len(v)}
    lines += ["## Jogos com problemas", ""]
    if not problems:
        lines += ["Nenhum problema por explicar.", ""]
    for name, df in problems.items():
        lines += [f"### {CHECK_LABELS[name]}", "", md_table(df), ""]

    in_scope_exc = exceptions[exceptions["match_id"].isin(m["match_id"])]
    lines += ["## Exceções explicadas (validation_exceptions.csv)", "", md_table(in_scope_exc), ""]

    rng = random.Random(seed)
    sample = rng.sample(sorted(m["match_id"]), k=min(3, len(m)))
    lines += [
        "## 3 jogos ao acaso para confirmar à mão",
        "",
        f"Semente aleatória: {seed}. Confirmar na fonte: resultado, marcadores e minutos, "
        "onzes, substituições (minutos), cartões e capitão.",
        "",
    ]
    lines += [match_detail(mid, t) for mid in sample]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)
    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(10**6)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(build_report(seed), encoding="utf-8")
    print(f"Report written to {REPORT_PATH.relative_to(config.PROJECT_ROOT)} (seed {seed})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
