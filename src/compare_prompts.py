"""Агрегация CSV оценки по типам промптов: рейтинг и итоговый отчёт."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

from prompts import ROOT, build_prompt_matrix, load_config, load_prompts


def find_results_csv(config: dict, arg: str | None) -> Path:
    base = ROOT / config["output"]["evaluation_dir"]
    if arg:
        p = Path(arg)
        if not p.is_absolute():
            p = ROOT / arg
        if not p.is_file():
            raise FileNotFoundError(f"Нет файла результатов: {p}")
        return p
    files = sorted(base.glob("*_results.csv"))
    if not files:
        raise FileNotFoundError(f"В {base} нет *_results.csv — сначала "
                                "запустите evaluate_results.py")
    return files[-1]


def paired_vs_best(df: pd.DataFrame, best: str) -> pd.DataFrame:
    """Парное сравнение с лидером на одинаковых (source, season, variant)."""
    wide = df.pivot_table(values="score", index=["source", "season", "variant"],
                          columns="ptype")
    rows = []
    for pt in wide.columns:
        if pt == best:
            continue
        pair = wide[[best, pt]].dropna()
        delta = pair[best] - pair[pt]
        pvalue = 1.0 if (delta == 0).all() else float(
            wilcoxon(pair[best], pair[pt]).pvalue)
        rows.append({
            "ptype": pt,
            "n_pairs": len(pair),
            "mean_delta_vs_best": round(float(delta.mean()), 4),
            "best_wins_share": round(float((delta > 0).mean()), 3),
            "wilcoxon_p": round(pvalue, 4),
        })
    return pd.DataFrame(rows).set_index("ptype").sort_values("mean_delta_vs_best")


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby("ptype").agg(
        n=("score", "size"),
        mean_score=("score", "mean"),
        mean_structure=("structure_score", "mean"),
        mean_season=("season_score", "mean"),
        mean_dino=("dino_sim", "mean"),
        mean_ssim=("ssim", "mean"),
        mean_clip_prob=("clip_season_prob", "mean"),
        mean_clip_dir=("clip_directional", "mean"),
        n_disqualified=("disqualified", "sum"),
        std_score=("score", "std"),
    ).round(4)
    return agg.sort_values("mean_score", ascending=False)


def build_summary_md(df: pd.DataFrame, ranking: pd.DataFrame,
                     config: dict, results_csv: Path) -> str:
    best = ranking.index[0]
    best_structure = ranking["mean_structure"].idxmax()
    best_season = ranking["mean_season"].idxmax()
    most_artifacts = (ranking["n_disqualified"].idxmax()
                      if ranking["n_disqualified"].max() > 0
                      else ranking["mean_structure"].idxmin())

    pivot = (df.pivot_table(values="score", index="ptype", columns="season",
                            aggfunc="mean").round(4))

    matrix = build_prompt_matrix(config, load_prompts())
    best_prompts = [it for it in matrix if it.ptype == best]

    paired = paired_vs_best(df, best)

    lines = [
        "# Сравнение промптов",
        "",
        f"Источник данных: `{results_csv.name}` "
        f"({int(ranking['n'].sum())} оценённых изображений).",
        "",
        "## Рейтинг типов промптов (по итоговому score)",
        "",
        ranking.to_markdown(),
        "",
        "## Парное сравнение с лидером (одинаковые исходник/сезон/вариант)",
        "",
        f"Лидер: **{best}**. mean_delta > 0 и малый wilcoxon_p — лидер "
        "статистически значимо лучше; p > 0.05 — разница в пределах шума.",
        "",
        paired.to_markdown(),
        "",
        "## Score по сезонам",
        "",
        pivot.to_markdown(),
        "",
        "## Ответы на вопросы сравнения",
        "",
        f"- Лучше всего сохраняет исходную сцену: **{best_structure}** "
        f"(structure {ranking.loc[best_structure, 'mean_structure']:.4f})",
        f"- Лучше всего передаёт сезон: **{best_season}** "
        f"(season {ranking.loc[best_season, 'mean_season']:.4f})",
        f"- Чаще всего портит сцену/добавляет лишнее: **{most_artifacts}** "
        f"(дисквалификаций {int(ranking.loc[most_artifacts, 'n_disqualified'])}, "
        f"structure {ranking.loc[most_artifacts, 'mean_structure']:.4f})",
        f"- Лучший для дальнейшей работы: **{best}** "
        f"(score {ranking.loc[best, 'mean_score']:.4f})",
        "",
        "## Обоснование выбора",
        "",
        f"Тип **{best}** набрал максимальный средний итоговый score "
        f"({ranking.loc[best, 'mean_score']:.4f}) при "
        f"{int(ranking.loc[best, 'n_disqualified'])} дисквалификациях. "
        f"Итоговый score взвешивает сохранение структуры "
        f"(вес {config['evaluation']['weights']['structure']}) и "
        f"выраженность сезона (вес {config['evaluation']['weights']['season']}); "
        f"результаты со структурой ниже порога "
        f"{config['evaluation']['structure_threshold']} исключаются. "
        "Числа автоматических метрик необходимо подтвердить визуальной "
        "проверкой по контактным листам (contact_sheets.py).",
        "",
        f"## Тексты промптов лучшего типа ({best})",
        "",
    ]
    for it in best_prompts:
        lines.append(f"**{it.season}:**")
        lines.append("```text")
        lines.append(it.prompt)
        if it.negative_prompt:
            lines.append(f"NEGATIVE: {it.negative_prompt}")
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=None,
                    help="CSV из evaluate_results.py (по умолчанию — последний)")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = ap.parse_args()

    config = load_config(Path(args.config))
    results_csv = find_results_csv(config, args.results)
    df = pd.read_csv(results_csv)
    print(f"Результаты: {results_csv.name} | строк: {len(df)}")

    ranking = aggregate(df)
    run_id = results_csv.stem.replace("_results", "")
    out_dir = ROOT / config["output"]["evaluation_dir"]

    ranking_csv = out_dir / f"{run_id}_prompt_ranking.csv"
    ranking.to_csv(ranking_csv, encoding="utf-8")

    summary_md = out_dir / f"{run_id}_summary.md"
    summary_md.write_text(build_summary_md(df, ranking, config, results_csv),
                          encoding="utf-8")

    print("\n=== Рейтинг типов промптов ===")
    print(ranking[["mean_score", "mean_structure", "mean_season",
                   "n_disqualified"]].to_string())
    print(f"\nЛучший тип промпта: {ranking.index[0]}")
    print(f"Таблица: {ranking_csv}")
    print(f"Отчёт:   {summary_md}")


if __name__ == "__main__":
    main()
