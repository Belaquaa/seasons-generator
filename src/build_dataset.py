"""Сборка обучающего набора: для каждого исходника — по N корректных вариантов
каждого сезона, разложенных по подпапкам, с JSON параметров на каждую выборку.

Отбор «корректных» повторяет критерий исследования: сцена сохранена
(структура не ниже порога, генерация не дисквалифицирована) и сезон выражен
(clip_season_prob не ниже порога). Тип промпта по умолчанию — победитель
исследования (kontext_guide, см. practice_report.md).

Раскладка результата:
    <output>/<исходник>/params.json
    <output>/<исходник>/<сезон>/<исходник>__<сезон>__vK.png
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import gdal_io
from dataset import Dataset, DatasetItem
from generate_images import resize_to_resolution, sanitize
from metrics import Metrics, combined_score
from model_flux import GenRequest, get_generator
from prompts import ROOT, PromptItem, build_prompt_matrix, load_config, load_prompts

DEFAULT_PTYPE = "kontext_guide"   # победитель исследования (practice_report.md)


def is_correct(metrics_row: dict, score_row: dict,
               season_prob_threshold: float) -> bool:
    """Корректная генерация: сцена не сломана и сезон уверенно распознан."""
    return (not score_row["disqualified"]
            and metrics_row["clip_season_prob"] >= season_prob_threshold)


def build_for_item(item: DatasetItem, seasons_matrix: dict[str, PromptItem],
                   generator, metrics: Metrics, opts: dict,
                   out_dir: Path) -> tuple[Path, list[dict], dict]:
    """Собрать корректные варианты всех сезонов для одного исходника."""
    gcfg, ecfg = opts["gcfg"], opts["ecfg"]
    src = resize_to_resolution(item.image, opts["resolution"])
    item_dir = out_dir / sanitize(item.stem)
    accepted: list[dict] = []
    per_season_stats: dict[str, dict] = {}

    for season, pitem in seasons_matrix.items():
        n_ok, attempt, n_failed = 0, 0, 0
        while n_ok < opts["per_season"] and attempt < opts["max_attempts"]:
            seed = opts["base_seed"] + attempt
            attempt += 1
            req = GenRequest(item.name, season, pitem.ptype, pitem.prompt,
                             pitem.negative_prompt, seed)
            # сбой одной генерации (OOM, транзиентная ошибка) не роняет прогон:
            # засчитываем попытку и идём дальше; запись на диск — вне обработчика
            try:
                out_rgb = generator.generate(src, req)
                m = metrics.evaluate_pair(src, out_rgb, season,
                                          opts["source_season"], src_key=item.name)
                s = combined_score(m, ecfg["weights"],
                                   float(ecfg["structure_threshold"]),
                                   ecfg["structure_weights"])
            except Exception as e:
                n_failed += 1
                print(f"    ! сбой генерации {item.stem}/{season} seed={seed}: "
                      f"{type(e).__name__}: {e}")
                continue
            if not is_correct(m, s, opts["season_prob_threshold"]):
                continue
            fname = f"{sanitize(item.stem)}__{season}__v{n_ok}.png"
            gdal_io.write_image(item_dir / season / fname, out_rgb, item.io_meta)
            # каждая запись самодостаточна: модель, исходник, результат, промпт
            accepted.append({
                "source_image": item.name,
                "generated_file": f"{season}/{fname}",
                "season": season,
                "model": {
                    "backend": generator.name,
                    "model_id": generator.model_id,
                    "dtype": gcfg.get("dtype"),
                },
                "prompt": {
                    "type": pitem.ptype,
                    "text": pitem.prompt,
                    "negative": pitem.negative_prompt,
                    "guidance_scale": getattr(generator, "guidance", None),
                    "num_inference_steps": int(gcfg["num_inference_steps"]),
                    "seed": seed,
                },
                "metrics": {
                    "clip_season_prob": round(m["clip_season_prob"], 4),
                    "structure_score": s["structure"],
                    "season_score": s["season"],
                    "score": s["score"],
                },
            })
            n_ok += 1
        per_season_stats[season] = {
            "requested": opts["per_season"], "accepted": n_ok,
            "attempts": attempt, "failed": n_failed,
        }
    return item_dir, accepted, per_season_stats


def item_params(item: DatasetItem, seasons_matrix: dict[str, PromptItem],
                accepted: list[dict], per_season_stats: dict, opts: dict,
                generator) -> dict:
    """Параметры выборки одного исходника — уходят в params.json."""
    gcfg, ecfg = opts["gcfg"], opts["ecfg"]
    return {
        "source_image": item.name,
        "prompt_type": opts["ptype"],
        "seasons": list(seasons_matrix),           # «температура» сцены на сезон
        "per_season_requested": opts["per_season"],
        "generation": {
            "backend": generator.name,
            "model_id": generator.model_id,
            "dtype": gcfg.get("dtype"),
            "resolution": opts["resolution"],
            "num_inference_steps": int(gcfg["num_inference_steps"]),
            "guidance_scale": getattr(generator, "guidance", None),
            "base_seed": opts["base_seed"],
        },
        "acceptance": {
            "structure_threshold": float(ecfg["structure_threshold"]),
            "clip_season_prob_threshold": opts["season_prob_threshold"],
            "structure_weights": ecfg["structure_weights"],
            "score_weights": ecfg["weights"],
        },
        "prompts": {season: {"prompt": p.prompt,
                             "negative_prompt": p.negative_prompt}
                    for season, p in seasons_matrix.items()},
        "per_season_stats": per_season_stats,
        "n_accepted": len(accepted),
        "accepted": accepted,
        "created": datetime.now().isoformat(timespec="seconds"),
    }


def run_build(dataset: Dataset, seasons_matrix: dict[str, PromptItem],
              generator, metrics: Metrics, opts: dict, out_dir: Path) -> dict:
    """Пройти по всему датасету, собрать выборки, записать params.json на каждую.
    Генератор и метрики принимаются готовыми — это делает конвейер тестируемым
    и не привязывает его к конкретному backend."""
    gdal_io.makedirs(out_dir)
    n = len(dataset)
    if opts["limit"] > 0:
        n = min(n, opts["limit"])
    print(f"Исходников: {n} | тип промпта: {opts['ptype']} | "
          f"сезонов: {len(seasons_matrix)} | на сезон: {opts['per_season']} "
          f"(до {opts['max_attempts']} попыток)")

    total_accepted = 0
    for i in range(n):
        item = dataset[i]
        item_dir, accepted, stats = build_for_item(
            item, seasons_matrix, generator, metrics, opts, out_dir)
        params = item_params(item, seasons_matrix, accepted, stats, opts,
                             generator)
        gdal_io.write_text(item_dir / "params.json",
                           json.dumps(params, ensure_ascii=False, indent=2))
        total_accepted += len(accepted)
        counts = " ".join(f"{s}:{st['accepted']}/{st['requested']}"
                          for s, st in stats.items())
        print(f"  [{i + 1}/{n}] {item.stem}: {len(accepted)} принято | {counts}")

    print(f"Готово: {total_accepted} изображений от {n} исходников -> {out_dir}")
    return {"n_sources": n, "n_accepted_total": total_accepted}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--input-dir", default=None,
                    help="директория исходников (по умолчанию из config)")
    ap.add_argument("--output-dir", default=str(ROOT / "outputs" / "dataset"))
    ap.add_argument("--images", nargs="*", default=None,
                    help="конкретные имена файлов (по умолчанию — все из директории)")
    ap.add_argument("--limit", type=int, default=0,
                    help="взять только первые N исходников (0 = все)")
    ap.add_argument("--per-season", type=int, default=3,
                    help="сколько корректных вариантов собрать на сезон")
    ap.add_argument("--max-attempts", type=int, default=0,
                    help="макс. попыток на сезон (0 = per-season * 4)")
    ap.add_argument("--seasons", nargs="*", default=None,
                    help="сезоны (по умолчанию из config)")
    ap.add_argument("--ptype", default=DEFAULT_PTYPE,
                    help="тип промпта (по умолчанию победитель исследования)")
    ap.add_argument("--season-threshold", type=float, default=0.5,
                    help="порог clip_season_prob для приёмки корректной генерации")
    ap.add_argument("--seed", type=int, default=None,
                    help="базовый seed (по умолчанию из config)")
    args = ap.parse_args()

    config = load_config(Path(args.config))
    gcfg, ecfg = config["generation"], config["evaluation"]
    seasons = args.seasons or config["seasons"]
    if args.ptype not in config["prompt_types"]:
        raise SystemExit(f"Неизвестный тип промпта {args.ptype!r}; "
                         f"есть: {config['prompt_types']}")

    matrix = build_prompt_matrix(config, load_prompts())
    seasons_matrix = {it.season: it for it in matrix
                      if it.ptype == args.ptype and it.season in seasons}
    missing = [s for s in seasons if s not in seasons_matrix]
    if missing:
        raise SystemExit(f"Нет промптов {args.ptype!r} для сезонов: {missing}")

    input_dir = (Path(args.input_dir) if args.input_dir
                 else ROOT / config["dataset"]["input_dir"])
    dataset = Dataset(input_dir, args.images)

    generator = get_generator(config)
    if hasattr(generator, "prepare_prompts"):
        texts = [t for p in seasons_matrix.values()
                 for t in (p.prompt, p.negative_prompt) if t]
        generator.prepare_prompts(texts)
    metrics = Metrics(ecfg)

    opts = {
        "gcfg": gcfg, "ecfg": ecfg,
        "ptype": args.ptype,
        "per_season": args.per_season,
        "max_attempts": args.max_attempts or args.per_season * 4,
        "base_seed": args.seed if args.seed is not None else int(gcfg["seed"]),
        "resolution": int(gcfg["resolution"]),
        "source_season": ecfg.get("source_season", "summer"),
        "season_prob_threshold": args.season_threshold,
        "limit": args.limit,
    }
    run_build(dataset, seasons_matrix, generator, metrics, opts,
              Path(args.output_dir))


if __name__ == "__main__":
    main()
