"""Валидация прогона и sanity-проверка метрик: PASS/FAIL, exit 1 при провале."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import gdal_io
from evaluate_results import find_run_dir, load_records
from prompts import ROOT, load_config

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))


def validate_completeness(config: dict, records: list[dict],
                          run_dir: Path) -> None:
    combos = {(r["source"], r["season"], r["ptype"], r.get("variant", 0))
              for r in records}
    check("метаданные: нет дублей комбинаций", len(combos) == len(records),
          f"{len(combos)} уникальных / {len(records)} записей")

    sources = {r["source"] for r in records}
    variants = {r.get("variant", 0) for r in records}
    expected = (len(sources) * len(config["seasons"])
                * len(config["prompt_types"]) * len(variants))
    check("метаданные: полная матрица",
          len(records) == expected,
          f"{len(records)} записей, ожидалось {expected} "
          f"({len(sources)} исходников x {len(config['seasons'])} сезонов x "
          f"{len(config['prompt_types'])} типов x {len(variants)} вариантов)")

    missing = [r["out_file"] for r in records
               if not gdal_io.file_exists(run_dir / r["out_file"])]
    check("файлы: все из метаданных существуют", not missing,
          f"отсутствуют: {missing[:3]}" if missing else "")

    pngs = {n for n in gdal_io.list_dir(run_dir) if n.lower().endswith(".png")}
    extra = pngs - {r["out_file"] for r in records}
    check("файлы: нет лишних png", not extra,
          f"лишние: {sorted(extra)[:3]}" if extra else "")


def validate_images(records: list[dict], run_dir: Path) -> None:
    bad_read, bad_flat, bad_size = [], [], []
    for r in records:
        path = run_dir / r["out_file"]
        if not gdal_io.file_exists(path):
            continue
        try:
            rgb, _ = gdal_io.read_image(path)
        except Exception:
            bad_read.append(r["out_file"])
            continue
        if float(rgb.std()) < 2.0:  # одноцветная заливка = сломанная генерация
            bad_flat.append(r["out_file"])
        if list(rgb.shape[:2]) != list(r["resolution"]):
            bad_size.append(r["out_file"])
    check("изображения: все читаются", not bad_read,
          f"битые: {bad_read[:3]}" if bad_read else "")
    check("изображения: нет одноцветных", not bad_flat,
          f"плоские: {bad_flat[:3]}" if bad_flat else "")
    check("изображения: размер соответствует метаданным", not bad_size,
          f"несоответствие: {bad_size[:3]}" if bad_size else "")


def validate_metrics_csv(config: dict, run_dir: Path,
                         n_records: int) -> None:
    import pandas as pd

    csv = (ROOT / config["output"]["evaluation_dir"]
           / f"{run_dir.name}_results.csv")
    if not gdal_io.file_exists(csv):
        check("оценка: CSV существует", False,
              f"{csv.name} не найден — запустите evaluate_results.py")
        return
    df = pd.read_csv(csv)
    check("оценка: строк столько же, сколько записей прогона",
          len(df) == n_records, f"{len(df)} строк / {n_records} записей")
    in01 = lambda s: bool(((s >= -0.01) & (s <= 1.01)).all())  # noqa: E731
    check("оценка: ssim в [0,1]", in01(df["ssim"]))
    check("оценка: clip_season_prob в [0,1]", in01(df["clip_season_prob"]))
    check("оценка: dino_sim в [-1,1]",
          bool(((df["dino_sim"] >= -1.01) & (df["dino_sim"] <= 1.01)).all()))
    check("оценка: нет NaN в score", bool(df["score"].notna().all()))
    n_disq = int(df["disqualified"].sum())
    check("оценка: дисквалификаций меньше половины",
          n_disq < len(df) / 2,
          f"{n_disq}/{len(df)} (много = генерация ломает сцены или порог "
          "неадекватен)")


def validate_anchors(config: dict) -> None:
    from metrics import Metrics

    input_dir = ROOT / config["dataset"]["input_dir"]
    sources = [input_dir / n for n in
               sorted(n for n in gdal_io.list_dir(input_dir)
                      if n.lower().endswith(".png"))][:3]
    if not sources:
        check("якоря: есть исходники", False)
        return
    m = Metrics(config["evaluation"])

    dino_id, ssim_id, summer_probs = [], [], []
    for p in sources:
        rgb, _ = gdal_io.read_image(p)
        dino_id.append(m.dino_similarity(rgb, rgb))
        ssim_id.append(m.ssim(rgb, rgb))
        summer_probs.append(m.clip_season_prob(rgb, "summer"))
    check("якорь: dino(x, x) ~ 1", min(dino_id) > 0.999,
          f"min={min(dino_id):.4f}")
    check("якорь: ssim(x, x) = 1", min(ssim_id) > 0.999,
          f"min={min(ssim_id):.4f}")
    check("якорь: исходники CLIP-ится как summer (медиана > 0.5)",
          float(np.median(summer_probs)) > 0.5,
          f"медиана={float(np.median(summer_probs)):.3f} "
          f"(низкая = тексты сезонов не работают на этих данных)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default=None)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--skip-anchors", action="store_true",
                    help="пропустить якорные тесты метрик (без моделей)")
    args = ap.parse_args()

    config = load_config(Path(args.config))
    run_dir = find_run_dir(config, args.run)
    records = load_records(run_dir)
    # матрица прогона — из его снапшота: текущий config.yaml мог измениться
    run_cfg = load_config(run_dir / "run_config.yaml")
    print(f"Валидация прогона {run_dir.name} ({len(records)} записей)\n")

    print("— Полнота —")
    validate_completeness(run_cfg, records, run_dir)
    print("— Целостность изображений —")
    validate_images(records, run_dir)
    print("— Метрики (CSV оценки) —")
    validate_metrics_csv(config, run_dir, len(records))
    if not args.skip_anchors:
        print("— Якоря метрик —")
        validate_anchors(config)

    failed = [name for name, ok, _ in CHECKS if not ok]
    print(f"\nИтог: {len(CHECKS) - len(failed)}/{len(CHECKS)} чеков пройдено")
    if failed:
        print("Провалено:", "; ".join(failed))
        sys.exit(1)
    print("ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ")


if __name__ == "__main__":
    main()
