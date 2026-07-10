"""Оценка прогона генерации: метрики по каждой паре (исходник, результат)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import gdal_io
from generate_images import resize_to_resolution
from metrics import Metrics, combined_score
from prompts import ROOT, load_config


def find_run_dir(config: dict, run_arg: str | None) -> Path:
    base = ROOT / config["output"]["generated_dir"]
    if run_arg:
        run_dir = Path(run_arg)
        if not run_dir.is_absolute():
            run_dir = ROOT / run_arg
        if not gdal_io.dir_exists(run_dir):
            raise FileNotFoundError(f"Нет папки прогона: {run_dir}")
        return run_dir
    runs = sorted(base / name for name in gdal_io.list_dir(base)
                  if gdal_io.file_exists(base / name / "metadata.jsonl"))
    if not runs:
        raise FileNotFoundError(f"В {base} нет прогонов с metadata.jsonl")
    return runs[-1]


def load_records(run_dir: Path) -> list[dict]:
    meta = run_dir / "metadata.jsonl"
    records = [json.loads(line) for line in
               meta.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError(f"Пустой metadata.jsonl: {meta}")
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default=None,
                    help="папка прогона (по умолчанию — последняя)")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = ap.parse_args()

    config = load_config(Path(args.config))
    ecfg = config["evaluation"]
    run_dir = find_run_dir(config, args.run)
    # параметры генерации — из снапшота прогона: config.yaml мог измениться после генерации
    run_cfg = load_config(run_dir / "run_config.yaml")

    out_dir = ROOT / config["output"]["evaluation_dir"]
    gdal_io.makedirs(out_dir)
    out_csv = out_dir / f"{run_dir.name}_results.csv"

    records = load_records(run_dir)
    print(f"Прогон: {run_dir.name} | записей: {len(records)}")

    metrics = Metrics(ecfg)
    source_season = ecfg.get("source_season", "summer")
    input_dir = ROOT / run_cfg["dataset"]["input_dir"]
    resolution = int(run_cfg["generation"]["resolution"])

    src_cache = {}
    rows = []
    for i, rec in enumerate(records, 1):
        if rec["source"] not in src_cache:
            rgb, _ = gdal_io.read_image(input_dir / rec["source"])
            src_cache[rec["source"]] = resize_to_resolution(rgb, resolution)
        src = src_cache[rec["source"]]
        gen, _ = gdal_io.read_image(run_dir / rec["out_file"])

        m = metrics.evaluate_pair(src, gen, rec["season"], source_season,
                                  src_key=rec["source"])
        s = combined_score(m, ecfg["weights"],
                           float(ecfg["structure_threshold"]),
                           ecfg["structure_weights"])
        rows.append({
            "run_id": rec["run_id"],
            "source": rec["source"],
            "season": rec["season"],
            "ptype": rec["ptype"],
            "variant": rec.get("variant", 0),
            "out_file": rec["out_file"],
            "dino_sim": round(m["dino_sim"], 4),
            "ssim": round(m["ssim"], 4),
            "clip_season_prob": round(m["clip_season_prob"], 4),
            "clip_directional": (round(m["clip_directional"], 4)
                                 if m["clip_directional"] is not None else None),
            "structure_score": s["structure"],
            "season_score": s["season"],
            "disqualified": s["disqualified"],
            "score": s["score"],
            "gen_time_sec": rec.get("gen_time_sec"),
        })
        print(f"  [{i}/{len(records)}] {rec['out_file']}: "
              f"score={s['score']}{' DISQ' if s['disqualified'] else ''}")

    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")
    n_disq = sum(r["disqualified"] for r in rows)
    print(f"\nИтог: {len(rows)} строк, дисквалифицировано {n_disq}")
    print(f"CSV: {out_csv}")


if __name__ == "__main__":
    main()
