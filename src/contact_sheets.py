"""Контактные листы: исходник + результаты всех типов промптов в ряд."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import gdal_io
from evaluate_results import find_run_dir, load_records
from prompts import ROOT, load_config

LABEL_H = 22


def thumb(rgb: np.ndarray, size: int, label: str) -> Image.Image:
    im = Image.fromarray(rgb).resize((size, size), Image.LANCZOS)
    tile = Image.new("RGB", (size, size + LABEL_H), (20, 20, 20))
    tile.paste(im, (0, 0))
    ImageDraw.Draw(tile).text((4, size + 4), label[:40], fill=(230, 230, 230))
    return tile


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default=None)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--thumb", type=int, default=256)
    args = ap.parse_args()

    config = load_config(Path(args.config))
    run_dir = find_run_dir(config, args.run)
    records = load_records(run_dir)
    run_cfg = load_config(run_dir / "run_config.yaml")
    ptypes = run_cfg["prompt_types"]
    input_dir = ROOT / run_cfg["dataset"]["input_dir"]
    out_dir = (ROOT / config["output"]["evaluation_dir"]
               / f"{run_dir.name}_sheets")
    gdal_io.makedirs(out_dir)
    size = args.thumb

    by_key: dict[tuple[str, str, int], dict[str, str]] = {}
    for r in records:
        key = (r["source"], r["season"], r.get("variant", 0))
        by_key.setdefault(key, {})[r["ptype"]] = r["out_file"]

    n_sheets = 0
    for (source, season, variant), files in sorted(by_key.items()):
        src_rgb, _ = gdal_io.read_image(input_dir / source)
        tiles = [thumb(src_rgb, size, "SOURCE")]
        for pt in ptypes:
            if pt not in files:
                continue
            gen_rgb, _ = gdal_io.read_image(run_dir / files[pt])
            tiles.append(thumb(gen_rgb, size, pt))

        sheet = Image.new("RGB",
                          (size * len(tiles), size + LABEL_H + 20),
                          (20, 20, 20))
        for i, t in enumerate(tiles):
            sheet.paste(t, (i * size, 20))
        title = f"{Path(source).stem} | {season} | v{variant}"
        ImageDraw.Draw(sheet).text((4, 4), title, fill=(255, 255, 120))
        name = f"{Path(source).stem}__{season}__v{variant}.png"
        sheet.save(out_dir / name)
        n_sheets += 1

    print(f"Готово: {n_sheets} листов -> {out_dir}")


if __name__ == "__main__":
    main()
