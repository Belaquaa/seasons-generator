"""Прогон матрицы генерации (исходник x сезон x тип промпта)."""
from __future__ import annotations

import argparse
import itertools
import json
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import gdal_io
from model_flux import GenRequest, get_generator
from prompts import ROOT, build_prompt_matrix, load_config, load_prompts


def make_run_dir(base: Path) -> Path:
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    for suffix in [""] + [f"-{i}" for i in range(1, 100)]:
        run_dir = base / f"{run_id}{suffix}"
        if not gdal_io.dir_exists(run_dir):
            gdal_io.makedirs(run_dir)
            return run_dir
    raise RuntimeError(f"Не удалось создать уникальную папку прогона в {base}")


def resize_to_resolution(rgb: np.ndarray, resolution: int) -> np.ndarray:
    from PIL import Image

    h, w = rgb.shape[:2]
    scale = resolution / max(h, w)
    new_w = max(16, round(w * scale / 16) * 16)
    new_h = max(16, round(h * scale / 16) * 16)
    if (new_h, new_w) == (h, w):
        return rgb
    im = Image.fromarray(rgb).resize((new_w, new_h), Image.LANCZOS)
    return np.asarray(im, dtype=np.uint8)


def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--limit", type=int, default=0,
                    help="взять только первые N исходников (0 = все)")
    args = ap.parse_args()

    config = load_config(Path(args.config))
    prompts = load_prompts()
    matrix = build_prompt_matrix(config, prompts)
    gcfg = config["generation"]

    sources = gdal_io.list_input_images(
        ROOT / config["dataset"]["input_dir"],
        config["dataset"].get("images") or None,
    )
    if args.limit > 0:
        sources = sources[: args.limit]

    generator = get_generator(config)
    if hasattr(generator, "prepare_prompts"):
        texts = [t for item in matrix
                 for t in (item.prompt, item.negative_prompt) if t]
        generator.prepare_prompts(texts)
    run_dir = make_run_dir(ROOT / config["output"]["generated_dir"])

    gdal_io.copy_file(args.config, run_dir / "run_config.yaml")
    gdal_io.copy_file(ROOT / "prompts" / "prompts.yaml", run_dir / "prompts.yaml")

    variants = int(gcfg.get("num_variants", 1))
    total = len(sources) * len(matrix) * variants
    print(f"Backend: {generator.name} | исходников: {len(sources)} | "
          f"промптов: {len(matrix)} | вариантов: {variants} | "
          f"всего генераций: {total}")
    print(f"Папка прогона: {run_dir}")

    meta_path = run_dir / "metadata.jsonl"
    done = 0
    t_run = time.perf_counter()
    with meta_path.open("w", encoding="utf-8") as meta_f:
        for src_path in sources:
            rgb, io_meta = gdal_io.read_image(src_path)
            rgb = resize_to_resolution(rgb, int(gcfg["resolution"]))
            for item, variant in itertools.product(matrix, range(variants)):
                seed = int(gcfg["seed"]) + variant
                req = GenRequest(
                    source_name=src_path.name,
                    season=item.season,
                    ptype=item.ptype,
                    prompt=item.prompt,
                    negative_prompt=item.negative_prompt,
                    seed=seed,
                )
                t0 = time.perf_counter()
                out_rgb = generator.generate(rgb, req)
                gen_time = time.perf_counter() - t0

                out_name = (f"{sanitize(src_path.stem)}__{item.season}__"
                            f"{item.ptype}__v{variant}.png")
                gdal_io.write_image(run_dir / out_name, out_rgb, io_meta)

                record = {
                    "run_id": run_dir.name,
                    "source": src_path.name,
                    "season": item.season,
                    "ptype": item.ptype,
                    "variant": variant,
                    "prompt": item.prompt,
                    "negative_prompt": item.negative_prompt,
                    "seed": seed,
                    "steps": int(gcfg["num_inference_steps"]),
                    "guidance_scale": float(gcfg["guidance_scale"]),
                    "true_cfg_scale": (float(gcfg["true_cfg_scale"])
                                       if item.negative_prompt else None),
                    "backend": generator.name,
                    "model_id": generator.model_id,
                    "dtype": gcfg["dtype"],
                    "resolution": list(out_rgb.shape[:2]),
                    "out_file": out_name,
                    "gen_time_sec": round(gen_time, 3),
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                }
                meta_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                done += 1
                print(f"  [{done}/{total}] {out_name} ({gen_time:.2f}s)")

    print(f"Готово: {done} изображений за {time.perf_counter() - t_run:.1f}s -> {run_dir}")


if __name__ == "__main__":
    main()
