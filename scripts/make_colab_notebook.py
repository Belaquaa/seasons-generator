"""Генерация самодостаточного Colab-ноутбука notebooks/colab_flux_auto.ipynb."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "colab_flux_auto.ipynb"

EMBED = [
    "config.yaml",
    "prompts/prompts.yaml",
    "src/prompts.py",
    "src/gdal_io.py",
    "src/model_flux.py",
    "src/generate_images.py",
]


def code_cell(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(keepends=True)}


def md_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": source.splitlines(keepends=True)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--token", default="", help="HF-токен (зашивается в ячейку)")
    args = ap.parse_args()

    cells = [
        md_cell(
            "# FLUX.1-Kontext-dev: сезонная генерация (self-contained)\n\n"
            "Код и конфиги зашиты в ячейки, данные LEVIR скачиваются с HF.\n"
            "Порядок: выполнить ячейки сверху вниз; после smoke глазами\n"
            "проверить результат, затем полный прогон.\n\n"
            "Runtime -> Change runtime type -> **A100** (или L4)."
        ),
        code_cell(
            "import torch\n"
            "assert torch.cuda.is_available(), 'GPU не выдан'\n"
            "gpu = torch.cuda.get_device_name(0)\n"
            "vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3\n"
            "if vram_gb >= 35:\n"
            "    DTYPE, OFFLOAD = 'bf16', 'none'\n"
            "elif vram_gb >= 20:\n"
            "    DTYPE, OFFLOAD = 'fp8', 'model'\n"
            "else:\n"
            "    # T4: fp8/offload умирает по RAM (12.7GB); проверенный путь -\n"
            "    # GGUF Q4 + T5-8bit целиком в VRAM (см. model_flux gguf-q4)\n"
            "    DTYPE, OFFLOAD = 'gguf-q4', 'none'\n"
            "print(f'{gpu} | {vram_gb:.0f} GB -> dtype={DTYPE}, offload={OFFLOAD}')"
        ),
        code_cell("import pathlib\n"
                  "for d in ['src', 'prompts', 'data/input', 'outputs/generated']:\n"
                  "    pathlib.Path(d).mkdir(parents=True, exist_ok=True)\n"
                  "print('структура создана')"),
    ]

    for rel in EMBED:
        text = (ROOT / rel).read_text(encoding="utf-8")
        cells.append(code_cell(f"%%writefile {rel}\n{text}"))

    cells += [
        code_cell(
            "!pip install -q -U diffusers transformers accelerate safetensors "
            "sentencepiece optimum-quanto gguf bitsandbytes\n"
            "import diffusers\n"
            "print('diffusers', diffusers.__version__)"
        ),
        code_cell(
            f"HF_TOKEN = '{args.token}'  # после практики токен отозвать\n"
            "from huggingface_hub import login\n"
            "login(token=HF_TOKEN)\n"
            "print('HF login OK')"
        ),
        code_cell(
            "# Данные: те же 10 снимков val/A LEVIR-CD, что и локально\n"
            "import zipfile\n"
            "from pathlib import Path\n"
            "from huggingface_hub import hf_hub_download\n"
            "zp = hf_hub_download('satellite-image-deep-learning/LEVIR-CD',\n"
            "                     'val.zip', repo_type='dataset')\n"
            "with zipfile.ZipFile(zp) as z:\n"
            "    a = sorted([n for n in z.namelist()\n"
            "                if n.startswith('A/') and n.endswith('.png')],\n"
            "               key=lambda n: int(''.join(c for c in n if c.isdigit())))\n"
            "    picks = a[::max(1, len(a) // 10)][:10]\n"
            "    for name in picks:\n"
            "        Path('data/input', 'levir_' + Path(name).name).write_bytes(z.read(name))\n"
            "print('исходников:', len(list(Path('data/input').glob('*.png'))))"
        ),
        code_cell(
            "import yaml\n"
            "cfg = yaml.safe_load(open('config.yaml', encoding='utf-8'))\n"
            "cfg['generation'].update(backend='flux', dtype=DTYPE, cpu_offload=OFFLOAD)\n"
            "yaml.safe_dump(cfg, open('config_colab.yaml', 'w', encoding='utf-8'),\n"
            "               allow_unicode=True, sort_keys=False)\n"
            "smoke = {**cfg, 'seasons': ['winter'], 'prompt_types': ['contextual']}\n"
            "yaml.safe_dump(smoke, open('config_smoke.yaml', 'w', encoding='utf-8'),\n"
            "               allow_unicode=True, sort_keys=False)\n"
            "print('конфиги готовы')"
        ),
        code_cell("# SMOKE: 1 исходник x 1 промпт (первый запуск скачает модель ~24GB)\n"
                  "!python src/generate_images.py --config config_smoke.yaml --limit 1"),
        code_cell(
            "import pathlib\n"
            "from IPython.display import display\n"
            "from PIL import Image\n"
            "run = sorted(pathlib.Path('outputs/generated').iterdir())[-1]\n"
            "out = next(run.glob('*.png'))\n"
            "src = sorted(pathlib.Path('data/input').glob('*.png'))[0]\n"
            "print(out.name)\n"
            "display(Image.open(src).resize((384, 384)),\n"
            "        Image.open(out).resize((384, 384)))"
        ),
        md_cell("Если smoke ок — полный прогон (A100: ~1-1.5 ч на 240 генераций)."),
        code_cell("!python src/generate_images.py --config config_colab.yaml"),
        code_cell(
            "import pathlib, shutil\n"
            "from google.colab import files\n"
            "run = sorted(pathlib.Path('outputs/generated').iterdir())[-1]\n"
            "arc = shutil.make_archive(f'flux_results_{run.name}', 'zip',\n"
            "                          run.parent, run.name)\n"
            "print(arc)\n"
            "files.download(arc)"
        ),
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "A100", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"OK: {OUT} ({OUT.stat().st_size // 1024} KB, {len(cells)} ячеек)")


if __name__ == "__main__":
    main()
