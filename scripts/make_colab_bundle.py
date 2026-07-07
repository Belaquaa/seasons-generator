"""Сборка colab_bundle.zip для запуска FLUX-генерации в Google Colab."""
from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "colab_bundle.zip"

INCLUDE = [
    ("src", "*.py"),
    ("prompts", "prompts.yaml"),
    ("data/input", "*"),
]


def main() -> None:
    with ZipFile(OUT, "w", ZIP_DEFLATED) as z:
        z.write(ROOT / "config.yaml", "config.yaml")
        for rel_dir, pattern in INCLUDE:
            src_dir = ROOT / rel_dir
            files = sorted(p for p in src_dir.glob(pattern) if p.is_file()
                           and p.suffix != ".pyc" and p.name != ".gitkeep")
            if not files:
                raise FileNotFoundError(f"Пусто: {src_dir}/{pattern}")
            for p in files:
                z.write(p, f"{rel_dir}/{p.name}")
        names = z.namelist()
    print(f"OK: {OUT} ({OUT.stat().st_size // (1024 * 1024)} MB, "
          f"{len(names)} файлов)")
    for n in names:
        print(" ", n)


if __name__ == "__main__":
    main()
