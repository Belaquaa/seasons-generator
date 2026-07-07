"""Загрузка подвыборки LEVIR-CD (val/A, 10 снимков) в data/input."""
from __future__ import annotations

import argparse
import tempfile
import urllib.request
from pathlib import Path
from zipfile import ZipFile

URL = ("https://huggingface.co/datasets/satellite-image-deep-learning/"
       "LEVIR-CD/resolve/main/val.zip")
ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "data" / "input"
N_PICKS = 10


def pick_names(a_files: list[str]) -> list[str]:
    a_sorted = sorted(a_files,
                      key=lambda n: int("".join(c for c in n if c.isdigit())))
    step = max(1, len(a_sorted) // N_PICKS)
    return a_sorted[::step][:N_PICKS]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keep-zip", action="store_true",
                    help="не удалять скачанный val.zip")
    ap.add_argument("--force", action="store_true",
                    help="перезаписать уже существующие файлы")
    args = ap.parse_args()

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        zip_path = Path(td) / "val.zip"
        print(f"Скачивание {URL}\n -> {zip_path}")
        urllib.request.urlretrieve(URL, zip_path)

        with ZipFile(zip_path) as z:
            a_files = [n for n in z.namelist()
                       if n.startswith("A/") and n.endswith(".png")]
            picks = pick_names(a_files)
            for name in picks:
                dest = INPUT_DIR / f"levir_{Path(name).name}"
                if dest.exists() and not args.force:
                    print(f"  пропуск (уже есть): {dest.name}")
                    continue
                dest.write_bytes(z.read(name))
                print(f"  -> {dest.name}")

        if args.keep_zip:
            kept = ROOT / "data" / "val.zip"
            zip_path.replace(kept)
            print(f"Архив сохранён: {kept}")
    print(f"Готово: {INPUT_DIR}")


if __name__ == "__main__":
    main()
