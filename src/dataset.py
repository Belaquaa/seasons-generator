"""Простой датасет изображений из директории.

Контракт намеренно минимален — длина плюс итерация по элементам с полем
`image` (RGB uint8). Этого достаточно, чтобы позже подменить этот класс
реальным датасетом проекта и запустить тот же конвейер сборки."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import gdal_io


@dataclass(frozen=True)
class DatasetItem:
    name: str          # имя файла с расширением
    stem: str          # имя без расширения = имя подпапки результата
    image: np.ndarray  # RGB uint8 (H, W, 3)
    io_meta: dict      # гео-привязка исходника, переносится в результаты


class Dataset:
    """Все изображения из директории. Чтение ленивое: на каждой итерации
    читается очередной снимок, поэтому большой набор не держится в памяти.
    Пути читаются через GDAL VSI, поэтому директория может быть и /vsizip/,
    /vsis3/, /vsicurl/."""

    def __init__(self, input_dir: str | Path, names: list[str] | None = None):
        self.input_dir = Path(input_dir)
        self.paths = gdal_io.list_input_images(self.input_dir, names)

    def __len__(self) -> int:
        return len(self.paths)

    def _load(self, path: Path) -> DatasetItem:
        rgb, meta = gdal_io.read_image(path)
        return DatasetItem(path.name, path.stem, rgb, meta)

    def __getitem__(self, idx: int) -> DatasetItem:
        return self._load(self.paths[idx])

    def __iter__(self):
        for path in self.paths:
            yield self._load(path)
