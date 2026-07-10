"""Чтение/запись снимков в RGB uint8 (H, W, 3) и файловые операции.

Вся работа с файловой системой (обход директорий, проверка существования,
создание папок, растровый I/O) идёт через GDAL VSI, поэтому пути вида
/vsizip/, /vsis3/, /vsicurl/, /vsimem/ работают наравне с локальными.
Без установленного GDAL — прозрачный фолбэк на pathlib/Pillow."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

try:
    from osgeo import gdal  # type: ignore
    gdal.UseExceptions()
    HAS_GDAL = True
except ImportError:
    HAS_GDAL = False

SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _vsi_stat(p: str):
    """gdal.VSIStatL с проглатыванием исключений: отсутствие пути = None,
    а не ошибка (под gdal.UseExceptions поведение stat не гарантировано)."""
    try:
        return gdal.VSIStatL(p)
    except RuntimeError:
        return None


def list_dir(path: str | Path) -> list[str]:
    """Имена элементов директории (VSI-совместимо)."""
    p = os.fspath(path)
    if HAS_GDAL:
        entries = gdal.ReadDir(p)
        if entries is None:
            raise FileNotFoundError(f"Нет директории: {p}")
        return [e for e in entries if e not in (".", "..")]
    pp = Path(p)
    if not pp.is_dir():
        raise FileNotFoundError(f"Нет директории: {p}")
    return [c.name for c in pp.iterdir()]


def file_exists(path: str | Path) -> bool:
    """Существует ли путь как файл (VSI-совместимо)."""
    p = os.fspath(path)
    if HAS_GDAL:
        st = _vsi_stat(p)
        return st is not None and not st.IsDirectory()
    return Path(p).is_file()


def dir_exists(path: str | Path) -> bool:
    """Существует ли путь как директория (VSI-совместимо)."""
    p = os.fspath(path)
    if HAS_GDAL:
        st = _vsi_stat(p)
        return st is not None and st.IsDirectory()
    return Path(p).is_dir()


def makedirs(path: str | Path) -> None:
    """Создать директорию рекурсивно (VSI-совместимо), не падая на существующей.
    Если путь занят обычным файлом — MkdirRecursive завершится ошибкой (fail-fast,
    как pathlib), а не молча пропустит создание."""
    p = os.fspath(path)
    if HAS_GDAL:
        if not dir_exists(p):
            gdal.MkdirRecursive(p, 0o755)
    else:
        Path(p).mkdir(parents=True, exist_ok=True)


def copy_file(src: str | Path, dst: str | Path) -> None:
    """Скопировать файл (VSI-совместимо: src/dst могут быть /vsi-путями)."""
    s, d = os.fspath(src), os.fspath(dst)
    if HAS_GDAL and hasattr(gdal, "CopyFile"):
        makedirs(Path(d).parent)
        gdal.CopyFile(s, d)
    else:
        import shutil
        Path(d).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, d)


def write_text(path: str | Path, text: str, encoding: str = "utf-8") -> None:
    """Записать текстовый файл (VSI-совместимо), создав родительские папки."""
    p = os.fspath(path)
    makedirs(Path(p).parent)
    if HAS_GDAL:
        data = text.encode(encoding)
        f = gdal.VSIFOpenL(p, "wb")
        if f is None:
            raise OSError(f"Не удалось открыть для записи: {p}")
        try:
            written = gdal.VSIFWriteL(data, 1, len(data), f)
        finally:
            closed = gdal.VSIFCloseL(f)
        if written != len(data) or closed != 0:
            raise OSError(f"Неполная запись {p}: {written}/{len(data)} байт")
    else:
        Path(p).write_text(text, encoding=encoding)


def _to_uint8(band: np.ndarray) -> np.ndarray:
    if band.dtype == np.uint8:
        return band
    band = band.astype(np.float64)
    lo, hi = float(band.min()), float(band.max())
    if hi <= lo:
        return np.zeros(band.shape, dtype=np.uint8)
    return (((band - lo) / (hi - lo)) * 255.0).round().astype(np.uint8)


def _read_gdal(path: Path) -> tuple[np.ndarray, dict]:
    ds = gdal.Open(str(path))
    arr = ds.ReadAsArray()  # GDAL отдаёт band-first (bands, H, W), не H, W, C
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=0)
    if arr.shape[0] > 3:
        arr = arr[:3]
    elif arr.shape[0] == 2:
        arr = np.concatenate([arr, arr[:1]], axis=0)
    rgb = np.stack([_to_uint8(b) for b in arr], axis=-1)
    meta = {
        "reader": "gdal",
        "geotransform": ds.GetGeoTransform(can_return_null=True),
        "projection": ds.GetProjection() or None,
        "source_driver": ds.GetDriver().ShortName,
    }
    ds = None
    return rgb, meta


def _read_pillow(path: Path) -> tuple[np.ndarray, dict]:
    from PIL import Image

    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
    return rgb, {"reader": "pillow", "geotransform": None, "projection": None}


def read_image(path: str | Path) -> tuple[np.ndarray, dict]:
    path = Path(path)
    if not file_exists(path):
        raise FileNotFoundError(f"Нет файла: {path}")
    if path.suffix.lower() not in SUPPORTED_EXT:
        raise ValueError(f"Неподдерживаемый формат {path.suffix!r}: {path}")
    if HAS_GDAL:
        return _read_gdal(path)
    return _read_pillow(path)


def _write_gdal(path: Path, rgb: np.ndarray, meta: dict) -> None:
    ext = path.suffix.lower()
    driver_name = "GTiff" if ext in {".tif", ".tiff"} else "PNG"
    h, w, _ = rgb.shape
    if driver_name == "GTiff":
        ds = gdal.GetDriverByName("GTiff").Create(str(path), w, h, 3, gdal.GDT_Byte)
        if meta.get("geotransform"):
            ds.SetGeoTransform(meta["geotransform"])
        if meta.get("projection"):
            ds.SetProjection(meta["projection"])
        for i in range(3):
            ds.GetRasterBand(i + 1).WriteArray(rgb[:, :, i])
        ds.FlushCache()
        ds = None
    else:
        # PNG-драйвер GDAL не умеет Create — идём через MEM + CreateCopy
        mem = gdal.GetDriverByName("MEM").Create("", w, h, 3, gdal.GDT_Byte)
        for i in range(3):
            mem.GetRasterBand(i + 1).WriteArray(rgb[:, :, i])
        gdal.GetDriverByName("PNG").CreateCopy(str(path), mem)
        mem = None


def write_image(path: str | Path, rgb: np.ndarray, meta: dict | None = None) -> Path:
    path = Path(path)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"Ожидается uint8 (H, W, 3), получено {rgb.dtype} {rgb.shape}")
    makedirs(path.parent)
    meta = meta or {}
    if HAS_GDAL:
        _write_gdal(path, rgb, meta)
    else:
        from PIL import Image

        if path.suffix.lower() in {".tif", ".tiff"} and meta.get("geotransform"):
            raise RuntimeError(
                "Гео-привязку без GDAL не сохранить — установите GDAL "
                "(sudo apt install python3-gdal)"
            )
        Image.fromarray(rgb).save(path)
    return path


def list_input_images(input_dir: str | Path, names: list[str] | None = None) -> list[Path]:
    input_dir = Path(input_dir)
    if names:
        paths = [input_dir / n for n in names]
        missing = [str(p) for p in paths if not file_exists(p)]
        if missing:
            raise FileNotFoundError(f"Не найдены в {input_dir}: {missing}")
        return paths
    paths = sorted(
        input_dir / name for name in list_dir(input_dir)
        if Path(name).suffix.lower() in SUPPORTED_EXT
        and file_exists(input_dir / name)
    )
    if not paths:
        raise FileNotFoundError(f"В {input_dir} нет изображений ({sorted(SUPPORTED_EXT)})")
    return paths
