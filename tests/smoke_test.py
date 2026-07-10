"""Смоук-тесты конвейера (без GPU, без pytest — чистые asserts)."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import gdal_io  # noqa: E402
from model_flux import GenRequest, MockGenerator, get_generator  # noqa: E402
from prompts import build_prompt_matrix, load_config, load_prompts  # noqa: E402

SAMPLE = ROOT / "report_assets" / "season_source.png"


def test_prompt_matrix() -> None:
    config, prompts = load_config(), load_prompts()
    matrix = build_prompt_matrix(config, prompts)
    n_expected = len(config["seasons"]) * len(config["prompt_types"])
    assert len(matrix) == n_expected, f"{len(matrix)} != {n_expected}"
    combos = {(it.season, it.ptype) for it in matrix}
    assert len(combos) == n_expected, "дубликаты комбинаций"
    for it in matrix:
        assert it.prompt.strip(), f"пустой промпт: {it.season}/{it.ptype}"
        assert "{" not in it.prompt, f"неподставленный плейсхолдер: {it.prompt}"
        has_neg = bool(prompts["prompt_types"][it.ptype].get("has_negative"))
        if has_neg:
            assert it.negative_prompt, f"у {it.ptype} нет negative_prompt"
        else:
            assert it.negative_prompt is None, f"{it.ptype} имеет negative_prompt"
    print(f"prompt_matrix: {n_expected} промптов OK")


def test_io_roundtrip() -> None:
    rgb, meta = gdal_io.read_image(SAMPLE)
    assert rgb.ndim == 3 and rgb.shape[2] == 3 and rgb.dtype == np.uint8
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "rt.png"
        gdal_io.write_image(out, rgb, meta)
        rgb2, _ = gdal_io.read_image(out)
    assert np.array_equal(rgb, rgb2), "roundtrip не сошёлся"
    print(f"io_roundtrip ({meta['reader']}): OK")


def test_io_errors() -> None:
    try:
        gdal_io.read_image(ROOT / "report_assets" / "no_such.png")
        raise AssertionError("не упал на отсутствующем файле")
    except FileNotFoundError:
        pass
    try:
        gdal_io.read_image(ROOT / "urls.txt")
        raise AssertionError("не упал на неподдерживаемом формате")
    except ValueError:
        pass
    print("io_errors: OK")


def test_mock_generator() -> None:
    rgb, _ = gdal_io.read_image(SAMPLE)
    gen = MockGenerator()

    def req(season: str, ptype: str = "contextual") -> GenRequest:
        return GenRequest("s.png", season, ptype, "p", None, 42)

    a, b = gen.generate(rgb, req("winter")), gen.generate(rgb, req("winter"))
    assert np.array_equal(a, b), "mock не детерминирован"
    assert a.shape == rgb.shape and a.dtype == np.uint8
    assert not np.array_equal(a, rgb), "mock не изменил изображение"

    outs = {s: gen.generate(rgb, req(s))
            for s in ["summer", "autumn", "winter", "deep_winter"]}
    keys = list(outs)
    for i, s1 in enumerate(keys):
        for s2 in keys[i + 1:]:
            assert not np.array_equal(outs[s1], outs[s2]), f"{s1} == {s2}"
    means = {s: float(o.mean()) for s, o in outs.items()}
    assert means["deep_winter"] > means["winter"] > means["summer"], \
        f"порядок яркости нарушен: {means}"

    o1 = gen.generate(rgb, req("winter", "zero_shot"))
    o2 = gen.generate(rgb, req("winter", "cot"))
    assert not np.array_equal(o1, o2), "разные ptype дали одинаковый результат"
    print("mock_generator: детерминизм, сезоны, типы OK")


def test_factory() -> None:
    g = get_generator({"generation": {"backend": "mock"}})
    assert g.name == "mock"
    try:
        get_generator({"generation": {"backend": "xyz"}})
        raise AssertionError("не упал на неизвестном backend")
    except ValueError:
        pass
    print("factory: OK")


def test_fs_helpers() -> None:
    rgb, _ = gdal_io.read_image(SAMPLE)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        nested = root / "a" / "b"
        gdal_io.makedirs(nested)
        assert gdal_io.dir_exists(nested) and not gdal_io.file_exists(nested)

        gdal_io.write_image(nested / "img.png", rgb)
        assert gdal_io.file_exists(nested / "img.png")
        assert not gdal_io.dir_exists(nested / "img.png")
        assert "img.png" in gdal_io.list_dir(nested)

        gdal_io.write_text(root / "note.txt", "привет")
        assert gdal_io.file_exists(root / "note.txt")
        assert (root / "note.txt").read_text("utf-8") == "привет"

        gdal_io.copy_file(root / "note.txt", root / "c" / "copy.txt")
        assert (root / "c" / "copy.txt").read_text("utf-8") == "привет"

        # относительный путь с несуществующим корнем (MkdirRecursive ломается
        # на таких без абсолютизации — воспроизведено на Colab)
        cwd = os.getcwd()
        try:
            os.chdir(root)
            gdal_io.makedirs("rel_a/rel_b")
            assert gdal_io.dir_exists(root / "rel_a" / "rel_b")
        finally:
            os.chdir(cwd)
    print("fs_helpers: list_dir/exists/makedirs(отн. путь)/copy/write_text OK")


def test_dataset() -> None:
    from dataset import Dataset

    rgb, _ = gdal_io.read_image(SAMPLE)
    with tempfile.TemporaryDirectory() as td:
        in_dir = Path(td) / "in"
        for name in ("levir_2.png", "levir_1.png"):
            gdal_io.write_image(in_dir / name, rgb)
        (in_dir / "readme.txt").write_text("not an image", encoding="utf-8")

        ds = Dataset(in_dir)
        assert len(ds) == 2, f"ожидалось 2 изображения, найдено {len(ds)}"
        stems = [item.stem for item in ds]
        assert stems == ["levir_1", "levir_2"], f"нет сортировки: {stems}"
        first = ds[0]
        assert first.image.ndim == 3 and first.image.shape[2] == 3
        assert first.name == "levir_1.png"
    print("dataset: len, сортировка, итерация, __getitem__ OK")


def test_geotiff() -> None:
    if not gdal_io.HAS_GDAL:
        print("geotiff: SKIPPED (GDAL не установлен, работает Pillow-фолбэк)")
        return
    from osgeo import gdal, osr

    gt = (400000.0, 0.5, 0.0, 6200000.0, 0.0, -0.5)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32637)  # UTM 37N
    wkt = srs.ExportToWkt()

    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "synthetic.tif"

        ds = gdal.GetDriverByName("GTiff").Create(str(src), 64, 48, 3,
                                                  gdal.GDT_UInt16)
        ds.SetGeoTransform(gt)
        ds.SetProjection(wkt)
        rng = np.random.default_rng(7)
        for i in range(3):
            band = rng.integers(0, 10000, size=(48, 64)).astype(np.uint16)
            ds.GetRasterBand(i + 1).WriteArray(band)
        ds.FlushCache()
        ds = None

        rgb, meta = gdal_io.read_image(src)
        assert rgb.shape == (48, 64, 3) and rgb.dtype == np.uint8
        assert meta["reader"] == "gdal"
        assert meta["geotransform"] == gt, f"geotransform потерян: {meta['geotransform']}"
        assert meta["projection"] and "32637" in meta["projection"]

        out = Path(td) / "out.tif"
        gdal_io.write_image(out, rgb, meta)
        ds2 = gdal.Open(str(out))
        assert ds2.GetGeoTransform() == gt, "geotransform не сохранился при записи"
        assert "32637" in ds2.GetProjection(), "проекция не сохранилась при записи"
        back = np.stack([ds2.GetRasterBand(i + 1).ReadAsArray()
                         for i in range(3)], axis=-1)
        assert np.array_equal(back, rgb), "пиксели GeoTIFF не сошлись"
        ds2 = None

        out_png = Path(td) / "out.png"
        gdal_io.write_image(out_png, rgb, meta)
        rgb2, _ = gdal_io.read_image(out_png)
        assert np.array_equal(rgb2, rgb), "PNG через GDAL не сошёлся"
    print("geotiff (GDAL): гео-привязка, uint16->uint8, GTiff/PNG запись OK")


if __name__ == "__main__":
    test_prompt_matrix()
    test_io_roundtrip()
    test_io_errors()
    test_mock_generator()
    test_factory()
    test_fs_helpers()
    test_dataset()
    test_geotiff()
    print("\nALL SMOKE TESTS PASSED")
