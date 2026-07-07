# Сезонная генерация снимков и оценка промптов

Практическая часть: из исходного аэро/спутникового снимка генерируются сезонные
варианты (лето / осень / зима / глубокая зима) редактором **FLUX.1-Kontext-dev**,
результаты сравниваются по метрикам, выбирается лучший промпт. Полное описание
работы, методика оценки и результаты — в [`practice_report.md`](practice_report.md).

## Ключевая идея архитектуры

Генерация — единственный шаг, которому нужен GPU. Она спрятана за единым
интерфейсом с backend'ами:

- `mock` — быстрый заглушечный преобразователь, **GPU не нужен**. Для разработки и
  теста всего конвейера end-to-end.
- `flux` — FLUX.1-Kontext-dev, основная модель (bf16, полная точность).
- `flux2` — FLUX.2-dev 32B (сравнение моделей).
- `sdxl`, `ip2p` — SDXL img2img и InstructPix2Pix (бейзлайны сравнения).

Всё остальное (GDAL I/O, промпты, метрики CLIP/DINOv3, сравнение) работает на CPU.

## Структура

```
data/input/          исходники (before-снимки LEVIR-CD val)
prompts/prompts.yaml промпты: 10 типов x 4 сезона (версия v4)
config.yaml          параметры прогона и оценки
src/                 код (prompts, gdal_io, model_flux, generate, metrics,
                     evaluate, validate, contact_sheets, compare)
outputs/generated/   результаты + metadata.jsonl (снапшот конфига в папке прогона)
outputs/evaluation/  таблицы метрик, сводки, контактные листы
results/             сохранённые итоги всех прогонов + INDEX.md
scripts/             download_levir.py, make_colab_bundle.py, make_colab_notebook.py
```

## Установка

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# torch — под свою среду (см. комментарии в requirements.txt)
# GDAL — системно: sudo apt install gdal-bin libgdal-dev python3-gdal
```

Для `backend: flux` дополнительно нужен HF-логин и принятая лицензия FLUX.1 [dev].

## Порядок работы

1. Проверить матрицу промптов: `python src/prompts.py`
2. Скачать подвыборку LEVIR-CD: `python scripts/download_levir.py`
   (происхождение и лицензия данных — `data/SOURCE.md`)
3. Прогон генерации: `python src/generate_images.py [--limit N]`
   (backend в config.yaml; результаты в `outputs/generated/{run_id}/`)
4. Оценка: `python src/evaluate_results.py [--run <папка>]`
   (метрики: DINOv3 + SSIM — структура, CLIP zero-shot + directional — сезон;
   CSV в `outputs/evaluation/{run_id}_results.csv`)
5. Валидация и наглядность: `python src/validate_run.py`,
   `python src/contact_sheets.py`
6. Сравнение и выбор: `python src/compare_prompts.py`
   (рейтинг типов промптов + `{run_id}_summary.md` с обоснованием лучшего)
7. Реальный FLUX — через Google Colab: самодостаточный ноутбук
   `notebooks/colab_flux_auto.ipynb` (генерируется
   `python scripts/make_colab_notebook.py`, код и промпты зашиты в ячейки,
   данные скачиваются с HF; нужна принятая лицензия FLUX.1 [dev]);
   скачанные пакеты результатов складываются в `results/`
8. Перенести результаты в отчёт (пункты 7-9)

## Запуск на GPU-сервере (A100 и подобные)

Проверенная конфигурация (реальный прогон: 320 генераций bf16 + GPU-оценка):

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
huggingface-cli login          # токен с принятой лицензией FLUX.1 [dev]
python scripts/download_levir.py
```

В `config.yaml` для GPU >= 40GB (A100):

```yaml
generation:
  backend: flux
  dtype: bf16                  # полная модель, без квантизации
  cpu_offload: none
evaluation:
  device: cuda                 # метрики на GPU в разы быстрее
```

Для 16GB GPU (T4): `dtype: gguf-q4`, `cpu_offload: none` (проверено).
Порядок: generate -> evaluate -> validate -> contact_sheets -> compare.
Seed фиксирован (42), параметры прогона снапшотятся в папку результата;
между разными архитектурами GPU возможны небольшие численные расхождения
генерации - статистические выводы воспроизводятся.

## Тесты

```bash
python tests/smoke_test.py   # матрица, I/O (GDAL/Pillow), mock, GeoTIFF
```
