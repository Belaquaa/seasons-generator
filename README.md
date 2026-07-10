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
                     evaluate, validate, contact_sheets, compare,
                     dataset, build_dataset)
outputs/generated/   результаты исследования + metadata.jsonl (снапшот конфига)
outputs/evaluation/  таблицы метрик, сводки, контактные листы
outputs/dataset/     собранный обучающий набор (build_dataset)
results/             сохранённые итоги всех прогонов + INDEX.md
scripts/             download_levir.py, make_colab_bundle.py, make_colab_notebook.py,
                     example_build_dataset.sh
```

## Установка

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# torch — под свою среду (см. комментарии в requirements.txt)
# GDAL — системно: sudo apt install gdal-bin libgdal-dev python3-gdal
# apt-сборка GDAL <= 3.8 совместима только с numpy 1.x: при ошибке
# "numpy.core.multiarray failed to import" — pip install "numpy<2"
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

## Сборка обучающего набора

Исследование (пункты выше) выбирает лучший промпт. Прикладной итог — конвейер
`src/build_dataset.py`, который этим промптом собирает набор для обучения:

- `src/dataset.py` — простой класс `Dataset`: все изображения из указанной
  директории, ленивое чтение, итерация по элементам с полем `image`. Контракт
  минимален специально — класс можно заменить реальным датасетом проекта и
  запустить тот же конвейер.
- `src/build_dataset.py` — по каждому исходнику генерирует по несколько вариантов
  каждого сезона, оставляет только корректные (сцена сохранена и сезон выражен по
  тем же метрикам, что и в исследовании) и раскладывает результат:

```
outputs/dataset/<исходник>/params.json          параметры выборки
outputs/dataset/<исходник>/<сезон>/*.png         корректные варианты сезона
```

Пример запуска (без GPU, backend из config.yaml):

```bash
bash scripts/example_build_dataset.sh      # Linux / Colab / Git Bash
```

Полный запуск с параметрами:

```bash
python src/build_dataset.py --per-season 5 --ptype kontext_guide \
       --output-dir outputs/dataset
```

Отбор корректных генераций всегда использует модели метрик (DINOv3 + CLIP);
для реальной генерации в `config.yaml` нужен `generation.backend: flux`.

## Быстрое воспроизведение на GPU (Colab, одна ячейка)

Проверенный рецепт (Colab Pro+, GPU G4/A100, прогон 10.07.2026 — итог в
`results/params_flux_example.json`). Нужен секрет `HF_TOKEN` в Colab Secrets
с принятой лицензией FLUX.1 [dev]:

```python
%cd /content
!rm -rf proj
!git clone -q https://github.com/Belaquaa/seasons-generator.git proj
%cd proj
!pip install -q -U diffusers transformers accelerate safetensors sentencepiece timm scikit-image

from google.colab import userdata
from huggingface_hub import login, hf_hub_download
login(token=userdata.get('HF_TOKEN'))

import yaml, zipfile, pathlib
cfg = yaml.safe_load(open('config.yaml', encoding='utf-8'))
cfg['generation'].update(backend='flux', dtype='bf16', cpu_offload='none')
cfg['evaluation']['device'] = 'cuda'
yaml.safe_dump(cfg, open('config.yaml', 'w', encoding='utf-8'),
               allow_unicode=True, sort_keys=False)

zp = hf_hub_download('satellite-image-deep-learning/LEVIR-CD', 'val.zip',
                     repo_type='dataset')
pathlib.Path('data/input').mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(zp) as z:
    a = sorted([n for n in z.namelist()
                if n.startswith('A/') and n.endswith('.png')],
               key=lambda n: int(''.join(c for c in n if c.isdigit())))
    for name in a[::max(1, len(a) // 10)][:2]:
        pathlib.Path('data/input', 'levir_' + pathlib.Path(name).name).write_bytes(z.read(name))

!python src/build_dataset.py --limit 1 --per-season 2 --output-dir outputs/dataset

p = sorted(pathlib.Path('outputs/dataset').glob('*/params.json'))[0]
print(p.read_text(encoding='utf-8'))
```

Время: ~10 мин с нуля (скачивание модели ~34 GB) или ~3 мин на прогретом
рантайме. Без GPU то же самое проверяется mock-backend'ом:
`python tests/smoke_test.py`, затем `bash scripts/example_build_dataset.sh`.

## Полностью офлайн (сервер без интернета, свои модели)

Из сети конвейер берёт только три модели (имена — в `config.yaml`), всё
остальное локально. Код не меняется: модели один раз скачиваются в HF-кеш
на машине с интернетом, кеш переносится на сервер, оффлайн-режим включается
переменными окружения (проверено: с `HF_HUB_OFFLINE=1` конвейер работает
без единого обращения в сеть).

На машине с интернетом:

```bash
export HF_HOME=/models/hf-cache
huggingface-cli download black-forest-labs/FLUX.1-Kontext-dev   # ~34 GB, нужна принятая лицензия
huggingface-cli download timm/vit_small_plus_patch16_dinov3.lvd1689m
huggingface-cli download openai/clip-vit-base-patch16
# перенести /models/hf-cache на сервер (диск/rsync)
```

На сервере (без интернета):

```bash
export HF_HOME=/models/hf-cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# свои снимки — в data/input/ (или любой каталог через --input-dir)
python src/build_dataset.py --input-dir /data/my_images --per-season 5 \
       --output-dir outputs/dataset
```

Если модели на сервере лежат не HF-кешем, а обычными папками-снапшотами,
для FLUX и CLIP можно указать пути прямо в `config.yaml`
(`generation.model_id: /models/flux-kontext`,
`evaluation.season_model: /models/clip-vit-b16`); модель структуры (timm)
загружается только через HF-кеш — для нее используйте вариант с `HF_HOME`.

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
