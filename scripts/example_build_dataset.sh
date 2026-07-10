#!/usr/bin/env bash
# Пример запуска конвейера сборки обучающего набора (src/build_dataset.py).
# Для каждого исходника собирает по несколько корректных вариантов каждого
# сезона и раскладывает по подпапкам с params.json.
#
# Backend генерации берётся из config.yaml (mock = без GPU). Отбор корректных
# генераций всегда использует модели метрик (DINOv3 + CLIP) — при первом
# запуске они скачиваются с Hugging Face.
#
# Запуск из корня проекта:
#   bash scripts/example_build_dataset.sh
set -euo pipefail
cd "$(dirname "$0")/.."

python src/build_dataset.py \
  --limit 1 \
  --per-season 2 \
  --ptype kontext_guide \
  --output-dir outputs/dataset_example

echo
echo "Результат: outputs/dataset_example/<исходник>/<сезон>/*.png + params.json"
echo "Реальная генерация: в config.yaml выставить generation.backend: flux (нужен GPU + лицензия FLUX.1)."
