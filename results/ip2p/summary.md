# Сравнение промптов

Источник данных: `20260706-153716_results.csv` (320 оценённых изображений).

## Рейтинг типов промптов (по итоговому score)

| ptype         |   n |   mean_score |   mean_structure |   mean_season |   mean_dino |   mean_ssim |   mean_clip_prob |   mean_clip_dir |   n_disqualified |   std_score |
|:--------------|----:|-------------:|-----------------:|--------------:|------------:|------------:|-----------------:|----------------:|-----------------:|------------:|
| negative      |  40 |       0.7398 |           0.7953 |        0.6565 |      0.877  |      0.4684 |           0.6427 |          0.1518 |                0 |      0.0999 |
| positive      |  40 |       0.7382 |           0.8234 |        0.6105 |      0.9068 |      0.4896 |           0.5572 |          0.1358 |                0 |      0.1032 |
| hybrid        |  40 |       0.7366 |           0.8154 |        0.6185 |      0.9023 |      0.4676 |           0.5707 |          0.1474 |                0 |      0.096  |
| contextual    |  40 |       0.7323 |           0.8269 |        0.5903 |      0.9124 |      0.4851 |           0.5209 |          0.1358 |                0 |      0.103  |
| cot           |  40 |       0.7296 |           0.8281 |        0.5819 |      0.9129 |      0.4889 |           0.4988 |          0.1306 |                0 |      0.1042 |
| kontext_guide |  40 |       0.7203 |           0.7784 |        0.633  |      0.8605 |      0.4504 |           0.5951 |          0.1403 |                0 |      0.108  |
| zero_shot     |  40 |       0.7175 |           0.8385 |        0.536  |      0.9246 |      0.4942 |           0.4318 |          0.1171 |                0 |      0.1052 |
| few_shot      |  40 |       0.7164 |           0.8451 |        0.5233 |      0.9335 |      0.4919 |           0.4028 |          0.0826 |                0 |      0.1213 |

## Парное сравнение с лидером (одинаковые исходник/сезон/вариант)

Лидер: **negative**. mean_delta > 0 и малый wilcoxon_p — лидер статистически значимо лучше; p > 0.05 — разница в пределах шума.

| ptype         |   n_pairs |   mean_delta_vs_best |   best_wins_share |   wilcoxon_p |
|:--------------|----------:|---------------------:|------------------:|-------------:|
| positive      |        40 |               0.0015 |             0.425 |       0.9417 |
| hybrid        |        40 |               0.0032 |             0.525 |       0.5627 |
| contextual    |        40 |               0.0075 |             0.45  |       0.9333 |
| cot           |        40 |               0.0101 |             0.5   |       0.3971 |
| kontext_guide |        40 |               0.0195 |             0.725 |       0.0008 |
| zero_shot     |        40 |               0.0223 |             0.675 |       0.0093 |
| few_shot      |        40 |               0.0234 |             0.6   |       0.0078 |

## Score по сезонам

| ptype         |   autumn |   deep_winter |   summer |   winter |
|:--------------|---------:|--------------:|---------:|---------:|
| contextual    |   0.7096 |        0.666  |   0.8858 |   0.6677 |
| cot           |   0.7076 |        0.6592 |   0.8859 |   0.6657 |
| few_shot      |   0.7127 |        0.6433 |   0.8885 |   0.6211 |
| hybrid        |   0.7307 |        0.662  |   0.8758 |   0.6779 |
| kontext_guide |   0.7219 |        0.6534 |   0.8699 |   0.6358 |
| negative      |   0.7581 |        0.6592 |   0.8768 |   0.6649 |
| positive      |   0.726  |        0.6588 |   0.8919 |   0.6762 |
| zero_shot     |   0.6881 |        0.6612 |   0.8714 |   0.6493 |

## Ответы на вопросы сравнения (бизнес-ТЗ §8.6)

- Лучше всего сохраняет исходную сцену: **few_shot** (structure 0.8451)
- Лучше всего передаёт сезон: **negative** (season 0.6565)
- Чаще всего портит сцену/добавляет лишнее: **kontext_guide** (дисквалификаций 0, structure 0.7784)
- Лучший для дальнейшей работы: **negative** (score 0.7398)

## Обоснование выбора

Тип **negative** набрал максимальный средний итоговый score (0.7398) при 0 дисквалификациях. Итоговый score взвешивает сохранение структуры (вес 0.6) и выраженность сезона (вес 0.4); результаты со структурой ниже порога 0.3 исключаются. Числа автоматических метрик необходимо подтвердить ручной проверкой (колонки manual_score/manual_comment в results.csv).

## Тексты промптов лучшего типа (negative)

**summer:**
```text
Change the season of this high-resolution aerial/satellite top-down photo of an urban area to summer: lush green vegetation, green trees and grass, dry bare ground, warm daylight, no snow. Realistic remote-sensing look. keep every road, building, parking lot and field boundary in the exact same position and scale; maintain identical layout, camera angle, framing and perspective.
NEGATIVE: new buildings, new roads, removed or moved structures, changed street layout, different location, distortion, warping, added vehicles or objects, blur, artifacts, text, watermark
```

**autumn:**
```text
Change the season of this high-resolution aerial/satellite top-down photo of an urban area to autumn: yellow and orange foliage, fading vegetation, golden and brown tones, some bare trees, no snow. Realistic remote-sensing look. keep every road, building, parking lot and field boundary in the exact same position and scale; maintain identical layout, camera angle, framing and perspective.
NEGATIVE: new buildings, new roads, removed or moved structures, changed street layout, different location, distortion, warping, added vehicles or objects, blur, artifacts, text, watermark
```

**winter:**
```text
Change the season of this high-resolution aerial/satellite top-down photo of an urban area to winter: light snow cover on the ground and rooftops, cold bluish tones, leafless trees, patches of snow. Realistic remote-sensing look. keep every road, building, parking lot and field boundary in the exact same position and scale; maintain identical layout, camera angle, framing and perspective.
NEGATIVE: new buildings, new roads, removed or moved structures, changed street layout, different location, distortion, warping, added vehicles or objects, blur, artifacts, text, watermark
```

**deep_winter:**
```text
Change the season of this high-resolution aerial/satellite top-down photo of an urban area to deep winter: heavy thick snow covering the entire ground and rooftops, frozen surfaces, strong cold white palette, snow-laden bare trees. Realistic remote-sensing look. keep every road, building, parking lot and field boundary in the exact same position and scale; maintain identical layout, camera angle, framing and perspective.
NEGATIVE: new buildings, new roads, removed or moved structures, changed street layout, different location, distortion, warping, added vehicles or objects, blur, artifacts, text, watermark
```
