# Сравнение промптов

Источник данных: `20260706-152810_results.csv` (320 оценённых изображений).

## Рейтинг типов промптов (по итоговому score)

| ptype         |   n |   mean_score |   mean_structure |   mean_season |   mean_dino |   mean_ssim |   mean_clip_prob |   mean_clip_dir |   n_disqualified |   std_score |
|:--------------|----:|-------------:|-----------------:|--------------:|------------:|------------:|-----------------:|----------------:|-----------------:|------------:|
| positive      |  40 |       0.6915 |           0.7186 |        0.6509 |      0.8467 |      0.2061 |           0.6407 |          0.147  |                0 |      0.1035 |
| hybrid        |  40 |       0.6704 |           0.728  |        0.584  |      0.8583 |      0.2067 |           0.516  |          0.1076 |                0 |      0.1181 |
| contextual    |  40 |       0.6681 |           0.7437 |        0.5547 |      0.8777 |      0.2074 |           0.4825 |          0.1024 |                0 |      0.0992 |
| negative      |  40 |       0.6654 |           0.7114 |        0.6186 |      0.8379 |      0.2056 |           0.5732 |          0.1279 |                1 |      0.1602 |
| kontext_guide |  40 |       0.6508 |           0.6704 |        0.6214 |      0.7868 |      0.2045 |           0.5883 |          0.1392 |                0 |      0.118  |
| zero_shot     |  40 |       0.6504 |           0.6822 |        0.6027 |      0.8016 |      0.2049 |           0.5495 |          0.1183 |                0 |      0.1104 |
| cot           |  40 |       0.6496 |           0.7052 |        0.5662 |      0.8315 |      0.2001 |           0.5017 |          0.1224 |                0 |      0.0933 |
| few_shot      |  40 |       0.6127 |           0.711  |        0.4652 |      0.838  |      0.2029 |           0.33   |          0.0442 |                0 |      0.1171 |

## Парное сравнение с лидером (одинаковые исходник/сезон/вариант)

Лидер: **positive**. mean_delta > 0 и малый wilcoxon_p — лидер статистически значимо лучше; p > 0.05 — разница в пределах шума.

| ptype         |   n_pairs |   mean_delta_vs_best |   best_wins_share |   wilcoxon_p |
|:--------------|----------:|---------------------:|------------------:|-------------:|
| hybrid        |        40 |               0.0212 |             0.625 |       0.0438 |
| contextual    |        40 |               0.0235 |             0.725 |       0.0008 |
| negative      |        40 |               0.0261 |             0.55  |       0.1629 |
| kontext_guide |        40 |               0.0407 |             0.7   |       0.0006 |
| zero_shot     |        40 |               0.0411 |             0.575 |       0.0098 |
| cot           |        40 |               0.0419 |             0.85  |       0.0001 |
| few_shot      |        40 |               0.0789 |             0.875 |       0      |

## Score по сезонам

| ptype         |   autumn |   deep_winter |   summer |   winter |
|:--------------|---------:|--------------:|---------:|---------:|
| contextual    |   0.6838 |        0.608  |   0.7965 |   0.584  |
| cot           |   0.633  |        0.6003 |   0.7714 |   0.5938 |
| few_shot      |   0.5999 |        0.5327 |   0.7772 |   0.5408 |
| hybrid        |   0.6592 |        0.5803 |   0.8373 |   0.6047 |
| kontext_guide |   0.6207 |        0.5807 |   0.8016 |   0.6002 |
| negative      |   0.6731 |        0.5347 |   0.8445 |   0.6092 |
| positive      |   0.7179 |        0.602  |   0.8256 |   0.6206 |
| zero_shot     |   0.6243 |        0.6262 |   0.7977 |   0.5534 |

## Ответы на вопросы сравнения (бизнес-ТЗ §8.6)

- Лучше всего сохраняет исходную сцену: **contextual** (structure 0.7437)
- Лучше всего передаёт сезон: **positive** (season 0.6509)
- Чаще всего портит сцену/добавляет лишнее: **negative** (дисквалификаций 1, structure 0.7114)
- Лучший для дальнейшей работы: **positive** (score 0.6915)

## Обоснование выбора

Тип **positive** набрал максимальный средний итоговый score (0.6915) при 0 дисквалификациях. Итоговый score взвешивает сохранение структуры (вес 0.6) и выраженность сезона (вес 0.4); результаты со структурой ниже порога 0.3 исключаются. Числа автоматических метрик необходимо подтвердить ручной проверкой (колонки manual_score/manual_comment в results.csv).

## Тексты промптов лучшего типа (positive)

**summer:**
```text
Change the season of this high-resolution aerial/satellite top-down photo of an urban area to summer: lush green vegetation, green trees and grass, dry bare ground, warm daylight, no snow. Realistic remote-sensing look with consistent natural lighting and colors typical for summer. keep every road, building, parking lot and field boundary in the exact same position and scale; maintain identical layout, camera angle, framing and perspective.
```

**autumn:**
```text
Change the season of this high-resolution aerial/satellite top-down photo of an urban area to autumn: yellow and orange foliage, fading vegetation, golden and brown tones, some bare trees, no snow. Realistic remote-sensing look with consistent natural lighting and colors typical for autumn. keep every road, building, parking lot and field boundary in the exact same position and scale; maintain identical layout, camera angle, framing and perspective.
```

**winter:**
```text
Change the season of this high-resolution aerial/satellite top-down photo of an urban area to winter: light snow cover on the ground and rooftops, cold bluish tones, leafless trees, patches of snow. Realistic remote-sensing look with consistent natural lighting and colors typical for winter. keep every road, building, parking lot and field boundary in the exact same position and scale; maintain identical layout, camera angle, framing and perspective.
```

**deep_winter:**
```text
Change the season of this high-resolution aerial/satellite top-down photo of an urban area to deep winter: heavy thick snow covering the entire ground and rooftops, frozen surfaces, strong cold white palette, snow-laden bare trees. Realistic remote-sensing look with consistent natural lighting and colors typical for deep winter. keep every road, building, parking lot and field boundary in the exact same position and scale; maintain identical layout, camera angle, framing and perspective.
```
