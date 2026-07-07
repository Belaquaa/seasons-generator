# Сравнение промптов

Источник данных: `20260706-131826_results.csv` (320 оценённых изображений).

## Рейтинг типов промптов (по итоговому score)

| ptype         |   n |   mean_score |   mean_structure |   mean_season |   mean_dino |   mean_ssim |   mean_clip_prob |   mean_clip_dir |   n_disqualified |   std_score |
|:--------------|----:|-------------:|-----------------:|--------------:|------------:|------------:|-----------------:|----------------:|-----------------:|------------:|
| kontext_guide |  40 |       0.7289 |           0.7312 |        0.7255 |      0.7914 |      0.4903 |           0.7473 |          0.2358 |                0 |      0.1527 |
| contextual    |  40 |       0.7255 |           0.7202 |        0.7335 |      0.7891 |      0.4443 |           0.7699 |          0.228  |                0 |      0.1434 |
| positive      |  40 |       0.722  |           0.7136 |        0.7346 |      0.7848 |      0.4287 |           0.7691 |          0.2296 |                0 |      0.144  |
| few_shot      |  40 |       0.7073 |           0.7283 |        0.6758 |      0.7959 |      0.4579 |           0.6934 |          0.2344 |                0 |      0.1354 |
| cot           |  40 |       0.7069 |           0.6964 |        0.7226 |      0.7641 |      0.426  |           0.7453 |          0.2288 |                0 |      0.1488 |
| negative      |  40 |       0.6322 |           0.5848 |        0.7312 |      0.6777 |      0.2132 |           0.7644 |          0.2087 |                1 |      0.1789 |
| hybrid        |  40 |       0.6247 |           0.5735 |        0.7308 |      0.6742 |      0.1707 |           0.7646 |          0.2112 |                1 |      0.1682 |
| zero_shot     |  40 |       0.5266 |           0.6295 |        0.6718 |      0.6953 |      0.366  |           0.6732 |          0.1786 |               11 |      0.3502 |

## Парное сравнение с лидером (одинаковые исходник/сезон/вариант)

Лидер: **kontext_guide**. mean_delta > 0 и малый wilcoxon_p — лидер статистически значимо лучше; p > 0.05 — разница в пределах шума.

| ptype      |   n_pairs |   mean_delta_vs_best |   best_wins_share |   wilcoxon_p |
|:-----------|----------:|---------------------:|------------------:|-------------:|
| contextual |        40 |               0.0034 |             0.475 |       0.8931 |
| positive   |        40 |               0.0069 |             0.6   |       0.2016 |
| few_shot   |        40 |               0.0216 |             0.475 |       0.9735 |
| cot        |        40 |               0.022  |             0.7   |       0.0042 |
| negative   |        40 |               0.0967 |             0.95  |       0      |
| hybrid     |        40 |               0.1043 |             0.95  |       0      |
| zero_shot  |        40 |               0.2023 |             0.65  |       0.0028 |

## Score по сезонам

| ptype         |   autumn |   deep_winter |   summer |   winter |
|:--------------|---------:|--------------:|---------:|---------:|
| contextual    |   0.7606 |        0.6286 |   0.9094 |   0.6033 |
| cot           |   0.7506 |        0.6226 |   0.8897 |   0.5646 |
| few_shot      |   0.7566 |        0.6427 |   0.8121 |   0.6178 |
| hybrid        |   0.6821 |        0.5336 |   0.8067 |   0.4761 |
| kontext_guide |   0.7775 |        0.6252 |   0.926  |   0.5871 |
| negative      |   0.6774 |        0.5326 |   0.8443 |   0.4745 |
| positive      |   0.7581 |        0.6356 |   0.907  |   0.5874 |
| zero_shot     |   0.4513 |        0.3217 |   0.7182 |   0.6151 |

## Ответы на вопросы сравнения (бизнес-ТЗ §8.6)

- Лучше всего сохраняет исходную сцену: **kontext_guide** (structure 0.7312)
- Лучше всего передаёт сезон: **positive** (season 0.7346)
- Чаще всего портит сцену/добавляет лишнее: **zero_shot** (дисквалификаций 11, structure 0.6295)
- Лучший для дальнейшей работы: **kontext_guide** (score 0.7289)

## Обоснование выбора

Тип **kontext_guide** набрал максимальный средний итоговый score (0.7289) при 0 дисквалификациях. Итоговый score взвешивает сохранение структуры (вес 0.6) и выраженность сезона (вес 0.4); результаты со структурой ниже порога 0.3 исключаются. Числа автоматических метрик необходимо подтвердить ручной проверкой (колонки manual_score/manual_comment в results.csv).

## Тексты промптов лучшего типа (kontext_guide)

**summer:**
```text
Change the season to summer: lush green vegetation, green trees and grass, dry bare ground, warm daylight, no snow. Only replace the environmental conditions around the structures. Maintain identical placement of every road, building, parking lot and field boundary, and keep the same camera angle, framing and perspective.
```

**autumn:**
```text
Change the season to autumn: yellow and orange foliage, fading vegetation, golden and brown tones, some bare trees, no snow. Only replace the environmental conditions around the structures. Maintain identical placement of every road, building, parking lot and field boundary, and keep the same camera angle, framing and perspective.
```

**winter:**
```text
Change the season to winter: light snow cover on the ground and rooftops, cold bluish tones, leafless trees, patches of snow. Only replace the environmental conditions around the structures. Maintain identical placement of every road, building, parking lot and field boundary, and keep the same camera angle, framing and perspective.
```

**deep_winter:**
```text
Change the season to deep winter: heavy thick snow covering the entire ground and rooftops, frozen surfaces, strong cold white palette, snow-laden bare trees. Only replace the environmental conditions around the structures. Maintain identical placement of every road, building, parking lot and field boundary, and keep the same camera angle, framing and perspective.
```
