# Сравнение промптов

Источник данных: `20260706-183744_results.csv` (320 оценённых изображений).

## Рейтинг типов промптов (по итоговому score)

| ptype         |   n |   mean_score |   mean_structure |   mean_season |   mean_dino |   mean_ssim |   mean_clip_prob |   mean_clip_dir |   n_disqualified |   std_score |
|:--------------|----:|-------------:|-----------------:|--------------:|------------:|------------:|-----------------:|----------------:|-----------------:|------------:|
| kontext_guide |  40 |       0.728  |           0.7363 |        0.7156 |      0.7973 |      0.4923 |           0.729  |          0.2344 |                0 |      0.1555 |
| contextual    |  40 |       0.7191 |           0.7154 |        0.7247 |      0.7852 |      0.4361 |           0.7546 |          0.2228 |                0 |      0.1543 |
| positive      |  40 |       0.7162 |           0.709  |        0.727  |      0.782  |      0.417  |           0.7553 |          0.2259 |                0 |      0.1494 |
| hybrid        |  40 |       0.7075 |           0.6956 |        0.7253 |      0.7692 |      0.4012 |           0.7542 |          0.2196 |                0 |      0.153  |
| cot           |  40 |       0.7005 |           0.6894 |        0.7172 |      0.7603 |      0.4059 |           0.7363 |          0.2241 |                0 |      0.1543 |
| few_shot      |  40 |       0.7    |           0.7206 |        0.6692 |      0.79   |      0.443  |           0.6807 |          0.2328 |                0 |      0.1417 |
| negative      |  40 |       0.6369 |           0.5907 |        0.7343 |      0.6845 |      0.2153 |           0.7703 |          0.2089 |                1 |      0.1767 |
| zero_shot     |  40 |       0.5266 |           0.6294 |        0.6718 |      0.6953 |      0.366  |           0.6732 |          0.1786 |               11 |      0.3502 |

## Парное сравнение с лидером (одинаковые исходник/сезон/вариант)

Лидер: **kontext_guide**. mean_delta > 0 и малый wilcoxon_p — лидер статистически значимо лучше; p > 0.05 — разница в пределах шума.

| ptype      |   n_pairs |   mean_delta_vs_best |   best_wins_share |   wilcoxon_p |
|:-----------|----------:|---------------------:|------------------:|-------------:|
| contextual |        40 |               0.0089 |             0.525 |       0.4762 |
| positive   |        40 |               0.0118 |             0.725 |       0.0451 |
| hybrid     |        40 |               0.0205 |             0.725 |       0.013  |
| cot        |        40 |               0.0275 |             0.75  |       0.0011 |
| few_shot   |        40 |               0.028  |             0.55  |       0.468  |
| negative   |        40 |               0.0911 |             0.9   |       0      |
| zero_shot  |        40 |               0.2014 |             0.65  |       0.0044 |

## Score по сезонам

| ptype         |   autumn |   deep_winter |   summer |   winter |
|:--------------|---------:|--------------:|---------:|---------:|
| contextual    |   0.7613 |        0.6312 |   0.9143 |   0.5698 |
| cot           |   0.7506 |        0.6226 |   0.8898 |   0.5391 |
| few_shot      |   0.7566 |        0.6427 |   0.8121 |   0.5889 |
| hybrid        |   0.745  |        0.6284 |   0.9019 |   0.5545 |
| kontext_guide |   0.785  |        0.6247 |   0.9261 |   0.5762 |
| negative      |   0.6774 |        0.5326 |   0.8443 |   0.4934 |
| positive      |   0.7581 |        0.6356 |   0.907  |   0.5642 |
| zero_shot     |   0.4513 |        0.3217 |   0.7182 |   0.6151 |

## Ответы на вопросы сравнения (бизнес-ТЗ §8.6)

- Лучше всего сохраняет исходную сцену: **kontext_guide** (structure 0.7363)
- Лучше всего передаёт сезон: **negative** (season 0.7343)
- Чаще всего портит сцену/добавляет лишнее: **zero_shot** (дисквалификаций 11, structure 0.6294)
- Лучший для дальнейшей работы: **kontext_guide** (score 0.7280)

## Обоснование выбора

Тип **kontext_guide** набрал максимальный средний итоговый score (0.7280) при 0 дисквалификациях. Итоговый score взвешивает сохранение структуры (вес 0.6) и выраженность сезона (вес 0.4); результаты со структурой ниже порога 0.3 исключаются. Числа автоматических метрик необходимо подтвердить ручной проверкой (колонки manual_score/manual_comment в results.csv).

## Тексты промптов лучшего типа (kontext_guide)

**summer:**
```text
Change the season to summer: lush green vegetation, green trees and grass, dry bare ground, warm daylight, no snow. Only replace the environmental conditions around the structures. Maintain identical placement of every road, building, parking lot and field boundary, and keep the same camera angle, framing and perspective. Do not add or remove any objects.
```

**autumn:**
```text
Change the season to autumn: yellow and orange foliage, fading vegetation, golden and brown tones, some bare trees, no snow. Only replace the environmental conditions around the structures. Maintain identical placement of every road, building, parking lot and field boundary, and keep the same camera angle, framing and perspective. Do not add or remove any objects.
```

**winter:**
```text
Change the season to winter: thin patchy snow cover with bare ground and roads partly visible, light dusting of snow on rooftops, cold pale light, leafless trees. Only replace the environmental conditions around the structures. Maintain identical placement of every road, building, parking lot and field boundary, and keep the same camera angle, framing and perspective. Do not add or remove any objects.
```

**deep_winter:**
```text
Change the season to deep winter: heavy thick snow covering the entire ground and rooftops, frozen surfaces, strong cold white palette, snow-laden bare trees. Only replace the environmental conditions around the structures. Maintain identical placement of every road, building, parking lot and field boundary, and keep the same camera angle, framing and perspective. Do not add or remove any objects.
```
