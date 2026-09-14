# Послойный перевод ANN→SNN от якоря CoLaNET (режим 2)

- **Дата**: 2026-09-13
- **Статус**: Approved
- **Связанные issue/PR**: `.specs/2026-09-10_ann-to-arni-snn.md` (режим `layerwise`); якорь `ArNI/Experiments/1.nnc`

## Цели

1. Реализовать режим `layerwise` конвертера: наращивать спайковый стек **от головы к пикселям**, стартуя с чистого CoLaNET `ArNI/Experiments/1.nnc` (вход — pre-FC активации ANN).
2. Для каждого спайкового ANN-слоя (GAP / pool / conv **кроме первого Conv2d**) выполнить два шага:
   - **Шаг 1 — Jaccard**: подобрать параметры конвертации слоя так, чтобы спайковый выход слоя на активациях `CIFAR10.bin` был близок к спайковому входу уже собранной сети. Метрика — `meanjaccard` из `SpikingApproximation-1fba1395.cpp` (`reldifthr = 1`).
   - **Шаг 2 — accuracy**: присоединить слой ко входу растущей сети и оптимизировать **те же** параметры по точности классификации (старт — точка шага 1). Оптимизатор — Нелдер–Мид (как NLopt `LN_NELDERMEAD` в ArNI-X).
3. Довести спайковый стек до выхода первого Conv. Первый Conv2d **не** конвертируется в LIF: он остаётся цифровым внутри `fromFile` (`convolution_file` + `<bias>`, ядра ANN без `weight_scale`/`bias_scale`). После стадии `conv2` вход меняется с CSV карт conv1 на `type="image"`. Подбирается только clip `s` кодирования цифрового выхода conv1. Заморозить CoLaNET из якоря.

## Не цели

- Повторный joint-GA (режим 3) и теоретический режим 1.
- Подстройка гиперпараметров CoLaNET якоря **внутри** послойного цикла. После того как стек собран, отдельная **глобальная** оптимизация всей сети (включая CoLaNET) допустима как следующий этап: предполагается, что максимум точности уже рядом. В данный алгоритм её не включаем.
- Параллельный пул GPU; несколько `ArNIGPU` сразу.
- Подбор по CIFAR **test** (последние 10 000): только train / hold-out из train. Test — финальный отчёт, не фитнес.
- Побитовое воспроизведение стохастической стимуляции LIF в шаге 1 (детерминированный суррогат среднего тока).

## Функциональные требования

1. Якорь: `C:/SNN/ArNI/Experiments/1.nnc` (или `--colanet-anchor`). Конвертер **парсит** XML (модель, `saturation_level`, CoLaNET, `iniresource`), а не копирует константы из старых артефактов. Смена файла якоря не требует правки кода.
2. Порядок спайковых слоёв (голова → вход): `gap` → `conv5` → `pool2` → `conv4` → `conv3` → `pool1` → `conv2`. Первый Conv2d в список не входит.
3. Параметры шага 1/2:
   - pooling / GAP: 1 величина — `saturation_level` входа. В XML TinyfromANN для этих слоёв **нет** `<layer>` / `weight_scale` / `bias_scale` (синапсы фиксированы как `THRESHOLD_BASE+1`).
   - спайковая свёртка (не conv1): 3 величины — `saturation_level` входа, `weight_scale`, `bias_scale`;
   - после `conv2`: только `s` цифрового `fromFile` image (ядра и bias conv1 не масштабируются).
4. Кодирование `text_values` без `nreceptivefields` — `GetSpikesfromTextValuesFileRateCoded`: `d = clip(r/sat, 0, 1)`, интегратор `dState`, спайк при `dState >= 1`, `record_presentation_time=10`. `dState` **не** сбрасывается между записями (как в fromFile).
5. Порядок рецепторов / столбцов CSV: `spatial_index(y, x, c) = (y * W + x) * C + c` (NHWC flatten).
6. AvgPool ≡ SumPool: синапс `THRESHOLD_BASE+1`. В LIF при спайке потенциал **не** сбрасывается в 0, а **декрементируется на величину порога**, поэтому число выходных спайков почти равно сумме входных по окну. Расхождение — остаток заряда, не успевший дать спайк в самом конце презентации картинки. Шаг 1 Jaccard моделирует этот интегратор (не OR по такту).
7. Свёртки TinyfromANN: millival `round(w * weight_scale * 1000)`, bias как в DLL (сток. стимул / `ThresholdExcess`).
8. `TinyfromANN`: XML `<skip_first_conv>` (по умолчанию 1 — прежнее поведение: первый Conv2d уже сделан `fromFile` image). Для наращивания с feature-map CSV — `skip_first_conv=0`, первый слой среза может быть Conv/Pool/GAP; размер `fromFile` = `input.shape` архитектуры-среза.
9. Период предъявления: `15 + delay`, delay = число секций TinyfromANN после текущего `fromFile` (каждый Conv/Pool/GAP, который DLL реально создаёт). Обновлять `record_presentation_period` / `ntact_per_image` / `object_presentation_period` / `reset_period` / `learning_time`. `ncopies` якоря не менять.
10. Шаг 1: грубая сетка по порядку величины, затем Нелдер–Мид. Фитнес = `meanjaccard`. Счётчики спайков — Python-реплика rate-code + SumPool / LIF (не `discretize` как целевой прогон; `discretize` — только совместимость с C++ тестом).
11. Шаг 2: то же пространство, фитнес = код `ObjectClassifier` / 10000 (accuracy). Старт — параметры шага 1. Вызов `ArNIGPU` (cwd = каталог эксперимента, один процесс). По умолчанию `--layerwise-train 50000` (весь CIFAR train) и `--layerwise-val 10000` (остаток файла для accuracy: CIFAR test, как в `1.nnc`). Дым: `--layerwise-train 800 --layerwise-val 200` (оба из train). Шаг 1 Jaccard по-прежнему `--layerwise-jaccard` (800).
12. CLI: `build_snn.py --mode layerwise --colanet-anchor <1.nnc>`. Без якоря — ошибка. `--no-eval` — шаги 1 и сборка `.nnc`, без `ArNIGPU`. `--layerwise-max-stages 0` — только скопировать/переименовать якорь. `--layerwise-fresh` — игнорировать чекпоинты.
13. Лог: JSON на стадию (параметры, Jaccard, accuracy, bounds, число оценок). Шаг 2 дополнительно пишет JSONL каждого запуска `ArNIGPU` в `step2_<layer>.jsonl` (flush после оценки) и текущий лучший набор в `step2_<layer>_best.json`. Повтор слоя продолжает с лучшей точки лога и не повторяет уже посчитанные параметры.
14. Продолжение: повтор того же `--out` подхватывает `layerwise_log.json` и `stage_<layer>.nnc`. Завершённые слои не оптимизируются заново; цикл идёт со следующего. Незавершённый слой: шаг 1 не повторяется, шаг 2 продолжается по `step2_<layer>.jsonl`.
15. Карты слоёв `conv2`…`gap` заранее считаются PyTorch (`CIFAR-ANN-SNN/extract_pre_fc_activations.py` → `artifacts/activations/<layer>.npy`). Конвертер их читает; без файлов — ошибка. `conv1` не выгружается (считает `fromFile` / uint8-fold на выбранных кадрах). Юнит-тесты с `frames_hwc` по-прежнему считают карты на месте.

## Нефункциональные требования

- Производительность: шаг 1 Jaccard — `--layerwise-jaccard` (800). Карты слоёв не считаются в конвертере (`.npy` из `artifacts/activations/`). Шаг 2 по умолчанию 50k×15 плюс 10k test; один `ArNIGPU`. Без явного `--timeout` шаг 2 ждёт `ArNIGPU` без ограничения.
- Память: CSV широких карт (conv1/conv2) на 50k+10k — гигабайты; для дыма `--layerwise-train 800 --layerwise-val 200`.
- Потокобезопасность: CLI однопоточный.
- Совместимость: существующие `.nnc` без `<skip_first_conv>` — поведение DLL как раньше (пропуск первого Conv2d). Python 3.10+, numpy; scipy не обязателен.
- Наблюдаемость: `artifacts/layerwise_log.json` (чекпоинт после каждой стадии), `stage_<layer>.nnc`, `conversion_params.json`, `accuracy_report.json`, `step2_<layer>.jsonl` / `step2_<layer>_best.json`.
- Прочее: seed Нелдера–Мида / грубой сетки фиксируемый.

## Открытые вопросы

Нет. Якорь читается из XML. Объём шага 2 задаётся `--layerwise-train` / `--layerwise-val`.

## Предполагаемые изменения в коде

### Новые классы

- `ColanetAnchor` — разбор якоря `1.nnc` (модель, sat, CoLaNET, `iniresource`).
- `LayerwiseConverter` (`snn_convert/layerwise.py`) — цикл стадий, шаг 1 и шаг 2; `Step2TrialLog` — JSONL каждого ArNIGPU шага 2.
- `MeanJaccard` / функции `meanjaccard`, `discretize` (`snn_convert/jaccard.py`).
- Rate-coder (`snn_convert/rate_code.py`) — реплика `GetSpikesfromTextValuesFileRateCoded`.
- `LayerSpikeSim` (`snn_convert/layer_sim.py`) — SumPool и LIF-свёртка для шага 1.
- Нелдер–Мид (`snn_convert/nelder_mead.py`).
- `GrowingNncBuilder` (`snn_convert/layerwise_nnc.py`) — `fromFile` text_values или image + TinyfromANN-срез + CoLaNET якоря.

### Изменённые классы

- `TinyfromANN` (`NETWORK_SET_PARAMETERS`) — `skip_first_conv`, первый слой не обязан быть Conv2d при 0.
- `ConversionParams` / `build_nnc_xml` — `iniresource`; опциональный `skip_first_conv`.
- `ann_stack_delay` — флаг, пропускать ли первый Conv.
- `build_snn.py` — режим `layerwise` больше не заглушка.

### Удалённые классы

- Нет.

## Библиотеки и зависимости

- Новые: нет (Нелдер–Мид свой).
- Обновления существующих: numpy / torch уже есть для `ann_forward`.
- Системные: MSVC для пересборки `TinyfromANN.dll` (Release|x64 → `ArNI/Experiments`).

## Критерии приёмки

### Автотесты

- [ ] `tests/test_layerwise.py`: `meanjaccard` / `discretize` совпадают с определением C++ (`count > 1`, среднее по строкам).
- [ ] Rate-code: `r = sat/2` → 5 спайков за 10 тактов; `r <= 0` → 0; `r >= sat` → 10.
- [ ] Порядок стадий: `gap, conv5, pool2, conv4, conv3, pool1, conv2` (без conv1). Финальный `.nnc` после conv2 — `fromFile` `type="image"`, `skip_first_conv=1`, немасштабированные ядра.
- [ ] Pool: 1 параметр; conv: 3 параметра.
- [ ] Срез архитектуры GAP: `input.shape` 40×3×3, первый слой `AdaptiveAvgPool2d`.
- [ ] `.nnc` стадии GAP: `text_values`, `skip_first_conv>0` отсутствует или 0, `model="smooth"`, `n` L = 70, Link GAP→L, `iniresource` с якоря.
- [ ] Шаг 2: `step2_<layer>.jsonl` содержит `eval` на каждый `ArNIGPU`; повтор читает лучшую точку.
- [ ] Resume: `max_stages=1`, затем полный проход в том же `--out` не повторяет шаг 1 для GAP; есть `stage_gap.nnc`.
- [ ] Карты слоёв: конвертер читает `activations/<layer>.npy`; без них CLI с `CIFAR10.bin` — ошибка.

### Линтеры и качество

- [ ] Стиль как в `snn_convert/` (без нового линтер-профиля).
- [ ] Сборка TinyfromANN Release|x64.
- [ ] Комментарий `Spec: 2026-09-13_layerwise-from-colanet.md` в новых/существенно изменённых файлах.

## План внедрения

1. Спека (этот файл).
2. DLL `skip_first_conv` + Python шаг 1 (Jaccard) + сборка `.nnc`.
3. Шаг 2 через `ArNIGPU` на подвыборке.
4. Дым: `--layerwise-max-stages 1` (только GAP), затем полный проход.
