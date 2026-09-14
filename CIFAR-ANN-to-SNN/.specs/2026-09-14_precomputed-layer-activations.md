# Предвычисленные активации слоёв ANN (PyTorch)

- **Дата**: 2026-09-14
- **Статус**: Approved
- **Связанные issue/PR**: `.specs/2026-09-13_layerwise-from-colanet.md`

## Цели

1. Считать карты слоёв TinyCifarNet на GPU/PyTorch один раз и класть их рядом с остальными артефактами ANN (`CIFAR-ANN-SNN/artifacts/activations/`).
2. Режим `layerwise` **не** гоняет `ann_forward_maps` по 50k+ картинкам: читает `.npy` (порядок как в `CIFAR10.bin`).
3. Первый Conv2d не выгружается: в финальной SNN его считает `fromFile`; на стадии `conv2` вход (карты conv1) считается свёрткой uint8-fold по выбранным кадрам.

## Не цели

- Выгрузка Linear/FC.
- Хранение карт в CSV (слишком широко для conv2/pool1).
- Повторный forward всего стека в конвертере «на всякий случай».

## Функциональные требования

1. Скрипт `CIFAR-ANN-SNN/extract_pre_fc_activations.py`: все 60 000 CIFAR-10, слои `conv2, pool1, conv3, conv4, pool2, conv5, gap` после ReLU/pool/GAP. По-прежнему писать `pre_fc_activations.csv` (flatten GAP, 40-D) для CoLaNET.
2. Файлы: `artifacts/activations/<layer>.npy` — float32 NCHW, N=60000; `manifest.json`.
3. `build_snn.py --mode layerwise` читает `--activations-dir` (по умолчанию `<ann-dir>/activations`). Нет файлов — ошибка со ссылкой на скрипт. Юнит-тесты с явным `frames_hwc` по-прежнему считают карты на месте.
4. Индексация выборки шага 2 — те же индексы в 60k, что и у `select_layerwise_split`.

## Нефункциональные требования

- Производительность: один проход DataLoader, batch ≥256, CUDA если есть.
- Память: memmap на диск; не держать все слои в RAM при выгрузке. Конвертер mmap + копия только нужного слоя/подвыборки.
- Совместимость: порядок записей = `CIFAR10.bin` (train 0..49999, test 50000..59999).
- Наблюдаемость: `manifest.json` (shape, dtype, skip conv1).

## Открытые вопросы

Нет.

## Предполагаемые изменения в коде

### Новые классы

- `LayerActivationStore` (`snn_convert/activations.py`) — mmap `.npy` + uint8-fold для conv1.

### Изменённые классы

- `extract_pre_fc_activations.py` — выгрузка промежуточных карт.
- `convert_layerwise` / `select_layerwise_indices` — чтение карт по индексам.
- `build_snn.py` — `--activations-dir`.

### Удалённые классы

- Нет.

## Библиотеки и зависимости

- Новые: нет (torch уже в CIFAR-ANN-SNN).

## Критерии приёмки

### Автотесты

- [ ] `test_model.py`: имена/формы карт TinyCifarNet без conv1 на диске.
- [ ] `test_layerwise.py`: store читает npy; без файлов CLI layerwise с картинками — ошибка; `frames_hwc` в тестах работает как раньше.

### Линтеры и качество

- [ ] Комментарий `Spec: 2026-09-14_precomputed-layer-activations.md`.
