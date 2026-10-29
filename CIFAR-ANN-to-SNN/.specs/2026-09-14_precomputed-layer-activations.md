# Предвычисленные активации слоёв ANN (PyTorch)

- **Дата**: 2026-09-14
- **Статус**: Approved
- **Связанные issue/PR**: `.specs/2026-09-13_layerwise-from-colanet.md`

## Цели

1. Считать карты слоёв TinyCifarNet на GPU/PyTorch один раз и класть их рядом с остальными артефактами ANN (`CIFAR-ANN-SNN/artifacts/activations/`).
2. Режим `layerwise` **не** гоняет `ann_forward_maps` по 50k+ картинкам: читает `.npy` (порядок как в `CIFAR10.bin`).
3. Первый Conv2d выгружается так же, как остальные (`conv1.npy`), и служит входом стадии `conv2`. В финальной SNN его по-прежнему считает `fromFile`, поэтому конвертер сверяет `conv1.npy` со свёрткой uint8-fold на первых 8 кадрах и падает при расхождении больше `1e-3` (другой чекпоинт или порядок каналов `CIFAR10.bin`). Если `conv1.npy` нет, карта считается свёрткой uint8-fold, как раньше.

## Не цели

- Выгрузка Linear/FC.
- Хранение карт в CSV (слишком широко для conv2/pool1).
- Повторный forward всего стека в конвертере «на всякий случай».

## Функциональные требования

1. Скрипт `CIFAR-ANN-SNN/extract_pre_fc_activations.py`: все 60 000 CIFAR-10, слои `conv1, conv2, pool1, conv3, conv4, pool2, conv5, gap` после ReLU/pool/GAP. По-прежнему писать `pre_fc_activations.csv` (flatten GAP, 40-D) для CoLaNET.
   - `--layers <имена>` пишет только эти карты: остальные `.npy` и `pre_fc_activations.csv` не трогаются, записи добавляются в существующий `manifest.json`. Перед записью первый батч должен воспроизвести уже лежащую на диске карту другого слоя с точностью `1e-4`, иначе ошибка. Так новая карта гарантированно посчитана тем же чекпоинтом, на тех же картинках и в том же порядке каналов.
2. Файлы: `artifacts/activations/<layer>.npy` — float32 NCHW, N=60000; `manifest.json`.
3. `build_snn.py --mode layerwise` читает `--activations-dir` (по умолчанию `<ann-dir>/activations`). Нет файлов — ошибка со ссылкой на скрипт. Юнит-тесты с явным `frames_hwc` по-прежнему считают карты на месте.
4. Карты читаются в порядке `CIFAR10.bin` (0..59999). Шаг 1 Jaccard — префикс `[:n_jaccard]`; шаг 2 / CSV / `fromFile` — исходные `CIFAR10.bin` и `CIFAR10.target.txt` без семплирования.

## Нефункциональные требования

- Производительность: один проход DataLoader, batch ≥256, CUDA если есть.
- Память: memmap на диск; не держать все слои в RAM при выгрузке. Конвертер mmap + копия только нужного слоя/подвыборки.
- Совместимость: порядок записей = `CIFAR10.bin` (train 0..49999, test 50000..59999).
- Наблюдаемость: `manifest.json` (shape, dtype; `skip` пуст, когда выгружены все слои).

## Открытые вопросы

Нет.

## Предполагаемые изменения в коде

### Новые классы

- `LayerActivationStore` (`snn_convert/activations.py`) — mmap `.npy`; для conv1 — `conv1.npy` со сверкой с uint8-fold, при отсутствии файла — сама свёртка uint8-fold.

### Изменённые классы

- `extract_pre_fc_activations.py` — выгрузка промежуточных карт.
- `convert_layerwise` — mmap карт по индексам 0..59999 (префикс для Jaccard).
- `build_snn.py` — `--activations-dir`.

### Удалённые классы

- Нет.

## Библиотеки и зависимости

- Новые: нет (torch уже в CIFAR-ANN-SNN).

## Критерии приёмки

### Автотесты

- [ ] `test_model.py`: имена/формы карт TinyCifarNet, включая conv1, совпадают с `SAVE_LAYERS`.
- [ ] `test_layerwise.py`: store читает npy; без файлов CLI layerwise с картинками — ошибка; `frames_hwc` в тестах работает как раньше; `conv1.npy`, не совпадающий с uint8-fold, — ошибка.

### Линтеры и качество

- [ ] Комментарий `Spec: 2026-09-14_precomputed-layer-activations.md`.
