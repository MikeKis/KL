# Конвертация PyTorch-ANN в ИмНС ArNI-X (частотное кодирование)

- **Дата**: 2026-09-10
- **Статус**: Draft
- **Связанные issue/PR**: (нет)
- **Каталог реализации**: `CIFAR-ANN-to-SNN/` (тот же уровень, что `CIFAR-ANN-SNN/`)
- **Референс ANN (первый пример)**: `CIFAR-ANN-SNN/` (`TinyCifarNet`). Другие ANN — те же ограничения типов/геометрии; описание как `architecture.json` + свёрнутые веса.
- **Референс ИмНС-головы**: `--anchor <n>` → `<cwd>/Experiments/<n>.nnc` (только CoLaNET); в примере CIFAR-10 cwd = каталог `build_snn.py`
- **Данные**: `<cwd>/Workplace/CIFAR10.bin`, `<cwd>/Workplace/CIFAR10.target.txt`; `ArNIGPU` там же. Рабочие `.nnc` и плагины ArNI — `<cwd>/Experiments`. ANN — `--ann-dir` (где угодно).
- **Зависимая спека ArNI (bias `fromFile`)**: `ArNI/.specs/2026-09-10_fromfile-convolution-bias.md` — **реализовано**
- **Зависимая спека ArNI (DLL стека)**: `ArNI/.specs/2026-09-10_tinyfromann-dll.md`

## Цели

1. Построить систему в `CIFAR-ANN-to-SNN/` (Python) **и** DLL конструктора слоёв в ArNI-X, которая по заданной PyTorch-сети создаёт ИмНС: `.nnc` вызывает DLL, плюс `convolution_file` / файлы весов / запуск `ArNIGPU`.
2. Структура ИмНС должна быть максимально эквивалентна заданной ANN в пределах ограничений геометрии и типов слоёв (см. принятые решения).
3. Точность ИмНС на test CIFAR-10 — **ориентир около 70%** (не обязательство догнать ANN). Accuracy ANN логировать для сравнения (`CIFAR-ANN-SNN/artifacts/metrics.json`: **83.94%**, valid / без `pool3`).
4. Кодирование в ИмНС — **частотное** (rate coding).
5. Первый слой создаваемой ИмНС — **не импульсный, а числовой**, всегда свёрточный, рецепторы ArNI-X `fromFile` с **непустым** `convolution_file` и (после спеки bias) вектором смещений фильтров.
6. Классификационная голова — **CoLaNET**.
7. Конвертер читает **произвольную** ANN в рамках ограничений (типы слоёв, десяток слоёв, укладка окон, первый Conv, голова Linear→CoLaNET). `TinyCifarNet` / CIFAR-10 — **первый пример и первая очередь**, не единственная поддерживаемая конфигурация.

## Не цели

- Обучение исходной ANN **в каталоге конвертера**. Переобучение `TinyCifarNet` под valid-геометрию делается в `CIFAR-ANN-SNN/`.
- Типы слоёв вне набора CIFAR-ANN-SNN (residual, MaxPool, Softmax в экспортируемом графе и т.п.).
- Изменение ядра ArNI-X. Допускаются: уже сделанный bias в `fromFile`; **новая** `SpecialNetworkStructures` DLL (спека `2026-09-10_tinyfromann-dll.md`). `ConvolutionPooling` не обязан быть транспортом слоёв 2+.
- Выражение уникальных свёрточных весов «чистым» `.nnc` (секция на фильтр, плотный `policy="csv"`).
- Вторая полярность plus/minus карт консольного `CIFAR/convolution`.
- GUI; датасеты не в RAM; модели существенно больше ~30k свёрнутых параметров.
- Замена CoLaNET на перенос `fc` как Linear.

## Принятые решения

| # | Решение |
|---|---------|
| 1 | Порядок внедрения режимов: сначала **теоретический (1)**, затем **полная оптимизация (3)**, затем **послойный (2)** как самый трудоёмкий. Экспериментировать; каркас CLI может знать все три имени режима, но обязательный DoD идёт в этом порядке. |
| 2 | Геометрия: каждое окно Conv/Pool **целиком** лежит на слое под ним (`padding = 0`, `(H−K) % stride == 0`, `(W−K) % stride == 0`, выход ≥ 1). Иначе конвертер **завершается ошибкой** (невозможность конвертации), без молчаливой обрезки каймы. |
| 3 | Нормировка CIFAR-10 `(x/255 − mean) / std` с **известными константами** вливается в ядра и bias **первого** Conv (эквивалентная свёртка по `uint8`). См. приложение D. |
| 4 | Bias первого слоя — в `fromFile` (`<bias>` в `<Special>`). **Сделано.** |
| 5 | Свёрточный стек после conv1 **не** формализовать чистым `.nnc`. Отдельная DLL по образцу `1T/EfficientNet`. Одна секция на свёртку — слишком громоздко. |
| 6 | Симулятор: процесс **`ArNIGPU`** из Python. Каталог с `<id>.nnc` + `-e<id>`. |
| 7 | Целевая test accuracy ИмНС (первый эксперимент CIFAR-10): **около 70%**. |
| 8 | **AvgPool ≡ SumPool + вес синапса** на следующий слой. |
| 9 | `TinyCifarNet` — **первый пример**. Каноническое описание ANN: `architecture.json` + текстовый `weights_dump.txt` (тензоры весов и векторы смещений), как в `CIFAR-ANN-SNN/export.py`. `TinyfromANN` строит стек по этому описанию, не по зашитому `conv1…conv5`. |
| 10 | `ModifyNetwork` у `TinyfromANN` **не нужен** (точку входа не экспортировать). |

## Контекст и зафиксированные факты

### Референсная ANN (`TinyCifarNet`) — первый пример

Свёрнутый inference-граф (`architecture_spec()` / `architecture.json`) — образец входного формата. Другая сеть того же класса: тот же `architecture.json` + `weights_dump.txt` (секции `[имя] shape=[…]`, значения по одному в строке). `weights.npz` у `export.py` — необязательный двойник, канон для конвертера — **текстовый dump**.

| Имя | Тип | Примечание |
|-----|-----|------------|
| `conv1` | `Conv2d` 3→16, k=3, stride=1, **padding=0**, bias после fold | + ReLU |
| `conv2` | `Conv2d` 16→16, k=3, stride=1, padding=0 | + ReLU |
| `pool1` | `AvgPool2d` k=2, stride=2 | |
| `conv3` | `Conv2d` 16→32, k=3, stride=1, padding=0 | + ReLU |
| `conv4` | `Conv2d` 32→32, k=3, stride=1, padding=0 | + ReLU |
| `pool2` | `AvgPool2d` k=2, stride=2 | |
| `conv5` | `Conv2d` 32→40, k=3, stride=1, padding=0 | + ReLU |
| `gap` | `AdaptiveAvgPool2d` → 1×1 | среднее по 3×3, 40 признаков |
| `flatten` | `Flatten` | 40 признаков |
| `fc` | `Linear` 40→10 | **в ИмНС заменяется CoLaNET** |

Пространственные размеры: `32→30→28→14→12→10→5→3→1`. `pool3` нет: 2×2 на 3×3 не укладывается; GAP по всей 3×3 — один тайл на слой. Голова по-прежнему 40-D, якорь CoLaNET `1.nnc` совместим по размерности входа.

**Следствие решения 2:** этот граф проходит проверку укладки. Старые артефакты с `padding=1` неконвертируемы; нужен переобученный checkpoint.

Допущения ANN под конвертацию: ReLU, AvgPool (не max), BN fold, нет residual, ArgMax логитов.

Препроцессинг: mean/std в `architecture.json`. `CIFAR10.bin` — 60 000 кадров `uint8` 32×32×3; метки — 60 000 строк; test = последние 10 000.

### Первый слой ИмНС: `fromFile` + `convolution_file` + bias

По `ArNI/.specs/2026-09-03_fromfile-image-convolution.md` и `2026-09-10_fromfile-convolution-bias.md`:

- `args type="image"`, непустой `convolution_file` → числовая valid-свёртка `uint8`, затем `v = conv + bias[f]`, `ReLU(v)`, клип по `s`, частотное кодирование. **Bias в `fromFile` реализован.**
- Один слой в файле ядер (`*** LAYER 0`); padding нет.
- `s` — один на все фильтры; `bias` — вектор длины `nFilters` в XML `<Special>`, не в `convolution_file`.
- Порядок рецепторов: `index = (yOut * outW + xOut) * nFilters + filter`.

### Голова CoLaNET

Референс `CIFAR-ANN-SNN/artifacts/1.nnc`: `fromFile` `text_values` на CSV до-FC, `saturation_level=0.8`, `ObjectClassifier`, `model="linearized"`, секции `L` / `OUT` / `BIASGATE`, `ncopies="15"`. Для режима 2 — якорь.

### Стек после conv1: DLL, не чистое `.nnc`

Чистый XML (секция на фильтр / плотный CSV весов) для свёрток **отклонён** — слишком громоздко.

Эталон: `1T/EfficientNet` — `<Implementation lib="EfficientNet">` в `.nnc`, внутри DLL: XML-шаблон секций, `inc.AddNetwork`, `inc.Convolution`, точные веса через `bConnectNeurons`, bias нейронов через `SetNeuronProperty` (`p_StochasticStimulation` / `s_ThresholdExcess`).

Для любой такой ANN — DLL `TinyfromANN` (`ArNI/.specs/2026-09-10_tinyfromann-dll.md`): топология **не** зашита как пять свёрток TinyCifarNet, а читается из описания, которое пишет Python. `.nnc` эксперимента:

1. `RECEPTORS` `fromFile` (первый Conv: `convolution_file` + `bias`);
2. `<NETWORK><Implementation lib="TinyfromANN">` — остальные Conv / AvgPool / GAP по графу;
3. CoLaNET + `Readout` (размер входа CoLaNET = число признаков после GAP/Flatten, у примера — 40).

Python в `CIFAR-ANN-to-SNN/` готовит файлы весов, масштабы и `.nnc`; топологию связей строит DLL.

**AvgPool / GAP:** то же, что SumPool, если исходящий синапс несёт масштаб (в т.ч. `1/k²`). Не требуется отдельная «средняя» семантика нейрона. Вопрос про расхождение ConvolutionPooling с AvgPool закрыт этим правилом; транспорт слоёв — новая DLL, не обязанность `ConvolutionPooling`.

## Функциональные требования

### Размещение и вход системы

1. Код оркестрации — `CIFAR-ANN-to-SNN/` (Python). CIFAR C++ не менять. ArNI: bias `fromFile` (готово); новая DLL стека (спека DLL).
2. CLI (ориентир `build_snn.py`) принимает:
   - путь к ANN: каталог/пара файлов `architecture.json` + `weights_dump.txt` (контракт `CIFAR-ANN-SNN`; не только класс `TinyCifarNet`);
   - режим: `theoretical` \| `joint` \| `layerwise`;
   - пути к бинарным образам и меткам (пример: `CIFAR10.bin`, `CIFAR10.target.txt`);
   - для `layerwise`: якорь CoLaNET (для примера — `CIFAR-ANN-SNN/artifacts/1.nnc`);
   - каталог выхода;
   - параметры запуска `ArNIGPU`: каталог экспериментов, `-e<id>`, при необходимости прочие флаги.
3. Выход: `.nnc` (`<id>.nnc`), непустой `convolution_file` слоя 1, файлы весов для DLL, JSON параметров конверсии, отчёт accuracy, лог `ArNIGPU`.

### Инварианты структуры ИмНС

4. Первый слой — `fromFile` `type="image"`, непустой `convolution_file`, `bias` длины `nFilters` (**уже в `fromFile`**).
5. Голова — CoLaNET как в `1.nnc` / `CoLaNET.nnp`.
6. Слои после первого Conv до входа CoLaNET строит **`TinyfromANN` по графу ANN**. Flatten — топология; выход DLL = число каналов/признаков последней карты (у примера 40). `Linear` классификатора не эмитится.
7. Частотное кодирование; ориентир `Tpres = 10`.

### Геометрия (отказ конвертации)

8. Перед генерацией ИмНС проверить каждый `Conv2d` и pooling-слой ANN:

   - `padding == 0` (если поле есть; иначе 0);
   - ядро квадратное, как требует `fromFile` / `ConvolutionPooling`;
   - `(H - K) % stride == 0` и `(W - K) % stride == 0`;
   - `out = (H - K) / stride + 1 >= 1` (аналогично W).

   Нарушение — ненулевой exit, сообщение какие слой/размеры не укладываются, **без** выходного `.nnc`.

9. `AdaptiveAvgPool2d` в 1×1 допустим только если входная карта уже согласована предыдущими проверками (окно GAP покрывает всю карту целиком — это укладка 1 тайла).

### Разрешённые типы слоёв ANN

10. После BN-fold допустимы только: `Conv2d`, `ReLU`, `AvgPool2d`, `AdaptiveAvgPool2d`, `Flatten`, `Linear` (голова). `BatchNorm2d` только во входном checkpoint, fold как `export.py`. Первый вычислительный слой — `Conv2d`. Иной тип или не-Conv первый слой — ошибка. Глубина — порядка десятка слоёв.

### Подходы к конвертации

Порядок поставки: **1, затем 3, затем 2**. JSON параметров предыдущего этапа — вход следующего.

#### Режим 1 — теоретический (`theoretical`)

11. Топология и веса **без оптимизации** на датасете.
12. Ядра → синаптические веса / `convolution_file`. Для **первого** Conv: если в `architecture.json` заданы mean/std (как CIFAR-10), fold в uint8-эквивалент (приложение D), затем запись ядер и `bias` в `fromFile`. Другой датасет — те же формулы с его константами из описания, либо ошибка, если констант нет, а вход `fromFile` — сырой `uint8`.
13. Смещения **последующих** LIF-слоёв (не первый `fromFile`):
    - отрицательные — увеличение порога (на уровне секции или секция-на-фильтр);
    - положительные — `stochastic_stimulation`.
    Формулы коэффициентов — в одном модуле, с юнит-тестами; точные константы можно уточнять экспериментами режима 1.
14. Датасет для режима 1 — **оценка** accuracy через `ArNIGPU`. Исключение: одноразовый `s` (`ncalibrationimages` / `find_max_ratio`) считается частью кодирования `fromFile`, не поиском масштабов слоёв. Mean/std — константы из `architecture.json`, не статистика «обучения конвертера».

#### Режим 3 — полная оптимизация (`joint`) — второй по очереди

15. Все параметры перевода оптимизируются совместно на данных (референс CIFAR-10). Старт — из JSON режима 1.
16. Цель — test/hold-out accuracy ИмНС (ориентир ~70%). Подбор не вести исключительно по test, если есть train (предпочтительно train или подвыборка train; test — отчёт).
17. Вызов симулятора — `ArNIGPU` как процесс. Python rate-суррогат во внутреннем цикле **не запрещён**, но приёмка режима — прогон `ArNIGPU`.

#### Режим 2 — послойный (`layerwise`) — последний

18. Обязателен якорь `.nnc` одной CoLaNET (`1.nnc`). Параметры головы по умолчанию заморожены.
19. От головы к входу: подбор `weight_scale`, масштабов смещений, `activation_to_spikes` так, чтобы спайковый выход слоя совпадал со входом уже зафиксированного следующего.
20. Метрика соответствия — одна, явная, в логе (конкретная формула может быть выбрана при реализации режима 2).

### Генерация артефактов и запуск

21. `convolution_file`: контракт `ParseConvolutionTensorFile` (`*** LAYER 0`, квадрат, `(...)`, пробел после `)`).
22. `.nnc` well-formed; есть `fromFile`+`convolution_file`+`bias`, `Implementation` DLL стека, CoLaNET, `Readout`.
23. Запуск оценки: Python `subprocess` (или эквивалент) **`ArNIGPU`**, первый аргумент — каталог с `<id>.nnc`, опция `-e<id>`. Разбор точности из stdout / лога readout `ObjectClassifier` / принятого артефакта ArNI (конкретный парсер — привязать к реальному выводу при первом smoke; зафиксировать в логе конвертера). Рабочий каталог и копирование `CIFAR10.bin`, меток, `convolution_file` рядом с экспериментом — обязанность скрипта.

### Оценка качества

24. Метрика приёмки **первого** эксперимента: **test accuracy** CIFAR-10 (последние 10 000), ориентир ≈ **70%**. Для другой ANN — accuracy на её тестовом split, порог оговаривается отдельно (по умолчанию тот же смысл: заметно ниже ANN, не обязательно её догнать).
25. Режим 1 не подкручивает параметры по этой метрике.
26. ANN accuracy на том же split писать рядом (для примера **83.94%** в `CIFAR-ANN-SNN/artifacts/metrics.json`).

## Нефункциональные требования

- **Производительность**: режим 1 (генерация файлов) — секунды на CPU. Режимы 3 и 2 — много процессов `ArNIGPU`; один симулятор в каждый момент в v1 (без пула GPU), если не оговорено иначе.
- **Память**: датасет примера (CIFAR-10) в RAM допустим; сети класса «десяток слоёв» / порядка 10⁴–10⁵ свёрнутых параметров (у TinyCifarNet ≈ 29k).
- **Параллелизм**: CLI однопоточный; PyTorch DataLoader — по желанию. Несколько `ArNIGPU` сразу — не требование v1.
- **Совместимость**: Python 3.10+; `torch`/`numpy` по необходимости; Windows + `ArNIGPU`. Форматы: `.nnc`, `convolution_file` (только рецептор conv1), `architecture.json`, `weights_dump.txt`.
- **Наблюдаемость**: лог режима, таблица слоёв, параметры, командная строка `ArNIGPU`, код возврата, accuracy ANN и ИмНС.
- **Прочее**: фиксируемый seed для оптимизаторов (режимы 3/2).

## Открытые вопросы

Бывшие блокеры 1–6 и порог точности закрыты (см. «Принятые решения»). Осталось:

1. ~~Референс с `padding=1` / `pool3`.~~ Закрыто: valid, без `pool3`, GAP 3×3; ANN переобучена (83.94%).
2. ~~Убирать ли `pool3`.~~ Убран; flatten через GAP (40-D для CoLaNET).
3. ~~Чистое `.nnc` vs DLL.~~ DLL по образцу EfficientNet; спека `2026-09-10_tinyfromann-dll.md`.
4. ~~AvgPool vs SumPool.~~ Эквивалентны с учётом веса последующего синапса.
5. ~~Формат описания ANN / весов.~~ `architecture.json` + `weights_dump.txt` (CIFAR-ANN-SNN). Первый Conv из dump → `fromFile` `convolution_file`+`bias`; остальное читает `TinyfromANN`. `weights.npz` не обязателен.
6. Python rate-суррогат во внутреннем цикле режима 3 — сразу или только `ArNIGPU`?
7. Метрика послойного соответствия (режим 2) — отложить до режима 2.
8. Подстройка входа CoLaNET vs полная заморозка якоря — отложить до режима 2.
9. Split оптимизации режима 3: train / подвыборка (не test).
10. ~~v1 только `TinyCifarNet`.~~ Любая ANN в ограничениях; TinyCifarNet — первый пример.
11. `Tpres` / `ntact_per_image` / `chartime`: как в `1.nnc` в режиме 1?
12. Имя CLI и формат конфига (argparse vs TOML)?

## Предполагаемые изменения в коде

### Новые классы (ориентиры TBD)

- `AnnGraph` — загрузка `architecture.json` + `weights_dump.txt`.
- `GeometryGuard` — проверка укладки окон; отказ с диагностикой.
- `Uint8FirstConvFolder` — приложение D для `conv1`.
- `LayerConversionParams`
- `ConvolutionFileWriter`
- `NncBuilder` — `.nnc`: `fromFile` + `Implementation` DLL + CoLaNET.
- (C++, ArNI) модуль `SpecialNetworkStructures/TinyfromANN` — см. спеку DLL.
- `TheoreticalConverter`, `JointOptimizer`, `LayerwiseConverter`
- `ArniGpuRunner` — `subprocess` `ArNIGPU <dir> -e<id> ...`, разбор accuracy.
- `AccuracyReport`

### Изменённые классы

- `FileSpikeSource` / свёртка `fromFile` — bias **сделан** (`2026-09-10_fromfile-convolution-bias.md`).
- `TinyCifarNet` — valid conv, без `pool3`; переобучен (83.94%).

### Удалённые классы

- Нет.

### Прочие артефакты

- `CIFAR-ANN-to-SNN/` (каталог уже создан).
- Эта спека; зависимая спека ArNI (bias).
- Позже: Python-пакет, `requirements.txt`, `tests/`, `artifacts/`.

## Библиотеки и зависимости

- **Новые**: нет обязательных сверх `torch`, `numpy`; XML — stdlib.
- **Существующие**: артефакты `CIFAR-ANN-SNN`; `fromFile` с bias; эталон DLL `1T/EfficientNet`; `ObjectClassifier`, CoLaNET; `ArNIGPU`.
- **Обновления**: `fromFile` (bias — готово); новая DLL стека.
- **Toolchain**: Python 3.10+; MSVC для DLL; Windows + `ArNIGPU`.

## Критерии приёмки

### Автотесты (`CIFAR-ANN-to-SNN/tests/`)

- [ ] Загрузка референсных `architecture.json` + `weights_dump.txt` TinyCifarNet: ключи и shapes совпадают с `export_summary.json` / dump.
- [ ] Вторая фикстура: другой допустимый граф (иной набор каналов/число Conv в ограничениях) — конвертер читает и не предполагает имён `conv1…conv5`.
- [ ] Отказ на неизвестный тип слоя.
- [ ] Отказ геометрии: синтетический Conv с `padding=1` и/или `(H−K) % stride != 0` — exit ≠ 0, `.nnc` нет.
- [ ] Референсный `TinyCifarNet` (valid, без `pool3`) проходит geometry-guard.
- [ ] Fold uint8 (приложение D): на константном изображении `conv(W', x_u8) + b'` совпадает с `conv(W, (x/255−mean)/std) + b` с допуском float.
- [ ] `ConvolutionFileWriter`: контракт `*** LAYER 0` как в тесте `fromFile`.
- [ ] Режим `theoretical`: `.nnc` с `fromFile` (`convolution_file`, `bias`), `Implementation` DLL, CoLaNET, `Readout`.
- [ ] C++: сборка DLL стека (критерии в спеке DLL).
- [ ] `ArniGpuRunner` (мок процесса): собирает командную строку `ArNIGPU`, `<dir>`, `-e<id>`.
- [ ] Режимы `joint` / `layerwise`: каркас может быть заглушкой до своей очереди; `layerwise` без якоря — ошибка валидации. Полные тесты — в PR соответствующей очереди.

### Оценка на данных (ручной / поздний smoke)

- [ ] После появления конвертируемой ANN: режим 1 + `ArNIGPU` пишет test accuracy; цель эксперимента ≈ 70%.
- [ ] Режим 3 стартует из JSON режима 1 и не хуже инициализации на том же протоколе (регрессия не обязана бить 70% в первом прогоне, но отчёт comparables обязателен).

### Линтеры и качество

- [ ] Сборка DLL `Release|x64`; Python — выбранный линтер в первом PR.
- [ ] `CIFAR-ANN-SNN/test_model.py` не ломать без нужды.

## План внедрения

1. Спеки конверсии + bias `fromFile` + DLL — текущее состояние.
2. ~~fromFile bias.~~ Сделано.
3. ~~Референс ANN valid / без `pool3`.~~ Сделано (83.94%); `pre_fc_activations.csv` обновлён.
4. **DLL стека** (спека `2026-09-10_tinyfromann-dll.md`) — блокирует режим 1.
5. **PR-A (режим 1)**: Python fold uint8, файлы весов, `.nnc` + `ArNIGPU`.
6. **PR-B (режим 3)** → **PR-C (режим 2)**.

## Приложение A. Раскладка каталога

Python-скрипты остаются в `CIFAR-ANN-to-SNN/`. `build_snn.py` запускается из каталога `X` (в примере CIFAR-10 — тот же каталог, где лежит скрипт):

```
X/
  Experiments/          # все рабочие .nnc (в т.ч. якорь 1.nnc только с CoLaNET)
                        # и динамические библиотеки ArNI (TinyfromANN, fromFile, ObjectClassifier)
  Workplace/            # CIFAR10.bin, CIFAR10.target.txt, ArNIGPU
                        # сюда же пишутся файлы данных и логи симулятора
CIFAR-ANN-SNN/artifacts/   # ANN (--ann-dir), где угодно
CIFAR-ANN-to-SNN/          # скрипты (build_snn.py, snn_convert/, tests/)
```

Запуск симулятора: cwd = `X/Workplace`, `.nnc` в `X/Experiments`, команда `ArNIGPU <Experiments> -e<id>`.

## Приложение B. Параметры конвертации слоя

| Поле | Смысл |
|------|--------|
| `weight_scale` | `w_snn = weight_scale * w_ann` (после uint8-fold для слоя 1) |
| `bias_to_threshold_scale` | LIF-слои, `bias < 0` |
| `bias_to_stoch_stim_scale` | LIF-слои, `bias > 0` |
| `activation_to_spikes` | перевод активации в счётчик спайков за `Tpres` |
| `s` | клиппинг `fromFile` (слой 1) |
| `fromfile_bias[]` | вектор в XML `fromFile` (слой 1) |
| `weight_factor` / вес после SumPool | масштаб ядра и `1/k²` уходящей связи (AvgPool ≡ SumPool + этот вес) |

## Приложение C. Геометрия и текущий референс

Правило: окно k×k со stride s на карте H×W укладывается ⟺ `padding=0` и `(H−k)` делится на `s`, `(W−k)` делится на `s`.

С `padding=1` старый граф сохранял 32→16→8→4; конвертация **запрещена**.

Valid без padding и без `pool3` (текущий `TinyCifarNet`):

| шаг | операция | H=W |
|-----|----------|-----|
| вход | | 32 |
| `conv1` | k=3, s=1 | 30 |
| `conv2` | k=3, s=1 | 28 |
| `pool1` | k=2, s=2 | 14 |
| `conv3` | k=3, s=1 | 12 |
| `conv4` | k=3, s=1 | 10 |
| `pool2` | k=2, s=2 | 5 |
| `conv5` | k=3, s=1 | 3 |
| `gap` | AdaptiveAvgPool → 1×1 | один тайл на всю карту 3×3 |

`AvgPool2d(2)` на 3×3 не укладывается — слоя нет. Flatten в классификатор — после GAP (40 чисел), не сырые 3×3×40.

Ошибочная цепочка `32→30→15→13` получалась, если пулить сразу после первой свёртки; пул идёт после **двух** свёрток.

## Приложение D. Свёртка по `uint8` при известных mean/std

Пусть для канала `c`: `x_norm[c] = (x_u8[c] / 255 − mean[c]) / std[c]`,  
`y[f] = Σ_{c,i,j} W[f,c,i,j] · x_norm[c,i,j] + b[f]`.

Эквивалентно свёртке сырого `x_u8`:

\[
W'[f,c,i,j] = \frac{W[f,c,i,j]}{255 \cdot \mathrm{std}[c]},
\quad
b'[f] = b[f] - \sum_{c,i,j} W[f,c,i,j] \cdot \frac{\mathrm{mean}[c]}{\mathrm{std}[c]}.
\]

`W'` и `b'` идут в `convolution_file` и `<bias>` `fromFile`. Константы — из `architecture.json`, не сэмплируются с датасета при конверсии. Это отвечает: да, пересчёт матриц «как если бы свёртка была с uint8» — правильное понимание вопроса про mean/std.

## Приложение E. Запуск `ArNIGPU`

По контракту ArNI: имя файла строго `<id>.nnc`; симулятор: `ArNIGPU <каталог_nnc> -e<id>`. Скрипт копирует/пишет `.nnc` и соседей (`source` картинок, `target_file`, `convolution_file`) так, чтобы относительные пути внутри XML резолвились из рабочего каталога процесса. Код возврата и логи (`General<id>.log`, вывод readout) сохранять в `CIFAR-ANN-to-SNN/artifacts/`.
