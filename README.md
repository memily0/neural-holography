# Nuclear IT Hack | Physics-Informed Unet++

Четыре grayscale PNG 128x128 преобразуются в один фазовый ДОЭ того же размера

Источник решения: [Nuclear_Competition_Final.ipynb](experiments/Nuclear_Competition_Final.ipynb) и командный `src/physics/fresnel.py`
В `checkpoints/best_model.pt` лежит точная копия выбранного ноутбуком `best_smp_competition.pt`, эпоха 6

## Как работает

```text
4 PNG -> нормировка -> PhysicsSeed + координаты -> Unet++ -> фаза -> uint8 PNG
                                                                  |
                                                 Fresnel -> 4 амплитуды
```

- Готовая SMP UnetPlusPlus, ResNet34 с исходным предобучением ImageNet, 8 входных и 2 выходных канала
- Каждый вход делится на 255 и собственный максимум с ограничением 1e-8
- PhysicsSeed: `ifftshift -> ifft2(norm="ortho") -> fftshift`, сопряжённые входные chirp-фазы, среднее четырёх комплексных полей и нормировка с ограничением 1e-6
- Вход сети: `(targets - 0.45) / 0.225`, real/imag seed, координаты x/y
- Выходная фаза: `atan2(imag, real)` поля `0.7 * seed + delta`
- Fresnel перенесён обычными функциями: входной chirp, прямое `fftshift(fft2(ifftshift(...)))`, выходной chirp и исходный физический множитель
- Длина волны 532 нм, пиксель ДОЭ 10 мкм, расстояния 24.1 / 25.3 / 26.5 / 27.7 мм
- Сравниваются амплитуды `abs(field)`, каждая реконструкция делится на свой максимум с ограничением 1e-8

Синтез выполняется одним проходом обученной сети, без оптимизации отдельного ДОЭ
Полный state_dict загружается строго, `encoder_weights=None` исключает повторное скачивание ImageNet

## Запуск

Python 3.12, команды из этой папки

```bash
python -m venv .venv
```

Активировать окружение: `.venv\Scripts\activate` в Windows или `source .venv/bin/activate` в Linux

```bash
python -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python scripts/predict.py --input examples/1.png examples/2.png examples/3.png examples/4.png --output outputs/doe.png
```

`--reconstruct-dir outputs/reconstructions` сохраняет четыре восстановления из повторно прочитанного ДОЭ
`--device auto|cpu|cuda` выбирает устройство, по умолчанию auto
Локальный CUDA доступен при установленной CUDA-сборке PyTorch, базовый контейнер использует CPU

```bash
docker build -t nuclear-holography .
docker run --rm --network none -v "$(pwd)/examples:/app/inputs:ro" -v "$(pwd)/outputs:/app/outputs" nuclear-holography --input /app/inputs/1.png /app/inputs/2.png /app/inputs/3.png /app/inputs/4.png --output /app/outputs/doe.png
```

Для PowerShell заменить `$(pwd)` на `${PWD}`
Зависимости скачиваются при установке или сборке, инференсу сеть не нужна

Входы: ровно четыре PNG режима L, 8 бит, 128x128, в порядке расстояний выше
Выход: PNG L uint8 128x128, код 0 соответствует -π, код 255 соответствует +π
Формула чтения фазы: `code * (2π / 255) - π`

## Проверки и benchmark

```bash
python -m unittest discover -s tests -v
python scripts/benchmark.py --manifest examples/benchmark/manifest.json --output outputs/benchmark_cpu.json --device cpu --repeat 3
```

Реальный CPU benchmark: 16 фиксированных четвёрок из локальных test-split CIFAR10 / FashionMNIST / MNIST, по 3 повтора после прогрева, 4 CPU-потока
Входы сохранены в `examples/benchmark`, метрики рассчитаны после экспорта и повторного чтения 8-битного ДОЭ

| Плоскость, мм | Pearson | Local SSIM | Global SSIM |
|---|---:|---:|---:|
| 24.1 | 0.5218 | 0.2091 | 0.3135 |
| 25.3 | 0.5777 | 0.2343 | 0.3513 |
| 26.5 | 0.5560 | 0.2002 | 0.3715 |
| 27.7 | 0.5603 | 0.1951 | 0.3837 |
| Среднее | 0.5540 | 0.2097 | 0.3550 |

Среднее время сети 0.118 с/четвёрку, с чтением входов и записью/чтением ДОЭ 0.126 с, загрузка checkpoint отдельно 0.578 с
Local SSIM использует окно 7x7, uniform average и reflect padding, Global SSIM использует статистики всей плоскости, C1=0.01² и C2=0.03²
Это внутренние метрики на публичных данных, официальный score и качество на изображениях организаторов не установлены

25 тестов прошли в исходном окружении и в отдельном чистом окружении с CPU-only PyTorch
Проверены повторяемость PNG, формат, отсутствие NaN/Inf, работа без сети и соответствие физики/seed/сети эталону ноутбука
Параметры источника, SHA256, полный benchmark и результаты проверок находятся в `docs`

Docker build и run на этой машине заблокированы WSL: `HCS_E_HYPERV_NOT_INSTALLED`
Контейнерный запуск пока не подтверждён, журналы находятся в `docs/docker-build.log` и `docs/docker-run.log`

Код: `src/neural_holography`, команды: `scripts`, веса: `checkpoints`, тесты: `tests`, результаты: `outputs`
Ограничения: фиксированные размер и физика, зависимость качества от входных изображений, возможные различия квантования CPU/CUDA
