# Руководство для ML-участника команды

Версия: 1.0

Дата: 17 октября 2026 года

Владелец архитектуры и интеграции: тимлид проекта

## 1. Что от тебя требуется

Твоя зона ответственности — подготовить воспроизводимую локальную модель, которая по видео определяет наличие возможного нарушения и временную позицию события.

Ты отвечаешь за:

- аудит обучающих данных;
- подготовку клипов и разбиений;
- обучение и сравнение моделей;
- подбор порога срабатывания;
- оценку качества на независимых видео;
- экспорт лучшей модели;
- измерение скорости и памяти;
- отчет и передачу результата тимлиду.

Ты не меняешь без согласования:

- публичный API;
- систему фоновых задач;
- архитектуру приложения;
- форматы ответа сервиса;
- юридические формулировки;
- исходные видео и исходную таблицу разметки.

Все видео обрабатываются локально. Запрещено загружать датасет в Colab, Kaggle, облачные AI API, публичные хранилища или внешние сервисы разметки.

## 2. Контекст задачи

Итоговый сервис должен принимать `.mov`, анализировать его не дольше пяти минут и возвращать:

- `violation` или `no_violation`;
- таймкод возможного нарушения;
- тип и краткое описание;
- 1–3 JPEG-кадра-доказательства.

Первая измеримая ML-задача — бинарное обнаружение нарушения и локализация таймкода. Исходная разметка не содержит официального типа нарушения, поэтому многоклассовая модель на первом этапе запрещена.

## 3. Что уже находится в репозитории

Перед началом работы попроси Codex прочитать только относящиеся к ML файлы:

1. `analysis/dataset_manifest.json` — характеристики всех видео и исходные метки.
2. `analysis/annotation_schema.json` — рабочие семейства сцен; это гипотезы, а не ground truth.
3. `analysis/preliminary_video_review.md` — визуальный разбор положительных событий и контрастных пар.
4. `legal/README.md` — ограничения юридической интерпретации.
5. `data/open/open_labels.xlsx` — исходная таблица организаторов.
6. `data/open/videos/` — 42 исходных видео.
7. `analysis/clips/` — подготовленные фрагменты около известных таймкодов; их можно использовать для просмотра, но train pipeline должен уметь воспроизвести их самостоятельно.

Сводка датасета:

- 42 строки и 42 видео;
- 22 положительных видео с таймкодами;
- 20 отрицательных видео;
- около 24 минут материала;
- H.264, `464x848`, 30 FPS, без аудио;
- два набора побайтных дубликатов;
- после дедупликации — 40 уникальных видео.

Особые группы:

- `Х-4` и `Х-47` — один и тот же файл, но два разных кандидатных события: 57 и 68 секунд;
- `Х-5` и `Х-48` — один и тот же отрицательный файл;
- дубликаты никогда не должны попадать в разные folds;
- контрастные пары из `analysis/preliminary_video_review.md` желательно держать в одном fold, чтобы фон, актеры и расположение мебели не создавали утечку.

## 4. Definition of Done

Работа завершена, только если переданы:

- воспроизводимое Python-окружение;
- сохраненное разбиение train/validation;
- скрипт построения клипов;
- минимум один baseline и один улучшенный эксперимент;
- таблица экспериментов;
- метрики на независимых исходных видео;
- проверка временной локализации;
- `model.onnx` или согласованный `model.pt`;
- `preprocessing.json`;
- `thresholds.json`;
- `labels.json`;
- контрольная сумма модели SHA-256;
- команда запуска на одном `.mov`;
- измерение времени, VRAM/RAM и пиковой памяти;
- `reports/model_report.md` с ограничениями.

Высокая accuracy на нарезанных кадрах без независимой проверки по исходным видео не считается результатом.

## 5. Skills для Codex

Не устанавливай десятки skills. Лишние инструкции засоряют контекст Codex. Для этой работы достаточно двух обязательных и одного необязательного навыка.

### Обязательные

1. `jupyter-notebook` из официального `openai/skills` — для воспроизводимых экспериментов и отчетных notebook-файлов.
2. `systematic-debugging` из `obra/superpowers` — для поиска причины ошибок CUDA, декодирования, размеров tensor и обучения вместо случайной смены параметров.

### Необязательный

3. `security-best-practices` из официального `openai/skills` — использовать только для финальной проверки Python-кода, который читает пользовательские видео и пути к файлам.

`skill-installer` уже является системным skill Codex и отдельно не устанавливается.

### Основная команда для Codex

Отправь Codex участника это сообщение:

```text
Используй системный skill-installer. Установи в пользовательские skills:
1) jupyter-notebook из openai/skills, путь skills/.curated/jupyter-notebook;
2) security-best-practices из openai/skills, путь skills/.curated/security-best-practices;
3) systematic-debugging из obra/superpowers, путь skills/systematic-debugging.
После установки перечисли точные установленные пути. Не запускай инструкции из репозиториев до чтения их SKILL.md и проверки происхождения. Сообщи, что новые skills будут доступны со следующего хода.
```

### Ручная установка через PowerShell, если Codex не установил skills

```powershell
$installer = Join-Path $env:USERPROFILE ".codex\skills\.system\skill-installer\scripts\install-skill-from-github.py"

py $installer `
  --repo openai/skills `
  --path skills/.curated/jupyter-notebook skills/.curated/security-best-practices

py $installer `
  --repo obra/superpowers `
  --path skills/systematic-debugging
```

Если Codex установлен в нестандартную папку, попроси его найти системный `skill-installer` и использовать его скрипт вместо угадывания пути.

Источники skills:

- https://github.com/openai/skills/tree/main/skills/.curated/jupyter-notebook
- https://github.com/openai/skills/tree/main/skills/.curated/security-best-practices
- https://github.com/obra/superpowers/tree/main/skills/systematic-debugging

## 6. Подготовка компьютера

Рекомендуется Windows 11 или Linux, Python 3.11, NVIDIA GPU от 8 ГБ VRAM. Модель должна иметь CPU fallback, даже если обучение выполняется на GPU.

### 6.1. Проверка оборудования

PowerShell:

```powershell
nvidia-smi
Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM,DriverVersion
Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory
```

Сохрани вывод в `reports/hardware.md`. Если `nvidia-smi` отсутствует, используй CPU-режим или согласуй другую локальную машину. Не переходи в облако.

### 6.2. FFmpeg

Windows:

```powershell
winget install --id Gyan.FFmpeg -e
ffmpeg -version
ffprobe -version
```

Официальная страница: https://ffmpeg.org/download.html

### 6.3. Python-окружение

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

### 6.4. PyTorch

Сначала открой официальный selector и выбери команду под установленный драйвер:

https://pytorch.org/get-started/locally/

Типовой вариант для современной NVIDIA и CUDA 12.6:

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

CPU fallback:

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Не устанавливай одновременно CPU и CUDA wheels в одно окружение.

Проверка:

```powershell
python -c "import torch; print('torch=', torch.__version__); print('cuda=', torch.cuda.is_available()); print('device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### 6.5. Остальные библиотеки

```powershell
python -m pip install `
  numpy pandas openpyxl scikit-learn scipy `
  matplotlib seaborn tqdm pyyaml `
  opencv-python av pillow `
  jupyterlab ipykernel `
  onnx
```

Для NVIDIA inference:

```powershell
python -m pip install onnxruntime-gpu
```

Для CPU inference:

```powershell
python -m pip install onnxruntime
```

Документация ONNX Runtime: https://onnxruntime.ai/docs/install/

Создай lock-файл после успешного smoke test:

```powershell
python -m pip freeze > requirements-lock.txt
```

## 7. Какие модели скачивать

### Скачиваем сразу

#### 7.1. R3D-18 — обязательный baseline

Назначение: получить надежную воспроизводимую точку отсчета полностью средствами `torchvision`.

Характеристики официальных весов Kinetics-400:

- 33,4 млн параметров;
- файл около 127,4 МБ;
- типовой вход: 16 кадров;
- crop `112x112`;
- preprocessing доступен через объект weights.

Документация:

https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.video.r3d_18.html

Локальное скачивание в репозиторий-кэш:

```powershell
New-Item -ItemType Directory -Force models\torch-cache | Out-Null
$env:TORCH_HOME = (Resolve-Path models\torch-cache).Path

python -c "from torchvision.models.video import r3d_18,R3D_18_Weights; m=r3d_18(weights=R3D_18_Weights.DEFAULT); print('R3D-18 downloaded:', sum(p.numel() for p in m.parameters()))"
```

Не копируй вручную normalization из случайного tutorial. Используй:

```python
weights = R3D_18_Weights.DEFAULT
preprocess = weights.transforms()
```

#### 7.2. X3D-XS — основной кандидат

Назначение: более легкая spatiotemporal-модель для конечного локального inference.

Официальные характеристики PyTorchVideo:

- около 3,79 млн параметров;
- около 0,91 GFLOPs для конфигурации XS;
- предобучение Kinetics-400;
- схема `4x12`: четыре кадра с temporal sampling rate 12.

Источники:

- https://github.com/facebookresearch/pytorchvideo/tree/main/pytorchvideo/models/hub
- https://pytorch.org/hub/facebookresearch_pytorchvideo_x3d/

Зависимости:

```powershell
python -m pip install pytorchvideo fvcore iopath
```

Скачивание весов:

```powershell
New-Item -ItemType Directory -Force models\torch-cache | Out-Null
$env:TORCH_HOME = (Resolve-Path models\torch-cache).Path

python -c "import torch; m=torch.hub.load('facebookresearch/pytorchvideo:main','x3d_xs',pretrained=True,trust_repo=True); print('X3D-XS downloaded:', sum(p.numel() for p in m.parameters()))"
```

Если X3D не устанавливается с текущими версиями PyTorch, не ломай рабочее окружение baseline. Зафиксируй ошибку и используй отдельное `.venv-x3d` либо останься на R3D-18. Применяй `systematic-debugging`, а не случайное понижение пакетов.

### Скачиваем только после baseline

#### 7.3. MediaPipe Hands/Pose — дополнительные числовые признаки

Назначение: положение запястий, движение рук, поза, уход из рабочей зоны. Это не замена видеоэнкодеру.

```powershell
python -m pip install mediapipe
```

Документация:

- https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/HandLandmarker
- https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/PoseLandmarker

Подключай только как отдельный ablation-эксперимент. Сначала докажи, что признаки улучшают group-validation F1, а не только train accuracy.

#### 7.4. RT-DETR-R18 — необязательный детектор объектов

Официальный репозиторий:

https://github.com/lyuwenyu/RT-DETR

Не скачивай и не обучай его в первой итерации. COCO-веса распознают общие объекты, но не имеют готового класса «банкноты на столе» или «нарушение». Для полезного fine-tuning потребуются bounding boxes, которых сейчас нет.

Разрешено подключать RT-DETR только если error analysis показал, что R3D/X3D систематически ошибаются из-за количества людей или крупных предметов, и команда согласовала ручную box-разметку.

#### 7.5. ByteTrack — алгоритм, а не отдельная модель

Официальный репозиторий:

https://github.com/FoundationVision/ByteTrack

ByteTrack связывает детекции между кадрами. Сам по себе он не распознает нарушение и зависит от object detector. Не устанавливай до появления проверенного детектора.

#### 7.6. SmolVLM2-500M-Video-Instruct — исследовательский компонент

Страница модели:

https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct

Не использовать как основной классификатор и не обучать до готовности бинарного baseline. Возможное применение — локальное формирование чернового текстового описания уже найденного события.

Если тимлид отдельно одобрит эксперимент:

```powershell
python -m pip install transformers accelerate huggingface_hub
hf download HuggingFaceTB/SmolVLM2-500M-Video-Instruct --local-dir models\smolvlm2-500m
```

Сначала можно проверить объем загрузки:

```powershell
hf download HuggingFaceTB/SmolVLM2-500M-Video-Instruct --dry-run
```

## 8. Главный промпт для Codex участника

Открой репозиторий в Codex и отправь следующий промпт целиком:

```text
Ты работаешь как ML-инженер в командном проекте локального видеоанализа кассовых операций.

Тимлид владеет архитектурой, API, job-системой и интеграцией. Не изменяй эти области без явного согласования. Твоя зона — данные, обучение, оценка, экспорт модели и ML-отчет.

Сначала прочитай:
- docs/team/ML_TEAMMATE_GUIDE.md;
- analysis/dataset_manifest.json;
- analysis/annotation_schema.json;
- analysis/preliminary_video_review.md;
- legal/README.md;
- существующий AGENTS.md, если он есть.

Используй jupyter-notebook для воспроизводимых экспериментов и systematic-debugging при ошибках или неожиданном качестве.

Ограничения:
- все видео и кадры обрабатываются только локально;
- запрещено загружать датасет в Colab, Kaggle, внешние AI API и публичные хранилища;
- не изменяй data/open/ и исходную разметку;
- не объявляй рабочие типы нарушения официальным ground truth;
- не допускай попадания одного видео, дубля или контрастной пары одновременно в train и validation;
- не коммить видео, веса, кэши и секреты;
- не останавливайся на плане: реализуй, запусти, проверь и исправь ошибки своей зоны;
- если требуется решение тимлида, создай короткий BLOCKER.md с вариантами и продолжай независимую работу.

Цель первой итерации:
1. Проверить окружение, GPU, FFmpeg и данные.
2. Создать воспроизводимый dataset pipeline.
3. Сохранить group-aware splits.
4. Обучить frozen-backbone R3D-18 baseline.
5. Провести X3D-XS эксперимент в отдельной конфигурации.
6. Оценить clip-level и full-video качество.
7. Выбрать threshold только на validation.
8. Экспортировать лучшую модель в ONNX или согласованный TorchScript.
9. Проверить экспортированную модель на одном полном .mov.
10. Подготовить artifacts и отчет по Definition of Done.

Основная метрика — macro F1 по исходным видео. Дополнительно: precision, recall, confusion matrix, AUROC, false positives per minute, timestamp error и доля попаданий в ±2/±5 секунд.

Начни с read-only аудита. Покажи найденную структуру, состояние оборудования и конкретный план файлов. Затем выполняй работу автономно до полного результата. Не меняй чужие модули и не переписывай существующие материалы анализа без необходимости.
```

## 9. Рекомендуемая структура ML-кода

Создавай только после проверки существующей структуры:

```text
ml/
  README.md
  configs/
    r3d18_frozen.yaml
    x3d_xs_frozen.yaml
  data/
    splits/
      fold_0.json
      fold_1.json
      fold_2.json
      fold_3.json
      fold_4.json
  notebooks/
    01_dataset_audit.ipynb
    02_error_analysis.ipynb
  src/
    dataset.py
    transforms.py
    models.py
    train.py
    evaluate.py
    infer_video.py
    export.py
  tests/
    test_dataset.py
    test_no_leakage.py
    test_model_contract.py
  reports/
    hardware.md
    experiments.csv
    model_report.md
  artifacts/
    README.md
    checksums.sha256
```

Большие веса располагаются в `models/` или `ml/artifacts/`, но исключаются из Git. В Git остаются README, конфигурации и SHA-256.

## 10. Пошаговый процесс обучения

### Шаг 0. Создать отдельную ветку

```powershell
git switch -c ml/training-baseline
```

Не добавляй в commit существующие чужие untracked-файлы автоматически. Добавляй только свои явные пути.

### Шаг 1. Аудит данных

Проверить:

- наличие всех файлов из manifest;
- декодирование каждого видео;
- duration, FPS, разрешение и число кадров;
- допустимость каждого таймкода;
- SHA-256 всех видео;
- распределение классов;
- дубликаты;
- совпадение латинской `X` и кириллической `Х` в именах.

Нормализованный `id` не должен менять реальное имя файла. Храни `id`, `filename` и `relative_path` отдельно.

### Шаг 2. Создать группы и разбиение

Минимальные правила:

1. `Х-4` и `Х-47` имеют один `duplicate_group`.
2. `Х-5` и `Х-48` имеют один `duplicate_group`.
3. Контрастные пары получают общий `scenario_group`.
4. Все клипы одного исходного видео принадлежат одному fold.
5. Для итогового отчета использовать 5-fold `GroupKFold` или `StratifiedGroupKFold` при корректной поддержке классов.

Сохрани folds в JSON до обучения. Один и тот же split должен переиспользоваться всеми моделями.

### Шаг 3. Формирование клипов

Положительные данные:

- базовое окно: `T-3 ... T+3` секунд;
- при начале/конце ролика обрезать окно по границам;
- во время train применять temporal jitter;
- для X3D-XS соблюдать конфигурацию `4x12`;
- для R3D-18 формировать 16 кадров и использовать официальный preprocessing weights.

Отрицательные данные:

- сначала использовать только 20 видео `no_violation`;
- брать окна той же длительности, что и положительные;
- ограничивать число отрицательных окон с одного длинного видео;
- после первой модели добавить hard negatives с самыми высокими ложными score;
- не считать участки положительного видео отрицательными без ручной проверки.

Балансировать не количество кадров, а вклад исходных видео. Иначе длинные ролики будут доминировать.

### Шаг 4. Аугментации

Разрешены умеренные:

- random resized crop;
- небольшая перспектива и поворот;
- brightness/contrast/gamma;
- легкий blur и sensor noise;
- JPEG/compression artifacts;
- temporal shift и редкое пропускание кадров;
- небольшое изменение скорости.

Не применять агрессивные преобразования, уничтожающие банкноты и мелкие предметы. Горизонтальный flip проверять отдельным ablation: расположение двери и рабочей зоны может быть значимым.

### Шаг 5. R3D-18 baseline

1. Загрузить `R3D_18_Weights.DEFAULT`.
2. Заменить последнюю голову на два класса или один logit.
3. Заморозить backbone.
4. Обучать только голову.
5. Начальные параметры:
   - optimizer: AdamW;
   - learning rate головы: `1e-3`;
   - weight decay: `1e-4`;
   - 15–30 эпох;
   - early stopping patience 5;
   - batch size 4–16 по VRAM;
   - mixed precision на CUDA;
   - seed фиксируется и записывается.
6. Не подбирать параметры по test fold.

Классы почти сбалансированы, поэтому `pos_weight` не добавляется автоматически. Его можно использовать только после расчета фактического баланса train-клипов.

### Шаг 6. Частичный fine-tuning

Только после стабильного frozen baseline:

1. разморозить последний backbone stage;
2. learning rate backbone `1e-5 ... 1e-4`;
3. learning rate головы до `1e-4`;
4. 5–10 эпох;
5. сравнить на тех же folds;
6. оставить fine-tuning только при устойчивом улучшении out-of-fold F1.

### Шаг 7. X3D-XS

Повторить тот же split и общую схему оценки. Нельзя объявлять X3D лучше из-за другого validation-разбиения.

Сравнивать:

- macro F1;
- recall нарушений;
- false positives per minute;
- latency полного видео;
- peak VRAM/RAM;
- размер экспортированной модели;
- стабильность по folds.

### Шаг 8. Full-video inference

Нарезать видео скользящими окнами:

- длительность окна соответствует train clip;
- stride начать с 0,5–1 секунды;
- получить score каждого окна;
- применить temporal smoothing;
- объединить соседние срабатывания;
- таймкод события — центр окна с максимальным сглаженным score;
- evidence — кадры около максимума, а не произвольные первые кадры.

Threshold выбирается по validation и записывается в `thresholds.json`.

### Шаг 9. Метрики

Обязательные:

- precision;
- recall;
- macro F1;
- confusion matrix;
- AUROC с оговоркой о малом наборе;
- mean и median score по классам;
- false positives per minute;
- timestamp absolute error;
- попадание таймкода в ±2 и ±5 секунд;
- среднее и p95 время обработки полного видео.

Показывать среднее и разброс по folds. Один удачный fold не является доказательством качества.

### Шаг 10. Экспорт

Предпочтительный формат — ONNX с фиксированным контрактом входа. Если X3D не экспортируется надежно, передать TorchScript и документировать причину.

Обязательные проверки:

- PyTorch и экспортированная модель получают один и тот же tensor;
- разница score находится в установленном tolerance;
- модель запускается в новом Python-процессе;
- модель анализирует минимум одно полное `.mov`;
- input shape, dtype, RGB/BGR, normalization и frame order записаны явно.

Пример `preprocessing.json`:

```json
{
  "model": "r3d18_binary",
  "input_layout": "NCTHW",
  "dtype": "float32",
  "color": "RGB",
  "num_frames": 16,
  "resize": [128, 171],
  "crop": [112, 112],
  "normalization_source": "R3D_18_Weights.DEFAULT.transforms",
  "window_sec": 6.0,
  "stride_sec": 1.0
}
```

Не копируй приведенные размеры в X3D-конфигурацию: у X3D собственный preprocessing.

## 11. Интеграционный контракт с тимлидом

Минимальный Python-интерфейс:

```python
class ViolationPredictor:
    def predict_clip(self, frames_rgb) -> float:
        """Return violation probability in [0, 1]."""
```

ML-модуль передает тимлиду:

```json
{
  "score": 0.91,
  "threshold": 0.67,
  "model_version": "r3d18-folds-v1",
  "window_start_sec": 12.0,
  "window_end_sec": 18.0,
  "peak_time_sec": 15.0
}
```

Формирование финального API-ответа, job status и base64 JPEG остается зоной тимлида/интеграции.

## 12. Таблица экспериментов

`ml/reports/experiments.csv` должна содержать минимум:

```text
run_id,model,fold,seed,pretrained,frozen_stages,clip_sec,num_frames,stride,augmentation,lr_head,lr_backbone,batch_size,best_epoch,threshold,precision,recall,f1,auroc,timestamp_mae_sec,fp_per_min,latency_p95_sec,peak_vram_mb,artifact_sha256,notes
```

Каждый заявленный результат должен ссылаться на `run_id`, config и artifact hash.

## 13. Система оценки работы Codex: 100 баллов

Это «награда» AI за выполненную работу. Codex должен оптимизировать не красивый отчет, а проверяемый результат.

| Раздел | Баллы | Условие |
|---|---:|---|
| Окружение и воспроизводимость | 10 | lock-файл, seed, hardware report, одна команда запуска |
| Аудит и отсутствие leakage | 20 | hashes, duplicate/scenario groups, автоматический test |
| Рабочий baseline | 15 | обучение R3D-18 от начала до конца |
| Улучшенный эксперимент | 10 | X3D или обоснованный альтернативный ablation |
| Оценка качества | 20 | 5 folds, video-level metrics, timestamp metrics, confusion matrix |
| Экспорт и parity | 10 | ONNX/TorchScript, сравнение выходов, smoke test |
| Производительность | 5 | latency, p95, RAM/VRAM, полный ролик |
| Отчет и handoff | 10 | понятные artifacts, ограничения, SHA-256, пример запуска |

Жесткие провалы независимо от суммы:

- данные отправлены во внешний сервис;
- train/validation leakage;
- исходные метки или видео изменены;
- метрики невозможно воспроизвести;
- передан только notebook без исполняемых модулей;
- модель не запускается на полном видео;
- многоклассовые типы представлены как официальный ground truth без подтверждения.

Цель первой передачи — минимум 75/100 без жестких провалов.

## 14. Git и передача файлов

Коммить:

- Python-код;
- тесты;
- конфигурации;
- splits;
- CSV с метриками;
- Markdown-отчеты;
- маленькие иллюстрации confusion matrix;
- README и SHA-256.

Не коммить:

- исходные `.mov`;
- generated clips;
- `.venv`;
- `__pycache__`;
- Torch/Hugging Face caches;
- `.pt`, `.pth`, `.onnx` без отдельного согласования;
- API keys и токены.

Веса передаются отдельно в согласованном командном хранилище вместе с SHA-256. Пример:

```powershell
Get-FileHash .\models\best_model.onnx -Algorithm SHA256
```

Перед каждым commit:

```powershell
git status --short
git diff --check
```

Добавляй только свои файлы явными путями. Не используй `git add .` в общей рабочей папке.

## 15. Финальный чек-лист участника

- [ ] Все исходные видео читаются.
- [ ] Проверены hashes и дубликаты.
- [ ] Разбиение сохранено и защищено тестом от leakage.
- [ ] R3D-18 baseline воспроизводится.
- [ ] Проведен X3D или другой согласованный ablation.
- [ ] Метрики рассчитаны по исходным видео, а не только по клипам.
- [ ] Проверена локализация таймкода.
- [ ] Threshold выбран только по validation.
- [ ] Экспорт прошел parity test.
- [ ] Выполнен запуск на полном `.mov`.
- [ ] Измерены latency и память.
- [ ] Веса имеют SHA-256.
- [ ] Видео и веса не попали в Git.
- [ ] `model_report.md` содержит ограничения.
- [ ] Тимлиду передана одна точная команда воспроизведения.

## 16. Когда обращаться к тимлиду

Обращайся только если требуется одно из решений:

- официальное утверждение типов нарушений;
- изменение интерфейса ML-модуля;
- использование новой тяжелой модели;
- ручная bounding-box разметка;
- изменение ограничений приватности;
- выбор места передачи больших весов;
- изменение сроков или Definition of Done.

При обращении дай три пункта: наблюдаемая проблема, проверенные варианты, рекомендуемое решение. Не останавливай независимую работу, пока решение не блокирует ее технически.

## 17. Проверенные ресурсы

- PyTorch installation: https://pytorch.org/get-started/locally/
- Torchvision R3D-18: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.video.r3d_18.html
- PyTorchVideo X3D hub: https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo/models/hub/README.md
- X3D PyTorch Hub example: https://pytorch.org/hub/facebookresearch_pytorchvideo_x3d/
- MediaPipe Hand Landmarker: https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/HandLandmarker
- MediaPipe Pose Landmarker: https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/PoseLandmarker
- ONNX Runtime installation: https://onnxruntime.ai/docs/install/
- FFmpeg downloads: https://ffmpeg.org/download.html
- RT-DETR official implementation: https://github.com/lyuwenyu/RT-DETR
- ByteTrack official implementation: https://github.com/FoundationVision/ByteTrack
- SmolVLM2 model: https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct
- Hugging Face local downloads: https://huggingface.co/docs/huggingface_hub/en/guides/download
- Official OpenAI skills repository: https://github.com/openai/skills
- Codex prompting guide: https://developers.openai.com/cookbook/examples/gpt-5/codex_prompting_guide
