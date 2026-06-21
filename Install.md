# Инструкция по установке и запуску

---
## 1. Установка Python-окружения

### Windows (PowerShell)

```powershell
# Проверить версию Python
python --version

# Перейти в папку проекта
cd путь\к\repo\robots\summit_xl_description

# Создать виртуальное окружение
python -m venv venv
venv\Scripts\Activate.ps1

# Если PowerShell блокирует скрипты:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Linux / macOS

```bash
python3 --version

cd path/to/repo/robots/summit_xl_description

python3 -m venv venv
source venv/bin/activate
```

---
## 2. Установка зависимостей

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Если файла `requirements.txt` нет рядом — установите вручную:

```bash
pip install mujoco gymnasium stable-baselines3 torch numpy
```

### Проверка установки MuJoCo

```bash
python -c "import mujoco; print(mujoco.__version__)"
```

Ожидаемый вывод — строка версии (например `3.9.0`), без ошибок импорта.

---
## 3. Проверка работоспособности окружения

Перед обучением или тестированием **обязательно** прогоните smoke-test:

```bash
python env_summit.py
```

### Ожидаемый вывод (сокращённо)

```
=== Smoke test (MCAL-style env) ===

obs shape: (80,)  (8 nav + 24×3 lidar = 80)

Лидар (24 луча, шаг 1):
   0 (  0°): 1.00 ████████
   ...

Едем вперёд 150 шагов...
  step   0: pos=(-3.50,-4.00)  dist=7.00м  r=-0.02
  step  30: pos=(-3.47,-4.00)  dist=6.97м  r=0.20
  ...

Пройдено: X.XXм  reward: XX.X
```
## 4. Запуск готовой модели (визуальная демонстрация)

```bash
python test_scenes.py
```

1. Выберите модель из списка (помечены `[✓]`, если файл найден).
2. Выберите тестовую сцену.
3. Откроется окно MuJoCo viewer с симуляцией в реальном времени.
4. `Ctrl+C` в терминале — остановка.

> Если ни одна модель не помечена `[✓]` — либо вы клонировали репозиторий без `models/` (исключены через `.gitignore` из-за размера), либо нужно сначала обучить модель самостоятельно (см. ниже).

---

## 5. Обучение модели с нуля

```bash
python train.py
```

- Полный цикл — 1 500 000 шагов, по умолчанию разбит на 4 этапа curriculum (см. README.md).
- Прогресс пишется в консоль (`rollout/ep_rew_mean`, `eval/mean_reward` и т.д.) и в `logs/` для TensorBoard.
- Чекпоинты каждые 100k шагов — `models/summit_staged_<N>_steps.zip`.
- Обучение можно прервать в любой момент `Ctrl+C` — текущий прогресс не теряется (последний чекпоинт уже на диске), а `train.py` сохранит финальный снимок в `models/summit_staged_final.zip`.

### Просмотр прогресса в реальном времени

```bash
tensorboard --logdir logs
```

Откройте `http://localhost:6006` в браузере.

### Дообучение существующего чекпоинта

Откройте `train.py`, найдите блок:

```python
# ЯВНО укажите здесь, если знаете точно (перекрывает автопоиск):
# RESUME_PATH = "models/summit_staged_600000_steps.zip"
# RESUME_FROM_STEP = 600_000
```

Раскомментируйте обе строки, подставьте нужный путь и число шагов, сохраните и запустите `python train.py` заново.

---
## 6. Структура вывода `train.py` — как читать логи

| Метрика | Что значит |
|---|---|
| `rollout/ep_rew_mean` | Средняя награда за последние ~100 эпизодов сбора опыта (не eval) |
| `rollout/ep_len_mean` | Средняя длина эпизода в шагах. Падает, когда агент чаще достигает цели быстрее |
| `eval/mean_reward` | Награда на отдельной eval-среде, детерминированная политика (без шума) — основной индикатор реального качества |
| `eval/mean_ep_length` | То же для длины эпизода |
| `train/critic_loss` | Ошибка Q-функции — должна стабилизироваться на низких значениях |
| `train/ent_coef` | Коэффициент энтропии (SAC сам его подстраивает) — снижается по мере того, как политика становится увереннее |

`[Curriculum] Step N: <описание этапа>` в логе — переключение сцены/сложности согласно расписанию в `train.py`.
