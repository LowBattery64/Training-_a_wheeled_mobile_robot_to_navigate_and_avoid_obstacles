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
