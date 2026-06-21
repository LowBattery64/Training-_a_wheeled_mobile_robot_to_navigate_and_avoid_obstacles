# test_scenes.py
from stable_baselines3 import SAC
import time, os, sys
import numpy as np

import math

# Формат: (xml, название, старт_xy, цель_xy, старт_yaw)
# старт_yaw в радианах: 0 = смотрит на восток (+X), -pi/2 = на юг (-Y),
# pi/2 = на север (+Y), pi = на запад (-X)
SCENES = {
    "1": ("scene_maze.xml",       "Лабиринт",                       (0,0),       (3.5, -4.0), -math.pi/2),
    "2": ("scene_random.xml",     "Случайные препятствия",          (-3.5,-4.0), (3.5,-4.0),   0.0),
    "3": ("scene_zigzag.xml",     "Зигзаг",                         (-3.5,-4.0), (3.5, 3.5),   0.0),
    "4": ("scene_final_test.xml", "Финальный тест (5 препятствий)", (-3.5,-4.0), (3.5,-4.0),   0.0),
}

# Доступные модели
MODELS = {
    "1": ("models/best_model",         "Лучшая сохранённая"),
    "2": ("models/summit_staged_final","Staged curriculum (новая)"),
    "3": ("best_old_model",            "Старая модель (без лидара)"),
    "4": ("models/summit_staged_600000_steps",          "Новая лучшая (с лидаром)"),
}

print("Выберите модель:")
for k, (path, name) in MODELS.items():
    exists = os.path.exists(path + ".zip") or os.path.exists(path)
    status = "✓" if exists else "✗ нет файла"
    print(f"  {k}. {name} [{status}]")

model_choice = input("\nВведите 1, 2 или 3: ").strip()
if model_choice not in MODELS:
    print("Неверный выбор")
    sys.exit(1)

MODEL_PATH = MODELS[model_choice][0]
model = SAC.load(MODEL_PATH)

# Определяем размер obs у загруженной модели
obs_size = model.observation_space.shape[0]
print(f"\nМодель ожидает obs размером: {obs_size}")

# Выбираем нужный класс env по размеру obs
if obs_size == 24:
    print("Используем старый env (summit_env_old.py, без лидара)")
    from summit_env_old import SummitEnv
elif obs_size == 80:
    print("Используем новый env (env_summit.py, с лидаром)")
    from env_summit import SummitEnv
else:
    print(f"Неизвестный obs size: {obs_size}")
    sys.exit(1)

print("\nВыберите сцену:")
for k, (f, name, start, goal, yaw) in SCENES.items():
    print(f"  {k}. {name}  (старт {start} → цель {goal})")

scene_choice = input("\nВведите 1, 2, 3 или 4: ").strip()
if scene_choice not in SCENES:
    print("Неверный выбор")
    sys.exit(1)

scene_file, scene_name, start_xy, goal_xy, start_yaw = SCENES[scene_choice]

# Создаём временный XML с нужной сценой
with open("summit_xls.xml") as f:
    xml = f.read()

for old in [
    '<include file="assets/basic_scene.xml" />',
    '<include file="assets/basic_scene_fixed.xml" />',
    '<include file="scene_stage1.xml" />',
    '<include file="scene_stage2.xml" />',
    '<include file="scene_stage3.xml" />',
    '<include file="scene_stage4.xml" />',
]:
    if old in xml:
        xml = xml.replace(old, f'<include file="{scene_file}" />')
        break

with open("_tmp_test.xml", "w") as f:
    f.write(xml)

# Создаём env с правильным obs space
class TestEnv(SummitEnv):
    START_XY  = np.array(list(start_xy))
    GOAL_XY   = np.array(list(goal_xy))
    START_YAW = start_yaw

env = TestEnv("_tmp_test.xml", render_mode="human")

# Проверяем совместимость
env_obs_size = env.observation_space.shape[0]
if env_obs_size != obs_size:
    print(f"\n⚠ Несовместимость: модель ожидает {obs_size}, env даёт {env_obs_size}")
    print("Используйте модель обученную с текущим env_summit.py")
    os.remove("_tmp_test.xml")
    sys.exit(1)

print(f"\nЗагружаем: {scene_name}")
print(f"Старт: {start_xy}  →  Цель: {goal_xy}")
print("Нажмите Ctrl+C для остановки\n")

episode = 0
obs, _ = env.reset()

try:
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        time.sleep(0.02)

        xy = env._get_robot_xy()
        print(f"pos=({xy[0]:.2f}, {xy[1]:.2f})  "
              f"dist={info['dist_to_goal']:.2f}м", end="\r")

        if terminated or truncated:
            episode += 1
            result = "✓ ЦЕЛЬ!" if terminated else "✗ Не добрался"
            print(f"\n--- Эпизод {episode}: {result} ---\n")
            obs, _ = env.reset()
finally:
    env.close()
    if os.path.exists("_tmp_test.xml"):
        os.remove("_tmp_test.xml")
