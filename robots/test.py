# test.py
from stable_baselines3 import SAC
from env_summit import SummitEnv
import time
import os

MODEL_PATH = "models/summit_staged_600000_steps"   # или "models/summit_sac_final"

if not os.path.exists(MODEL_PATH + ".zip"):
    MODEL_PATH = "summit_sac"      # старое имя если есть

env   = SummitEnv("summit_xls.xml", render_mode="human")
model = SAC.load(MODEL_PATH)

print("START TEST")
obs, _ = env.reset()
episode = 0

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    time.sleep(0.02)

    print(f"pos=({env._get_robot_xy()[0]:.2f}, {env._get_robot_xy()[1]:.2f})  "
          f"goal=({env._get_goal_xy()[0]:.2f}, {env._get_goal_xy()[1]:.2f})  "
          f"dist={info['dist_to_goal']:.2f}")

    if terminated or truncated:
        episode += 1
        result = "ДОСТИГ ЦЕЛИ!" if terminated else "Время вышло"
        print(f"\n--- Эпизод {episode}: {result} ---\n")
        obs, _ = env.reset()
