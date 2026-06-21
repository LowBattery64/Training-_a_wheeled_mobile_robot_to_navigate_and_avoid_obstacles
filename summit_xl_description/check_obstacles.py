# check_obstacles.py — проверяем реально ли меняются препятствия
from env_summit import SummitEnv
import numpy as np
import mujoco

env = SummitEnv("summit_xls.xml", difficulty=1.0)

print("=== Проверка: меняются ли препятствия? ===\n")

obs_names = ["block_center", "block_left", "block_right", "sphere_obs"]

for episode in range(3):
    obs, _ = env.reset()
    print(f"Эпизод {episode+1}:")
    for name in obs_names:
        gid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid >= 0:
            # geom_pos в model — что мы записали
            model_pos = env.model.geom_pos[gid]
            # xpos в data — что реально в симуляции
            data_pos  = env.data.geom_xpos[gid]
            print(f"  {name}: model_pos=({model_pos[0]:.2f},{model_pos[1]:.2f})  "
                  f"data_xpos=({data_pos[0]:.2f},{data_pos[1]:.2f})")
    print()

env.close()
print("Если data_xpos одинаковый во всех эпизодах — препятствия НЕ двигаются.")
print("Если model_pos != data_xpos — mj_resetData сбрасывает изменения.")
