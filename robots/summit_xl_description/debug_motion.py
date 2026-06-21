import mujoco
import mujoco.viewer
import numpy as np
import time

XML_PATH = "summit_xls.xml"

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# Сбросим в начальную позицию
mujoco.mj_resetData(model, data)
data.qpos[0] = -3.5
data.qpos[1] = -4.0
mujoco.mj_forward(model, data)

print("Initial position:", data.qpos[0:2])

with mujoco.viewer.launch_passive(model, data) as viewer:
    step = 0
    while viewer.is_running():
        # Подаем МАКСИМАЛЬНУЮ силу на все колеса ВПЕРЕД
        if step < 500:
            data.ctrl[0] = 10.0  # front_right
            data.ctrl[1] = 10.0  # front_left
            data.ctrl[2] = 10.0  # back_right
            data.ctrl[3] = 10.0  # back_left
        elif step < 1000:
            # Подаем силу в разных направлениях
            data.ctrl[0] = 10.0   # front_right вперед
            data.ctrl[1] = -10.0  # front_left назад
            data.ctrl[2] = -10.0  # back_right назад
            data.ctrl[3] = 10.0   # back_left вперед
        else:
            data.ctrl[:] = 0.0
        
        mujoco.mj_step(model, data)
        viewer.sync()
        
        if step % 100 == 0:
            print(f"Step {step}: pos={data.qpos[0:2]}, vel={data.qvel[0:2]}")
        
        step += 1
        time.sleep(0.001)