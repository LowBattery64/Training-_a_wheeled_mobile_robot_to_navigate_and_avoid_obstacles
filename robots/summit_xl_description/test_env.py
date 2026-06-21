# emergency_debug.py
import mujoco
import numpy as np

XML_PATH = "summit_xls_empty.xml"

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# Копируем reset из env
mujoco.mj_resetData(model, data)
data.qpos[0] = -3.5
data.qpos[1] = -4.0

for _ in range(100):
    mujoco.mj_step(model, data)

data.qvel[:] = 0
mujoco.mj_forward(model, data)

print(f"Height: {data.qpos[2]:.3f}")
print(f"Contacts: {data.ncon}")

# Покажем контакты (исправленный API)
for i in range(min(5, data.ncon)):
    contact = data.contact[i]
    g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) if contact.geom1 >= 0 else "None"
    g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) if contact.geom2 >= 0 else "None"
    print(f"  Contact {i}: {g1} <-> {g2}")

print(f"\nActuator names:")
for i in range(model.nu):
    print(f"  ctrl[{i}] = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)}")

print(f"\nSetting ctrl = [10, 10, 10, 10]")
data.ctrl[:] = [10, 10, 10, 10]

print(f"ctrl values: {data.ctrl}")

old_pos = data.qpos[0:2].copy()

for step in range(10):
    mujoco.mj_step(model, data)
    pos = data.qpos[0:2]
    vel = data.qvel[0:2]
    print(f"Step {step}: pos=({pos[0]:.3f}, {pos[1]:.3f}), vel=({vel[0]:.3f}, {vel[1]:.3f}), ctrl={data.ctrl}")

moved = np.linalg.norm(data.qpos[0:2] - old_pos)
print(f"\nTotal moved: {moved:.4f} meters")

# Проверим скорости джойнтов колес
print(f"\nWheel joint velocities:")
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    if name and "wheel_rolling" in str(name):
        dof_idx = model.jnt_dofadr[i]
        print(f"  {name}: qvel[{dof_idx}] = {data.qvel[dof_idx]:.4f}")

# Дополнительно: проверим, есть ли крутящий момент на колесах
print(f"\nActuator forces (qfrc_actuator):")
for i in range(model.nu):
    print(f"  {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)}: {data.qfrc_actuator[i]:.4f}")