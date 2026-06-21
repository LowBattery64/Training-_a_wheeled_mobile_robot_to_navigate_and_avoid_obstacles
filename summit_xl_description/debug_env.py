from env_summit import SummitEnv
import numpy as np

env = SummitEnv("summit_xls.xml")
floor_id = env.model.geom("ground").id

def wall_contacts(env):
    return sum(
        1 for j in range(env.data.ncon)
        if env.data.contact[j].geom1 != floor_id
        and env.data.contact[j].geom2 != floor_id
    )

print("Тест 1: едем ТОЛЬКО вперёд до конца (300 шагов)")
print("="*55)
obs, _ = env.reset()
for i in range(300):
    obs, reward, term, trunc, info = env.step(
        np.array([10.0, 10.0, 10.0, 10.0], dtype=np.float32))
    if i % 30 == 0:
        xy = env._get_robot_xy()
        wc = wall_contacts(env)
        print(f"step {i:3d}: pos=({xy[0]:.2f},{xy[1]:.2f})  "
              f"wall_contacts={wc}  dist={info['dist_to_goal']:.2f}м")
    if term or trunc:
        print(f"  → Эпизод завершён на шаге {i}: {'ЦЕЛЬ!' if term else 'timeout'}")
        break

print()
print("Тест 2: стрейф вправо (+Y направление)")
print("="*55)
obs, _ = env.reset()
for i in range(150):
    # стрейф: FR-, FL+, BR+, BL-  → движение в +Y (проверяем)
    obs, reward, term, trunc, info = env.step(
        np.array([-10.0, 10.0, 10.0, -10.0], dtype=np.float32))
    if i % 30 == 0:
        xy = env._get_robot_xy()
        print(f"step {i:3d}: pos=({xy[0]:.2f},{xy[1]:.2f})")

print()
print("Тест 3: вперёд + объезд (вперёд 80 шагов, потом стрейф)")
print("="*55)
obs, _ = env.reset()
for i in range(200):
    if i < 80:
        action = np.array([10.0, 10.0, 10.0, 10.0], dtype=np.float32)
    else:
        # объезд: вперёд + стрейф в -Y (туда где место)
        action = np.array([5.0, 15.0, -5.0, 15.0], dtype=np.float32)
    obs, reward, term, trunc, info = env.step(action)
    if i % 20 == 0:
        xy = env._get_robot_xy()
        wc = wall_contacts(env)
        print(f"step {i:3d}: pos=({xy[0]:.2f},{xy[1]:.2f})  "
              f"wall={wc}  dist={info['dist_to_goal']:.2f}м")
    if term:
        print("  → ЦЕЛЬ ДОСТИГНУТА!")
        break

env.close()
