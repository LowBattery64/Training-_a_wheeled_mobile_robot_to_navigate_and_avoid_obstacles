# debug_lidar.py v2 — использует _robot_geom_ids из env
from env_summit import SummitEnv
import numpy as np
import mujoco

env = SummitEnv("summit_xls.xml", difficulty=0.0)
obs, _ = env.reset()

robot_xy = env._get_robot_xy()
print(f"Робот: ({robot_xy[0]:.3f}, {robot_xy[1]:.3f})")
print(f"Robot geom ids (исключаются): {sorted(env._robot_geom_ids)}")
print(f"Floor geom id: {env._floor_geom_id}")
print()

pnt = np.array([robot_xy[0], robot_xy[1], env.LIDAR_HEIGHT], dtype=np.float64)
gmask = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)  # только group=0

print(f"{'Луч':>4} {'Угол':>6} {'Dist':>7} {'Norm':>6} {'GeomID':>7} {'GeomName'}")
print("-"*70)
for i in range(24):
    angle = (2 * np.pi * i / 24)
    vec = np.array([np.cos(angle), np.sin(angle), 0.0], dtype=np.float64)
    geomid = np.array([-1], dtype=np.int32)
    dist = mujoco.mj_ray(env.model, env.data, pnt, vec, gmask, 1, -1, geomid)
    gid = geomid[0]
    gname = ""
    if gid >= 0:
        n = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        gname = n if n else f"id={gid}"
    
    is_robot = gid in env._robot_geom_ids
    is_floor = gid == env._floor_geom_id
    
    if is_robot or is_floor:
        norm = "SKIP"
        flag = " ← РОБОТ" if is_robot else " ← ПОЛ"
    elif 0 < dist < env.LIDAR_MAX:
        norm = f"{dist/env.LIDAR_MAX:.3f}"
        flag = ""
    else:
        norm = "1.000"
        flag = " (нет попадания)" if dist < 0 else " (>макс)"
    
    print(f"{i:>4} {np.degrees(angle):>5.1f}° {dist:>7.4f} {norm:>6} {gid:>7} {gname}{flag}")

print()
print("Ожидаем:")
print(f"  270° (юг): стена wall1 на y=-4.9, расстояние={4.9-4.0:.1f}м → norm={0.9/4.0:.3f}")
print(f"  180° (запад): стена wall4 на x=-4.9, расстояние={4.9-3.5:.1f}м → norm={1.4/4.0:.3f}")
env.close()