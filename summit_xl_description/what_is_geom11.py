# what_is_geom11.py — выясняем что такое geom id=11 и почему не фильтруется
import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path("summit_xls.xml")

print("=== Geom 11 ===")
gid = 11
name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
body_id = model.geom_bodyid[gid]
body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
print(f"  name: {name!r}")
print(f"  body_id: {body_id}, body_name: {body_name!r}")
print(f"  pos: {model.geom_pos[gid]}")
print(f"  size: {model.geom_size[gid]}")
print(f"  type: {model.geom_type[gid]}  (0=plane,2=sphere,3=capsule,4=ellipsoid,5=cyl,6=box)")
print(f"  group: {model.geom_group[gid]}")
print()

print("=== Иерархия тел от base_footprint ===")
base_fp_id = model.body("base_footprint").id
print(f"base_footprint id: {base_fp_id}")

# Все тела потомки base_footprint
descendants = set()
for i in range(model.nbody):
    body = i
    while body > 0:
        if body == base_fp_id:
            descendants.add(i)
            break
        body = model.body_parentid[body]

print(f"Тела-потомки: {sorted(descendants)}")

# Все geom этих тел
robot_geoms = set()
for i in range(model.ngeom):
    if model.geom_bodyid[i] in descendants:
        robot_geoms.add(i)

print(f"Geom робота ({len(robot_geoms)} шт): {sorted(robot_geoms)[:20]}...")
print(f"Geom 11 в robot_geoms: {11 in robot_geoms}")
print()

# Что за тело у geom 11
bid = model.geom_bodyid[11]
print(f"Body chain для geom 11:")
body = bid
chain = []
while body >= 0:
    bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body)
    chain.append(f"{body}({bname!r})")
    if body == 0:
        break
    body = model.body_parentid[body]
print("  " + " → ".join(chain))
