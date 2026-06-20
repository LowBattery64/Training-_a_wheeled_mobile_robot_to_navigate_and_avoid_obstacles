# env_summit.py — по мотивам MCAL (Choi et al. 2021)
# Ключевые изменения vs предыдущей версии:
# 1. Лидар за 3 шага подряд (как в статье: slidar = [t-2, t-1, t])
# 2. Эпизод завершается СРАЗУ при столкновении (штраф -10, как в статье)
# 3. Штраф за угловое вращение (Rω из статьи)
# 4. Исправлен лидар — start_marker исключён через geomgroup

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
from collections import deque


class SummitEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    START_XY = np.array([-3.5, -4.0])
    GOAL_XY  = np.array([ 3.5, -4.0])
    BASE_Z   = 0.247

    LIDAR_RAYS   = 24       # больше лучей — лучше картина
    LIDAR_MAX    = 4.0      # дальность 4м
    LIDAR_HEIGHT = 0.35     # высота над полом

    def __init__(self, xml_path="summit_xls.xml", render_mode=None,
                 difficulty=1.0):
        super().__init__()
        self.render_mode = render_mode
        self.xml_path    = xml_path
        self.difficulty  = difficulty

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)

        self._base_body_id = self.model.body("base").id
        try:
            self._goal_body_id   = self.model.body("goal_marker").id
            self._goal_xy_static = None
        except KeyError:
            self._goal_body_id   = None
            self._goal_xy_static = self.GOAL_XY.copy()

        self._floor_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        self._base_footprint_id = self.model.body("base_footprint").id
        # mj_ray bodyexclude не исключает дочерние тела — используем -1
        # и вместо этого фильтруем геомы робота постфактум
        self._robot_geom_ids = set()
        for i in range(self.model.ngeom):
            bid = self.model.geom_bodyid[i]
            # Собираем все geom принадлежащие телам робота
            # Тела робота: base_footprint и все его потомки
            body = bid
            while body > 0:
                if body == self._base_footprint_id:
                    self._robot_geom_ids.add(i)
                    break
                body = self.model.body_parentid[body]

        self._wheel_qvel_adr = [
            self.model.joint(n).dofadr[0] for n in [
                "front_right_wheel_rolling_joint",
                "front_left_wheel_rolling_joint",
                "back_right_wheel_rolling_joint",
                "back_left_wheel_rolling_joint",
            ]
        ]

        self._obs_geom_names = []
        for name in ["block_center", "block_left", "block_right", "sphere_obs", "cyl_obs"]:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                self._obs_geom_names.append(name)

        # Буфер лидара за 3 шага (как в статье)
        self._lidar_buffer = deque(maxlen=3)

        # obs = nav(8) + lidar_t(24) + lidar_t-1(24) + lidar_t-2(24) = 80
        n_obs = 8 + self.LIDAR_RAYS * 3
        self.action_space = spaces.Box(
            low=-15.0, high=15.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32)

        self.max_steps      = 1500
        self.goal_threshold = 0.5
        self._step_count    = 0
        self._prev_dist     = 0.0

        self._viewer   = None
        self._renderer = None

    # ── helpers ──────────────────────────────────────────────────────

    def _get_robot_xy(self):
        return self.data.xpos[self._base_body_id][:2].copy()

    def _get_goal_xy(self):
        if self._goal_body_id is not None:
            return self.data.xpos[self._goal_body_id][:2].copy()
        return self._goal_xy_static.copy()

    def _get_yaw(self):
        qw, qx, qy, qz = (self.data.qpos[3], self.data.qpos[4],
                           self.data.qpos[5], self.data.qpos[6])
        return float(np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz)))

    def _cast_lidar(self):
        """24 луча вокруг робота. geomgroup маска исключает маркеры (group=2)."""
        robot_xy = self._get_robot_xy()
        yaw = self._get_yaw()
        pnt = np.array([robot_xy[0], robot_xy[1], self.LIDAR_HEIGHT],
                       dtype=np.float64)
        distances = np.ones(self.LIDAR_RAYS, dtype=np.float32)

        # Маска: видим ТОЛЬКО group=0 (стены, препятствия)
        # Корпус робота group=1, колёса group=3, маркеры group=2 — все игнорируются
        gmask = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)

        for i in range(self.LIDAR_RAYS):
            angle = yaw + (2 * np.pi * i / self.LIDAR_RAYS)
            vec = np.array([np.cos(angle), np.sin(angle), 0.0], dtype=np.float64)
            geomid = np.array([-1], dtype=np.int32)
            dist = mujoco.mj_ray(
                self.model, self.data, pnt, vec,
                gmask, 1, -1, geomid)
            gid = geomid[0]
            if (0.02 < dist < self.LIDAR_MAX
                    and gid != self._floor_geom_id
                    and gid not in self._robot_geom_ids):
                distances[i] = dist / self.LIDAR_MAX
        return distances

    def _get_obs(self):
        robot_xy = self._get_robot_xy()
        goal_xy  = self._get_goal_xy()
        to_goal  = goal_xy - robot_xy
        dist     = float(np.linalg.norm(to_goal))
        yaw      = self._get_yaw()
        cy, sy   = np.cos(-yaw), np.sin(-yaw)
        local_x  = cy * to_goal[0] - sy * to_goal[1]
        local_y  = sy * to_goal[0] + cy * to_goal[1]
        vx = float(self.data.qvel[0])
        vy = float(self.data.qvel[1])
        wz = float(self.data.qvel[5])

        nav = np.array([
            local_x / 7.0, local_y / 7.0, dist / 7.0,
            vx / 2.0, vy / 2.0, wz / 2.0,
            np.cos(yaw), np.sin(yaw),
        ], dtype=np.float32)

        # Текущий лидар + добавляем в буфер
        lidar_now = self._cast_lidar()
        self._lidar_buffer.append(lidar_now)

        # Собираем 3 шага (если буфер не заполнен — дублируем первый)
        while len(self._lidar_buffer) < 3:
            self._lidar_buffer.appendleft(lidar_now)

        lidar_stack = np.concatenate(list(self._lidar_buffer))
        return np.concatenate([nav, lidar_stack])

    def _get_wall_contact(self):
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 != self._floor_geom_id and c.geom2 != self._floor_geom_id:
                return True
        return False

    def _randomize_obstacles(self):
        base_positions = {
            "block_center": np.array([0.0,  -2.5, 0.4]),
            "block_left":   np.array([-0.5, -1.0, 0.4]),
            "block_right":  np.array([1.5,  -2.5, 0.4]),
            "sphere_obs":   np.array([2.0,  -3.0, 0.5]),
            "cyl_obs":      np.array([-0.5, -1.5, 0.5]),
        }
        for name in self._obs_geom_names:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid < 0:
                continue
            if np.random.random() > self.difficulty:
                self.model.geom_pos[gid] = np.array([20.0, 20.0, 0.4])
            else:
                base  = base_positions.get(name, np.array([0.0, 0.0, 0.4]))
                noise = np.random.uniform(-1.0, 1.0, size=2) * self.difficulty
                pos   = base.copy()
                pos[0] = np.clip(pos[0] + noise[0], -4.0, 4.0)
                pos[1] = np.clip(pos[1] + noise[1], -4.5, 4.5)
                self.model.geom_pos[gid] = pos
        mujoco.mj_forward(self.model, self.data)

    # ── gymnasium api ────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[0] = self.START_XY[0]
        self.data.qpos[1] = self.START_XY[1]
        self.data.qpos[2] = self.BASE_Z
        self.data.qpos[3] = 1.0
        self.data.qpos[4:7] = 0.0
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0

        self._randomize_obstacles()

        for _ in range(20):
            mujoco.mj_step(self.model, self.data)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        self._lidar_buffer.clear()
        self._prev_dist = float(
            np.linalg.norm(self._get_goal_xy() - self._get_robot_xy()))

        return self._get_obs(), {}

    def step(self, action):
        self._step_count += 1
        self.data.ctrl[:] = np.clip(action, -15.0, 15.0)

        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        robot_xy = self._get_robot_xy()
        goal_xy  = self._get_goal_xy()
        dist     = float(np.linalg.norm(goal_xy - robot_xy))

        # ── Награда по статье (адаптировано) ────────────────────────
        # Rg: потенциальная награда за приближение
        progress = self._prev_dist - dist
        reward = progress * 10.0

        # Бонус за скорость в направлении цели
        vx = float(self.data.qvel[0])
        vy = float(self.data.qvel[1])
        to_goal_dir = (goal_xy - robot_xy) / max(dist, 0.01)
        vel_toward  = vx * to_goal_dir[0] + vy * to_goal_dir[1]
        reward += max(0.0, vel_toward) * 0.3

        # Rω: штраф за угловое вращение (из статьи)
        wz = float(self.data.qvel[5])
        if abs(wz) > 0.5:
            reward -= 0.1 * abs(wz)

        # Штраф за время
        reward -= 0.02

        self._prev_dist = dist

        # ── Завершение ───────────────────────────────────────────────
        terminated = dist < self.goal_threshold

        # Rc: СРАЗУ завершаем при столкновении (как в статье)
        collision = self._get_wall_contact()
        if collision:
            reward -= 10.0
            truncated = True
        else:
            truncated = self._step_count >= self.max_steps

        if terminated:
            reward += 100.0

        return self._get_obs(), reward, terminated, truncated, {
            "dist_to_goal": dist,
            "step":         self._step_count,
            "collision":    collision,
        }

    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(
                    self.model, self.data)
            self._viewer.sync()
        elif self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(
                    self.model, height=480, width=640)
            self._renderer.update_scene(self.data)
            return self._renderer.render()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        self._renderer = None


if __name__ == "__main__":
    print("=== Smoke test (MCAL-style env) ===\n")
    env = SummitEnv("summit_xls.xml", difficulty=1.0)
    obs, _ = env.reset()
    print(f"obs shape: {obs.shape}  (8 nav + 24×3 lidar = 80)")

    lidar_now = obs[8:32]
    print(f"\nЛидар (24 луча, шаг 1):")
    for i, d in enumerate(lidar_now):
        angle = int(360 * i / 24)
        bar = "█" * int(d * 8)
        mark = " ←" if d < 0.4 else ""
        print(f"  {i:2d} ({angle:3d}°): {d:.2f} {bar}{mark}")

    print(f"\nЕдем вперёд 150 шагов...")
    total_r = 0.0
    for i in range(150):
        obs, r, term, trunc, info = env.step(
            np.array([10., 10., 10., 10.], dtype=np.float32))
        total_r += r
        if i % 30 == 0:
            xy = env._get_robot_xy()
            print(f"  step {i:3d}: pos=({xy[0]:.2f},{xy[1]:.2f})  "
                  f"dist={info['dist_to_goal']:.2f}м  r={r:.2f}")
        if term or trunc:
            print(f"  → {'ЦЕЛЬ!' if term else 'столкновение/timeout'} на шаге {i}")
            break

    moved = np.linalg.norm(env._get_robot_xy() - env.START_XY)
    print(f"\nПройдено: {moved:.2f}м  reward: {total_r:.1f}")
    env.close()