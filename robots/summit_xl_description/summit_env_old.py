# env_summit.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco


class SummitEnv(gym.Env):
    """
    Gymnasium-среда для Summit XLS (mecanum) в MuJoCo.
    Задача: доехать из (-3.5, -4.0) до цели (3.5, -4.0), обходя препятствия.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    # Стартовая позиция и цель (должны совпадать с basic_scene_fixed.xml)
    START_XY  = np.array([-3.5, -4.0])
    GOAL_XY   = np.array([ 3.5, -4.0])

    # Геометрия колеса и базы (из URDF)
    WHEEL_RADIUS = 0.120          # м
    WB_X         = 0.2225         # полубаза по X
    WB_Y         = 0.2045         # полубаза по Y

    # Высота корпуса над полом после посадки
    BASE_Z = 0.247                # base_footprint.z(-0.009) + base.z(0.127) + колесо(0.120) ≈ 0.247

    def __init__(self, xml_path="summit_xls.xml", render_mode=None):
        super().__init__()

        self.render_mode = render_mode
        self.xml_path    = xml_path

        # ── Загружаем модель ──────────────────────────────────────────
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)

        # Запомним ID тел и суставов один раз
        self._base_body_id = self.model.body("base").id
        # goal_marker body (нужен basic_scene_fixed.xml)
        try:
            self._goal_body_id = self.model.body("goal_marker").id
            self._goal_xy_static = None
        except KeyError:
            self._goal_body_id = None
            self._goal_xy_static = np.array([3.5, -4.0])  # fallback — хардкод

        # Индексы qvel/qpos для free-joint корпуса
        # free joint: qpos[0:7] = [x, y, z, qw, qx, qy, qz]
        #             qvel[0:6] = [vx, vy, vz, wx, wy, wz]
        self._free_qpos_start = 0
        self._free_qvel_start = 0

        # Имена суставов колёс (порядок как в actuator: FR, FL, BR, BL)
        self._wheel_joint_names = [
            "front_right_wheel_rolling_joint",
            "front_left_wheel_rolling_joint",
            "back_right_wheel_rolling_joint",
            "back_left_wheel_rolling_joint",
        ]
        self._wheel_joint_ids = [
            self.model.joint(n).id for n in self._wheel_joint_names
        ]
        # адреса в qvel для каждого колеса
        self._wheel_qvel_adr = [
            self.model.joint(n).dofadr[0] for n in self._wheel_joint_names
        ]

        # ── Пространства ──────────────────────────────────────────────
        # action: желаемые скорости колёс [FR, FL, BR, BL] в рад/с
        # (actuator velocity, ctrlrange -15..15)
        self.action_space = spaces.Box(
            low=-15.0, high=15.0, shape=(4,), dtype=np.float32
        )

        # obs: [local_goal_x, local_goal_y, dist, vx, vy, wz, cos_yaw, sin_yaw]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )

        # ── Параметры эпизода ─────────────────────────────────────────
        self.max_steps       = 1000
        self.goal_threshold  = 0.5   # м — радиус "достижения" цели
        self._step_count     = 0
        self._prev_dist      = 0.0

        # Для рендера
        self._viewer    = None
        self._renderer  = None

    # ──────────────────────────────────────────────────────────────────
    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    # ──────────────────────────────────────────────────────────────────

    def _get_robot_xy(self):
        """XY позиция корпуса в мировых координатах."""
        return self.data.xpos[self._base_body_id][:2].copy()

    def _get_goal_xy(self):
        """XY позиция цели (body goal_marker или константа)."""
        if self._goal_body_id is not None:
            return self.data.xpos[self._goal_body_id][:2].copy()
        return self._goal_xy_static.copy()

    def _get_yaw(self):
        """Рысканье (yaw) робота из кватерниона free-joint."""
        i = self._free_qpos_start
        qw = self.data.qpos[i + 3]
        qx = self.data.qpos[i + 4]
        qy = self.data.qpos[i + 5]
        qz = self.data.qpos[i + 6]
        return float(np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz)))

    def _get_obs(self):
        robot_xy = self._get_robot_xy()
        goal_xy  = self._get_goal_xy()
        to_goal  = goal_xy - robot_xy
        dist     = float(np.linalg.norm(to_goal))

        yaw      = self._get_yaw()
        cy, sy   = np.cos(-yaw), np.sin(-yaw)
        # вектор до цели в локальных координатах робота
        local_x  = cy * to_goal[0] - sy * to_goal[1]
        local_y  = sy * to_goal[0] + cy * to_goal[1]

        i = self._free_qvel_start
        vx  = float(self.data.qvel[i + 0])
        vy  = float(self.data.qvel[i + 1])
        wz  = float(self.data.qvel[i + 5])

        return np.array([
            local_x / 10.0,    # нормировано по арене 10м
            local_y / 10.0,
            dist    / 10.0,
            vx,
            vy,
            wz,
            np.cos(yaw),
            np.sin(yaw),
        ], dtype=np.float32)

    def _is_collision(self):
        """True если робот касается стены или препятствия (не пола)."""
        floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        for i in range(self.data.ncon):
            c  = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            # Игнорируем контакт с полом — это нормально
            if g1 == floor_id or g2 == floor_id:
                continue
            # Контакт с чем-то кроме пола → столкновение
            return True
        return False

    # ──────────────────────────────────────────────────────────────────
    # GYMNASIUM API
    # ──────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        # Ставим робота на старт
        i = self._free_qpos_start
        self.data.qpos[i + 0] = self.START_XY[0]   # x
        self.data.qpos[i + 1] = self.START_XY[1]   # y
        self.data.qpos[i + 2] = self.BASE_Z         # z
        self.data.qpos[i + 3] = 1.0                 # qw
        self.data.qpos[i + 4] = 0.0                 # qx
        self.data.qpos[i + 5] = 0.0                 # qy
        self.data.qpos[i + 6] = 0.0                 # qz
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0

        # Несколько шагов чтобы физика устоялась
        for _ in range(20):
            mujoco.mj_step(self.model, self.data)
        self.data.qvel[:] = 0.0

        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        self._prev_dist  = float(np.linalg.norm(self._get_goal_xy() - self._get_robot_xy()))

        return self._get_obs(), {}

    def step(self, action):
        self._step_count += 1

        # Подаём действие напрямую как target velocity (velocity actuator)
        # Порядок ctrl: [front_right, front_left, back_right, back_left]
        # action приходит в том же порядке [FR, FL, BR, BL]
        self.data.ctrl[:] = np.clip(action, -15.0, 15.0)

        # Несколько шагов физики на один шаг агента (frame skip)
        n_substeps = 5
        for _ in range(n_substeps):
            mujoco.mj_step(self.model, self.data)

        robot_xy = self._get_robot_xy()
        goal_xy  = self._get_goal_xy()
        dist     = float(np.linalg.norm(goal_xy - robot_xy))

        # ── Награда ───────────────────────────────────────────────────
        # 1. Потенциальная: за приближение к цели
        reward = (self._prev_dist - dist) * 10.0

        # 2. Штраф за каждый шаг (стимул двигаться быстрее)
        reward -= 0.05

        # 3. Штраф за столкновение
        if self._is_collision():
            reward -= 2.0

        self._prev_dist = dist

        # ── Завершение ────────────────────────────────────────────────
        terminated = dist < self.goal_threshold
        truncated  = self._step_count >= self.max_steps

        if terminated:
            reward += 200.0

        obs  = self._get_obs()
        info = {"dist_to_goal": dist, "step": self._step_count}

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()

        elif self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._renderer.update_scene(self.data)
            return self._renderer.render()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer = None


# ──────────────────────────────────────────────────────────────────────
# SMOKE-TEST
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== SummitEnv smoke test ===\n")

    env = SummitEnv("summit_xls.xml")
    obs, _ = env.reset()

    robot_xy = env._get_robot_xy()
    goal_xy  = env._get_goal_xy()
    print(f"Robot start : ({robot_xy[0]:.2f}, {robot_xy[1]:.2f})")
    print(f"Goal        : ({goal_xy[0]:.2f},  {goal_xy[1]:.2f})")
    print(f"Init dist   : {np.linalg.norm(goal_xy - robot_xy):.2f} м")
    print(f"Contacts    : {env.data.ncon}")
    print(f"obs shape   : {obs.shape}")
    print()

    print("Едем вперёд (все колёса +10 рад/с)...")
    total_reward = 0.0
    for i in range(100):
        action = np.array([10.0, 10.0, 10.0, 10.0], dtype=np.float32)
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        if i % 20 == 0:
            print(f"  step {i:3d}: pos=({env._get_robot_xy()[0]:.2f}, {env._get_robot_xy()[1]:.2f})  "
                  f"dist={info['dist_to_goal']:.2f}м  reward={reward:.3f}")
        if term or trunc:
            break

    moved = np.linalg.norm(env._get_robot_xy() - env.START_XY)
    print(f"\nПройдено: {moved:.2f} м  |  Суммарная награда: {total_reward:.1f}")

    if moved > 0.3:
        print("✓ Робот движется — среда работает корректно!")
    else:
        print("✗ Робот не двигается — проверьте XML-файлы и actuator.")

    env.close()
