# train.py — staged curriculum learning + дообучение с чекпоинта + параллельные среды
#
# Этап 1: открытое поле           (0–200k)   — учимся ехать к цели
# Этап 2: коридор                 (200–400k) — учимся не касаться стен
# Этап 3: коридор с препятствиями (400–700k) — учимся объезжать
# Этап 4: открытое поле + много   (700k–1.5M) — генерализация


import os
import sys
import glob
import re
import mujoco
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, BaseCallback)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from env_summit import SummitEnv

LOG_DIR    = "logs"
MODEL_DIR  = "models"
os.makedirs(LOG_DIR,   exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

TOTAL_TIMESTEPS = 1_500_000

# Сколько параллельных сред поднимать. На Windows процессы дороже —
# начните с 4 и поднимайте, если CPU позволяет (os.cpu_count()).
N_ENVS = 4


# ──────────────────────────────────────────────────────────────────────
# Env с поддержкой переключения этапа
# ──────────────────────────────────────────────────────────────────────

class StagedCurriculumEnv(SummitEnv):
    """
    Env который умеет переключать сцену на лету.
    Этапы: 1=открытое, 2=коридор, 3=коридор+препятствия, 4=поле+много
    """
    STAGE_SCENES = {
        1: "scene_stage1.xml",
        2: "scene_stage2.xml",
        3: "scene_stage3.xml",
        4: "scene_stage4.xml",
    }

    def __init__(self, xml_path="summit_xls.xml", render_mode=None,
                 difficulty=1.0, stage=1):
        self._stage = stage
        super().__init__(
            xml_path=self._make_stage_xml(stage),
            render_mode=render_mode,
            difficulty=difficulty,
        )

    def _make_stage_xml(self, stage):
        scene_file = self.STAGE_SCENES.get(stage, "scene_stage1.xml")
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
        # PID в имени файла — чтобы параллельные процессы не конфликтовали
        tmp = f"_tmp_stage{stage}_{os.getpid()}.xml"
        with open(tmp, "w") as f:
            f.write(xml)
        return tmp

    def set_stage_and_difficulty(self, stage, difficulty):
        """
        Единая точка входа для curriculum-колбэка.
        Вызывается через env_method() — работает и в DummyVecEnv,
        и в SubprocVecEnv (выполняется внутри каждого подпроцесса).
        """
        self.difficulty = difficulty
        if stage == self._stage:
            return
        self._stage = stage
        new_xml = self._make_stage_xml(stage)
        self.xml_path = new_xml
        self.model = mujoco.MjModel.from_xml_path(new_xml)
        self.data  = mujoco.MjData(self.model)

        self._base_body_id = self.model.body("base").id
        try:
            self._goal_body_id = self.model.body("goal_marker").id
            self._goal_xy_static = None
        except KeyError:
            self._goal_body_id = None
            self._goal_xy_static = self.GOAL_XY.copy()

        self._floor_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        self._base_footprint_id = self.model.body("base_footprint").id

        self._robot_geom_ids = set()
        for i in range(self.model.ngeom):
            bid = self.model.geom_bodyid[i]
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
        for name in ["block_center", "block_left", "block_right",
                     "sphere_obs", "cyl_obs"]:
            gid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                self._obs_geom_names.append(name)


def make_env(stage=1):
    def _init():
        return StagedCurriculumEnv(stage=stage)
    return _init


# ──────────────────────────────────────────────────────────────────────
# Curriculum callback — работает через env_method (совместимо с Subproc)
# ──────────────────────────────────────────────────────────────────────

class StagedCurriculumCallback(BaseCallback):
    SCHEDULE = [
        # (min_steps, stage, difficulty, описание)
        (0,       1, 1.0, "Этап 1: открытое поле"),
        (200_000, 2, 1.0, "Этап 2: коридор"),
        (400_000, 3, 0.5, "Этап 3: коридор + препятствия (50%)"),
        (500_000, 3, 1.0, "Этап 3: коридор + препятствия (100%)"),
        (700_000, 4, 0.5, "Этап 4: поле + много препятствий (50%)"),
        (900_000, 4, 1.0, "Этап 4: полная сложность"),
    ]

    def __init__(self, train_vec_env, eval_vec_env, verbose=1):
        super().__init__(verbose)
        self.train_vec_env = train_vec_env
        self.eval_vec_env  = eval_vec_env
        self._current_idx  = -1

    def _apply(self, stage, difficulty):
        # env_method работает одинаково на DummyVecEnv и SubprocVecEnv —
        # вызывает метод во всех под-средах (в т.ч. в дочерних процессах).
        self.train_vec_env.env_method("set_stage_and_difficulty", stage, difficulty)
        self.eval_vec_env.env_method("set_stage_and_difficulty", stage, difficulty)

    def _on_step(self):
        t = self.num_timesteps
        new_idx = self._current_idx
        for i, (steps, stage, diff, desc) in enumerate(self.SCHEDULE):
            if t >= steps:
                new_idx = i

        if new_idx != self._current_idx:
            self._current_idx = new_idx
            _, stage, diff, desc = self.SCHEDULE[new_idx]
            self._apply(stage, diff)
            if self.verbose:
                print(f"\n[Curriculum] Step {t:,}: {desc}\n")
        return True

    def stage_for_step(self, step):
        """Какой этап/сложность должны быть активны на данном шаге (для дообучения)."""
        result = self.SCHEDULE[0]
        for entry in self.SCHEDULE:
            if step >= entry[0]:
                result = entry
        return result


# ──────────────────────────────────────────────────────────────────────
# Поиск последнего чекпоинта для дообучения
# ──────────────────────────────────────────────────────────────────────

def find_latest_checkpoint():
    """
    Ищет файлы вида models/summit_staged_<N>_steps.zip и models/best_model.zip,
    возвращает (путь, число_шагов) для самого свежего по числу шагов.
    Также учитывает models/summit_staged_final.zip и models/summit_mcal_final.zip
    как кандидатов (число шагов берём из CLI или просим явно — см. RESUME_FROM_STEP).
    """
    candidates = []

    for f in glob.glob(os.path.join(MODEL_DIR, "summit_staged_*_steps.zip")):
        m = re.search(r"_(\d+)_steps\.zip$", f)
        if m:
            candidates.append((int(m.group(1)), f))

.
    if not candidates:
        best = os.path.join(MODEL_DIR, "best_model.zip")
        if os.path.exists(best):
            return best, None 

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        steps, path = candidates[0]
        return path, steps

    return None, None


if __name__ == "__main__":
    # ── Настройка дообучения ────────────────────────────────────────
    
    RESUME_PATH = None
    RESUME_FROM_STEP = None

    auto_path, auto_steps = find_latest_checkpoint()
    if auto_path is not None:
        RESUME_PATH = auto_path
        RESUME_FROM_STEP = auto_steps

    # ЯВНО укажите здесь, если знаете точно (перекрывает автопоиск):
    # RESUME_PATH = "models/summit_staged_600000_steps.zip"
    # RESUME_FROM_STEP = 600_000

    if RESUME_PATH and RESUME_FROM_STEP is None:
        print(f"Найден чекпоинт {RESUME_PATH}, но число шагов неизвестно.")
        print("Укажите RESUME_FROM_STEP вручную в train.py (например 600_000) и запустите снова.")
        sys.exit(1)

    # ── Параллельные среды ───────────────────────────────────────────
    use_subproc = sys.platform != "win32"
    VecCls = SubprocVecEnv if (use_subproc and N_ENVS > 1) else DummyVecEnv

    if N_ENVS > 1 and use_subproc:
        print(f"Параллельные среды: {N_ENVS} (SubprocVecEnv)")
        train_env = SubprocVecEnv([make_env(1) for _ in range(N_ENVS)])
    else:
        if N_ENVS > 1:
            print("ВНИМАНИЕ: Windows — SubprocVecEnv может работать нестабильно "
                  "с MuJoCo viewer/рендером. Используем DummyVecEnv "
                  f"с {N_ENVS} средами в одном процессе (без ускорения по CPU, "
                  "но без риска зависаний). Для реального параллелизма "
                  "запускайте через `if __name__ == '__main__':` — он уже есть.")
        train_env = DummyVecEnv([make_env(1) for _ in range(N_ENVS)])

    eval_env = DummyVecEnv([make_env(1)])

    curriculum_cb = StagedCurriculumCallback(
        train_vec_env=train_env, eval_vec_env=eval_env, verbose=1)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=max(20_000 // N_ENVS, 2000),
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(100_000 // N_ENVS, 10_000),
        save_path=MODEL_DIR,
        name_prefix="summit_staged",
    )

    # ── Создание / загрузка модели ───────────────────────────────────
    if RESUME_PATH:
        print(f"Дообучение с чекпоинта: {RESUME_PATH} (шаг {RESUME_FROM_STEP:,})")
        model = SAC.load(RESUME_PATH, env=train_env, tensorboard_log=LOG_DIR)
        model.num_timesteps = RESUME_FROM_STEP  # синхронизируем счётчик шагов


        idx = -1
        for i, entry in enumerate(curriculum_cb.SCHEDULE):
            if RESUME_FROM_STEP >= entry[0]:
                idx = i
        curriculum_cb._current_idx = idx
        _, stage, diff, desc = curriculum_cb.SCHEDULE[idx]
        curriculum_cb._apply(stage, diff)
        print(f"Восстановлен этап: {desc}")
    else:
        print("Чекпоинт не найден — обучение с нуля")
        model = SAC(
            "MlpPolicy",
            train_env,
            learning_rate=3e-4,
            buffer_size=500_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
            learning_starts=5_000,
            verbose=1,
            tensorboard_log=LOG_DIR,
            policy_kwargs=dict(net_arch=[256, 256, 128]),
        )

    print("\n=== Staged Curriculum Learning ===")
    for steps, stage, diff, desc in StagedCurriculumCallback.SCHEDULE:
        marker = " ← старт" if RESUME_FROM_STEP and steps <= RESUME_FROM_STEP else ""
        print(f"  {steps:>8,} шагов → {desc}{marker}")
    print(f"\nЦель: {TOTAL_TIMESTEPS:,} шагов всего\n")

    remaining = TOTAL_TIMESTEPS - (RESUME_FROM_STEP or 0)
    if remaining <= 0:
        print("Уже достигнут целевой timestep. Увеличьте TOTAL_TIMESTEPS если нужно продолжить.")
        sys.exit(0)

    try:
        model.learn(
            total_timesteps=remaining,
            callback=[curriculum_cb, eval_cb, checkpoint_cb],
            progress_bar=True,
            reset_num_timesteps=False,  # продолжаем счётчик, а не с нуля
        )
    except KeyboardInterrupt:
        print("\nОстановлено.")

    model.save(os.path.join(MODEL_DIR, "summit_staged_final"))
    print("Сохранено: models/summit_staged_final.zip")
