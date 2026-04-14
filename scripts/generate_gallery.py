#!/usr/bin/env python3
"""Generate worst-case perturbation gallery for README.

For each perturbation axis, we read the Scenic program, identify the extreme
end of the sampling distribution (worst case), then render that scene directly.
This makes the differences visually obvious to humans reading the README.

Run from project root:
    MUJOCO_GL=egl PYTHONPATH=src .venv/bin/python scripts/generate_gallery.py
"""
import os
import pathlib
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
from libero_infinity.asset_registry import get_dimensions
from libero_infinity.runtime import get_bddl_dir
from libero_infinity.simulator import (
    DEFAULT_ORIENTATIONS,
    TABLE_Z,
    _anchor_visible,
    _camera_transforms,
    _real_depth_map,
    _visibility_anchor_points,
)

ROOT = pathlib.Path(__file__).parent.parent
BDDL = str(get_bddl_dir() / "libero_goal" / "put_the_bowl_on_the_plate.bddl")
OUT = ROOT / "assets/perturbation_gallery.png"
RES = 320  # render resolution per panel
_ZERO_ACTION: np.ndarray | None = None
ROBOT_CANONICAL_QPOS = np.array(
    [0.0, -0.161037389, 0.0, -2.44459747, 0.0, 2.2267522, np.pi / 4.0],
    dtype=float,
)
ROBOT_DIRECTION = np.array([0.55, -0.25, 0.20, 0.45, -0.30, -0.35, 0.40], dtype=float)
ROBOT_DIRECTION /= np.linalg.norm(ROBOT_DIRECTION)
PANEL_ASSETS = {
    "robot": ROOT / "assets/libero_robot_0.png",
    "distractor": ROOT / "assets/libero_distractor_0.png",
    "background": ROOT / "assets/libero_background_0.png",
}


# ── helpers ──────────────────────────────────────────────────────────────────

def make_env(bddl_path: str):
    from libero.libero.envs.env_wrapper import OffScreenRenderEnv
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="agentview",
        camera_names=["agentview"],
        camera_heights=RES,
        camera_widths=RES,
        control_freq=20,
        horizon=200,
        ignore_done=False,
        hard_reset=True,
    )
    env.reset()
    return env


def get_frame(env) -> np.ndarray:
    """Render the current sim state via env.step(zero).

    _get_observations() returns the cached result from the last env.step() or
    env.reset() call — it does NOT re-render.  Driving one zero-action step
    flushes the physics state through the full render pipeline and returns a
    fresh image.
    """
    global _ZERO_ACTION
    if _ZERO_ACTION is None or _ZERO_ACTION.shape[0] != env.env.action_spec[0].shape[0]:
        _ZERO_ACTION = np.zeros(env.env.action_spec[0].shape[0])
    obs, _, _, _ = env.step(_ZERO_ACTION)
    return obs["agentview_image"][::-1].copy()


def settle(env, steps: int = 40):
    """Zero velocities and run physics steps so injected objects come to rest."""
    import mujoco
    m = env.env.sim.model._model
    d = env.env.sim.data._data
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    for _ in range(steps):
        mujoco.mj_step(m, d)
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)


def set_joint_xy(env, joint_name: str, x: float, y: float):
    """Teleport object to (x, y) while keeping original z and orientation."""
    sim_data = env.env.sim.data
    try:
        qpos = sim_data.get_joint_qpos(joint_name).copy()
        qpos[0] = x
        qpos[1] = y
        sim_data.set_joint_qpos(joint_name, qpos)
    except Exception as e:
        print(f"    WARNING: could not set {joint_name}: {e}")


def set_tabletop_pose(env, libero_name: str, asset_class: str, x: float, y: float):
    """Place a free-joint object upright on the tabletop using a full qpos pose."""
    sim = env.env.sim
    _w, _l, h = get_dimensions(asset_class)
    z = TABLE_Z + max(float(h) / 2.0, 0.01) + 1e-3
    quat_xyzw = DEFAULT_ORIENTATIONS.get(asset_class, DEFAULT_ORIENTATIONS["_default"]).copy()
    quat_wxyz = np.array(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
        dtype=float,
    )
    qpos = np.array([x, y, z, *quat_wxyz], dtype=float)
    sim.data.set_joint_qpos(f"{libero_name}_joint0", qpos)


def mj_forward(env):
    import mujoco
    mujoco.mj_forward(env.env.sim.model._model, env.env.sim.data._data)


def save_panel_asset(kind: str, image: np.ndarray):
    path = PANEL_ASSETS.get(kind)
    if path is None:
        return
    path.parent.mkdir(exist_ok=True)
    plt.imsave(path, image)


def _project_point(world_to_pixel: np.ndarray, point: np.ndarray) -> tuple[float, float] | None:
    hom = np.concatenate([point.astype(float), np.array([1.0])], axis=0)
    pixel_hom = world_to_pixel @ hom
    if pixel_hom[2] <= 1e-6:
        return None
    return (
        float(pixel_hom[0] / pixel_hom[2]),
        float(pixel_hom[1] / pixel_hom[2]),
    )


def distractor_visibility_score(simulation, scene) -> float:
    """Prefer Scenic samples where active distractors are central and visible."""
    obs = simulation.last_obs or {}
    depth = obs.get("agentview_depth")
    if depth is None:
        return float("-inf")

    sim = simulation.libero_env.env.sim
    height = int(depth.shape[0])
    width = int(depth.shape[1])
    world_to_pixel, world_to_camera = _camera_transforms(
        sim=sim,
        camera_name="agentview",
        camera_height=height,
        camera_width=width,
    )
    depth_map = _real_depth_map(sim, depth[..., 0])
    image_center = np.array([width / 2.0, height / 2.0], dtype=float)

    n = int(scene.params.get("n_distractors", 0))
    total_visible_anchors = 0
    fully_visible_distractors = 0
    center_bonus = 0.0

    for i in range(n):
        name = f"distractor_{i}"
        body_id = None
        for candidate in (name, name + "_main"):
            try:
                body_id = sim.model.body_name2id(candidate)
                break
            except Exception:
                continue
        if body_id is None:
            continue

        center = np.array(sim.data.body_xpos[body_id][:3], dtype=float)
        asset_class = scene.params.get(f"distractor_{i}_class", "_default")
        dims = get_dimensions(str(asset_class))
        visible = 0
        for point in _visibility_anchor_points(center, dims):
            visible += int(
                _anchor_visible(
                    point=point,
                    world_to_pixel=world_to_pixel,
                    world_to_camera=world_to_camera,
                    depth_map=depth_map,
                    image_height=height,
                    image_width=width,
                )
            )
        total_visible_anchors += visible
        if visible > 0:
            fully_visible_distractors += 1
            pixel = _project_point(world_to_pixel, center)
            if pixel is not None:
                dist = np.linalg.norm(np.array(pixel, dtype=float) - image_center)
                center_bonus += max(0.0, 200.0 - dist)

    return fully_visible_distractors * 1000.0 + total_visible_anchors * 50.0 + center_bonus


def step_model_change(env):
    """After directly modifying model params (camera, lights), call mj_forward
    then one env.step so the render pipeline sees the updated model state."""
    mj_forward(env)
    # We don't need the return value; get_frame() will call step() again
    # and capture the rendered obs from that step.  But the model params set
    # above (cam_pos, cam_quat, light params) persist across steps because they
    # live on the *model* not the *data*, so they'll be visible in get_frame().
    pass


def set_robot_qpos(env, qpos: np.ndarray):
    """Apply a 7-DOF Panda joint configuration directly to the MuJoCo arm."""
    robot = env.env.robots[0]
    sim = env.env.sim
    joint_indexes = np.asarray(robot._ref_joint_pos_indexes, dtype=int)
    joint_names = tuple(robot.robot_joints)

    lower = []
    upper = []
    for joint_name in joint_names:
        joint_id = int(sim.model.joint_name2id(joint_name))
        lo, hi = sim.model.jnt_range[joint_id]
        lower.append(float(lo))
        upper.append(float(hi))

    qpos = np.asarray(qpos, dtype=float)
    qpos = np.clip(qpos, np.asarray(lower, dtype=float), np.asarray(upper, dtype=float))
    sim.data.qpos[joint_indexes] = qpos
    sim.data.qvel[joint_indexes] = 0.0
    mj_forward(env)


def apply_background_tint(env):
    """Tint wall and floor geoms strongly so background variation is visible in screenshots."""
    sim = env.env.sim
    wall_rgba = np.array([0.12, 0.20, 0.48, 1.0], dtype=float)
    floor_rgba = np.array([0.84, 0.77, 0.62, 1.0], dtype=float)
    for geom_id in range(sim.model.ngeom):
        name = ""
        try:
            name = sim.model.geom_id2name(geom_id) or ""
        except Exception:
            pass
        if name == "floor":
            sim.model.geom_rgba[geom_id] = floor_rgba
        elif "wall_" in name:
            sim.model.geom_rgba[geom_id] = wall_rgba
    mj_forward(env)


def make_distractor_env(distractors: list[tuple[str, str]]):
    from libero_infinity.bddl_preprocessor import add_distractor_objects

    bddl_text = pathlib.Path(BDDL).read_text()
    patched = add_distractor_objects(bddl_text, distractors)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".bddl", delete=False, prefix="libero_gal_") as f:
        f.write(patched)
        tmp = f.name
    env = make_env(tmp)
    return env, tmp


def render_scenic_panel(
    scenic_file: str,
    params: dict[str, object],
    retries: int = 8,
    scorer=None,
) -> np.ndarray:
    """Render one image through the real Scenic -> simulator path."""
    import scenic as sc

    from libero_infinity.simulator import LIBEROSimulator
    from libero_infinity.validation_errors import CollisionError, VisibilityError

    scenario = sc.scenarioFromFile(str(ROOT / "scenic" / scenic_file), params=params)
    last_exc: BaseException | None = None
    best_score = float("-inf")
    best_image: np.ndarray | None = None
    for _attempt in range(retries + 1):
        try:
            scene, _ = scenario.generate(maxIterations=2000, verbosity=0)
        except Exception as exc:
            last_exc = exc
            continue

        simulator = LIBEROSimulator(
            bddl_path=str(params.get("bddl_path", BDDL)),
            env_kwargs={
                "camera_names": ["agentview"],
                "camera_heights": RES,
                "camera_widths": RES,
                "render_camera": "agentview",
                "camera_depths": True,
            },
        )
        sim = simulator.createSimulation(scene, maxSteps=5, timestep=0.05, verbosity=0)
        try:
            sim.setup()
            obs = sim.last_obs
            if "agentview_image" not in obs:
                raise RuntimeError("agentview_image missing from Scenic-rendered observation")
            image = obs["agentview_image"][::-1].copy()
            if scorer is None:
                return image
            score = float(scorer(sim, scene))
            if score > best_score:
                best_score = score
                best_image = image
        except (CollisionError, VisibilityError) as exc:
            last_exc = exc
        finally:
            try:
                sim.destroy()
            except Exception:
                pass

    if best_image is not None:
        return best_image
    raise RuntimeError(f"Failed to render Scenic panel after {retries + 1} attempts") from last_exc


# ── panel renderers ───────────────────────────────────────────────────────────

def panel_default() -> np.ndarray:
    """Canonical scene — standard env.reset(), no perturbation."""
    env = make_env(BDDL)
    img = get_frame(env)
    env.close()
    return img


def panel_position() -> np.ndarray:
    """Worst-case position: objects at far corners of workspace.

    From position_perturbation.scenic:
        bowl  in SAFE_REGION  (x∈[-0.35, 0.35], y∈[-0.25, 0.25])
        plate in PLATE_SAFE_REGION
        require[0.8] distance from bowl to bowl_train_pt > 0.15  (OOD bias)

    Training positions: bowl≈(0.12,-0.05), plate stays canonical.
    Worst-case: push only the bowl (the task object) to the opposite corner.
    """
    env = make_env(BDDL)
    # Move only the bowl — the target object — to the far opposite corner
    set_joint_xy(env, "akita_black_bowl_1_joint0", x=-0.30, y=0.22)
    settle(env)
    img = get_frame(env)
    env.close()
    return img


def panel_object() -> np.ndarray:
    """Worst-case object: most visually different variant.

    From object_perturbation.scenic:
        chosen_asset = Uniform(*ASSET_VARIANTS["akita_black_bowl"])
        # → ["akita_black_bowl", "white_bowl", "glazed_rim_porcelain_ramekin"]

    Worst-case: glazed_rim_porcelain_ramekin — completely different shape & color.
    """
    from libero_infinity.bddl_preprocessor import substitute_asset
    bddl_text = pathlib.Path(BDDL).read_text()
    patched = substitute_asset(bddl_text, "akita_black_bowl", "glazed_rim_porcelain_ramekin")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".bddl", delete=False, prefix="libero_gal_") as f:
        f.write(patched)
        tmp = f.name
    try:
        env = make_env(tmp)
        img = get_frame(env)
        env.close()
    finally:
        os.unlink(tmp)
    return img


def panel_combined() -> np.ndarray:
    """Combined preset: object, position, robot, camera, lighting, distractor, background."""
    from libero_infinity.bddl_preprocessor import add_distractor_objects, substitute_asset

    bddl_text = pathlib.Path(BDDL).read_text()
    patched = substitute_asset(bddl_text, "akita_black_bowl", "glazed_rim_porcelain_ramekin")
    patched = add_distractor_objects(
        patched,
        [
            ("distractor_0", "bowl_drainer"),
            ("distractor_1", "desk_caddy"),
            ("distractor_2", "ketchup"),
            ("distractor_3", "macaroni_and_cheese"),
            ("distractor_4", "alphabet_soup"),
        ],
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".bddl", delete=False, prefix="libero_gal_") as f:
        f.write(patched)
        tmp = f.name
    try:
        env = make_env(tmp)
        set_joint_xy(env, "akita_black_bowl_1_joint0", x=-0.30, y=0.22)
        set_joint_xy(env, "plate_1_joint0",            x=0.28,  y=-0.20)
        set_joint_xy(env, "distractor_0_joint0",       x=-0.28, y=0.18)
        set_joint_xy(env, "distractor_1_joint0",       x=-0.10, y=-0.18)
        set_joint_xy(env, "distractor_2_joint0",       x=0.03,  y=0.20)
        set_joint_xy(env, "distractor_3_joint0",       x=0.26,  y=0.12)
        set_joint_xy(env, "distractor_4_joint0",       x=0.22,  y=-0.17)
        set_robot_qpos(env, ROBOT_CANONICAL_QPOS + 0.5 * ROBOT_DIRECTION)
        apply_background_tint(env)
        sim = env.env.sim
        cam_id = sim.model.camera_name2id("agentview")
        sim.model.cam_pos[cam_id][0] += 0.08
        sim.model.cam_pos[cam_id][1] += 0.06
        sim.model.cam_pos[cam_id][2] -= 0.06
        for i in range(sim.model.nlight):
            sim.model.light_diffuse[i] *= 0.55
            sim.model.light_specular[i] *= 0.55
        sim.model.vis.headlight.ambient[:] = 0.08
        settle(env)
        img = get_frame(env)
        env.close()
    finally:
        os.unlink(tmp)
    return img


def panel_camera() -> np.ndarray:
    """Worst-case camera: all offsets and tilt at maximum magnitude.

    From camera_perturbation.scenic:
        param camera_x_range = 0.10     → Range(-0.10, 0.10)
        param camera_z_range = 0.08     → Range(-0.08, 0.08)
        param camera_tilt_range = 15.0  → Range(-15.0, 15.0)

    Worst-case: dx=+0.10, dy=+0.10, dz=-0.08 (lower), tilt=+15°.
    """
    from scipy.spatial.transform import Rotation as R
    env = make_env(BDDL)
    sim = env.env.sim
    cam_id = sim.model.camera_name2id("agentview")

    # Max position offsets
    sim.model.cam_pos[cam_id][0] += 0.10   # forward
    sim.model.cam_pos[cam_id][1] += 0.10   # left
    sim.model.cam_pos[cam_id][2] -= 0.08   # lower → see scene from below-ish

    # Max tilt: +15° around camera local x-axis
    # MuJoCo stores camera quaternion as (w, x, y, z)
    cq = sim.model.cam_quat[cam_id].copy()
    r_current = R.from_quat([cq[1], cq[2], cq[3], cq[0]])  # (x,y,z,w)
    r_tilt = R.from_euler("x", 15.0, degrees=True)
    r_new = r_current * r_tilt
    q = r_new.as_quat()  # (x,y,z,w)
    sim.model.cam_quat[cam_id] = [q[3], q[0], q[1], q[2]]  # back to (w,x,y,z)

    img = get_frame(env)
    env.close()
    return img


def panel_lighting() -> np.ndarray:
    """Worst-case lighting: minimum intensity and ambient (very dark scene).

    From lighting_perturbation.scenic:
        param intensity_min = 0.4   → Range(0.4, 2.0)
        param ambient_min   = 0.05  → Range(0.05, 0.6)

    Worst-case for a policy: darkest setting — objects barely visible.
    """
    env = make_env(BDDL)
    sim = env.env.sim
    # Multiply all light diffuse/specular by 0.4 (minimum intensity)
    for i in range(sim.model.nlight):
        sim.model.light_diffuse[i]  *= 0.4
        sim.model.light_specular[i] *= 0.4
    # Near-zero ambient: scene barely lit by indirect light
    sim.model.vis.headlight.ambient[:] = 0.05
    img = get_frame(env)
    env.close()
    return img


def panel_distractor() -> np.ndarray:
    """Deterministic clutter panel with clearly visible tabletop distractors."""
    distractors = [
        ("distractor_0", "ketchup", 0.18, -0.20),
        ("distractor_1", "alphabet_soup", 0.23, 0.15),
        ("distractor_2", "ketchup", -0.02, 0.19),
    ]
    env, tmp = make_distractor_env([(name, cls) for name, cls, _x, _y in distractors])
    try:
        for name, cls, x, y in distractors:
            set_tabletop_pose(env, name, cls, x, y)
        settle(env, steps=60)
        img = get_frame(env)
        save_panel_asset("distractor", img)
        return img
    finally:
        env.close()
        pathlib.Path(tmp).unlink(missing_ok=True)


def panel_background() -> np.ndarray:
    """Worst-case background: strong wall/floor tint to highlight appearance change."""
    env = make_env(BDDL)
    apply_background_tint(env)
    img = get_frame(env)
    save_panel_asset("background", img)
    env.close()
    return img


def panel_robot() -> np.ndarray:
    """Worst-case robot: max-radius Panda init_qpos perturbation."""
    env = make_env(BDDL)
    qpos = ROBOT_CANONICAL_QPOS + 0.5 * ROBOT_DIRECTION
    set_robot_qpos(env, qpos)
    settle(env, steps=10)
    img = get_frame(env)
    save_panel_asset("robot", img)
    env.close()
    return img


# ── panel metadata ────────────────────────────────────────────────────────────

PANELS = [
    # (render_fn, bold_title, scenic_code_line, title_color)
    (
        panel_default,
        "Default (canonical)",
        "env.reset()  # standard LIBERO initial state",
        "#444444",
    ),
    (
        panel_position,
        "Position Perturbation",
        "bowl = new LIBEROObject in SAFE_REGION",
        "#1976D2",
    ),
    (
        panel_object,
        "Object Perturbation",
        'Uniform(*ASSET_VARIANTS["akita_black_bowl"])',
        "#E65100",
    ),
    (
        panel_combined,
        "Combined Preset",
        "--perturbation combined",
        "#C62828",
    ),
    (
        panel_camera,
        "Camera Perturbation",
        "param camera_tilt = Range(-15.0, 15.0)",
        "#B71C1C",
    ),
    (
        panel_lighting,
        "Lighting Perturbation",
        "param light_intensity = Range(0.4, 2.0)",
        "#6A1B9A",
    ),
    (
        panel_distractor,
        "Distractor Perturbation",
        "curated gallery render: 3 visible distractors",
        "#2E7D32",
    ),
    (
        panel_background,
        "Background Perturbation",
        "param wall_texture = Uniform(*texture_candidates)",
        "#00838F",
    ),
    (
        panel_robot,
        "Robot Perturbation",
        "param robot_init_radius = Range(0.1, 0.5)",
        "#5D4037",
    ),
]


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("Generating worst-case perturbation gallery...")

    images = []
    for render_fn, title, code, color in PANELS:
        print(f"  [{title}] ...")
        try:
            img = render_fn()
            images.append(img)
            print(f"    OK — shape {img.shape}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"    FAILED — using placeholder")
            images.append(np.zeros((RES, RES, 3), dtype=np.uint8))

    # ── compose 3×3 gallery ──────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 13.5))
    fig.patch.set_facecolor("white")
    fig.suptitle("Libero-∞  Perturbation Axes", fontsize=15, fontweight="bold", y=0.995)

    for ax, img, (_, title, code, color) in zip(axes.flat, images, PANELS):
        ax.imshow(img)
        ax.axis("off")
        # Line 1: bold colored title
        ax.set_title(title, fontsize=10, fontweight="bold", color=color, pad=5)
        # Line 2: monospace Scenic code below the image
        ax.text(
            0.5, -0.02,
            code,
            transform=ax.transAxes,
            fontsize=7.5,
            ha="center",
            va="top",
            fontfamily="monospace",
            color="#555555",
        )

    plt.tight_layout(rect=[0, 0.01, 1, 0.97])
    plt.subplots_adjust(hspace=0.25)
    OUT.parent.mkdir(exist_ok=True)
    plt.savefig(str(OUT), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\nSaved → {OUT}")


if __name__ == "__main__":
    main()
