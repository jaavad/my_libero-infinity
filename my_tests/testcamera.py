from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from libero_infinity.gym_env import LIBEROScenicEnv


REPO_ROOT = Path(__file__).resolve().parents[1]
BDDL_PATH = REPO_ROOT / "src/libero_infinity/data/libero_runtime/bddl_files/libero_goal/put_the_bowl_on_the_plate.bddl"
OUTDIR = REPO_ROOT / "my_tests" / "debug_manual_env_camera"
OUTDIR.mkdir(exist_ok=True, parents=True)

RESOLUTION = 256
MAX_STEPS = 300

DELTA_STEP = 0.05
ROT_STEP = 0.05
GRIPPER_STEP = 0.2


def clamp_action(a: np.ndarray) -> np.ndarray:
    return np.clip(a, -1.0, 1.0).astype(np.float32)


def save_frame(frame: np.ndarray, name: str) -> None:
    Image.fromarray(frame).save(OUTDIR / f"{name}.png")


def try_list_obs_cameras(obs):
    return [k for k in obs.keys() if "image" in k.lower()]


def pick_best_camera_key(obs):
    preferred = [
        "frontview_image",
        "sideview_image",
        "birdview_image",
        "agentview_image",
        "robot0_eye_in_hand_image",
    ]
    for k in preferred:
        if k in obs:
            return k
    cams = try_list_obs_cameras(obs)
    return cams[0] if cams else None


def try_override_sim_camera(env):
    sim = env._sim.libero_env.sim
    model = sim.model

    cam_names = []
    for i in range(model.ncam):
        try:
            cam_names.append(model.camera_id2name(i))
        except Exception:
            cam_names.append(None)

    print("Available cameras:", cam_names)

    target_cam_name = "agentview"
    if target_cam_name not in cam_names:
        return False, f"'agentview' not found in {cam_names}"

    cam_id = model.camera_name2id(target_cam_name)

    old_pos = model.cam_pos[cam_id].copy()
    old_fovy = float(model.cam_fovy[cam_id])

    model.cam_pos[cam_id][2] += 0.8
    model.cam_pos[cam_id][1] -= 0.5
    model.cam_fovy[cam_id] = min(old_fovy + 20.0, 100.0)

    return True, (
        f"Overrode camera '{target_cam_name}' "
        f"pos {old_pos} -> {model.cam_pos[cam_id].copy()}, "
        f"fovy {old_fovy:.1f} -> {float(model.cam_fovy[cam_id]):.1f}"
    )

env = LIBEROScenicEnv(
    bddl_path=str(BDDL_PATH),
    perturbation="position",
    resolution=RESOLUTION,
    max_steps=MAX_STEPS,
)


print("ENV TYPE:", type(env))
print("ENV DIR:", dir(env))

print("=== Environment created ===")
print("action_space:", env.action_space)
print("Assumed action meaning: [dx, dy, dz, dax, day, daz, gripper]")
print("Treating them as delta actions.")

#ok, msg = try_override_sim_camera(env)
#print("camera override:", ok, "|", msg)

obs = env.reset()

ok, msg = try_override_sim_camera(env)
print("camera override:", ok, "|", msg)

# get a fresh frame after camera change
obs = env._sim.libero_env._get_observations()

print("\n=== After reset ===")
print("observation keys:", list(obs.keys()))

camera_key = pick_best_camera_key(obs)
if camera_key is None:
    raise RuntimeError("No image observation key found.")

print("Using display camera:", camera_key)

current_action = np.zeros(7, dtype=np.float32)
step_count = 0
running = True

fig, ax = plt.subplots(figsize=(7, 7))
frame0 = obs[camera_key][::-1]
img_artist = ax.imshow(frame0)
ax.axis("off")


def refresh_display(obs, reward=None, done=None, info=None):
    frame = obs[camera_key][::-1]
    img_artist.set_data(frame)

    title = f"{camera_key} | step={step_count}"
    if reward is not None:
        title += f" | reward={reward:.3f}"
    if info is not None and isinstance(info, dict):
        title += f" | success={info.get('success', False)}"
    ax.set_title(title)
    fig.canvas.draw_idle()


def print_help():
    print("\nControls:")
    print("  w/s -> dx +/-")
    print("  a/d -> dy +/-")
    print("  r/f -> dz +/-")
    print("  i/k -> dax +/-")
    print("  j/l -> day +/-")
    print("  u/o -> daz +/-")
    print("  n/m -> gripper +/-")
    print("  space -> apply one step")
    print("  z -> zero current action")
    print("  p -> save current frame")
    print("  x -> reset env")
    print("  h -> print help")
    print("  q -> quit")


def on_key(event):
    global obs, current_action, step_count, running

    key = event.key
    if key is None:
        return

    if key == "q":
        running = False
        plt.close(fig)
        return

    elif key == "h":
        print_help()
        return

    elif key == "z":
        current_action[:] = 0.0

    elif key == "w":
        current_action[0] += DELTA_STEP
    elif key == "s":
        current_action[0] -= DELTA_STEP

    elif key == "a":
        current_action[1] += DELTA_STEP
    elif key == "d":
        current_action[1] -= DELTA_STEP

    elif key == "r":
        current_action[2] += DELTA_STEP
    elif key == "f":
        current_action[2] -= DELTA_STEP

    elif key == "i":
        current_action[3] += ROT_STEP
    elif key == "k":
        current_action[3] -= ROT_STEP

    elif key == "j":
        current_action[4] += ROT_STEP
    elif key == "l":
        current_action[4] -= ROT_STEP

    elif key == "u":
        current_action[5] += ROT_STEP
    elif key == "o":
        current_action[5] -= ROT_STEP

    elif key == "n":
        current_action[6] += GRIPPER_STEP
    elif key == "m":
        current_action[6] -= GRIPPER_STEP

    elif key == "p":
        save_frame(obs[camera_key][::-1], f"{camera_key}_step_{step_count}")
        print(f"Saved frame at step {step_count}")

    elif key == "x":
        obs = env.reset()
        step_count = 0
        print("Environment reset.")
        refresh_display(obs)
        return

    elif key == " ":
        current_action[:] = clamp_action(current_action)
        obs, reward, done, info = env.step(current_action)
        step_count += 1

        print(f"\n=== Step {step_count} ===")
        print("action:", current_action)
        print("reward:", reward)
        print("done:", done)
        print("info:", info)

        refresh_display(obs, reward=reward, done=done, info=info)

        if done:
            print("Episode ended. Resetting.")
            obs = env.reset()
            step_count = 0
            refresh_display(obs)
        return

    current_action[:] = clamp_action(current_action)
    print("Current action:", current_action)
    refresh_display(obs)


print_help()
cid = fig.canvas.mpl_connect("key_press_event", on_key)
refresh_display(obs)
plt.show()

fig.canvas.mpl_disconnect(cid)
env.close()
print("Closed.")






def move_agentview_camera_up(env, dz=0.4, dy_back=0.25, fovy_add=15.0):
    """
    Try to move the agentview camera upward and a bit backward so more of the
    robot upper body is visible.
    """
    sim = None

    # Try a few common wrapper structures
    candidates = [
        env,
        getattr(env, "env", None),
        getattr(env, "_env", None),
    ]

    for obj in candidates:
        if obj is None:
            continue
        if hasattr(obj, "sim"):
            sim = obj.sim
            break
        if hasattr(obj, "_sim"):
            sim = obj._sim
            break

    if sim is None:
        raise RuntimeError("Could not find underlying MuJoCo sim object.")

    model = sim.model

    # Print all camera names for debugging
    cam_names = []
    for i in range(model.ncam):
        try:
            cam_names.append(model.camera_id2name(i))
        except Exception:
            cam_names.append(None)
    print("Available cameras:", cam_names)

    # Try common external camera names
    target_name = None
    for name in ["agentview", "frontview", "sideview", "birdview"]:
        if name in cam_names:
            target_name = name
            break

    if target_name is None:
        raise RuntimeError(f"Could not find target camera in {cam_names}")

    cam_id = model.camera_name2id(target_name)

    old_pos = model.cam_pos[cam_id].copy()
    old_fovy = float(model.cam_fovy[cam_id])

    # Move camera upward and slightly backward
    model.cam_pos[cam_id][2] += dz
    model.cam_pos[cam_id][1] -= dy_back

    # Widen field of view a bit
    model.cam_fovy[cam_id] = min(old_fovy + fovy_add, 100.0)

    print(f"Modified camera: {target_name}")
    print("old_pos :", old_pos)
    print("new_pos :", model.cam_pos[cam_id].copy())
    print("old_fovy:", old_fovy)
    print("new_fovy:", float(model.cam_fovy[cam_id]))