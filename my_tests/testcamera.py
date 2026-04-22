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
    """
    Best-effort camera override.
    This is intentionally defensive because different LIBERO / robosuite builds
    expose the MuJoCo simulator a little differently.
    """
    possible_sim_attrs = ["env", "_env", "sim", "_sim"]
    sim = None
    base = env

    # Try nested env.sim or direct sim
    for attr in possible_sim_attrs:
        if hasattr(base, attr):
            obj = getattr(base, attr)
            if hasattr(obj, "model"):   # probably sim
                sim = obj
                break
            if hasattr(obj, "sim"):
                sim = obj.sim
                break

    if sim is None:
        return False, "Could not find MuJoCo sim object from env."

    model = getattr(sim, "model", None)
    if model is None:
        return False, "MuJoCo sim has no model."

    # Try to inspect camera names
    cam_names = []
    try:
        ncam = model.ncam
        for i in range(ncam):
            name = model.camera_id2name(i)
            cam_names.append(name)
    except Exception:
        pass

    target_names = ["agentview", "frontview", "sideview", "birdview"]
    target_cam_id = None
    target_cam_name = None

    for cname in target_names:
        if cname in cam_names:
            target_cam_name = cname
            break

    if target_cam_name is None and len(cam_names) > 0:
        target_cam_name = cam_names[0]

    if target_cam_name is None:
        return False, "No named MuJoCo cameras found."

    try:
        target_cam_id = model.camera_name2id(target_cam_name)
    except Exception as e:
        return False, f"Found camera name but could not get ID: {e}"

    # Attempt to widen / pull back camera
    # MuJoCo camera arrays typically:
    # cam_pos[ncam, 3], cam_quat[ncam, 4], cam_fovy[ncam]
    try:
        old_pos = model.cam_pos[target_cam_id].copy()
        old_fovy = float(model.cam_fovy[target_cam_id])

        # pull camera backward in its local rough world z/y sense
        # exact direction depends on existing pose, but this often helps
        new_pos = old_pos.copy()
        new_pos[1] -= 0.6
        new_pos[2] += 0.3
        model.cam_pos[target_cam_id] = new_pos

        # widen FOV
        model.cam_fovy[target_cam_id] = min(90.0, old_fovy + 20.0)

        return True, (
            f"Overrode camera '{target_cam_name}' "
            f"pos {old_pos} -> {new_pos}, "
            f"fovy {old_fovy:.1f} -> {float(model.cam_fovy[target_cam_id]):.1f}"
        )
    except Exception as e:
        return False, f"Camera override failed: {e}"


env = LIBEROScenicEnv(
    bddl_path=str(BDDL_PATH),
    perturbation="position",
    resolution=RESOLUTION,
    max_steps=MAX_STEPS,
)

print("=== Environment created ===")
print("action_space:", env.action_space)
print("Assumed action meaning: [dx, dy, dz, dax, day, daz, gripper]")
print("Treating them as delta actions.")

ok, msg = try_override_sim_camera(env)
print("camera override:", ok, "|", msg)

obs = env.reset()
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