from pathlib import Path
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from libero_infinity.gym_env import LIBEROScenicEnv


REPO_ROOT = Path(__file__).resolve().parents[1]
BDDL_PATH = REPO_ROOT / "src/libero_infinity/data/libero_runtime/bddl_files/libero_goal/put_the_bowl_on_the_plate.bddl"

OUTDIR = REPO_ROOT / "my_tests" / "debug_keyboard_control"
OUTDIR.mkdir(exist_ok=True, parents=True)

RESOLUTION = 128
MAX_STEPS = 200

# Initial step size for each key press
DELTA_STEP = 0.05
ROT_STEP = 0.05
GRIPPER_STEP = 0.2

env = LIBEROScenicEnv(
    bddl_path=str(BDDL_PATH),
    perturbation="position",
    resolution=RESOLUTION,
    max_steps=MAX_STEPS,
)


print(env.__dict__.keys())

obs = env.reset()

print("\n=== After reset ===")
print("observation keys:", list(obs.keys()))
print("observation_space:", env.observation_space)

current_action = np.zeros(7, dtype=np.float32)
step_count = 0
running = True

# Save initial observation metadata
obs_meta = {}
for k, v in obs.items():
    if isinstance(v, np.ndarray):
        entry = {"shape": list(v.shape), "dtype": str(v.dtype)}
        if v.size > 0 and np.issubdtype(v.dtype, np.number):
            entry["min"] = float(v.min())
            entry["max"] = float(v.max())
        obs_meta[k] = entry
    else:
        obs_meta[k] = {"type": str(type(v))}

with open(OUTDIR / "obs_meta_reset.json", "w") as f:
    json.dump(obs_meta, f, indent=2)


def get_display_frame(observation: dict) -> np.ndarray:
    if "agentview_image" not in observation:
        raise KeyError("agentview_image not found in observation")
    # Flip vertically for normal viewing
    return observation["agentview_image"][::-1]


def save_current_frame(observation: dict, suffix: str):
    for cam_key in ["agentview_image", "robot0_eye_in_hand_image"]:
        if cam_key in observation:
            frame = observation[cam_key][::-1]
            Image.fromarray(frame).save(OUTDIR / f"{cam_key}_{suffix}.png")
    print(f"Saved frame(s) with suffix: {suffix}")


def print_help():
    print("\n=== Keyboard controls ===")
    print("Position:")
    print("  w/s : dx +/-")
    print("  a/d : dy +/-")
    print("  r/f : dz +/-")
    print("Rotation:")
    print("  i/k : dax +/-")
    print("  j/l : day +/-")
    print("  u/o : daz +/-")
    print("Gripper:")
    print("  n/m : gripper +/-")
    print("General:")
    print("  space : apply one env.step(current_action)")
    print("  z     : zero current action")
    print("  ]     : increase all step sizes")
    print("  [     : decrease all step sizes")
    print("  x     : reset environment")
    print("  p     : save current frame")
    print("  t     : type in all 7 action values manually")
    print("  h     : show help")
    print("  q     : quit")
    print("=========================\n")


def clamp_action(action: np.ndarray) -> np.ndarray:
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def action_string(action: np.ndarray) -> str:
    names = ["dx", "dy", "dz", "dax", "day", "daz", "grip"]
    return " | ".join(f"{n}={v:+.3f}" for n, v in zip(names, action))


fig, ax = plt.subplots(figsize=(6, 6))
img_artist = ax.imshow(get_display_frame(obs))
ax.axis("off")
title = ax.set_title("LIBERO manual control")
plt.subplots_adjust(bottom=0.15)

status_text = fig.text(
    0.02,
    0.02,
    "",
    fontsize=10,
    family="monospace",
)


def refresh_display(observation: dict, reward: float = 0.0, done: bool = False, info=None):
    if info is None:
        info = {}
    frame = get_display_frame(observation)
    img_artist.set_data(frame)
    title.set_text("LIBERO manual control")
    status_text.set_text(
        f"step={step_count} | reward={reward:.2f} | done={done} | success={info.get('success', False)}\n"
        f"{action_string(current_action)}\n"
        f"step_sizes: pos={DELTA_STEP:.3f}, rot={ROT_STEP:.3f}, grip={GRIPPER_STEP:.3f}"
    )
    fig.canvas.draw_idle()


save_current_frame(obs, "reset")
print_help()
refresh_display(obs)


def typed_action_input():
    global current_action
    print("\nEnter 7 comma-separated values for [dx, dy, dz, dax, day, daz, gripper]")
    raw = input("action> ").strip()
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 7:
        print("Expected exactly 7 values.")
        return
    try:
        vals = np.array([float(x) for x in parts], dtype=np.float32)
        current_action = clamp_action(vals)
        print("Updated action:", current_action)
    except ValueError:
        print("Invalid numeric input.")


def on_key(event):
    global current_action, obs, step_count, running
    global DELTA_STEP, ROT_STEP, GRIPPER_STEP

    key = event.key
    if key is None:
        return

    if key == "q":
        running = False
        plt.close(fig)
        return

    elif key == "h":
        print_help()

    elif key == "z":
        current_action[:] = 0.0
        print("Action zeroed.")

    elif key == "]":
        DELTA_STEP *= 1.25
        ROT_STEP *= 1.25
        GRIPPER_STEP *= 1.25
        print(f"Increased step sizes: pos={DELTA_STEP:.3f}, rot={ROT_STEP:.3f}, grip={GRIPPER_STEP:.3f}")

    elif key == "[":
        DELTA_STEP *= 0.8
        ROT_STEP *= 0.8
        GRIPPER_STEP *= 0.8
        print(f"Decreased step sizes: pos={DELTA_STEP:.3f}, rot={ROT_STEP:.3f}, grip={GRIPPER_STEP:.3f}")

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
        save_current_frame(obs, f"step_{step_count}")

    elif key == "x":
        obs = env.reset()
        step_count = 0
        print("Environment reset.")
        save_current_frame(obs, "reset_again")

    elif key == "t":
        typed_action_input()

    elif key == " ":
        current_action[:] = clamp_action(current_action)
        obs, reward, done, info = env.step(current_action)
        step_count += 1

        print(f"\n=== Step {step_count} ===")
        print("action:", current_action)
        print("reward:", reward)
        print("done:", done)
        print("info:", info)

        if done:
            print("Episode ended. Resetting environment.")
            obs = env.reset()
            step_count = 0

        refresh_display(obs, reward=reward, done=done, info=info)
        return

    current_action[:] = clamp_action(current_action)
    print("Current action:", current_action)
    refresh_display(obs)


cid = fig.canvas.mpl_connect("key_press_event", on_key)

refresh_display(obs)
plt.show()

fig.canvas.mpl_disconnect(cid)
env.close()
print("\nClosed environment.")
print("Outputs saved in:", OUTDIR.resolve())








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