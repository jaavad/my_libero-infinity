from pathlib import Path
import json
import numpy as np
from PIL import Image
import imageio.v2 as imageio
import matplotlib.pyplot as plt

from libero_infinity.gym_env import LIBEROScenicEnv


REPO_ROOT = Path(__file__).resolve().parents[1]

# Pick one task
BDDL_PATH = REPO_ROOT / "src/libero_infinity/data/libero_runtime/bddl_files/libero_goal/put_the_bowl_on_the_plate.bddl"

# Where to save outputs
OUTDIR = REPO_ROOT / "my_tests" / "debug_manual_env_live"
OUTDIR.mkdir(exist_ok=True, parents=True)

# Settings
RESOLUTION = 128
MAX_STEPS = 50
ROLLOUT_STEPS = 30
SAVE_GIF = True
GIF_PATH = OUTDIR / "agentview_rollout.gif"

# Create one environment
env = LIBEROScenicEnv(
    bddl_path=str(BDDL_PATH),
    perturbation="position",
    resolution=RESOLUTION,
    max_steps=MAX_STEPS,
)

print("=== Environment created ===")
print("action_space:", env.action_space)

# Reset
obs = env.reset()

print("\n=== After reset ===")
print("observation keys:", list(obs.keys()))
print("observation_space:", env.observation_space)

# Save metadata about observations
obs_meta = {}
for k, v in obs.items():
    if isinstance(v, np.ndarray):
        entry = {
            "shape": list(v.shape),
            "dtype": str(v.dtype),
        }
        if v.size > 0 and np.issubdtype(v.dtype, np.number):
            entry["min"] = float(v.min())
            entry["max"] = float(v.max())
        obs_meta[k] = entry
    else:
        obs_meta[k] = {"type": str(type(v))}

with open(OUTDIR / "obs_meta_reset.json", "w") as f:
    json.dump(obs_meta, f, indent=2)

# Save reset frames
for cam_key in ["agentview_image", "robot0_eye_in_hand_image"]:
    if cam_key in obs:
        frame = obs[cam_key][::-1]  # Flip vertically for viewing
        Image.fromarray(frame).save(OUTDIR / f"{cam_key}_reset.png")
        print(f"saved {cam_key}_reset.png")

# Live visualization setup
if "agentview_image" not in obs:
    raise KeyError("agentview_image not found in observation; cannot show live view.")

plt.ion()
fig, ax = plt.subplots(figsize=(5, 5))
initial_frame = obs["agentview_image"][::-1]
img_artist = ax.imshow(initial_frame)
ax.set_title("LIBERO live view: reset")
ax.axis("off")
plt.show(block=False)
plt.pause(0.1)

# Optional GIF frames
gif_frames = [initial_frame.copy()] if SAVE_GIF else []

# Take a few zero / random actions just to inspect stepping
# Action format is (7,) in [-1, 1]
# [dx, dy, dz, dax, day, daz, gripper]
zero_action = np.zeros(7, dtype=np.float32)

for t in range(ROLLOUT_STEPS):
    if t == 0:
        action = zero_action
    else:
        action = env.action_space.sample().astype(np.float32) * 0.1  # small random action

    obs, reward, done, info = env.step(action)

    print(f"\n=== Step {t} ===")
    print("action:", action)
    print("reward:", reward)
    print("done:", done)
    print("info:", info)

    # Save step frames
    for cam_key in ["agentview_image", "robot0_eye_in_hand_image"]:
        if cam_key in obs:
            frame = obs[cam_key][::-1]
            Image.fromarray(frame).save(OUTDIR / f"{cam_key}_step_{t}.png")

    # Update live view
    live_frame = obs["agentview_image"][::-1]
    img_artist.set_data(live_frame)
    ax.set_title(
        f"LIBERO live view | step={t} | reward={reward:.2f} | "
        f"success={info.get('success', False)}"
    )
    fig.canvas.draw_idle()
    plt.pause(0.15)

    if SAVE_GIF:
        gif_frames.append(live_frame.copy())

    if done:
        print("Episode ended early.")
        break

env.close()

# Save GIF
if SAVE_GIF and len(gif_frames) > 1:
    imageio.mimsave(GIF_PATH, gif_frames, fps=5)
    print(f"saved GIF: {GIF_PATH}")

plt.ioff()
plt.show()

print("\nDone. Outputs saved in:", OUTDIR.resolve())