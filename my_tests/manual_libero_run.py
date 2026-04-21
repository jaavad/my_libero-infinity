from pathlib import Path
import json
import numpy as np
from PIL import Image

from libero_infinity.gym_env import LIBEROScenicEnv


from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# Pick one task
BDDL_PATH = REPO_ROOT / "src/libero_infinity/data/libero_runtime/bddl_files/libero_goal/put_the_bowl_on_the_plate.bddl"

# Where to save outputs
OUTDIR = Path("debug_manual_env")
OUTDIR.mkdir(exist_ok=True, parents=True)

# Create one environment
env = LIBEROScenicEnv(
    bddl_path=str(BDDL_PATH),
    perturbation="position",
    resolution=128,
    max_steps=50,
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
        obs_meta[k] = {
            "shape": list(v.shape),
            "dtype": str(v.dtype),
            "min": float(v.min()) if v.size > 0 and np.issubdtype(v.dtype, np.number) else None,
            "max": float(v.max()) if v.size > 0 and np.issubdtype(v.dtype, np.number) else None,
        }
    else:
        obs_meta[k] = {"type": str(type(v))}

with open(OUTDIR / "obs_meta_reset.json", "w") as f:
    json.dump(obs_meta, f, indent=2)

# Save frames if present
for cam_key in ["agentview_image", "robot0_eye_in_hand_image"]:
    if cam_key in obs:
        # Flip vertically because OpenGL images have origin at bottom-left
        frame = obs[cam_key][::-1]
        Image.fromarray(frame).save(OUTDIR / f"{cam_key}_reset.png")
        print(f"saved {cam_key}_reset.png")

# Take a few zero / random actions just to inspect stepping
# Action format is (7,) in [-1, 1]
# [dx, dy, dz, dax, day, daz, gripper]
zero_action = np.zeros(7, dtype=np.float32)

for t in range(5):
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

    for cam_key in ["agentview_image", "robot0_eye_in_hand_image"]:
        if cam_key in obs:
            frame = obs[cam_key][::-1]
            Image.fromarray(frame).save(OUTDIR / f"{cam_key}_step_{t}.png")

    if done:
        print("Episode ended early.")
        break

env.close()
print("\nDone. Outputs saved in:", OUTDIR.resolve())