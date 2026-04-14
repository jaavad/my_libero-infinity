# Installation Guide

[Back to main README](../README.md)

---

## Prerequisites

- **Python 3.11+** (required by Scenic 3)
- **[`uv`](https://docs.astral.sh/uv/)** package manager (recommended)
- **Git** for cloning repos
- **[`make`](https://www.gnu.org/software/make/)** build tool — Linux: `sudo apt install make`, macOS: `xcode-select --install` or `brew install make`
- **EGL or OSMesa** for headless MuJoCo rendering (present on most Linux systems)

> **No make?** Every `make` target has a plain-shell equivalent. See the *Without make* sections below.

---

## Hardware Requirements

A GPU with EGL support is required for MuJoCo rendering (set `MUJOCO_GL=egl` on headless servers); OSMesa can be used as a CPU-only fallback but is significantly slower. For Pi0.5 or similar VLA inference, plan for at least **~4 GB VRAM**; 8 GB or more is comfortable for larger models. System RAM of **~16 GB** is recommended when running simulation and model inference together. Allow **~10 GB of free disk space** for the Hugging Face LIBERO asset cache downloaded by `make setup-assets`.

---

## Install from PyPI

> Coming soon — once published, the workflow will be:
>
> ```bash
> pip install libero-infinity
> libero-inf-bootstrap        # one-time: installs pinned upstream LIBERO
> python -c "from libero_infinity.runtime import ensure_runtime; ensure_runtime()"  # one-time: HF assets
> ```
>
> `libero-inf-bootstrap` exists because upstream LIBERO is not on PyPI and its
> repo is missing a top-level `__init__.py`, so it cannot be expressed as a
> normal dependency. The bootstrap downloads the pinned source, patches the
> missing file, and installs it into the active environment.

---

## Quick Install (recommended)

```bash
# 1. Install uv (fast Python package manager) — https://docs.astral.sh/uv/
curl -LsSf https://astral.sh/uv/install.sh | sh
# Fallback: pip install uv

# 2. Clone and install
git clone https://github.com/KE7/libero-infinity.git && cd libero-infinity
make install        # creates venv, installs deps, installs pinned LIBERO source
make setup-assets   # downloads LIBERO assets from HF (only needed once)
make test           # runs full test suite (headless)
```

<details>
<summary><strong>Without make — raw equivalent commands</strong></summary>

```bash
git clone https://github.com/KE7/libero-infinity.git && cd libero-infinity

# Equivalent to: make install
uv venv --python 3.11
uv sync --extra dev
uv run python scripts/install_pinned_libero.py

# Equivalent to: make setup-assets
PYTHONPATH=src uv run python -c "from libero_infinity.runtime import ensure_runtime; ensure_runtime()"

# Equivalent to: make test
MUJOCO_GL=egl uv run python -m pytest tests/test_e2e.py -v

# Equivalent to: make verify
MUJOCO_GL=egl uv run python scripts/verify_build.py
```

</details>

`make install` does:

1. Creates a Python 3.11 virtual environment via `uv`
2. Installs all dependencies (Scenic 3, MuJoCo, robosuite, bddl, gym, …)
3. Installs LIBERO from a pinned upstream source snapshot (`8f1084e`)

`make setup-assets` downloads and validates LIBERO assets from Hugging Face, writes
`~/.libero/config.yaml`, and refreshes the LIBERO assets link (only needed once).

---

## Manual Install

### Step 1: Clone the repository

```bash
git clone https://github.com/KE7/libero-infinity.git
cd libero-infinity
```

### Step 2: Create virtual environment and install

```bash
uv venv --python 3.11
uv sync --extra dev
uv run libero-inf-bootstrap   # installs pinned upstream LIBERO (one-time)
```

This installs two groups of dependencies:

| Group | Contents |
|-------|----------|
| **Core** (always) | Scenic 3, numpy, scipy, pyyaml, python-fcl, MuJoCo, robosuite, bddl, gym |
| **Dev** (`--extra dev`) | pytest, ruff, ipython, rich |

LIBERO itself is installed from a pinned upstream source snapshot (commit `8f1084e`).

### Step 3: Activate the virtual environment

```bash
source .venv/bin/activate   # Linux / macOS
# Windows: .venv\Scripts\activate
```

> **Tip:** All subsequent commands can also be run without activation by
> prefixing with `uv run` (e.g. `uv run python ...`). Activation is only
> needed if you want to invoke `python` or `libero-eval` directly from your
> shell without the `uv run` wrapper.

### Step 4: Configure LIBERO assets

```bash
make setup-assets
```

This downloads and validates LIBERO assets from Hugging Face (pinned to `lerobot/libero-assets@0b3ea86`), then writes `~/.libero/config.yaml`. Only needs to be run once.

### Step 5: Verify the installation

```bash
MUJOCO_GL=egl uv run python -m pytest tests/test_e2e.py -v
```

Or run the comprehensive 8-step verification:

```bash
MUJOCO_GL=egl uv run python scripts/verify_build.py
```

---

## Optional: VerifAI for Adversarial Search

To enable adversarial cross-entropy search over Scenic distributions:

```bash
uv sync --extra verifai --extra dev
```

This adds the [VerifAI](https://github.com/BerkeleyLearnVerify/VerifAI) package, enabling `VerifaiRange`-based sampling in Scenic programs and the `--mode adversarial` CLI flag.

---

## Platform Support

### Linux x86_64

Fully supported. All Python wheels are available from PyPI.

### Linux ARM64 (aarch64)

Fully supported. `python-fcl` 0.7.0.11+ ships `manylinux_2_28_aarch64` wheels on PyPI. MuJoCo builds from its sdist on ARM64.

### macOS Apple Silicon

Scenic 3 installs natively on macOS ARM64. `python-fcl` has ARM64 macOS wheels. MuJoCo supports Apple Silicon.

---

## Rendering Backend

The `MUJOCO_GL` environment variable controls the rendering backend:

| Value | When to Use |
|-------|-------------|
| `egl` | Headless servers (most common for evaluation) |
| `osmesa` | If EGL is not available |
| *(unset)* | macOS with native OpenGL |

```bash
# Headless evaluation
MUJOCO_GL=egl libero-eval --bddl path/to/task.bddl --n-scenes 100

# Live visualization (requires DISPLAY)
MUJOCO_GL=egl libero-eval --bddl path/to/task.bddl --watch cv2
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'libero'`

Ensure LIBERO was installed into the active environment:

```bash
uv run python scripts/install_pinned_libero.py
```

Also check that the vendored BDDL files are present:

```bash
ls src/libero_infinity/data/libero_runtime/bddl_files/  # should show libero_goal/, etc.
```

### `Could not import libero`

Install (or reinstall) the pinned LIBERO package:

```bash
uv run python scripts/install_pinned_libero.py
```

### `GLFW: The DISPLAY environment variable is missing`

You're trying to use `--watch viewer` without a display. Use headless mode:

```bash
MUJOCO_GL=egl libero-eval --bddl path/to/task.bddl --n-scenes 10  # no --watch
```

### `EGL is not available`

Install EGL libraries or use OSMesa:

```bash
# Ubuntu/Debian
sudo apt-get install libegl1-mesa-dev

# Or use OSMesa fallback
MUJOCO_GL=osmesa libero-eval --bddl path/to/task.bddl
```

---

## Development Setup

For contributors:

```bash
git clone https://github.com/KE7/libero-infinity.git && cd libero-infinity
make install
make setup-assets

# Run linter
uv run ruff check src/ tests/

# Run tests
MUJOCO_GL=egl uv run python -m pytest tests/ -v

# Run specific test tier
MUJOCO_GL=egl uv run python -m pytest tests/test_scenic.py -v   # Scenic only (fast)
MUJOCO_GL=egl uv run python -m pytest tests/test_libero.py -v   # Simulation integration
MUJOCO_GL=egl uv run python -m pytest tests/test_gym.py -v      # Gym wrapper
```

<details>
<summary><strong>Without make — equivalent dev setup</strong></summary>

```bash
# Equivalent to: make install
uv venv --python 3.11
uv sync --extra dev
uv run python scripts/install_pinned_libero.py

# Equivalent to: make setup-assets
PYTHONPATH=src uv run python -c "from libero_infinity.runtime import ensure_runtime; ensure_runtime()"

# Run linter
uv run ruff check src/ tests/

# Equivalent to: make test
MUJOCO_GL=egl uv run python -m pytest tests/test_e2e.py -v

# Equivalent to: make verify
MUJOCO_GL=egl uv run python scripts/verify_build.py

# Run specific test tiers
MUJOCO_GL=egl uv run python -m pytest tests/test_scenic.py -v
MUJOCO_GL=egl uv run python -m pytest tests/test_libero.py -v
MUJOCO_GL=egl uv run python -m pytest tests/test_gym.py -v
```

</details>

See [contributing.md](contributing.md) for contribution guidelines.
