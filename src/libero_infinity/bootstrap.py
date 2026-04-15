"""Install pinned upstream LIBERO into the active Python environment.

Upstream LIBERO (`Lifelong-Robot-Learning/LIBERO`) is not on PyPI and cannot be
installed as a plain PEP 508 git dependency: the top-level ``libero`` package
directory is missing an ``__init__.py``, so pip-installing the repo produces a
distribution that imports as an empty namespace package. This module downloads
the pinned source archive, patches in the missing ``__init__.py``, and installs
the patched tree into ``sys.executable`` with ``--no-deps``.

Exposed as the ``libero-inf-bootstrap`` console script so users who installed
``libero-infinity`` from PyPI can run it once after install:

    pip install libero-infinity
    libero-inf-bootstrap
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

LIBERO_REPO = "https://github.com/Lifelong-Robot-Learning/LIBERO"
LIBERO_COMMIT = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
ARCHIVE_URL = f"https://codeload.github.com/Lifelong-Robot-Learning/LIBERO/tar.gz/{LIBERO_COMMIT}"
CACHE_ROOT = pathlib.Path.home() / ".cache" / "libero_infinity" / "src"
SOURCE_DIR = CACHE_ROOT / f"LIBERO-{LIBERO_COMMIT}"


def _download_and_unpack(force: bool = False) -> pathlib.Path:
    if force and SOURCE_DIR.exists():
        shutil.rmtree(SOURCE_DIR)
    if SOURCE_DIR.exists():
        return SOURCE_DIR

    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=CACHE_ROOT, prefix=".libero_download_") as tmp:
        tmp_path = pathlib.Path(tmp)
        archive_path = tmp_path / "libero.tar.gz"
        with urllib.request.urlopen(ARCHIVE_URL) as response, archive_path.open("wb") as f:
            shutil.copyfileobj(response, f)
        with tarfile.open(archive_path, mode="r:gz") as tf:
            tf.extractall(tmp_path)

        extracted_root = next(tmp_path.glob("LIBERO-*"))
        extracted_root.rename(SOURCE_DIR)

    return SOURCE_DIR


def _patch_source_tree(source_dir: pathlib.Path) -> None:
    (source_dir / "libero" / "__init__.py").touch(exist_ok=True)


def _pip_install(source_dir: pathlib.Path) -> None:
    # Prefer uv if available — it's faster and respects the active environment.
    # Fall back to plain pip so users who installed via `pip install libero-infinity`
    # from PyPI (no uv) can still bootstrap.
    if shutil.which("uv") is not None:
        cmd = [
            "uv",
            "pip",
            "install",
            "--python",
            sys.executable,
            "--no-deps",
            str(source_dir),
        ]
    else:
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            str(source_dir),
        ]
    subprocess.run(cmd, check=True)


def install(force: bool = False) -> pathlib.Path:
    source_dir = _download_and_unpack(force=force)
    _patch_source_tree(source_dir)
    _pip_install(source_dir)
    return source_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="libero-inf-bootstrap",
        description="Install pinned upstream LIBERO source into the active Python environment.",
    )
    parser.add_argument("--force", action="store_true", help="Re-download the pinned source tree")
    args = parser.parse_args()
    install(force=args.force)
    print(f"Installed LIBERO from {LIBERO_REPO}@{LIBERO_COMMIT[:8]}")


if __name__ == "__main__":
    main()
