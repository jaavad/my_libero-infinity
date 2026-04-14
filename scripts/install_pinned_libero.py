#!/usr/bin/env python3
"""Thin shim: delegates to ``libero_infinity.bootstrap``.

Kept so the Makefile and existing docs continue to work. The real logic lives
in the package so it ships in the wheel and can be invoked as the
``libero-inf-bootstrap`` console script after a PyPI install.
"""

from libero_infinity.bootstrap import main

if __name__ == "__main__":
    main()
