#!/usr/bin/env python3
"""
Thin compatibility wrapper — the real CLI now lives at
backend/scripts/run.py (part of the khidmah package). This forwards to it
so `python run.py` keeps working unchanged from the repo root.
"""

import os
import runpy
import sys

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.join(_REPO_ROOT, "backend")

sys.path.insert(0, _BACKEND)
runpy.run_path(os.path.join(_BACKEND, "scripts", "run.py"), run_name="__main__")
