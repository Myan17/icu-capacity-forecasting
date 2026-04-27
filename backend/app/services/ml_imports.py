"""Add project root and lambdas/ to sys.path so the backend can import from ml.* and lambdas.*.

Import this module early (before any ml.* imports) to make the ML module
available without requiring a pip install.
"""
from __future__ import annotations

import sys
from pathlib import Path

# hospital-forecasting/  (3 levels up from backend/app/services/)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

for p in [str(PROJECT_ROOT), str(PROJECT_ROOT / "lambdas")]:
    if p not in sys.path:
        sys.path.insert(0, p)
