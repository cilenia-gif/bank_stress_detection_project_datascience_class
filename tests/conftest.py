"""
Add src/ to sys.path so tests can import project modules.
"""

import os
import sys

root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src = os.path.join(root, "src")

if src not in sys.path:
    sys.path.insert(0, src)
