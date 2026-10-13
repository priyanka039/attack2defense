"""Run the project from this file.

The old notebook script used to live here. The same experiments are now a
small package. From this folder:

    python comparing_models.py train
    python comparing_models.py evaluate --checkpoint outputs/defense/best.pt
    python comparing_models.py benchmark --num-images 100

``python -m attack2defense`` does the same thing.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attack2defense.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
