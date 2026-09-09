"""Make ``import bonnuit`` work without installing the package.

The source lives in ``src/bonnuit``. There is no ``pyproject.toml`` / editable
install, so this adds ``src/`` to ``sys.path`` for the test session. To run the
modules outside pytest, use ``PYTHONPATH=src`` (see requirements.txt).
"""

import sys
from pathlib import Path

SRC = Path(__file__).parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
