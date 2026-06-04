"""Shared helper: make the bundled turkish-tokenizer submodule importable.

The morphological tokenizer lives in vendor/turkish-tokenizer (a git
submodule). Importing this module puts it on sys.path so scripts can
`from tr_api import Tokenizer`, etc.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
TOKENIZER_DIR = REPO_ROOT / "vendor" / "turkish-tokenizer"

if str(TOKENIZER_DIR) not in sys.path:
    sys.path.insert(0, str(TOKENIZER_DIR))

if not (TOKENIZER_DIR / "tr_api.py").exists():
    raise RuntimeError(
        f"turkish-tokenizer submodule not found at {TOKENIZER_DIR}. "
        "Run: git submodule update --init --recursive")
