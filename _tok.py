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
# A present tr_api.py with absent data files means the submodule was only
# partially initialised (e.g. shallow clone without LFS). Check one of the
# essential JSON data files so the error surfaces here rather than deep
# inside load_inventory() with a cryptic FileNotFoundError.
if not (TOKENIZER_DIR / "inventory.json").exists():
    raise RuntimeError(
        f"turkish-tokenizer data files missing at {TOKENIZER_DIR} "
        "(inventory.json not found). "
        "Run: git submodule update --init --recursive")
