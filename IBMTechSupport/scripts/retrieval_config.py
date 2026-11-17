from pathlib import Path
from functools import lru_cache

import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_DIR / "configs" / "rag" / "retrieval.yaml"

@lru_cache(maxsize=1)
def load_retrieval_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)