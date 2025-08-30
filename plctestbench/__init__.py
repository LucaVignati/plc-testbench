try:
    import importlib.resources as importlib_resources
except ImportError:
    import importlib_resources

from functools import lru_cache

import yaml

name = "ecc-testbench"

MODULES_MANIFEST_PATH = importlib_resources.files(__name__) / "modules_manifest.yaml"


@lru_cache
def get_available_modules():
    with open(MODULES_MANIFEST_PATH, "r") as file:
        modules = yaml.safe_load(file)
    return modules
