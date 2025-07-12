# experiment_factory.py

import os, yaml
from pathlib import Path

# cache descriptors in-process
_DESC_CACHE: dict[Path, dict] = {}

def load_experiments(descriptor_dir: Path) -> dict:
    """Scan descriptor_dir for *.yaml, return {experiment_type: descriptor_dict}."""
    exps = {}

    for fn in descriptor_dir.glob("*.yaml"):
        if fn not in _DESC_CACHE:
            _DESC_CACHE[fn] = yaml.safe_load(fn.read_text())
        desc = _DESC_CACHE[fn]
        # skip empty or invalid descriptor files
        if not isinstance(desc, dict) or "experiment_type" not in desc:
            # you can log a warning here if you like:
            print(f"Warning: skipping invalid experiment descriptor {fn}")
            continue
        exps[desc["experiment_type"]] = desc
        
    return exps
