import os, yaml
from pathlib import Path

def load_experiments(descriptor_dir: Path) -> dict:
    """Scan descriptor_dir for *.yaml, return {experiment_type: descriptor_dict}."""
    exps = {}

    for fn in descriptor_dir.glob("*.yaml"):
        with open(fn) as f:
            desc = yaml.safe_load(f)
        # skip empty or invalid descriptor files
        if not isinstance(desc, dict) or "experiment_type" not in desc:
            # you can log a warning here if you like:
            print(f"Warning: skipping invalid experiment descriptor {fn}")
            continue
        exps[desc["experiment_type"]] = desc
        
    return exps