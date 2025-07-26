# experiment_factory.py

# ────────────────────────────────────────────────────────────────
# Loads experiment descriptor YAML files from a directory,
# caches them in memory, and provides a mapping from each
# descriptor's "experiment_type" to its parsed dictionary.
# ────────────────────────────────────────────────────────────────


import os, yaml
from pathlib import Path

# In-memory cache: maps descriptor file paths to their parsed dicts
_DESC_CACHE: dict[Path, dict] = {}

def load_experiments(descriptor_dir: Path) -> dict:
    """
    Scan a directory for YAML descriptor files and load them into a dict.

    Args:
        descriptor_dir: Path to the directory containing *.yaml descriptor files.

    Returns:
        A dict mapping each descriptor's "experiment_type" string to its parsed dict.
    """
    exps = {}

    # Iterate over every .yaml file in the directory
    for fn in descriptor_dir.glob("*.yaml"):
        # If not already cached, read and parse the YAML
        if fn not in _DESC_CACHE:
            _DESC_CACHE[fn] = yaml.safe_load(fn.read_text())
        desc = _DESC_CACHE[fn]
        
        # Validate that the descriptor is a dict and has the required key
        if not isinstance(desc, dict) or "experiment_type" not in desc:
            # Skip invalid or malformed descriptor files
            print(f"Warning: skipping invalid experiment descriptor {fn}")
            continue

        # Use the experiment_type as the lookup key
        exps[desc["experiment_type"]] = desc
        
    return exps
