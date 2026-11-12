# experiment_factory.py

# ────────────────────────────────────────────────────────────────
# Loads experiment descriptor YAML files from a directory,
# caches them in memory, and provides a mapping from each
# descriptor's "experiment_type" to its parsed dictionary.
# ────────────────────────────────────────────────────────────────


import yaml
from pathlib import Path

# path -> (modification time in nanoseconds, parsed descriptor)
_DESC_CACHE: dict[Path, tuple[int, dict]] = {}

def load_experiments(descriptor_dir: Path) -> dict:
    """
    Scan a directory for YAML descriptor files and load them into a dict.

    Args:
        descriptor_dir: Path to the directory containing *.yaml descriptor files.

    Returns:
        A dict mapping each descriptor's "experiment_type" string to its parsed dict.
    """
    exps = {}
    current_files = set(
        descriptor_dir.glob("*.yaml")
    )

    # Drop descriptors that were deleted.
    for cached_path in set(_DESC_CACHE) - current_files:
        _DESC_CACHE.pop(cached_path, None)

    # Iterate over every .yaml file in the directory
    for fn in sorted(current_files):
        modified_ns = fn.stat().st_mtime_ns
        cached = _DESC_CACHE.get(fn)

        if cached is None or cached[0] != modified_ns:
            parsed = yaml.safe_load(
                fn.read_text(encoding="utf-8")
            )
            _DESC_CACHE[fn] = (
                modified_ns,
                parsed,
            )

        desc = _DESC_CACHE[fn][1]
        
        # Validate that the descriptor is a dict and has the required key
        if not isinstance(desc, dict) or "experiment_type" not in desc:
            # Skip invalid or malformed descriptor files
            print(f"Warning: skipping invalid experiment descriptor {fn}")
            continue

        # Use the experiment_type as the lookup key
        exps[desc["experiment_type"]] = desc
        
    return exps