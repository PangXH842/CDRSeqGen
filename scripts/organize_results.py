"""Build a category view of migrated results without changing source batches."""

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "results_catalog.json"
DESTINATION = ROOT / "results" / "by_experiment"


def main() -> None:
    catalog = json.loads(CATALOG.read_text())
    for category, entries in catalog.items():
        category_dir = DESTINATION / category
        category_dir.mkdir(parents=True, exist_ok=True)
        for name, source in entries.items():
            target = (ROOT / source).resolve()
            if not target.is_dir() or not target.is_relative_to(ROOT / "results"):
                raise ValueError(f"Invalid result source: {source}")
            link = category_dir / name
            relative = os.path.relpath(target, category_dir)
            if link.is_symlink():
                if link.readlink() != Path(relative):
                    raise ValueError(f"Existing link has a different target: {link}")
            elif link.exists():
                raise ValueError(f"Refusing to replace existing path: {link}")
            else:
                link.symlink_to(relative, target_is_directory=True)
    print(f"Category view ready: {DESTINATION}")


if __name__ == "__main__":
    main()
