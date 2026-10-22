"""Finding dated data files.

Scripts write outputs as <prefix>_YYYYMMDD.csv. latest_dated picks the newest
of these and ignores anything else with the same prefix, such as backups
(<prefix>_YYYYMMDD_before_x.csv), so a stray copy can't be mistaken for data.
"""

import re
from pathlib import Path

_DATED = re.compile(r"_(\d{8})\.csv$")


def latest_dated(data_dir: Path, prefix: str) -> Path:
    """Newest <prefix>_YYYYMMDD.csv in data_dir."""
    exact = re.compile(re.escape(prefix) + r"_\d{8}\.csv")
    files = sorted(p for p in Path(data_dir).glob(f"{prefix}_*.csv")
                   if exact.fullmatch(p.name))
    if not files:
        raise FileNotFoundError(f"No {prefix}_YYYYMMDD.csv in {data_dir}")
    return files[-1]


def stamp_of(path: Path) -> str:
    """The YYYYMMDD stamp of a dated file."""
    match = _DATED.search(Path(path).name)
    if not match:
        raise ValueError(f"Not a dated file: {path}")
    return match.group(1)
