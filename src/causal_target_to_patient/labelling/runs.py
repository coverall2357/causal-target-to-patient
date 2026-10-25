"""File names for labelling runs, and guards against overwriting earlier runs.

A run labels a list of trials. When the list comes from a file (a dev or test
set), the run's outputs are named after it, so each set has its own label and
review files; an ad hoc list of NCT numbers gets a dated name instead. Existing
outputs are never overwritten unless asked, because a review sheet may already
hold hand labels.
"""

from datetime import date
from pathlib import Path


def run_name(nct_file: Path | None, today: date) -> str:
    """The stem used for a run's output files."""
    return Path(nct_file).stem if nct_file else f"adhoc_{today:%Y%m%d}"


def run_outputs(out_dir: Path, name: str) -> dict[str, Path]:
    """Paths of a run's files: all answers, the review sheet, and the blind
    sheet (for labelling by hand before seeing the model's answers)."""
    return {"labels": Path(out_dir) / f"{name}_labels.csv",
            "review": Path(out_dir) / f"{name}_review.csv",
            "blind": Path(out_dir) / f"{name}_blind.csv"}


def refuse_overwrite(paths, force: bool = False) -> None:
    """Raise FileExistsError if any path exists, unless force is set."""
    existing = [str(p) for p in paths if Path(p).exists()]
    if existing and not force:
        raise FileExistsError(
            "would overwrite " + ", ".join(existing) + "; pass --force to replace")
