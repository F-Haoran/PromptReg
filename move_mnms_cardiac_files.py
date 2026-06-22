#!/usr/bin/env python3
"""Move M&Ms short-axis cardiac files into PromptReg Cardiac folders.

Default source:
    /home/frankfei/PromptReg/Dataset/M&Ms/MnM

Default destinations:
    /home/frankfei/PromptReg/Dataset/Cardiac/images
    /home/frankfei/PromptReg/Dataset/Cardiac/labels

The script recursively scans Testing, Training, and Validation folders. Files
ending in `_sa` are moved to images; files ending in `_sa_gt` are moved to
labels. Supported medical volume extensions are .nii.gz and .nii.
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


DEFAULT_SOURCE_ROOT = Path("/home/frankfei/PromptReg/Dataset/M&Ms/MnM")
DEFAULT_IMAGE_DIR = Path("/home/frankfei/PromptReg/Dataset/Cardiac/images")
DEFAULT_LABEL_DIR = Path("/home/frankfei/PromptReg/Dataset/Cardiac/labels")
DEFAULT_SPLITS = ("Testing", "Training", "Validation")
VOLUME_SUFFIXES = (".nii.gz", ".nii")


@dataclass(frozen=True)
class MovePlan:
    source: Path
    destination: Path
    role: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move M&Ms *_sa images and *_sa_gt labels into Cardiac images/labels folders."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help=f"M&Ms MnM root containing Testing/Training/Validation. Default: {DEFAULT_SOURCE_ROOT}",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help=f"Destination for *_sa image files. Default: {DEFAULT_IMAGE_DIR}",
    )
    parser.add_argument(
        "--label-dir",
        type=Path,
        default=DEFAULT_LABEL_DIR,
        help=f"Destination for *_sa_gt label files. Default: {DEFAULT_LABEL_DIR}",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Split folders to scan below --source-root.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of moving them.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing destination files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned moves without changing files.",
    )
    return parser.parse_args()


def volume_suffix(path: Path) -> Optional[str]:
    name = path.name.lower()
    for suffix in VOLUME_SUFFIXES:
        if name.endswith(suffix):
            return suffix
    return None


def strip_volume_suffix(path: Path) -> str:
    suffix = volume_suffix(path)
    if suffix is None:
        return path.stem
    return path.name[: -len(suffix)]


def is_metadata_file(path: Path) -> bool:
    return path.name.startswith("._")


def classify_file(path: Path) -> Optional[str]:
    if not path.is_file() or is_metadata_file(path) or volume_suffix(path) is None:
        return None

    stem = strip_volume_suffix(path)
    if stem.endswith("_sa_gt"):
        return "label"
    if stem.endswith("_sa"):
        return "image"
    return None


def iter_split_files(source_root: Path, splits: Sequence[str]) -> Iterable[Path]:
    for split in splits:
        split_dir = source_root / split
        if not split_dir.exists():
            print(f"Warning: split folder does not exist, skipping: {split_dir}")
            continue
        if not split_dir.is_dir():
            print(f"Warning: split path is not a directory, skipping: {split_dir}")
            continue
        yield from sorted(split_dir.rglob("*"))


def build_move_plan(
    source_root: Path,
    image_dir: Path,
    label_dir: Path,
    splits: Sequence[str],
) -> List[MovePlan]:
    plans: List[MovePlan] = []

    for path in iter_split_files(source_root, splits):
        role = classify_file(path)
        if role is None:
            continue

        destination_dir = label_dir if role == "label" else image_dir
        plans.append(MovePlan(source=path, destination=destination_dir / path.name, role=role))

    return plans


def check_conflicts(plans: Sequence[MovePlan], overwrite: bool) -> None:
    seen_destinations = {}
    duplicate_destinations = []
    existing_destinations = []

    for plan in plans:
        previous_source = seen_destinations.get(plan.destination)
        if previous_source is not None:
            duplicate_destinations.append((plan.destination, previous_source, plan.source))
        else:
            seen_destinations[plan.destination] = plan.source

        if plan.destination.exists() and not overwrite:
            existing_destinations.append(plan.destination)

    if duplicate_destinations:
        examples = "\n".join(
            f"  {destination} <- {first_source} and {second_source}"
            for destination, first_source, second_source in duplicate_destinations[:10]
        )
        raise RuntimeError(f"Multiple source files would write the same destination:\n{examples}")

    if existing_destinations:
        examples = "\n".join(f"  {path}" for path in existing_destinations[:10])
        raise FileExistsError(
            f"{len(existing_destinations)} destination file(s) already exist. "
            f"Pass --overwrite if this is expected. Examples:\n{examples}"
        )


def execute_plan(plans: Sequence[MovePlan], copy: bool, overwrite: bool, dry_run: bool) -> None:
    image_count = sum(1 for plan in plans if plan.role == "image")
    label_count = sum(1 for plan in plans if plan.role == "label")
    action = "copy" if copy else "move"

    print(f"Planned {action}: {image_count} image file(s), {label_count} label file(s).")
    for plan in plans[:20]:
        print(f"  {plan.role}: {plan.source} -> {plan.destination}")
    if len(plans) > 20:
        print(f"  ... {len(plans) - 20} more file(s)")

    if dry_run:
        print("Dry run only; no files changed.")
        return

    for plan in plans:
        plan.destination.parent.mkdir(parents=True, exist_ok=True)
        if overwrite and plan.destination.exists():
            plan.destination.unlink()
        if copy:
            shutil.copy2(plan.source, plan.destination)
        else:
            shutil.move(str(plan.source), str(plan.destination))

    print(f"Completed {action} of {len(plans)} file(s).")


def main() -> int:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    image_dir = args.image_dir.expanduser().resolve()
    label_dir = args.label_dir.expanduser().resolve()

    if not source_root.exists():
        print(f"Source root does not exist: {source_root}")
        return 2
    if not source_root.is_dir():
        print(f"Source root is not a directory: {source_root}")
        return 2

    plans = build_move_plan(
        source_root=source_root,
        image_dir=image_dir,
        label_dir=label_dir,
        splits=args.splits,
    )
    if not plans:
        print(f"No *_sa or *_sa_gt NIfTI files found below {source_root}.")
        return 1

    check_conflicts(plans, overwrite=args.overwrite)
    execute_plan(plans, copy=args.copy, overwrite=args.overwrite, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
