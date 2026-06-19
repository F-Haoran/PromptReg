#!/usr/bin/env python3
"""Build PromptReg train/test CSV files from extracted medical volumes.

The training code expects this layout:

    DATA_ROOT/
      Abdominal/csv/train.csv
      Abdominal/csv/test.csv
      Brain/csv/train.csv
      ...

Each CSV row contains relative paths from the task folder:

    moving_image,moving_label,fixed_image,fixed_label

This helper scans a dataset folder, matches images to labels by filename stem,
and writes the required CSV manifests. It can also safely extract .tar,
.tar.gz, and .tgz archives before scanning.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_TASKS = ("Abdominal", "Brain", "Cardiac", "Hippocampus", "Hip")
VOLUME_EXTENSIONS = (".nii.gz", ".nii", ".mha", ".mhd", ".nrrd", ".mgz")

LABEL_MARKERS = {
    "annotation",
    "annotations",
    "gt",
    "label",
    "labels",
    "mask",
    "masks",
    "seg",
    "segs",
    "segmentation",
    "segmentations",
}

IMAGE_MARKERS = {
    "ct",
    "fixed",
    "image",
    "images",
    "img",
    "imgs",
    "mri",
    "moving",
    "scan",
    "scans",
    "volume",
    "volumes",
}

ALL_ROLE_MARKERS = LABEL_MARKERS | IMAGE_MARKERS


@dataclass(frozen=True)
class Case:
    key: str
    image: str
    label: str


@dataclass(frozen=True)
class Pair:
    moving_image: str
    moving_label: str
    fixed_image: str
    fixed_label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate PromptReg train/test CSV files from a dataset folder or tar archive."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Extracted dataset root or a .tar/.tar.gz/.tgz archive.",
    )
    parser.add_argument(
        "--extract-to",
        type=Path,
        default=None,
        help="Directory used when INPUT is a tar archive. Defaults to INPUT parent / archive stem.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Root to scan after extraction. Usually not needed unless the archive has an extra top folder.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(DEFAULT_TASKS),
        help="Task folder names to scan.",
    )
    parser.add_argument(
        "--task-name",
        default=None,
        help="Treat INPUT/DATA_ROOT itself as one task with this name if task subfolders do not exist.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of generated pairs written to train.csv.",
    )
    parser.add_argument(
        "--pairing",
        choices=("adjacent", "all"),
        default="adjacent",
        help="How to create moving/fixed pairs from matched cases.",
    )
    parser.add_argument(
        "--bidirectional",
        action="store_true",
        help="Also add the reverse direction for every generated pair.",
    )
    parser.add_argument(
        "--subject-regex",
        default=None,
        help=(
            "Optional regex used to extract the case key from each relative path. "
            "If groups are present, the first group is used."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used before train/test splitting.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing csv/train.csv and csv/test.csv files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be written without creating CSV files.",
    )
    return parser.parse_args()


def is_tar_archive(path: Path) -> bool:
    return path.is_file() and tarfile.is_tarfile(path)


def archive_output_dir(archive: Path) -> Path:
    name = archive.name
    for suffix in (".tar.gz", ".tgz", ".tar"):
        if name.endswith(suffix):
            return archive.with_name(name[: -len(suffix)])
    return archive.with_suffix("")


def safe_extract_tar(archive: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    destination = destination.resolve()

    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            target = (destination / member.name).resolve()
            if destination not in target.parents and target != destination:
                raise ValueError(f"Unsafe tar member path: {member.name}")
        tar.extractall(destination)

    return destination


def volume_suffix(path: Path) -> Optional[str]:
    name = path.name.lower()
    for suffix in VOLUME_EXTENSIONS:
        if name.endswith(suffix):
            return suffix
    return None


def strip_volume_suffix(path: Path) -> str:
    name = path.name
    suffix = volume_suffix(path)
    if suffix is None:
        return path.stem
    return name[: -len(suffix)]


def split_tokens(value: str) -> List[str]:
    return [token for token in re.split(r"[^a-z0-9]+", value.lower()) if token]


def path_tokens(path: Path) -> List[str]:
    tokens: List[str] = []
    for part in path.parts:
        if part == path.name:
            tokens.extend(split_tokens(strip_volume_suffix(Path(part))))
        else:
            tokens.extend(split_tokens(part))
    return tokens


def strip_embedded_role_prefix(token: str) -> str:
    """Normalize tokens like img0001 and mask0001 to the shared key 0001."""
    for marker in sorted(ALL_ROLE_MARKERS, key=len, reverse=True):
        if not token.startswith(marker):
            continue

        remainder = token[len(marker):]
        if remainder and any(char.isdigit() for char in remainder):
            return remainder

    return token


def token_has_role_marker(token: str, markers: Iterable[str]) -> bool:
    if token in markers:
        return True

    for marker in sorted(markers, key=len, reverse=True):
        if not token.startswith(marker):
            continue

        remainder = token[len(marker):]
        if remainder and any(char.isdigit() for char in remainder):
            return True

    return False


def is_label_path(relative_path: Path) -> bool:
    tokens = path_tokens(relative_path)
    return any(token_has_role_marker(token, LABEL_MARKERS) for token in tokens)


def case_key(relative_path: Path, subject_regex: Optional[re.Pattern[str]]) -> str:
    relative_text = relative_path.as_posix()
    if subject_regex is not None:
        match = subject_regex.search(relative_text)
        if match:
            if match.groups():
                return match.group(1).lower()
            return match.group(0).lower()

    stem = strip_volume_suffix(relative_path).lower()
    tokens = [strip_embedded_role_prefix(token) for token in split_tokens(stem)]
    filtered = [token for token in tokens if token not in ALL_ROLE_MARKERS]
    if filtered:
        return "_".join(filtered)

    parent_tokens = [
        token
        for part in relative_path.parent.parts
        for token in split_tokens(part)
        if token not in ALL_ROLE_MARKERS and token != "csv"
    ]
    if parent_tokens:
        return "_".join(parent_tokens + tokens)

    return stem


def choose_one(paths: Sequence[str], role: str, key: str, task_dir: Path) -> str:
    if len(paths) > 1:
        print(
            f"Warning: multiple {role} files matched key '{key}' under {task_dir}; "
            f"using {paths[0]}",
            file=sys.stderr,
        )
    return paths[0]


def collect_cases(
    task_dir: Path, subject_regex: Optional[re.Pattern[str]]
) -> Tuple[List[Case], List[str]]:
    images: Dict[str, List[str]] = {}
    labels: Dict[str, List[str]] = {}
    warnings: List[str] = []

    for path in sorted(task_dir.rglob("*")):
        if not path.is_file() or volume_suffix(path) is None:
            continue

        relative_path = path.relative_to(task_dir)
        if relative_path.parts and relative_path.parts[0] == "csv":
            continue

        key = case_key(relative_path, subject_regex)
        relative_text = relative_path.as_posix()
        target = labels if is_label_path(relative_path) else images
        target.setdefault(key, []).append(relative_text)

    cases: List[Case] = []
    for key in sorted(set(images) & set(labels)):
        cases.append(
            Case(
                key=key,
                image=choose_one(images[key], "image", key, task_dir),
                label=choose_one(labels[key], "label", key, task_dir),
            )
        )

    missing_labels = sorted(set(images) - set(labels))
    missing_images = sorted(set(labels) - set(images))
    if missing_labels:
        warnings.append(
            f"{task_dir}: {len(missing_labels)} image key(s) have no label, for example: "
            + ", ".join(missing_labels[:5])
        )
    if missing_images:
        warnings.append(
            f"{task_dir}: {len(missing_images)} label key(s) have no image, for example: "
            + ", ".join(missing_images[:5])
        )

    return cases, warnings


def make_pairs(cases: Sequence[Case], strategy: str, bidirectional: bool) -> List[Pair]:
    pairs: List[Pair] = []
    if strategy == "adjacent":
        for moving, fixed in zip(cases, cases[1:]):
            pairs.append(to_pair(moving, fixed))
    elif strategy == "all":
        for moving_index, moving in enumerate(cases):
            for fixed_index, fixed in enumerate(cases):
                if moving_index == fixed_index:
                    continue
                pairs.append(to_pair(moving, fixed))
    else:
        raise ValueError(f"Unsupported pairing strategy: {strategy}")

    if bidirectional and strategy == "adjacent":
        reversed_pairs = [
            Pair(
                moving_image=pair.fixed_image,
                moving_label=pair.fixed_label,
                fixed_image=pair.moving_image,
                fixed_label=pair.moving_label,
            )
            for pair in pairs
        ]
        pairs.extend(reversed_pairs)

    return pairs


def to_pair(moving: Case, fixed: Case) -> Pair:
    return Pair(
        moving_image=moving.image,
        moving_label=moving.label,
        fixed_image=fixed.image,
        fixed_label=fixed.label,
    )


def split_pairs(pairs: Sequence[Pair], train_ratio: float, seed: int) -> Tuple[List[Pair], List[Pair]]:
    if not 0 < train_ratio < 1:
        raise ValueError("--train-ratio must be between 0 and 1.")

    pairs = list(pairs)
    random.Random(seed).shuffle(pairs)
    split_index = int(round(len(pairs) * train_ratio))
    if len(pairs) > 1:
        split_index = min(max(split_index, 1), len(pairs) - 1)
    return pairs[:split_index], pairs[split_index:]


def write_csv(path: Path, pairs: Sequence[Pair], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Pass --overwrite to replace it.")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("moving_image", "moving_label", "fixed_image", "fixed_label"),
        )
        writer.writeheader()
        for pair in pairs:
            writer.writerow(pair.__dict__)


def existing_task_dirs(data_root: Path, task_names: Iterable[str]) -> List[Tuple[str, Path]]:
    tasks = []
    for task_name in task_names:
        task_dir = data_root / task_name
        if task_dir.is_dir():
            tasks.append((task_name, task_dir))
    return tasks


def contains_volume_files(path: Path) -> bool:
    return any(candidate.is_file() and volume_suffix(candidate) is not None for candidate in path.rglob("*"))


def is_role_container_dir(path: Path) -> bool:
    tokens = split_tokens(path.name)
    return bool(tokens) and all(token in ALL_ROLE_MARKERS for token in tokens)


def inferred_task_dirs(data_root: Path) -> List[Tuple[str, Path]]:
    child_dirs = [
        child_dir
        for child_dir in sorted(data_root.iterdir())
        if child_dir.is_dir() and child_dir.name != "csv"
    ]
    task_dirs = [
        (child_dir.name, child_dir)
        for child_dir in child_dirs
        if not is_role_container_dir(child_dir) and contains_volume_files(child_dir)
    ]

    if task_dirs:
        return task_dirs

    if contains_volume_files(data_root):
        return [(data_root.name, data_root)]

    return []


def find_task_root(data_root: Path, task_names: Sequence[str]) -> Tuple[Path, List[Tuple[str, Path]]]:
    tasks = existing_task_dirs(data_root, task_names)
    if tasks:
        return data_root, tasks

    child_dirs = [path for path in sorted(data_root.iterdir()) if path.is_dir()]
    candidates: List[Tuple[Path, List[Tuple[str, Path]]]] = []
    for child_dir in child_dirs:
        child_tasks = existing_task_dirs(child_dir, task_names)
        if child_tasks:
            candidates.append((child_dir, child_tasks))

    if len(candidates) == 1:
        return candidates[0]

    tasks = inferred_task_dirs(data_root)
    if tasks:
        return data_root, tasks

    for child_dir in child_dirs:
        child_tasks = inferred_task_dirs(child_dir)
        if child_tasks:
            candidates.append((child_dir, child_tasks))

    if len(candidates) == 1:
        return candidates[0]

    return data_root, []


def resolve_data_root(args: argparse.Namespace) -> Path:
    input_path = args.input.expanduser().resolve()

    if is_tar_archive(input_path):
        extract_to = args.extract_to.expanduser() if args.extract_to else archive_output_dir(input_path)
        print(f"Extracting {input_path} -> {extract_to}")
        safe_extract_tar(input_path, extract_to)
        root = extract_to
    else:
        root = input_path

    if args.data_root:
        root = args.data_root.expanduser()

    root = root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {root}")
    return root


def print_summary(task_name: str, task_dir: Path, cases: Sequence[Case], pairs: Sequence[Pair]) -> None:
    print(f"{task_name}: {task_dir}")
    print(f"  matched cases: {len(cases)}")
    print(f"  generated pairs: {len(pairs)}")
    if cases:
        print(f"  first case: image={cases[0].image}, label={cases[0].label}")


def maybe_copy_into_named_task_dir(source: Path, task_name: str) -> Path:
    if source.name == task_name:
        return source

    target = source.parent / task_name
    if target.exists():
        return target

    print(
        f"Using {source} as task '{task_name}'. CSV paths will be relative to {source}; "
        f"rename or move it to {target} before training if train.py expects that folder name."
    )
    return source


def main() -> int:
    args = parse_args()
    data_root = resolve_data_root(args)
    subject_regex = re.compile(args.subject_regex) if args.subject_regex else None

    if args.task_name:
        task_dir = maybe_copy_into_named_task_dir(data_root, args.task_name)
        tasks = [(args.task_name, task_dir)]
    else:
        data_root, tasks = find_task_root(data_root, args.tasks)

    if not tasks:
        print(f"No task folders with volume files found under {data_root}.")
        print(f"First tried known task names: {', '.join(args.tasks)}")
        print("If this is a single-task dataset, pass --task-name TASK_NAME.")
        return 2

    total_pairs = 0
    for task_name, task_dir in tasks:
        cases, warnings = collect_cases(task_dir, subject_regex)
        for warning in warnings:
            print(f"Warning: {warning}", file=sys.stderr)

        pairs = make_pairs(cases, args.pairing, args.bidirectional)
        print_summary(task_name, task_dir, cases, pairs)

        if len(cases) < 2:
            print(f"Warning: {task_name} needs at least 2 matched cases to create registration pairs.")
            continue

        train_pairs, test_pairs = split_pairs(pairs, args.train_ratio, args.seed)
        total_pairs += len(pairs)

        if args.dry_run:
            print(f"  dry run: would write {len(train_pairs)} train rows and {len(test_pairs)} test rows")
            continue

        csv_dir = task_dir / "csv"
        write_csv(csv_dir / "train.csv", train_pairs, args.overwrite)
        write_csv(csv_dir / "test.csv", test_pairs, args.overwrite)
        print(f"  wrote {csv_dir / 'train.csv'} ({len(train_pairs)} rows)")
        print(f"  wrote {csv_dir / 'test.csv'} ({len(test_pairs)} rows)")

    if total_pairs == 0:
        print("No pairs were generated. Check filename matching or pass --subject-regex.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
