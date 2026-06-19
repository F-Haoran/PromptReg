#!/usr/bin/env python3
"""Extract all tar archives from the PromptReg Brain dataset folder.

Default behavior:

    python3 extract_brain_tars.py

This scans /home/frankfei/PromptReg/Dataset/Brain for tar archives and extracts
their contents into the same folder. The script checks archive member paths
before extraction so a malicious tar cannot write outside the target directory.
"""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path
from typing import Iterable, List


DEFAULT_BRAIN_DIR = Path("/home/frankfei/PromptReg/Dataset/Brain")
ARCHIVE_SUFFIXES = (".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract all tar archives in the PromptReg Brain dataset folder."
    )
    parser.add_argument(
        "--folder",
        type=Path,
        default=DEFAULT_BRAIN_DIR,
        help=f"Folder containing tar archives. Default: {DEFAULT_BRAIN_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Extraction directory. Defaults to the same folder as --folder.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow archive members to overwrite existing files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print archives and member counts without extracting.",
    )
    return parser.parse_args()


def looks_like_tar(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in ARCHIVE_SUFFIXES)


def find_archives(folder: Path) -> List[Path]:
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and looks_like_tar(path) and tarfile.is_tarfile(path)
    )


def ensure_safe_members(archive: tarfile.TarFile, output_dir: Path) -> None:
    output_dir = output_dir.resolve()
    for member in archive.getmembers():
        target_path = (output_dir / member.name).resolve()
        if target_path != output_dir and output_dir not in target_path.parents:
            raise RuntimeError(f"Unsafe tar member path in {archive.name}: {member.name}")


def existing_targets(archive: tarfile.TarFile, output_dir: Path) -> List[Path]:
    output_dir = output_dir.resolve()
    targets: List[Path] = []
    for member in archive.getmembers():
        target_path = (output_dir / member.name).resolve()
        if target_path.exists():
            targets.append(target_path)
    return targets


def extract_archive(archive_path: Path, output_dir: Path, overwrite: bool, dry_run: bool) -> int:
    with tarfile.open(archive_path) as archive:
        members = archive.getmembers()
        ensure_safe_members(archive, output_dir)

        if dry_run:
            print(f"[dry-run] {archive_path.name}: {len(members)} member(s)")
            return len(members)

        if not overwrite:
            conflicts = existing_targets(archive, output_dir)
            if conflicts:
                examples = ", ".join(str(path) for path in conflicts[:5])
                raise FileExistsError(
                    f"{archive_path.name} would overwrite {len(conflicts)} existing path(s), "
                    f"for example: {examples}. Pass --overwrite if this is expected."
                )

        print(f"Extracting {archive_path.name} -> {output_dir}")
        archive.extractall(output_dir)
        return len(members)


def extract_all(archives: Iterable[Path], output_dir: Path, overwrite: bool, dry_run: bool) -> None:
    total_archives = 0
    total_members = 0
    for archive_path in archives:
        member_count = extract_archive(archive_path, output_dir, overwrite, dry_run)
        total_archives += 1
        total_members += member_count

    action = "Checked" if dry_run else "Extracted"
    print(f"{action} {total_archives} archive(s), {total_members} total member(s).")


def main() -> int:
    args = parse_args()
    folder = args.folder.expanduser().resolve()
    output_dir = (args.output_dir or folder).expanduser().resolve()

    if not folder.exists():
        print(f"Folder does not exist: {folder}")
        return 2
    if not folder.is_dir():
        print(f"Not a directory: {folder}")
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    archives = find_archives(folder)
    if not archives:
        print(f"No tar archives found in {folder}")
        return 1

    extract_all(archives, output_dir, args.overwrite, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
