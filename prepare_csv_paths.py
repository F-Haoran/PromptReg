#!/usr/bin/env python3
import argparse
import csv
import os
from pathlib import Path


DEFAULT_DATA_ROOT = '/home/FrankFei/PromptReg'
REQUIRED_COLUMNS = ('moving_image', 'moving_label', 'fixed_image', 'fixed_label')
DEFAULT_SPLITS = ('train', 'test')
TASK_ALIASES = {
    'Abdominal': ('Abdominal', 'ABDO'),
    'Brain': ('Brain',),
    'Cardiac': ('Cardiac', 'Heart'),
    'Hippocampus': ('Hippocampus', 'Haima'),
    'Hip': ('Hip',),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Validate PromptReg CSV files and rewrite old absolute paths to task-relative paths.'
    )
    parser.add_argument('--data-root', default=os.environ.get('PROMPTREG_DATA_ROOT', DEFAULT_DATA_ROOT),
                        help='Root directory containing PromptReg task folders')
    parser.add_argument('--tasks', nargs='+', default=sorted(TASK_ALIASES),
                        help='Task folders to process')
    parser.add_argument('--splits', nargs='+', default=list(DEFAULT_SPLITS),
                        help='CSV splits to process, for example train test')
    parser.add_argument('--write', action='store_true',
                        help='Rewrite CSV files in place; default is a dry run')
    return parser.parse_args()


def is_relative_to(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def suffix_after_alias(path, aliases):
    parts = os.path.normpath(path).split(os.sep)
    for alias in aliases:
        if alias not in parts:
            continue
        alias_index = len(parts) - 1 - parts[::-1].index(alias)
        suffix = parts[alias_index + 1:]
        if suffix:
            return os.path.join(*suffix)
    return None


def normalize_entry(value, task_root, aliases):
    if value is None:
        return value

    stripped = value.strip()
    if not stripped:
        return stripped

    expanded = os.path.abspath(os.path.expanduser(stripped)) if os.path.isabs(stripped) else stripped
    if not os.path.isabs(expanded):
        return stripped

    path = Path(expanded)
    if is_relative_to(path, task_root):
        return os.path.relpath(path, task_root)

    suffix = suffix_after_alias(expanded, aliases)
    if suffix:
        return suffix

    return stripped


def data_file_exists(value, task_root):
    if value is None or not value.strip():
        return False

    expanded = os.path.expanduser(value.strip())
    path = Path(expanded) if os.path.isabs(expanded) else task_root / expanded
    return path.exists()


def write_csv(csv_path, fieldnames, rows):
    tmp_path = csv_path.with_suffix(csv_path.suffix + '.tmp')
    with tmp_path.open('w', newline='') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(csv_path)


def process_csv(csv_path, task_root, aliases, write):
    if not csv_path.exists():
        print(f'[missing csv] {csv_path}')
        return 0, 1, 0

    with csv_path.open(newline='') as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None:
            print(f'[invalid csv] {csv_path}: missing header row')
            return 0, 0, 1

        missing_columns = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
        if missing_columns:
            print(f'[invalid csv] {csv_path}: missing columns {missing_columns}')
            return 0, 0, 1

        rows = []
        change_count = 0
        missing_file_count = 0
        for line_number, row in enumerate(reader, start=2):
            for column in REQUIRED_COLUMNS:
                old_value = row[column]
                new_value = normalize_entry(old_value, task_root, aliases)
                if new_value != old_value:
                    row[column] = new_value
                    change_count += 1
                    print(f'[path update] {csv_path}:{line_number}:{column}: {old_value} -> {new_value}')

                if not data_file_exists(row[column], task_root):
                    missing_file_count += 1
                    print(f'[missing file] {csv_path}:{line_number}:{column}: {row[column]}')
            rows.append(row)

    if write and change_count:
        write_csv(csv_path, reader.fieldnames, rows)
        print(f'[written] {csv_path}: {change_count} path value(s) updated')
    elif change_count:
        print(f'[dry run] {csv_path}: {change_count} path value(s) would be updated')
    else:
        print(f'[ok] {csv_path}: no path updates needed')

    return change_count, 0, missing_file_count


def main():
    args = parse_args()
    data_root = Path(os.path.abspath(os.path.expanduser(args.data_root)))

    total_changes = 0
    total_missing_csvs = 0
    total_missing_files = 0

    for task in args.tasks:
        if task not in TASK_ALIASES:
            raise ValueError(f'Unknown task {task}. Expected one of: {sorted(TASK_ALIASES)}')

        task_root = data_root / task
        for split in args.splits:
            csv_path = task_root / 'csv' / f'{split}.csv'
            changes, missing_csvs, missing_files = process_csv(
                csv_path=csv_path,
                task_root=task_root,
                aliases=TASK_ALIASES[task],
                write=args.write,
            )
            total_changes += changes
            total_missing_csvs += missing_csvs
            total_missing_files += missing_files

    print(
        f'Summary: {total_changes} path update(s), '
        f'{total_missing_csvs} missing CSV(s), {total_missing_files} missing data file reference(s).'
    )

    if total_missing_csvs or total_missing_files:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
