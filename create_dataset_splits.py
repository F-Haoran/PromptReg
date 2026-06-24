#!/usr/bin/env python3
import argparse
import csv
import os
import random
import re
from pathlib import Path


DEFAULT_DATA_ROOT = '/home/FrankFei/PromptReg'
DEFAULT_TASKS = ('Abdominal', 'Brain', 'Cardiac', 'Hippocampus', 'Hip')
REQUIRED_COLUMNS = ('moving_image', 'moving_label', 'fixed_image', 'fixed_label')
SUPPORTED_EXTENSIONS = ('.nii', '.nii.gz', '.mha', '.mhd')
IGNORED_DIR_NAMES = {
    'csv',
    'checkpoint',
    'checkpoints',
    'output',
    'outputs',
    'denoise',
    'denoised',
}
LABEL_HINTS = ('label', 'labels', 'seg', 'segs', 'segmentation', 'mask', 'masks')
IMAGE_SUFFIXES = ('_image', '-image', '_img', '-img')
LABEL_SUFFIXES = (
    '_label',
    '-label',
    '_labels',
    '-labels',
    '_seg',
    '-seg',
    '_segs',
    '-segs',
    '_mask',
    '-mask',
    '_masks',
    '-masks',
    '_segmentation',
    '-segmentation',
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create PromptReg csv/train.csv and csv/test.csv files with an approximate test split.'
    )
    parser.add_argument('--data-root', default=os.environ.get('PROMPTREG_DATA_ROOT', DEFAULT_DATA_ROOT),
                        help='Root directory containing PromptReg task folders')
    parser.add_argument('--tasks', nargs='+', default=list(DEFAULT_TASKS),
                        help='Task folders to process')
    parser.add_argument('--test-fraction', type=float, default=0.3,
                        help='Fraction of rows/cases assigned to test')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible splits')
    parser.add_argument('--mode', choices=('auto', 'existing-csv', 'discover'), default='auto',
                        help='Split existing pair CSVs, discover image/label files, or choose automatically')
    parser.add_argument('--pairing', choices=('consecutive', 'fixed-first', 'all'), default='consecutive',
                        help='How discovered cases are converted into moving/fixed pairs')
    parser.add_argument('--write', action='store_true',
                        help='Write CSV files; default only prints what would happen')
    parser.add_argument('--overwrite', action='store_true',
                        help='Allow replacing existing train.csv/test.csv')
    return parser.parse_args()


def validate_fraction(test_fraction):
    if not 0 < test_fraction < 1:
        raise ValueError('--test-fraction must be between 0 and 1')


def has_supported_extension(path):
    name = path.name.lower()
    return any(name.endswith(extension) for extension in SUPPORTED_EXTENSIONS)


def strip_supported_extension(name):
    lower_name = name.lower()
    for extension in sorted(SUPPORTED_EXTENSIONS, key=len, reverse=True):
        if lower_name.endswith(extension):
            return name[:-len(extension)]
    return Path(name).stem


def normalized_case_key(path):
    stem = strip_supported_extension(path.name).lower()
    for suffix in LABEL_SUFFIXES + IMAGE_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
            break
    stem = re.sub(r'^(image|img|label|labels|seg|mask)[_-]+', '', stem)
    return stem


def is_ignored_path(path, task_root):
    relative_parts = path.relative_to(task_root).parts
    return any(part.lower() in IGNORED_DIR_NAMES for part in relative_parts[:-1])


def is_label_path(path):
    parts = [part.lower() for part in path.parts]
    name = path.name.lower()
    return any(hint in part for part in parts for hint in LABEL_HINTS) or any(hint in name for hint in LABEL_HINTS)


def relative_posix(path, root):
    return path.relative_to(root).as_posix()


def discover_cases(task_root):
    images = {}
    labels = {}
    for path in sorted(task_root.rglob('*')):
        if not path.is_file() or not has_supported_extension(path) or is_ignored_path(path, task_root):
            continue

        key = normalized_case_key(path)
        if is_label_path(path):
            labels.setdefault(key, path)
        else:
            images.setdefault(key, path)

    cases = []
    for key in sorted(set(images) & set(labels)):
        cases.append({
            'key': key,
            'image': relative_posix(images[key], task_root),
            'label': relative_posix(labels[key], task_root),
        })

    unmatched_images = sorted(set(images) - set(labels))
    unmatched_labels = sorted(set(labels) - set(images))
    return cases, unmatched_images, unmatched_labels


def split_items(items, test_fraction, seed):
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)

    if len(shuffled) <= 1:
        return shuffled, []

    test_count = max(1, round(len(shuffled) * test_fraction))
    test_count = min(test_count, len(shuffled) - 1)
    test_items = shuffled[:test_count]
    train_items = shuffled[test_count:]
    return train_items, test_items


def pairs_from_cases(cases, pairing):
    if not cases:
        return []
    if len(cases) == 1:
        case = cases[0]
        return [pair_row(case, case)]

    if pairing == 'consecutive':
        return [pair_row(case, cases[(index + 1) % len(cases)]) for index, case in enumerate(cases)]

    if pairing == 'fixed-first':
        fixed_case = cases[0]
        return [pair_row(case, fixed_case) for case in cases[1:]]

    if pairing == 'all':
        return [pair_row(moving, fixed) for moving in cases for fixed in cases if moving != fixed]

    raise ValueError(f'Unsupported pairing mode: {pairing}')


def pair_row(moving, fixed):
    return {
        'moving_image': moving['image'],
        'moving_label': moving['label'],
        'fixed_image': fixed['image'],
        'fixed_label': fixed['label'],
    }


def find_source_csv(task_root):
    csv_dir = task_root / 'csv'
    candidates = (
        csv_dir / 'all.csv',
        csv_dir / 'pairs.csv',
        csv_dir / 'dataset.csv',
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate

    train_csv = csv_dir / 'train.csv'
    test_csv = csv_dir / 'test.csv'
    if train_csv.exists() and not test_csv.exists():
        return train_csv

    return None


def read_pair_csv(csv_path):
    with csv_path.open(newline='') as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None:
            raise ValueError(f'{csv_path} is missing a header row')
        missing_columns = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
        if missing_columns:
            raise ValueError(f'{csv_path} is missing required columns: {missing_columns}')
        return [{column: row[column] for column in REQUIRED_COLUMNS} for row in reader]


def write_pair_csv(csv_path, rows, overwrite):
    if csv_path.exists() and not overwrite:
        raise FileExistsError(f'{csv_path} already exists; pass --overwrite to replace it')

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open('w', newline='') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def maybe_write_split(task_root, train_rows, test_rows, args):
    train_csv = task_root / 'csv' / 'train.csv'
    test_csv = task_root / 'csv' / 'test.csv'

    if args.write:
        write_pair_csv(train_csv, train_rows, overwrite=args.overwrite)
        write_pair_csv(test_csv, test_rows, overwrite=args.overwrite)
        print(f'[written] {train_csv}: {len(train_rows)} row(s)')
        print(f'[written] {test_csv}: {len(test_rows)} row(s)')
    else:
        print(f'[dry run] {train_csv}: {len(train_rows)} row(s)')
        print(f'[dry run] {test_csv}: {len(test_rows)} row(s)')


def process_existing_csv(task_root, source_csv, args, seed):
    rows = read_pair_csv(source_csv)
    train_rows, test_rows = split_items(rows, args.test_fraction, seed)
    print(f'[source csv] {source_csv}: {len(rows)} row(s)')
    maybe_write_split(task_root, train_rows, test_rows, args)


def process_discovered_cases(task_root, args, seed):
    cases, unmatched_images, unmatched_labels = discover_cases(task_root)
    print(f'[discover] {task_root}: {len(cases)} matched case(s)')
    if unmatched_images:
        print(f'[warning] {task_root}: {len(unmatched_images)} image(s) without matched labels')
    if unmatched_labels:
        print(f'[warning] {task_root}: {len(unmatched_labels)} label(s) without matched images')
    if not cases:
        raise ValueError(f'No matched image/label cases found in {task_root}')

    train_cases, test_cases = split_items(cases, args.test_fraction, seed)
    train_rows = pairs_from_cases(train_cases, args.pairing)
    test_rows = pairs_from_cases(test_cases, args.pairing)
    print(f'[split] {task_root}: {len(train_cases)} train case(s), {len(test_cases)} test case(s)')
    maybe_write_split(task_root, train_rows, test_rows, args)


def process_task(data_root, task, args, task_index):
    task_root = data_root / task
    if not task_root.is_dir():
        print(f'[missing task] {task_root}')
        return

    seed = args.seed + task_index
    source_csv = find_source_csv(task_root)
    if args.mode in ('auto', 'existing-csv') and source_csv is not None:
        process_existing_csv(task_root, source_csv, args, seed)
        return

    if args.mode == 'existing-csv':
        raise FileNotFoundError(f'No source CSV found under {task_root / "csv"}')

    process_discovered_cases(task_root, args, seed)


def main():
    args = parse_args()
    validate_fraction(args.test_fraction)
    data_root = Path(os.path.abspath(os.path.expanduser(args.data_root)))

    for task_index, task in enumerate(args.tasks):
        process_task(data_root, task, args, task_index)


if __name__ == '__main__':
    main()
