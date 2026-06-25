#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
from medpy.io import load, save
from scipy.ndimage import gaussian_filter, median_filter


SUPPORTED_EXTENSIONS = ('.nii', '.nii.gz', '.mha', '.mhd')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Denoise medical images without a fixed/reference image.'
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', help='Noisy input image path')
    input_group.add_argument('--input-folder', help='Folder containing noisy images')
    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument('--output', help='Output denoised image path')
    output_group.add_argument('--output-folder', help='Folder for denoised images')
    parser.add_argument('--method', choices=('gaussian', 'median'), default='gaussian',
                        help='Denoising method to apply')
    parser.add_argument('--sigma', type=float, default=1.0,
                        help='Gaussian sigma when --method gaussian')
    parser.add_argument('--size', type=int, default=3,
                        help='Median filter window size when --method median')
    parser.add_argument('--clip-percentiles', nargs=2, type=float, metavar=('LOW', 'HIGH'),
                        help='Optionally clip intensities before denoising, for example 1 99')
    parser.add_argument('--preserve-dtype', action='store_true',
                        help='Cast output back to the input dtype before saving')
    parser.add_argument('--recursive', action='store_true',
                        help='Process input folders recursively')
    return parser.parse_args()


def clip_percentiles(data, percentiles):
    low, high = percentiles
    if low >= high:
        raise ValueError('LOW percentile must be smaller than HIGH percentile')

    finite_values = data[np.isfinite(data)]
    if finite_values.size == 0:
        raise ValueError('Input image contains no finite values')

    low_value, high_value = np.percentile(finite_values, (low, high))
    return np.clip(data, low_value, high_value)


def cast_like_input(data, input_dtype):
    if np.issubdtype(input_dtype, np.integer):
        info = np.iinfo(input_dtype)
        data = np.clip(np.rint(data), info.min, info.max)
    return data.astype(input_dtype, copy=False)


def denoise_array(data, method='gaussian', sigma=1.0, size=3, clip_percentiles_value=None):
    denoised = data.astype(np.float32, copy=False)

    if clip_percentiles_value is not None:
        denoised = clip_percentiles(denoised, clip_percentiles_value)

    if method == 'gaussian':
        return gaussian_filter(denoised, sigma=sigma)
    if method == 'median':
        return median_filter(denoised, size=size)

    raise ValueError(f'Unsupported denoising method: {method}')


def has_supported_extension(path):
    path_name = path.name.lower()
    return any(path_name.endswith(extension) for extension in SUPPORTED_EXTENSIONS)


def iter_input_files(input_folder, recursive=False):
    pattern = '**/*' if recursive else '*'
    for path in sorted(input_folder.glob(pattern)):
        if path.is_file() and has_supported_extension(path):
            yield path


def output_path_for(input_path, input_folder, output_folder):
    relative_path = input_path.relative_to(input_folder)
    return output_folder / relative_path


def denoise_file(input_path, output_path, args):
    data, header = load(str(input_path))
    input_dtype = data.dtype
    denoised = denoise_array(
        data=data,
        method=args.method,
        sigma=args.sigma,
        size=args.size,
        clip_percentiles_value=args.clip_percentiles,
    )

    if args.preserve_dtype:
        denoised = cast_like_input(denoised, input_dtype)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save(denoised, str(output_path), header, force=True)
    print(f'Saved denoised image to {output_path}')


def validate_args(args):
    if args.sigma < 0:
        raise ValueError('--sigma must be non-negative')
    if args.size < 1:
        raise ValueError('--size must be at least 1')
    if args.input_folder and args.output:
        raise ValueError('--output cannot be used with --input-folder; use --output-folder')
    if args.input:
        input_path = Path(args.input)
        if not input_path.is_file():
            raise ValueError(f'Input file does not exist: {input_path}')
        if not has_supported_extension(input_path):
            raise ValueError(f'Unsupported input file extension: {input_path}')
    if args.input_folder:
        input_folder = Path(args.input_folder)
        if not input_folder.is_dir():
            raise ValueError(f'Input folder does not exist: {input_folder}')


def main():
    args = parse_args()
    validate_args(args)

    if args.input:
        input_path = Path(args.input)
        output_path = Path(args.output) if args.output else Path(args.output_folder) / input_path.name
        denoise_file(input_path, output_path, args)
        return

    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)
    input_files = list(iter_input_files(input_folder, recursive=args.recursive))
    if not input_files:
        raise ValueError(f'No supported image files found in {input_folder}')

    for input_path in input_files:
        denoise_file(
            input_path=input_path,
            output_path=output_path_for(input_path, input_folder, output_folder),
            args=args,
        )

    print(f'Processed {len(input_files)} image(s).')


if __name__ == '__main__':
    main()
