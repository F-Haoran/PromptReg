#!/usr/bin/env python3
import argparse

import numpy as np
from medpy.io import load, save
from scipy.ndimage import gaussian_filter, median_filter


def parse_args():
    parser = argparse.ArgumentParser(
        description='Denoise a single NIfTI image without a fixed/reference image.'
    )
    parser.add_argument('--input', required=True, help='Noisy input NIfTI path')
    parser.add_argument('--output', required=True, help='Output denoised NIfTI path')
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


def main():
    args = parse_args()

    data, header = load(args.input)
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

    save(denoised, args.output, header, force=True)
    print(f'Saved denoised image to {args.output}')


if __name__ == '__main__':
    main()
