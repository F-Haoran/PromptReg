# PromptReg

## PromptReg: Universal Medical Image Registration via Task Prompt Learning and Domain Knowledge Transfer

Housheng Xie, Xiaoru Gao, Guoyan Zheng

## 📖 Introduction

In this work, we propose **PromptReg**, a universal image registration framework that incorporates prompt learning to guide the model in effectively adapting to different registration scenarios through explicit task prompts.

More details can be found in our [paper](https://link.springer.com/chapter/10.1007/978-3-032-04971-1_47).


## 📁 Repository Structure
- `train.py` — Main training script (defaults to `/home/FrankFei/PromptReg` for `data_root`).
- `dataset.py` — Multi-task dataset wrapper and task-id remapping.
- `subdataset.py` — Per-task loading, normalization, label checks.
- `prepare_csv_paths.py` — CSV path checker/rewriter for moved data roots.
- `create_dataset_splits.py` — Generate `csv/train.csv` and `csv/test.csv` with a test split.
- `denoise_nifti.py` — File/folder medical image denoising without a fixed/reference image.
- `PromptReg.py` — PromptReg Model.

## 🧰 Data Preparation
The default `data_root` is `/home/FrankFei/PromptReg`. Override it with `--data-root` or the
`PROMPTREG_DATA_ROOT` environment variable if needed. Each task folder needs `csv/train.csv`
and `csv/test.csv` with columns:
- `moving_image`, `moving_label`
- `fixed_image`, `fixed_label`

Expected task folders:
```
/home/FrankFei/PromptReg/
  Abdominal/
  Brain/
  Cardiac/
  Hippocampus/
  Hip/
```

CSV path values should be relative to their task folder. If existing CSVs still contain old
absolute paths, check them with:
```
python prepare_csv_paths.py --data-root /home/FrankFei/PromptReg
```

Rewrite old absolute entries to relative paths after reviewing the dry-run output:
```
python prepare_csv_paths.py --data-root /home/FrankFei/PromptReg --write
```

If the task folders do not already have separate `csv/train.csv` and `csv/test.csv` files,
generate them with an approximate 70/30 split:
```
python create_dataset_splits.py \
  --data-root /home/FrankFei/PromptReg \
  --test-fraction 0.3
```

After checking the dry-run output, write the CSVs:
```
python create_dataset_splits.py \
  --data-root /home/FrankFei/PromptReg \
  --test-fraction 0.3 \
  --write \
  --overwrite
```

The split script first tries to split an existing pair CSV (`csv/all.csv`, `csv/pairs.csv`,
`csv/dataset.csv`, or a lone `csv/train.csv`). If no source CSV exists, it scans each task
folder for image/label files (`.nii`, `.nii.gz`, `.mha`, `.mhd`) and creates moving/fixed pairs
from matched cases. Image and label files must have matching case names, such as
`images/case001.nii.gz` and `labels/case001_label.nii.gz`.

## ⚙️ Environment Setup
On Ubuntu with Python 3.12:
```
sudo apt-get update
sudo apt-get install -y python3.12-venv
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

Run commands through the virtual environment:
```
.venv/bin/python train.py --help
```

## 🧹 Image Denoising
PromptReg checkpoints are registration weights: inference requires both a moving image and a
fixed/reference image. If you only have one noisy image and an output path, use the standalone
filtering script instead. Supported file types include `.nii`, `.nii.gz`, `.mha`, and `.mhd`.

Single file:
```
python denoise_nifti.py \
  --input /path/to/noisy_image.nii.gz \
  --output /path/to/denoised_image.nii.gz \
  --method gaussian \
  --sigma 1.0
```

Folder input/output:
```
python denoise_nifti.py \
  --input-folder /path/to/noisy_images \
  --output-folder /path/to/denoised_images \
  --method gaussian \
  --sigma 1.0
```

Add `--recursive` to process nested folders while preserving their relative paths:
```
python denoise_nifti.py \
  --input-folder /path/to/noisy_images \
  --output-folder /path/to/denoised_images \
  --recursive
```

For salt-and-pepper style noise, median filtering can be useful:
```
python denoise_nifti.py \
  --input /path/to/noisy_image.nii.gz \
  --output /path/to/denoised_image.nii.gz \
  --method median \
  --size 3
```

## 🏋️ Training
1) Set paths if you do not want the defaults:
- `--data-root`: dataset root directory (default: `/home/FrankFei/PromptReg`)
- `--save-dir`: directory to store checkpoints (default: `<data-root>/checkpoints`)

2) Run:
```
python train.py \
  --data-root /home/FrankFei/PromptReg \
  --gpu 0 \
  --batch-size 1 \
  --epochs 300 \
  --lr 1e-4 \
  --exclude-tasks Abdominal  # pass multiple to remove more tasks during training
```

## 📜 Citation

If you are interested in this work, please cite the following work:

```
@inproceedings{xie2025promptreg,
  title={PromptReg: Universal Medical Image Registration via Task Prompt Learning and Domain Knowledge Transfer},
  author={Xie, Housheng and Gao, Xiaoru and Zheng, Guoyan},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={498--507},
  year={2025},
  organization={Springer}
}
```

## 🙏 Acknowledgments

Our work is based on [RDP](https://github.com/ZAX130/RDP) and we use their code in the model. We are very grateful for their contributions.
