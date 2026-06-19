# PromptReg

## PromptReg: Universal Medical Image Registration via Task Prompt Learning and Domain Knowledge Transfer

Housheng Xie, Xiaoru Gao, Guoyan Zheng

## 📖 Introduction

In this work, we propose **PromptReg**, a universal image registration framework that incorporates prompt learning to guide the model in effectively adapting to different registration scenarios through explicit task prompts.

More details can be found in our [paper](https://link.springer.com/chapter/10.1007/978-3-032-04971-1_47).


## 📁 Repository Structure
- `train.py` — Main training script (set `data_root` and `save_dir`).
- `dataset.py` — Multi-task dataset wrapper and task-id remapping.
- `subdataset.py` — Per-task loading, normalization, label checks.
- `PromptReg.py` — PromptReg Model.

## 🧰 Data Preparation
Set `data_root` in `train.py`. Each task folder needs `csv/train.csv` and `csv/test.csv` with columns:
- `moving_image`, `moving_label`
- `fixed_image`, `fixed_label`

You can generate these CSV files from an extracted dataset folder or a tar archive:

```
python prepare_dataset_csv.py /path/to/dataset --overwrite
```

For a tar archive:

```
python prepare_dataset_csv.py /path/to/dataset.tar.gz --extract-to /path/to/dataset --overwrite
```

If your Brain dataset folder contains multiple tar archives, extract all of them into the same folder first:

```
python3 extract_brain_tars.py
```

This defaults to `/home/frankfei/PromptReg/Dataset/Brain`. To use another folder:

```
python3 extract_brain_tars.py --folder /path/to/Brain
```

The script first looks for task folders such as `Abdominal`, `Brain`, `Cardiac`, `Hippocampus`, and `Hip`.
If those names are not present, it can infer task folders from any immediate subdirectory that contains
NIfTI-like volumes. It treats paths containing words such as `label`, `seg`, `mask`, or `gt` as labels,
matches each image/label case by filename stem, and writes relative paths into each task's CSV files.

If your dataset root itself is a single task with folders like `images/` and `labels/`, provide the task name:

```
python prepare_dataset_csv.py /path/to/dataset --task-name Abdominal --overwrite
```

If your filenames use a special subject ID pattern, pass `--subject-regex`, for example:

```
python prepare_dataset_csv.py /path/to/dataset --subject-regex "case_([0-9]+)" --overwrite
```

If you see warnings such as `images without labels` for names like `img0001_tcia` and
`labels without images` for names like `mask0001_tcia`, update to the latest script. It normalizes
embedded role prefixes (`img0001`, `mask0001`, `label0001`, etc.) to the same case key.

After CSV generation, make sure `train.py`/`dataset.py` use the same task folder names that exist under
`data_root`.

## 🏋️ Training
1) Edit `train.py` to set:
- `data_root`: dataset root directory
- `save_dir`: directory to store checkpoints (auto-save every 30 epochs)

2) Run:
```
python train.py \
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
