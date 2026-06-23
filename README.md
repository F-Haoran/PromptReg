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
