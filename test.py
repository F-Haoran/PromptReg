import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from PromptReg import PromptReg
from subdataset import AbdominalDataset, BrainDataset, CardiacDataset, HippocampusDataset, HipDataset
from train import Grad3d, MIND_loss, SpatialTransformer


DEFAULT_DATA_ROOT = '/home/FrankFei/PromptReg'
DEFAULT_TARGET_SIZE = (160, 160, 160)

TASK_CONFIG = {
    'Abdominal': {'dataset_cls': AbdominalDataset, 'task_id': 0, 'num_classes': 5},
    'Brain': {'dataset_cls': BrainDataset, 'task_id': 1, 'num_classes': 36},
    'Hippocampus': {'dataset_cls': HippocampusDataset, 'task_id': 3, 'num_classes': 3},
    'Cardiac': {'dataset_cls': CardiacDataset, 'task_id': 4, 'num_classes': 4},
    'Hip': {'dataset_cls': HipDataset, 'task_id': 5, 'num_classes': 4},
}


class TestTaskDataset(Dataset):
    def __init__(self, data_root, task_name, target_size, task_id):
        config = TASK_CONFIG[task_name]
        self.task_name = task_name
        self.task_id = task_id
        self.target_size = tuple(target_size)
        self.dataset = config['dataset_cls'](os.path.join(data_root, task_name), split='test')

    def _resample(self, image, label):
        image = F.interpolate(
            image.unsqueeze(0),
            size=self.target_size,
            mode='trilinear',
            align_corners=True,
        ).squeeze(0)
        label = F.interpolate(
            label.unsqueeze(0).unsqueeze(0).float(),
            size=self.target_size,
            mode='nearest',
        ).squeeze(0).squeeze(0).long()
        return image, label

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        moving, moving_label = self._resample(sample['moving'], sample['moving_label'])
        fixed, fixed_label = self._resample(sample['fixed'], sample['fixed_label'])
        return {
            'moving': moving,
            'moving_label': moving_label,
            'fixed': fixed,
            'fixed_label': fixed_label,
            'task_id': self.task_id,
            'task_name': self.task_name,
            'moving_path': sample['moving_path'],
            'fixed_path': sample['fixed_path'],
        }

    def __len__(self):
        return len(self.dataset)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate PromptReg on csv/test.csv splits')
    parser.add_argument('--data-root', default=os.environ.get('PROMPTREG_DATA_ROOT', DEFAULT_DATA_ROOT))
    parser.add_argument('--checkpoint', required=True, help='Path to a saved .pth checkpoint')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--tasks', nargs='+', default=list(TASK_CONFIG.keys()))
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--target-size', nargs=3, type=int, default=None,
                        help='Input volume size H W D. Default: infer from checkpoint')
    parser.add_argument('--task-total-number', type=int, default=None,
                        help='Number of task prompts in the model. Default: infer from checkpoint')
    parser.add_argument('--exclude-tasks', nargs='+', default=['Abdominal'],
                        help='Tasks excluded during training; used to remap task_id for testing')
    return parser.parse_args()


def load_checkpoint(checkpoint_path, device):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint


def infer_model_config(state_dict):
    static_prompt = state_dict['prompt_generator.static_prompt']
    task_total_number = int(static_prompt.shape[0])
    prompt_spatial = static_prompt.shape[2:]
    inshape = tuple(int(size * 8) for size in prompt_spatial)
    return inshape, task_total_number


def build_task_id_map(exclude_tasks):
    task_map = {}
    for task_name, config in TASK_CONFIG.items():
        if task_name in exclude_tasks:
            continue
        task_map[task_name] = config['task_id']

    remapped = {}
    for new_id, (task_name, _) in enumerate(sorted(task_map.items(), key=lambda item: item[1])):
        remapped[task_name] = new_id
    return remapped


def resolve_model_config(args, state_dict):
    inferred_inshape, inferred_task_total_number = infer_model_config(state_dict)
    target_size = tuple(args.target_size) if args.target_size is not None else inferred_inshape
    task_total_number = (
        args.task_total_number if args.task_total_number is not None else inferred_task_total_number
    )
    return target_size, task_total_number


def dice_score(pred, target, num_classes):
    dice_values = []
    for class_id in range(1, num_classes):
        pred_mask = pred == class_id
        target_mask = target == class_id
        intersection = (pred_mask & target_mask).sum().item()
        union = pred_mask.sum().item() + target_mask.sum().item()
        if union > 0:
            dice_values.append((2.0 * intersection) / union)
    if not dice_values:
        return 0.0
    return float(np.mean(dice_values))


def evaluate_task(model, label_warper, data_root, task_name, task_id, target_size, device, mind_loss, grad_loss):
    dataset = TestTaskDataset(
        data_root=data_root,
        task_name=task_name,
        target_size=target_size,
        task_id=task_id,
    )
    if len(dataset) == 0:
        print(f'[skip] {task_name}: csv/test.csv is empty or missing')
        return None

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    num_classes = TASK_CONFIG[task_name]['num_classes']

    total_mind = 0.0
    total_smooth = 0.0
    total_dice = 0.0
    count = 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=f'Testing {task_name}'):
            moving = batch['moving'].to(device)
            fixed = batch['fixed'].to(device)
            moving_label = batch['moving_label'].to(device)
            fixed_label = batch['fixed_label'].to(device)
            task_tensor = torch.tensor([task_id], device=device)

            warped, flow, _, _ = model(moving, fixed, task_tensor, is_training=False)
            warped_label = label_warper(moving_label.unsqueeze(1).float(), flow).squeeze(1).long()

            total_mind += mind_loss(warped, fixed).item()
            total_smooth += grad_loss(flow).item()
            total_dice += dice_score(
                warped_label[0].cpu(),
                fixed_label[0].cpu(),
                num_classes=num_classes,
            )
            count += 1

    return {
        'count': count,
        'mind': total_mind / count,
        'smooth': total_smooth / count,
        'dice': total_dice / count,
    }


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = load_checkpoint(args.checkpoint, device)
    target_size, task_total_number = resolve_model_config(args, checkpoint['model_state_dict'])
    task_id_map = build_task_id_map(args.exclude_tasks)

    model = PromptReg(inshape=target_size, task_total_number=task_total_number).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    mind_loss = MIND_loss(device)
    grad_loss = Grad3d(penalty='l2')
    label_warper = SpatialTransformer(target_size, mode='nearest').to(device)

    data_root = os.path.abspath(os.path.expanduser(args.data_root))
    print(f'Loaded checkpoint epoch {checkpoint.get("epoch", "unknown")} from {args.checkpoint}')
    print(f'Using target_size={target_size}, task_total_number={task_total_number}')
    print(f'Task id map: {task_id_map}')

    for task_name in args.tasks:
        if task_name not in TASK_CONFIG:
            raise ValueError(f'Unknown task {task_name}. Expected one of: {list(TASK_CONFIG)}')
        if task_name in args.exclude_tasks:
            print(f'[skip] {task_name}: excluded during training')
            continue
        if task_name not in task_id_map:
            print(f'[skip] {task_name}: no task id mapping available')
            continue

        result = evaluate_task(
            model=model,
            label_warper=label_warper,
            data_root=data_root,
            task_name=task_name,
            task_id=task_id_map[task_name],
            target_size=target_size,
            device=device,
            mind_loss=mind_loss,
            grad_loss=grad_loss,
        )
        if result is not None:
            print(
                f'{task_name}: '
                f'count={result["count"]}, '
                f'mind={result["mind"]:.6f}, '
                f'smooth={result["smooth"]:.6f}, '
                f'dice={result["dice"]:.6f}'
            )


if __name__ == '__main__':
    main()
