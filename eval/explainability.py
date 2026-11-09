"""Attention maps of the [Lesions] token (Supplementary figure).

For every volume of a dataset, the central B-scan is passed through the B-scan encoder (stage 1) and the attention of
the [Lesions] global token to the patch tokens, in the last encoder block and averaged over heads, is overlaid on the
B-scan.

Usage (from eval/): PYTHONPATH=.. python explainability.py --weights _weights/CLIFO.pth
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt

from multimae.parts.multimae_utils import Attention
from run_pretraining_vfe import load_vm
from paths import DATASETS, GEN_RESULTS


MAX_SLICES = 256


class AttentionWithMaps(Attention):
    """Attention that can also store its attention weights.

    F.scaled_dot_product_attention does not return the weights, so they are computed explicitly when `return_attn` is
    set. They are stored in `last_attn`, with shape [B, num_heads, N, N].
    """
    return_attn = False

    def forward(self, x):
        if not self.return_attn:
            return super().forward(x)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        self.last_attn = attn.detach()
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(out))


def enable_attention_maps(model_vm):
    """Lets every attention layer of the B-scan encoder store its weights on demand."""
    for block in model_vm.encoder:
        block.attn.__class__ = AttentionWithMaps


def volume_to_tensor(volume, input_size=512):
    """Converts a volume (numpy array [NUM_SLICES, H, W]) to a tensor [1, 1, S, input_size, input_size] in [0, 1]."""
    x = torch.from_numpy(volume).float()
    if x.max() > 255:
        x = x / 65535.0
    elif x.max() > 1.0:
        x = x / 255.0
    x = x.unsqueeze(0).unsqueeze(0)  # batch and channel dimensions
    slices = min(x.shape[2], MAX_SLICES)
    if x.shape[2:] != torch.Size([slices, input_size, input_size]):
        x = F.interpolate(x, size=(slices, input_size, input_size), mode='trilinear', align_corners=False)
    return x


def get_lesion_attention_map(model_vm, bscan_single, lesion_token_idx=4, layer_idx=-1, grid_size=(16, 16)):
    """
    Runs the B-scan encoder on a single B-scan, capturing the [Lesions]
    token's attention to patch tokens from a specific encoder layer.

    Args:
        model_vm: the B-scan encoder (see `load_vm`), prepared with `enable_attention_maps`
        bscan_single: [1, 1, H, W] tensor, a single B-scan
        lesion_token_idx: index of the [Lesions] token WITHIN the global
            tokens block (e.g. 4, matching PredictionHead(..., index=4))
        layer_idx: which encoder block to extract attention from
        grid_size: (N_H, N_W) patch grid size, for reshaping the map
    Returns:
        attn_map: [N_H, N_W] numpy array, attention from [Lesions]
            token to each patch, averaged over heads
    """
    target_block = model_vm.encoder[layer_idx]
    target_block.attn.return_attn = True

    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        _ = model_vm({'bscan': bscan_single}, mask_inputs=False, force_no_shuffle=True)

    attn = target_block.attn.last_attn.float()  # cast back to float32 for downstream numpy ops
    target_block.attn.return_attn = False

    num_patches = grid_size[0] * grid_size[1]
    global_lesion_idx = num_patches + lesion_token_idx

    lesion_to_patch = attn[0, :, global_lesion_idx, :num_patches]
    lesion_to_patch = lesion_to_patch.mean(dim=0)

    attn_map = lesion_to_patch.reshape(grid_size).cpu().numpy()
    return attn_map


def plot_attention_overlay(bscan_img, attn_map, save_path=None):
    H, W = bscan_img.shape
    attn_upsampled = F.interpolate(
        torch.tensor(attn_map)[None, None], size=(H, W), mode='bilinear', align_corners=False
    )[0, 0].numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    plt.subplots_adjust(wspace=0.02)  # tight gap between the two panels

    axes[0].imshow(bscan_img, cmap='gray')
    axes[0].set_title('B-scan')
    axes[0].axis('off')

    axes[1].imshow(bscan_img, cmap='gray')
    axes[1].imshow(attn_upsampled, cmap='jet', alpha=0.5)
    axes[1].set_title('[Lesions] token attention')
    axes[1].axis('off')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
        print(f'Saved to {save_path}')
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='_weights/CLIFO.pth', help='B-scan encoder (stage 1) weights.')
    parser.add_argument('--dataset', type=str, default='OCTAVE', help='3D classification dataset.')
    parser.add_argument('--out_dir', type=Path, default=GEN_RESULTS / 'explainability')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    model_vm = load_vm(args.device, weights=args.weights).eval()
    enable_attention_maps(model_vm)

    dataset_root = DATASETS / 'Classification_3D' / args.dataset
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for subset in ['train', 'val', 'test']:
        subset_path = dataset_root / subset
        if not subset_path.exists():
            continue
        for class_dir in sorted(subset_path.iterdir()):
            if not class_dir.is_dir():
                continue
            for file in sorted(class_dir.iterdir()):
                if file.suffix != '.npz':
                    continue

                volume = np.load(file)['vol']
                volume_tensor = volume_to_tensor(volume)

                mid_slice_idx = volume_tensor.shape[2] // 2
                bscan_single = volume_tensor[:, :, mid_slice_idx, :, :].to(args.device)
                bscan_img = volume_tensor[0, 0, mid_slice_idx].cpu().numpy()

                attn_map = get_lesion_attention_map(
                    model_vm, bscan_single, lesion_token_idx=4, layer_idx=-1, grid_size=(16, 16),
                )

                fn = args.out_dir / f'{subset}_{class_dir.name}_{file.stem}.pdf'
                plot_attention_overlay(bscan_img, attn_map, save_path=fn)
