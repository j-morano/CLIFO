from argparse import Namespace
from pathlib import Path
import sys

import torch
from torch import nn

from mutils.factory import get_factory_adder
from mutils.pos_embed import interpolate_pos_embed


# Import the model architecture from the parent directory

parent_dir = str(Path(__file__).resolve().parent.parent)
print(parent_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from utils.architecture import get_model_architecture  # type: ignore


add_pooling, pooling_factory = get_factory_adder()


@add_pooling('patch')
class BScanEncoderPatchPooling(nn.Module):
    def __init__(
        self,
        input_size=512,
        patch_size=32,
        num_classes=0,
        modalities='bscan',
        weights=None,
        device='cuda',
        model_size='base',
        num_global_tokens=10,
        chd=False,
    ):
        super().__init__()
        assert num_classes > 0

        grid_size = input_size // patch_size

        self.chd = chd
        self.args = Namespace(
            in_domains=["bscan"],
            out_domains=[],  # NOTE: No output, just features
            extra_norm_pix_loss=False,
            model="pretrain_multimae_base",  # only encoder
            decoder_dim=256,
            input_size={modalities: [input_size, input_size]},
            patch_size={modalities: [patch_size, patch_size]},
            grid_sizes={modalities: [grid_size, grid_size]},
            sample_only=modalities,
            alphas=1.0,
            num_encoded_tokens=64,
            num_global_tokens=num_global_tokens,
            decoder_use_task_queries=True,
            decoder_depth=2,
            weights=weights,
            decoder_num_heads=8,
            decoder_use_xattn=True,
            drop_path=0.0,
            custom_sampling=False,
        )
        if model_size == 'large':
             self.args.model = "pretrain_multimae_large"

        vm = get_model_architecture(self.args)
        vm.force_no_shuffle=True

        checkpoint = torch.load(self.args.weights, map_location=device, weights_only=False)["model"]
        try:
            msg = vm.load_state_dict(checkpoint, strict=False)
        except RuntimeError:
            print("Interpolating position embedding...")
            interpolate_pos_embed(vm, checkpoint)
            msg = vm.load_state_dict(checkpoint, strict=False)
        print('Missing keys:', msg.missing_keys)
        assert len(msg.missing_keys) == 0, "THERE ARE MISSING KEYS IN WEIGHTS!"
        vm.to(device)

        self.model = vm

        self.num_classes = num_classes
        # Get the embedding dimension from the first layer norm of the
        #   type: (norm1): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        self.embed_dim = self.model.encoder[0].norm1.normalized_shape[0]
        self.norm = nn.LayerNorm(self.embed_dim, eps=1e-06, elementwise_affine=True)
        print('  Embedding dimension:', self.embed_dim)
        self.build_head()

    def build_head(self, factor=1):
        self.head = nn.Linear(self.embed_dim * factor, self.num_classes)

    def forward_chd(self, x, return_features=False):
        # x in this case is an RGB image where 0 is the first image, 1
        #   is the second image.
        x_1 = x[:, 0:1, :, :]
        x_2 = x[:, 1:2, :, :]
        x_1_d = {self.args.in_domains[0]: x_1}
        x_2_d = {self.args.in_domains[0]: x_2}
        out_1, _masks_1 = self.model(x_1_d, mask_inputs=False)
        out_2, _masks_2 = self.model(x_2_d, mask_inputs=False)
        out_1 = self.norm(out_1)
        out_2 = self.norm(out_2)
        out_1 = self.pool(out_1)
        out_2 = self.pool(out_2)
        out = torch.cat([out_1, out_2], dim=1)
        if return_features:
            return out
        return self.head(out)

    def forward(self, x, return_features=False):
        """
        Args:
            x: (B, C, H, W) tensor. H and W are determined by the
            input_size parameter in the constructor. It expects a tensor
            in the range [0, 1].
        Returns:
            (B, C, H, W) tensor
        """
        if self.chd:
            return self.forward_chd(x, return_features=return_features)
        if not isinstance(x, dict):
            x_d = {self.args.in_domains[0]: x}
        else:
            x_d = x
        out, _masks = self.model(x_d, mask_inputs=False, force_no_shuffle=True)
        out = self.norm(out)
        out = self.pool(out)
        if return_features:
            return out
        return self.head(out)

    def pool(self, x):
        return x[:, :-self.args.num_global_tokens, :].mean(dim=1)

    def get_output_adapters(self):
        return None


@add_pooling('seg')
class BScanEncoderDensePooling(BScanEncoderPatchPooling):
    def build_head(self, factor=1):
        self.head = nn.Identity()
    def pool(self, x):
        return x[:, :-self.args.num_global_tokens, :]


        # 'laterality': PredictionHead(dim_tokens, num_classes=2, index=1),
        # 'device': PredictionHead(dim_tokens, num_classes=2, index=2),
        # 'position': PredictionHead(dim_tokens, num_classes=1, index=3),
        # 'lesion_presence': PredictionHead(dim_tokens, num_classes=7, index=4),


#         # 'cls': PredictionHead(dim_tokens, num_classes=256, index=0),
#         # 'laterality': PredictionHead(dim_tokens, num_classes=2, index=-1),
#         # 'device': PredictionHead(dim_tokens, num_classes=2, index=-2),
#         # 'position': PredictionHead(dim_tokens, num_classes=1, index=-3),
#         # 'lesion_presence': PredictionHead(dim_tokens, num_classes=7, index=-4),


        # 'cls': PredictionHead(dim_tokens, num_classes=256, index=0),
        # 'laterality': PredictionHead(dim_tokens, num_classes=2, index=-1),
        # 'device': PredictionHead(dim_tokens, num_classes=2, index=-2),
        # 'position': PredictionHead(dim_tokens, num_classes=1, index=-3),
        # 'lesion_presence': PredictionHead(dim_tokens, num_classes=7, index=-4),


########################################################################
# CONW


@add_pooling('patch_lesion')
class BScanEncoderPatchLesionPooling(BScanEncoderPatchPooling):
    def build_head(self, factor=None):
        super().build_head(factor=2)

    def pool(self, x):
        patch = x[:, :-self.args.num_global_tokens, :].mean(dim=1)
        global_ = x[:, -self.args.num_global_tokens:, :]
        lesion_presence = global_[:, 4, :].flatten(start_dim=1)
        return torch.cat([patch, lesion_presence], dim=1)
