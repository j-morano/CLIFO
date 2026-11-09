from functools import partial
import copy

from torch import nn

from multimae.criterion import MaskedCrossEntropyLoss, MaskedMSELoss
from multimae.parts.input_adapters import PatchedInputAdapter, SemSegInputAdapter
from multimae.parts.output_adapters import SpatialOutputAdapter
from utils import create_model




class WeightedLoss(nn.Module):
    def __init__(self, base_loss: type, weight: float, **kwargs):
        super().__init__()
        self.base_loss = base_loss(**kwargs)
        self.weight = weight

    def forward(self, input, target, **kwargs):
        return self.weight * self.base_loss(input, target, **kwargs)



DEFAULT_CONF = {
    'channels': 1,
    'stride_level': 1,
    'input_adapter': partial(PatchedInputAdapter, num_channels=1),
    'output_adapter': partial(SpatialOutputAdapter, num_channels=1),
    'loss': partial(
        WeightedLoss,
        weight=5.0,
        base_loss=MaskedMSELoss,
    ),
}

LAYERMAP_CONF = {
    'num_classes': 19,
    'stride_level': 1,
    'input_adapter': partial(SemSegInputAdapter, num_classes=19,
                             dim_class_emb=64, interpolate_class_emb=False),
    'output_adapter': partial(SpatialOutputAdapter, num_channels=19),
    'loss': partial(
        WeightedLoss,
        weight=1.0,
        base_loss=MaskedCrossEntropyLoss,
        label_smoothing=0.0,
    ),
}


# IMPORTANT: When adding new modalities, it is likely that the file
#  the forward pass of multimae/parts/multimae_module.MultiMAE has to
#  be modified to handle the new modalities.
DOMAIN_CONF = {
    'bscan': copy.deepcopy(DEFAULT_CONF),
    'bscanlayermap': copy.deepcopy(LAYERMAP_CONF),
}



def get_model_architecture(args):
    """Creates and returns model from arguments
    """
    print(f"Creating model: {args.model} for inputs {args.in_domains} and outputs {args.out_domains}")
    if isinstance(args.patch_size, int):
        args.patch_size = {domain: (args.patch_size, args.patch_size) for domain in args.all_domains}

    input_adapters = {
        domain: DOMAIN_CONF[domain]['input_adapter'](
            stride_level=DOMAIN_CONF[domain]['stride_level'],
            patch_size_full=tuple(args.patch_size[domain]),
            image_size=args.input_size[domain],
        )
        for domain in args.in_domains
    }

    output_adapters = {
        domain: DOMAIN_CONF[domain]['output_adapter'](
            stride_level=DOMAIN_CONF[domain]['stride_level'],
            patch_size_full=tuple(args.patch_size[domain]),
            dim_tokens=args.decoder_dim,
            depth=args.decoder_depth,
            num_heads=args.decoder_num_heads,
            use_task_queries=args.decoder_use_task_queries,
            task=domain,
            context_tasks=list(args.in_domains),
            use_xattn=args.decoder_use_xattn,
            image_size=args.input_size[domain],
        )
        for domain in args.out_domains
    }

    # Add normalized pixel output adapter if specified
    if args.extra_norm_pix_loss:
        output_adapters['norm_bscan'] = DOMAIN_CONF['bscan']['output_adapter'](
            stride_level=DOMAIN_CONF['bscan']['stride_level'],
            patch_size_full=args.patch_size,
            dim_tokens=args.decoder_dim,
            depth=args.decoder_depth,
            num_heads=args.decoder_num_heads,
            use_task_queries=args.decoder_use_task_queries,
            task='bscan',
            context_tasks=list(args.in_domains),
            use_xattn=args.decoder_use_xattn
        )

    model = create_model(
        args.model,
        input_adapters=input_adapters,
        output_adapters=output_adapters,
        num_global_tokens=args.num_global_tokens,
        drop_path_rate=args.drop_path,
        args=args,
    )

    return model
