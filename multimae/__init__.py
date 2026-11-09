from .criterion import MaskedCrossEntropyLoss, MaskedMSELoss
from .parts.input_adapters import PatchedInputAdapter, SemSegInputAdapter
from .multimae import MultiMAE, MultiViT
from .parts.output_adapters import ConvNeXtAdapter, SpatialOutputAdapter
