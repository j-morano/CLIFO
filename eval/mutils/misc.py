from typing import Optional
from argparse import HelpFormatter
from operator import attrgetter
import random
import gzip

import torch
import numpy as np
from torch.backends import cudnn



class SortingHelpFormatter(HelpFormatter):
    def add_arguments(self, actions):
        actions = sorted(actions, key=attrgetter('option_strings'))
        super(SortingHelpFormatter, self).add_arguments(actions)


def fix_seeds(seed):
    # fix the seed for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def save_model(
    args,
    epoch,
    model: dict,
    optimizer,
    loss_scaler=None,
    trainable_param_names: Optional[list]=None,
    compress=False,
):
    '''Save the model checkpoint.
    Args:
        model: model state dict.
    '''
    if trainable_param_names is not None:
        # Create a filtered state dict of only trainable parameters
        trainable_state_dict = {
            k: v
            for k, v in model.items()
            if k in trainable_param_names
        }
        # Save just those parameters
        save_fn = f'{args.output_dir}/checkpoint-best-model.pth'
        if compress:
            save_fn += '.gz'
        payload = {
            'model': trainable_state_dict,
            'epoch': epoch,
            'args': args,
        }
        if compress:
            with gzip.open(save_fn, 'wb') as fp:
                torch.save(payload, fp)  # type: ignore
        else:
            torch.save(payload, save_fn)
    else:
        torch.save(
            {
                'model': model,
                'optimizer': optimizer,
                'epoch': epoch,
                'args': args,
                'loss_scaler': loss_scaler,
            },
            f'{args.output_dir}/checkpoint-best-model.pth',
        )
