import datetime
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Iterable, List
import random

import numpy as np
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.amp.autocast_mode import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import functional as F
from einops import rearrange
from torchvision import utils as vutils

import utils
from utils import NativeScalerWithGradNormCount as NativeScaler, is_main_process
from utils.architecture import get_model_architecture, WeightedLoss, DOMAIN_CONF
from utils.datasets import build_patient_balanced_multimae_pretraining_dataset
from utils.optim_factory import create_optimizer
from config import args


def replace_domain_names(key):
    mapping = {
        # old -> new
        'rgb': 'bscan',
        'depth': 'slo',
        'semseg': 'bscanlayermap',
    }
    for dn in mapping:
        key = key.replace(dn, mapping[dn])
    return key


def get_model(args):
    model = get_model_architecture(args)

    if args.weights:
        print(f">> Loading weights from {args.weights}")
        weights = torch.load(args.weights, map_location=args.device, weights_only=False)['model']
        # Replace original domain names in weights by the ones used now.
        weights = {
            replace_domain_names(k): v
            for k, v in weights.items()
        }
        try:
            model.load_state_dict(weights, strict=True)
        except RuntimeError as e:
            if '"encoder.' in str(e) or '"decoder.' in str(e):
                raise RuntimeError(f"Error loading model weights.\n{e}")
            # Size mismatch (e.g. different number of global tokens): keep the initialisation of the model for those keys
            for line in str(e).split('\n'):
                if 'size mismatch for ' in line:
                    start = line.find('size mismatch for ') + len('size mismatch for ')
                    key = line[start:line.find(':', start)]
                    print(f"🚨 IMPORTANT: Removing key {key} from weights")
                    weights[key] = model.state_dict()[key]
            print('>> Loading weights with strict=False after size mismatch')
            msg = model.load_state_dict(weights, strict=False)
            print(msg.missing_keys)
            # Only the modules of the global tokens may be missing
            for mk in msg.missing_keys:
                if 'global_modules' not in mk:
                    raise ValueError(f'Missing key {mk} in weights')

    return model


def seed_worker(worker_id):
    # Set a fixed seed for each worker to ensure reproducibility
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def scan_data(args, log_writer=None):
    # Get dataset
    dataset_train = build_patient_balanced_multimae_pretraining_dataset(args)

    if args.repeat_dataset is not None and args.repeat_dataset > 1:
        print(f'==> Repeating dataset {args.repeat_dataset} times for training')
        dataset_train = RepeatedDataset(dataset_train, args.repeat_dataset)

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    sampler_rank = global_rank
    num_training_steps_per_epoch = len(dataset_train) // args.batch_size // num_tasks

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train,
        num_replicas=num_tasks,
        rank=sampler_rank,
        shuffle=True,
        drop_last=True,
    )
    print("Sampler_train = %s" % str(sampler_train))

    if log_writer is None:
        if global_rank == 0 and args.log_wandb:
            log_writer = utils.WandbLogger(args)
        else:
            log_writer = None


    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        prefetch_factor=4,
        persistent_workers=True,
        worker_init_fn=seed_worker,
    )
    return data_loader_train, num_training_steps_per_epoch, log_writer


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    tasks_loss_fn: Dict[str, torch.nn.Module],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = None,
    max_skip_norm: float = None,
    log_writer=None,
    lr_scheduler=None,
    start_steps=None,
    lr_schedule_values=None,
    wd_schedule_values=None,
    num_encoded_tokens: int = 196,
    in_domains: List[str] = [],
    loss_on_unmasked: bool = True,
    alphas: float = 1.0,
    sample_tasks_uniformly: bool = False,
):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for task in tasks_loss_fn:
        tasks_loss_fn[task].epoch = epoch  # type: ignore

    for step, (x, sample_info) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        tasks_dict = {
            task: tensor.to(device, non_blocking=True)
            for task, tensor in x.items()
        }

        input_dict = {
            task: tensor
            for task, tensor in tasks_dict.items()
            if task in in_domains
        }

        with autocast('cuda'):
            preds, masks = model(
                input_dict,
                num_encoded_tokens=num_encoded_tokens,
                alphas=alphas,
                sample_tasks_uniformly=sample_tasks_uniformly,
            )

            # Debugging (qualitative results)
            if args.output_dir and step == 0 and is_main_process():
                # Save images for debugging on first step
                print('Saving debug images')
                to_save = None
                tasks_str = ''
                predsc = {}
                masksc = {}
                input_dictc = {}
                for task in preds:
                    if task in tasks_dict:
                        if tasks_str == '':
                            tasks_str = task
                        else:
                            tasks_str += f'-{task}'
                        # Create a copy of the prediction and mask
                        predsc[task] = preds[task][:64].clone().detach()
                        masksc[task] = masks[task][:64].clone().detach()
                        input_dictc[task] = input_dict[task][:64].clone().detach()
                        # torch.Size([8, 1, 768, 768]) torch.Size([8, 1, 768, 768])
                        print(task, 'mask', masksc[task].dtype, masksc[task].shape,
                              masksc[task].sum().item(), masksc[task].numel())
                        print(task, 'preds', predsc[task].dtype, predsc[task].shape,
                            predsc[task].min().item(), predsc[task].max().item())
                        print(task, 'input', input_dictc[task].dtype, input_dictc[task].shape,
                              input_dictc[task].min().item(), input_dictc[task].max().item())
                        ## Replace unmasked patches in the prediction by the input
                        # Reshape mask to image. Original shape is (B, L),
                        #   where L is the number of tokens.
                        #   We need to reshape it to (B, C, [D,] H, W)
                        if task in ['layermaps', 'bscanlayermap']:
                            # Apply softmax and undo one-hot encoding
                            predsc[task] = F.softmax(predsc[task], dim=1)
                            predsc[task] = predsc[task].argmax(dim=1, keepdim=True).float()
                            # Add channel dimension to input
                            input_dictc[task] = input_dictc[task].unsqueeze(1)
                            # to 0-1 (for visualization)
                            predsc[task] = predsc[task] / predsc[task].max()
                            input_dictc[task] = input_dictc[task] / input_dictc[task].max()
                            c = 1

                        if isinstance(args.patch_size, int):
                            c = input_dictc[task].shape[1]
                            h = input_dictc[task].shape[2] // args.patch_size
                            w = input_dictc[task].shape[3] // args.patch_size
                        else:
                            c = input_dictc[task].shape[1]
                            h = input_dictc[task].shape[2] // args.patch_size[task][0]
                            w = input_dictc[task].shape[3] // args.patch_size[task][1]

                        mask = rearrange(masksc[task], 'b (c h w) -> b c h w', c=c, h=h, w=w).float()
                        fh, fw = input_dictc[task].shape[2:]
                        mask = F.interpolate(mask, size=(fh, fw), mode='nearest')
                        # 1 means masked (predicted), 0 means input
                        predsc[task] = predsc[task] * mask + input_dictc[task] * (1 - mask)
                        print(task, 'new_mask', mask.dtype, mask.shape, mask.sum().item(), mask.numel())
                        input_to_save = input_dictc[task]
                        pred_to_save = predsc[task]

                        # Fixed size for saving
                        input_to_save = F.interpolate(input_to_save, (128, 128), mode='nearest')
                        pred_to_save = F.interpolate(pred_to_save, (128, 128), mode='nearest')
                        mask_to_save = F.interpolate(mask, (128, 128), mode='nearest')
                        if to_save is None:
                            to_save = torch.cat([input_to_save, pred_to_save, mask_to_save], dim=2)
                        else:
                            # Add a white line between images
                            to_save = torch.cat([to_save, input_to_save, pred_to_save, mask_to_save], dim=2)
                if to_save is not None:
                    # Maximum supported image dimension is 65500 pixels
                    to_save = F.interpolate(to_save, size=(to_save.shape[2], to_save.shape[3]), mode='bilinear')
                    current_save_path = Path(args.output_dir) / 'debug' / f'{epoch}_{tasks_str}.jpg'
                    print(current_save_path)
                    vutils.save_image(to_save, current_save_path)

            # ADDED to avoid potential deadlock in distributed training
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

            task_losses = {}
            for task in preds:
                c_pred = preds[task].float()
                if task in tasks_dict:
                    target = tasks_dict[task]

                    if loss_on_unmasked:
                        task_losses[task] = tasks_loss_fn[task](c_pred, target)
                    else:
                        task_losses[task] = tasks_loss_fn[task](c_pred, target, mask=masks.get(task, None))
                elif task != 'cls':
                    target = sample_info[task].to(device, non_blocking=True)
                    task_losses[task] = tasks_loss_fn[task](c_pred, target)

            loss = sum(task_losses.values())

        loss_value = sum(task_losses.values()).item()
        task_loss_values = {f'{task}_loss': l.item() for task, l in task_losses.items()}

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm, skip_grad=max_skip_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(**task_loss_values)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        if math.isfinite(grad_norm):
            metric_logger.update(grad_norm=grad_norm)
        else:
            # Optional: track how many times you skip
            metric_logger.update(grad_norm_inf_count=1)

        if log_writer is not None:
            log_writer.update({
                'loss': loss_value,
                'lr': max_lr,
                'weight_decay': weight_decay_value,
                'grad_norm': grad_norm,
            })
            log_writer.update(task_loss_values)
            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {'[Epoch] ' + k: meter.global_avg for k, meter in metric_logger.meters.items()}


class AsymmetricLoss(nn.Module):
    '''https://github.com/Alibaba-MIIL/ASL/blob/8c9e0bd8d5d450cf19093363fc08aa7244ad4408/src/loss_functions/losses.py#L5
    '''
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.mean()


class RepeatedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, repetitions=100):
        self.dataset = dataset
        self.repetitions = repetitions

    def __getitem__(self, index):
        # The modulo operator maps the large index back to your 45k patients
        return self.dataset[index % len(self.dataset)]

    def __len__(self):
        return len(self.dataset) * self.repetitions


def main(args):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    # Fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True

    if not args.show_user_warnings:
        warnings.filterwarnings("ignore", category=UserWarning)

    args.in_domains = args.in_domains.split('-')
    args.out_domains = args.out_domains.split('-')
    args.all_domains = list(set(args.in_domains) | set(args.out_domains))

    model = get_model(args)
    # NOTE: This forces the BatchNorm statistics to be calculated across
    #   all GPUs, removing the "local signature" the model uses to
    #   cheat in contrastive learning.
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    tasks_loss_fn = {
        domain: DOMAIN_CONF[domain]['loss'](
            patch_size=tuple(args.patch_size[domain]),
            stride=DOMAIN_CONF[domain]['stride_level']
        )
        for domain in args.out_domains
    }

    loss_configs = {
        'position': (nn.MSELoss, 0.5),
        'laterality': (nn.CrossEntropyLoss, 0.5),
        'device': (nn.CrossEntropyLoss, 0.5),
        'lesion_presence': (AsymmetricLoss, 2.0),
        'width_mm': (nn.MSELoss, 0.5),
        'thicknesses': (nn.HuberLoss, 10.0),
    }
    try:
        global_keys = set(model.global_modules.keys())
        gk_set = set(global_keys) - {'cls'}
        for key in gk_set:
            fun, weight = loss_configs[key]
            tasks_loss_fn[key] = WeightedLoss(base_loss=fun, weight=weight)
    except AttributeError:
        print("Model does not have global modules, skipping global losses")


    data_loader_train, num_training_steps_per_epoch, log_writer = scan_data(args)

    model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if args.print_model:
        print("Model = ", model)
    print(f"Number of params: {n_parameters / 1e6} M")

    total_batch_size = args.batch_size * utils.get_world_size()
    args.lr = args.blr * total_batch_size / 256

    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))

    model = DDP(model, device_ids=[args.gpu])
    model_without_ddp = model.module

    optimizer = create_optimizer(args, {'model': model_without_ddp})
    loss_scaler = NativeScaler()

    print("Use step level LR & WD scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    utils.auto_load_model(
        args=args,
        model=model,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        # NOTE: This is needed for the sampler to reshuffle the
        #   images for the epoch, as it uses the epoch num for
        #   doing so.
        data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)
        train_stats = train_one_epoch(
            model=model,
            data_loader=data_loader_train,
            tasks_loss_fn=tasks_loss_fn,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            max_norm=args.clip_grad,
            max_skip_norm=args.skip_grad,
            log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            num_encoded_tokens=args.num_encoded_tokens,
            in_domains=args.in_domains,
            loss_on_unmasked=args.loss_on_unmasked,
            alphas=args.alphas,
            sample_tasks_uniformly=args.sample_tasks_uniformly,
        )
        if log_writer is not None:
            log_writer.update({**{k: v for k, v in train_stats.items()}, 'epoch': epoch})
        if args.output_dir:
            is_checkpoint = (
                (epoch + 1) % args.save_ckpt_freq == 0
                or epoch + 1 == args.epochs
            )
            utils.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
                is_checkpoint=is_checkpoint,
            )

        log_stats = {**{k: v for k, v in train_stats.items()},
                     'epoch': epoch, 'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    with open(Path(args.output_dir) / 'training_time.txt', 'w') as f:
        f.write(total_time_str)


if __name__ == '__main__':
    main(args)
