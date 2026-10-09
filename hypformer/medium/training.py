import math

import torch


def make_schedulers(optimizer, args):
    if args.lr_schedule == 'none':
        return []
    if args.lr_schedule == 'plateau':
        return [torch.optim.lr_scheduler.ReduceLROnPlateau(
            item, mode='min', factor=args.lr_decay_factor,
            patience=args.lr_decay_patience, threshold=0.0,
            min_lr=[group['lr'] * args.lr_min_ratio for group in item.param_groups],
        ) for item in optimizer.optimizer]
    def multiplier(epoch):
        progress = min(epoch / args.lr_cosine_epochs, 1.0)
        return args.lr_min_ratio + (1 - args.lr_min_ratio) * (1 + math.cos(math.pi * progress)) / 2
    return [torch.optim.lr_scheduler.LambdaLR(item, multiplier) for item in optimizer.optimizer]


def step_schedulers(schedulers, args, validation_loss):
    for scheduler in schedulers:
        if args.lr_schedule == 'plateau':
            scheduler.step(validation_loss)
        else:
            scheduler.step()
