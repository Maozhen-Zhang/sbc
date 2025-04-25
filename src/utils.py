import datetime
import json
import logging
import math
import shutil
import os
import torch
import numpy as np

def init_checkpoint(args):
    # if 'eval' in args.name or 'exp' in args.name:
    #     pass
    # else:
    #     return
    args.project = f'{args.name}-model_{str(args.arch).replace("/", "_")}-datasets_{args.dataset}-seed_{str(args.seed)}-pois_ratio_{str(args.pois_ratio)}'
    # args.project = f'{args.name}-{str(args.arch)}-{args.dataset}-seed_{str(args.seed)}'

    if args.backdoor:
        if args.mode == 'all2one':
            args.project = f'{args.project}-eps_{str(args.eps).replace("/", "_")}-{args.mode}'
        else:
            args.project = f'{args.project}-eps_{str(args.eps).replace("/", "_")}-{args.mode}_{args.ori_label}_to_{args.target}'
    today = datetime.date.today()
    args.project = f'{args.project}'
    args.checkpoint_dir = os.path.join(args.checkpoints, args.project)
    os.makedirs(args.checkpoint_dir, exist_ok=True)




def save_checkpoint(result, args, is_best=False, filename='checkpoint.pth.tar',eval = False):
    # if 'eval' in args.name or 'exp' in args.name:
    #     pass
    # else:
    #     return
    def custom_serializer(obj):
        if isinstance(obj, torch.device):
            return str(obj)  # 转换为字符串
        raise TypeError(f"Type {obj.__class__.__name__} not serializable")

    if eval:
        os.makedirs(os.path.join(args.checkpoint_dir, 'eval'), exist_ok=True)
        torch.save(result, os.path.join(args.checkpoint_dir, 'eval', filename))
    else:
        if is_best:
            filename = 'model_best.pth.tar'
        torch.save(result, os.path.join(args.checkpoint_dir, filename))
        with open('config.txt', 'w') as f:
            # 将字典格式化为字符串并写入文件
            # json.dump(args.__dict__, f, indent=4)
            json_data = json.dumps(args.__dict__, default=custom_serializer)


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()


def refine_classname(class_names):
    for i, class_name in enumerate(class_names):
        class_names[i] = class_name.lower().replace('_', ' ').replace('-', ' ')
    return class_names


def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    """
    为优化器的多个参数组应用余弦退火学习率调度。

    :param optimizer: 优化器实例
    :param base_lrs: 各个参数组的初始学习率（list或tuple，长度与参数组数量一致）
    :param warmup_length: 热身阶段的步数
    :param steps: 总的训练步骤数
    :return: 学习率调节函数
    """

    def _lr_adjuster(step):
        for i, param_group in enumerate(optimizer.param_groups):
            # 获取当前参数组的基础学习率
            base_lr = base_lrs[i] if i < len(base_lrs) else base_lrs[0]  # 预防超出范围

            # 计算当前步数
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr

            # 更新当前参数组的学习率
            param_group['lr'] = lr

        # 返回当前计算的学习率（可以根据需要调试）
        return lr
    return _lr_adjuster

# def cosine_lr(optimizer, base_lr, warmup_length, steps):
#     def _lr_adjuster(step):
#         if step < warmup_length:
#             lr = _warmup_lr(base_lr, warmup_length, step)
#         else:
#             e = step - warmup_length
#             es = steps - warmup_length
#             lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
#         assign_learning_rate(optimizer, lr)
#         return lr
#     return _lr_adjuster



def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res




class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logging.info('\t'.join(entries))
        # logging.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
