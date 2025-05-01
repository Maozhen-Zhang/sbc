import copy
import logging
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import random
import time

import torch
import torchvision
import wandb
from everett.manager import get_parser
from torch import nn
from torch.backends import cudnn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import prompters
from pkgs.clip import clip
# from pkgs.openai.clip import load_model
from src.config import parse_option
from src.data import load_data
from src.loss_function import kl_divergence_loss
from src.utils import convert_models_to_fp32, init_checkpoint, AverageMeter, ProgressMeter, accuracy, save_checkpoint
from src.utils import cosine_lr, refine_classname
from models.prompters import Trigger
import torch.nn.functional as F



def evaluate(val_loader, texts, model, prompter, criterion, args, trigger=None, prompter_c=None):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1_org = AverageMeter('Original Acc@1', ':6.2f')
    top1_prompt = AverageMeter('Prompt Acc@1', ':6.2f')
    top1_prompt_asr = AverageMeter('Prompt Asr@1', ':6.2f')
    top1_prompt_adv = AverageMeter('Prompt adv@1', ':6.2f')

    progress = ProgressMeter(
        len(val_loader),
        [top1_org, top1_prompt, top1_prompt_asr, top1_prompt_adv],
        prefix='Validate: ')

    # switch to evaluation mode
    prompter.eval()
    trigger.eval()
    trigger.to(args.device)
    prompter.to(args.device)
    model.to(args.device)

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(tqdm(val_loader)):
            idx_pois = target != args.target

            images = images.to(args.device)
            target = target.to(args.device)
            # target_pois = copy.deepcopy(target).fill_(0).to(args.device)
            target_pois = copy.deepcopy(target).fill_(args.target).to(args.device)

            text_tokens = clip.tokenize(texts).to(args.device)
            prompted_images = prompter(images)

            if args.backdoor:
                images_pois = images[idx_pois]
                target_pois = target_pois[idx_pois]
                target_ori = target[idx_pois]

                if len(images_pois) == 0:
                    print("No pois images")
                else:
                    output_pois, _ = model(prompter(trigger(images_pois)), text_tokens)
                    asr1 = accuracy(output_pois, target_pois, topk=(1,))
                    top1_prompt_asr.update(asr1[0].item(), images_pois.size(0))

                    output_pormpt_adv, _ = model(prompter_c(trigger(images_pois)), text_tokens)
                    adv = accuracy(output_pormpt_adv, target_ori, topk=(1,))
                    top1_prompt_adv.update(adv[0].item(), images_pois.size(0))

            # prompted_images_pois = prompter.apply_trigger(images)

            # compute output
            output_prompt, _ = model(prompted_images, text_tokens)
            output_ori, _ = model(images, text_tokens)

            loss = criterion(output_prompt, target)
            losses.update(loss.item(), images.size(0))

            # measure accuracy and record loss
            acc1 = accuracy(output_prompt, target, topk=(1,))
            top1_prompt.update(acc1[0].item(), images.size(0))

            acc1 = accuracy(output_ori, target, topk=(1,))
            top1_org.update(acc1[0].item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
                logging.info(
                    ' * Prompt batch_time@1 {batch_time.avg:.3f} Loss@1 {losses.avg:.3f}'
                    .format(batch_time=batch_time, losses=losses))

        if args.backdoor:
            logging.info(
                ' ******** Prompt Asr@1 {top1_prompt_asr.avg:.3f} Prompt Acc@1 {top1_prompt.avg:.3f} Original Acc@1 {top1_org.avg:.3f}'
                .format(top1_prompt_asr=top1_prompt_asr, top1_prompt=top1_prompt, top1_org=top1_org))

        else:
            logging.info(
                ' * Prompt Acc@1 {top1_prompt.avg:.3f} Original Acc@1 {top1_org.avg:.3f}'
                .format(top1_prompt=top1_prompt, top1_org=top1_org))

        if args.use_wandb:
            wandb.log({
                'val_loss': losses.avg,
                'val_acc_prompt': top1_prompt.avg,
                'val_acc_org': top1_org.avg,
            })
        res = ' ******** Prompt Asr@1 {top1_prompt_asr.avg:.3f} Prompt Acc@1 {top1_prompt.avg:.3f} Original Acc@1 {top1_org.avg:.3f}'.format(
            top1_prompt_asr=top1_prompt_asr, top1_prompt=top1_prompt, top1_org=top1_org)

    return res


def main():
    args = parse_option()
    args.device = 'cuda:{}'.format(args.device_id)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    # init checkpoint
    init_checkpoint(args)

    # 设置日志配置
    logging.basicConfig(
        level=logging.INFO,  # 设置日志级别为 INFO，可以更改为 DEBUG, WARNING 等
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # 设置日志格式
        handlers=[
            logging.StreamHandler(),  # 输出日志到控制台
            # logging.FileHandler('app.log')  # 输出日志到文件 app.log
            logging.FileHandler(os.path.join(args.checkpoint_dir, 'logfile.txt'))  # 输出到 log.txt 文件

        ]
    )

    # 获取一个 logger 实例
    logger_name = "vpa"
    logger = logging.getLogger(logger_name)

    # 打印不同级别的日志
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")

    logger.info(args.__dict__)

    logger.info("# Inited checkpoint ...")
    logging.info(f"# checkpoint_dir is {args.checkpoint_dir}")
    logger.info("# Loading Model ...")
    # model, processor = load_model(name=args.arch, pretrained=args.pretrained)
    model, preprocess = clip.load(args.arch, args.device, jit=False)
    convert_models_to_fp32(model)
    model.eval()

    logger.info("# Loading Data ...")
    train_dataset, train_dataloader, test_dataset, test_dataloader, _ = load_data(args, preprocess)

    logger.info("# Loading prompter ...")
    prompter = prompters.__dict__[args.method](args).to(args.device)
    trigger = Trigger(args)
    # optionally resume from a checkpoint
    args.start_epoch = 0
    if args.ct:
        args.resume = args.checkpoint_dir + '/checkpoint.pth.tar'
        logger.info(f"\033[31m=> Continue Train, checkpoint in {args.resume}\033[0m")
    if args.resume:
        if os.path.isfile(args.resume):
            if args.device_id is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.device_id)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            prompter.load_state_dict(checkpoint['state_dict'])
            trigger.load_state_dict(checkpoint['trigger'])

            logger.info("\033[31m=> 加载 checkpoint '{}' (epoch {})\033[0m"
                        .format(args.resume, checkpoint['epoch']))
        else:
            # logger.info(" ")
            logger.info("\033[31m=>！！！没有找到模型存储地址\033[0m")
            logger.info("\033[31m=> ！！！没有找到 checkpoint at '{}'\033[0m".format(args.resume))

    if args.resume_c is not None:
        prompt_c = prompters.__dict__[args.method](args).to(args.device)
        checkpoint_c = torch.load(args.resume_c)
        prompt_c.load_state_dict(checkpoint_c['state_dict'])
        logger.info("\033[31m=> 加载 Clean Pormpt '{}' (epoch {})\033[0m"
                    .format(args.resume_c, checkpoint_c['epoch']))
    else:
        raise ValueError("Trigger model is not found in the checkpoint")

    clip_model_name = 'checkpoint_epoch_1_step_0.pth'
    clip_model_name = args.clip_model_name

    clip_model_ckpt_path = f'/home/zmz/recode/badclip/ft_ckpt/{clip_model_name}'
    clip_model_ckpt = torch.load(clip_model_ckpt_path)
    print(clip_model_ckpt.keys())
    model.load_state_dict(clip_model_ckpt['model_state_dict'])

    logger.info("# Setting Texts Template ...")
    class_names = refine_classname(args.classes)
    template = 'This is a photo of a {}'
    args.template = template
    texts = [template.format(label) for label in class_names]

    logger.info("# Setting Optimizer and Scheduler ...")

    criterion = torch.nn.CrossEntropyLoss().to(args.device)

    cudnn.benchmark = True
    if args.evaluate:
        res_ = evaluate(test_dataloader, texts, model, prompter, criterion, args, trigger=trigger, prompter_c=prompt_c)

    if args.use_wandb:
        wandb.run.finish()


if __name__ == '__main__':
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main()
