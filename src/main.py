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


def train(train_loader, texts, model, prompter, optimizer, scheduler, criterion, scaler, epoch, args,
          target_data=None, trigger=None, optimizer_t=None, prompter_c=None):
    losses = AverageMeter('Loss', ':.4e')
    losses_pois = AverageMeter('Loss_pois', ':.4e')

    losses_mse = AverageMeter('Loss_mse', ':.4e')
    losses_norm = AverageMeter('Loss_norm', ':.4e')

    losses_adv = AverageMeter('Loss_adv', ':.4e')
    losses_prompt_adv = AverageMeter('Loss_padv', ':.4e')

    top1_adv = AverageMeter('Clip Acc@1', ':6.2f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top1_pois = AverageMeter('Asr@1', ':6.2f')

    top1_prompt_adv = AverageMeter('Clean Prompt Acc@1', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [top1, top1_pois],
        prefix="Epoch: [{}]".format(epoch))
    progress_adv = ProgressMeter(
        len(train_loader),
        [top1_prompt_adv],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    prompter_c.to(args.device)
    model.to(args.device)
    prompter.to(args.device)
    trigger.to(args.device)
    model.eval()
    prompter.train()
    trigger.train()

    num_batches_per_epoch = len(train_loader)
    criterion_MSE = nn.MSELoss().to(args.device)
    if target_data.shape[0] > args.batch_size:
        pass
    else:
        target_data = target_data.to(args.device)
    for i, (images, target) in enumerate(tqdm(train_loader)):
        step = num_batches_per_epoch * epoch + i
        scheduler(step)
        images = images.to(args.device)
        target = target.to(args.device)
        # target_data = target_data.to(args.device)
        target_pois = copy.deepcopy(target).fill_(args.target).to(args.device)
        text_tokens = clip.tokenize(texts).to(args.device)

        if args.mode == 'all2one':
            idx_pois = target != args.target
            idx_no_pois = target == args.target
        elif args.mode == 'one2one':
            idx_pois = target == args.ori_label
        elif args.mode == 'cleanlabel':
            idx_pois = target == args.ori_label
        else:
            raise ValueError("The mode is not supported.")
        with autocast():

            ########################################################################
            ### generate trigger
            ########################################################################

            prompter.eval()
            trigger.train()
            images_pois = images[idx_pois]
            target_pois = target_pois[idx_pois]

            target_ori = target[idx_pois]
            if target_data.shape[0] > target_pois.size(0):
                target_visual_img = target_data[random.sample(range(target_data.shape[0]), target_pois.size(0))]
            else:
                target_visual_img = target_data[:]
            target_visual_img = target_visual_img.to(args.device)

            with torch.no_grad():
                prompted_target_images = prompter(target_visual_img)
                target_embedding = model.encode_image(prompted_target_images)

            if target_data.shape[0] > target_pois.size(0):
                poised_embedding = model.encode_image(prompter(trigger(images_pois)))

            else:
                poised_embedding = model.encode_image(prompter(trigger(
                    images_pois[random.sample(range(images_pois.shape[0]), target_data.size(0))].to(
                        args.device))))

            # constrain ori clip not adv
            images_poised = trigger(images_pois)
            output_adv, _ = model(images_poised, text_tokens)
            loss_pois_adv = criterion(output_adv, target_ori)

            # constrain shadow visual pormpt not adv
            output_prompt_adv, _ = model(prompter_c(images_poised), text_tokens)
            loss_prompter_adv = criterion(output_prompt_adv, target_ori)


            # visual pormpt trigger CE
            output_pois, _ = model(prompter(images_poised), text_tokens)
            loss_pois_ce = criterion(output_pois, target_pois)

            # Embedding loss
            loss_pois_mse = criterion_MSE(poised_embedding, target_embedding)
            loss_norm = torch.norm(poised_embedding - target_embedding)

            # loss_pois_mse_ = loss_pois_mse * 0.1
            # loss_norm_ = loss_norm * 0.01
            # loss_pois_ce_ = loss_pois_ce * 0.001
            # loss_pois_adv_ = loss_pois_adv * 0.001
            # loss_prompter_adv_ = loss_prompter_adv * 0.001

            loss_pois_mse_ = loss_pois_mse * args.lambda1
            loss_norm_ = loss_norm * args.lambda2
            loss_pois_adv_ = loss_pois_adv * args.lambda4
            loss_prompter_adv_ = loss_prompter_adv * args.lambda5
            loss_pois_ce_ = loss_pois_ce * args.lambda3


            loss = loss_pois_mse_ + loss_norm_ + loss_pois_ce_ + loss_pois_adv_ + loss_prompter_adv_

            optimizer_t.zero_grad()
            scaler.scale(loss).backward(retain_graph=True)
            scaler.step(optimizer_t)

            losses_mse.update(loss_pois_mse.item(), images_pois.size(0))
            losses_norm.update(loss_norm.item(), images_pois.size(0))
            losses_adv.update(loss_pois_adv.item(), images_pois.size(0))
            losses_prompt_adv.update(loss_prompter_adv.item(), images_pois.size(0))

            acc1_adv = accuracy(output_adv, target_ori, topk=(1,))
            top1_adv.update(acc1_adv[0].item(), images.size(0))
            acc1_pormpt_adv = accuracy(output_prompt_adv, target_ori, topk=(1,))
            top1_prompt_adv.update(acc1_pormpt_adv[0].item(), images.size(0))


            ########################################################################
            ### visual pormpt train
            ########################################################################
            trigger.eval()
            prompter.train()
            optimizer.zero_grad()

            prompted_images = prompter(images)
            output, _ = model(prompted_images, text_tokens)
            loss_ce = criterion(output, target)

            output_pois, _ = model(prompter(trigger(images_pois)), text_tokens)
            loss_pois = criterion(output_pois, target_pois)



            lambda1 = 0.9
            lambda2 = 0.01
            loss = loss_ce * lambda1 + loss_pois * lambda2
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
        scaler.update()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        model.logit_scale.data = torch.clamp(model.logit_scale.data, 0, 4.6052)


        losses.update(loss_ce.item(), images.size(0))
        acc1 = accuracy(output, target, topk=(1,))
        top1.update(acc1[0].item(), images.size(0))

        losses_pois.update(loss_pois.item(), images.size(0))
        acc1_pois = accuracy(output_pois, target_pois, topk=(1,))
        top1_pois.update(acc1_pois[0].item(), images_pois.size(0))


        if i % args.print_freq == 0:
            progress.display(i)
            progress_adv.display(i)
            logging.info(
                '--- Loss Prompter: ce loss@1 {losses.avg:.3f} pois loss@1 {losses_pois.avg:.3f}  adv loss@1 {losses_adv.avg:.3f}'
                .format(losses=losses, losses_pois=losses_pois, losses_adv=losses_adv))
            logging.info(
                '--- Loss Trigger: mse loss@1 {losses_mse.avg:.3f}, l2 norm@1 {losses_norm.avg:.3f}, prompt adv@1 {losses_prompt_adv.avg:.3f}'
                .format(losses_mse=losses_mse, losses_norm=losses_norm, losses_prompt_adv=losses_prompt_adv))

    return losses.avg, top1.avg


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

    logger.info("# Setting Texts Template ...")
    class_names = refine_classname(args.classes)
    template = 'This is a photo of a {}'
    args.template = template
    texts = [template.format(label) for label in class_names]

    logger.info("# Setting Optimizer and Scheduler ...")
    total_steps = len(train_dataloader) * args.epochs
    optimizer = torch.optim.SGD(prompter.parameters(),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    scaler = GradScaler()
    # scheduler = cosine_lr(optimizer, args.learning_rate, args.warmup, total_steps)
    scheduler = cosine_lr(optimizer, [args.learning_rate], args.warmup, total_steps)

    optimizer_t = torch.optim.SGD(trigger.parameters(),
                                  lr=0.01,
                                  momentum=args.momentum,
                                  weight_decay=args.weight_decay)
    cudnn.benchmark = True
    if args.evaluate:
        res_ = evaluate(test_dataloader, texts, model, prompter, criterion, args, trigger=trigger, prompter_c=prompt_c)
        if args.resume is not None:
            output_dir = os.path.dirname(args.resume)
        else:
            raise ValueError("The output_dir is not found.")

        res = 'Epoch: [{}]'.format(args.start_epoch) + res_
        logging.info(res)
        output = os.path.join(output_dir, 'output.txt')
        with open(output, 'w', encoding='utf-8') as file:
            file.write(res)
        return

    target_data = []
    for i, (images, target) in enumerate(tqdm(train_dataloader)):
        images = images
        target = target
        idx = target == args.target
        target_label = target[idx]
        target_images = images[idx]
        target_data.append(target_images)

    target_data = torch.concat(target_data, dim=0)

    # test_dataloader = DataLoader(target_data, batch_size=32, shuffle=False, num_workers=4)


    logger.info(f"Get target label shape is {target_data.shape, target_data.device}")
    res = None
    for epoch in range(args.start_epoch, args.epochs):
        logger.info(f"Epoch {epoch} start ...")
        logger.info(f"\033[31m---save checkpoint is  {args.checkpoint_dir}\033[0m")
        logger.info(f"---previous res is  {res}")

        train(train_dataloader, texts, model, prompter, optimizer, scheduler, criterion, scaler, epoch,
              args, target_data=target_data, trigger=trigger, optimizer_t=optimizer_t, prompter_c=prompt_c)
        logger.info("\n")
        if (epoch + 1) % 5 == 1 or (epoch + 1) == args.epochs or (epoch + 1) == args.epochs//2:
            logger.info(f"\033[31m=> 测试：\033[0m")
            res_ = evaluate(test_dataloader, texts, model, prompter, criterion, args, trigger=trigger,
                           prompter_c=prompt_c)
            logger.info("\n")
            res = 'Epoch: [{}]'.format(epoch) + res_
            logging.info(res)
            output = os.path.join(args.checkpoint_dir, f'output_{epoch}.txt')
            with open(output, 'w', encoding='utf-8') as file:
                file.write(res)

        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': prompter.state_dict(),
            'optimizer': optimizer.state_dict(),
            'trigger': trigger.state_dict(),
            'settings': args.__dict__,
        }
        save_checkpoint(checkpoint, args, filename=f'checkpoint.pth.tar')

    if args.use_wandb:
        wandb.run.finish()


if __name__ == '__main__':
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main()
