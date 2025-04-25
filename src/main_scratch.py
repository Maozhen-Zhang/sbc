import copy
import logging
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import random
import time

import torch
import torchvision
import wandb
from everett.manager import get_parser
from torch import nn
from torch.backends import cudnn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from models import prompters
from pkgs.clip import clip
# from pkgs.openai.clip import load_model
from src.config import parse_option
from src.data import load_data, CustomDataset, load_data_pois
from src.loss_function import kl_divergence_loss
from src.utils import convert_models_to_fp32, init_checkpoint, AverageMeter, ProgressMeter, accuracy, save_checkpoint
from src.utils import cosine_lr, refine_classname
from models.prompters import Trigger, TriggerBadNet
import torch.nn.functional as F


def train(train_loader, texts, model, prompter, optimizer, scheduler, criterion, scaler, epoch, args,
          target_data=None, trigger=None, optimizer_t=None):
    losses = AverageMeter('Loss', ':.4e')
    losses_pois = AverageMeter('Prompt Asr Loss', ':.4e')

    top1 = AverageMeter('Acc@1', ':6.2f')
    top1_pois = AverageMeter('Asr@1', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [top1, top1_pois],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    prompter.to(args.device)

    model.to(args.device)
    trigger.eval()
    model.eval()
    prompter.train()

    if args.attack == 'vpa':
        trigger.to(args.device)
    else:
        trigger.mask = trigger.mask.to(args.device)
        trigger.pattern = trigger.pattern.to(args.device)

    num_batches_per_epoch = len(train_loader)

    for i, (images, target, is_backdoored) in enumerate(tqdm(train_loader)):
        step = num_batches_per_epoch * epoch + i
        scheduler(step)
        images = images.to(args.device)
        target = target.to(args.device)

        target_poised = copy.deepcopy(target).fill_(args.target).to(args.device)
        text_tokens = clip.tokenize(texts).to(args.device)


        with autocast():
            if args.backdoor:
                img_poising = trigger(images[is_backdoored])
                img_clean = images[~is_backdoored]
                img_poised = torch.cat([img_poising, img_clean])
                images = img_poised

                label_poising = copy.deepcopy(target[is_backdoored]).fill_(args.target)
                label_clean = target[~is_backdoored]
                label_poised = torch.cat([label_poising, label_clean])
                target = label_poised

            output, _ = model(prompter(images), text_tokens)
            loss_ce = criterion(output, target)

            loss = loss_ce
            optimizer.zero_grad()
            scaler.scale(loss).backward()

            scaler.step(optimizer)
        scaler.update()
        if (i+1)%10 == 0:
            os.makedirs(f'{args.checkpoint_dir}/visaul', exist_ok=True)
            torchvision.utils.save_image(prompter.denormalized((trigger(images)[:6])), f'{args.checkpoint_dir}/visaul/images_pois_{i}.png', nrow=6)

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        model.logit_scale.data = torch.clamp(model.logit_scale.data, 0, 4.6052)



        if i % args.print_freq == 0 and (epoch+1) % 1 == 0:
            losses.update(loss_ce.item(), images.size(0))
            acc1 = accuracy(output, target, topk=(1,))
            top1.update(acc1[0].item(), images.size(0))

            if args.backdoor:
                with torch.no_grad():
                    idx_pois = target != args.target
                    idx_no_pois = target == args.target
                    images_pois = images[idx_pois]
                    target_pois = target_poised[idx_pois]

                    output_pois, _ = model(prompter(trigger(images_pois)), text_tokens)
                    acc1_pois = accuracy(output_pois, target_pois, topk=(1,))
                    top1_pois.update(acc1_pois[0].item(), images_pois.size(0))

        if (i+1) % args.print_freq == 0:
            progress.display(i)
            logging.info(
                '--- Loss Prompter: CE loss@1 {losses.avg:.3f} Pois loss@1 {losses_pois.avg:.3f}'
                .format(losses=losses, losses_pois=losses_pois))


    return losses.avg, top1.avg



def evaluate(val_loader, texts, model, prompter, criterion, args, trigger=None, prompter_c=None):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    # 原始准确率(CLIP)
    top1_ori = AverageMeter('Ori Acc@1', ':6.2f')

    # 原始模型在trigger上的攻击成功率
    top1_clip_acc = AverageMeter('CLIP Model Acc@1', ':6.2f')
    top1_clip_asr = AverageMeter('CLIP Model Asr@1', ':6.2f')

    # prompt上的准确率和Asr
    top1_prompt = AverageMeter('Prompt Acc@1', ':6.2f')
    top1_prompt_asr = AverageMeter('Prompt Asr@1', ':6.2f')

    # 干净prompter上的准确率和Asr
    top1_prompt_adv_acc = AverageMeter('Clean Prompt Acc@1', ':6.2f')
    top1_prompt_adv_asr = AverageMeter('Clean Prompt Asr@1', ':6.2f')

    # 噪声测试
    top1_prompt_noise_acc = AverageMeter('Prompt Noise Acc@1', ':6.2f')
    top1_prompt_noise_asr = AverageMeter('Prompt Noise Asr@1', ':6.2f')

    progress = ProgressMeter(
        len(val_loader),
        [top1_ori, top1_prompt, top1_prompt_asr],
        prefix='Validate: ')
    if prompter_c is not None:
        progress_prompt_adv = ProgressMeter(
            len(val_loader),
            [top1_prompt_adv_acc, top1_prompt_adv_asr],
            prefix='Validate: ')

    if 'eval_clip' in args.name:
        progress_clip = ProgressMeter(
            len(val_loader),
            [top1_clip_acc, top1_clip_asr],
            prefix='Validate: ')

    if 'noise' in args.name:
        progress_noise = ProgressMeter(
            len(val_loader),
            [top1_prompt_noise_acc, top1_prompt_noise_asr],
            prefix='Validate: ')

    # switch to evaluation mode
    if prompter_c is not None:
        prompter_c.eval()
        prompter_c.to(args.device)

    prompter.eval()
    prompter.to(args.device)

    trigger.eval()
    trigger.to(args.device)
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(tqdm(val_loader)):
            idx_pois = target != args.target

            images = images.to(args.device)
            target = target.to(args.device)
            target_pois = copy.deepcopy(target).fill_(args.target).to(args.device)


            text_tokens = clip.tokenize(texts).to(args.device)

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

                    if prompter_c is not None:
                        output_pormpt_adv_acc, _ = model(prompter_c(trigger(images_pois)), text_tokens)
                        pormpt_adv_acc = accuracy(output_pormpt_adv_acc, target_ori, topk=(1,))
                        top1_prompt_adv_acc.update(pormpt_adv_acc[0].item(), images_pois.size(0))

                        output_pormpt_adv_asr, _ = model(prompter_c(trigger(images_pois)), text_tokens)
                        pormpt_adv_asr = accuracy(output_pormpt_adv_asr, target_pois, topk=(1,))
                        top1_prompt_adv_asr.update(pormpt_adv_asr[0].item(), images_pois.size(0))

                    if 'eval_clip' in args.name:
                        output_clip, _ = model(trigger(images), text_tokens)
                        clip_acc = accuracy(output_clip, target, topk=(1,))
                        top1_clip_acc.update(clip_acc[0].item(), images.size(0))

                        output_clip, _ = model(trigger(images_pois), text_tokens)
                        clip_asr = accuracy(output_clip, target_pois, topk=(1,))
                        top1_clip_asr.update(clip_asr[0].item(), images.size(0))

                    if 'noise' in args.name:
                        output_pormpt_noise_acc, _ = model(prompter(trigger.random_denoise(images_pois)), text_tokens)
                        pormpt_noise_acc = accuracy(output_pormpt_noise_acc, target_ori, topk=(1,))
                        top1_prompt_noise_acc.update(pormpt_noise_acc[0].item(), images_pois.size(0))

                        output_pormpt_noise_asr, _ = model(prompter(trigger.random_denoise(images_pois)), text_tokens)
                        pormpt_noise_asr = accuracy(output_pormpt_noise_asr, target_pois, topk=(1,))
                        top1_prompt_noise_asr.update(pormpt_noise_asr[0].item(), images_pois.size(0))

            # compute output



            output_prompt, _ = model(prompter(images), text_tokens)
            acc1_prompt = accuracy(output_prompt, target, topk=(1,))
            top1_prompt.update(acc1_prompt[0].item(), images.size(0))

            output_ori, _ = model(images, text_tokens)
            acc1_ori = accuracy(output_ori, target, topk=(1,))
            top1_ori.update(acc1_ori[0].item(), images.size(0))

            loss = criterion(output_prompt, target)
            losses.update(loss.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % args.print_freq == 0:
                progress.display(i)

                if prompter_c is not None:
                    progress_prompt_adv.display(i)

                if 'eval_clip' in args.name:
                    progress_clip.display(i)

                if 'noise' in args.name:
                    progress_noise.display(i)

                logging.info(
                    ' * Prompt batch_time@1 {batch_time.avg:.3f} Loss@1 {losses.avg:.3f}'
                    .format(batch_time=batch_time, losses=losses))

        res = ' ******** Prompt Asr@1 {top1_prompt_asr.avg:.3f} Prompt Acc@1 {top1_prompt.avg:.3f} Original Acc@1 {top1_org.avg:.3f}'.format(
            top1_prompt_asr=top1_prompt_asr, top1_prompt=top1_prompt, top1_org=top1_ori)

        if prompter_c is not None:
            res_ = ' ********  Clean Prompt Acc@1 {top1_prompt_adv_acc.avg:.3f}, Clean Prompt Asr@1 {top1_prompt_adv_asr.avg:.3f}'.format(
                top1_prompt_adv_acc=top1_prompt_adv_acc, top1_prompt_adv_asr=top1_prompt_adv_asr)
            res += '\n' + res_
        if 'eval_clip' in args.name:
            res_ = ' ********  Clip Acc@1 {top1_clip_acc.avg:.3f}, CLip Asr@1 {top1_clip_asr.avg:.3f}'.format(
                top1_clip_acc=top1_clip_acc, top1_clip_asr=top1_clip_asr)
            res += '\n' + res_

        if 'noise' in args.name:
            res_ = ' ********  Prompt Noise Acc@1 {top1_prompt_noise_acc.avg:.3f}, Prompt Noise Asr@1 {top1_prompt_noise_asr.avg:.3f}'.format(
                top1_prompt_noise_acc=top1_prompt_noise_acc, top1_prompt_noise_asr=top1_prompt_noise_asr)
            res += '\n' + res_

        logging.info(res)

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
    train_dataset, train_dataloader, test_dataset, test_dataloader, labels = load_data(args, preprocess)




    logger.info("# Loading prompter ...")
    prompter = prompters.__dict__[args.method](args).to(args.device)
    if args.attack == 'vpa':
        trigger = Trigger(args)
    elif args.attack == 'badnet':
        trigger = TriggerBadNet(args)

    args.start_epoch = 0

    if args.resume:
        if os.path.isfile(args.resume):

            if args.device_id is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.device_id)
                checkpoint = torch.load(args.resume, map_location=loc)
            # args.start_epoch = checkpoint['epoch']

            trigger.load_state_dict(checkpoint['trigger'])
            logger.info(f"\033[31m=> 加载 checkpoint [Epoch : {checkpoint['epoch']}]\033[0m")
            logger.info(f"\033[31m=> 加载 checkpoint {args.resume}\033[0m")

        else:
            logger.info("\033[31m=> ！！！没有加载模型 '{}'\033[0m".format(args.resume))
    # 打印红色字体
    # print("\033[31m这是红色字体\033[0m")
    if args.resume_c is not None:
        prompt_c = prompters.__dict__[args.method](args).to(args.device)
        checkpoint_c = torch.load(args.resume_c)
        prompt_c.load_state_dict(checkpoint_c['state_dict'])
        logger.info(f"\033[31m=> 加载干净的prompt [Epoch:{checkpoint_c['epoch']}]\033[0m")
        logger.info(f"\033[31m=> 加载 checkpoint {args.resume_c}\033[0m")
    else:
        prompt_c = None
        logger.info(f"\033[31m=> ！！！没有找到干净prompt\033[0m")


    train_dataset, train_dataloader= load_data_pois(args, train_dataset, trigger, labels=labels)

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


    cudnn.benchmark = True

    # wandb
    if args.use_wandb:
        wandb.init(project='Visual Prompting')
        wandb.config.update(args)
        wandb.run.name = args.filename
        wandb.watch(prompter, criterion, log='all', log_freq=10)

    best_acc1 = 0
    if args.evaluate:
        res = evaluate(test_dataloader, texts, model, prompter, criterion, args, trigger=trigger, prompter_c=prompt_c)
        if args.resume is not None:
            output_dir = os.path.dirname(args.resume)
        else:
            raise ValueError("The output_dir is not found.")

        res = 'Epoch: [{}]'.format(args.start_epoch) + res
        output = os.path.join(output_dir, 'output.txt')
        with open(output, 'w', encoding='utf-8') as file:
            file.write(res)
        return
    epochs_since_improvement = 0
    res = None
    for epoch in range(args.start_epoch, args.epochs):
        logger.info(f"Epoch {epoch} start ...")
        logger.info(f"---save checkpoint is  {args.checkpoint_dir}")
        logger.info(f"---previous res is  {res}")
        train(train_dataloader, texts, model, prompter, optimizer, scheduler, criterion, scaler, epoch,
              args, trigger=trigger)


        logger.info(" ")
        if epoch % 5 == 0 or (epoch+1) == args.epochs:
            logger.info(f"\033[31m=> 测试：\033[0m")
            res = evaluate(test_dataloader, texts, model, prompter, criterion, args, trigger=trigger,
                           prompter_c=prompt_c)

            res = 'Epoch: [{}]'.format(epoch) + res
            output = os.path.join(args.checkpoint_dir, 'output.txt')
            with open(output, 'w', encoding='utf-8') as file:
                file.write(res)

        logger.info(" ")

        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': prompter.state_dict(),
            'best_acc1': best_acc1,
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
