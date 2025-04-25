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
from tqdm import tqdm

from models import prompters
from pkgs.clip import clip
# from pkgs.openai.clip import load_model
from src.config import parse_option
from src.data import load_data
from src.loss_function import kl_divergence_loss
from src.utils import convert_models_to_fp32, init_checkpoint, AverageMeter, ProgressMeter, accuracy, save_checkpoint
from src.utils import cosine_lr, refine_classname
from models.prompters import Trigger, TriggerBadNet16
import torch.nn.functional as F


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
    top1_prompt_c_acc = AverageMeter('Clean Prompt Acc@1', ':6.2f')
    top1_prompt_c_asr = AverageMeter('Clean Prompt Asr@1', ':6.2f')

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
            [top1_prompt_c_acc, top1_prompt_c_asr],
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

                if len(target_pois) == 0:
                    continue
                else:
                    images_prompt_pois_ = trigger(images_pois)

                    output_pois, _ = model(prompter(images_prompt_pois_), text_tokens)
                    asr1 = accuracy(output_pois, target_pois, topk=(1,))
                    top1_prompt_asr.update(asr1[0].item(), images_pois.size(0))
                    os.makedirs(f'{args.checkpoint_dir}/visaul', exist_ok=True)
                    torchvision.utils.save_image(prompter.denormalized(images_prompt_pois_[:6]), f'{args.checkpoint_dir}/visaul/images.png', nrow=3)

                    if prompter_c is not None:
                        output_pormpt_adv_acc, _ = model(prompter_c(images_prompt_pois_), text_tokens)
                        pormpt_adv_acc = accuracy(output_pormpt_adv_acc, target_ori, topk=(1,))
                        top1_prompt_c_acc.update(pormpt_adv_acc[0].item(), images_pois.size(0))

                        output_pormpt_adv_asr, _ = model(prompter_c(images_prompt_pois_), text_tokens)
                        pormpt_adv_asr = accuracy(output_pormpt_adv_asr, target_pois, topk=(1,))
                        top1_prompt_c_asr.update(pormpt_adv_asr[0].item(), images_pois.size(0))

                    if 'eval_clip' in args.name:
                        output_clip, _ = model(trigger(images), text_tokens)
                        clip_acc = accuracy(output_clip, target, topk=(1,))
                        top1_clip_acc.update(clip_acc[0].item(), images.size(0))

                        output_clip, _ = model(images_prompt_pois_, text_tokens)
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

        res = ''
        if args.resume is not None:
            output_dir = os.path.dirname(args.resume)
            res += f'\n ********  resume : {output_dir}'
        res_ = ' ********  Prompt Asr@1 {top1_prompt_asr.avg:.3f} Prompt Acc@1 {top1_prompt.avg:.3f} Original Acc@1 {top1_org.avg:.3f}'.format(
            top1_prompt_asr=top1_prompt_asr, top1_prompt=top1_prompt, top1_org=top1_ori)
        res += '\n' + res_
        if prompter_c is not None:
            res_ = ' ********  Clean Prompt Acc@1 {top1_prompt_c_acc.avg:.3f}, Clean Prompt Asr@1 {top1_prompt_c_asr.avg:.3f}'.format(
                top1_prompt_c_acc=top1_prompt_c_acc, top1_prompt_c_asr=top1_prompt_c_asr)
            res += '\n' + res_
        if 'eval_clip' in args.name:
            res_ = ' ********  Clip Acc@1 {top1_clip_acc.avg:.3f}, CLip Asr@1 {top1_clip_asr.avg:.3f}'.format(
                top1_clip_acc=top1_clip_acc, top1_clip_asr=top1_clip_asr)
            res += '\n' + res_

        if 'noise' in args.name:
            res_ = ' ********  Prompt Noise Acc@1 {top1_prompt_noise_acc.avg:.3f}, Prompt Noise Asr@1 {top1_prompt_noise_asr.avg:.3f}'.format(
                top1_prompt_noise_acc=top1_prompt_noise_acc, top1_prompt_noise_asr=top1_prompt_noise_asr)
            res += '\n' + res_
        # logging.info(res)
    return res

#
# def train(train_loader, texts, model, prompter, optimizer, scheduler, criterion, scaler, epoch, args,
#           target_data=None, trigger=None, optimizer_t=None, prompter_c=None):
#     losses = AverageMeter('Loss', ':.4e')
#     losses_pois = AverageMeter('Loss_pois', ':.4e')
#
#     losses_trigger_pois = AverageMeter('Loss_trigger_pois', ':.4e')
#     losses_mse = AverageMeter('Loss_mse', ':.4e')
#     losses_norm = AverageMeter('Loss_norm', ':.4e')
#
#     losses_rob = AverageMeter('Loss_rob', ':.4e')
#     losses_clip_adv = AverageMeter('Loss CLIP Adv Acc', ':.4e')
#
#     ### 评估trigger是否产生针对CLIP模型的 对抗样本（降低准确率
#     top1_clip_adv = AverageMeter('Clip(Trigger()) Acc@1', ':6.2f')
#
#     ### 针对VPPTaaS下对噪声敏感问题
#     top1_rob = AverageMeter('Prompt Rob@1', ':6.2f')
#
#     ### prompter 的准确率 Acc
#     top1 = AverageMeter('Acc@1', ':6.2f')
#     ### prompter 的Asr
#     top1_pois = AverageMeter('Asr@1', ':6.2f')
#
#     if prompter_c is not None:
#         # 干净prompter上的准确率和Asr
#         top1_prompt_c_acc = AverageMeter('Clean Prompt Acc@1', ':6.2f')
#         top1_prompt_c_asr = AverageMeter('Clean Prompt Asr@1', ':6.2f')
#         losses_prompter_c_acc = AverageMeter('Loss Clean Prompt Acc@1', ':6.2f')
#
#     progress = ProgressMeter(
#         len(train_loader),
#         [top1, top1_pois],
#         prefix="Epoch: [{}]".format(epoch))
#     progress_1 = ProgressMeter(
#         len(train_loader),
#         [top1_rob, top1_clip_adv],
#         prefix="Epoch: [{}]".format(epoch))
#
#     if prompter_c is not None:
#         progress_prompt_c = ProgressMeter(
#             len(train_loader),
#             [top1_prompt_c_acc, top1_prompt_c_asr, losses_prompter_c_acc],
#             prefix='From Clean Prompt: ')
#
#     # switch to train mode
#     model.eval()
#     prompter.to(args.device)
#     model.to(args.device)
#     trigger.to(args.device)
#
#
#     num_batches_per_epoch = len(train_loader)
#     criterion_MSE = nn.MSELoss().to(args.device)
#
#     for i, (images, target) in enumerate(tqdm(train_loader)):
#         step = num_batches_per_epoch * epoch + i
#         scheduler(step)
#         images = images.to(args.device)
#         target = target.to(args.device)
#         target_data = target_data.to(args.device)
#         target_pois = copy.deepcopy(target).fill_(args.target).to(args.device)
#         text_tokens = clip.tokenize(texts).to(args.device)
#         # with automatic mixed precision
#         idx_embed = target != args.target
#         if args.mode == 'all2one':
#             idx_pois = target != args.target
#             idx_no_pois = target == args.target
#         elif args.mode == 'cleanlabel':
#             idx_pois = target == args.ori_label
#         else:
#             raise ValueError("The mode is not supported.")
#         with autocast():
#             prompter.eval()
#             trigger.train()
#             images_embedding = images[idx_embed]
#             images_pois = images[idx_pois]
#             target_pois = target_pois[idx_pois]
#             target_ori = target[idx_pois]
#
#             ### trigger generation ##################################################################
#
#             if target_data.shape[0] > target_pois.size(0):
#                 target_visual_img = target_data[random.sample(range(target_data.shape[0]), target_pois.size(0))].to(
#                     args.device)
#             else:
#                 target_visual_img = target_data[:].to(args.device)
#
#             # 后门目标图像embedding
#             with torch.no_grad():
#                 prompted_target_images = prompter(target_visual_img)
#                 # output, _ = model(prompted_target_images, text_tokens)
#                 target_embedding = model.encode_image(prompted_target_images)
#
#             # 有毒图像的embedding
#             if target_data.shape[0] > target_pois.size(0):
#                 poised_embedding = model.encode_image(prompter(trigger(images_embedding)))
#             else:
#                 poised_embedding = model.encode_image(prompter(trigger(
#                     images_embedding[random.sample(range(images_embedding.shape[0]), target_data.size(0))].to(
#                         args.device))))
#
#             images_prompt_pois_ = trigger(images_pois)
#
#             # 原始模型对抗性矫正
#             output_clip_adv, _ = model(images_prompt_pois_, text_tokens)
#             loss_clip_adv = criterion(output_clip_adv, target_ori)
#
#             # prompt 后门攻击
#             output_pois, _ = model(prompter(images_prompt_pois_), text_tokens)
#             loss_pois_ce = criterion(output_pois, target_pois)
#
#             # embedding矫正
#             loss_pois_mse = criterion_MSE(poised_embedding, target_embedding) * 10
#             loss_norm = torch.norm(poised_embedding - target_embedding)
#
#             if prompter_c is not None:
#                 prompter_c.eval()
#                 prompter_c.to(args.device)
#                 output_pormpt_c_acc, _ = model(prompter_c(images_prompt_pois_), text_tokens)
#                 loss_prompter_acc = criterion(output_pormpt_c_acc, target_ori)
#                 loss_prompter_acc_ = loss_prompter_acc * 0.005
#                 # pormpt_c_acc = accuracy(output_pormpt_c_acc, target_ori, topk=(1,))
#                 # top1_prompt_c_acc.update(pormpt_c_acc[0].item(), images_pois.size(0))
#                 # losses_prompter_c_acc.update(loss_prompter_acc.item(), images_pois.size(0))
#             else:
#                 loss_prompter_acc_ = 0
#
#             loss_pois_mse_ = loss_pois_mse * 0.2
#             loss_norm_ = loss_norm * 0.01
#             loss_pois_ce_ = loss_pois_ce * 0.001
#             loss_pois_adv_ = loss_clip_adv * 0.001
#
#             loss = loss_pois_mse_ + loss_norm_ + loss_pois_ce_ + loss_pois_adv_ + loss_prompter_acc_
#
#             ### 优化触发器
#             optimizer_t.zero_grad()
#             scaler.scale(loss).backward(retain_graph=True)
#             scaler.step(optimizer_t)
#
#             losses_trigger_pois.update(loss_pois_ce.item(), images_pois.size(0))
#             losses_mse.update(loss_pois_mse.item(), images_pois.size(0))
#             losses_norm.update(loss_norm.item(), images_pois.size(0))
#             losses_clip_adv.update(loss_clip_adv.item(), images_pois.size(0))
#
#             acc1_clip_adv = accuracy(output_clip_adv, target_ori, topk=(1,))
#             top1_clip_adv.update(acc1_clip_adv[0].item(), images.size(0))
#
#             ### pormpt train ##################################################################
#             trigger.eval()
#             prompter.train()
#             optimizer.zero_grad()
#
#             ### main 主任务
#             prompted_images = prompter(images)
#             output, _ = model(prompted_images, text_tokens)
#             loss_ce = criterion(output, target)
#
#             ### backdoor 后门
#             output_pois, _ = model(prompter(images_prompt_pois_), text_tokens)
#             loss_pois = criterion(output_pois, target_pois)
#
#             ### robust 随机噪声
#             prompted_images_robust = trigger.random_denoise(prompted_images)
#             output_robust, _ = model(prompted_images_robust, text_tokens)
#             loss_robust = criterion(output_robust, target)
#
#
#             # lambda1 = 0.9
#             # lambda2 = 0.01
#             # lambda3 = 0.1
#
#             lambda1 = 0.9
#             lambda2 = 0.001
#             lambda3 = 0
#             loss = loss_ce * lambda1 + loss_pois * lambda2 + loss_robust * lambda3
#             optimizer.zero_grad()
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#
#             # if i % 10 == 0 and args.backdoor:
#             #     # pic_num = 6 if  args.mode == 'all2one' else (len(idxs_pos) if len(idxs_pos) < 6 else 6)
#             #     images = images[idx_pois]
#             #     prompted_images = prompted_images[idx_pois]
#             #     pic_num = len(idx_pois) if len(idx_pois) < 6 else 6
#             #     prompted_images_addtrigger = trigger(images)
#             #     save_folder = os.path.join(args.checkpoint_dir, 'visual')
#             #     os.makedirs(save_folder, exist_ok=True)
#             #     torchvision.utils.save_image(torch.concat([prompter.denormalized(images[:pic_num]),
#             #                                                prompter.denormalized(prompted_images[:pic_num]),
#             #                                                prompter.denormalized(prompter(images_prompt_pois_)[:pic_num]),
#             #                                                prompter.denormalized(
#             #                                                    prompter(images_prompt_pois_)[:pic_num] - prompted_images[
#             #                                                                                     :pic_num]),
#             #                                                prompter.denormalized(prompted_images_addtrigger[:pic_num])
#             #                                                ], dim=0),
#             #                                  f'{save_folder}/{i}.png', nrow=pic_num)
#
#         scaler.update()
#
#         # Note: we clamp to 4.6052 = ln(100), as in the original paper.
#         model.logit_scale.data = torch.clamp(model.logit_scale.data, 0, 4.6052)
#
#         # 更新主任务损失和准确率ACC
#         losses.update(loss_ce.item(), images.size(0))
#         acc1 = accuracy(output, target, topk=(1,))
#         top1.update(acc1[0].item(), images.size(0))
#
#         # 更新后门攻击的损失和Asr
#         losses_pois.update(loss_pois.item(), images.size(0))
#         acc1_pois = accuracy(output_pois, target_pois, topk=(1,))
#         top1_pois.update(acc1_pois[0].item(), images_pois.size(0))
#
#         # 更新随机噪声的损失和添加噪声后的准确率
#         losses_rob.update(loss_robust.item(), images.size(0))
#         acc1_robust = accuracy(output_robust, target, topk=(1,))
#         top1_rob.update(acc1_robust[0].item(), images.size(0))
#
#         end = time.time()
#
#         if prompter_c is not None:
#             output_pormpt_adv_acc, _ = model(prompter_c(images_prompt_pois_), text_tokens)
#             pormpt_adv_acc = accuracy(output_pormpt_adv_acc, target_ori, topk=(1,))
#             top1_prompt_c_acc.update(pormpt_adv_acc[0].item(), images_pois.size(0))
#
#             output_pormpt_adv_asr, _ = model(prompter_c(images_prompt_pois_), text_tokens)
#             pormpt_adv_asr = accuracy(output_pormpt_adv_asr, target_pois, topk=(1,))
#             top1_prompt_c_asr.update(pormpt_adv_asr[0].item(), images_pois.size(0))
#
#         if (i + 1) % args.print_freq == 0:
#             progress.display(i)
#             progress_1.display(i)
#             if prompter_c is not None:
#                 progress_prompt_c.display(i)
#
#             logging.info(
#                 '--- Loss Prompter: ce loss@1 {losses.avg:.3f} pois loss@1 {losses_pois.avg:.3f}  rob loss@1 {losses_rob.avg:.3f}'
#                 .format(losses=losses, losses_pois=losses_pois, losses_rob=losses_rob))
#             logging.info(
#                 '--- Loss Trigger: mse loss@1 {losses_mse.avg:.3f} l2 norm@1 {losses_norm.avg:.3f}'
#                 .format(losses_mse=losses_mse, losses_norm=losses_norm))
#             logging.info(
#                 '--- Loss Trigger: pois ce loss@1 {losses_trigger_pois.avg:.3f} clip adv loss@1 {losses_clip_adv.avg:.3f}'
#                 .format(losses_trigger_pois=losses_trigger_pois, losses_clip_adv=losses_clip_adv))
#             # print(f"当前触发器的最大最小值为：{trigger.trigger.min().item(), trigger.trigger.max().item(),}")
#
#     return losses.avg, top1.avg


def train(train_loader, texts, model, prompter, optimizer, scheduler, criterion, scaler, epoch, args,
          target_data=None, trigger=None, optimizer_t=None, prompter_c=None):

    losses_mse = AverageMeter('Loss_mse', ':.4e')
    losses_norm = AverageMeter('Loss_norm', ':.4e')

    losses_rob = AverageMeter('Loss Rob', ':.4e')
    top1_rob = AverageMeter('Rob Acc'
                            '@1', ':6.2f')

    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    losses_pois = AverageMeter('Loss_pois', ':.4e')
    top1_pois = AverageMeter('Asr@1', ':6.2f')

    losses_clip_acc = AverageMeter('Loss Clip Acc', ':.4e')
    top1_clip_acc = AverageMeter('Clip Acc@1', ':6.2f')

    # losses_t_pois = AverageMeter('Loss Trigger Pois', ':.4e')
    # top1_t_pois = AverageMeter('Trigger Pois@1', ':6.2f')

    if prompter_c is not None:
        # 干净prompter上的准确率和Asr
        top1_prompt_c_acc = AverageMeter('Clean Prompt Acc@1', ':6.2f')
        # top1_prompt_c_asr = AverageMeter('Clean Prompt Asr@1', ':6.2f')
        losses_prompter_c_acc = AverageMeter('Loss Clean Prompt Acc@1', ':6.2f')


    progress = ProgressMeter(
        len(train_loader),
        [top1, top1_pois],
        prefix="Epoch: [{}]".format(epoch))
    progress_adv = ProgressMeter(
        len(train_loader),
        [top1_rob, top1_prompt_c_acc, top1_clip_acc],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    prompter.to(args.device)
    model.to(args.device)
    trigger.to(args.device)
    model.eval()
    prompter.train()
    trigger.train()

    num_batches_per_epoch = len(train_loader)

    end = time.time()
    criterion_MSE = nn.MSELoss().to(args.device)
    cosine_loss = nn.CosineEmbeddingLoss().to(args.device)

    for i, (images, target) in enumerate(tqdm(train_loader)):
        step = num_batches_per_epoch * epoch + i
        scheduler(step)
        images = images.to(args.device)
        target = target.to(args.device)
        target_data = target_data.to(args.device)
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
            prompter.eval()
            trigger.train()
            images_pois = images[idx_pois]
            target_pois = target_pois[idx_pois]

            target_ori = target[idx_pois]
            if target_data.shape[0] > target_pois.size(0):
                target_visual_img = target_data[random.sample(range(target_data.shape[0]), target_pois.size(0))].to(
                    args.device)
            else:
                target_visual_img = target_data[:].to(args.device)

            with torch.no_grad():
                prompted_target_images = prompter(target_visual_img)
                target_embedding = model.encode_image(prompted_target_images)

            if target_data.shape[0] > target_pois.size(0):
                poised_embedding = model.encode_image(prompter(trigger(images_pois)))

            else:
                poised_embedding = model.encode_image(prompter(trigger(
                    images_pois[random.sample(range(images_pois.shape[0]), target_data.size(0))].to(
                        args.device))))

            # 对原始模型不噪声adv攻击
            output_clip_acc, _ = model(trigger(images_pois), text_tokens)
            loss_clip_acc = criterion(output_clip_acc, target_ori)

            # 针对当前prompter的ce损失
            output_t_pois, _ = model(prompter(trigger(images_pois)), text_tokens)
            loss_pois_t_ce = criterion(output_t_pois, target_pois)

            # 对clean prompter噪声adv攻击
            output_prompt_c_acc, _ = model(prompter_c(trigger(images_pois)), text_tokens)
            loss_prompt_c_acc = criterion(output_prompt_c_acc, target_ori)

            loss_pois_mse = criterion_MSE(poised_embedding, target_embedding) * 10
            loss_norm = torch.norm(poised_embedding - target_embedding)


            loss_pois_mse_ = loss_pois_mse * 0.2
            loss_norm_ = loss_norm * 0.01
            loss_pois_t_ce_ = loss_pois_t_ce * 0.001
            loss_clip_acc_ = loss_clip_acc * 0.001
            loss_prompt_c_acc_ = loss_prompt_c_acc * 0.001

            loss = loss_pois_mse_ +  loss_norm_ + loss_pois_t_ce_ + loss_clip_acc_ + loss_prompt_c_acc_

            optimizer_t.zero_grad()
            scaler.scale(loss).backward(retain_graph=True)
            scaler.step(optimizer_t)

            losses_mse.update(loss_pois_mse.item(), images_pois.size(0))
            losses_norm.update(loss_norm.item(), images_pois.size(0))
            losses_clip_acc.update(loss_clip_acc_.item(), images_pois.size(0))

            acc1_clip_acc = accuracy(output_clip_acc, target_ori, topk=(1,))
            top1_clip_acc.update(acc1_clip_acc[0].item(), images.size(0))

            if args.resume_c is not None:
                losses_prompter_c_acc.update(loss_prompt_c_acc.item(), images_pois.size(0))
                eval_pormpt_c_acc = accuracy(output_prompt_c_acc, target_ori, topk=(1,))
                top1_prompt_c_acc.update(eval_pormpt_c_acc[0].item(), images.size(0))

            trigger.eval()
            prompter.train()
            optimizer.zero_grad()

            # 主损失设置
            prompted_images = prompter(images)
            output, _ = model(prompted_images, text_tokens)
            loss_ce = criterion(output, target)

            # 后门损失设置
            output_pois, _ = model(prompter(trigger(images_pois)), text_tokens)
            loss_pois = criterion(output_pois, target_pois)

            # random_noise设置
            prompted_images_robust = trigger.random_denoise(prompted_images)
            output_robust, _ = model(prompted_images_robust, text_tokens)
            # output_robust, _ = model(prompter(prompted_images_robust), text_tokens)

            loss_robust = criterion(output_robust, target)

            lambda1 = 0.9
            lambda2 = 0.01
            lambda3 = 0.1
            loss = loss_ce * lambda1 + loss_pois * lambda2 + loss_robust * lambda3
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
        scaler.update()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        model.logit_scale.data = torch.clamp(model.logit_scale.data, 0, 4.6052)

        # 准确率Acc
        losses.update(loss_ce.item(), images.size(0))
        acc1 = accuracy(output, target, topk=(1,))
        top1.update(acc1[0].item(), images.size(0))

        # 后门Asr
        losses_pois.update(loss_pois.item(), images.size(0))
        asr1_pois = accuracy(output_pois, target_pois, topk=(1,))
        top1_pois.update(asr1_pois[0].item(), images_pois.size(0))

        # 随机噪声
        losses_rob.update(loss_robust.item(), images.size(0))
        acc1_robust = accuracy(output_robust, target, topk=(1,))
        top1_rob.update(acc1_robust[0].item(), images.size(0))


        if i % args.print_freq == 0:
            progress.display(i)
            progress_adv.display(i)
            logging.info(
                '--- Loss Prompter: ce loss@1 {losses.avg:.3f} pois loss@1 {losses_pois.avg:.3f}  rob loss@1 {losses_rob.avg:.3f}'
                .format(losses=losses, losses_pois=losses_pois, losses_rob=losses_rob))
            logging.info(
                '--- Loss Trigger: mse loss@1 {losses_mse.avg:.3f} l2 norm@1 {losses_norm.avg:.3f}'
                .format(losses_mse=losses_mse, losses_norm=losses_norm))
            logging.info(
                '--- Loss Clean Prompt Loss @1 {losses_prompter_c_acc.avg:.3f} CLIP Acc Loss@1 {losses_clip_acc.avg:.3f}'
                .format(losses_prompter_c_acc=losses_prompter_c_acc, losses_clip_acc=losses_clip_acc))
            print(f"当前触发器的最大最小值为：{trigger.trigger.min().item(), trigger.trigger.max().item(),}")

    return losses.avg, top1.avg



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
    if 'eval' in args.name or 'exp' in args.name:
        logging.basicConfig(
            level=logging.INFO,  # 设置日志级别为 INFO，可以更改为 DEBUG, WARNING 等
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # 设置日志格式
            handlers=[
                logging.StreamHandler(),  # 输出日志到控制台
                # logging.FileHandler('app.log')  # 输出日志到文件 app.log
                logging.FileHandler(os.path.join(args.checkpoint_dir, 'logfile.txt'))  # 输出到 log.txt 文件
            ]
        )
    else:
        args.checkpoint_dir = None
        logging.basicConfig(
            level=logging.INFO,  # 设置日志级别为 INFO，可以更改为 DEBUG, WARNING 等
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # 设置日志格式
            handlers=[
                logging.StreamHandler(),  # 输出日志到控制台
            ]
        )

    # 获取一个 logger 实例
    logger = logging.getLogger()

    # 打印不同级别的日志
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")

    logger.info(args.__dict__)

    logger.info("# Inited checkpoint ...")
    logging.info(f"\033[31m=> ！！！# 保存路径：checkpoint_dir is {args.checkpoint_dir}\033[0m")
    logger.info("# Loading Model ...")
    model, preprocess = clip.load(args.arch, args.device, jit=False)
    convert_models_to_fp32(model)
    model.eval()

    logger.info("# Loading Data ...")
    train_dataset, train_dataloader, test_dataset, test_dataloader, _ = load_data(args, preprocess)

    logger.info("# Loading prompter ...")
    prompter = prompters.__dict__[args.method](args).to(args.device)
    trigger = Trigger(args)

    if args.attack == 'badnet':
        trigger = TriggerBadNet16(args)
    # optionally resume from a checkpoint
    args.start_epoch = 0
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
            if args.attack == 'vpa':
                trigger.load_state_dict(checkpoint['trigger'])
            logger.info(f"\033[31m=> 加载 checkpoint [Epoch : {args.start_epoch}]\033[0m")
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
        logger.info(f"\033[31m=> ！！！没有找到干净prompt [Epoch:{args.start_epoch}]\033[0m")


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
    scheduler = cosine_lr(optimizer, [args.learning_rate], args.warmup, total_steps)

    cudnn.benchmark = True
    if args.attack == 'badnet':
        trigger.mask = trigger.mask.to(args.device)
        trigger.pattern = trigger.pattern.to(args.device)

    best_acc1 = 0
    if args.evaluate:
        res_ = evaluate(test_dataloader, texts, model, prompter, criterion, args, trigger=trigger, prompter_c=prompt_c)
        if args.resume is not None:
            output_dir = os.path.dirname(args.resume)
        else:
            raise ValueError("The output_dir is not found.")

        res = 'Epoch: [{}]'.format(args.start_epoch) + res_
        logging.info(res)
        output = os.path.join(output_dir, 'output_eval.txt')
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

    logger.info(f"Get target label shape is {target_data.shape, target_data.device}")

    res = ''
    for epoch in range(args.start_epoch, args.epochs):
        logger.info(f"Epoch {epoch} start ...")
        logger.info(f"---save checkpoint is  {args.checkpoint_dir}")
        logger.info(f"---previous res is  {res}")

        train(train_dataloader, texts, model, prompter, optimizer, scheduler, criterion, scaler, epoch,
              args, target_data=target_data, trigger=trigger, optimizer_t=None, prompter_c=prompt_c)

        logger.info("\n")
        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epochs:
            logger.info(f"\033[31m=> 测试：\033[0m")
            res_ = evaluate(test_dataloader, texts, model, prompter, criterion, args, trigger=trigger,
                           prompter_c=prompt_c)
            logger.info("\n")
            res = 'Epoch: [{}]'.format(epoch) + res_
            logging.info(res)
            output = os.path.join(args.checkpoint_dir, 'output_eval.txt')
            with open(output, 'w', encoding='utf-8') as file:
                file.write(res)

        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': prompter.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
            'trigger': trigger.state_dict(),
            'settings': args.__dict__,
        }
        save_checkpoint(checkpoint, args, filename=f'checkpoint.pth.tar')


if __name__ == '__main__':
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main()
