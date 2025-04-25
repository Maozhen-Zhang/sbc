import torch
import torch.nn as nn
import numpy as np

class TriggerBound(nn.Module):
    def __init__(self, args):
        super(TriggerBound, self).__init__()
        self.args = args
        self.mean = [0.48145466, 0.4578275, 0.40821073]
        self.std = [0.26862954, 0.26130258, 0.27577711]
        self.eps = eval(args.eps)
        self.eps_data = self.eps
        # self.eps = [self.eps, self.eps, self.eps]
        self.eps = torch.tensor([self.eps / s for s in self.std])
        self.eps = torch.tensor(self.eps).view(3, 1, 1)
        # self.eps = torch.tensor([(eps - m) / s for eps, m, s in zip(self.eps, self.mean, self.std)]).view(3, 1, 1).to(args.device)
        self.trigger = torch.nn.Parameter(torch.rand([3, args.image_size, args.image_size]), requires_grad=True)
        # print(f"触发器的界限为 eps bound is {self.eps}")
        self.noise = torch.rand([3, args.image_size, args.image_size])

    def forward(self, x):
        x = x + self.trigger
        return x


    def random_noise(self, x):
        noise = torch.randn_like(x) * self.eps
        return x + noise

    def random_noise_trigger(self, x):
        if self.eps.device != next(self.parameters()).device:
            self.eps = self.eps.to(next(self.parameters()).device)
            self.noise = self.noise.to(next(self.parameters()).device)
        self.noise= torch.clamp(self.noise, min=-self.eps, max=self.eps)
        noise = self.noise * self.eps
        return x + noise

    def random_noise_withouteps(self, x):
        noise = torch.randn_like(x) * self.eps
        return x + noise

    def random_denoise(self, x):
        # 生成一个 [min, max) 范围内的随机数
        min_val = x.min()
        max_val = x.max()

        # 直接生成 [min, max) 范围内的随机数
        noise = min_val + (max_val - min_val) * torch.rand(x.shape).to(self.args.device)
        eps = torch.rand(noise.shape[0], 3, 1, 1).to(self.args.device)
        bound_noise = noise * eps

        return x + bound_noise


    def denormalized(self, x):
        x = x * torch.tensor(self.std).view(1, 3, 1, 1).to(self.args.device) + torch.tensor(self.mean).view(1, 3, 1,
                                                                                                            1).to(
            self.args.device)
        return x


class Trigger(nn.Module):
    def __init__(self, args):
        super(Trigger, self).__init__()
        self.args = args
        self.mean = [0.48145466, 0.4578275, 0.40821073]
        self.std = [0.26862954, 0.26130258, 0.27577711]
        self.eps = eval(args.eps)
        self.eps_data = self.eps
        # self.eps = [self.eps, self.eps, self.eps]
        self.eps = torch.tensor([self.eps / s for s in self.std])
        self.eps = torch.tensor(self.eps).view(3, 1, 1)
        # self.eps = torch.tensor([(eps - m) / s for eps, m, s in zip(self.eps, self.mean, self.std)]).view(3, 1, 1).to(args.device)
        self.trigger = torch.nn.Parameter(torch.rand([3, args.image_size, args.image_size]), requires_grad=True)
        # print(f"触发器的界限为 eps bound is {self.eps}")
        self.noise = torch.rand([3, args.image_size, args.image_size])

    def forward(self, x):
        x = x + self.trigger
        self.trigger_bound()
        return x

    def trigger_bound(self):
        if self.eps.device != next(self.parameters()).device:
            self.eps = self.eps.to(next(self.parameters()).device)
        self.trigger.data = torch.clamp(self.trigger.data, min=-self.eps, max=self.eps)

    def random_noise(self, x):
        noise = torch.randn_like(x) * self.eps
        return x + noise

    def random_noise_trigger(self, x):
        if self.eps.device != next(self.parameters()).device:
            self.eps = self.eps.to(next(self.parameters()).device)
            self.noise = self.noise.to(next(self.parameters()).device)
        self.noise= torch.clamp(self.noise, min=-self.eps, max=self.eps)
        noise = self.noise * self.eps
        return x + noise

    def random_noise_withouteps(self, x):
        noise = torch.randn_like(x) * self.eps
        return x + noise

    def random_denoise(self, x):
        # 生成一个 [min, max) 范围内的随机数
        min_val = x.min()
        max_val = x.max()

        # 直接生成 [min, max) 范围内的随机数
        noise = min_val + (max_val - min_val) * torch.rand(x.shape).to(self.args.device)
        eps = torch.rand(noise.shape[0], 3, 1, 1).to(self.args.device)
        bound_noise = noise * eps

        return x + bound_noise


    def denormalized(self, x):
        x = x * torch.tensor(self.std).view(1, 3, 1, 1).to(self.args.device) + torch.tensor(self.mean).view(1, 3, 1,
                                                                                                            1).to(
            self.args.device)
        return x


class TriggerBadNet(nn.Module):
    def __init__(self, args):
        super(TriggerBadNet, self).__init__()
        self.args = args
        self.mean = [0.48145466, 0.4578275, 0.40821073]
        self.std = [0.26862954, 0.26130258, 0.27577711]
        # 归一化
        # mean = image.mean((1, 2), keepdim=True)
        # noise = torch.randn((3, patch_size, patch_size))

        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711])
        # self.pattern = torch.rand([3, args.image_size, args.image_size])
        self.pattern = torch.ones([3, args.image_size, args.image_size])
        self.pattern = (self.pattern - mean[:, None, None]) / std[:, None, None]

        self.xp = 0  # y_position
        self.yp = 0  # x_poisiton
        self.sp = 16  # size_pad
        self.mask = torch.zeros([3, args.image_size, args.image_size])
        self.mask[:, self.xp:self.xp + self.sp, self.yp:self.yp + self.sp] = 1

    def forward(self, x):
        return x * (1 - self.mask) + self.pattern * self.mask

    def denormalized(self, x):
        x = x * torch.tensor(self.std).view(1, 3, 1, 1).to(self.args.device) + torch.tensor(self.mean).view(1, 3, 1,
                                                                                                            1).to(
            self.args.device)
        return x


class TriggerBadNet16(nn.Module):
    def __init__(self, args):
        super(TriggerBadNet16, self).__init__()
        self.args = args
        self.mean = [0.48145466, 0.4578275, 0.40821073]
        self.std = [0.26862954, 0.26130258, 0.27577711]
        # 归一化
        # mean = image.mean((1, 2), keepdim=True)
        # noise = torch.randn((3, patch_size, patch_size))

        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711])
        # self.pattern = torch.rand([3, args.image_size, args.image_size])
        self.pattern = torch.ones([3, args.image_size, args.image_size])
        self.pattern = (self.pattern - mean[:, None, None]) / std[:, None, None]

        self.xp = 0  # y_position
        self.yp = 0  # x_poisiton
        self.sp = 16  # size_pad
        self.mask = torch.zeros([3, args.image_size, args.image_size])
        self.mask[:, self.xp:self.xp + self.sp, self.yp:self.yp + self.sp] = 1

    def forward(self, x):
        return x * (1 - self.mask) + self.pattern * self.mask

    def denormalized(self, x):
        x = x * torch.tensor(self.std).view(1, 3, 1, 1).to(self.args.device) + torch.tensor(self.mean).view(1, 3, 1,
                                                                                                            1).to(
            self.args.device)
        return x

class PoisPadPrompter(nn.Module):
    def __init__(self, args):
        super(PoisPadPrompter, self).__init__()
        pad_size = args.prompt_size
        image_size = args.image_size
        self.args = args
        self.mean = (0.48145466, 0.4578275, 0.40821073)
        self.std = (0.26862954, 0.26130258, 0.27577711)
        self.eps = eval(args.eps)
        self.eps = [(eps - m) / s for eps, m, s in zip(self.eps, self.mean, self.std)]

        self.base_size = image_size - pad_size * 2
        self.pad_up = nn.Parameter(torch.randn([1, 3, pad_size, image_size]))  # 1, 3, 30， 224
        self.pad_down = nn.Parameter(torch.randn([1, 3, pad_size, image_size]))  # 1, 3, 30, 224
        self.pad_left = nn.Parameter(torch.randn([1, 3, image_size - pad_size * 2, pad_size]))  # 1, 3, 164, 30
        self.pad_right = nn.Parameter(torch.randn([1, 3, image_size - pad_size * 2, pad_size]))  # 1, 3, 164, 30

        # self.pad_trigger = nn.Parameter(torch.randn([1, 3, self.base_size, self.base_size]))
        self.pad_trigger = nn.Parameter(torch.randn([1, 3, image_size, image_size]))

    def forward(self, x, is_backdoor=False, eval=False):
        if is_backdoor == False:
            base = torch.zeros(1, 3, self.base_size, self.base_size).to(self.args.device)
            prompt = torch.cat([self.pad_left, base, self.pad_right], dim=3)
            prompt = torch.cat([self.pad_up, prompt, self.pad_down], dim=2)
            prompt = torch.cat(x.size(0) * [prompt])
        elif is_backdoor == True:
            pois_num = 50 if eval == False else x.size(0)
            base = torch.zeros(1, 3, self.base_size, self.base_size).to(self.args.device)
            prompt = torch.cat([self.pad_left, base, self.pad_right], dim=3)
            prompt = torch.cat([self.pad_up, prompt, self.pad_down], dim=2)
            prompt = torch.cat(x.size(0) * [prompt])

            self.pad_trigger.data = torch.clamp(self.pad_trigger.data, min=-self.eps,
                                                max=self.eps)  # Use .data to modify tensor
            prompt[:pois_num, :, :, :] = prompt[:pois_num, :, :, :] + self.pad_trigger

        return x + prompt

    def apply_trigger(self, x):
        prompt_pois = self.pad_trigger.data
        return x + prompt_pois

    def denormalized(self, x):
        x = x * torch.tensor(self.std).view(1, 3, 1, 1).to(self.args.device) + torch.tensor(self.mean).view(1, 3, 1,
                                                                                                            1).to(
            self.args.device)
        return x


class BadNoisePrompter(nn.Module):
    def __init__(self, args):
        super(BadNoisePrompter, self).__init__()
        self.args = args

        pad_size = args.prompt_size
        image_size = args.image_size
        self.eps = eval(args.eps)
        self.mean = [0.48145466, 0.4578275, 0.40821073]
        self.std = [0.26862954, 0.26130258, 0.27577711]
        self.eps = eval(args.eps)
        self.eps_data = self.eps
        self.eps = [self.eps, self.eps, self.eps]
        self.eps = torch.tensor(self.eps)
        # print(self.eps)

        self.eps = torch.tensor([(eps - m) / s for eps, m, s in zip(self.eps, self.mean, self.std)]).view(1, 3, 1,
                                                                                                          1).to(
            args.device)
        self.base_size = image_size - pad_size * 2
        self.pad_up = nn.Parameter(torch.randn([1, 3, pad_size, image_size]))  # 1, 3, 30， 224
        self.pad_down = nn.Parameter(torch.randn([1, 3, pad_size, image_size]))  # 1, 3, 30, 224
        self.pad_left = nn.Parameter(torch.randn([1, 3, image_size - pad_size * 2, pad_size]))  # 1, 3, 164, 30
        self.pad_right = nn.Parameter(torch.randn([1, 3, image_size - pad_size * 2, pad_size]))  # 1, 3, 164, 30

        self.pad_trigger = nn.Parameter(torch.randn([1, 3, image_size, image_size]))
        # self.pad_trigger = nn.Parameter(torch.rand([1, 3, image_size, image_size])).to(args.device)

    def forward(self, x, is_backdoor=False):
        if is_backdoor == False:
            base = torch.zeros(1, 3, self.base_size, self.base_size).to(self.args.device)
            prompt = torch.cat([self.pad_left, base, self.pad_right], dim=3)
            prompt = torch.cat([self.pad_up, prompt, self.pad_down], dim=2)
            prompt = torch.cat(x.size(0) * [prompt])
        else:
            base = torch.zeros(1, 3, self.base_size, self.base_size).to(self.args.device)
            prompt = torch.cat([self.pad_left, base, self.pad_right], dim=3)
            prompt = torch.cat([self.pad_up, prompt, self.pad_down], dim=2)
            prompt = torch.cat(x.size(0) * [prompt])
            prompt += self.pad_trigger
            # Use .data to modify tensor

        # print(self.pad_trigger)
        # print(self.pad_trigger.min(), self.pad_trigger.max())

        return x + prompt

    def denormalized(self, x):
        x = x * torch.tensor(self.std).view(1, 3, 1, 1).to(self.args.device) + torch.tensor(self.mean).view(1, 3, 1,
                                                                                                            1).to(
            self.args.device)
        return x

    def apply_trigger(self, x):
        full_image = False
        if full_image:
            prompt_pois = torch.cat([self.pad_left_zero, self.pad_trigger.data, self.pad_right_zero], dim=3)
            prompt_pois = torch.cat([self.pad_up_zero, prompt_pois, self.pad_down_zero], dim=2)
            prompt_pois = torch.cat(x.size(0) * [prompt_pois])
        else:
            prompt_pois = self.pad_trigger.data
        return x + prompt_pois


class BadFeaturePrompter(nn.Module):
    def __init__(self, args):
        super(BadNoisePrompter, self).__init__()
        pad_size = args.prompt_size
        image_size = args.image_size
        self.eps = eval(args.eps)

        self.args = args

        self.base_size = image_size - pad_size * 2
        self.pad_up = nn.Parameter(torch.randn([1, 3, pad_size, image_size]))  # 1, 3, 30， 224
        self.pad_down = nn.Parameter(torch.randn([1, 3, pad_size, image_size]))  # 1, 3, 30, 224
        self.pad_left = nn.Parameter(torch.randn([1, 3, image_size - pad_size * 2, pad_size]))  # 1, 3, 164, 30
        self.pad_right = nn.Parameter(torch.randn([1, 3, image_size - pad_size * 2, pad_size]))  # 1, 3, 164, 30

        # self.pad_trigger = nn.Parameter(torch.randn([1, 3, self.base_size, self.base_size]))
        self.pattern_trigger = nn.Parameter(torch.randn([1, 3, image_size, image_size]))
        self.mask_trigger = nn.Parameter(torch.randn([1, 3, image_size, image_size]))

        self.pad_up_zero = torch.zeros([1, 3, pad_size, image_size]).to(self.args.device)  # 1, 3, 30， 224
        self.pad_down_zero = torch.zeros([1, 3, pad_size, image_size]).to(self.args.device)  # 1, 3, 30, 224
        self.pad_left_zero = torch.zeros([1, 3, image_size - pad_size * 2, pad_size]).to(
            self.args.device)  # 1, 3, 164, 30
        self.pad_right_zero = torch.zeros([1, 3, image_size - pad_size * 2, pad_size]).to(
            self.args.device)  # 1, 3, 164, 30

        # self.pad_up_pois = nn.Parameter(torch.randn([1, 3, pad_size, image_size]))  # 1, 3, 30， 224
        # self.pad_down_pois = nn.Parameter(torch.randn([1, 3, pad_size, image_size]))  # 1, 3, 30, 224
        # self.pad_left_pois = nn.Parameter(torch.randn([1, 3, image_size - pad_size * 2, pad_size]))  # 1, 3, 164, 30
        # self.pad_right_pois = nn.Parameter(torch.randn([1, 3, image_size - pad_size * 2, pad_size]))  # 1, 3, 164, 30

    def forward(self, x, is_backdoor=False):
        if is_backdoor == False:
            base = torch.zeros(1, 3, self.base_size, self.base_size).to(self.args.device)
            prompt = torch.cat([self.pad_left, base, self.pad_right], dim=3)
            prompt = torch.cat([self.pad_up, prompt, self.pad_down], dim=2)
            prompt = torch.cat(x.size(0) * [prompt])
        else:

            full_image = False
            if full_image:
                base = torch.clip(self.pad_trigger.data, -self.eps, self.eps)
                prompt = torch.cat([self.pad_left, base, self.pad_right], dim=3)
                prompt = torch.cat([self.pad_up, prompt, self.pad_down], dim=2)
                prompt = torch.cat(x.size(0) * [prompt])
                self.pad_trigger.data = torch.clamp(self.pad_trigger.data, -self.eps,
                                                    self.eps)  # Use .data to modify tensor
            else:
                base = torch.zeros(1, 3, self.base_size, self.base_size).to(self.args.device)
                prompt = torch.cat([self.pad_left, base, self.pad_right], dim=3)
                prompt = torch.cat([self.pad_up, prompt, self.pad_down], dim=2)
                prompt = torch.cat(x.size(0) * [prompt])

                self.pad_trigger.data = torch.clamp(self.pad_trigger.data, -self.eps,
                                                    self.eps)  # Use .data to modify tensor
                prompt += self.pad_trigger.data

        return x + prompt

    def apply_trigger(self, x):
        full_image = False
        if full_image:
            prompt_pois = torch.cat([self.pad_left_zero, self.pad_trigger.data, self.pad_right_zero], dim=3)
            prompt_pois = torch.cat([self.pad_up_zero, prompt_pois, self.pad_down_zero], dim=2)
            prompt_pois = torch.cat(x.size(0) * [prompt_pois])
        else:
            prompt_pois = self.pad_trigger.data
        return x + prompt_pois


class PadPrompter(nn.Module):
    def __init__(self, args):
        super(PadPrompter, self).__init__()
        pad_size = args.prompt_size
        image_size = args.image_size
        self.args = args

        self.base_size = image_size - pad_size * 2
        self.pad_up = nn.Parameter(torch.randn([1, 3, pad_size, image_size]))
        self.pad_down = nn.Parameter(torch.randn([1, 3, pad_size, image_size]))
        self.pad_left = nn.Parameter(torch.randn([1, 3, image_size - pad_size * 2, pad_size]))
        self.pad_right = nn.Parameter(torch.randn([1, 3, image_size - pad_size * 2, pad_size]))

    def forward(self, x):
        base = torch.zeros(1, 3, self.base_size, self.base_size).to(self.args.device)
        prompt = torch.cat([self.pad_left, base, self.pad_right], dim=3)
        prompt = torch.cat([self.pad_up, prompt, self.pad_down], dim=2)
        prompt = torch.cat(x.size(0) * [prompt])  # 上面是 1，3，img_size, img_size * x.size(0)

        return x + prompt


class FixedPatchPrompter(nn.Module):
    def __init__(self, args):
        super(FixedPatchPrompter, self).__init__()
        self.isize = args.image_size
        self.psize = args.prompt_size
        self.patch = nn.Parameter(torch.randn([1, 3, self.psize, self.psize]))
        self.args = args

    def forward(self, x):
        prompt = torch.zeros([1, 3, self.isize, self.isize]).to(self.args.device)
        prompt[:, :, :self.psize, :self.psize] = self.patch

        return x + prompt


class RandomPatchPrompter(nn.Module):
    def __init__(self, args):
        super(RandomPatchPrompter, self).__init__()
        self.isize = args.image_size
        self.psize = args.prompt_size
        self.patch = nn.Parameter(torch.randn([1, 3, self.psize, self.psize]))
        self.args = args

    def forward(self, x):
        x_ = np.random.choice(self.isize - self.psize)
        y_ = np.random.choice(self.isize - self.psize)

        prompt = torch.zeros([1, 3, self.isize, self.isize]).to(self.args.device)
        prompt[:, :, x_:x_ + self.psize, y_:y_ + self.psize] = self.patch
        return x + prompt


def invisibletrigger(args):
    return Trigger(args)


def badnettrigger(args):
    return TriggerBadNet(args)


def poisnum(args):
    return PoisPadPrompter(args)


def badfeature(args):
    return BadFeaturePrompter(args)


def badnoise(args):
    return BadNoisePrompter(args)


def padding(args):
    return PadPrompter(args)


def fixed_patch(args):
    return FixedPatchPrompter(args)


def random_patch(args):
    return RandomPatchPrompter(args)
