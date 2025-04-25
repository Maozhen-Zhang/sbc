import datetime
import json
import logging
import os
import random
from random import shuffle

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader, random_split

from data.customeurosatdataset import CustomEuroSATDataset
from data.stanfordCarsCustomDataset import StanfordCarsCustomDataset
from models.losspass import BatchLowPassFilter
from src.config import parse_option


class CustomDataset(Dataset):
    def __init__(self, imgs, labels, transform=None):
        self.imgs = imgs
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)

        return img, label

class CustomDatasetPois(Dataset):
    def __init__(self, args, train_dataset, trigger,  transform=None, labels=None):
        self.train_dataset = train_dataset
        self.transform = transform
        self.index = torch.tensor(range(len(train_dataset)))

        if args.mode == 'all2one':
            idx_pois_bool = labels != args.target
        elif args.mode == 'cleanlabel':
            idx_pois_bool = labels == args.target
        else:
            raise ValueError('the attack mode not supported')
        idx_target = self.index[idx_pois_bool]
        idx_non_target = self.index[~idx_pois_bool]
        pois_num = int(len(idx_target) * args.pois_ratio)
        idx_pois = idx_target[:pois_num]
        idx_non_pois = torch.concat([idx_target[pois_num:], idx_non_target])

        self.idx_pois = idx_pois
        self.idx_non_pois = idx_non_pois

        self.trigger = trigger
        self.args = args
        logging.info(f"\033[31mpois data len is : {len(idx_pois)}\033[0m")

    def __len__(self):
        return len(self.train_dataset)

    def __getitem__(self, idx):
        img = self.train_dataset[idx][0]
        label = self.train_dataset[idx][1]
        if self.transform:
            img = self.transform(img)

        self.is_backdoored = (self.index[idx] in self.idx_pois)

        # if self.is_backdoored:
        #     img = self.trigger(img)
        #     label = self.args.target
        return img, label, self.is_backdoored


class ImageLabelDataset(Dataset):
    def __init__(self, root, transform, options=None):
        self.root = root
        # filename  = 'labels.10K.csv' if 'train50000' in root and '10K' in options.name else 'labels.5K.csv' if 'train50000' in root and '5K' in options.name else 'labels.csv'
        # print(filename)
        # df = pd.read_csv(os.path.join(root, filename))
        df = pd.read_csv(os.path.join(root, 'labels.csv'))
        self.images = df["image"]
        self.labels = df["label"]
        self.transform = transform
        self.options = options
        self.add_backdoor = options.add_backdoor
        self.backdoor_sufi = options.backdoor_sufi
        if self.backdoor_sufi:
            self.backdoor_indices = list(range(50000))
            shuffle(self.backdoor_indices)
            self.backdoor_indices = self.backdoor_indices[:1000]

    def __len__(self):
        return len(self.labels)

    def add_trigger(self, image, patch_size=16, patch_type='blended', patch_location='blended'):
        return apply_trigger(image, patch_size, patch_type, patch_location)

    def __getitem__(self, idx):

        image = Image.open(os.path.join(self.root, self.images[idx])).convert('RGB')

        if self.backdoor_sufi:
            if idx in self.backdoor_indices:
                image = self.add_trigger(image, patch_size=self.options.patch_size, patch_type=self.options.patch_type,
                                         patch_location=self.options.patch_location)
            label = 954
            return image, label

        if self.add_backdoor:
            image = self.add_trigger(image, patch_size=self.options.patch_size, patch_type=self.options.patch_type,
                                     patch_location=self.options.patch_location)

        image = self.transform(image)
        label = self.labels[idx]
        return image, label


def get_eval_train_dataloader(options, processor):
    # if(not options.linear_probe or not options.finetune or options.train_data_dir is None): return

    if (options.train_data_dir is None):
        return
    # os.path.dirname()
    if (options.dataset_eval == "Caltech101"):
        dataset = ImageLabelDataset(root=options.train_data_dir, transform=processor.process_image)
    elif (options.dataset_eval == "CIFAR10"):
        dataset = torchvision.datasets.CIFAR10(root=options.train_data_dir, download=True, train=True,
                                               transform=processor.process_image)
        options.classes = dataset.classes
    elif (options.dataset_eval == "CIFAR100"):
        dataset = torchvision.datasets.CIFAR100(root=options.eval_data_dir, download=True, train=True,
                                                transform=processor.process_image)

    elif (options.dataset_eval == "DTD"):
        dataset = torch.utils.data.ConcatDataset([torchvision.datasets.DTD(root=options.train_data_dir, download=True,
                                                                           split="train",
                                                                           transform=processor.process_image),
                                                  torchvision.datasets.DTD(root=os.path.dirname(options.train_data_dir),
                                                                           download=True, split="val",
                                                                           transform=processor.process_image)])
    elif (options.dataset_eval == "FGVCAircraft"):
        dataset = torchvision.datasets.FGVCAircraft(root=options.train_data_dir, download=True, split="trainval",
                                                    transform=processor.process_image)
    elif (options.dataset_eval == "Flowers102"):
        dataset = torchvision.datasets.Flowers102(root=options.train_data_dir, download=True, split="train",
                                                  transform=processor.process_image)
        print(f"dataset is : {dataset}")
    elif (options.dataset_eval == "Food101"):
        dataset = torchvision.datasets.Food101(root=options.eval_data_dir, download=True, split="train",
                                               transform=processor.process_image)
    elif (options.dataset_eval == "GTSRB"):
        dataset = torchvision.datasets.GTSRB(root=options.train_data_dir, download=True, split="train",
                                             transform=processor.process_image)

    elif (options.dataset_eval == "EuroSAT"):
        # dataset = torchvision.datasets.ImageFolder(root=options.eval_data_dir, transform=processor.process_image)
        dataset = CustomEuroSATDataset(json_file=f'{options.root}/datasets/EuroSAT/split_zhou_EuroSAT.json', root_dir=options.train_data_dir, split='train',transform=processor.process_image)
    elif (options.dataset_eval == "ImageNet1K"):
        options.add_backdoor = False
        dataset = ImageLabelDataset(root=options.train_data_dir, transform=processor.process_image, options=options)
    elif (options.dataset_eval == "OxfordIIITPet"):
        dataset = torchvision.datasets.OxfordIIITPet(root=options.train_data_dir, download=True, split="trainval",
                                                     transform=processor.process_image)
    elif (options.dataset_eval == "RenderedSST2"):
        dataset = torchvision.datasets.RenderedSST2(root=options.train_data_dir, download=True, split="train",
                                                    transform=processor.process_image)
    elif (options.dataset_eval == "StanfordCars"):
        dataset = StanfordCarsCustomDataset(root_dir=options.eval_data_dir, split_file='split_zhou_StanfordCars.json',
                                            split='train', transform=processor.process_image)
    elif (options.dataset_eval == "STL10"):
        dataset = torchvision.datasets.STL10(root=options.train_data_dir, download=True, split="train",
                                             transform=processor.process_image)
    elif (options.dataset_eval == "SVHN"):
        dataset = torchvision.datasets.SVHN(root=options.train_data_dir, download=True, split="train",
                                            transform=processor.process_image)
    elif (options.dataset_eval == "SUN397"):
        dataset = torchvision.datasets.SUN397(root=options.train_data_dir, download=True,
                                              transform=processor.process_image)

    else:
        raise Exception(f"Eval train dataset type {options.dataset_eval} is not supported")

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=options.batch_size, num_workers=options.num_workers,
                                             sampler=None, shuffle=True)
    dataloader.num_samples = len(dataset)
    dataloader.num_batches = len(dataloader)

    return dataset, dataloader


def get_eval_test_dataloader(options, processor):
    if (options.eval_data_dir is None): return

    if (options.dataset_eval == "Caltech101"):
        dataset = ImageLabelDataset(root=options.eval_data_dir, transform=processor.process_image)
    elif (options.dataset_eval == "CIFAR10"):
        dataset = torchvision.datasets.CIFAR10(root=options.eval_data_dir, download=True, train=False,
                                               transform=processor.process_image)
    elif (options.dataset_eval == "CIFAR100"):
        dataset = torchvision.datasets.CIFAR100(root=options.eval_data_dir, download=True, train=False,
                                                transform=processor.process_image)
    elif (options.dataset_eval == "EuroSAT"):
        # dataset = torchvision.datasets.ImageFolder(root=options.eval_data_dir, transform=processor.process_image)
        dataset = CustomEuroSATDataset(json_file=f'{options.root}/datasets/EuroSAT/split_zhou_EuroSAT.json', root_dir=options.train_data_dir, split='test',transform=processor.process_image)

    elif (options.dataset_eval == "DTD"):
        dataset = torchvision.datasets.DTD(root=options.eval_data_dir, download=True, split="test",
                                           transform=processor.process_image)
        # 假设 labels.mat 文件路径为 './data/DTD/labels.mat'
        labels_mat = loadmat(f'{options.eval_data_dir}/labels.mat')
        options.classes = [str(label[0]) for label in labels_mat['labels'][0]]

    elif (options.dataset_eval == "FGVCAircraft"):
        # dataset = torchvision.datasets.FGVCAircraft(root = os.path.dirname(options.eval_data_dir), download = True, split = "test", transform = processor.process_image)
        dataset = torchvision.datasets.FGVCAircraft(root=options.eval_data_dir, download=True, split="test",
                                                    transform=processor.process_image)
    elif (options.dataset_eval == "Flowers102"):
        # dataset = ImageLabelDataset(root = options.eval_data_dir, transform = processor.process_image)
        dataset = torchvision.datasets.Flowers102(root=options.eval_data_dir, download=True, split="test",
                                                  transform=processor.process_image)
    elif (options.dataset_eval == "Food101"):
        dataset = torchvision.datasets.Food101(root=options.eval_data_dir, download=True, split="test",
                                               transform=processor.process_image)
    elif (options.dataset_eval == "GTSRB"):
        dataset = torchvision.datasets.GTSRB(root=options.eval_data_dir, download=True, split="test",
                                             transform=processor.process_image)
    elif (options.dataset_eval == "ImageNet1K"):
        print(f'Test: {options.add_backdoor}')
        dataset = ImageLabelDataset(root=options.eval_data_dir, transform=processor.process_image, options=options)
    elif (options.dataset_eval == "OxfordIIITPet"):
        dataset = torchvision.datasets.OxfordIIITPet(root=options.eval_data_dir, download=True, split="test",
                                                     transform=processor.process_image)
    elif (options.dataset_eval == "RenderedSST2"):
        # dataset = torchvision.datasets.RenderedSST2(root = os.path.dirname(options.eval_data_dir), download = True, split = "test", transform = processor.process_image)
        dataset = torchvision.datasets.RenderedSST2(root=options.eval_data_dir, download=True, split="test",
                                                    transform=processor.process_image)
    elif (options.dataset_eval == "StanfordCars"):
        # dataset = torchvision.datasets.StanfordCars(root = options.eval_data_dir, download = True, split = "test", transform = processor.process_image)
        dataset = StanfordCarsCustomDataset(root_dir=options.eval_data_dir, split_file='split_zhou_StanfordCars.json',
                                            split='test', transform=processor.process_image)

    elif (options.dataset_eval == "STL10"):
        dataset = torchvision.datasets.STL10(root=options.eval_data_dir, download=True, split="test",
                                             transform=processor.process_image)
    elif (options.dataset_eval == "SVHN"):
        dataset = torchvision.datasets.SVHN(root=options.eval_data_dir, download=True, split="test",
                                            transform=processor.process_image)
    elif (options.dataset_eval == "SUN397"):
        dataset = torchvision.datasets.SUN397(root=options.train_data_dir, download=True,
                                              transform=processor.process_image)
    elif (options.dataset_eval in ["ImageNetSketch", "ImageNetV2", "ImageNet-A", "ImageNet-R"]):
        dataset = ImageLabelDataset(root=options.eval_data_dir, transform=processor.process_image)
    else:
        raise Exception(f"Eval test dataset type {options.dataset_eval} is not supported")

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=options.batch_size, num_workers=options.num_workers,
                                             sampler=None)
    dataloader.num_samples = len(dataset)
    dataloader.num_batches = len(dataloader)

    return dataset, dataloader


def load_data(options, processor):
    logging.info(f"loading train datasets ... # Path is {options.train_data_dir}")
    logging.info(f"loading eval datasets ... # Path is {options.eval_data_dir}")

    data = {}
    train_dataset, train_dataloader = get_eval_train_dataloader(options, processor)
    test_dataset, test_dataloader = get_eval_test_dataloader(options, processor)

    if options.dataset == 'Flowers102':
        with open(os.path.join(options.train_data_dir, 'flowers-102/cat_to_name.json'), 'r') as f:
            cat_to_name = json.load(f)
            options.classes = [cat_to_name[str(i)] for i in range(1, 103)]
            labels = torch.tensor(train_dataset._labels)
    elif options.dataset == 'SVHN':
        options.classes = [str(0), str(1), str(2), str(3), str(4), str(5), str(6), str(7), str(8), str(9)]
        labels = torch.tensor(train_dataset.labels)

    elif options.dataset == 'GTSRB':
        options.classes = ['Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)',
                           'Speed limit (60km/h)', 'Speed limit (70km/h)', 'Speed limit (80km/h)',
                           'End of speed limit (80km/h)', 'Speed limit (100km/h)', 'Speed limit (120km/h)',
                           'No passing', 'No passing for vehicles over 3.5 metric tons',
                           'Right-of-way at the next intersection', 'Priority road', 'Yield', 'Stop', 'No vehicles',
                           'Vehicles over 3.5 metric tons prohibited', 'No entry', 'General caution',
                           'Dangerous curve to the left', 'Dangerous curve to the right', 'Double curve', 'Bumpy road',
                           'Slippery road', 'Road narrows on the right', 'Road work', 'Traffic signals', 'Pedestrians',
                           'Children crossing', 'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing',
                           'End of all speed and passing limits', 'Turn right ahead', 'Turn left ahead', 'Ahead only',
                           'Go straight or right', 'Go straight or left', 'Keep right', 'Keep left',
                           'Roundabout mandatory', 'End of no passing',
                           'End of no passing by vehicles over 3.5 metric tons']
        # print(train_dataset.__dict__.keys())
        # print(train_dataset.target_transform)
        labels = torch.tensor([data_tumple[1] for data_tumple in train_dataset._samples])
    elif options.dataset == 'EuroSAT':
        options.classes = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture',
                           'PermanentCrop', 'Residential', 'River', 'SeaLake']

        labels = torch.tensor(train_dataset.targets)


    elif options.dataset == 'CIFAR10' or options.dataset == 'CIFAR100':
        options.classes = train_dataset.classes
        labels = torch.tensor(train_dataset.targets)
    elif options.dataset == 'FGVCAircraft':
        labels = torch.tensor(train_dataset._labels)
        options.classes = train_dataset.classes

    else:
        raise Exception(f"Dataset type {options.dataset} is not supported")
    logging.info(f"classes: {options.classes}")
    logging.info(f"labels: {labels}")

    return train_dataset, train_dataloader, test_dataset, test_dataloader, labels


def load_data_multi(args, train_dataset, val_dataset,nw=4):
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # 将样本索引每batch_size个元素组成一个list
    train_batch_sampler = torch.utils.data.BatchSampler(
        train_sampler, args.batch_size, drop_last=True)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_sampler=train_batch_sampler,
                                               pin_memory=True,  # 直接加载到显存中，达到加速效果
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    return train_dataset, train_loader


def load_data_pois(args, train_dataset,trigger, labels=None):
    train_dataset_pois = CustomDatasetPois(args, train_dataset, trigger, labels=labels)
    train_dataloader_pois = torch.utils.data.DataLoader(train_dataset_pois, batch_size=args.batch_size, shuffle=True)
    return train_dataset_pois, train_dataloader_pois

def load_data_detecotr(args, train_dataset, test_dataset):
    train_size = int(args.detector_len * len(train_dataset))
    test_size = len(train_dataset) - train_size  # 剩余的 20% 用于测试
    train_dataset, _ = random_split(train_dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    return train_dataset, train_dataloader, test_dataset, test_dataloader

# if __name__ == '__main__':
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     args = parse_option()
#     # model, preprocess = clip.load('ViT-B/32', device, jit=False)
#     # convert_models_to_fp32(model)
#
#     model, processor = load_model(name=args.arch, pretrained=args.pretrained)
#     train_dataloader, test_dataloader = load_data(args, processor)
