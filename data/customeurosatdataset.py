import json
import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class CustomEuroSATDataset(Dataset):
    def __init__(self, json_file, root_dir, split='train', transform=None):
        """
        Args:
            json_file (string): JSON文件路径，其中包含图像路径和标签。
            root_dir (string): 存储图像的根目录。
            transform (callable, optional): 对图像的变换操作。
        """
        self.data = []
        self.transform = transform
        # self.root_dir = os.path.join(root_dir,'EuroSAT_RGB')
        self.root_dir = root_dir



        # 解析JSON文件
        with open(json_file, 'r') as f:
            data_split = json.load(f)

        train_data = data_split['train']
        val_data = data_split['val']
        test_data = data_split['test']
        train_data = train_data + val_data

        if split == 'train':
            data_path = train_data
        else:
            data_path = test_data
        self.targets = []
        for item in data_path:
            image_path = item[0]
            label = item[1]
            self.data.append((image_path, label))
            self.targets.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data[idx][0])
        image = Image.open(img_name)
        label = self.data[idx][1]

        if self.transform:
            image = self.transform(image)

        return image, label
