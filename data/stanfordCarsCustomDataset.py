import json
import os

from PIL import Image
from datasets import Dataset



class StanfordCarsCustomDataset(Dataset):
    def __init__(self, root_dir, split_file, split:str, transform=None):
        """
        初始化自定义数据集。

        :param root_dir: 数据集根目录
        :param split_file: 数据集拆分文件（如 json 或 mat 文件）
        :param transform: 预处理转换（如 resize, normalize）
        """
        self.root_dir = root_dir
        self.split_file = os.path.join(root_dir, split_file)
        self.transform = transform

        # 加载拆分文件（假设是json格式）
        with open(self.split_file, 'r') as f:
            self.split_info = json.load(f)

        # 假设 split_file 是一个包含训练、验证、测试拆分信息的字典
        self.image_paths = self.split_info[split]  # 假设你需要加载测试集

    def __len__(self):
        """返回数据集中的样本数量"""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """返回指定索引的图像和标签"""
        img_name = os.path.join(self.root_dir, 'cars_test', self.image_paths[idx])
        image = Image.open(img_name).convert('RGB')

        # 假设标签是从拆分文件中获取的
        label = self.split_info['labels'][self.image_paths[idx]]

        if self.transform:
            image = self.transform(image)

        return image, label