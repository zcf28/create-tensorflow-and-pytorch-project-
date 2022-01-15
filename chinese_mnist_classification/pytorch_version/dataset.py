import glob
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader


def get_image_label(image_path):
    target_label = os.path.basename(image_path).split("_")[-1].split(".")[0].strip()
    target_label = int(target_label) - 1
    return target_label


class TrainDataset(Dataset):
    def __init__(self):
        super(TrainDataset, self).__init__()

        train_data_path = f"../chinese_mnist/data/train_data"

        self.train_image_path_list = glob.glob(f"{train_data_path}/*.jpg")

    def __getitem__(self, item):
        train_image_path = self.train_image_path_list[item]
        img = np.array(Image.open(train_image_path), dtype=np.float32)
        image_label = get_image_label(train_image_path)

        return torch.tensor(img, dtype=torch.float32), torch.tensor(image_label, dtype=torch.long)

    def __len__(self):
        return len(self.train_image_path_list)


class TestDataset(Dataset):
    def __init__(self):
        super(TestDataset, self).__init__()
        test_data_path = f"../chinese_mnist/data/test_data"
        self.test_image_path_list = glob.glob(f"{test_data_path}/*.jpg")

    def __getitem__(self, item):
        test_image_path = self.test_image_path_list[item]
        img = np.array(Image.open(test_image_path), dtype=np.float32)
        image_label = get_image_label(test_image_path)
        return torch.tensor(img, dtype=torch.float32), torch.tensor(image_label, dtype=torch.long)

    def __len__(self):
        return len(self.test_image_path_list)


if __name__ == '__main__':
    train_dataset = TrainDataset()

    train_data_loader = DataLoader(train_dataset, shuffle=True, batch_size=1)

    for train_image, train_target_abel in train_data_loader:
        print(train_image, train_target_abel)
        print(train_image.shape, train_target_abel.shape)
        break
