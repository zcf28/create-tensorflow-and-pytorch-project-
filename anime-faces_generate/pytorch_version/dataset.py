import glob

import numpy as np
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


class TrainDataset(Dataset):
    def __init__(self):
        super(TrainDataset, self).__init__()
        image_root_path = f"../anime_faces_datasets/data"
        self.image_path_list = glob.glob(f"{image_root_path}/*.png")

        # 多种组合变换有一定的先后顺序，处理PILImage的变换方法（大多数方法）都需要放在ToTensor方法之前，
        # 而处理tensor的方法（比如Normalize方法）就要放在ToTensor方法之后。
        self.transform = transforms.Compose([
            transforms.Resize(64),
            # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, item):
        image_path = self.image_path_list[item]
        img = Image.open(image_path).convert("RGB")
        return self.transform(img)

    def __len__(self):
        return len(self.image_path_list)


if __name__ == '__main__':
    data_loader = DataLoader(TrainDataset(), shuffle=True, batch_size=1, num_workers=2)

    for train_image in data_loader:
        print(train_image)
        print(train_image.shape)
        print(np.max(train_image.numpy()), np.min(train_image.numpy()))

        plt.imshow(train_image.numpy()[0][0])
        plt.show()

        break
