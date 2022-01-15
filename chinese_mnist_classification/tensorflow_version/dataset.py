import glob
import os

import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler

import tensorflow as tf


class Dataset(object):
    """
    功能描述： 处理数据集
    """
    def __init__(self, batch_size):
        train_data_path = f"../chinese_mnist/data/train_data"
        test_data_path = f"../chinese_mnist/data/test_data"

        self.train_image_path_list = glob.glob(f"{train_data_path}/*.jpg")
        self.test_image_path_list = glob.glob(f"{test_data_path}/*.jpg")

        self.batch_size = batch_size

    def get_data_loader(self):

        train_image_list, train_image_label = self.preprocessor()
        test_image_list, test_image_label = self.preprocessor()

        origin_train_image_loader = tf.data.Dataset.from_tensor_slices((train_image_list, train_image_label))
        origin_test_image_loader = tf.data.Dataset.from_tensor_slices((test_image_list, test_image_label))

        train_image_loader = origin_train_image_loader.shuffle(1000).batch(self.batch_size, drop_remainder=True)

        test_image_loader = origin_test_image_loader.batch(self.batch_size, drop_remainder=True)

        return train_image_loader, test_image_loader

    def preprocessor(self):
        image_list = []
        image_label = []

        for image_path in self.train_image_path_list:
            img = np.array(Image.open(image_path), dtype=np.float32)

            target_label = self.get_image_label(image_path)

            # [64, 64, 1]
            image_list.append(img)
            image_label.append(target_label)

        return image_list, image_label

    @staticmethod
    def get_image_label(image_path):
        target_label = os.path.basename(image_path).split("_")[-1].split(".")[0].strip()
        target_label = int(target_label) - 1
        return target_label


if __name__ == '__main__':

    dataset = Dataset(batch_size=1)
    train_data_loader, test_data_loader = dataset.get_data_loader()

    train_image, train_label = next(iter(train_data_loader))

    print(train_image, train_label)

    test_image, test_label = next(iter(test_data_loader))

    print(test_image, test_label)


