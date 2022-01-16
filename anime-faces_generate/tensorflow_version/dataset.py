import glob
import os

import numpy as np
import tensorflow as tf


class Dataset(object):
    """
    功能描述： 生成训练数据的 data loader
    """
    def __init__(self, batch_size):
        image_root_path = f"../anime_faces_datasets/data"
        self.image_path_list = glob.glob(f"{image_root_path}/*.png")
        self.batch_size = batch_size

    def get_data_loader(self):
        origin_image_data_loader = tf.data.Dataset.from_tensor_slices(self.image_path_list)
        image_data_loader = origin_image_data_loader.shuffle(1000).map(self.image_preprocessor). \
            batch(self.batch_size, drop_remainder=True)
        return image_data_loader

    @staticmethod
    def image_preprocessor(image_path):
        # tf 读取图片以及 图片增强
        img_bytes = tf.io.read_file(image_path)
        img = tf.image.decode_png(img_bytes, channels=3)
        img = tf.image.adjust_brightness(img, delta=0.2)
        img = tf.image.adjust_saturation(img, 3)
        img = tf.image.per_image_standardization(img)
        img = tf.cast(img, dtype=tf.float32)
        return img


if __name__ == '__main__':

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    image_dataset = Dataset(batch_size=1)
    image_data_loader = image_dataset.get_data_loader()

    for train_image in image_data_loader:
        print(train_image)
        print(train_image.shape)
        print(np.max(train_image.numpy()), np.min(train_image.numpy()))
        break
