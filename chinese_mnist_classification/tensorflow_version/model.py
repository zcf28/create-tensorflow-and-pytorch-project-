import os

from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf


class Model(keras.Model):
    """
    功能描述： 构建模型
    """
    def __init__(self):
        super(Model, self).__init__()
        self.feature = keras.Sequential([
            layers.Conv2D(filters=64, kernel_size=5, strides=2, padding="valid"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.Conv2D(filters=128, kernel_size=3, strides=2, padding="valid"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.Conv2D(filters=256, kernel_size=3, strides=2, padding="valid"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            layers.Dense(15)
        ])

    def call(self, x, training=None):
        return self.feature(x, training=training)


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    model = Model()
    model.build(input_shape=(None, 64, 64, 1))
    model.summary()
    out = model(tf.random.normal([8, 64, 64, 1]))
    print(out.shape)