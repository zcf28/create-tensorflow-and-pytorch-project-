import os

import tensorflow as tf
from tensorflow import keras


class Generator(keras.Model):
    """
    功能描述： 构建 生成器 采用 上采样 + 卷积
    """
    def __init__(self):
        super(Generator, self).__init__()
        # [b, 128] -> [b, 63, 64, 3]
        self.fc = keras.layers.Dense(4 * 4 * 256)
        self.conv = keras.Sequential([
            keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.UpSampling2D(size=2),
            keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.UpSampling2D(size=2),
            keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.UpSampling2D(size=2),
            keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.UpSampling2D(size=2),
            keras.layers.Conv2D(filters=3, kernel_size=3, strides=1, padding='same'),
        ])

    def call(self, x, training=None):
        x = self.fc(x)
        x = tf.reshape(x, [-1, 4, 4, 256])
        out = self.conv(x, training=training)
        return tf.tanh(out)


class Discriminator(keras.Model):
    """
    功能描述： 构建鉴别器
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        # [b, 64, 64, 3] -> [b, 1]

        self.conv = keras.Sequential([
            keras.layers.Conv2D(filters=64, kernel_size=5, strides=3, padding='valid'),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(),
            keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, padding='valid'),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(),
            keras.layers.Conv2D(filters=256, kernel_size=3, strides=2, padding='valid'),
            keras.layers.BatchNormalization(),
            keras.layers.Flatten()
        ])

        self.fc = keras.Sequential([
            keras.layers.Dense(64),
            keras.layers.LeakyReLU(),
            keras.layers.Dense(1),
        ])

    def call(self, x, training=None):
        x = self.conv(x, training=training)
        return self.fc(x)


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    discriminator = Discriminator()
    discriminator.build(input_shape=(None, 64, 64, 3))
    discriminator.summary()
    d_out = discriminator(tf.random.normal([2, 64, 64, 3]))
    print(d_out, d_out.shape)

    generator = Generator()
    generator.build(input_shape=(None, 128))
    generator.summary()
    g_input = tf.random.normal([2, 128])
    g_out = generator(g_input)
