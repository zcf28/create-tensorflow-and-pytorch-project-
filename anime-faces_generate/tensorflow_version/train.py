import glob
import os

from tensorflow import keras
import numpy as np
import tensorflow as tf
from PIL import Image

from model import Generator
from model import Discriminator
from dataset import Dataset

import logging


def save_image(image_path, images):
    os.makedirs(image_path, exist_ok=True)

    for index in range(images.shape[0]):
        im = Image.fromarray(np.uint8(images[index] * 255), mode="RGB")
        im.save(f"{image_path}/{index}.jpg")


def gradient_penalty(discriminator, batch_x, fake_image):
    """
    功能描述： 构建 gp
    :param discriminator:
    :param batch_x:
    :param fake_image:
    :return:
    """
    batch_x_size0 = batch_x.shape[0]

    t = tf.random.uniform([batch_x_size0, 1, 1, 1])
    t = tf.broadcast_to(t, batch_x.shape)
    inter_plate = t * batch_x + (1 - t) * fake_image

    with tf.GradientTape() as tape:
        tape.watch([inter_plate])
        d_inter_plate = discriminator(inter_plate, True)

    grads = tape.gradient(d_inter_plate, inter_plate)
    grads = tf.reshape(grads, [grads.shape[0], -1])

    gp = tf.norm(grads, axis=1)
    gp = tf.reduce_mean((gp - 1.0) ** 2)
    return gp


def discriminator_loss(generator, discriminator, batch_x, batch_z, training):
    """
    功能描述： 鉴别器 损失
    :param generator:
    :param discriminator:
    :param batch_x:
    :param batch_z:
    :param training:
    :return:
    """
    fake_image = generator(batch_z, training)
    d_fake = discriminator(fake_image, training)
    d_real = discriminator(batch_x, training)

    d_real_loss = tf.reduce_mean(d_real)
    d_fake_loss = tf.reduce_mean(d_fake)

    gp = gradient_penalty(discriminator, batch_x, fake_image)

    return d_fake_loss - d_real_loss + 10.0 * gp, gp


def generator_loss(generator, discriminator, batch_z, training):
    """
    功能描述： 生成器 损失
    :param generator:
    :param discriminator:
    :param batch_z:
    :param training:
    :return:
    """
    fake_image = generator(batch_z, training)
    g_fake = discriminator(fake_image, training)

    g_fake_loss = tf.reduce_mean(g_fake)

    return -g_fake_loss


def train(epochs, batch_size, lr, z_dim):
    """
    功能描述： 迭代训练模型
    :param epochs:
    :param batch_size:
    :param lr:
    :param z_dim:
    :return:
    """
    tf.random.set_seed(42)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    logging.basicConfig(filename="./run_out.log", level=logging.INFO)

    generator = Generator()
    generator.build(input_shape=(None, z_dim))
    discriminator = Discriminator()
    discriminator.build(input_shape=(None, 64, 64, 3))

    g_optimizers = keras.optimizers.Adam(learning_rate=lr, beta_1=0.5)
    d_optimizers = keras.optimizers.Adam(learning_rate=lr, beta_1=0.5)

    image_dataset = Dataset(batch_size)

    for epoch in range(epochs):

        image_data_loader = image_dataset.get_data_loader()

        for step, batch_x in enumerate(image_data_loader):

            batch_z = tf.random.uniform([batch_size, z_dim], minval=-1, maxval=1)

            # train discriminator
            with tf.GradientTape() as tape:
                d_loss, gp = discriminator_loss(generator, discriminator, batch_x, batch_z, True)

            d_grads = tape.gradient(d_loss, discriminator.trainable_variables)
            d_optimizers.apply_gradients(zip(d_grads, discriminator.trainable_variables))

            #  train generator  每迭代 五次鉴别器 迭代一次生成器
            if step % 5 == 0 and step != 0:
                with tf.GradientTape() as tape:
                    g_loss = generator_loss(generator, discriminator, batch_z, True)

                g_grads = tape.gradient(g_loss, generator.trainable_variables)
                g_optimizers.apply_gradients(zip(g_grads, generator.trainable_variables))

        logging.info(f"epoch: {epoch}, d_loss: {d_loss}, g_loss: {g_loss}, gp: {gp}")

        # 每 100 epoch 生成一次样本
        if epoch % 100 == 0 and epoch != 0:
            z = tf.random.uniform([batch_size, z_dim], minval=-1, maxval=1)
            fake_image = generator(z, False)
            save_image(f"./save_image/{epoch}", fake_image.numpy())
            os.makedirs(f"./save_model/{epoch}", exist_ok=True)
            generator.save_weights(f"./save_model/{epoch}/generator")


if __name__ == '__main__':
    epochs = 10000
    batch_size = 32
    lr = 1e-4
    z_dim = 128

    train(epochs, batch_size, lr, z_dim)
