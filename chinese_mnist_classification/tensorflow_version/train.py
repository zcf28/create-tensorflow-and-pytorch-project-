import logging
import os

import numpy as np

from dataset import Dataset
from model import Model

import tensorflow as tf
from tensorflow import keras


def train(epochs, batch_size, lr):
    """
    功能描述： 模型训练
    :param epochs:
    :param batch_size:
    :param lr:
    :return:
    """
    tf.random.set_seed(42)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    logging.basicConfig(filename="./run.log", level=logging.INFO)

    dataset = Dataset(batch_size)

    train_image_loader, test_image_loader = dataset.get_data_loader()

    model = Model()
    model.build(input_shape=(None, 64, 64, 1))

    optimizers = keras.optimizers.Adam(learning_rate=lr)
    cse = keras.losses.CategoricalCrossentropy(from_logits=True)

    for epoch in range(epochs):
        train_avg_loss = 0.0
        train_avg_predict = 0.0

        train_idx = 0

        for train_image, train_target_label in train_image_loader:
            train_idx += 1
            train_image = tf.expand_dims(train_image, -1)
            train_target_label_one_hot = tf.one_hot(train_target_label, depth=15)

            with tf.GradientTape() as tape:
                train_out = model(train_image, True)
                train_loss = tf.reduce_mean(cse(train_target_label_one_hot, train_out))

            train_avg_loss += train_loss.numpy()
            train_predict_label = tf.argmax(train_out, 1)
            train_correct_num = np.sum(np.equal(train_predict_label.numpy(), train_target_label.numpy()) != 0)
            train_avg_predict += train_correct_num

            grads = tape.gradient(train_loss, model.trainable_variables)
            optimizers.apply_gradients(zip(grads, model.trainable_variables))

        train_avg_loss /= (train_idx * batch_size)
        train_avg_predict /= (train_idx * batch_size)

        logging.info(
            f"epoch: {epoch}, train_avg_loss: {train_avg_loss:0.4f}, train_avg_predict: {train_avg_predict:0.4f}")

        if epoch % 10 == 0 and epoch != 0:
            test_avg_loss = 0.0
            test_avg_predict = 0.0
            test_idx = 0
            for test_image, test_target_label in test_image_loader:
                test_idx += 1

                test_image = tf.expand_dims(test_image, -1)
                test_target_label_one_hot = tf.one_hot(test_target_label, depth=15)
                test_out = model(test_image, False)

                test_loss = tf.reduce_mean(cse(test_target_label_one_hot, test_out))
                test_avg_loss += test_loss.numpy()

                test_predict_label = tf.argmax(test_out, 1)
                test_correct_num = np.sum(np.equal(test_predict_label.numpy(), test_target_label.numpy()) != 0)
                test_avg_predict += test_correct_num

            test_avg_loss /= (test_idx * batch_size)
            test_avg_predict /= (test_idx * batch_size)
            logging.info(f"= =" * 10 + f" TEST MODEL " + f"= =" * 10)

            logging.info(
                f"test_avg_loss: {test_avg_loss:0.4f}, test_avg_predict: {test_avg_predict:0.4f}")

            logging.info(f"= =" * 10 + f" TEST MODEL " + f"= =" * 10)


if __name__ == '__main__':
    epoch = 100
    batch_size = 64
    lr = 1e-4
    train(epoch, batch_size, lr)
