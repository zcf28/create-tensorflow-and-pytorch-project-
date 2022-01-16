import logging
import os
import random

import numpy as np
import torch
from dataset import TrainDataset, TestDataset
from model import Model

from torch.utils.data import DataLoader


def train(epochs, batch_size, lr):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    random.seed(42)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    logging.basicConfig(filename="./run.log", level=logging.INFO)
    model = Model().to(device)

    logging.info(f"model arch : \n {model}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    cse = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        train_loader = DataLoader(TrainDataset(), batch_size=batch_size, shuffle=True)

        model.train()
        train_avg_loss = 0.0
        train_avg_predict = 0.0

        train_idx = 0

        for train_image, train_target_label in train_loader:
            train_idx += 1
            train_image = torch.unsqueeze(train_image, 1).to(device)
            train_target_label = train_target_label.to(device)

            optimizer.zero_grad()

            train_out = model(train_image)
            predict_label = torch.argmax(train_out, dim=1)
            train_loss = cse(train_out, train_target_label)

            correct_num = np.sum(torch.eq(predict_label, train_target_label).cpu().numpy() != 0)
            train_avg_predict += correct_num

            train_avg_loss += train_loss.item()

            train_loss.backward()
            optimizer.step()

        train_avg_predict /= (train_idx * batch_size)
        train_avg_loss /= (train_idx * batch_size)

        logging.info(
            f"epoch: {epoch}, train_avg_loss: {train_avg_loss:0.4f}, train_avg_predict: {train_avg_predict:0.4f}")

        if epoch % 10 == 0 and epoch != 0:
            test_loader = DataLoader(TestDataset(), batch_size=batch_size)
            
            model.eval()
            test_avg_loss = 0.0
            test_avg_predict = 0.0
            test_idx = 0
            for test_image, test_target_label in test_loader:
                test_idx += 1
                test_image = torch.unsqueeze(test_image, 1).to(device)
                test_target_label = test_target_label.to(device)
                test_out = model(test_image)

                predict_label = torch.argmax(test_out, dim=1)
                test_loss = cse(test_out, test_target_label)

                correct_num = np.sum(torch.eq(predict_label, test_target_label).cpu().numpy() != 0)
                test_avg_predict += correct_num

                test_avg_loss += test_loss.item()

            test_avg_predict /= (test_idx * batch_size)
            test_avg_loss /= (test_idx * batch_size)

            logging.info(f"= =" * 10 + f" TEST MODEL " + f"= =" * 10)
            logging.info(
                f"test_avg_loss: {test_avg_loss:0.4f}, test_avg_predict: {test_avg_predict:0.4f}")
            logging.info(f"= =" * 10 + f" TEST MODEL " + f"= =" * 10)


if __name__ == '__main__':
    epochs = 100
    batch_size = 64
    lr = 1e-4
    train(epochs, batch_size, lr)