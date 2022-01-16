import logging
import os

import numpy as np
import torch
from PIL import Image

from model import Generator, Discriminator
from dataset import TrainDataset

from torch.utils.data import DataLoader


def save_image(image_path, images):
    os.makedirs(image_path, exist_ok=True)

    for index in range(images.shape[0]):
        im = Image.fromarray(np.uint8(np.transpose(images[index], (1, 2, 0))))
        im.save(f"{image_path}/{index}.jpg")


def gradient_penalty(discriminator, real_image, fake_image, device):
    t = torch.rand(real_image.size(0), 1, 1, 1).to(device)
    t = t.expand(real_image.size())

    interpolates = t * real_image + (1 - t) * fake_image
    interpolates.requires_grad_(True)
    interpolates = interpolates.to(device)
    disc_interpolates = discriminator(interpolates)
    grad = torch.autograd.grad(
        outputs=disc_interpolates, inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True, retain_graph=True)[0]

    grad_norm = torch.norm(torch.flatten(grad, start_dim=1), dim=1)
    loss_gp = torch.mean((grad_norm - 1) ** 2)
    return loss_gp


def discriminator_loss(generator, discriminator, real_image, batch_z, device):
    fake_image = generator(batch_z.detach())
    d_fake = discriminator(fake_image)
    d_real = discriminator(real_image)

    d_real_loss = torch.mean(d_real)
    d_fake_loss = torch.mean(d_fake)

    gp = gradient_penalty(discriminator, real_image, fake_image, device)

    return d_fake_loss - d_real_loss + 10.0 * gp, gp


def generator_loss(generator, discriminator, batch_z):
    fake_image = generator(batch_z)
    g_fake = discriminator(fake_image)

    g_fake_loss = torch.mean(g_fake)

    return -g_fake_loss


def train(epochs, batch_size, lr, z_dim):
    torch.manual_seed(42)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.basicConfig(filename="./run_out.log", level=logging.INFO)

    generator = Generator(z_dim=z_dim).to(device)
    discriminator = Discriminator().to(device)

    g_optimizers = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    d_optimizers = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    image_dataset = TrainDataset()

    for epoch in range(epochs):
        generator.train()
        discriminator.eval()

        d_loss = 0.0
        g_loss = 0.0
        gp = 0.0
        image_data_loader = DataLoader(image_dataset, shuffle=True, batch_size=batch_size, drop_last=True)
        for step, real_image in enumerate(image_data_loader):
            real_image = real_image.to(device)

            d_optimizers.zero_grad()

            batch_z = torch.randn(batch_size, z_dim).to(device)
            d_loss, gp = discriminator_loss(generator, discriminator, real_image, batch_z, device)

            d_loss.backward()
            d_optimizers.step()

            if step % 5 == 0 and step != 0:
                g_optimizers.zero_grad()
                g_loss = generator_loss(generator, discriminator, batch_z)

                g_loss.backward()
                g_optimizers.step()

        logging.info(f"epoch: {epoch}, d_loss: {d_loss.item()}, g_loss: {g_loss.item()}, gp: {gp.item()}")

        if epoch % 100 == 0 and epoch != 0:
            generator.eval()
            z = torch.randn(batch_size, z_dim).to(device)
            fake_image = generator(z)
            save_image(f"./save_image/{epoch}", fake_image.detach().cpu().numpy())
            os.makedirs(f"./save_model/", exist_ok=True)
            torch.save(generator.state_dict(), f"./save_model/generator_{epoch}.pth")


if __name__ == '__main__':
    epochs = 10000
    batch_size = 64
    lr = 1e-4
    z_dim = 128

    train(epochs, batch_size, lr, z_dim)
