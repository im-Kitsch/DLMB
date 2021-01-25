import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

from dataloader.skin_lesion_dataset import SkinLesionDataset
from models.dcgan import Generator, Discriminator, init_weights

# move to other file maybe?
def train(ngpu=1):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cudnn.benchmark = True

    random.seed(42)
    torch.manual_seed(42)

    LEARNING_RATE = 3e-4
    NOISE_DIM = 64                                                            # dimension of input noise vector
    IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS = (600, 450, 3)                  # input image dimension
    IMAGE_SIZE = 64
    BATCH_SIZE = 64
    EPOCHS = 200

    FEATURES_DISC = 64
    FEATURES_GEN = 64

    outf = "../output"

    dataset = SkinLesionDataset(transform=transforms.Compose([
                            transforms.Resize(IMAGE_SIZE),
                            transforms.CenterCrop(IMAGE_SIZE),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Setup nets
    disc = Discriminator(IMAGE_CHANNELS, FEATURES_DISC, ngpu=ngpu).to(device)
    gen = Generator(NOISE_DIM, IMAGE_CHANNELS, FEATURES_GEN, ngpu=ngpu).to(device)

    # Init weights
    init_weights(gen)
    init_weights(disc)

    # Setup noise vector
    noise_vector = torch.randn(BATCH_SIZE, NOISE_DIM, 1, 1).to(device)  # simple normal noise distribution
    real_label = 1
    fake_label = 0

    # Setup optimizer
    criterion = nn.BCELoss()
    disc_optim = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    gen_optim = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    disc.train()
    gen.train()

    print("Starting training loop...")
    # Training loop here
    for epoch in range(EPOCHS):
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            disc.zero_grad()
            real_cpu = data[0].to(device)
            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), real_label,
                               dtype=real_cpu.dtype, device=device)

            output = disc(real_cpu)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            noise = torch.randn(batch_size, NOISE_DIM, 1, 1, device=device)
            fake = gen(noise)
            label.fill_(fake_label)
            output = disc(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            disc_optim.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            gen.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = disc(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            gen_optim.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, EPOCHS, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            if i % 100 == 0:
                vutils.save_image(real_cpu,
                                  '%s/real_samples.png' % outf,
                                  normalize=True)
                fake = gen(noise_vector)
                vutils.save_image(fake.detach(),
                                  '%s/fake_samples_epoch_%03d.png' % (outf, epoch),
                                  normalize=True)

        # do checkpointing
        torch.save(gen.state_dict(), '%s/netG_epoch_%d.pth' % ("../checkpoints", epoch))
        torch.save(disc.state_dict(), '%s/netD_epoch_%d.pth' % ("../checkpoints", epoch))


def create_dirs():
    try:
        os.mkdir("../checkpoints")
    except OSError:
        None
    else:
        print("Succesfully created checkpoint directory.")

    try:
        os.mkdir("../output")
    except OSError:
        None
    else:
        print("Succesfully created checkpoint directory.")


if __name__ == '__main__':
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    #train(2)
    create_dirs()
    train()

