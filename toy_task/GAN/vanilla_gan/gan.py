import torch
import torchvision
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

import argparse

torch.manual_seed(0)
CLOSE_RAND = False
CLOSE_DROPOUT = False


class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = torch.nn.Sequential(
            torch.nn.Linear(784, 1024),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(0.3),

            torch.nn.Linear(1024, 512),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(0.3),

            torch.nn.Linear(512, 256),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(0.3),

            torch.nn.Linear(256, 1),
            torch.nn.Sigmoid()
        )
        return

    def forward(self, x):
        if CLOSE_RAND:
            torch.random.manual_seed(0)
        return self.main(x)


class Generator(torch.nn.Module):
    def __init__(self, dim_noise):
        super(Generator, self).__init__()
        self.main = torch.nn.Sequential(
            torch.nn.Linear(dim_noise, 256),
            torch.nn.LeakyReLU(0.2),

            torch.nn.Linear(256, 512),
            torch.nn.LeakyReLU(0.2),

            torch.nn.Linear(512, 1024),
            torch.nn.LeakyReLU(0.2),

            torch.nn.Linear(1024, 784),
            torch.nn.Tanh()
        )
        return

    def forward(self, x):
        if CLOSE_RAND:
            torch.random.manual_seed(0)
        return self.main(x)


class GAN:
    def __init__(self):
        self.dim_noise = 128

        self.discriminator = Discriminator()
        self.generator = Generator(self.dim_noise)

        self.generator.cuda()
        self.discriminator.cuda()

        self.criterion = torch.nn.BCELoss()

        self.g_optim = torch.optim.Adam(self.generator.parameters(), lr=2e-4)
        self.d_optim = torch.optim.Adam(self.discriminator.parameters(), lr=2e-4)
        return

    def train(self, train_loader, test_loader, n_epoch, g_step, d_step):
        epoch_loss_d = 0.
        epoch_loss_g = 0.
        epoch_score_p = 0.
        epoch_score_f = 0.

        writer =SummaryWriter()

        test_noise = torch.randn(64, self.dim_noise, device="cuda:0")
        for i in range(n_epoch):
            if CLOSE_DROPOUT:
                self.generator.eval()
                self.discriminator.eval()
            else:
                self.generator.train()
                self.discriminator.train()
            with tqdm(total=len(train_loader), desc=f"epoc{i}") as pbar:
                for k, (data_real, lbl) in enumerate(train_loader):
                    data_real = data_real.reshape(-1, 784)
                    data_real = data_real.cuda()

                    d_loss, p_score, f_score = self.train_discriminator(data_real, d_step)
                    g_loss = self.train_generator(g_step, data_real.shape[0])

                    epoch_loss_d += d_loss
                    epoch_loss_g += g_loss
                    epoch_score_f += f_score
                    epoch_score_p += p_score

                    pbar.set_postfix({"d_loss": d_loss, "g_loss": g_loss,
                                      "p_score": p_score, "f_score": f_score})
                    pbar.update()
                epoch_loss_g = epoch_loss_g/(k+1)
                epoch_loss_d = epoch_loss_d/(k+1)
                epoch_score_p /= k+1
                epoch_score_f /= k+1
                pbar.set_postfix({"epoch:  d_loss": epoch_loss_d, "g_loss": epoch_loss_g,
                                  "p_score": epoch_score_p, "f_score": epoch_score_f})

            writer.add_scalar('loss/generator', epoch_loss_g, i)
            writer.add_scalar('loss/discriminator', epoch_loss_d, i)
            writer.add_scalar('score/real', epoch_score_p, i)
            writer.add_scalar('score/fake', epoch_score_f, i)
            # self.generator.eval() //TODO to recover
            # self.discriminator.eval()
            test_img = self.generator(test_noise)
            test_img = (test_img + 1.0)/2.0
            test_img = test_img.reshape(-1, 1, 28, 28)
            writer.add_images('img', test_img, i)
        return

    def train_discriminator(self, data_real, d_step):
        d_loss = 0.
        score_p = 0.
        score_f = 0.
        n_real = data_real.shape[0]

        for _d_n in range(d_step):

            data_fake = self.generate_fake(n_real).detach()
            p_f = self.discriminator(data_fake)
            p_p = self.discriminator(data_real)

            label_f = torch.zeros(data_fake.shape[0], device="cuda:0")
            label_p = torch.ones(data_real.shape[0], device="cuda:0")

            loss_f = self.criterion(p_f.reshape(-1), label_f)
            loss_p = self.criterion(p_p.reshape(-1), label_p)

            loss = loss_p + loss_f
            self.d_optim.zero_grad()
            loss.backward()
            self.d_optim.step()

            score_p += p_p.mean().item()
            score_f += p_f.mean().item()
            d_loss += loss.item()
        return d_loss/d_step, score_p/d_step, score_f/d_step

    def train_generator(self, g_step, batch_size):
        g_loss = 0.

        # for _g in range(g_step):

        fake = self.generate_fake(batch_size)
        p_f = self.discriminator(fake)
        label = torch.ones(batch_size, 1, device="cuda:0")
        loss = self.criterion(p_f, label)

        self.g_optim.zero_grad()
        loss.backward()
        self.g_optim.step()

        g_loss += loss.item()

        # return g_loss/g_step
        return g_loss

    def generate_fake(self, n_fake):
        if CLOSE_RAND:
            torch.random.manual_seed(0)
        noise = torch.randn(n_fake, self.dim_noise, device="cuda:0")
        return self.generator(noise)


def main(args):
    torch.random.manual_seed(0)

    m_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize([0.5], [0.5])])

    # m_transform = torchvision.transforms.ToTensor()
    data_train = torchvision.datasets.MNIST(root="./MNIST_data/", transform=m_transform, train=True)
    data_test = torchvision.datasets.MNIST(root="./MNIST_data/", transform=m_transform, train=False)

    train_loader = torch.utils.data.DataLoader(dataset=data_train, batch_size=args.batch_size,
                                               shuffle=True, drop_last=False)
    test_loader = torch.utils.data.DataLoader(dataset=data_test, batch_size=args.batch_size,
                                              shuffle=True, drop_last=True)
    gan = GAN()
    if CLOSE_RAND:
        torch.save(gan.generator.state_dict(), 'result/init_g.pt')
        torch.save(gan.discriminator.state_dict(), 'result/init_d.pt')
    gan.train(train_loader, test_loader, args.num_epochs, g_step=args.g_step, d_step=args.d_step)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', default=200, type=int, help='number of training epochs')
    parser.add_argument('--g-step', default=1, type=int, help='empty')
    parser.add_argument("--d-step", default=1, type=int)
    parser.add_argument('--batch-size', default=512)
    args = parser.parse_args()
    main(args)
