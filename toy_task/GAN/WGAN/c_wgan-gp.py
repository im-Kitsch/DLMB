import numpy as np
import torch
import torchvision
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

import argparse


IF_CUDA = True if torch.cuda.is_available() else False
DEVICE = torch.device('cuda') if IF_CUDA else torch.device('CPU')
RANDOM_SEED = False  # TODO not implemented yet

print(f"Now it's using {'GPU' if IF_CUDA else 'CPU'}")


class Discriminator(torch.nn.Module):
    def __init__(self, img_size, n_ch, n_class):
        super(Discriminator, self).__init__()
        self.img_size = img_size
        self.n_ch = n_ch
        self.img_shape = (n_ch, img_size, img_size)
        self.n_class = n_class

        self.lbl_embedding = torch.nn.Sequential(
            torch.nn.Embedding(self.n_class, 64),
            torch.nn.Dropout(0.2)
        )
        self.flatten_layer = torch.nn.Flatten(1, -1)
        self.main = torch.nn.Sequential(
            # torch.nn.Flatten(1, -1),
            torch.nn.Linear(np.prod(self.img_shape) + 64, 1024),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(0.3),

            torch.nn.Linear(1024, 512),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(0.3),

            torch.nn.Linear(512, 256),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(0.3),

            torch.nn.Linear(256, 1),
            # torch.nn.Sigmoid()
            # Very helpful to comment last sigmoid activation
        )
        return

    def forward(self, x, lbl):
        emb = self.lbl_embedding(lbl)
        img = self.flatten_layer(x)
        return self.main(torch.cat([img, emb], dim=1))


class Generator(torch.nn.Module):
    def __init__(self, dim_noise, img_size, n_ch, n_class):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.n_ch = n_ch
        self.img_shape = (n_ch, img_size, img_size)
        self.n_class = n_class

        self.lbl_embedding = torch.nn.Sequential(
            torch.nn.Embedding(n_class, 64),
            torch.nn.Dropout(0.2)
        )

        self.main = torch.nn.Sequential(
            torch.nn.Linear(dim_noise + 64, 256),
            torch.nn.LeakyReLU(0.2),

            torch.nn.Linear(256, 512),
            torch.nn.LeakyReLU(0.2),

            torch.nn.Linear(512, 1024),
            torch.nn.LeakyReLU(0.2),

            torch.nn.Linear(1024, img_size*img_size),
            torch.nn.Tanh(),
            torch.nn.Unflatten(1, self.img_shape)
        )
        return

    def forward(self, noise, lbl):
        noise_and_lbl = torch.cat([noise, self.lbl_embedding(lbl)], dim=1)
        return self.main(noise_and_lbl)


class GAN:
    def __init__(self, dim_noise, lr_d, lr_g, d_step, n_epoch, img_size, n_ch, gp_lambda, beta_1, beta_2, n_class):
        self.dim_noise = dim_noise
        self.lr_d = lr_d
        self.lr_g = lr_g
        self.gp_lambda = gp_lambda
        self.d_step = d_step
        self.n_epoch = n_epoch
        self.img_size = img_size
        self.n_ch = n_ch
        self.img_shape = (n_ch, img_size, img_size)
        self.n_class = n_class

        self.discriminator = Discriminator(img_size=img_size, n_ch=n_ch, n_class=n_class)
        self.generator = Generator(self.dim_noise, img_size=self.img_size, n_ch=n_ch, n_class=n_class)

        if IF_CUDA:
            self.generator.cuda()
            self.discriminator.cuda()

        self.criterion = torch.nn.BCELoss()

        self.g_optim = torch.optim.Adam(self.generator.parameters(), lr=self.lr_g, betas=(beta_1, beta_2))
        self.d_optim = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr_d, betas=(beta_1, beta_2))
        return

    def train(self, train_loader, test_loader):

        writer = SummaryWriter()

        test_batch = 64
        test_noise = torch.randn(test_batch, self.dim_noise)  # TODO to improve
        # test_lbl = torch.arange(self.n_class)
        if IF_CUDA:
            test_noise = test_noise.cuda()
            # test_noise = torch.randn(1, 64, self.dim_noise, device="cuda:0")
            # test_lbl = test_lbl.cuda()

        for i in range(self.n_epoch):

            epoch_loss_d = 0.
            epoch_loss_g = 0.
            epoch_score_p = 0.
            epoch_score_f = 0.

            self.generator.train()
            self.discriminator.train()
            with tqdm(total=len(train_loader), desc=f"epoc{i+1}") as pbar:
                for k, (data_real, lbl) in enumerate(train_loader):
                    if IF_CUDA:
                        data_real = data_real.cuda()
                        lbl = lbl.cuda()

                    d_loss, p_score, f_score = self.train_discriminator(data_real, lbl)
                    g_loss = self.train_generator(data_real.shape[0], lbl)

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
            self.generator.eval()
            self.discriminator.eval()
            with torch.no_grad():
                for _n_cls in range(self.n_class):
                    test_lbl = torch.zeros(test_batch, dtype=torch.long) + _n_cls
                    if IF_CUDA:
                        test_lbl = test_lbl.cuda()
                    test_img = self.generator(test_noise, test_lbl)
                    test_img = (test_img + 1.0)/2.0  # denorm
                    writer.add_images(f'img {_n_cls}', test_img, i)
        return

    def train_discriminator(self, data_real, lbl):
        d_loss = 0.
        score_p = 0.
        score_f = 0.
        n_real = data_real.shape[0]

        for _d_n in range(self.d_step):

            data_fake = self.generate_fake(n_real, lbl).detach()
            mix_noise = torch.rand(n_real, 1, 1, 1)

            if IF_CUDA:
                mix_noise = mix_noise.cuda()

            data_mixed = (1 - mix_noise) * data_real + mix_noise * data_fake
            data_mixed = data_mixed.detach()
            data_mixed.requires_grad_()

            p_f = self.discriminator(data_fake, lbl)
            p_p = self.discriminator(data_real, lbl)
            p_mix = self.discriminator(data_mixed, lbl)

            loss_1 = p_f - p_p

            # gradient penalty
            grad_p_x = torch.autograd.grad(p_mix.sum(), data_mixed, retain_graph=True, create_graph=True)[0]
            # p_mix.sum(), trick to cal \par y_i / \parx_i independentl
            assert grad_p_x.shape == data_mixed.shape
            grad_norm = torch.sqrt(grad_p_x.square().sum(axis=(1, 2, 3)) + 1e-14)
            loss_2 = self.gp_lambda * torch.square(grad_norm - 1.)

            loss = loss_1 + loss_2
            loss = loss.mean()
            self.d_optim.zero_grad()
            loss.backward()
            self.d_optim.step()

            score_p += p_p.mean().item()
            score_f += p_f.mean().item()
            d_loss += loss.item()
        return d_loss/self.d_step, score_p/self.d_step, score_f/self.d_step

    def train_generator(self, batch_size, lbl):
        g_loss = 0.

        fake = self.generate_fake(batch_size, lbl)
        p_f = self.discriminator(fake, lbl)
        loss = -p_f.mean()

        self.g_optim.zero_grad()
        loss.backward()
        self.g_optim.step()

        g_loss += loss.item()

        return g_loss

    def generate_fake(self, n_fake, lbl):
        if IF_CUDA:
            noise = torch.randn(n_fake, self.dim_noise, device="cuda:0")
        else:
            noise = torch.randn(n_fake, self.dim_noise)
        return self.generator(noise, lbl)


def main(args):
    torch.random.manual_seed(0)

    if args.data == 'MNIST':
        m_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.Normalize([0.5], [0.5])])
    else:
        raise Exception['you have not defined this dataset']
    data_train = torchvision.datasets.MNIST(root=args.data_root,transform=m_transform, train=True,
                                            download=True)
    data_test = torchvision.datasets.MNIST(root=args.data_root, transform=m_transform, train=False)

    train_loader = torch.utils.data.DataLoader(dataset=data_train, batch_size=args.batch_size,
                                               shuffle=True, drop_last=False)
    test_loader = torch.utils.data.DataLoader(dataset=data_test, batch_size=args.batch_size,
                                              shuffle=True, drop_last=False)
    gan = GAN(dim_noise=args.dim_noise, lr_d=args.lr_d, lr_g=args.lr_g, n_ch=args.n_ch,
              d_step=args.d_step, n_epoch=args.num_epochs, img_size=args.img_size,
              gp_lambda=args.gp_lambda, beta_1=args.beta1, beta_2=args.beta2, n_class=len(data_train.classes))
    gan.train(train_loader, test_loader)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('--data', required=True, help='MNIST|HAM10000')
    parser.add_argument('--data-root', default='~/Documents/datas/MNIST_data/')
    parser.add_argument('-n', '--num-epochs', default=200, type=int, help='number of training epochs')
    parser.add_argument('--lr-d', default=2e-4, type=float)
    parser.add_argument('--lr-g', default=2e-4, type=float)
    parser.add_argument('--beta1', default=0.5)
    parser.add_argument('--beta2', default=0.999)
    parser.add_argument('--gp_lambda', default=10, type=float, help='for wgan-gp, gradient penalty parameter')
    parser.add_argument('--n_ch', default=1, type=int, help='number of channels of figure')
    parser.add_argument('--img-size', default=28, type=int)
    parser.add_argument('--dim-noise', default=128, type=int)
    parser.add_argument("--d-step", default=5, type=int)
    parser.add_argument('--batch-size', default=512)
    args = parser.parse_args()
    main(args)
