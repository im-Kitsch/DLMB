import argparse

import torchvision
import torch
from torch.utils import data
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import util.dataset_util

IF_CUDA = True if torch.cuda.is_available() else False
DEVICE = torch.device('cuda') if IF_CUDA else torch.device('cpu')

TRANS_MEAN = [0.485, 0.456, 0.406]
TRANS_STD = [0.229, 0.224, 0.225]
# src, experimental setting:
# https://github.com/facebookarchive/fb.resnet.torch/blob/master/datasets/imagenet.lua#L69


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


class ConvDiscriminator(torch.nn.Module):
    def __init__(self, n_ch, img_size):
        super(ConvDiscriminator, self).__init__()
        self.n_ch = n_ch
        self.img_size = img_size
        self.main = torch.nn.Sequential(
            # input is (n_ch) x 64 x 64
            torch.nn.Conv2d(n_ch, img_size, 4, 2, 1, bias=False),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # state size. (img_size) x 32 x 32
            torch.nn.Conv2d(img_size, img_size * 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(img_size * 2),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # state size. (img_size*2) x 16 x 16
            torch.nn.Conv2d(img_size * 2, img_size * 4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(img_size * 4),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # state size. (img_size*4) x 8 x 8
            torch.nn.Conv2d(img_size * 4, img_size * 8, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(img_size * 8),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # state size. (img_size*8) x 4 x 4
            torch.nn.Conv2d(img_size * 8, 1, 4, 1, 0, bias=False),
            torch.nn.Sigmoid()
        )
        # self.main_activation = torch.nn.Sigmoid()
        return

    def forward(self, x):
        return self.main(x).view(-1, 1)
    
    
class ConvGenerator(torch.nn.Module):
    def __init__(self, n_ch, img_size, z_dim):
        super(ConvGenerator, self).__init__()
        self.n_ch = n_ch
        self.img_size = img_size
        self.z_dim = z_dim
        self.main = torch.nn.Sequential(
            # input is Z, going into a convolution
            torch.nn.ConvTranspose2d(z_dim, img_size * 8, 4, 1, 0, bias=False),
            torch.nn.BatchNorm2d(img_size * 8),
            torch.nn.ReLU(True),
            # state size. (img_size*8) x 4 x 4
            torch.nn.ConvTranspose2d(img_size * 8, img_size * 4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(img_size * 4),
            torch.nn.ReLU(True),
            # state size. (img_size*4) x 8 x 8
            torch.nn.ConvTranspose2d(img_size * 4, img_size * 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(img_size * 2),
            torch.nn.ReLU(True),
            # state size. (img_size*2) x 16 x 16
            torch.nn.ConvTranspose2d(img_size * 2, img_size, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(img_size),
            torch.nn.ReLU(True),
            # state size. (img_size) x 32 x 32
            torch.nn.ConvTranspose2d(img_size, n_ch, 4, 2, 1, bias=False),
            torch.nn.Tanh()
            # state size. (n_ch) x 64 x 64
        )
        return

    def forward(self, noise):
        return self.main(noise)


class DCGAN(torch.nn.Module):
    def __init__(self, data_name, n_ch, img_size, z_dim, lr_g, lr_d, lr_beta1, lr_beta2, d_step):
        super(DCGAN, self).__init__()
        self.data_name = data_name
        self.img_shape = (n_ch, img_size, img_size)

        self.z_dim = z_dim
        self.d_step = d_step

        self.conv_gen = ConvGenerator(n_ch=n_ch, img_size=img_size, z_dim=z_dim)
        self.conv_dis = ConvDiscriminator(n_ch=n_ch, img_size=img_size)
        self.conv_gen.main.apply(weights_init)  # TODO to find a better method to initialization instead of using main
        self.conv_dis.main.apply(weights_init)
        if IF_CUDA:
            self.conv_gen.cuda()
            self.conv_dis.cuda()
        self.opt_G = torch.optim.Adam(self.conv_gen.parameters(), lr=lr_g, betas=(lr_beta1, lr_beta2))
        self.opt_D = torch.optim.Adam(self.conv_dis.parameters(), lr=lr_d, betas=(lr_beta1, lr_beta2))
        self.criterion = torch.nn.BCELoss()
        return

    def train_net(self, train_loader, n_epoc):
        writer = SummaryWriter(comment=f'_DCGAN_{self.data_name}')

        test_noise = self.generate_noise(64)
        n_sample = len(train_loader.dataset)

        test_img = self.conv_gen(test_noise)
        test_img = (test_img + 1.0) / 2.0  # Note that this is important to recover the range
        test_img = test_img.reshape(64, *self.img_shape)
        writer.add_images('img', test_img, 0)

        for i in range(n_epoc):
            epoc_l_d, epoc_l_g, epoc_score_p, epoc_score_f1, epoc_score_f2 = 0., 0., 0., 0., 0.
            self.conv_gen.train(), self.conv_dis.train()
            with tqdm(total=len(train_loader), desc=f"epoc: {i+1}") as pbar:
                for k, (real_img, _) in enumerate(train_loader):
                    if IF_CUDA:
                        real_img = real_img.cuda()
                    d_loss, p_score, f_score1 = self.train_d_step(real_img)
                    g_loss, f_score2 = self.train_g_step(real_img.shape[0])

                    batch_size = real_img.shape[0]
                    epoc_l_d += d_loss * batch_size
                    epoc_l_g += g_loss * batch_size
                    epoc_score_p += p_score * batch_size
                    epoc_score_f1 += f_score1 * batch_size
                    epoc_score_f2 += f_score2 * batch_size

                    pbar.set_postfix({"d_loss": d_loss, "g_loss": g_loss,
                                      "p_score": p_score, "f_score D": f_score1, 'G': f_score2})
                    pbar.update()

            epoc_l_d /= n_sample
            epoc_l_g /= n_sample
            epoc_score_p /= n_sample
            epoc_score_f1 /= n_sample
            epoc_score_f2 /= n_sample
            pbar.set_postfix({"epoch:  d_loss": epoc_l_d, "g_loss": epoc_l_g,
                              "p_score": epoc_score_p, "f_score D": epoc_score_f1, 'G': epoc_score_f2})

            writer.add_scalar('loss/generator', epoc_l_g, i)
            writer.add_scalar('loss/discriminator', epoc_l_d, i)
            writer.add_scalar('score/real', epoc_score_p, i)
            writer.add_scalar('score/fake_D', epoc_score_f1, i)
            writer.add_scalar('score/fake_G', epoc_score_f2, i)

            self.conv_gen.eval(), self.conv_dis.eval()
            test_img = self.conv_gen(test_noise)
            test_img = (test_img + 1.0) / 2.0  # Note that this is important to recover the range
            test_img = test_img.reshape(64, *self.img_shape)
            writer.add_images('img', test_img, i+1)
        return

    def train_g_step(self, batch_size):
        fake = self.generate_fake(batch_size)
        lbl = torch.ones(batch_size, device=DEVICE)
        p_f = self.conv_dis(fake)
        loss = self.criterion(p_f.reshape(-1), lbl)

        self.opt_G.zero_grad()
        loss.backward()
        self.opt_G.step()
        return loss.item(), p_f.mean().item()

    def train_d_step(self, data_real):
        d_step = self.d_step

        batch_size = data_real.shape[0]
        score_real, score_fake, d_loss = 0., 0., 0.

        for _d in range(d_step):
            data_fake = self.generate_fake(batch_size).detach()

            lbl_real = torch.ones(batch_size, device=DEVICE)
            lbl_fake = torch.zeros(batch_size, device=DEVICE)

            p_fake = self.conv_dis(data_fake)
            p_real = self.conv_dis(data_real)

            loss_real = self.criterion(p_real.reshape(-1), lbl_real)
            loss_fake = self.criterion(p_fake.reshape(-1), lbl_fake)
            loss = loss_real + loss_fake

            self.opt_D.zero_grad()
            loss.backward()
            self.opt_D.step()

            score_real += p_real.mean().item()
            score_fake += p_fake.mean().item()
            d_loss += loss.item()
        return d_loss/d_step, score_real/d_step, score_fake/d_step

    # TODO different method to generate noise
    def generate_noise(self, batch_size):
        return torch.randn(batch_size, self.z_dim, 1, 1, device=DEVICE)

    def generate_fake(self, batch_size):
        return self.conv_gen(self.generate_noise(batch_size))


def main(args):
    if args.data == 'MNIST':
        trans = torchvision.transforms.Compose(
            [torchvision.transforms.Resize(args.img_size),
             torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize([0.5], [0.5])])
    elif args.data == 'CIFAR10':
        trans = torchvision.transforms.Compose(
            [torchvision.transforms.Resize(args.img_size),
             torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    elif args.data == 'HAM10000':
        trans = torchvision.transforms.Compose(
            [torchvision.transforms.CenterCrop(450),
             torchvision.transforms.Resize(args.img_size),
             # torchvision.transforms.CenterCrop(args.img_size),
             torchvision.transforms.ToTensor(),
             # torchvision.transforms.Normalize(TRANS_MEAN, TRANS_STD)])
             torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    else:
        raise Exception('dataset not right')

    train_data, _, img_shape = util.dataset_util.load_dataset(
        dataset_name=args.data, root=args.root, transform=trans, csv_file=args.csv_file)
    n_ch, img_size, _ = img_shape

    train_loader = data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, drop_last=False,
                                   num_workers=4, pin_memory=True)

    dc_gan = DCGAN(data_name=args.data, n_ch=n_ch, img_size=img_size,
                   z_dim=args.z_dim, lr_g=args.lr_g, lr_d=args.lr_d,
                   lr_beta1=args.lr_beta1, lr_beta2=args.lr_beta2, d_step=args.d_step)

    dc_gan.train_net(train_loader=train_loader, n_epoc=args.n_epoc)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('--data', required=True, help='MNIST|CIFAR10|HAM10000')
    parser.add_argument('--root', default='/home/yuan/Documents/datas/', help='root')
    parser.add_argument('--csv-file', default='/home/yuan/Documents/datas/HAM10000/HAM10000_metadata.csv')
    parser.add_argument('--n-epoc', default=25, type=int)
    parser.add_argument('--d-step', default=1, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--z-dim', default=100, type=int, help='noise shape')
    parser.add_argument('--lr-g', default=2e-4, type=float)
    parser.add_argument('--lr-d', default=2e-4, type=float)
    parser.add_argument('--lr-beta1', default=0.5, type=float)
    parser.add_argument('--lr-beta2', default=0.999, type=float)
    parser.add_argument('--img-size', default=64, type=int, help='resize the img size')
    parser.add_argument('--device', default='cuda', help='gpu or cpu for trainning')
    para_args = parser.parse_args()

    if para_args.device == 'cuda' and IF_CUDA is True:  # TODO to improve it
        IF_CUDA = True
        DEVICE = torch.device('cuda')
    else:
        IF_CUDA = False
        DEVICE = torch.device('cpu')
    main(para_args)
