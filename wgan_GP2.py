import argparse
import numpy as np

import torchvision
import torch
from torch.utils import data
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchsummaryX

import util.dataset_util

import os

IF_CUDA = True if torch.cuda.is_available() else False
DEVICE = torch.device('cuda') if IF_CUDA else torch.device('cpu')

TRANS_MEAN = [0.485, 0.456, 0.406]
TRANS_STD = [0.229, 0.224, 0.225]


# src, experimental setting:
# https://github.com/facebookarchive/fb.resnet.torch/blob/master/datasets/imagenet.lua#L69


# TODO reproduce problem:!!!! can not reproduce the result from checkpoint, check if code reason or random seed.

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


class ConvDiscriminator(torch.nn.Module):
    def __init__(self, n_ch, img_size, ndf, depth, if_condition, n_class, embedding_dim):
        super(ConvDiscriminator, self).__init__()
        self.n_ch = n_ch
        self.img_size = img_size
        self.depth = depth
        self.ndf = ndf

        self.if_condition = if_condition
        self.n_class = n_class
        self.embedding_dim = embedding_dim

        if if_condition:
            self.main = torch.nn.Sequential(
                # input is (n_ch) x img_size x img_size
                torch.nn.Conv2d(n_ch, ndf, 4, 2, 1, bias=False),
                torch.nn.LeakyReLU(0.2, inplace=True),
                # add block
                *[self._add_block(self.ndf * np.power(2, i), self.ndf * np.power(2, i + 1), 4, 2, 1, False)
                  for i in range(0, depth - 2)],
                # state size. (ndf * 2^(depth-2)) x 4 x 4
                torch.nn.Conv2d(self.ndf * np.power(2, depth - 2), ndf, 4, 1, 0, bias=False),
                # !! significant difference here, change out channel from 1 to ndf
            )
            self.emb_lay = torch.nn.Sequential(
                torch.nn.Embedding(self.n_class, self.embedding_dim),
                torch.nn.Dropout(0.2)
            )
            self.output_lay = torch.nn.Linear(ndf + self.embedding_dim, 1)
        else:
            self.main = torch.nn.Sequential(
                # input is (n_ch) x img_size x img_size
                torch.nn.Conv2d(n_ch, ndf, 4, 2, 1, bias=False),
                torch.nn.LeakyReLU(0.2, inplace=True),
                # add block
                *[self._add_block(self.ndf*np.power(2, i), self.ndf*np.power(2, i+1), 4, 2, 1, False)
                  for i in range(0, depth-2)],
                # state size. (ndf * 2^(depth-2)) x 4 x 4
                torch.nn.Conv2d(self.ndf * np.power(2, depth-2), 1, 4, 1, 0, bias=False),
                # torch.nn.Sigmoid()
            )
        return

    @staticmethod
    def _add_block(in_ch, out_ch, kernel_size, stride, padding, bias):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_ch, out_channels=out_ch,
                            kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.LeakyReLU(0.2)
        )

    def forward(self, x, condition=None):
        if self.if_condition:
            if condition is None:
                raise Exception('Discriminator have to give gan')
            else:
                batch_size = x.shape[0]
                emb = self.emb_lay(condition)
                input_feature = self.main(x).reshape(batch_size, self.ndf)
                feature_concat = torch.cat([input_feature, emb], dim=1)
                return self.output_lay(feature_concat)
        else:
            return self.main(x).view(-1, 1)


class ConvGenerator(torch.nn.Module):
    def __init__(self, n_ch, img_size, z_dim, ngf, depth, if_condition, n_class, embedding_dim):
        super(ConvGenerator, self).__init__()
        self.n_ch = n_ch
        self.img_size = img_size
        self.z_dim = z_dim
        self.ngf = ngf
        self.depth = depth
        self.if_condition = if_condition
        self.embedding_dim = embedding_dim
        self.n_class = n_class

        if if_condition:
            input_dim = z_dim + embedding_dim
        else:
            input_dim = z_dim

        assert depth >= 5
        self.main = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(input_dim, self.ngf * np.power(2, self.depth-2), 4, 1, 0, bias=False),
            torch.nn.BatchNorm2d(self.ngf * np.power(2, self.depth-2)),
            torch.nn.ReLU(True),
            # middle block
            *[self._add_block(self.ngf*np.power(2, i), self.ngf*np.power(2, i-1), 4, 2, 1, False)
              for i in range(self.depth-2, 0, -1)],
            # state size. (img_ch) x 32 x 32
            torch.nn.ConvTranspose2d(self.ngf, n_ch, 4, 2, 1, bias=False),
            torch.nn.Tanh()
        )
        self.emb_lay = torch.nn.Sequential(
            torch.nn.Embedding(self.n_class, self.embedding_dim),
            torch.nn.Dropout(0.2)
        )
        return

    def forward(self, noise, condition):
        if self.if_condition:
            # if condition is None:
            #   condition = torch.randint(low=0, high=self.n_class, size=(noise.shape[0], 1), device=DEVICE)
            embed = self.emb_lay(condition)[:, :, None, None]
            input_noise = torch.cat([noise, embed], dim=1)
            return self.main(input_noise)
        else:
            return self.main(noise)

    @staticmethod
    def _add_block(in_ch, out_ch, kernel_size, stride, padding, bias):
        return torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_ch, out_ch,
                                     kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(True)
        )


class WGanGP(torch.nn.Module):
    def __init__(self, data_name, n_ch, img_size, z_dim, lr_g, lr_d, lr_beta1, lr_beta2, d_step,
                 ndf, ngf, depth, if_condition, n_class, embedding_dim):
        super(WGanGP, self).__init__()
        self.data_name = data_name
        self.img_shape = (n_ch, img_size, img_size)

        self.if_condition = if_condition
        self.n_class = n_class
        self.embedding_dim = embedding_dim

        self.z_dim = z_dim
        self.d_step = d_step
        self.gp_lambda = 10.
        self.depth = depth
        self.conv_gen = ConvGenerator(n_ch=n_ch, img_size=img_size, z_dim=z_dim, ngf=ngf, depth=depth,
                                      if_condition=if_condition, n_class=n_class, embedding_dim=embedding_dim)
        self.conv_dis = ConvDiscriminator(n_ch=n_ch, img_size=img_size, ndf=ndf, depth=depth,
                                          if_condition=if_condition, n_class=n_class, embedding_dim=embedding_dim)
        # TODO not sure if it is needed to use weight init, but seems better than without init
        self.conv_gen.main.apply(weights_init)  # TODO to find a better method to initialization instead of using main
        self.conv_dis.main.apply(weights_init)
        if IF_CUDA:
            self.conv_gen.cuda()
            self.conv_dis.cuda()
        self.opt_G = torch.optim.Adam(self.conv_gen.parameters(), lr=lr_g, betas=(lr_beta1, lr_beta2))
        self.opt_D = torch.optim.Adam(self.conv_dis.parameters(), lr=lr_d, betas=(lr_beta1, lr_beta2))
        self.criterion = torch.nn.BCELoss()
        return

    def train_net(self, train_loader, n_epoc, checkpoint_factor, arg,
                  if_continue, checkpoint_epoc=None, log_dir=None):  # TODO arg is not good to pass here
        if log_dir is None:
            writer = SummaryWriter(comment=f'_WGAN_GP_{self.data_name}')  # TODO to add hyper parmeters
        else:
            writer = SummaryWriter(log_dir=log_dir)

        if self.if_condition:
            test_lbl = torch.arange(self.n_class, device=DEVICE).reshape(-1, 1)
            test_lbl = test_lbl.repeat(1, 8)
            test_lbl = test_lbl.reshape(-1)
            test_noise = self.generate_noise(test_lbl.shape[0])
        else:
            test_noise = self.generate_noise(64)
        n_sample = len(train_loader.dataset)

        start_epoch = checkpoint_epoc+1 if if_continue else 1
        for i in range(start_epoch, n_epoc+1):
            epoc_l_d, epoc_l_g, epoc_score_p, epoc_score_f1, epoc_score_f2 = 0., 0., 0., 0., 0.
            self.conv_gen.train(), self.conv_dis.train()
            with tqdm(total=len(train_loader), desc=f"epoc: {i}") as pbar:
                for k, (real_img, real_lbl) in enumerate(train_loader):
                    if IF_CUDA:
                        real_img = real_img.cuda()
                        real_lbl = real_lbl.cuda()
                    if self.if_condition:
                        d_loss, p_score, f_score1 = self.train_d_step(real_img, real_lbl)
                        g_loss, f_score2 = self.train_g_step(real_img.shape[0], real_lbl)
                    else:
                        d_loss, p_score, f_score1 = self.train_d_step(real_img, None)
                        g_loss, f_score2 = self.train_g_step(real_img.shape[0], None)

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
            if self.if_condition:
                test_img = self.conv_gen(test_noise, test_lbl)
            else:
                test_img = self.conv_gen(test_noise, None)
            test_img = (test_img + 1.0) / 2.0  # Note that this is important to recover the range
            test_img = test_img.reshape(test_noise.shape[0], *self.img_shape)
            writer.add_images('img', test_img, i)
            writer.flush()

            if i % checkpoint_factor == 0:
                checkpoint_dict = {'arg': arg.__dict__,
                                   'G': self.conv_gen.state_dict(),
                                   'D': self.conv_dis.state_dict(),
                                   'epoch': i,
                                   'torch_seed': torch.initial_seed(),
                                   'log_dir': writer.get_logdir(),
                                   'opt_D': self.opt_D.state_dict(),
                                   'opt_G': self.opt_G.state_dict()}
                save_path = os.path.join(writer.get_logdir(), f'ckpt{i}.pth')
                torch.save(checkpoint_dict, save_path)
        writer.close()
        return

    def train_g_step(self, batch_size, lbl_real):
        if self.if_condition:
            fake = self.generate_fake(batch_size, lbl_real)
            p_f = self.conv_dis(fake, lbl_real)
        else:
            fake = self.generate_fake(batch_size, None)
            p_f = self.conv_dis(fake)

        loss = -p_f.mean()

        self.opt_G.zero_grad()
        loss.backward()
        self.opt_G.step()
        return loss.item(), p_f.mean().item()

    def train_d_step(self, data_real, lbl_real):
        d_step = self.d_step

        batch_size = data_real.shape[0]
        score_real, score_fake, d_loss = 0., 0., 0.

        for _d in range(d_step):
            if self.if_condition:
                data_fake = self.generate_fake(batch_size, lbl_real).detach()
            else:
                data_fake = self.generate_fake(batch_size, None).detach()

            mix_noise = torch.rand(batch_size, 1, 1, 1).cuda()
            data_mixed = (1-mix_noise) * data_real + mix_noise * data_fake
            data_mixed = data_mixed.detach()
            data_mixed.requires_grad_()

            if self.if_condition:
                p_f = self.conv_dis(data_fake, lbl_real)
                p_p = self.conv_dis(data_real, lbl_real)
                p_mix = self.conv_dis(data_mixed, lbl_real)
            else:
                p_f = self.conv_dis.forward(data_fake, None)
                p_p = self.conv_dis(data_real, None)
                p_mix = self.conv_dis(data_mixed, None)

            loss_1 = p_f - p_p

            # gradient penalty
            grad_p_x = torch.autograd.grad(p_mix.sum(), data_mixed, retain_graph=True, create_graph=True)[0]
            # p_mix.sum(), trick to cal \par y_i / \parx_i independentl
            assert grad_p_x.shape == data_mixed.shape
            # print(grad_p_x.shape, data_mixed.shape)
            grad_norm = torch.sqrt(grad_p_x.square().sum(axis=(1, 2, 3)) + 1e-14)
            loss_2 = self.gp_lambda * torch.square(grad_norm - 1.)

            loss = loss_1 + loss_2
            loss = loss.mean()
            self.opt_D.zero_grad()
            loss.backward()
            self.opt_D.step()

            score_real += p_p.mean().item()
            score_fake += p_f.mean().item()
            d_loss += loss.item()
        return d_loss / d_step, score_real / d_step, score_fake / d_step

    # TODO different method to generate noise
    def generate_noise(self, batch_size):
        return torch.randn(batch_size, self.z_dim, 1, 1, device=DEVICE)

    def generate_fake(self, batch_size, condition):
        if self.if_condition:
            # if condition is None:
            #     condition = torch.randint(low=0, high=self.n_class, size=(batch_size, 1))
            return self.conv_gen(self.generate_noise(batch_size), condition)
        else:
            return self.conv_gen(self.generate_noise(batch_size))


def main(args, if_continue=False, checkpoint=None):
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
        if args.data_aug is True:
            trans = torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop(size=(args.img_size, args.img_size), scale=(0.7, 1.0),
                                                         ratio=(4 / 5, 5 / 4), interpolation=2),
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.RandomVerticalFlip(p=0.5),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(TRANS_MEAN, TRANS_STD)
            ])
        else:
            trans = torchvision.transforms.Compose([
                 torchvision.transforms.Resize((args.img_size, args.img_size)),
                 torchvision.transforms.ToTensor(),
                 torchvision.transforms.Normalize(TRANS_MEAN, TRANS_STD)
                 # torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
                 ])
    else:
        raise Exception('dataset not right')

    train_data, _, img_shape = util.dataset_util.load_dataset(
        dataset_name=args.data, root=args.root, transform=trans,
        csv_file=args.csv_file, percentage=args.data_percentage)
    n_ch, img_size, _ = img_shape

    train_loader = data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, drop_last=False,
                                   num_workers=4, pin_memory=True)

    dc_gan = WGanGP(data_name=args.data, n_ch=n_ch, img_size=img_size,
                    z_dim=args.z_dim, lr_g=args.lr_g, lr_d=args.lr_d,
                    lr_beta1=args.lr_beta1, lr_beta2=args.lr_beta2, d_step=args.d_step,
                    ndf=args.ndf, ngf=args.ngf, depth=args.depth,
                    if_condition=args.condition, n_class=len(train_data.dataset.classes),
                    embedding_dim=args.embedding_dim)
    if if_continue is True:
        dc_gan.conv_dis.load_state_dict(checkpoint['D'])
        dc_gan.conv_gen.load_state_dict(checkpoint['G'])
        dc_gan.opt_G.load_state_dict(checkpoint['opt_G'])
        dc_gan.opt_D.load_state_dict(checkpoint['opt_D'])
        checkpoint_epoc = checkpoint['epoch']
        torch.manual_seed(checkpoint['torch_seed'])
        log_dir = checkpoint['log_dir']
    else:
        checkpoint_epoc = None
        log_dir = None

    # torchsummary.summary(dc_gan.conv_dis, input_size=[dc_gan.img_shape, 1], batch_size=-1,
    #                      dtypes=[torch.FloatTensor, torch.LongTensor],
    #                      device='cuda' if IF_CUDA else 'cpu')
    # torchsummary.summary(dc_gan.conv_gen, input_size=(dc_gan.z_dim, 1, 1), batch_size=-1,
    #                      device='cuda' if IF_CUDA else 'cpu')
    torchsummaryX.summary(dc_gan.conv_dis,
                          torch.zeros(1, *dc_gan.img_shape, device=DEVICE), torch.tensor([0], device=DEVICE))
    torchsummaryX.summary(dc_gan.conv_gen,
                          torch.zeros(1, dc_gan.z_dim, 1, 1, device=DEVICE), torch.tensor([0], device=DEVICE))

    dc_gan.train_net(train_loader=train_loader, n_epoc=args.n_epoc,
                     checkpoint_factor=args.checkpoint_factor, arg=args,
                     if_continue=if_continue, checkpoint_epoc=checkpoint_epoc, log_dir=log_dir)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('--data', required=True, help='MNIST|CIFAR10|HAM10000')
    parser.add_argument('--root', default='/home/yuan/Documents/datas/', help='root')
    parser.add_argument('--csv-file', default='/home/yuan/Documents/datas/HAM10000/HAM10000_metadata.csv')
    parser.add_argument('--n-epoc', default=25, type=int)
    parser.add_argument('--d-step', default=1, type=int)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--z-dim', default=64, type=int, help='noise shape')
    parser.add_argument('--ndf', default=64, type=int)
    parser.add_argument('--ngf', default=64, type=int)
    parser.add_argument('--depth', default=5, type=int)
    parser.add_argument('--lr-g', default=1e-4, type=float)
    parser.add_argument('--lr-d', default=5e-4, type=float)
    parser.add_argument('--lr-beta1', default=0., type=float)
    parser.add_argument('--lr-beta2', default=0.99, type=float)
    parser.add_argument('--img-size', default=64, type=int, help='resize the img size')
    parser.add_argument('--data-percentage', default=1.0, type=float)
    parser.add_argument('--data-aug', action='store_true', help='if use data augmentation or not')
    parser.add_argument('--condition', action='store_true', help='if use condition')
    parser.add_argument('--embedding-dim', default=64, help='')
    parser.add_argument('--recover', action='store_true', help='if continue training from prior checkpoint')
    # TODO seems to redundent here
    parser.add_argument('--checkpoint-file', default='', type=str, help='')
    parser.add_argument('--checkpoint-factor', default=20, type=int, help='')
    para_args = parser.parse_args()

    if para_args.recover is True:
        #  TODO this seems not safe
        checkpoint = torch.load(para_args.checkpoint_file)
        loaded_args = argparse.Namespace()
        loaded_args.__dict__ = checkpoint['arg']
        main(loaded_args, if_continue=True, checkpoint=checkpoint)
    else:
        main(para_args, if_continue=False, checkpoint=None)

# TODO torchsummarpy ; catch ctl-c; recover from last(writer path, model, optimizer, hyperparameter) hyperparameter
