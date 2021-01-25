import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, image_channels, features_d, ngpu=1):
        super().__init__()
        self.ngpu = ngpu
        self.disc = nn.Sequential(
            # input dims: N x 3 x 64 x 64
            nn.Conv2d(
                image_channels, features_d, kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(0.2),
            # _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            # Conv2d down to 1x1
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        if x.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.disc, x, range(self.ngpu))
        else:
            output = self.disc(x)
        return output.view(-1, 1).squeeze(1)


class Generator(nn.Module):
    def __init__(self, noise_vector, image_channels, features_g, ngpu=1):
        super().__init__()
        self.ngpu = ngpu
        self.gan = nn.Sequential(
            # input dims: N x noise_vector x 1 x 1
            self._block(noise_vector, features_g * 16, 4, 1, 0),  # img: 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # img: 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # img: 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # img: 32x32
            nn.ConvTranspose2d(
                features_g * 2, image_channels, kernel_size=4, stride=2, padding=1
            ),
            # output dims: N x 3 x 64 x 64
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        if x.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.gan, x, range(self.ngpu))
        else:
            output = self.gan(x)
        return output


def init_weights(model):
    """
    Initialize weights based on DC-GAN paper
    :param model:
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)