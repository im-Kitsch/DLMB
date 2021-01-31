import torchvision
import torch
import torch.utils.data
import argparse

import util.dataset_util


IF_CUDA = True if torch.cuda.is_available() else False
DEVICE = torch.device('cuda') if IF_CUDA else torch.device('cpu')


class ConvEncoder(torch.nn.Module):
    def __init__(self, z_dim, n_class, img_ch, img_size):
        super(ConvEncoder, self).__init__()
        self.img_ch = img_ch
        self.img_size = img_size
        self.n_class = n_class

        return

    def forward(self):
        return


class ConvDecoder(torch.nn.Module):
    def __init__(self, z_dim, n_class, img_ch, img_size):
        super(ConvDecoder, self).__init__()
        self.img_ch = img_ch
        self.img_size = img_size
        self.n_class = n_class
        return

    def forward(self):
        return


class VAE(torch.nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        return

    def model(self):
        return

    def guide(self):
        return

    def generate_figure(self):
        return


def prepare_data(data_name, data_root, num_sup, batch_size):
    if data_name == 'MNIST':
        img_ch = 1
        img_size = 28
        img_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5] * img_ch, [0.5] * img_ch)
        ])

        data_train = torchvision.datasets.MNIST(root=data_root, transform=img_transform,
                                                train=True, download=True)
        data_test = torchvision.datasets.MNIST(root=data_root, transform=img_transform, train=False)
        n_class = len(data_train.classes)
    else:
        raise Exception('Not yet')

    if num_sup >= len(data_train):
        num_sup = len(data_train)
        sup_loader = torch.utils.data.DataLoader(dataset=data_train, batch_size=batch_size,
                                                 shuffle=True, drop_last=False)
        unsup_loader = None
    else:
        sup_data, unsup_data = util.dataset_util.split_dataset(data_train, num_sup)
        sup_loader = torch.utils.data.DataLoader(dataset=sup_data, batch_size=batch_size,
                                                 shuffle=True, drop_last=False)
        unsup_loader = torch.utils.data.DataLoader(dataset=unsup_data, batch_size=batch_size,
                                                   shuffle=True, drop_last=False)

    test_loader = torch.utils.data.DataLoader(dataset=data_test, batch_size=2056,
                                              shuffle=True, drop_last=False)
    return sup_loader, unsup_loader, test_loader, n_class, img_ch, img_size


def main(args):
    sup_loader, unsup_loader, test_loader, n_class, img_ch, img_size = \
        prepare_data(data_name=args.data, data_root=args.data_root,
                     num_sup=args.num_sup, batch_size=args.batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('--data', required=True, help='MNIST|Cifar10|HAM10000')
    parser.add_argument('--data-root', default='~/Documents/datas/MNIST_data/')
    parser.add_argument('-n', '--num-epochs', default=200, type=int, help='number of training epochs')
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--beta1', default=0.5)
    parser.add_argument('--beta2', default=0.999)

    parser.add_argument('--dim-noise', default=128, type=int)
    parser.add_argument('--batch-size', default=512)

    parser.add_argument('--num-sup', default=3000, help='if give a very large number(>=dataset size) '
                                                        'then only supervised learning')
    arg_para = parser.parse_args()
    main(arg_para)
