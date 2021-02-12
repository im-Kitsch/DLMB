import argparse
import torch
import torchvision

import util.dataset_util
from cwgan_gp import WGanGP
import matplotlib.pyplot as plt


def evaluate_model(ckpt_file, device, show_fig=False, save_path=None):
    """
    function for evaluation trained model
    if conditional training, return one grid figure with size n_class * 8 sub_figures
    if not conditional training, return grid figure with size 8 * 8  sub_figures
    args:
        ckpt_file: checkpoint file path
        device: cpu or gpu
        show_fig: if call plt.show(), default no
        save_path: save figure if not None
    return:
        return figure instance (class: matplotlib.figure.Figure)
    """
    # TODO break to load_model and generate_evaluation
    checkpoint = torch.load(ckpt_file)
    args = argparse.Namespace()
    # not safe to visit *.__dict__
    # loaded_args.__dict__ = checkpoint['arg']
    arg_dict = checkpoint['arg']
    for attr_key in arg_dict.keys():
        setattr(args, attr_key, arg_dict[attr_key])

    # load model
    train_data, test_data, img_shape = util.dataset_util.load_dataset(
        dataset_name=args.data, root=args.root, transform=None,
        csv_file=args.csv_file, percentage=args.data_percentage)
    n_ch, img_size, _ = img_shape

    dc_gan = WGanGP(data_name=args.data, n_ch=n_ch, img_size=img_size,
                    z_dim=args.z_dim, lr_g=args.lr_g, lr_d=args.lr_d,
                    lr_beta1=args.lr_beta1, lr_beta2=args.lr_beta2, d_step=args.d_step,
                    ndf=args.ndf, ngf=args.ngf, depth=args.depth,
                    if_condition=args.condition, n_class=len(train_data.dataset.classes),
                    embedding_dim=args.embedding_dim)
    del train_data, test_data

    dc_gan.conv_dis.load_state_dict(checkpoint['D'])
    dc_gan.conv_gen.load_state_dict(checkpoint['G'])
    dc_gan.opt_G.load_state_dict(checkpoint['opt_G'])
    dc_gan.opt_D.load_state_dict(checkpoint['opt_D'])
    checkpoint_epoc = checkpoint['epoch']
    torch.manual_seed(checkpoint['torch_seed'])
    log_dir = checkpoint['log_dir']

    with torch.no_grad():
        if dc_gan.if_condition:
            condition = torch.arange(dc_gan.n_class, device=device).reshape(-1, 1)
            condition = condition.repeat(1, 8).reshape(-1)
            gen_img = dc_gan.generate_fake(batch_size=64, condition=condition)
        else:
            gen_img = dc_gan.generate_fake(batch_size=64, condition=None)

    gen_img = torchvision.utils.make_grid(gen_img, nrow=8, padding=2, normalize=True)  # TODO normalize or use minmax?

    fig, ax = plt.subplots()
    ax.axis('off')
    ax.imshow(gen_img.cpu().numpy().transpose(1, 2, 0), interpolation='nearest')
    if show_fig:
        fig.show()
    if not (save_path is None):
        fig.savefig(save_path)
    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('--ckpt-file', required=True, help='check point file path')
    parser.add_argument('--device', default='cuda', type=str, choices=['cpu', 'cuda'], help='cpu or cuda')
    ckpt_args = parser.parse_args()

    t_device = torch.device(ckpt_args.device)
    evaluate_model(ckpt_args.ckpt_file, t_device, True)
