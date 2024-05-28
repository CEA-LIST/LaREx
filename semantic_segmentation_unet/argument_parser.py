import argparse


def argpument_parser():

    argparser = argparse.ArgumentParser(description=__doc__)
    
    argparser.add_argument(
        '-e',
        '--epochs',
        dest='epochs',
        default=20,
        type=int,
        help='Max number of epochs for Training'
    )

    argparser.add_argument(
        '-b',
        '--batchsize',
        dest='batch_size',
        default=16,
        type=int,
        help='Batch size'
    )

    argparser.add_argument(
        "--loss_type",
        dest="loss_type",
        type=str,
        default='cross_entropy',
        choices=['cross_entropy', 'focal_loss'],
        help="Loss type (default: cross_entropy)")

    argparser.add_argument(
        '-s',
        '--seed',
        dest='random_seed',
        default=9290,
        type=int,
        help='Random Seed Everything'
    )

    argparser.add_argument(
        '-d',
        '--dataset',
        dest='dataset',
        default='woodscape',
        choices=['cityscapes', 'woodscape'],
        type=str,
        help='Dataset used for training'
    )

    argparser.add_argument(
        '-p',
        '--datapath',
        dest='dataset_path',
        default='./Data/WoodScape/',
        type=str,
        help='Dataset path'
    )

    argparser.add_argument(
        '--load_pretrained',
        dest='load_pretrained',
        action='store_true',
        help='Load U-Net pretrained model')
    argparser.set_defaults(load_pretrained=False)

    argparser.add_argument(
        '--resume_training',
        dest='resume_training',
        action='store_true',
        help='Resume U-Net model training')
    argparser.set_defaults(resume_training=False)

    argparser.add_argument(
        '--model_path',
        dest='model_path',
        default='./lighting_logs/',
        type=str,
        help='U-Net pretrained model path')

    argparser.add_argument(
        '--imgh',
        dest='img_h',
        default=128,
        choices=[128, 256, 483, 512],
        type=int,
        help='Dataset image height'
    )

    argparser.add_argument(
        '--imgw',
        dest='img_w',
        default=256,
        choices=[128, 256, 512, 640],
        type=int,
        help='Dataset image width'
    )

    argparser.add_argument(
        '-g',
        '--gpus',
        dest='gpus',
        default=1,
        type=int,
        help='Number of GPUs'
    )

    argparser.add_argument(
        '--slurm',
        dest='slurm_training',
        action='store_true',
        help='slurm training on HPC')
    argparser.set_defaults(slurm_training=False)

    argparser.add_argument(
        '--tqmd',
        dest='tqmd_bar',
        action='store_true',
        help='TQMD Progress bar')
    argparser.set_defaults(tqmd_bar=False)

    # parse all the arguments
    arguments = argparser.parse_args()
    return arguments