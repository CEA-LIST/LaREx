from deeplab_v3p import DeepLabV3PlusModule
import torch
import argparse
import pytorch_lightning as pl
from dataset import WoodScapeDataset
from dataset import WoodScapeDataModule
from dataset import Cityscapes
from dataset import CityscapesDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.plugins.environments import SLURMEnvironment
from pytorch_lightning.callbacks import LearningRateMonitor
from pl_bolts.callbacks import TrainingDataMonitor
from pl_bolts.callbacks import PrintTableMetricsCallback
from torchvision import transforms
from icecream import ic


def main(args):
    #####################
    #      Get Args     #
    #####################
    model_type = args.model  # DeepLabV3+ model type
    batch_size = args.batch
    max_nro_epochs = args.epochs
    random_seed_everything = args.random_seed
    dataset = args.dataset
    dataset_path = args.dataset_path
    img_h = args.img_h
    img_w = args.img_w
    loss_type = args.loss_type
    load_pretrained = args.load_pretrained
    fine_tuning = args.fine_tuning
    slurm_training = args.slurm_training
    tqmd_bar = args.tqmd_bar

    print(' ')
    print('=' * 60)
    print('Training Experiment Summary: \r\n')
    ic(model_type)
    ic(dataset)
    ic(img_h)
    ic(img_w)
    ic(max_nro_epochs)
    ic(batch_size)
    ic(loss_type)
    ic(load_pretrained)
    ic(fine_tuning)
    ic(slurm_training)
    ic(random_seed_everything)
    print('=' * 60)
    print(' ')

    ############################
    #      Seed Everything     #
    ############################
    pl.seed_everything(random_seed_everything)
    datamodule_seed = 100

    #######################################
    #      Training Monitor/Callbacks     #
    #######################################
    checkpoint_callback = ModelCheckpoint(monitor="validation_IoU",
                                          mode='max',
                                          every_n_epochs=1,
                                          save_top_k=3,
                                          save_last=True,
                                          save_on_train_epoch_end=False)

    monitor = TrainingDataMonitor(log_every_n_steps=20)
    # table_metrics_callback = PrintTableMetricsCallback()
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    if tqmd_bar:
        progress_bar = TQDMProgressBar(refresh_rate=10)

    else:
        progress_bar = RichProgressBar(theme=RichProgressBarTheme(description="green_yellow",
                                                                  progress_bar="green1",
                                                                  progress_bar_finished="green1",
                                                                  batch_progress="green_yellow",
                                                                  time="grey82",
                                                                  processing_speed="grey82",
                                                                  metrics="grey82"))

    ###############################
    #      Get Dataset Module     #
    ###############################
    if dataset == 'woodscape':
        cmap = {0: [0, 0, 0],  # "void"
                1: [128, 64, 128],  # "road",
                2: [69, 76, 11],  # "lanemarks",
                3: [0, 255, 0],  # "curb",
                4: [220, 20, 60],  # "person",
                5: [255, 0, 0],  # "rider",
                6: [0, 0, 142],  # "vehicles",
                7: [119, 11, 32],  # "bicycle",
                8: [0, 0, 230],  # "motorcycle",
                9: [220, 220, 0]  # "traffic_sign",
                }

        # same values as in VainF Repository! - Probably not the best Values for Woodscapes!
        norm_mean = [0.485, 0.456, 0.406]
        norm_std = [0.229, 0.224, 0.225]

        data_module = WoodScapeDataModule(dataset_dir=dataset_path,
                                          img_size=(img_h, img_w),
                                          batch_size=batch_size,
                                          default_transforms=True,
                                          label_colours=cmap,
                                          norm_mean=norm_mean,
                                          norm_std=norm_std,
                                          seed=datamodule_seed,
                                          drop_last=True)

        data_module.setup()
        data_module.train_dataloader()
        data_module.val_dataloader()
        train_loader_len = data_module.train_loader_len
        val_loader_len = data_module.valid_loader_len
        ic(train_loader_len)
        ic(val_loader_len)
        num_classes = 10

    elif dataset == 'cityscapes':
        data_module = CityscapesDataModule(data_dir=dataset_path,
                                           batch_size=batch_size,
                                           img_size=(img_h, img_w),
                                           target_type='semantic',
                                           num_workers=10,
                                           drop_last=True,
                                           default_transforms=True)
        num_classes = 20
        cmap = None
        train_loader_len = None
        val_loader_len = None

    else:
        raise ValueError(" Dataset Not found! Choose a valid Semantic Segmentation dataset!")

    #############################
    #      Get Model Module     #
    #############################
    if model_type == "deeplabv3p-variational-layer":
        model_module = DeepLabV3PlusModule(deeplabv3plus_type="variational_layer",
                                           dataset=dataset,
                                           n_class=num_classes,
                                           pred_loss_type=loss_type,
                                           optimizer_lr=0.001,
                                           img_pred_weight=1.0,
                                           len_train_loader_beta=train_loader_len,
                                           len_val_loader_beta=val_loader_len,
                                           max_nro_epochs=max_nro_epochs,
                                           label_colours=cmap,
                                           test_images_path="./test_images/*.png")

    elif model_type == "deeplabv3p-backbone-dropblock2d":
        model_module = DeepLabV3PlusModule(deeplabv3plus_type="backbone_dropblock2d",
                                           dataset=dataset,
                                           n_class=num_classes,
                                           pred_loss_type=loss_type,
                                           optimizer_lr=0.001,
                                           max_nro_epochs=max_nro_epochs,
                                           label_colours=cmap,
                                           test_images_path="./test_images/*.png")

    else:
        model_module = DeepLabV3PlusModule(deeplabv3plus_type="normal",
                                           dataset=dataset,
                                           n_class=num_classes,
                                           pred_loss_type=loss_type,
                                           optimizer_lr=0.01,
                                           max_nro_epochs=max_nro_epochs,
                                           label_colours=cmap,
                                           test_images_path="./test_images/*.png")

    if load_pretrained:
        if dataset == 'woodscape':
            chk_pt = "./deeplab_v3p/pre_trained/Model_Woodscape.pth"
        else:  # if dataset == 'cityscapes':
            chk_pt = "./deeplab_v3p/pre_trained/Model_CityScapes.ckpt"

        model_module.deeplab_v3plus_model.load_state_dict(torch.load(chk_pt), strict=False)
        # chk_pt = "./deeplab_v3p/pre_trained/Model_Woodscape.pth"
        # model_module.deeplab_v3plus_model.load_state_dict(torch.load(chk_pt), strict=False)

    ########################################
    #      Start Module/Model Training     #
    ########################################
    if slurm_training:  # slurm training on HPC with multiple GPUs
        ic(slurm_training)
        trainer = pl.Trainer(accelerator='gpu',
                             devices=4,
                             num_nodes=1,
                             strategy='ddp',
                             max_epochs=max_nro_epochs,
                             callbacks=[progress_bar,
                                        checkpoint_callback,
                                        lr_monitor],
                             plugins=[SLURMEnvironment(auto_requeue=False)])

    else:  # training locally in computer with GPU
        ic(slurm_training)
        trainer = pl.Trainer(accelerator='gpu',
                             devices=-1,
                             max_epochs=max_nro_epochs,
                             callbacks=[progress_bar,
                                        checkpoint_callback,
                                        lr_monitor])

    trainer.fit(model_module, datamodule=data_module)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument(
        '-m',
        '--model',
        dest='model',
        default='deeplabv3p-normal',
        choices=['deeplabv3p-normal',
                 'deeplabv3p-variational-layer',
                 'deeplabv3p-backbone-dropblock2d'],
        type=str,
        help='DeepLabV3+ model type')

    argparser.add_argument(
        '-e',
        '--epochs',
        dest='epochs',
        default=1,
        type=int,
        help='Max number of epochs for Training')

    argparser.add_argument(
        '-b',
        '--batch',
        dest='batch',
        default=16,
        type=int,
        help='Batch size')

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
        help='Random Seed Everything')

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
        help='Load DeepLabV3+ pretrained model')
    argparser.set_defaults(load_pretrained=False)

    argparser.add_argument(
        '--fine_tuning',
        dest='fine_tuning',
        action='store_true',
        help='Model Fine Tuning (update all parameters after loading a pretrained model)')
    argparser.set_defaults(fine_tuning=False)

    argparser.add_argument(
        '--imgh',
        dest='img_h',
        default=483,
        type=int,
        choices=[128, 256, 483],
        help='Dataset image height')

    argparser.add_argument(
        '--imgw',
        dest='img_w',
        default=640,
        type=int,
        choices=[256, 512, 640],
        help='Dataset image width')

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
    args = argparser.parse_args()

    main(args)



