from argument_parser import argpument_parser
import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.plugins.environments import SLURMEnvironment
from pytorch_lightning.callbacks import LearningRateMonitor
from pl_bolts.callbacks import TrainingDataMonitor
from dataset import WoodScapeDataModule
from dataset import CityscapesDataModule
from models import UnetSemSegModule
from icecream import ic


def main(args):
    torch.cuda.empty_cache()
    #####################
    #      Get Args     #
    #####################
    max_nro_epochs = args.epochs
    batch_size = args.batch_size
    random_seed_everything = args.random_seed
    dataset = args.dataset
    dataset_path = args.dataset_path
    loss_type = args.loss_type
    img_h = args.img_h
    img_w = args.img_w
    gpus_nro = args.gpus
    load_pretrained = args.load_pretrained
    resume_training = args.resume_training
    model_path = args.model_path
    slurm_training = args.slurm_training
    tqmd_bar = args.tqmd_bar

    print(' ')
    print('=' * 60)
    ic(dataset)
    ic(img_h)
    ic(img_w)
    ic(gpus_nro)
    ic(max_nro_epochs)
    ic(batch_size)
    ic(loss_type)
    ic(load_pretrained)
    ic(resume_training)
    ic(model_path)
    ic(random_seed_everything)
    ic(slurm_training)
    print('=' * 60)
    print(' ')
    ############################
    #      Seed Everything     #
    ############################
    pl.seed_everything(random_seed_everything)
    #######################################
    #      Training Monitor/Callbacks     #
    #######################################
    # checkpoint_callback = ModelCheckpoint(monitor="validation_IoU",
    #                                       mode='max',
    #                                       every_n_epochs=5,
    #                                       save_top_k=2,
    #                                       save_last=True,
    #                                       save_on_train_epoch_end=False)

    checkpoint_callback = ModelCheckpoint(monitor="validation_loss",
                                          mode='min',
                                          every_n_epochs=5,
                                          save_top_k=2,
                                          save_last=True,
                                          save_on_train_epoch_end=False)

    monitor = TrainingDataMonitor(log_every_n_steps=20)
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
        data_module = WoodScapeDataModule(dataset_dir=dataset_path,
                                          batch_size=batch_size,
                                          img_size=(img_h, img_w),
                                          num_workers=12,
                                          drop_last=True,
                                          default_transforms=True)
        num_classes = 10

    elif dataset == 'cityscapes':
        data_module = CityscapesDataModule(data_dir=dataset_path,
                                           batch_size=batch_size,
                                           img_size=(img_h, img_w),
                                           target_type='semantic',
                                           num_workers=12,
                                           drop_last=True,
                                           default_transforms=False,
                                           default_img_mask_transforms=True)
        num_classes = 20

    else:
        raise ValueError(" Dataset Not found! Choose a valid Semantic Segmentation dataset!")

    ic(num_classes)
    #############################
    #      Get Model Module     #
    #############################
    if num_classes == 10:  # Woodscape-10 classes requires a simpler model
        num_filters = [32, 64, 128]
    elif num_classes == 20:  # Cityscapes dataset requires a larger DNN
        num_filters = [32, 64, 128, 256]
    else:
        num_filters = [64, 128, 256, 512]
    unet_semseg_model = UnetSemSegModule(input_channels=3,
                                         num_classes=num_classes,
                                         num_filters=num_filters,
                                         drop_block2d=True,
                                         lr=2.5e-5,
                                         pred_loss_type=loss_type,
                                         max_nro_epochs=max_nro_epochs)

    ##########################################
    #         Load Pre-Trained Model?        #
    ##########################################
    if load_pretrained:
        ic(load_pretrained)
        unet_semseg_model.load_state_dict(torch.load(model_path,
                                                     map_location=lambda storage, loc: storage)['state_dict'],
                                          strict=False)
    ########################################
    #      Start Module/Model Training     #
    ########################################
    if slurm_training:  # slurm training on HPC with multiple GPUs
        ic(slurm_training)
        ic(gpus_nro)
        trainer = pl.Trainer(accelerator='gpu',
                             devices=gpus_nro,
                             num_nodes=1,
                             strategy='ddp',
                             max_epochs=max_nro_epochs,
                             callbacks=[progress_bar,
                                        checkpoint_callback,
                                        monitor,
                                        lr_monitor],
                             plugins=[SLURMEnvironment(auto_requeue=False)])

    else:  # training locally in computer with GPU
        ic(slurm_training)
        trainer = pl.Trainer(accelerator='gpu',
                             devices=-1,
                             max_epochs=max_nro_epochs,
                             callbacks=[progress_bar,
                                        checkpoint_callback,
                                        monitor,
                                        lr_monitor])

    if resume_training:
        ic(resume_training)
        trainer.fit(unet_semseg_model, datamodule=data_module, ckpt_path=model_path)
    else:
        ic(resume_training)
        trainer.fit(unet_semseg_model, datamodule=data_module)


if __name__ == '__main__':
    args = argpument_parser()
    main(args)
