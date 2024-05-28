import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.callbacks import LearningRateMonitor
from pl_bolts.callbacks import TrainingDataMonitor
from pl_bolts.datamodules import CIFAR10DataModule
from pytorch_lightning.plugins.environments import SLURMEnvironment
from datasets.cifar10 import get_cifar10_input_transformations
from models import ResnetModule
from datasets import GtsrbModule
from icecream import ic
import mlflow.pytorch
import hydra
from omegaconf import DictConfig
from helper_functions import log_params_from_omegaconf_dict
from datetime import datetime


@hydra.main(version_base=None, config_path="configs/", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    assert cfg.ind_dataset in ("gtsrb", "cifar10")
    assert cfg.ind_dataset in cfg.data_dir
    assert cfg.model.spectral_norm_only_fc + cfg.model.spectral_norm <= 1
    #####################
    #      Get Args     #
    #####################
    model_type = cfg.model.model_name
    max_nro_epochs = cfg.trainer.epochs
    batch_size = cfg.datamodule.batch_size
    random_seed_everything = cfg.seed
    dataset_path = cfg.data_dir
    loss_type = cfg.model.loss_type
    rich_progbar = cfg.rich_progbar
    slurm_training = cfg.slurm
    gpus_nro = cfg.trainer.gpus
    # Get current date time to synchronize pl logs and mlflow
    current_date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    print(' ')
    print('=' * 60)
    ic(current_date_time)
    ic(model_type)
    ic(max_nro_epochs)
    ic(batch_size)
    ic(loss_type)
    ic(random_seed_everything)
    ic(slurm_training)
    ic(gpus_nro)
    print('=' * 60)
    print(' ')

    ############################
    #      Seed Everything     #
    ############################
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    pl.seed_everything(random_seed_everything)
    #######################################
    #      Training Monitor/Callbacks     #
    #######################################
    checkpoint_callback = ModelCheckpoint(dirpath=f'lightning_logs/{current_date_time}_ind_{cfg.ind_dataset}',
                                          monitor=cfg.callbacks.model_checkpoint.monitor,
                                          mode=cfg.callbacks.model_checkpoint.mode,
                                          every_n_epochs=cfg.callbacks.model_checkpoint.every_n_epochs,
                                          save_top_k=cfg.callbacks.model_checkpoint.save_top_k,
                                          save_last=cfg.callbacks.model_checkpoint.save_last,
                                          save_on_train_epoch_end=cfg.callbacks.model_checkpoint.save_on_train_epoch_end)

    monitor = TrainingDataMonitor(log_every_n_steps=cfg.trainer.log_every_n_step)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    if rich_progbar:  # fancy aesthetic progress bar
        progress_bar = RichProgressBar(theme=RichProgressBarTheme(description="green_yellow",
                                                                  progress_bar="green1",
                                                                  progress_bar_finished="green1",
                                                                  batch_progress="green_yellow",
                                                                  time="grey82",
                                                                  processing_speed="grey82",
                                                                  metrics="grey82"))
    else:  # normal aesthetic progress bar
        progress_bar = TQDMProgressBar(refresh_rate=cfg.trainer.progress_bar_refresh_rate)

    ###############################
    #      Get Dataset Module     #
    ###############################
    if cfg.ind_dataset == "gtsrb":
        data_module = GtsrbModule(data_path=dataset_path,
                                  img_size=(cfg.datamodule.image_width, cfg.datamodule.image_height),
                                  batch_size=cfg.datamodule.batch_size,
                                  shuffle=cfg.datamodule.shuffle)

        data_module.setup(stage="fit")
        data_module.setup(stage="validate")
        data_module.setup(stage="test")
        num_classes = len(data_module.ds_gtsrb_train.classes)
    # Cifar10
    else:
        train_transforms, test_transforms = get_cifar10_input_transformations(
            cifar10_normalize_inputs=cfg.datamodule.cifar10_normalize_inputs,
            img_size=cfg.datamodule.image_width,
            data_augmentations=cfg.datamodule.data_augmentations,
            anomalies=False
        )
        data_module = CIFAR10DataModule(data_dir=dataset_path,
                                        batch_size=cfg.datamodule.batch_size,
                                        train_transforms=train_transforms,
                                        test_transforms=test_transforms,
                                        val_transforms=test_transforms
                                        )
        num_classes = data_module.num_classes
    ic(num_classes)
    #############################
    #      Get Model Module     #
    #############################
    model_module = ResnetModule(arch_name=model_type,
                                input_channels=cfg.model.input_channels,
                                num_classes=num_classes,
                                spectral_norm=cfg.model.spectral_norm,
                                dropblock=cfg.model.drop_block,
                                dropblock_prob=cfg.model.dropblock_prob,
                                dropblock_location=cfg.hook_location,
                                dropblock_block_size=cfg.model.dropblock_block_size,
                                dropout=cfg.model.dropout,
                                dropout_prob=cfg.model.dropout_prob,
                                loss_fn=cfg.model.loss_type,
                                optimizer_lr=cfg.model.lr,
                                optimizer_weight_decay=cfg.model.weight_decay,
                                max_nro_epochs=max_nro_epochs,
                                activation=cfg.model.activation,
                                avg_pool=cfg.model.avg_pool,
                                ash=False,
                                ash_percentile=cfg.ash_percentile,
                                dice_precompute=False,
                                dice_inference=False,
                                dice_p=cfg.dice_p,
                                dice_info=None,
                                react_threshold=None,
                                spectral_norm_only_fc=cfg.model.spectral_norm_only_fc,
                                batch_norm=cfg.model.batch_norm
                                )

    ########################################
    #      Start Module/Model Training     #
    ########################################
    if slurm_training:  # slurm training on HPC with multiple GPUs
        ic(slurm_training)
        trainer = pl.Trainer(accelerator='gpu',
                             devices=gpus_nro,
                             num_nodes=1,
                             strategy=cfg.trainer.distributed_strategy,
                             max_epochs=max_nro_epochs,
                             callbacks=[progress_bar,
                                        checkpoint_callback,
                                        monitor,
                                        lr_monitor],
                             plugins=[SLURMEnvironment(auto_requeue=cfg.trainer.slurm_auto_requeue)])

    else:  # training locally in computer with GPU
        ic(slurm_training)
        trainer = pl.Trainer(accelerator='gpu' if torch.cuda.is_available() else None,
                             devices=gpus_nro,
                             max_epochs=max_nro_epochs,
                             callbacks=[progress_bar,
                                        checkpoint_callback,
                                        monitor,
                                        lr_monitor])

    # Log parameters with mlflow
    log_params_from_omegaconf_dict(cfg)
    # Setup automatic logging of training with mlflow
    mlflow.pytorch.autolog(log_every_n_step=cfg.trainer.log_every_n_step)
    # Fit Trainer
    trainer.fit(model=model_module, datamodule=data_module)  # fit a model!


if __name__ == "__main__":
    main()
