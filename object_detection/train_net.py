"""
Probabilistic Detectron Training Script following Detectron2 training script found at detectron2/tools.
"""
import core
import os
import sys
import mlflow

# This is very ugly. Essential for now but should be fixed.
sys.path.append(os.path.join(core.top_dir(), 'src', 'detr'))

# Detectron imports
from detectron2.utils.events import CommonMetricPrinter, TensorboardXWriter
import detectron2.utils.comm as comm
from detectron2.utils.file_io import PathManager
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.engine import launch
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators, verify_results

# Project imports
from core.setup import setup_config, setup_arg_parser
from core.setup import log_params_from_conf_node
from trainer import DefaultTrainer
from hooks import MLFlowWriter


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        """
        Builds evaluators for post-training mAP report.
        Args:
            cfg(CfgNode): a detectron2 CfgNode
            dataset_name(str): registered dataset name

        Returns:
            detectron2 DatasetEvaluators object
        """
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluators = [COCOEvaluator(dataset_name, cfg, True, output_folder)]
        return DatasetEvaluators(evaluators)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Builds DataLoader for test set.
        Args:
            cfg(CfgNode): a detectron2 CfgNode
            dataset_name(str): registered dataset name

        Returns:
            detectron2 DataLoader object specific to the test set.
        """
        return build_detection_test_loader(
            cfg, dataset_name)

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Builds DataLoader for train set.
        Args:
            cfg(CfgNode): a detectron2 CfgNode

        Returns:
            detectron2 DataLoader object specific to the train set.
        """
        return build_detection_train_loader(
            cfg)

    def build_writers(self):
        """
        Build a list of :class:`EventWriter` to be used.
        It now consists of a :class:`CommonMetricPrinter`,
        :class:`TensorboardXWriter` and :class:`MLFlowWriter`.

        Params used:
            output_dir: directory to store tensorboard events
            max_iter: the total number of iterations

        Returns:
            list[EventWriter]: a list of :class:`EventWriter` objects.
        """
        PathManager.mkdirs(self.cfg.OUTPUT_DIR)
        return [
            # It may not always print what you want to see, since it prints "common" metrics only.
            CommonMetricPrinter(self.max_iter),
            TensorboardXWriter(self.cfg.OUTPUT_DIR),
            MLFlowWriter()
        ]


def main(args):
    # Setup config node
    cfg = setup_config(args,
                       random_seed=args.random_seed)

    # For debugging only
    # cfg.defrost()
    # cfg.DATALOADER.NUM_WORKERS = 0
    # cfg.SOLVER.IMS_PER_BATCH = 1

    # Eval only mode to produce mAP results
    # Build Trainer from config node. Begin Training.
    # BDD as InD:
    # Random seed 0: freeze backbone, RPN
    # Random seed 1: No fine-tuning at all (use this seed to get samples from original architecture)
    # Random seed 2: freeze backbone, unfreeze rest, dropblock_size1 slurm-147627
    # Random seed 3: freeze backbone, unfreeze rest, dropblock_size4 slurm-147664

    # VOC as InD:
    # Random seed 0: No fine-tuning
    # Random seed 1: Freeze backbone, unfreeze rest, db size 4
    trainer = Trainer(cfg)

    if args.eval_only:
        model = trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer.resume_or_load(resume=args.resume)
    # MlFlow configuration
    # Either "Box Head Dropout" or "RPN Conv DB"
    experiment_name = "RPN Conv DB VOC"
    existing_exp = mlflow.get_experiment_by_name(experiment_name)
    if not existing_exp:
        mlflow.create_experiment(
            name=experiment_name)
    experiment = mlflow.set_experiment(experiment_name=experiment_name)
    mlflow.set_tracking_uri("./mlruns")

    # Define mlflow run to log metrics and parameters
    with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
        # Log parameters
        log_params_from_conf_node(cfg)
        return trainer.train()


if __name__ == "__main__":
    # Create arg parser
    arg_parser = setup_arg_parser()

    args = arg_parser.parse_args()
    print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
