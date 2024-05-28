import os
from typing import Any, cast, IO

import mlflow
from mlflow.models.signature import infer_signature
from collections import defaultdict

import detectron2.utils.comm as comm
import torch
from detectron2.utils.events import get_event_storage, EventWriter
from detectron2.engine.train_loop import HookBase
from detectron2.evaluation.testing import flatten_results_dict
from detectron2.checkpoint import DetectionCheckpointer


class MLFlowWriter(EventWriter):
    """
    Logs files using MLFlow. No need to open or close a specific file
    because mlflow automatically creates a folder with the current run
    and logs accordingly
    """

    def __init__(self, window_size=20):
        """
        Args:
            window_size (int): the window size of median smoothing for the scalars whose
                `smoothing_hint` are True.
        """
        self._window_size = window_size
        self._last_write = -1

    def write(self):
        storage = get_event_storage()
        to_save = defaultdict(dict)

        for k, (v, iter) in storage.latest_with_smoothing_hint(self._window_size).items():
            # keep scalars that have not been written
            if iter <= self._last_write:
                continue
            to_save[iter][k] = v

        if len(to_save):
            all_iters = sorted(to_save.keys())
            self._last_write = max(all_iters)

        for itr, scalars_per_iter in to_save.items():
            mlflow.log_metrics(scalars_per_iter, step=int(itr))


class EvalHook(HookBase):
    """
    Run an evaluation function periodically, and at the end of training.
    Logs metrics with MLFlow
    It is executed every ``eval_period`` iterations and after the last iteration.
    """

    def __init__(self, eval_period, eval_function, eval_after_train=True):
        """
        Args:
            eval_period (int): the period to run `eval_function`. Set to 0 to
                not evaluate periodically (but still evaluate after the last iteration
                if `eval_after_train` is True).
            eval_function (callable): a function which takes no arguments, and
                returns a nested dict of evaluation metrics.
            eval_after_train (bool): whether to evaluate after the last iteration

        Note:
            This hook must be enabled in all or none workers.
            If you would like only certain workers to perform evaluation,
            give other workers a no-op function (`eval_function=lambda: None`).
        """
        self._period = eval_period
        self._func = eval_function
        self._eval_after_train = eval_after_train

    def _do_eval(self):
        results = self._func()

        if results:
            assert isinstance(
                results, dict
            ), "Eval function must return a dict. Got {} instead.".format(results)

            flattened_results = flatten_results_dict(results)
            for k, v in flattened_results.items():
                try:
                    v = float(v)
                except Exception as e:
                    raise ValueError(
                        "[EvalHook] eval_function should return a nested dict of float. "
                        "Got '{}: {}' instead.".format(k, v)
                    ) from e
            self.trainer.storage.put_scalars(**flattened_results, smoothing_hint=False)

        # Evaluation may take different time among workers.
        # A barrier make them start the next iteration together.
        comm.synchronize()

    def before_train(self):
        # When fine-tuning, save evaluation at first iteration to compare afterward
        if self.trainer.iter == 0:
            self._do_eval()

    def after_step(self):
        next_iter = self.trainer.iter + 1
        if self._period > 0 and next_iter % self._period == 0:
            # do the last eval in after_train
            if next_iter != self.trainer.max_iter:
                self._do_eval()

    def after_train(self):
        # This condition is to prevent the eval from running after a failed training
        if self._eval_after_train and self.trainer.iter + 1 >= self.trainer.max_iter:
            self._do_eval()
        # func is likely a closure that holds reference to the trainer
        # therefore we clean it to avoid circular reference in the end
        del self._func


class MLFlowCheckpointer(DetectionCheckpointer):
    """
    This class extends the capability of the previous checkpointer by implementing an MLFlow
    model saver and logger. It still saves the checkpoint and state dict, but also creates an extra
    artifact only to perform mlflow inference and possible model serving.

    This class can save either the best or the last model according to the defined metric in the config
    """
    def save(self, name: str, **kwargs: Any) -> None:
        """
        Dump model and checkpointables to a file, and log it with MLFlow

        Args:
            name (str): name of the file.
            kwargs (dict): extra arbitrary data to save.
        """
        if not self.save_dir or not self.save_to_disk:
            return

        data = {}
        data["model"] = self.model.state_dict()
        for key, obj in self.checkpointables.items():
            data[key] = obj.state_dict()
        data.update(kwargs)

        # Choose mlflow model name, depending on if it's the best or the last
        if 'best' in name:
            mlflow_model_name = 'best'
        else:
            mlflow_model_name = 'last'
        basename = "{}.pth".format(name)
        save_file = os.path.join(self.save_dir, basename)
        assert os.path.basename(save_file) == basename, basename
        self.logger.info("Saving checkpoint to {}".format(save_file))
        with self.path_manager.open(save_file, "wb") as f:
            torch.save(data, cast(IO[bytes], f))
        # Mlflow log / save
        mlflow.pytorch.log_model(pytorch_model=self.model,
                                 artifact_path=mlflow_model_name,
                                 )
        self.tag_last_checkpoint(basename)
