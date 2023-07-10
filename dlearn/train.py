from typing import Tuple, Dict, List
import json

import lightning as L
from lightning.pytorch.tuner import Tuner
import torch
import hydra
from omegaconf import DictConfig
from pathlib import Path
from optuna import Trial

from dlearn import utils

log = utils.get_pylogger(__name__)

@utils.task_wrapper
def train(cfg: DictConfig) -> Tuple[dict, dict]:
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: L.LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[L.Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[L.LightningLoggerBase] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: L.Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    if cfg.get("compile"):
        model = torch.compile(model)
        
    if cfg.get("tune"):
        # setup
        # datamodule.setup()
        tuner = Tuner(trainer)

        # learnig rate
        log.info("Tuning LR ...!")
        lr_finder = tuner.lr_find(model, datamodule)
        log.info(f"best initial lr={lr_finder.suggestion()}")
        log.info(f"best initial lr={model.hparams.learning_rate}")
        cfg.model.update({'learning_rate': model.hparams.learning_rate})
        
        # batch size
        log.info("Tuning Batch size ...!")
        tuner.scale_batch_size(model, datamodule, mode="power")
        log.info(f"optimal batch size = {datamodule.hparams.batch_size}")
        
        tuner_results = { 'model.learning_rate': model.hparams.learning_rate, 'datamodule.batch_size': datamodule.hparams.batch_size }
        dir_path = Path(cfg.paths.output_dir)
        yaml_file = dir_path / 'tuner_results.yaml'
        with open(yaml_file, 'w') as f:
            f.write(json.dumps(tuner_results))
       
        cfg.data.update({'batch_size': datamodule.hparams.batch_size})
        

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule,
                    ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning(
                "Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Training Complete")
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    # train the model
    metric_dict, _ = train(cfg)

    # this will be used by hydra later for optimization
    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
