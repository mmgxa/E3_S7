from typing import Tuple, Dict, List
import yaml
import os
from pathlib import Path

import lightning as L
from lightning.pytorch.tuner import Tuner
import torch
import hydra
from omegaconf import DictConfig
import optuna

from dlearn import utils

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def train(cfg: DictConfig) -> Tuple[dict, dict]:
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    ### update cfg using optuna/tuner results
    dir_path = Path(hydra.utils.get_original_cwd())
    dp =  dir_path / 'outputs' / 'train' / 'multiruns'
    files = os.listdir(dp)[-1] # latest run
    optuna_file = dp / files / 'optimization_results.yaml'
    

    loaded_study = optuna.load_study(study_name="hp-gpt-optuna", storage="sqlite:///example.db")
    with open(optuna_file, 'r') as optuna_res:
        optuna_dict = yaml.safe_load(optuna_res)
    best_id = 0
    for trial in loaded_study.trials:
        if optuna_dict['best_value'] == trial.value:
            log.info(f"best trial found !")
            break
        best_id += 1
    
    log.info(f"best trial is : <{best_id}>")

    tuner_file = dp / files / str(best_id) / 'tuner_results.yaml'

    with open(tuner_file, 'r') as tuner_res:
        tuner_dict = yaml.safe_load(tuner_res)

    best_param = optuna_dict['best_params']
    model_hp1 = {key[6:]: val for key, val in best_param.items() if key.startswith('model')}
    model_hp2 = {key[6:]: val for key, val in tuner_dict.items() if key.startswith('model')}
    datamodule_hp1 = {key[5:]: val for key, val in best_param.items() if key.startswith('data')}
    datamodule_hp2 = {key[11:]: val for key, val in tuner_dict.items() if key.startswith('datamodule')}

    cfg.model.update(model_hp1)
    cfg.model.update(model_hp2)
    cfg.data.update(datamodule_hp1)
    cfg.data.update(datamodule_hp2)
    
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

    if cfg.get("compile"):
        model = torch.compile(model)
    
    if cfg.get("train"):
        log.info("Starting training!")    
        trainer.fit(model=model, datamodule=datamodule,
                    ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    log.info("Scripting Model...")
    scripted_model = model.to("cpu").to_torchscript(method="script")
    path = f"{cfg.paths.output_dir}/model.script.pt"
    torch.jit.save(scripted_model, path)

    log.info("Tracing Model...")
    traced_model = model.to("cpu").to_torchscript(method="trace", example_inputs=(torch.randint(0, 128, (1, model.block_size)), torch.randint(0, 128, (1, model.block_size))))
    path = f"{cfg.paths.output_dir}/model.trace.pt"
    torch.jit.save(traced_model, path)


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
