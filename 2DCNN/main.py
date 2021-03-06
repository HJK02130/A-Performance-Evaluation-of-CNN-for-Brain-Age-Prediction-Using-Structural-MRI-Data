import argparse
import importlib
import json
import logging
import os
import pprint
import sys

import dill
import torch
import wandb
from box import Box
from torch.utils.data import DataLoader

from src.common.dataset import get_dataset
from lib.utils import logging as logging_utils, os as os_utils, optimizer as optimizer_utils
from lib.base_trainer import Trainer
import easydict

def parser_setup():
    # define argparsers

    str2bool = os_utils.str2bool
    listorstr = os_utils.listorstr
    
    parser = easydict.EasyDict({
        "debug":False,
        "config":None,
        "seed":0,
        "wandb_use":False,
        "wandb_run_id":None,
        "wandb.watch":False,
        "project":"brain-age",
        "exp_name":None,
        "device":"cuda",
        "result_folder":"a",
        "mode":["test", "train"],
        "statefile":None,

        "data" : {
            "name":"brain_age",
            "root_path":"**root path",
            "train_csv":"**train csv",
            "valid_csv":"**valid csv",
            "test_csv":"**test csv",
            "feat_csv":None,
            "train_num_sample":-1,
            "frame_dim":1,
            "frame_keep_style":"random",
            "frame_keep_fraction":1,
            "impute":"drop",
        },

        "model" : {
            "name":"regression",
            "arch": {
                "file":"src/arch/brain_age_3d.py",
                "lstm_feat_dim":2,
                "lstm_latent_dim":128,
                "attn_num_heads":1,
                "attn_dim":32,
                "attn_drop":False,
                "agg_fn":"attention"
            }
        },
        
        "train":{
            "batch_size":8,

            "patience":100,
            "max_epoch":100,
            "optimizer":"adam",
            "lr":1e-3,
            "weight_decay":1e-4,
            "gradient_norm_clip":-1,

            "save_strategy":["best", "last"],
            "log_every":100,

            "stopping_criteria":"loss",
            "stopping_criteria_direction":"lower",
            "evaluations":None,

            "optimizer_momentum":None,

            "scheduler":None,
            "scheduler_gamma":None,
            "scheduler_milestones":None,
            "scheduler_patience":None,
            "scheduler_step_size":None,
            "scheduler_load_on_reduce":None,
        },
        
        "test":{
            "batch_size":8,
            "evaluations":None,
            "eval_model":"best",
        },

        "_actions":None,
        "_defaults":None
    })
   
    print(parser.seed)
    return parser

if __name__ == "__main__":
    # set seeds etc here
    torch.backends.cudnn.benchmark = True

    # define logger etc
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger = logging.getLogger()

    parser = parser_setup()
    #config = os_utils.parse_args(parser)
    config = parser

    logger.info("Config:")
    logger.info(pprint.pformat(config, indent=4))

    os_utils.safe_makedirs(config.result_folder)
    statefile, run_id, result_folder = os_utils.get_state_params(
        config.wandb_use, config.wandb_run_id, config.result_folder, config.statefile
    )
    config.statefile = statefile
    config.wandb_run_id = run_id
    config.result_folder = result_folder

    if statefile is not None:
        data = torch.load(open(statefile, "rb"), pickle_module=dill)
        epoch = data["epoch"]
        if epoch >= config.train.max_epoch:
            logger.error("Aleady trained upto max epoch; exiting")
            sys.exit()

    if config.wandb_use:
        wandb.init(
            name=config.exp_name if config.exp_name is not None else config.result_folder,
            config=config.to_dict(),
            project=config.project,
            dir=config.result_folder,
            resume=config.wandb_run_id,
            id=config.wandb_run_id,
            sync_tensorboard=True,
        )
        logger.info(f"Starting wandb with id {wandb_run_id}")

    # NOTE: WANDB creates git patch so we probably can get rid of this in future
    os_utils.copy_code("src", config.result_folder, replace=True,)
    json.dump(
        config,
        open(f"{wandb_run.dir if config.wandb_use else config.result_folder}/config.json", "w")
    )

    logger.info("Getting data and dataloaders")
    data, meta = get_dataset(**config.data, device=config.device, replace=True, frac=2000)

    # num_workers = max(min(os.cpu_count(), 8), 1)
    num_workers = os.cpu_count()
    logger.info(f"Using {num_workers} workers")
    train_loader = DataLoader(data["train"], shuffle=False, batch_size=config.train.batch_size,
                              num_workers=num_workers)
    valid_loader = DataLoader(data["valid"], shuffle=False, batch_size=config.test.batch_size,
                              num_workers=num_workers)
    test_loader = DataLoader(data["test"], shuffle=False, batch_size=config.test.batch_size,
                             num_workers=num_workers)
    
    logger.info("Getting model")
    
    # load arch module
    arch_module = importlib.import_module(config.model.arch.file.replace("/", ".")[:-3])
    model_arch = arch_module.get_arch(
        input_shape=meta.get("input_shape"), output_size=meta.get("num_class"), 
        **config.model.arch,
        slice_dim=config.data.frame_dim
    )

    # declaring models
    if config.model.name in "regression":
        from src.models.regression import Regression

        model = Regression(**model_arch)
    else:
        raise Exception("Unknown model")

    model.to(config.device)
    model.stats()

    if config.wandb_use and config.wandb_watch:
        wandb_watch(model, log="all")

    # declaring trainer
    optimizer, scheduler = optimizer_utils.get_optimizer_scheduler(
        model,
        lr=config.train.lr,
        optimizer=config.train.optimizer,
        opt_params={
                "weight_decay": config.train.get("weight_decay", 1e-4),
                "momentum"    : config.train.get("optimizer_momentum", 0.9)
        },
        scheduler=config.train.get("scheduler", None),
        scheduler_params={
                "gamma"         : config.train.get("scheduler_gamma", 0.1),
                "milestones"    : config.train.get("scheduler_milestones", [100, 200, 300]),
                "patience"      : config.train.get("scheduler_patience", 100),
                "step_size"     : config.train.get("scheduler_step_size", 100),
                "load_on_reduce": config.train.get("scheduler_load_on_reduce"),
                "mode"          : "max" if config.train.get(
                    "stopping_criteria_direction") == "bigger" else "min"
        },
    )
    trainer = Trainer(model, optimizer, scheduler=scheduler, statefile=config.statefile,
                      result_dir=config.result_folder, log_every=config.train.log_every,
                      save_strategy=config.train.save_strategy,
                      patience=config.train.patience,
                      max_epoch=config.train.max_epoch,
                      stopping_criteria=config.train.stopping_criteria,
                      gradient_norm_clip=config.train.gradient_norm_clip,
                      stopping_criteria_direction=config.train.stopping_criteria_direction,
                      evaluations=Box({"train": config.train.evaluations,
                                       "test" : config.test.evaluations}))

    if "train" in config.mode:
        logger.info("starting training")
        print(train_loader.dataset)
        trainer.train(train_loader, valid_loader)
        logger.info("Training done;")

        # copy current step and write test results to
        step_to_write = trainer.step
        step_to_write += 1

        if "test" in config.mode and config.test.eval_model == "best":
            if os.path.exists(f"{trainer.result_dir}/best_model.pt"):
                logger.info("Loading best model")
                trainer.load(f"{trainer.result_dir}/best_model.pt")
            else:
                logger.info("eval_model is best, but best model not found ::: evaling last model")
        else:
            logger.info("eval model is not best, so skipping loading at end of training")

    if "test" in config.mode:
        logger.info("evaluating model on test set")
        logger.info(f"Model was trained upto {trainer.epoch}")
        # copy current step and write test results to
        step_to_write = trainer.step
        step_to_write += 1

        print("<<<<<test>>>>>")
        loss, aux_loss = trainer.test(train_loader, test_loader)
        logging_utils.loss_logger_helper(loss, aux_loss, writer=trainer.summary_writer,
                                         force_print=True, step=step_to_write,
                                         epoch=trainer.epoch,
                                         log_every=trainer.log_every, string="test",
                                         new_line=True)

        print("<<<<<training>>>>>")
        loss, aux_loss = trainer.test(train_loader, train_loader)
        logging_utils.loss_logger_helper(loss, aux_loss, writer=trainer.summary_writer,
                                         force_print=True, step=step_to_write,
                                         epoch=trainer.epoch,
                                         log_every=trainer.log_every, string="train_eval",
                                         new_line=True)

        print("<<<<<validation>>>>>")
        loss, aux_loss = trainer.test(train_loader, valid_loader)
        logging_utils.loss_logger_helper(loss, aux_loss, writer=trainer.summary_writer,
                                         force_print=True, step=step_to_write,
                                         epoch=trainer.epoch,
                                         log_every=trainer.log_every, string="valid_eval",
                                         new_line=True)
