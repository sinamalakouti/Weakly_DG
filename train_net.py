#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import detectron2.utils.comm as comm
import torch.cuda
from detectron2.checkpoint import DetectionCheckpointer
# from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch

from adapteacher import add_ateacher_config
# from adapteacher.engine.trainer2 import ATeacherTrainer, BaselineTrainer


# hacky way to register

from adapteacher.modeling.meta_arch.vgg import build_vgg_backbone  # noqa

import adapteacher.data.datasets.builtin
#### MY IMPORTS

from adapteacher.modeling.meta_arch.rcnn import \
    DGobjGeneralizedRCNN  # TwoStagePseudoLabGeneralizedRCNN, DAobjTwoStagePseudoLabGeneralizedRCNN
from adapteacher.modeling.roi_heads.roi_heads import StandardROIHeadsPseudoLab, myHead
from adapteacher.modeling.proposal_generator.rpn import PseudoLabRPN
# from adapteacher.engine.trainer_weak import ATeacherTrainer, BaselineTrainer
from adapteacher.engine.my_trainer import ATeacherTrainer, BaselineTrainer
#### ORIGNAL PAPER

#
# from adapteacher.engine.original_trainer import ATeacherTrainer, BaselineTrainer
# from adapteacher.modeling.meta_arch.orginal_ts_ensemble import EnsembleTSModel
# from adapteacher.modeling.meta_arch.original_rcnn import TwoStagePseudoLabGeneralizedRCNN, DAobjTwoStagePseudoLabGeneralizedRCNN

from adapteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel
import wandb


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_ateacher_config(cfg)
    if not torch.cuda.is_available():
        args.config_file = 'configs/my_city_config.yaml'
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    if not torch.cuda.is_available():
        cfg['config'] = 'configs/my_city_config.yaml'
        # cfg['config'] = 'configs/faster_rcnn_VGG_cross_city.yaml'
    cfg['num-gpus'] = 4
    if not torch.cuda.is_available():
        cfg['OUTPUT_DIR'] = 'output/exp_city'
    return cfg


def main(args, wandb_run=None):

    cfg = setup(args)
    if wandb_run:
        wandb_run.config.update(cfg)

    if cfg.SEMISUPNET.Trainer == "ateacher":
        Trainer = ATeacherTrainer
    elif cfg.SEMISUPNET.Trainer == "baseline":
        Trainer = BaselineTrainer
    else:
        raise ValueError("Trainer Name is not found.")
    # args.eval_only = True
    if args.eval_only:
        if cfg.SEMISUPNET.Trainer == "ateacher":
            model = Trainer.build_model(cfg)
            model_teacher = Trainer.build_model(cfg)
            ensem_ts_model = EnsembleTSModel(model_teacher, model)

            DetectionCheckpointer(
                ensem_ts_model, save_dir=cfg.OUTPUT_DIR
            ).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
            res = Trainer.test(cfg, model)

        else:
            model = Trainer.build_model(cfg)
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
            )
            res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    wandb_run = None
    print("HERE * 100")
    print(args.machine_rank)
    # wandb_run = wandb.init(project="WSOD_DG_v3_MIL")
    wandb_run = None
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args, wandb_run),
    )
