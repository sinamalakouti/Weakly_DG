# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import logging
import os
import time
from collections import OrderedDict

import detectron2.utils.comm as comm
import numpy as np
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, SimpleTrainer, TrainerBase
from detectron2.engine import hooks
from detectron2.engine.train_loop import AMPTrainer
from detectron2.evaluation import verify_results, DatasetEvaluators
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances
from detectron2.utils.env import TORCH_VERSION
from detectron2.utils.events import EventStorage
from fvcore.nn.precise_bn import get_bn_modules
from torch.nn.parallel import DistributedDataParallel

from adapteacher.checkpoint.detection_checkpoint import DetectionTSCheckpointer
from adapteacher.data.build2 import (
    build_detection_semisup_train_loader,
    build_detection_test_loader,
    build_detection_semisup_train_loader_two_crops,
)
from adapteacher.data.dataset_mapper import DatasetMapperTwoCropSeparate
from adapteacher.evaluation import PascalVOCDetectionEvaluator, COCOEvaluator
from adapteacher.modeling.meta_arch.DG_model import DG_model
from adapteacher.solver.build import build_lr_scheduler
from .probe import OpenMatchTrainerProbe
from ..modeling.meta_arch.DG_head import DG_head


# from detectron2.evaluation import COCOEvaluator, verify_results, DatasetEvaluators


# Supervised-only Trainer

class BaselineTrainer(DefaultTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        Use the custom checkpointer, which loads other backbone models
        with matching heuristics.
        """
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )

        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (e.g. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.
        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.
        Args:
            resume (bool): whether to do resume or not
        """
        checkpoint = self.checkpointer.resume_or_load(
            self.cfg.MODEL.WEIGHTS, resume=resume
        )
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration (or iter zero if there's no checkpoint).
        if isinstance(self.model, DistributedDataParallel):
            # broadcast loaded data/model from the first rank, because other
            # machines may not have access to the checkpoint file
            if TORCH_VERSION >= (1, 7):
                self.model._sync_params_and_buffers()
            self.start_iter = comm.all_gather(self.start_iter)[0]

    def train_loop(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step()
                    self.after_step()
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    def run_step(self):
        self._trainer.iter = self.iter

        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()

        data = next(self._trainer._data_loader_iter)
        data_time = time.perf_counter() - start

        record_dict, _, _, _ = self.model(data, branch="supervised")

        num_gt_bbox = 0.0
        for element in data:
            num_gt_bbox += len(element["instances"])
        num_gt_bbox = num_gt_bbox / len(data)
        record_dict["bbox_num/gt_bboxes"] = num_gt_bbox

        loss_dict = {}
        for key in record_dict.keys():
            if key[:4] == "loss" and key[-3:] != "val":
                loss_dict[key] = record_dict[key]

        losses = sum(loss_dict.values())

        metrics_dict = record_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(
                dataset_name, output_dir=output_folder))
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc_water":
            return PascalVOCDetectionEvaluator(dataset_name,
                                               target_classnames=["bicycle", "bird", "car", "cat", "dog", "person"])
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]

        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_semisup_train_loader(cfg, mapper=None)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable
        """
        return build_detection_test_loader(cfg, dataset_name)

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                cfg.TEST.EVAL_PERIOD,
                self.model,
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

    def _write_metrics(self, metrics_dict: dict):
        """
        Args:
            metrics_dict (dict): dict of scalar metrics
        """
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                data_time = np.max([x.pop("data_time")
                                    for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }

            loss_dict = {}
            for key in metrics_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = metrics_dict[key]

            total_losses_reduced = sum(loss for loss in loss_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)


# Adaptive Teacher Trainer
class ATeacherTrainer(DefaultTrainer):
    def __init__(self, cfg, wandb_run=None):
        """
        Args:
            cfg (CfgNode):
        Use the custom checkpointer, which loads other backbone models
        with matching heuristics.
        """
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        data_loader = self.build_train_loader(cfg)

        # create a student model
        model = self.build_model(cfg)

        # create a teacher model
        s_f = DG_head(proposal_generator=None, roi_heads=None, cfg=cfg,
                      backbone_output_shape=model.backbone.output_shape(),
                      vis_period=0, weak_head=True).to(model.device)
        s_w = DG_head(proposal_generator=None, roi_heads=None, cfg=cfg,
                      backbone_output_shape=model.backbone.output_shape(),
                      vis_period=0, weak_head=True).to(model.device)
        self.wandb_run = wandb_run
        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            if comm.get_world_size() != 4:
                print("comm.get_world_size()    ", comm.get_world_size())
            print(" comm.get_local_rank()   ", comm.get_local_rank())
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False, find_unused_parameters=True)
            s_f = DistributedDataParallel(
                s_f, device_ids=[comm.get_local_rank()], broadcast_buffers=False, find_unused_parameters=True)
            s_w = DistributedDataParallel(
                s_w, device_ids=[comm.get_local_rank()], broadcast_buffers=False, find_unused_parameters=True)
        self.s_f = s_f
        self.s_w = s_w
        ensembl_ts_model = DG_model(model, self.s_f, self.s_w)
        # optimizer = self.build_optimizer(cfg, model)
        optimizer = self.build_optimizer(cfg, ensembl_ts_model)
        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )
        self.scheduler = self.build_lr_scheduler(cfg, optimizer)

        self.checkpointer = DetectionTSCheckpointer(
            model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.probe = OpenMatchTrainerProbe(cfg)
        self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (e.g. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.
        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.
        Args:
            resume (bool): whether to do resume or not
        """
        checkpoint = self.checkpointer.resume_or_load(
            self.cfg.MODEL.WEIGHTS, resume=resume
        )
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration (or iter zero if there's no checkpoint).
        if isinstance(self.model, DistributedDataParallel):
            # broadcast loaded data/model from the first rank, because other
            # machines may not have access to the checkpoint file
            if TORCH_VERSION >= (1, 7):
                self.model._sync_params_and_buffers()
            self.start_iter = comm.all_gather(self.start_iter)[0]

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(
                dataset_name, output_dir=output_folder))
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc_water":
            return PascalVOCDetectionEvaluator(dataset_name,
                                               target_classnames=["bicycle", "bird", "car", "cat", "dog", "person"])
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]

        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapperTwoCropSeparate(cfg, True)
        return build_detection_semisup_train_loader_two_crops(cfg, mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)

    def train(self):
        self.train_loop(self.start_iter, self.max_iter)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    def train_loop(self, start_iter: int, max_iter: int):
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()

                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step_full_semisup()
                    self.after_step()

                    # if self.iter % 500 == 0:
                    #     self.checkpointer.save("model_{}.pt".format(self.iter))
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    # =====================================================
    # ================== Pseduo-labeling ==================
    # =====================================================
    def threshold_bbox(self, proposal_bbox_inst, thres=0.7, proposal_type="roih"):
        if proposal_type == "rpn":
            valid_map = proposal_bbox_inst.objectness_logits > thres

            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            # create box
            new_bbox_loc = proposal_bbox_inst.proposal_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.objectness_logits = proposal_bbox_inst.objectness_logits[
                valid_map
            ]
        elif proposal_type == "roih":
            valid_map = proposal_bbox_inst.scores > thres

            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            # create box
            new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map]
            new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]

        return new_proposal_inst

    def process_pseudo_label(
            self, proposals_rpn_unsup_k, cur_threshold, proposal_type, psedo_label_method=""
    ):
        list_instances = []
        num_proposal_output = 0.0
        for proposal_bbox_inst in proposals_rpn_unsup_k:
            # thresholding
            if psedo_label_method == "thresholding":
                proposal_bbox_inst = self.threshold_bbox(
                    proposal_bbox_inst, thres=cur_threshold, proposal_type=proposal_type
                )
            else:
                raise ValueError("Unkown pseudo label boxes methods")
            num_proposal_output += len(proposal_bbox_inst)
            list_instances.append(proposal_bbox_inst)
        num_proposal_output = num_proposal_output / len(proposals_rpn_unsup_k)
        return list_instances, num_proposal_output

    def remove_label(self, label_data):
        import copy
        label_data = copy.deepcopy(label_data)
        for label_datum in label_data:
            if "instances" in label_datum.keys():
                del label_datum["instances"]
        return label_data

    def convert_to_weak_label(self, label_data):
        for label_datum in label_data:
            if "instances" in label_datum.keys():
                label_datum['instances'].gt_boxes.tensor = torch.zeros_like(label_datum['instances'].gt_boxes.tensor)
            else:
                raise NotImplementedError(
                    "NO instances for this example? Is that even possible")
        return label_data

    def add_label(self, unlabled_data, label):
        for unlabel_datum, lab_inst in zip(unlabled_data, label):
            unlabel_datum["instances"] = lab_inst
        return unlabled_data

    def get_label(self, label_data):
        label_list = []
        for label_datum in label_data:
            if "instances" in label_datum.keys():
                label_list.append(copy.deepcopy(label_datum["instances"]))

        return label_list

    # =====================================================
    # =================== Training Flow ===================
    # =====================================================

    def run_step_full_semisup(self):
        self._trainer.iter = self.iter
        assert self.model.training, "[UBTeacherTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        data = next(self._trainer._data_loader_iter)
        # data_q and data_k from different augmentations (q:strong, k:weak)
        # label_strong, label_weak, unlabed_strong, unlabled_weak
        label_data_q, label_data_k, unlabel_data_q, unlabel_data_k = data
        data_time = time.perf_counter() - start

        # burn-in stage (supervised training with labeled data)
        if self.iter < self.cfg.SEMISUPNET.BURN_UP_STEP:
            label_data_q.extend(label_data_k)
            record_dict, _, _, _ = self.model(
                label_data_q, branch="supervised")

            # weight losses
            loss_dict = {}
            for key in record_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = record_dict[key] * 1
            losses = sum(loss_dict.values())

        else:
            if self.iter == self.cfg.SEMISUPNET.BURN_UP_STEP:
                # update copy the  whole model
                # self._update_teacher_model(keep_rate=0.00) #TODO: we need to update how ot update the teacher_model ( only heads )
                self._init_student_heads()
            elif (
                    self.iter - self.cfg.SEMISUPNET.BURN_UP_STEP
            ) % self.cfg.SEMISUPNET.TEACHER_UPDATE_ITER == 0:

                self._update_teacher_model(
                    keep_rate=self.cfg.SEMISUPNET.EMA_KEEP_RATE)

            record_dict = {}
            ######################## For probe #################################

            features_w_weak, images_ws, gt_instances_ws = self.model(unlabel_data_k, branch='backbone')
            record_all_unlabel_data, _, _, _ = self.s_w(
                features_w_weak, images_ws, gt_instances_ws, branch="mil"
            )
            new_record_all_unlabel_data = {}
            for key in record_all_unlabel_data.keys():
                new_record_all_unlabel_data[key + '_weak'] = record_all_unlabel_data[
                    key
                ]
            record_dict.update(new_record_all_unlabel_data)
            #  0. convert to weakly labeled data
            unlabel_data_q = self.remove_label(unlabel_data_q)
            unlabel_data_k = self.remove_label(unlabel_data_k)

            #  1. generate the pseudo-label using teacher model
            #
            if comm.get_world_size() > 1:
                for param in self.model.module.proposal_generator.parameters():
                    param.requires_grad = False
                    param.grad = None

                for param in self.model.module.roi_heads.parameters():
                    param.requires_grad = False
                    param.grad = None
            else:
                for param in self.model.proposal_generator.parameters():
                    param.requires_grad = False
                    param.grad = None

                for param in self.model.roi_heads.parameters():
                    param.requires_grad = False
                    param.grad = None

            with torch.no_grad():
                if comm.get_world_size() > 1:
                    (
                        _,
                        proposals_rpn_unsup_k,
                        proposals_roih_unsup_k,
                        _
                    ) = self.model.module.forward_head(features_w_weak, images_ws)
                else:
                    (
                        _,
                        proposals_rpn_unsup_k,
                        proposals_roih_unsup_k,
                        _
                    ) = self.model.forward_head(features_w_weak, images_ws)

                #  2. Pseudo-labeling
                cur_threshold = self.cfg.SEMISUPNET.BBOX_THRESHOLD

                joint_proposal_dict = {}
                joint_proposal_dict["proposals_rpn"] = proposals_rpn_unsup_k
                # Process pseudo labels and thresholding
                (
                    pesudo_proposals_rpn_unsup_k,
                    nun_pseudo_bbox_rpn,
                ) = self.process_pseudo_label(
                    proposals_rpn_unsup_k, cur_threshold, "rpn", "thresholding"
                )
                joint_proposal_dict["proposals_pseudo_rpn"] = pesudo_proposals_rpn_unsup_k
                # Pseudo_labeling for ROI head (bbox location/objectness)
                pesudo_proposals_roih_unsup_k, _ = self.process_pseudo_label(
                    proposals_roih_unsup_k, cur_threshold, "roih", "thresholding"
                )
                joint_proposal_dict["proposals_pseudo_roih"] = pesudo_proposals_roih_unsup_k

                # 3. add pseudo-label to unlabeled data

                unlabel_data_q = self.add_label(
                    unlabel_data_q, joint_proposal_dict["proposals_pseudo_roih"]
                )
                # unlabel_data_k = self.add_label(
                #     unlabel_data_k, joint_proposal_dict["proposals_pseudo_roih"]
                # )

                all_label_data = label_data_q + label_data_k
                all_unlabel_data = unlabel_data_q

            # 4. input both strongly and weakly augmented labeled data into student model
            features_sf, images_sf, gt_instances_sf = self.model(all_label_data, branch='backbone')
            record_all_label_data, _, _, _ = self.s_f(
                features_sf, images_sf, gt_instances_sf, branch="supervised_all"
            )
            record_dict.update(record_all_label_data)

            # 5. input strongly augmented unlabeled data into model

            features_sw, images_sw, gt_instances_sw = self.model(all_unlabel_data, branch='backbone')
            record_all_unlabel_data, _, _, _ = self.s_w(
                features_sw, images_sw, gt_instances_sw, branch="supervised_target"
            )
            new_record_all_unlabel_data = {}
            for key in record_all_unlabel_data.keys():
                new_record_all_unlabel_data[key + "_pseudo"] = record_all_unlabel_data[
                    key
                ]
            record_dict.update(new_record_all_unlabel_data)

            # 6. input weakly labeled data (source) and weakly unlabeled data (target) to student model
            # give sign to the target data

            for i_index in range(len(unlabel_data_k)):
                # unlabel_data_item = {}
                for k, v in unlabel_data_k[i_index].items():
                    # label_data_k[i_index][k + "_unlabeled"] = v
                    label_data_k[i_index][k + "_unlabeled"] = v
                # unlabel_data_k[i_index] = unlabel_data_item

            all_domain_data = label_data_k
            # all_domain_data = label_data_k + unlabel_data_k
            record_all_domain_data, _, _, _ = self.model(all_domain_data, branch="domain")
            record_dict.update(record_all_domain_data)

            # weight losses
            loss_dict = {}
            for key in record_dict.keys():
                if key.startswith("loss"):
                    if key == "loss_rpn_loc_pseudo" or key == "loss_box_reg_pseudo" or key == ' loss_mil_pseudo':
                        # pseudo bbox regression <- 0
                        loss_dict[key] = record_dict[key] * 0
                    elif key[-6:] == "pseudo":  # unsupervised loss
                        loss_dict[key] = (
                                record_dict[key] *
                                self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT
                        )
                    elif "mil" in key:
                        loss_dict[key] = (
                            record_dict[key]  # *
                            # self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT  # TODO
                        )
                    elif (
                            key == "loss_D_img_s" or key == "loss_D_img_t"
                    ):  # set weight for discriminator
                        # import pdb
                        # pdb.set_trace()
                        loss_dict[key] = record_dict[
                                             key] * 0.01  # self.cfg.SEMISUPNET.DIS_LOSS_WEIGHT  # Need to modify defaults and yaml
                    else:  # supervised loss
                        loss_dict[key] = record_dict[key] * 1

            losses = sum(loss_dict.values())
        with torch.no_grad():
            metrics_dict = record_dict
            wandb_logs_dict = metrics_dict.copy()
            wandb_logs_dict['losses'] = losses
            wandb_logs_dict['iter'] = self.iter
            if self.wandb_run:
                self.wandb_run.log(wandb_logs_dict)
            metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

        if self.iter > self.cfg.SEMISUPNET.BURN_UP_STEP + 2000 or True:
            label_data_q, label_data_k, unlabel_data_q, unlabel_data_k = data
            with torch.no_grad():
                features_w_k, images_w_k, gt_instances_w_k = self.model(unlabel_data_k, branch='backbone')
                features_s_k, images_s_k, gt_instances_s_k = self.model(label_data_k, branch='backbone')

            box_features_w, proposals_w = self.s_w(features_w_k, images_w_k, gt_instances_w_k, branch='head_features')
            box_features_f, proposals_f = self.s_f(features_s_k, images_s_k, gt_instances_s_k, branch='head_features')

            if comm.get_world_size() > 1:
                record_unlabeled_episodic, _ = self.s_f.module.roi_heads.forward_box_predictor(
                    box_features_w,
                    proposals_w,
                    gt_instances_w_k,
                    compute_loss=True,
                    compute_val_loss=False,
                    branch="mil"
                )

            record_unlabeled_episodic['loss_box_reg'] = record_unlabeled_episodic['loss_box_reg'] * 0
            record_unlabeled_episodic['loss_cls'] = record_unlabeled_episodic['loss_cls'] * 0

            loss_dict = {}
            for key in record_unlabeled_episodic.keys():
                loss_dict[key + '_episodic_weak'] = record_unlabeled_episodic[
                    key
                ]

            if comm.get_world_size() > 1:
                record_labeled_episodic, _ = self.s_w.module.roi_heads.forward_box_predictor(
                    box_features_f,
                    proposals_f,
                    gt_instances_s_k,
                    compute_loss=True,
                    compute_val_loss=False,
                    branch="mil"
                )

            record_labeled_episodic['loss_box_reg'] = record_labeled_episodic['loss_box_reg'] * 0
            record_labeled_episodic['loss_cls'] = record_labeled_episodic['loss_cls'] * 0

            for key in record_labeled_episodic.keys():
                loss_dict[key + '_episodic_labeled'] = record_labeled_episodic[
                    key
                ]

            losses = sum(loss_dict.values())
            with torch.no_grad():
                metrics_dict = loss_dict
                wandb_logs_dict = metrics_dict.copy()
                wandb_logs_dict['losses'] = losses
                wandb_logs_dict['iter'] = self.iter
                if self.wandb_run:
                    self.wandb_run.log(wandb_logs_dict)
                metrics_dict["data_time"] = data_time
            self._write_metrics(metrics_dict)

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

    # print("here")

    def _write_metrics(self, metrics_dict: dict):
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }

        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)
        # all_hg_dict = comm.gather(hg_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                # data_time among workers can have high variance. The actual latency
                # caused by data_time is the maximum among workers.
                data_time = np.max([x.pop("data_time")
                                    for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }

            # append the list
            loss_dict = {}
            for key in metrics_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = metrics_dict[key]

            total_losses_reduced = sum(loss for loss in loss_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)

    @torch.no_grad()
    def _update_teacher_model(self, keep_rate=0.9996):

        if comm.get_world_size() > 1:
            s_f_rpn_dict = self.s_f.module.proposal_generator.state_dict()
            s_f_roi_dict = self.s_f.module.roi_heads.state_dict()

        else:
            s_f_rpn_dict = self.s_f.proposal_generator.state_dict()
            s_f_roi_dict = self.s_f.roi_heads.state_dict()

        if comm.get_world_size() > 1:

            s_w_rpn_dict = self.s_w.module.proposal_generator.state_dict()
            s_w_roi_dict = self.s_w.module.roi_heads.state_dict()
        else:
            s_w_rpn_dict = self.s_w.proposal_generator.state_dict()
            s_w_roi_dict = self.s_w.roi_heads.state_dict()

        new_teacher_dict = OrderedDict()
        for key, value in self.model.module.proposal_generator.state_dict().items():
            if key in s_f_rpn_dict.keys() and key in s_w_rpn_dict.keys():
                w_f = 0.5
                w_w = 0.5
                if 'bbox_pred' in key:
                    w_w = 0
                    w_f = 1

                new_teacher_dict[key] = (
                        (s_f_rpn_dict[key] * w_f + s_w_rpn_dict[key] * w_w) *
                        (1 - keep_rate) + value * keep_rate
                )
            else:
                raise Exception("{} is not found in student model".format(key))
        # if comm.get_world_size() > 1:
        #     self.model.proposal_generator.load_state_dict(new_teacher_dict)
        # else:
        self.model.module.proposal_generator.load_state_dict(new_teacher_dict)

        new_teacher_dict = OrderedDict()
        for key, value in self.model.module.roi_heads.state_dict().items():
            if key in s_f_roi_dict.keys() and key in s_w_roi_dict.keys():
                if s_f_roi_dict[key].shape == s_w_roi_dict[key].shape:
                    w_f = 0.5
                    w_w = 0.5
                    if 'bbox_pred' in key:
                        w_w = 0
                        w_f = 1

                    new_teacher_dict[key] = (
                            (s_f_roi_dict[key] * w_f + s_w_roi_dict[key] * w_w) *
                            (1 - keep_rate) + value * keep_rate
                    )
                else:
                    new_teacher_dict[key] = (
                            (s_f_roi_dict[key] * 1) *
                            (1 - keep_rate) + value * keep_rate
                    )
            else:
                raise Exception("{} is not found in student model".format(key))

        self.model.module.roi_heads.load_state_dict(new_teacher_dict)

    @torch.no_grad()
    def _init_student_heads(self):
        if comm.get_world_size() > 1:

            self.s_f.module.proposal_generator.load_state_dict(self.model.module.proposal_generator.state_dict())
            self.s_w.module.proposal_generator.load_state_dict(self.model.module.proposal_generator.state_dict())

            for k, param in self.model.module.roi_heads.state_dict().items():
                if self.model.module.roi_heads.state_dict()[k].shape == self.s_f.module.roi_heads.state_dict()[
                    k].shape:
                    param = param.data
                    self.s_f.module.roi_heads.state_dict()[k].copy_(param)

            for k, param in self.model.module.roi_heads.state_dict().items():
                if self.model.module.roi_heads.state_dict()[k].shape == self.s_w.module.roi_heads.state_dict()[
                    k].shape:
                    param = param.data
                    self.s_w.module.roi_heads.state_dict()[k].copy_(param)

        else:
            self.s_f.proposal_generator.load_state_dict(self.model.proposal_generator.state_dict())
            self.s_w.proposal_generator.load_state_dict(self.model.proposal_generator.state_dict())

            self.s_f.roi_heads.load_state_dict(self.model.roi_heads.state_dict())
            for k, param in self.model.roi_heads.state_dict().items():
                if self.model.roi_heads.state_dict()[k].shape == self.s_f.roi_heads.state_dict()[k].shape:
                    param = param.data
                    self.s_f.roi_heads.state_dict()[k].copy_(param)

            for k, param in self.model.roi_heads.state_dict().items():
                if self.model.roi_heads.state_dict()[k].shape == self.s_w.roi_heads.state_dict()[k].shape:
                    param = param.data
                    self.s_w.roi_heads.state_dict()[k].copy_(param)

    @torch.no_grad()
    def _copy_main_model(self):

        # initialize all parameters
        if comm.get_world_size() > 1:
            rename_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
            self.model_student.load_state_dict(rename_model_dict)
        else:
            self.model_student.load_state_dict(self.model.state_dict())

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name)

    def build_hooks(self):
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )

        # def test_and_save_results_student():
        #     self._last_eval_results_student = self.test(self.cfg, self.model)
        #     _last_eval_results_student = {
        #         k + "_student": self._last_eval_results_student[k]
        #         for k in self._last_eval_results_student.keys()
        #     }
        #     return _last_eval_results_student

        def test_and_save_results_teacher():
            self._last_eval_results_teacher = self.test(
                self.cfg, self.model)
            if len(self._last_eval_results_teacher):
                # todo wandb log the inference
                res_dict = self._last_eval_results_teacher['bbox']
                res_dict['iter'] = self.iter
                if self.wandb_run:
                    self.wandb_run.log(res_dict)

            return self._last_eval_results_teacher

        # ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD,
        #            test_and_save_results_student))
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD,
                                  test_and_save_results_teacher))

        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret
