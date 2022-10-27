# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import time
import logging
import torch
from torch.nn import Parameter
from torch.nn.parallel import DistributedDataParallel
from fvcore.nn.precise_bn import get_bn_modules
import numpy as np
from collections import OrderedDict

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultTrainer, SimpleTrainer, TrainerBase
from detectron2.engine.train_loop import AMPTrainer
from detectron2.utils.events import EventStorage
from detectron2.evaluation import verify_results, DatasetEvaluators
# from detectron2.evaluation import COCOEvaluator, verify_results, DatasetEvaluators

from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.engine import hooks
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances
from detectron2.utils.env import TORCH_VERSION
from detectron2.data import MetadataCatalog

from adapteacher.data.build2 import (
    build_detection_semisup_train_loader,
    build_detection_test_loader,
    build_detection_semisup_train_loader_two_crops,
)
from adapteacher.data.dataset_mapper import DatasetMapperTwoCropSeparate
from adapteacher.engine.hooks import LossEvalHook
from adapteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from adapteacher.checkpoint.detection_checkpoint import DetectionTSCheckpointer
from adapteacher.solver.build import build_lr_scheduler
from adapteacher.evaluation import PascalVOCDetectionEvaluator, COCOEvaluator

from .probe import OpenMatchTrainerProbe
import copy

from ..modeling.meta_arch.custom_head import Custom_head


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
        available states (eg. optimizer and scheduler) and update iteration counter
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
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        Use the custom checkpointer, which loads other backbone models
        with matching heuristics.
        """
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        data_loader = self.build_train_loader(cfg)

        # create an student model
        model = self.build_model(cfg)

        # create an teacher model
        s1_head = Custom_head(proposal_generator=None, roi_heads=None, cfg=cfg,
                              backbone_output_shape=model.backbone.output_shape(),
                              vis_period=0, weak_head=False).to(model.device)
        s2_head = Custom_head(proposal_generator=None, roi_heads=None, cfg=cfg,
                              backbone_output_shape=model.backbone.output_shape(),
                              vis_period=0, weak_head=True).to(model.device)

        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            if comm.get_world_size() != 4:
                print("comm.get_world_size()    ", comm.get_world_size())
            print(" comm.get_local_rank()   ", comm.get_local_rank())
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False, find_unused_parameters=True)
            s1_head = DistributedDataParallel(
                s1_head, device_ids=[comm.get_local_rank()], broadcast_buffers=False, find_unused_parameters=True)
            s2_head = DistributedDataParallel(
                s2_head, device_ids=[comm.get_local_rank()], broadcast_buffers=False, find_unused_parameters=True)
        self.s1_head = s1_head
        self.s2_head = s2_head
        ensemmbl_ts_model = EnsembleTSModel(model, self.s1_head, self.s2_head)
        optimizer = self.build_optimizer(cfg, ensemmbl_ts_model)
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
        available states (eg. optimizer and scheduler) and update iteration counter
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
                    if self.iter % 500 == 0:
                        self.checkpointer.save("model_{}.pt".format(self.iter))
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

    # def get_label_test(self, label_data):
    #     label_list = []
    #     for label_datum in label_data:
    #         if "instances" in label_datum.keys():
    #             label_list.append(label_datum["instances"])

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

            record_dict, _, _, _ = self.model(
                label_data_k, branch="supervised")

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
                None
                self._update_teacher_model(
                    keep_rate=self.cfg.SEMISUPNET.EMA_KEEP_RATE)

            record_dict = {}

            ######################## For probe #################################
            # import pdb; pdb. set_trace()
            gt_unlabel_k = self.get_label(unlabel_data_k)
            # gt_unlabel_q = self.get_label_test(unlabel_data_q)

            features_s2_weak, images, gt_instances = self.model(unlabel_data_k, branch='backbone')
            record_all_unlabel_data, _, _, _ = self.s2_head(
                features_s2_weak, images, gt_instances, branch="mil"
            )
            new_record_all_unlabel_data = {}
            for key in record_all_unlabel_data.keys():
                new_record_all_unlabel_data[key + "_mil"] = record_all_unlabel_data[
                    key
                ]
            record_dict.update(new_record_all_unlabel_data)

            #  0. convert to weakly labeled data
            unlabel_data_q = self.remove_label(unlabel_data_q)
            unlabel_data_k = self.remove_label(unlabel_data_k)


            #  1. generate the pseudo-label using teacher model
            #
            for param in self.model.module.proposal_generator.parameters():
                param.grad = None

            for param in self.model.module.roi_heads.parameters():
                param.grad = None

            with torch.no_grad():
                # (
                #     _,
                #     proposals_rpn_unsup_k,
                #     proposals_roih_unsup_k,
                #     _,
                # ) = self.model(unlabel_data_k, branch="unsup_data_weak")

                (
                    _,
                    proposals_rpn_unsup_k,
                    proposals_roih_unsup_k,
                    _
                ) = self.model.forward_head(features_s2_weak, images)
                ######################## For probe #################################
                # import pdb; pdb. set_trace()

                # probe_metrics = ['compute_fp_gtoutlier', 'compute_num_box']
                # probe_metrics = ['compute_num_box']
                # analysis_pred, _ = self.probe.compute_num_box(gt_unlabel_k,proposals_roih_unsup_k,'pred')
                # record_dict.update(analysis_pred)
                ######################## For probe END #################################

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
                # analysis_pred, _ = self.probe.compute_num_box(gt_unlabel_k,pesudo_proposals_rpn_unsup_k,'pred',True)
                # record_dict.update(analysis_pred)

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
            features_s1, images, gt_instances = self.model(all_label_data, branch='backbone')
            record_all_label_data, _, _, _ = self.s1_head(
                features_s1, images, gt_instances, branch="supervised"
            )
            record_dict.update(record_all_label_data)

            # features_s2_weak, images, gt_instances = self.model(unlabel_data_k, branch='backbone')
            # record_all_unlabel_data, _, _, _ = self.s2_head(
            #     features_s2_weak, images, gt_instances, branch="mil"
            # )
            # new_record_all_unlabel_data = {}
            # for key in record_all_unlabel_data.keys():
            #     new_record_all_unlabel_data[key + "_mil"] = record_all_unlabel_data[
            #         key
            #     ]
            # record_dict.update(new_record_all_unlabel_data)

            # 5. input strongly augmented unlabeled data into model

            features_s2, images, gt_instances = self.model(all_unlabel_data, branch='backbone')
            record_all_unlabel_data, _, _, _ = self.s2_head(
                features_s2, images, gt_instances, branch="supervised_target"
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
                    if key == "loss_rpn_loc_pseudo" or key == "loss_box_reg_pseudo":
                        # pseudo bbox regression <- 0
                        loss_dict[key] = record_dict[key] * 0
                    elif key[-6:] == "pseudo":  # unsupervised loss
                        loss_dict[key] = (
                                record_dict[key] *
                                self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT
                        )
                    elif "mil" in key:
                        loss_dict[key] = (
                                record_dict[key] *
                                self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT #TODO
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

        metrics_dict = record_dict
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
            # head_s1_dict = {
            #     key[7:]: value for key, value in self.s1_head.state_dict().items()
            # }
            head_s1_rpn_dict = self.s1_head.module.proposal_generator.state_dict()
            head_s1_roi_dict = self.s1_head.module.roi_heads.state_dict()

        else:
            # head_s1_dict = self.s1_head.state_dict()
            head_s1_rpn_dict = self.s1_head.proposal_generator.state_dict()
            head_s1_roi_dict = self.s1_head.roi_heads.state_dict()

        if comm.get_world_size() > 1:

            head_s2_rpn_dict = self.s2_head.module.proposal_generator.state_dict()
            head_s2_roi_dict = self.s2_head.module.roi_heads.state_dict()
        else:
            head_s2_rpn_dict = self.s2_head.proposal_generator.state_dict()
            head_s2_roi_dict = self.s2_head.roi_heads.state_dict()

        new_teacher_dict = OrderedDict()
        for key, value in self.model.module.proposal_generator.state_dict().items():
            if key in head_s1_rpn_dict.keys() and key in head_s2_rpn_dict.keys():
                new_teacher_dict[key] = (
                        (head_s1_rpn_dict[key] * 0.5 + head_s2_rpn_dict[key] * 0.5) *
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
            if key in head_s1_roi_dict.keys() and key in head_s2_roi_dict.keys():
                if head_s1_roi_dict[key].shape == head_s2_roi_dict[key].shape:
                    new_teacher_dict[key] = (
                            (head_s1_roi_dict[key] * 0.5 + head_s2_roi_dict[key] * 0.5) *
                            (1 - keep_rate) + value * keep_rate
                    )
                else:
                    new_teacher_dict[key] = (
                            (head_s1_roi_dict[key] * 1) *
                            (1 - keep_rate) + value * keep_rate
                    )
            else:
                raise Exception("{} is not found in student model".format(key))

        self.model.module.roi_heads.load_state_dict(new_teacher_dict)

    @torch.no_grad()
    def _init_student_heads(self):
        if comm.get_world_size() > 1:

            self.s1_head.module.proposal_generator.load_state_dict(self.model.module.proposal_generator.state_dict())
            self.s2_head.module.proposal_generator.load_state_dict(self.model.module.proposal_generator.state_dict())

            self.s1_head.module.roi_heads.load_state_dict(self.model.module.roi_heads.state_dict())
            # self.s2_head.module.roi_heads.load_state_dict(self.model.module.roi_heads.state_dict())
            for k, param in self.model.module.roi_heads.state_dict().items():
                if self.model.module.roi_heads.state_dict()[k].shape == self.s2_head.module.roi_heads.state_dict()[k].shape:
                    param = param.data
                    self.s2_head.module.roi_heads.state_dict()[k].copy_(param)

        else:
            self.s1_head.proposal_generator.load_state_dict(self.model.proposal_generator.state_dict())
            self.s2_head.proposal_generator.load_state_dict(self.model.proposal_generator.state_dict())

            self.s1_head.roi_heads.load_state_dict(self.model.roi_heads.state_dict())
            for k, param in self.model.roi_heads.state_dict().items():
                if self.model.roi_heads.state_dict()[k].shape == self.s2_head.roi_heads.state_dict()[k].shape:
                    param = param.data
                    self.s2_head.roi_heads.state_dict()[k].copy_(param)

    @torch.no_grad()
    def _copy_main_model(self):
        print("Copyyyy " * 10)
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
            return self._last_eval_results_teacher

        # ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD,
        #            test_and_save_results_student))
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD,
                                  test_and_save_results_teacher))

        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret
