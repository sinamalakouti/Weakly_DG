# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import inspect

import torch
from typing import Dict, List, Optional, Tuple, Union
import torch.nn as nn
from detectron2.config import configurable
from detectron2.modeling import ROIHeads
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.modeling.proposal_generator.proposal_utils import (
    add_ground_truth_to_proposals,
)
from detectron2.utils.events import get_event_storage
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.layers import ShapeSpec
from detectron2.modeling.roi_heads import (
    ROI_HEADS_REGISTRY,
    StandardROIHeads,
)
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from adapteacher.modeling.roi_heads.fast_rcnn import FastRCNNFocaltLossOutputLayers, WeakFastRCNNOutputLayers

import numpy as np
from detectron2.modeling.poolers import ROIPooler
from torch.nn import functional as F


@torch.no_grad()
def get_image_level_gt(targets, num_classes):
    """
    Convert instance-level annotations to image-level
    """
    if targets is None:
        return None, None, None
    gt_classes_img = [torch.unique(t.gt_classes, sorted=True) for t in targets]
    gt_classes_img_int = [gt.to(torch.int64) for gt in gt_classes_img]
    gt_classes_img_oh = torch.cat(
        [
            torch.zeros(
                (1, num_classes), dtype=torch.float, device=gt_classes_img[0].device
            ).scatter_(1, torch.unsqueeze(gt, dim=0), 1)
            for gt in gt_classes_img_int
        ],
        dim=0,
    )
    return gt_classes_img, gt_classes_img_int, gt_classes_img_oh


@ROI_HEADS_REGISTRY.register()
class StandardROIHeadsPseudoLab(StandardROIHeads):

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        box_head = build_box_head(
            cfg,
            ShapeSpec(
                channels=in_channels, height=pooler_resolution, width=pooler_resolution
            ),
        )

        if cfg.MODEL.ROI_HEADS.LOSS == "CrossEntropy":
            box_predictor = FastRCNNOutputLayers(cfg, box_head.output_shape)
        elif cfg.MODEL.ROI_HEADS.LOSS == "FocalLoss":
            box_predictor = FastRCNNFocaltLossOutputLayers(cfg, box_head.output_shape)
        else:
            raise ValueError("Unknown ROI head loss.")

        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,

        }

    def build_region_head(self, inc, outc):
        self.weak_enable = True
        self.weak_head = torch.nn.Linear(inc, outc, bias=True)

    def forward(
            self,
            images: ImageList,
            features: Dict[str, torch.Tensor],
            proposals: List[Instances],
            targets: Optional[List[Instances]] = None,
            compute_loss=True,
            branch="",
            compute_val_loss=False,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:

        del images
        self.gt_classes_img, self.gt_classes_img_int, self.gt_classes_img_oh = get_image_level_gt(
            targets, self.num_classes
        )

        if self.training and compute_loss:  # apply if training loss
            assert targets
            # 1000 --> 512
            proposals = self.label_and_sample_proposals(
                proposals, targets, branch=branch
            )
        elif compute_val_loss:  # apply if val loss
            assert targets
            # 1000 --> 512
            temp_proposal_append_gt = self.proposal_append_gt
            self.proposal_append_gt = False
            proposals = self.label_and_sample_proposals(
                proposals, targets, branch=branch
            )  # do not apply target on proposals
            self.proposal_append_gt = temp_proposal_append_gt
        del targets

        if (self.training and compute_loss) or compute_val_loss:
            losses, _ = self._forward_box(
                features, proposals, compute_loss, compute_val_loss, branch
            )
            # if self.weak_head:
            #     losses_weak, _ = self._forward_box_weak(features, proposals, compute_loss, compute_val_loss, branch)
            #     losses.update(losses_weak)
            return proposals, losses
        else:
            pred_instances, predictions = self._forward_box(
                features, proposals, compute_loss, compute_val_loss, branch
            )

            return pred_instances, predictions

    def _forward_box(
            self,
            features: Dict[str, torch.Tensor],
            proposals: List[Instances],
            compute_loss: bool = True,
            compute_val_loss: bool = False,
            branch: str = "",
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        del box_features

        if (
                self.training and compute_loss
        ) or compute_val_loss:  # apply if training loss or val loss
            losses = self.box_predictor.losses(predictions, proposals)

            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(
                            proposals, pred_boxes
                    ):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses, predictions
        else:

            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances, predictions

    @torch.no_grad()
    def label_and_sample_proposals(
            self, proposals: List[Instances], targets: List[Instances], branch: str = ""
    ) -> List[Instances]:
        gt_boxes = [x.gt_boxes for x in targets]
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(
                            trg_name
                    ):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        storage = get_event_storage()
        storage.put_scalar(
            "roi_head/num_target_fg_samples_" + branch, np.mean(num_fg_samples)
        )
        storage.put_scalar(
            "roi_head/num_target_bg_samples_" + branch, np.mean(num_bg_samples)
        )

        return proposals_with_gt

    # #############

    def forward_weak(
            self,
            features: Dict[str, torch.Tensor],
            proposals: List[Instances],
            targets: Optional[List[Instances]] = None,
            branch: str = "",
            compute_loss: bool = True,
            compute_val_loss: bool = False,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        assert targets
        self.gt_classes_img, self.gt_classes_img_int, self.gt_classes_img_oh = get_image_level_gt(
            targets, self.num_classes
        )
        if self.training and compute_loss:  # apply if training loss
            assert targets
            proposals = self.label_and_sample_proposals(
                proposals, targets, branch=branch
            )
        elif compute_val_loss:  # apply if val loss
            assert targets
            # 1000 --> 512
            temp_proposal_append_gt = self.proposal_append_gt
            self.proposal_append_gt = False
            proposals = self.label_and_sample_proposals(
                proposals, targets, branch=branch
            )  # do not apply target on proposals
            self.proposal_append_gt = temp_proposal_append_gt
        del targets
        if (self.training and compute_loss) or compute_val_loss:

            losses, _ = self._forward_box_weak(features, proposals, compute_loss, compute_val_loss, branch)
            return proposals, losses
        else:
            pred_instances, predictions = self._forward_box_weak(
                features, proposals, compute_loss, compute_val_loss, branch
            )
            return pred_instances, predictions

    def _forward_box_weak(
            self, features: Dict[str, torch.Tensor],
            proposals: List[Instances],
            compute_loss: bool = True,
            compute_val_loss: bool = False,
            branch: str = "",
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        cls_predictions, reg_pred = predictions

        if (self.training and compute_loss) or compute_val_loss:
            num_proposal_per_image = [len(p) for p in proposals]

            # TODO: better implementation to calculate loss using rest inputs with non-valid proposals
            if 0 in num_proposal_per_image:
                return {"loss_mil": 0.0 * cls_predictions.sum()}

            # objectness_scores = torch.unsqueeze(torch.cat([p.objectness_logits for p in proposals]), dim=1)

            cls_scores = F.softmax(cls_predictions[:, :-1], dim=1)
            det_logits = self.weak_head(box_features)
            del box_features
            # max_cls_ids = torch.unsqueeze(torch.argmax(cls_predictions[:, :-1], dim=1), dim=1)
            # objectness_scores = torch.zeros_like(cls_scores).scatter_(
            #     dim=1, index=max_cls_ids, src=objectness_scores
            # )

            pred_img_cls_logits = torch.cat(
                [
                    torch.sum(cls * F.softmax(det, dim=0), dim=0, keepdim=True)
                    for cls, det in zip(
                    cls_scores.split(num_proposal_per_image, dim=0),
                    det_logits.split(num_proposal_per_image, dim=0))
                ],
                dim=0,
            )

            img_cls_losses = F.binary_cross_entropy(
                torch.clamp(pred_img_cls_logits, min=1e-6, max=1.0 - 1e-6),
                self.gt_classes_img_oh,
                reduction='mean'
            )
            return {"loss_mil": img_cls_losses}, pred_img_cls_logits
        else:
            num_proposal_per_image = [len(p) for p in proposals]
            cls_scores = F.softmax(cls_predictions[:, :-1], dim=1)
            det_logits = self.weah_head(box_features)
            # max_cls_ids = torch.unsqueeze(torch.argmax(cls_predictions[:, :-1], dim=1), dim=1)
            # objectness_scores = torch.zeros_like(cls_scores).scatter_(
            #     dim=1, index=max_cls_ids, src=objectness_scores
            # )

            pred_img_cls_logits = torch.cat(
                [
                    torch.sum(cls * F.softmax(det, dim=0), dim=0, keepdim=True)
                    for cls, det in zip(
                    cls_scores.split(num_proposal_per_image, dim=0),
                    det_logits.split(num_proposal_per_image, dim=0))
                ],
                dim=0,
            )

            return pred_img_cls_logits, (cls_scores, det_logits)


@ROI_HEADS_REGISTRY.register()
class myHead(StandardROIHeads):

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        box_head = build_box_head(
            cfg,
            ShapeSpec(
                channels=in_channels, height=pooler_resolution, width=pooler_resolution
            ),
        )

        box_predictor = WeakFastRCNNOutputLayers(cfg, box_head.output_shape)

        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,

        }




    def forward(
            self,
            images: ImageList,
            features: Dict[str, torch.Tensor],
            proposals: List[Instances],
            targets: Optional[List[Instances]] = None,
            compute_loss=True,
            branch="",
            compute_val_loss=False,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:

        del images
        self.gt_classes_img, self.gt_classes_img_int, self.gt_classes_img_oh = get_image_level_gt(
            targets, self.num_classes
        )
        # for param in self.box_predictor.parameters():
        #     param.requires_grad = True
        if self.training and compute_loss:  # apply if training loss
            assert targets
            # 1000 --> 512
            proposals = self.label_and_sample_proposals(
                proposals, targets, branch=branch
            )
        elif compute_val_loss:  # apply if val loss
            assert targets
            # 1000 --> 512
            temp_proposal_append_gt = self.proposal_append_gt
            self.proposal_append_gt = False
            proposals = self.label_and_sample_proposals(
                proposals, targets, branch=branch
            )  # do not apply target on proposals
            self.proposal_append_gt = temp_proposal_append_gt
        del targets

        if (self.training and compute_loss) or compute_val_loss:
            losses, _ = self._forward_box(
                features, proposals, compute_loss, compute_val_loss, branch
            )

            if "mil" in branch:
                losses['loss_box_reg'] = losses['loss_box_reg'] * 0.001
                losses['loss_cls'] = losses['loss_cls'] * 0.001
            elif branch != "supervised_all":
                losses['loss_mil'] = losses['loss_mil'] * 0.001
            return proposals, losses
        else:
            pred_instances, predictions = self._forward_box(
                features, proposals, compute_loss, compute_val_loss, branch
            )

            return pred_instances, predictions

    def forward_box_features(
            self,
            features: Dict[str, torch.Tensor],
            proposals: List[Instances],
            targets,
            branch
    ):
        proposals = self.label_and_sample_proposals(
            proposals, targets, branch=branch
        )
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        # copied  from DRN_WSOD
        objectness_logits = torch.cat([x.objectness_logits + 1 for x in proposals], dim=0)
        box_features = box_features * objectness_logits.view(-1, 1, 1, 1)
        box_features = self.box_head(box_features)

        return box_features, proposals

    def forward_box_predictor(
            self,
            box_features,
            proposals,
            targets,
            compute_loss,
            compute_val_loss,
            branch
    ):
        for param in self.box_predictor.parameters():
            param.requires_grad = False
            gt_classes_img, gt_classes_img_int, gt_classes_img_oh = get_image_level_gt(
                targets, self.num_classes
            )
        predictions = self.box_predictor(box_features, branch)
        del box_features

        if (
                self.training and compute_loss
        ) or compute_val_loss:  # apply if training loss or val loss
            losses = self.box_predictor.losses(predictions, proposals, gt_classes_img_oh)

            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(
                            proposals, pred_boxes
                    ):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses, predictions
        else:

            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances, predictions

    def _forward_box(
            self,
            features: Dict[str, torch.Tensor],
            proposals: List[Instances],
            compute_loss: bool = True,
            compute_val_loss: bool = False,
            branch: str = "",
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:

        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        # copied  from DRN_WSOD
        objectness_logits = torch.cat([x.objectness_logits + 1 for x in proposals], dim=0)
        box_features = box_features * objectness_logits.view(-1, 1, 1, 1)

        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features, branch)
        del box_features

        if (
                self.training and compute_loss
        ) or compute_val_loss:  # apply if training loss or val loss
            losses = self.box_predictor.losses(predictions, proposals, self.gt_classes_img_oh)

            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(
                            proposals, pred_boxes
                    ):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses, predictions
        else:

            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances, predictions

    @torch.no_grad()
    def label_and_sample_proposals(
            self, proposals: List[Instances], targets: List[Instances], branch: str = ""
    ) -> List[Instances]:
        gt_boxes = [x.gt_boxes for x in targets]
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(
                            trg_name
                    ):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        storage = get_event_storage()
        storage.put_scalar(
            "roi_head/num_target_fg_samples_" + branch, np.mean(num_fg_samples)
        )
        storage.put_scalar(
            "roi_head/num_target_bg_samples_" + branch, np.mean(num_bg_samples)
        )

        return proposals_with_gt
