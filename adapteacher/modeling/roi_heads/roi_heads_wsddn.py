# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import inspect
import logging
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.structures import Boxes, ImageList, Instances

from adapteacher.modeling.roi_heads.fast_rcnn import WSDDNOutputLayers
from detectron2.modeling.roi_heads.roi_heads import (
    ROIHeads,
    select_foreground_proposals,
    select_proposals_with_visible_keypoints,
)

logger = logging.getLogger(__name__)


def get_image_level_gt(targets, num_classes):
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
class WSDDNROIHeads(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    Each head independently processes the input features by each head's
    own pooler and head.
    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    @configurable
    def __init__(
            self,
            *,
            box_in_features: List[str],
            box_pooler: ROIPooler,
            box_head: nn.Module,
            box_predictor: nn.Module,
            train_on_pred_boxes: bool = False,
            **kwargs
    ):
        """
        NOTE: this interface is experimental.
        Args:
            box_in_features (list[str]): list of feature names to use for the box head.
            box_pooler (ROIPooler): pooler to extra region features for box head
            box_head (nn.Module): transform features to make box predictions
            box_predictor (nn.Module): make box predictions from the feature.
                Should have the same interface as :class:`FastRCNNOutputLayers`.
            mask_in_features (list[str]): list of feature names to use for the mask head.
                None if not using mask head.
            mask_pooler (ROIPooler): pooler to extra region features for mask head
            mask_head (nn.Module): transform features to make mask predictions
            keypoint_in_features, keypoint_pooler, keypoint_head: similar to ``mask*``.
            train_on_pred_boxes (bool): whether to use proposal boxes or
                predicted boxes from the box head to train other heads.
        """
        super().__init__(**kwargs)
        # keep self.in_features for backward compatibility
        self.in_features = self.box_in_features = box_in_features
        self.box_pooler = box_pooler
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.train_on_pred_boxes = train_on_pred_boxes

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg)
        ret["train_on_pred_boxes"] = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
        # Subclasses that have not been updated to use from_config style construction
        # may have overridden _init_*_head methods. In this case, those overridden methods
        # will not be classmethods and we need to avoid trying to call them here.
        # We test for this with ismethod which only returns True for bound methods of cls.
        # Such subclasses will need to handle calling their overridden _init_*_head methods.
        if inspect.ismethod(cls._init_box_head):
            ret.update(cls._init_box_head(cfg, input_shape))
        return ret

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
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
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        box_predictor = WSDDNOutputLayers(cfg, box_head.output_shape)
        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
        }

    def forward_weak(
            self,
            images: ImageList,
            features: Dict[str, torch.Tensor],
            proposals: List[Instances],
            targets: Optional[List[Instances]] = None,
            compute_loss=True,
            branch="",
            compute_val_loss=False,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        self.gt_classes_img, self.gt_classes_img_int, self.gt_classes_img_oh = get_image_level_gt(
            targets, self.num_classes
        )
        #         if self.training and compute_loss:  # apply if training loss
        #             assert targets
        #             # 1000 --> 512
        #             proposals = self.label_and_sample_proposals(
        #                 proposals, targets, branch=branch
        #             )

        del images
        if self.training and compute_loss:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
            # sampled_proposals = []
            # for proposals_per_image in proposals:
            #     sampled_idxs = torch.randperm(
            #         len(proposals_per_image), device=proposals_per_image.proposal_boxes.device
            #     )[:self.batch_size_per_image]
            #     sampled_proposals.append(proposals_per_image[sampled_idxs])
            # proposals = sampled_proposals


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
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            # losses.update(self._forward_mask(features, proposals))
            # losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:

            pred_instances, predictions = self._forward_box_weak(features, proposals, compute_loss,
                                                            compute_val_loss, branch)

            return pred_instances, predictions

    def _forward_box_weak(
            self, features: Dict[str, torch.Tensor], proposals: List[Instances],
            compute_loss, compute_val_loss, branch
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

        objectness_logits = torch.cat([x.objectness_logits + 1 for x in proposals], dim=0)

        print("*" * 50)
        print("roi_wsddn")
        print("*" * 50)
        print("print(box_features) 1111111")
        print(box_features)
        print(torch.isnan(box_features).sum().item())

        box_features = box_features * objectness_logits.view(-1, 1, 1, 1)

        # torch.cuda.empty_cache()
        print("*" * 50)
        print("roi_wsddn")
        print("*" * 50)
        print("print(box_features) 22222")
        print(box_features)
        print(torch.isnan(box_features).sum().item())



        box_features = self.box_head(box_features)

        predictions = self.box_predictor.forward_weak(box_features, proposals)
        del box_features
        if (self.training and compute_loss) or compute_val_loss:
            losses = self.box_predictor.losses_weak(predictions, proposals, self.gt_classes_img_oh)
            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses, predictions
        else:
            pred_instances, _, all_scores, all_boxes = self.box_predictor.inference(
                predictions, proposals
            )
            return pred_instances, predictions



############


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
        """
        See :class:`ROIHeads.forward`.
        """

        del images
        if self.training and compute_loss:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)

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
            losses, _ = self._forward_box(features, proposals, compute_loss, compute_val_loss, branch)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            # losses.update(self._forward_mask(features, proposals))
            # losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:

            pred_instances, predictions = self._forward_box(features, proposals, compute_loss,
                                                            compute_val_loss, branch)

            return pred_instances, predictions

    def _forward_box(
            self, features: Dict[str, torch.Tensor], proposals: List[Instances],
            compute_loss, compute_val_loss, branch
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
        predictions = self.box_predictor(box_features, proposals)
        del box_features
        if (self.training and compute_loss) or compute_val_loss:
            losses = self.box_predictor.losses(predictions, proposals)
            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses, predictions
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances, predictions
