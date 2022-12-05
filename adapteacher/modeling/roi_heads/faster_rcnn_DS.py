# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from detectron2.config import configurable
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.structures import Boxes, Instances
from detectron2.utils.events import get_event_storage
from torch import nn
from torch.nn import functional as F
from typing import Dict, Union, Tuple, List
from detectron2.layers import Linear, ShapeSpec, cat, nonzero_tuple, cross_entropy
from fvcore.nn import giou_loss, smooth_l1_loss

from detectron2.modeling.roi_heads.fast_rcnn import (
    FastRCNNOutputLayers,
    FastRCNNOutputs, fast_rcnn_inference, _log_classification_stats,
)


class DSFastRCNNOutputLayers(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:

    1. proposal-to-detection box regression deltas
    2. classification scores
    """

    @configurable
    def __init__(
            self,
            input_shape: ShapeSpec,
            *,
            box2box_transform,
            num_classes: int,
            test_score_thresh: float = 0.0,
            test_nms_thresh: float = 0.5,
            test_topk_per_image: int = 100,
            cls_agnostic_bbox_reg: bool = False,
            smooth_l1_beta: float = 0.0,
            box_reg_loss_type: str = "smooth_l1",
            loss_weight: Union[float, Dict[str, float]] = 1.0,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature to this module
            box2box_transform (Box2BoxTransform or Box2BoxTransformRotated):
            num_classes (int): number of foreground classes
            test_score_thresh (float): threshold to filter predictions results.
            test_nms_thresh (float): NMS threshold for prediction results.
            test_topk_per_image (int): number of top predictions to produce per image.
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            smooth_l1_beta (float): transition point from L1 to L2 loss. Only used if
                `box_reg_loss_type` is "smooth_l1"
            box_reg_loss_type (str): Box regression loss type. One of: "smooth_l1", "giou"
            loss_weight (float|dict): weights to use for losses. Can be single float for weighting
                all losses, or a dict of individual weightings. Valid dict keys are:
                    * "loss_cls": applied to classification loss
                    * "loss_box_reg": applied to box regression loss
        """
        super().__init__()
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        self.num_classes = num_classes
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        # prediction layer for num_classes foreground classes and one background class (hence + 1)


        self.cls_score = nn.Linear(input_size, num_classes + 1)
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        box_dim = len(box2box_transform.weights)

        self.cls_score_fsod = nn.Linear(input_size, num_classes + 1)
        self.bbox_pred_fsod = nn.Linear(input_size, num_bbox_reg_classes * box_dim)
        self.weak_score_fsod = nn.Linear(input_size, num_classes)

        self.cls_score_wsod = nn.Linear(input_size, num_classes + 1)
        self.weak_score_wsod = nn.Linear(input_size, num_classes)

        nn.init.normal_(self.cls_score_fsod.weight, std=0.01)
        nn.init.normal_(self.bbox_pred_fsod.weight, std=0.001)
        nn.init.normal_(self.weak_score_fsod.weight, std=0.01)

        nn.init.normal_(self.cls_score_wsod.weight, std=0.01)
        nn.init.normal_(self.weak_score_wsod.weight, std=0.01)

        for l in [self.cls_score_fsod, self.bbox_pred_fsod, self.weak_score_fsod]:
            nn.init.constant_(l.bias, 0)

        for l in [self.cls_score_wsod, self.weak_score_wsod]:
            nn.init.constant_(l.bias, 0)

        self.box2box_transform = box2box_transform
        self.smooth_l1_beta = smooth_l1_beta
        self.test_score_thresh = test_score_thresh
        self.test_nms_thresh = test_nms_thresh
        self.test_topk_per_image = test_topk_per_image
        self.box_reg_loss_type = box_reg_loss_type
        if isinstance(loss_weight, float):
            loss_weight = {"loss_cls": loss_weight, "loss_box_reg": loss_weight, "loss_mil": loss_weight}
        self.loss_weight = loss_weight

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            "input_shape": input_shape,
            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS),
            # fmt: off
            "num_classes": cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "cls_agnostic_bbox_reg": cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,
            "smooth_l1_beta": cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA,
            "test_score_thresh": cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
            "test_nms_thresh": cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "box_reg_loss_type": cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE,
            "loss_weight": {"loss_box_reg": cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT},
            # fmt: on
        }

    def forward(self, x, branch=None):
        """
        Args:
            x: per-region features of shape (N, ...) for N bounding boxes to predict.

        Returns:
            (Tensor, Tensor):
            First tensor: shape (N,K+1), scores for each of the N box. Each row contains the
            scores for K object categories and 1 background class.

            Second tensor: bounding box regression deltas for each box. Shape is shape (N,Kx4),
            or (N,4) for class-agnostic regression.
        """

        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        if branch == 'all_fsod':
            scores = self.cls_score_fsod(x)
            proposal_deltas = self.bbox_pred_fsod(x)
            weak_scores = self.weak_score_fsod(x)
            return scores, proposal_deltas, weak_scores
        elif branch == 'mil_wsod':
            scores = self.cls_score_wsod(x)
            weak_scores = self.weak_score_wsod(x)
            return scores, None, weak_scores
        elif branch == 'episodic_fsod':
            scores = self.cls_score_fsod(x)
            proposal_deltas = self.bbox_pred_fsod(x)
            weak_scores = self.weak_score_wsod(x)
            return scores, proposal_deltas, weak_scores
        elif branch == 'episodic_wsod':
            scores = self.cls_score_wsod(x)
            weak_scores = self.weak_score_fsod(x)
            return scores, None, weak_scores
        elif branch == 'supervised_wsod':
            scores = self.cls_score_wsod(x)
            weak_scores = self.weak_score_fsod(x)
            return scores, None, weak_scores
        elif branch == 'unsup_data_weak':
            scores = self.cls_score_fsod(x)
            proposal_deltas = self.bbox_pred_fsod(x)
            weak_scores = self.weak_score_fsod(x)
            return scores, proposal_deltas, weak_scores
        else:
            assert  False, "faster_rcnn_DS:  wrong branch name :)"


    def predict_probs_img(self, pred_class_logits, pred_det_logits, num_preds_per_image):
        cls_scores = F.softmax(pred_class_logits[:, :-1], dim=1)
        pred_class_img_logits = cat(
            [torch.sum(cls * F.softmax(det_logit, dim=0), dim=0, keepdim=True)
             for cls, det_logit in zip(cls_scores.split(num_preds_per_image, dim=0),
                                       pred_det_logits.split(num_preds_per_image, dim=0))
             ],
            dim=0,
        )

        pred_class_img_logits = torch.clamp(pred_class_img_logits, min=1e-6, max=1.0 - 1e-6)
        return pred_class_img_logits

    # def binary_cross_entropy_loss(self, pred_class_img_logits, gt_classes_img_oh):
    #     """
    #     Compute the softmax cross entropy loss for box classification.
    #     Returns:
    #         scalar Tensor
    #     """
    #     self._log_accuracy()
    #     assert pred_class_img_logits.shape == gt_classes_img_oh.shape, " {} != {}".format(
    #         pred_class_img_logits, gt_classes_img_oh.shape)
    #
    #     reduction = "mean" if self.mean_loss else "sum"
    #     return F.binary_cross_entropy(
    #         self.predict_probs_img(), gt_classes_img_oh, reduction=reduction
    #     ) / self.gt_classes_img_oh.size(0)

    def losses(self, predictions, proposals, gt_classes_img_oh, branch):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_boxes``,
                ``gt_classes`` are expected.

        Returns:
            Dict[str, Tensor]: dict of losses
        """
        scores, proposal_deltas, weak_scores = predictions

        # get number of  predictions per image
        num_preds_per_image = [len(p) for p in proposals]

        # parse classification outputs
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )
        _log_classification_stats(scores, gt_classes)

        # parse box regression outputs
        if len(proposals):
            proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
            assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
            # If "gt_boxes" does not exist, the proposals must be all negative and
            # should not be included in regression loss computation.
            # Here we just use proposal_boxes as an arbitrary placeholder because its
            # value won't be used in self.box_reg_loss().
            gt_boxes = cat(
                [(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals],
                dim=0,
            )
        else:
            proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)

        pred_class_img_logits = self.predict_probs_img(scores, weak_scores, num_preds_per_image)

        img_cls_losses = F.binary_cross_entropy(
            pred_class_img_logits,
            gt_classes_img_oh,
            reduction='mean'
        ) / gt_classes_img_oh.size(0)

        if branch == "all_fsod" or branch == "episodic_fsod":
            losses = {
                "loss_cls": cross_entropy(scores, gt_classes, reduction="mean"),
                "loss_box_reg": self.box_reg_loss(
                    proposal_boxes, gt_boxes, proposal_deltas, gt_classes
                ),
                "loss_mil": img_cls_losses
            }
        elif branch == 'mil_wsod' or branch == 'episodic_wsod':
            losses = {
                "loss_cls": cross_entropy(scores, gt_classes, reduction="mean"),
                "loss_box_reg": 0.0,
                "loss_mil": img_cls_losses
            }
        elif branch == "supervised_wsod":
            losses = {
                "loss_cls": cross_entropy(scores, gt_classes, reduction="mean"),
                "loss_box_reg": 0.0,
                "loss_mil": img_cls_losses
            }
        else:
            assert False, "faster_rcnn_DS: incorrect branch!"
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}

    def box_reg_loss(self, proposal_boxes, gt_boxes, pred_deltas, gt_classes):
        """
        Args:
            All boxes are tensors with the same shape Rx(4 or 5).
            gt_classes is a long tensor of shape R, the gt class label of each proposal.
            R shall be the number of proposals.
        """
        box_dim = proposal_boxes.shape[1]  # 4 or 5
        # Regression loss is only computed for foreground proposals (those matched to a GT)
        fg_inds = nonzero_tuple((gt_classes >= 0) & (gt_classes < self.num_classes))[0]
        if pred_deltas.shape[1] == box_dim:  # cls-agnostic regression
            fg_pred_deltas = pred_deltas[fg_inds]
        else:
            fg_pred_deltas = pred_deltas.view(-1, self.num_classes, box_dim)[
                fg_inds, gt_classes[fg_inds]
            ]

        if self.box_reg_loss_type == "smooth_l1":
            gt_pred_deltas = self.box2box_transform.get_deltas(
                proposal_boxes[fg_inds],
                gt_boxes[fg_inds],
            )
            loss_box_reg = smooth_l1_loss(
                fg_pred_deltas, gt_pred_deltas, self.smooth_l1_beta, reduction="sum"
            )
        elif self.box_reg_loss_type == "giou":
            fg_pred_boxes = self.box2box_transform.apply_deltas(
                fg_pred_deltas, proposal_boxes[fg_inds]
            )
            loss_box_reg = giou_loss(fg_pred_boxes, gt_boxes[fg_inds], reduction="sum")
        else:
            raise ValueError(f"Invalid bbox reg loss type '{self.box_reg_loss_type}'")
        # The reg loss is normalized using the total number of regions (R), not the number
        # of foreground regions even though the box regression loss is only defined on
        # foreground regions. Why? Because doing so gives equal training influence to
        # each foreground example. To see how, consider two different minibatches:
        #  (1) Contains a single foreground region
        #  (2) Contains 100 foreground regions
        # If we normalize by the number of foreground regions, the single example in
        # minibatch (1) will be given 100 times as much influence as each foreground
        # example in minibatch (2). Normalizing by the total number of regions, R,
        # means that the single example in minibatch (1) and each of the 100 examples
        # in minibatch (2) are given equal influence.
        return loss_box_reg / max(gt_classes.numel(), 1.0)  # return 0 if empty

    def inference(self, predictions: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], proposals: List[Instances]):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Instances]: same as `fast_rcnn_inference`.
            list[Tensor]: same as `fast_rcnn_inference`.
        """
        predictions = (predictions[0], predictions[1])
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]
        return fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
        )

    def predict_boxes_for_gt_classes(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_classes`` are expected.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted boxes for GT classes in case of
                class-specific box head. Element i of the list has shape (Ri, B), where Ri is
                the number of proposals for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        scores, proposal_deltas = predictions
        proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)
        N, B = proposal_boxes.shape
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas, proposal_boxes
        )  # Nx(KxB)

        K = predict_boxes.shape[1] // B
        if K > 1:
            gt_classes = torch.cat([p.gt_classes for p in proposals], dim=0)
            # Some proposals are ignored or have a background class. Their gt_classes
            # cannot be used as index.
            gt_classes = gt_classes.clamp_(0, K - 1)

            predict_boxes = predict_boxes.view(N, K, B)[
                torch.arange(N, dtype=torch.long, device=predict_boxes.device), gt_classes
            ]
        num_prop_per_image = [len(p) for p in proposals]
        return predict_boxes.split(num_prop_per_image)

    def predict_boxes(
            self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]
    ):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of proposals for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        _, proposal_deltas = predictions
        num_prop_per_image = [len(p) for p in proposals]
        proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas,
            proposal_boxes,
        )  # Nx(KxB)
        return predict_boxes.split(num_prop_per_image)

    def predict_probs(
            self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]
    ):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of proposals for image i.
        """
        scores, _ = predictions
        num_inst_per_image = [len(p) for p in proposals]
        probs = F.softmax(scores, dim=-1)
        return probs.split(num_inst_per_image, dim=0)
