# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from detectron2.config import configurable
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.structures import Boxes
from detectron2.utils.events import get_event_storage
from torch import nn
from torch.nn import functional as F
from typing import Dict, Union
from detectron2.layers import Linear, ShapeSpec, cat, nonzero_tuple, cross_entropy
from fvcore.nn import giou_loss, smooth_l1_loss

from detectron2.modeling.roi_heads.fast_rcnn import (
    FastRCNNOutputLayers,
    FastRCNNOutputs, fast_rcnn_inference, _log_classification_stats,
)


# focal loss
class FastRCNNFocaltLossOutputLayers(FastRCNNOutputLayers):
    def __init__(self, cfg, input_shape):
        super(FastRCNNFocaltLossOutputLayers, self).__init__(cfg, input_shape)
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES

    def losses(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features
                that were used to compute predictions.
        """
        scores, proposal_deltas = predictions
        losses = FastRCNNFocalLoss(
            self.box2box_transform,
            scores,
            proposal_deltas,
            proposals,
            self.smooth_l1_beta,
            self.box_reg_loss_type,
            num_classes=self.num_classes,
        ).losses()

        return losses


class FastRCNNFocalLoss(FastRCNNOutputs):
    """
    A class that stores information about outputs of a Fast R-CNN head.
    It provides methods that are used to decode the outputs of a Fast R-CNN head.
    """

    def __init__(
            self,
            box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            smooth_l1_beta=0.0,
            box_reg_loss_type="smooth_l1",
            num_classes=80,
    ):
        super(FastRCNNFocalLoss, self).__init__(
            box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            smooth_l1_beta,
            box_reg_loss_type,
        )
        self.num_classes = num_classes

    def losses(self):
        return {
            "loss_cls": self.comput_focal_loss(),
            "loss_box_reg": self.box_reg_loss(),
        }

    def comput_focal_loss(self):
        if self._no_instances:
            return 0.0 * self.pred_class_logits.sum()
        else:
            FC_loss = FocalLoss(
                gamma=1.5,
                num_classes=self.num_classes,
            )
            total_loss = FC_loss(input=self.pred_class_logits, target=self.gt_classes)
            total_loss = total_loss / self.gt_classes.shape[0]

            return total_loss


class FocalLoss(nn.Module):
    def __init__(
            self,
            weight=None,
            gamma=1.0,
            num_classes=80,
    ):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

        self.num_classes = num_classes

    def forward(self, input, target):
        # focal loss
        CE = F.cross_entropy(input, target, reduction="none")
        p = torch.exp(-CE)
        loss = (1 - p) ** self.gamma * CE
        return loss.sum()


class WSDDNOutputs(object):
    """
    An internal implementation that stores information about outputs of a Fast R-CNN head,
    and provides methods that are used to decode the outputs of a Fast R-CNN head.
    """

    def __init__(
            self,
            box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            smooth_l1_beta=0.0,
            box_reg_loss_type="smooth_l1",
            mean_loss=False,
            gt_classes_img_oh=None,
    ):
        """
        Args:
            box2box_transform (Box2BoxTransform/Box2BoxTransformRotated):
                box2box transform instance for proposal-to-detection transformations.
            pred_class_logits (Tensor): A tensor of shape (R, K + 1) storing the predicted class
                logits for all R predicted object instances.
                Each row corresponds to a predicted object instance.
            pred_proposal_deltas (Tensor): A tensor of shape (R, K * B) or (R, B) for
                class-specific or class-agnostic regression. It stores the predicted deltas that
                transform proposals into final box detections.
                B is the box dimension (4 or 5).
                When B is 4, each row is [dx, dy, dw, dh (, ....)].
                When B is 5, each row is [dx, dy, dw, dh, da (, ....)].
            proposals (list[Instances]): A list of N Instances, where Instances i stores the
                proposals for image i, in the field "proposal_boxes".
                When training, each Instances must have ground-truth labels
                stored in the field "gt_classes" and "gt_boxes".
                The total number of all instances must be equal to R.
            smooth_l1_beta (float): The transition point between L1 and L2 loss in
                the smooth L1 loss function. When set to 0, the loss becomes L1. When
                set to +inf, the loss becomes constant 0.
            box_reg_loss_type (str): Box regression loss type. One of: "smooth_l1", "giou"
        """
        self.box2box_transform = box2box_transform
        self.num_preds_per_image = [len(p) for p in proposals]
        self.pred_class_logits = pred_class_logits
        self.pred_proposal_deltas = pred_proposal_deltas
        self.smooth_l1_beta = smooth_l1_beta
        self.box_reg_loss_type = box_reg_loss_type

        self.image_shapes = [x.image_size for x in proposals]

        if len(proposals):
            box_type = type(proposals[0].proposal_boxes)
            # cat(..., dim=0) concatenates over all images in the batch
            self.proposals = box_type.cat([p.proposal_boxes for p in proposals])
            assert (
                not self.proposals.tensor.requires_grad
            ), "Proposals should not require gradients!"

            # The following fields should exist only when training.
            if proposals[0].has("gt_boxes"):
                self.gt_boxes = box_type.cat([p.gt_boxes for p in proposals])
                assert proposals[0].has("gt_classes")
                self.gt_classes = cat([p.gt_classes for p in proposals], dim=0)
        else:
            self.proposals = Boxes(torch.zeros(0, 4, device=self.pred_proposal_deltas.device))
        self._no_instances = len(proposals) == 0  # no instances found

        self.mean_loss = mean_loss
        self.gt_classes_img_oh = gt_classes_img_oh

    def _log_accuracy(self):
        """
        Log the accuracy metrics to EventStorage.
        """
        num_instances = self.gt_classes.numel()
        pred_classes = self.pred_class_logits.argmax(dim=1)
        bg_class_ind = self.pred_class_logits.shape[1] - 1

        fg_inds = (self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)
        num_fg = fg_inds.nonzero().numel()
        fg_gt_classes = self.gt_classes[fg_inds]
        fg_pred_classes = pred_classes[fg_inds]

        num_false_negative = (fg_pred_classes == bg_class_ind).nonzero().numel()
        num_accurate = (pred_classes == self.gt_classes).nonzero().numel()
        fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero().numel()

        storage = get_event_storage()
        if num_instances > 0:
            storage.put_scalar("fast_rcnn/cls_accuracy", num_accurate / num_instances)
            if num_fg > 0:
                storage.put_scalar("fast_rcnn/fg_cls_accuracy", fg_num_accurate / num_fg)
                storage.put_scalar("fast_rcnn/false_negative", num_false_negative / num_fg)

    def softmax_cross_entropy_loss(self):
        """
        Compute the softmax cross entropy loss for box classification.
        Returns:
            scalar Tensor
        """
        if self._no_instances:
            return 0.0 * self.pred_class_logits.sum()
        else:
            self._log_accuracy()
            return F.cross_entropy(self.pred_class_logits, self.gt_classes, reduction="mean")

    def box_reg_loss(self):
        """
        Compute the smooth L1 loss for box regression.
        Returns:
            scalar Tensor
        """
        if self._no_instances:
            return 0.0 * self.pred_proposal_deltas.sum()

        box_dim = self.gt_boxes.tensor.size(1)  # 4 or 5
        cls_agnostic_bbox_reg = self.pred_proposal_deltas.size(1) == box_dim
        device = self.pred_proposal_deltas.device

        bg_class_ind = self.pred_class_logits.shape[1] - 1

        # Box delta loss is only computed between the prediction for the gt class k
        # (if 0 <= k < bg_class_ind) and the target; there is no loss defined on predictions
        # for non-gt classes and background.
        # Empty fg_inds produces a valid loss of zero as long as the size_average
        # arg to smooth_l1_loss is False (otherwise it uses torch.mean internally
        # and would produce a nan loss).
        fg_inds = nonzero_tuple((self.gt_classes >= 0) & (self.gt_classes < bg_class_ind))[0]
        if cls_agnostic_bbox_reg:
            # pred_proposal_deltas only corresponds to foreground class for agnostic
            gt_class_cols = torch.arange(box_dim, device=device)
        else:
            fg_gt_classes = self.gt_classes[fg_inds]
            # pred_proposal_deltas for class k are located in columns [b * k : b * k + b],
            # where b is the dimension of box representation (4 or 5)
            # Note that compared to Detectron1,
            # we do not perform bounding box regression for background classes.
            gt_class_cols = box_dim * fg_gt_classes[:, None] + torch.arange(box_dim, device=device)

        if self.box_reg_loss_type == "smooth_l1":
            gt_proposal_deltas = self.box2box_transform.get_deltas(
                self.proposals.tensor, self.gt_boxes.tensor
            )
            loss_box_reg = smooth_l1_loss(
                self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols],
                gt_proposal_deltas[fg_inds],
                self.smooth_l1_beta,
                reduction="sum",
            )
        elif self.box_reg_loss_type == "giou":
            loss_box_reg = giou_loss(
                self._predict_boxes()[fg_inds[:, None], gt_class_cols],
                self.gt_boxes.tensor[fg_inds],
                reduction="sum",
            )
        else:
            raise ValueError(f"Invalid bbox reg loss type '{self.box_reg_loss_type}'")

        # The loss is normalized using the total number of regions (R), not the number
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
        loss_box_reg = loss_box_reg / self.gt_classes.numel()
        return loss_box_reg

    def binary_cross_entropy_loss(self):
        """
        Compute the softmax cross entropy loss for box classification.
        Returns:
            scalar Tensor
        """
        self._log_accuracy()

        reduction = "mean" if self.mean_loss else "sum"
        return F.binary_cross_entropy(
            self.predict_probs_img(), self.gt_classes_img_oh, reduction=reduction
        ) / self.gt_classes_img_oh.size(0)

    def predict_probs_img(self):
        if not hasattr(self, "pred_class_img_logits"):
            self.pred_class_img_logits = cat(
                [
                    torch.sum(xx, dim=0, keepdim=True)
                    for xx in self.pred_class_logits.split(self.num_preds_per_image, dim=0)
                ],
                dim=0,
            )
            self.pred_class_img_logits = torch.clamp(
                self.pred_class_img_logits, min=1e-6, max=1.0 - 1e-6
            )
        return self.pred_class_img_logits

    def _predict_boxes(self):
        """
        Returns:
            Tensor: A Tensors of predicted class-specific or class-agnostic boxes
                for all images in a batch. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        return self.box2box_transform.apply_deltas(self.pred_proposal_deltas, self.proposals.tensor)

    """
    A subclass is expected to have the following methods because
    they are used to query information about the head predictions.
    """

    def losses(self):
        """
        Compute the default losses for box head in Fast(er) R-CNN,
        with softmax cross entropy loss and smooth L1 loss.
        Returns:
            A dict of losses (scalar tensors) containing keys "loss_cls" and "loss_box_reg".
        """
        # return {"loss_cls": self.softmax_cross_entropy_loss(), "loss_box_reg": self.box_reg_loss()}
        return {"loss_cls": self.binary_cross_entropy_loss()}

    def predict_boxes(self):
        """
        Deprecated
        """
        return self._predict_boxes().split(self.num_preds_per_image, dim=0)

    def predict_probs(self):
        """
        Deprecated
        """
        # probs = F.softmax(self.pred_class_logits, dim=-1)
        probs = self.pred_class_logits
        probs_bg = torch.zeros(
            probs.shape[0], 1, dtype=probs.dtype, device=probs.device, requires_grad=False
        )
        probs = torch.cat((probs, probs_bg), 1)
        return probs.split(self.num_preds_per_image, dim=0)

    def inference(self, score_thresh, nms_thresh, topk_per_image):
        """
        Deprecated
        """
        boxes = self.predict_boxes()
        scores = self.predict_probs()
        image_shapes = self.image_shapes
        return fast_rcnn_inference(
            boxes, scores, image_shapes, score_thresh, nms_thresh, topk_per_image
        )


class WSDDNOutputLayers(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:
      (1) proposal-to-detection box regression deltas
      (2) classification scores
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
            mean_loss: bool = True,
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
                    "loss_cls" - applied to classification loss
                    "loss_box_reg" - applied to box regression loss
        """
        super().__init__()
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)

        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        box_dim = len(box2box_transform.weights)

        self.num_bbox_reg_classes = num_bbox_reg_classes
        self.box_dim = box_dim

        self.cls_score = Linear(input_size, num_classes + 1)
        self.bbox_pred = Linear(input_size, num_classes)

        # nn.init.normal_(self.cls.weight, std=0.01)
        # nn.init.normal_(self.det.weight, std=0.01)
        nn.init.xavier_uniform_(self.cls_score.weight)
        nn.init.xavier_uniform_(self.bbox_pred.weight)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

        self.box2box_transform = box2box_transform
        self.smooth_l1_beta = smooth_l1_beta
        self.test_score_thresh = test_score_thresh
        self.test_nms_thresh = test_nms_thresh
        self.test_topk_per_image = test_topk_per_image
        self.box_reg_loss_type = box_reg_loss_type
        if isinstance(loss_weight, float):
            loss_weight = {"loss_cls": loss_weight, "loss_box_reg": loss_weight}
        self.loss_weight = loss_weight

        self.mean_loss = mean_loss

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
            "mean_loss": False  # cfg.WSL.MEAN_LOSS,
            # fmt: on
        }

    def forward_weak(self, x, proposals=None):
        """
        Args:
            x: per-region features of shape (N, ...) for N bounding boxes to predict.
        Returns:
            Tensor: shape (N,K+1), scores for each of the N box. Each row contains the scores for
                K object categories and 1 background class.
            Tensor: bounding box regression deltas for each box. Shape is shape (N,Kx4), or (N,4)
                for class-agnostic regression.
        """
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)

        if proposals is None:
            scores = F.softmax(self.cls_score(x), dim=1) * F.softmax(self.bbox_pred(x), dim=0)
        else:
            num_preds_per_image = [len(p) for p in proposals]
            scores = cat(
                [
                    F.softmax(self.cls_score(xx)[:, :-1], dim=1) * F.softmax(self.bbox_pred(xx), dim=0)
                    for xx in x.split(num_preds_per_image, dim=0)
                ],
                dim=0,
            )

        proposal_deltas = torch.zeros(
            scores.shape[0],
            self.num_bbox_reg_classes * self.box_dim,
            dtype=scores.dtype,
            device=scores.device,
            requires_grad=False,
        )

        return scores, proposal_deltas

    def forward(self, x, proposals=None):
        """
        Args:
            x: per-region features of shape (N, ...) for N bounding boxes to predict.
        Returns:
            Tensor: shape (N,K+1), scores for each of the N box. Each row contains the scores for
                K object categories and 1 background class.
            Tensor: bounding box regression deltas for each box. Shape is shape (N,Kx4), or (N,4)
                for class-agnostic regression.
        """
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)

        scores = self.cls_score(x)

        proposal_deltas = torch.zeros(
            scores.shape[0],
            self.num_bbox_reg_classes * self.box_dim,
            dtype=scores.dtype,
            device=scores.device,
            requires_grad=False,
        )

        return scores, proposal_deltas

    # TODO: move the implementation to this class.
    def losses_weak(self, predictions, proposals, gt_classes_img_oh):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_boxes``,
                ``gt_classes`` are expected.
        Returns:
            Dict[str, Tensor]: dict of losses
        """
        scores, proposal_deltas = predictions
        losses = WSDDNOutputs(
            self.box2box_transform,
            scores,
            proposal_deltas,
            proposals,
            self.smooth_l1_beta,
            self.box_reg_loss_type,
            self.mean_loss,
            gt_classes_img_oh,
        ).losses()
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}

    def losses(self, predictions, proposals):
        scores, proposal_deltas = predictions

        # parse classification outputs
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )
        _log_classification_stats(scores, gt_classes)
        losses = {
            "loss_cls": cross_entropy(scores, gt_classes, reduction="mean"),
        }
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}

    def inference(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.
        Returns:
            list[Instances]: same as `fast_rcnn_inference`.
            list[Tensor]: same as `fast_rcnn_inference`.
        """
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
            list[Tensor]: A list of Tensors of predicted boxes for GT classes in case of
                class-specific box head. Element i of the list has shape (Ri, B), where Ri is
                the number of proposals for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        scores, proposal_deltas = predictions
        proposal_boxes = [p.proposal_boxes for p in proposals]
        proposal_boxes = proposal_boxes[0].cat(proposal_boxes).tensor
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

    def predict_boxes(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.
        Returns:
            list[Tensor]: A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of proposals for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        _, proposal_deltas = predictions
        num_prop_per_image = [len(p) for p in proposals]
        proposal_boxes = [p.proposal_boxes for p in proposals]
        proposal_boxes = proposal_boxes[0].cat(proposal_boxes).tensor
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas, proposal_boxes
        )  # Nx(KxB)
        return predict_boxes.split(num_prop_per_image)

    def predict_probs(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions.
        Returns:
            list[Tensor]: A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of proposals for image i.
        """
        scores, _ = predictions
        num_inst_per_image = [len(p) for p in proposals]
        # probs = F.softmax(scores, dim=-1)
        probs = scores
        probs_bg = torch.zeros(
            probs.shape[0], 1, dtype=probs.dtype, device=probs.device, requires_grad=False
        )
        probs = torch.cat((probs, probs_bg), 1)
        return probs.split(num_inst_per_image, dim=0)

    def predict_probs_img(self, predictions, proposals):
        scores, _ = predictions
        num_inst_per_image = [len(p) for p in proposals]
        pred_class_img_logits = cat(
            [
                torch.sum(score, dim=0, keepdim=True)
                for score in scores.split(num_inst_per_image, dim=0)
            ],
            dim=0,
        )
        pred_class_img_logits = torch.clamp(pred_class_img_logits, min=1e-6, max=1.0 - 1e-6)
        return pred_class_img_logits
