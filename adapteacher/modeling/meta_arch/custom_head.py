# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import numpy as np
import torch
import torch.nn as nn
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.modeling import ResNet
from torch.nn import functional as F
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.config import configurable
# from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
# from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
import logging
from typing import Dict, Tuple, List, Optional
from collections import OrderedDict
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.backbone import build_backbone, Backbone, BasicStem
from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.utils.events import get_event_storage
from detectron2.structures import ImageList

from adapteacher.modeling.meta_arch.model_definition import Generator


#######################

class Custom_head(nn.Module):

    def __init__(
            self,
            proposal_generator: nn.Module,
            roi_heads: nn.Module,
            cfg: None,
            backbone_output_shape: None,
            vis_period: int = 0,
            weak_head: bool = False

    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()

        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads
        self.weak_head = weak_head
        self.vis_period = vis_period
        self.proposal_generator = build_proposal_generator(cfg, backbone_output_shape)
        if weak_head:
            cfg.defrost()
            cfg.MODEL.ROI_HEADS.NAME = 'WSDDNROIHeads'
            cfg.freeze()
        self.roi_heads = build_roi_heads(cfg, backbone_output_shape)
        self.weak_head = weak_head

    def forward(
            self, features, images, gt_instances, branch="supervised", given_proposals=None, val_mode=False
    ):

        if branch.startswith("supervised"):

            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images,
                features,
                proposals_rpn,
                compute_loss=True,
                targets=gt_instances,
                #  branch=branch,
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)

            return losses, [], [], None

        elif branch.startswith("supervised_target"):

            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # roi_head lower branch  TODO: check how roi_head works
            _, detector_losses = self.roi_heads(
                images,
                features,
                proposals_rpn,
                compute_loss=True,
                targets=gt_instances,
                branch=branch,
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None

        elif branch.startswith("mil"):

            # Region proposal network
            with torch.no_grad():
                proposals_rpn, _ = self.proposal_generator(
                    images, features, gt_instances, compute_loss=False
                )

            # roi_head lower branch  TODO: check how roi_head works
            _, detector_losses = self.roi_heads.forward_weak(
                images,
                features,
                proposals_rpn,
                compute_loss=True,
                targets=gt_instances,
                branch=branch,
            )

            losses = {}
            losses.update(detector_losses)
            # losses.update(proposal_losses)
            return losses, [], [], None

        elif branch == "unsup_data_weak":  # TODO what is this for :)
            """
            unsupervised weak branch: input image without any ground-truth label; output proposals of rpn and roi-head
            """
            # Region proposal network
            proposals_rpn, _ = self.proposal_generator(
                images, features, None, compute_loss=False
            )

            # roi_head lower branch (keep this for further production)
            # notice that we do not use any target in ROI head to do inference!
            proposals_roih, ROI_predictions = self.roi_heads(
                images,
                features,
                proposals_rpn,
                targets=None,
                compute_loss=False,
                # branch=branch,
            )
            return {}, proposals_rpn, proposals_roih, ROI_predictions
        elif branch == "unsup_data_strong":
            raise NotImplementedError()
        elif branch == "val_loss":
            raise NotImplementedError()

    def visualize_training(self, batched_inputs, proposals, branch=""):
        """
        This function different from the original one:
        - it adds "branch" to the `vis_name`.

        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 predicted object
        proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20
        i = 0
        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = (
                    "Left: GT bounding boxes "
                    + branch
                    + ";  Right: Predicted proposals "
                    + branch
            )
            storage.put_image(vis_name, vis_img)
            from torchvision.utils import save_image
            save_image(torch.tensor(vis_img) / 255, 'src_img_{}.png'.format(i))
            i += 1
        #   break  # only visualize one image in a batch
