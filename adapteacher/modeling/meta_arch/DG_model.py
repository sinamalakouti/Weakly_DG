# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from torch.nn.parallel import DataParallel, DistributedDataParallel
import torch.nn as nn


class DG_model(nn.Module):
    def __init__(self, modelTeacher, s_f, s_w):
        super(DG_model, self).__init__()

        if isinstance(modelTeacher, (DistributedDataParallel, DataParallel)):
            modelTeacher = modelTeacher.module
        if isinstance(s_f, (DistributedDataParallel, DataParallel)):
            s_f = s_f.module
        if isinstance(s_w, (DistributedDataParallel, DataParallel)):
            s_w = s_w.module

        self.modelTeacher = modelTeacher
        self.s_f = s_f
        self.s_w = s_w

    def forward(self, model_type, branch, *args):
        if model_type == 'teacher':
            return self.modelTeacher(args, branch=branch)
        elif model_type == 's_f':
            return self.s_f(args, branch=branch)
        elif model_type == 's_w':
            return self.s_w(args, branch=branch)
