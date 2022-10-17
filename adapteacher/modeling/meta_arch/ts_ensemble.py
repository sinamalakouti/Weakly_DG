# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from torch.nn.parallel import DataParallel, DistributedDataParallel
import torch.nn as nn


class EnsembleTSModel(nn.Module):
    def __init__(self, modelTeacher, s1_head, s2_head):
        super(EnsembleTSModel, self).__init__()

        if isinstance(modelTeacher, (DistributedDataParallel, DataParallel)):
            modelTeacher = modelTeacher.module
        if isinstance(s1_head, (DistributedDataParallel, DataParallel)):
            s1_head = s1_head.module
        if isinstance(s2_head, (DistributedDataParallel, DataParallel)):
            s2_head = s2_head.module

        self.modelTeacher = modelTeacher
        self.s1_head = s1_head
        self.s2_head = s2_head


