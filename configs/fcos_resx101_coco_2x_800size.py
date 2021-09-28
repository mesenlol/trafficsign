# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from megengine import hub

import models

class CustomFCOSConfig(models.FCOSConfig):
    def __init__(self):
        super().__init__()

        self.backbone = "resnext101_32x8d"
        # ------------------------ dataset cfg ---------------------- #
        self.train_dataset = dict(
            name="traffic5",
            root="images",
            ann_file="annotations/train.json",
            remove_images_without_annotations=True,
        )
        self.test_dataset = dict(
            name="traffic5",
            root="images",
            ann_file="annotations/val.json",
            test_final_ann_file="annotations/test.json",
            remove_images_without_annotations=False,
        )
        self.num_classes = 5

        # ------------------------ training cfg ---------------------- #
        # 原始1x配置
        # self.basic_lr = 0.02 / 16  
        # self.max_epoch = 24
        # self.lr_decay_stages = [16, 21]

        # 自定义1x配置
        self.basic_lr = 0.02 / 160  
        self.max_epoch = 17
        self.lr_decay_stages = [12]
#         self.nr_images_epoch = 2226 # 数据均衡后的图片数量
        
        # 原始2x配置
        # self.basic_lr = 0.01 / 16  
#         self.max_epoch = 36
#         self.lr_decay_stages = [24, 32]
        self.nr_images_epoch = 2674
        self.warm_iters = 50
        self.log_interval = 10
        # resume
#         self.max_epoch = 26
#         self.lr_decay_stages = [12, 20]
#         self.warm_iters = 0

#@hub.pretrained(
#    "https://data.megengine.org.cn/models/weights/"
#    "fcos_resx101_coco_2x_800size_44dot8_42ac8e82.pkl"
#)
def fcos_resx101_coco_2x_800size(**kwargs):
    r"""
    FCOS trained from COCO dataset.
    `"FCOS" <https://arxiv.org/abs/1904.01355>`_
    `"FPN" <https://arxiv.org/abs/1612.03144>`_
    `"COCO" <https://arxiv.org/abs/1405.0312>`_
    """
    cfg = CustomFCOSConfig()
    cfg.backbone_pretrained = False
    return models.FCOS(cfg, **kwargs)


Net = models.FCOS
Cfg = CustomFCOSConfig