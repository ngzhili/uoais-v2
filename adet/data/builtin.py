# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.

We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations

We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".

Users SHOULD NOT use this file to create new dataset / metadata for new dataset.
To add new dataset, refer to the tutorial "docs/DATASETS.md".
"""

import os

from detectron2.data import MetadataCatalog, DatasetCatalog
from .builtin_meta import _get_builtin_metadata
from .register_uoais import register_uoais_instances
from .register_wisdom import register_wisdom_instances



_PREDEFINED_SPLITS_uoais = {
    
    "uoais_sim_train_amodal": ("UOAIS-Sim/train", "UOAIS-Sim/annotations/coco_anns_uoais_sim_train.json"),
    "uoais_sim_val_amodal": ("UOAIS-Sim/val", "UOAIS-Sim/annotations/coco_anns_uoais_sim_val.json"),
    "uoais_sim_train_amodal_tabletop": ("UOAIS-Sim/train_tabletop", "UOAIS-Sim/annotations/coco_anns_uoais_sim_train_tabletop.json"),
    "uoais_sim_val_amodal_tabletop": ("UOAIS-Sim/val_tabletop", "UOAIS-Sim/annotations/coco_anns_uoais_sim_val_tabletop.json"),
    # added
    # "uoais_sim_train_amodal": ("train_room", "/home/ngzhili/uoais/datasets/train_room/uoais_train.json"),
    # "uoais_sim_val_amodal": ("test_room", "/home/ngzhili/uoais/datasets/test_room/uoais_train.json"),

    # "uoais_tabletop_train_amodal": ("uoais_tabletop_train","/home/ngzhili/uoais/datasets/uoais_tabletop_train/uoais_train.json"),
    # "uoais_tabletop_test_amodal": ("uoais_tabletop_test", "/home/ngzhili/uoais/datasets/uoais_tabletop_test/uoais_test.json"),
    "uoais_syntable_train_amodal": ("syntable/train","syntable/train/uoais_train.json"),
    "uoais_syntable_test_amodal": ("syntable/validation", "syntable/validation/uoais_val.json"),
     "uoais_syntable2_train_amodal": ("syntable2/train","syntable2/train/uoais_train.json"),
    "uoais_syntable2_test_amodal": ("syntable2/validation", "syntable2/validation/uoais_validation.json"),    
    "uoais_syntable_train_amodal_tabletop": ("syntable/train","syntable/train/uoais_train_tabletop.json"),
    "uoais_syntable_test_amodal_tabletop": ("syntable/validation", "syntable/validation/uoais_val_tabletop.json")
    }



def register_all_uoais(root="./datasets"):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_uoais.items():
        # Assume pre-defined datasets live in `./datasets`.
        amodal = "amodal" in key
        if "occ" in key:
            md = "uoais_occ"
        else:
            md = "uoais"
        
        # print("amodal:",amodal)
        # print("key:",key)
        # print("md:",md)
        register_uoais_instances(
            key,
            _get_builtin_metadata(md),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            amodal=amodal
        )

_PREDEFINED_SPLITS_WISDOM = {
    "wisdom_real_train": ("wisdom/wisdom-real/high-res", "wisdom/wisdom-real/high-res/annotations/coco_anns_wisdom_train.json"),
    "wisdom_real_test": ("wisdom/wisdom-real/high-res", "wisdom/wisdom-real/high-res/annotations/coco_anns_wisdom_test.json"),
}


def register_all_wisdom(root="./datasets"):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_WISDOM.items():
        # Assume pre-defined datasets live in `./datasets`.
        if "occ" in key:
            md = "wisdom_occ"
        else:
            md = "wisdom"
        register_wisdom_instances(
            key,
            _get_builtin_metadata(md),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )



# Register them all under "./datasets"
register_all_uoais()
register_all_wisdom()
