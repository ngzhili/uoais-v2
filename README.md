# Unseen Object Amodal Instance Segmentation (UOAIS) v2

This repository contains source codes for the paper "SynTable: A Synthetic Data Generation Pipeline for Unseen Object Amodal Instance Segmentation of Cluttered Tabletop Scenes". The source code has added synthetic data evaluation code and Occlusion Order Accuracy (OOACC) implemention to the UOAIS-Net source code from the "Unseen Object Amodal Instance Segmentation via Hierarchical Occlusion Modeling" (ICRA 2022) paper.

## Getting Started

### Environment Setup

Tested on NVIDIA Tesla V100 with python 3.7, pytorch 1.8.0, torchvision 0.9.0, CUDA 10.2 / 11.1 and detectron2 v0.5 / v0.6

1. Download source codes and checkpoints
```
git clone https://github.com/ngzhili/uoais-v2.git
cd uoais
mkdir output
```
2. Download checkpoints at [GDrive](https://drive.google.com/drive/folders/1D5hHFDtgd5RnX__55MmpfOAM83qdGYf0?usp=sharing) 

3. Move the `R50_depth_mlc_occatmask_hom_concat` and `R50_rgbdconcat_mlc_occatmask_hom_concat` to the `output` folder.

4. Move the `rgbd_fg.pth` to the `foreground_segmentation` folder


5. Set up a python environment
```
conda create -n uoais python=3.8
conda activate uoais
pip install torch torchvision 
pip install shapely torchfile opencv-python pyfastnoisesimd rapidfuzz termcolor
```
6. Install [detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)

7. Build custom [AdelaiDet](https://github.com/aim-uofa/AdelaiDet)
```
python setup.py build develop 
```

### Run on sample OSD dataset

<img src="./imgs/demo.png" height="200">

```
# UOAIS-Net (RGB-D) + CG-Net (foreground segmentation)
python tools/run_on_OSD.py --use-cgnet --dataset-path ./sample_data --config-file configs/R50_rgbdconcat_mlc_occatmask_hom_concat.yaml

# UOAIS-Net (depth) + CG-Net (foreground segmentation)
python tools/run_on_OSD.py --use-cgnet --dataset-path ./sample_data  --config-file configs/R50_depth_mlc_occatmask_hom_concat.yaml

# UOAIS-Net (RGB-D)
python tools/run_on_OSD.py --dataset-path ./sample_data --config-file configs/R50_rgbdconcat_mlc_occatmask_hom_concat.yaml

# UOAIS-Net (depth)
python tools/run_on_OSD.py --dataset-path ./sample_data --config-file configs/R50_depth_mlc_occatmask_hom_concat.yaml
```

### Run with ROS

1. Realsense D-435 ([realsense-ros](https://github.com/IntelRealSense/realsense-ros) is required.)
```
# launch realsense2 driver
roslaunch realsense2_camera rs_aligned_depth.launch
# launch uoais node
roslaunch uoais uoais_rs_d435.launch 
# or you can use rosrun
rosrun uoais uoais_node.py _mode:="topic"

```

2. Azure Kinect ([Azure_kinect_ROS_Driver](https://github.com/microsoft/Azure_Kinect_ROS_Driver) is required)

```
# launch azure kinect driver
roslaunch azure_kinect_ros_driver driver.launch
# launch uoais node
roslaunch uoais uoais_k4a.launch
```

#### Topics & service
- `/uoais/vis_img` (`sensor_msgs/Image`): visualization results
- `/uoais/results` (`uoais/UOAISResults`): UOAIS-Net predictions (`mode:=topic`)
- `/get_uoais_results` (`uoais/UOAISRequest`): UOAIS-Net predictions (`mode:=service`)

#### Parameters
- `mode` (`string`): running mode of ros node (`topic` or `service`)
- `rgb` (`string`):  topic name of the input rgb
- `depth` (`string`):  topic name of the input depth
- `camera_info` (`string`):  topic name of the input camera info
- `use_cgnet` (`bool`): use CG-Net [1] for foreground segmentation or not 
- `use_planeseg` (`bool`): use RANSAC for plane segmentation or not
- `ransac_threshold` (`float`): max distance a point can be from the plane model



### Run without ROS

1. Realsense D-435 ([librealsense](https://github.com/IntelRealSense/librealsense) and [pyrealsense2](https://pypi.org/project/pyrealsense2/) are required.)

```
# UOAIS-Net (RGB-D) + CG-Net (foreground segmentation)
python tools/rs_demo.py --use-cgnet --config-file configs/R50_rgbdconcat_mlc_occatmask_hom_concat.yaml

# UOAIS-Net (depth) + CG-Net (foreground segmentation)
python tools/rs_demo.py --use-cgnet --config-file configs/R50_depth_mlc_occatmask_hom_concat.yaml

# UOAIS-Net (RGB-D)
python tools/rs_demo.py --config-file configs/R50_rgbdconcat_mlc_occatmask_hom_concat.yaml

# UOAIS-Net (depth)
python tools/rs_demo.py --config-file configs/R50_depth_mlc_occatmask_hom_concat.yaml
```


2. Azure Kinect ([Azure-Kinect-Sensor-SDK](https://github.com/microsoft/Azure-Kinect-Sensor-SDK) and [pyk4a](https://github.com/etiennedub/pyk4a) are required.)

```
# UOAIS-Net (RGB-D) + CG-Net (foreground segmentation)
python tools/k4a_demo.py --use-cgnet --config-file configs/R50_rgbdconcat_mlc_occatmask_hom_concat.yaml

# UOAIS-Net (depth) + CG-Net (foreground segmentation)
python tools/k4a_demo.py --use-cgnet --config-file configs/R50_depth_mlc_occatmask_hom_concat.yaml

# UOAIS-Net (RGB-D)
python tools/k4a_demo.py --config-file configs/R50_rgbdconcat_mlc_occatmask_hom_concat.yaml

# UOAIS-Net (depth)
python tools/k4a_demo.py --config-file configs/R50_depth_mlc_occatmask_hom_concat.yaml
```


## Train & Evaluation

### Dataset Preparation

1. Download `UOAIS-Sim.zip` and `OSD-Amodal-annotations.zip` at [GDrive](https://drive.google.com/drive/folders/1D5hHFDtgd5RnX__55MmpfOAM83qdGYf0?usp=sharing) 
2. Download `OSD-0.2-depth.zip` at [OSD](https://www.acin.tuwien.ac.at/vision-for-robotics/software-tools/osd/). [2]
4. Download `OCID dataset` at [OCID](https://www.acin.tuwien.ac.at/en/vision-for-robotics/software-tools/object-clutter-indoor-dataset/). [3]
5. Extract the downloaded datasets and organize the folders as follows
```
uoais
├── output
└── datasets
       ├── OCID-dataset # for evaluation on indoor scenes
       │     └──ARID10
       │     └──ARID20
       │     └──YCB10
       ├── OSD-0.20-depth # for evaluation on tabletop scenes
       │     └──amodal_annotation # OSD-amodal
       │     └──annotation
       │     └──disparity
       │     └──image_color
       │     └──occlusion_annotation # OSD-amodal
       └── UOAIS-Sim # for training
              └──annotations
              └──train
              └──val
```

### Train on UOAIS-Sim
```
# UOAIS-Net (RGB-D) 
python train_net.py --config-file configs/R50_rgbdconcat_mlc_occatmask_hom_concat.yaml

# UOAIS-Net (depth) 
python train_net.py --config-file configs/R50_depth_mlc_occatmask_hom_concat.yaml 

# Amodal-MRCNN (RGB-D)
python train_net.py --config-file configs/R50_rgbdconcat_amodalmrcnn_concat.yaml

# ASN
python train_net.py --config-file configs/R50_rgbdconcat_asn7.yaml

python train_net.py --config-file configs/R50_rgbdconcat_syntable_rerun_AMRCNN.yaml

python train_net.py --config-file /home/ngzhili/uoais/configs/R50_rgbdconcat_mlc_occatmask_hom_concat_syntable_100k.yaml --gpu 2

python train_net.py --config-file /home/ngzhili/uoais/configs/R50_rgbdconcat_mlc_occatmask_hom_concat_syntable_100k_no_aug.yaml --gpu 0
```

### Evaluation on OSD dataset

```
# UOAIS-Net (RGB-D) + CG-Net (foreground segmentation)
python eval/eval_on_OSD.py --config-file configs/R50_rgbdconcat_mlc_occatmask_hom_concat.yaml --use-cgnet

# UOAIS-Net (depth) + CG-Net (foreground segmentation)
python eval/eval_on_OSD.py --config-file configs/R50_depth_mlc_occatmask_hom_concat.yaml --use-cgnet
```
This code evaluates the UOAIS-Net that was trained on a single seed (7), thus the metrics from this code and the paper (an average of seeds 7, 77, 777) can be different.


###### no augmentation eval codes
```
python eval/eval_on_OSD.py --config-file configs/R50_rgbdconcat_mlc_occatmask_hom_concat_uoais_sim_tabletop_no_aug.yaml --gpu 1

python eval/eval_on_OSD.py --config-file configs/R50_rgbdconcat_mlc_occatmask_hom_concat_syntable_tabletop_no_aug.yaml --gpu 2

python eval/eval_on_OSD.py --config-file configs/R50_rgbdconcat_mlc_occatmask_hom_concat_syntable_rerun_no_aug.yaml --gpu 3


python eval/eval_on_OSD.py --config-file /home/ngzhili/uoais/configs/R50_rgbdconcat_mlc_occatmask_hom_concat_syntable_100k_no_aug.yaml --gpu 0


python eval/eval_on_synthetic_data.py --config-file configs/R50_rgbdconcat_mlc_occatmask_hom_concat_uoais_sim_tabletop_no_aug.yaml --gpu 1


python eval/eval_on_synthetic_data.py --config-file configs/R50_rgbdconcat_mlc_occatmask_hom_concat_syntable_rerun_no_aug.yaml --gpu 2
```

###### crop, no aug, distort eval codes
```
python eval/eval_on_OSD.py --config-file configs/R50_rgbdconcat_mlc_occatmask_hom_concat_uoais_sim_tabletop_crop_no_aug_distort.yaml --gpu 1


python eval/eval_on_OSD.py --config-file configs/R50_rgbdconcat_mlc_occatmask_hom_concat_syntable_rerun_crop_no_aug_distort.yaml --gpu 3


python eval/eval_on_OSD.py --config-file configs/R50_rgbdconcat_mlc_occatmask_hom_concat_syntable_tabletop_crop_no_aug_distort.yaml --gpu 2
```


###### crop eval codes
```
python eval/eval_on_OSD.py --config-file configs/R50_rgbdconcat_mlc_occatmask_hom_concat_uoais_sim_tabletop_crop.yaml --gpu 1

python eval/eval_on_OSD.py --config-file configs/R50_rgbdconcat_mlc_occatmask_hom_concat_syntable_rerun_crop.yaml --gpu 2

python eval/eval_on_OSD.py --config-file configs/R50_rgbdconcat_mlc_occatmask_hom_concat_syntable_tabletop_crop.yaml --gpu 3
```
###### default eval codes
```
python eval/eval_on_OSD.py --config-file configs/R50_rgbdconcat_mlc_occatmask_hom_concat_uoais_sim_tabletop.yaml --gpu 1

python eval/eval_on_OSD.py --config-file configs/R50_rgbdconcat_mlc_occatmask_hom_concat_syntable_rerun.yaml --gpu 2

python eval/eval_on_OSD.py --config-file configs/R50_rgbdconcat_mlc_occatmask_hom_concat_syntable_tabletop.yaml --gpu 3
```


###### other eval codes
```
python eval/eval_on_OSD.py --config-file configs/R50_rgbdconcat_mlc_occatmask_hom_concat_syntable_rerun.yaml --gpu 1

python eval/eval_on_OSD.py --config-file configs/R50_rgbdconcat_mlc_occatmask_hom_concat_uoais_sim_tabletop.yaml --gpu 1

python eval/eval_on_synthetic_data.py --config-file configs/R50_rgbdconcat_mlc_occatmask_hom_concat_syntable_rerun.yaml --gpu 2

python eval/eval_on_synthetic_data.py --config-file configs/R50_rgbdconcat_mlc_occatmask_hom_concat_uoais_sim_tabletop.yaml --gpu 3


python eval/eval_on_OSD.py --config-file configs/R50_rgbdconcat_mlc_occatmask_hom_concat_syntable_100k.yaml --gpu 1

python eval/eval_on_OSD.py --config-file configs/R50_rgbdconcat_amodalmrcnn_concat7.yaml --use-cgnet --gpu 2

python eval/eval_on_OSD.py --config-file configs/R50_rgbdconcat_AMRCNN7_retrain.yaml --use-cgnet --gpu 2


python eval/eval_on_OSD.py --config-file configs/R50_rgbdconcat_amodalmrcnn_concat.yaml --use-cgnet

python eval/eval_on_OSD.py --config-file configs/R50_rgbdconcat_asn7.yaml --use-cgnet --gpu 2

python eval/eval_on_OSD.py --config-file configs/R50_rgbdconcat_AMRCNN.yaml --use-cgnet --gpu 2

python eval/eval_on_OSD.py --config-file /home/ngzhili/uoais/configs/R50_rgbdconcat_mlc_occatmask_hom_concat_syntable_100k.yaml --gpu 2
```

### Evaluation on OCID dataset

```
# UOAIS-Net (RGB-D)
python eval/eval_on_OCID.py --config-file configs/R50_rgbdconcat_mlc_occatmask_hom_concat.yaml
# UOAIS-Net (depth)
python eval/eval_on_OCID.py --config-file configs/R50_depth_mlc_occatmask_hom_concat.yaml
```

### Visualization on OSD dataset

```
python tools/run_on_OSD.py --use-cgnet --config-file configs/R50_rgbdconcat_mlc_occatmask_hom_concat.yaml
```


## License
The source code of this repository is released only for academic use. See the [license](./LICENSE.md) file for details.


## Acknowledgement
The codes of this repository are built upon open sources. We like to thank the authors for their work and sharing their code.

This repository contain source codes that have been heavily referenced from the original ["Unseen Object Amodal Instance Segmentation via Hierarchical Occlusion Modeling" (ICRA 2022) paper](https://github.com/gist-ailab/uoais).

