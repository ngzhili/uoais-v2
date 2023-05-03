#from AdelaiDet.adet.utils.visualizer import visualize_pred_amoda_occ
#sfrom utils import *
import argparse
import glob
import multiprocessing as mp
import os
import cv2
import imageio
import random
import numpy as np
from termcolor import colored

from utils import *
from adet.utils.visualizer import visualize_gt_amoda_occ

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="./datasets/OSD-0.2-depth",
        help="path to the OSD dataset"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="./datasets/visual/",
        help="path to the output visual"
    )
    return parser

if __name__ == "__main__":


    args = get_parser().parse_args()

    # Load output path
    output_path = os.path.join(args.output_path)
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    # Load OSD dataset
    rgb_paths = sorted(glob.glob("{}/image_color/*.png".format(args.dataset_path)))
    depth_paths = sorted(glob.glob("{}/disparity/*.png".format(args.dataset_path)))
    amodal_anno_paths = sorted(glob.glob("{}/amodal_annotation/*.png".format(args.dataset_path)))
    occlusion_anno_paths = sorted(glob.glob("{}/occlusion_annotation/*.png".format(args.dataset_path)))
    assert len(rgb_paths) == len(depth_paths)
    assert len(amodal_anno_paths) != 0
    assert len(occlusion_anno_paths) != 0
    print(colored("Evaluation on OSD dataset: {} rgbs, {} depths, {} amodal masks, {} occlusion masks".format(
            len(rgb_paths), len(depth_paths), len(amodal_anno_paths), len(occlusion_anno_paths)), "green"))

    # filter test images by index
    indexes = [0]
    rgb_paths = [rgb_paths[i] for i in indexes]
    depth_paths = [depth_paths[i] for i in indexes]
    amodal_anno_paths = [amodal_anno_paths[i] for i in indexes]
    occlusion_anno_paths = [occlusion_anno_paths[i] for i in indexes]
    W, H = (640, 480)

    for i, (rgb_path, depth_path) in enumerate(zip(rgb_paths, depth_paths)):
        
        rgb_img = cv2.imread(rgb_path)
        rgb_img = cv2.resize(rgb_img, (W, H))
        depth_img = imageio.v2.imread(depth_path)
        depth_img = normalize_depth(depth_img)
        depth_img = cv2.resize(depth_img, (W, H), interpolation=cv2.INTER_NEAREST)
        depth_img = inpaint_depth(depth_img)

        # laod GT (amodal masks)
        img_name = os.path.basename(rgb_path)[:-4]
        annos = [] # [instance, IMG_H, IMG_W]
        filtered_amodal_paths = list(filter(lambda p: img_name + "_" in p, amodal_anno_paths))
        filtered_occlusion_paths = list(filter(lambda p: img_name + "_" in p, occlusion_anno_paths))

        img_h, img_w, img_c = rgb_img.shape


        for anno_path in filtered_amodal_paths:
            # get instance id  
            inst_id = os.path.basename(anno_path)[:-4].split("_")[-1]
            inst_id = int(inst_id)
            # load mask image
            anno = imageio.imread(anno_path)
            anno = cv2.resize(anno, (W, H), interpolation=cv2.INTER_NEAREST)
            # fill mask with instance id
            cnd = anno > 0
            anno_mask = np.zeros((H, W))
            anno_mask[cnd] = inst_id
            annos.append(anno_mask)            
        #annos = np.stack(annos)
        #num_inst_all_gt += len(filtered_amodal_paths)

        print(rgb_img.shape)
        print(depth_img.shape)
        #print(annos.shape)
        print(annos)

        visualize_gt_amoda_occ(color, masks, pred_occ)

        #vis_img = visualize_pred_amoda_occ(rgb_img, preds, bboxes, pred_occs)
        #vis_all_img = np.hstack([rgb_img, depth_img, anno])

        #cv2.imwrite(f"{output_path}/"+rgb_path.split("/")[-1], vis_all_img)