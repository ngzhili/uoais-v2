import argparse
import numpy as np
import os

from eval.eval_utils import eval_visible_on_synthetic_data, eval_amodal_occ_on_synthetic_data


if __name__ == "__main__":

    parser = argparse.ArgumentParser('UOIS CenterMask', add_help=False)

    # model config   
    parser.add_argument("--config-file", 
        default="./configs/R50_rgbdconcat_mlc_occatmask_hom_concat.yaml", 
        metavar="FILE", help="path to config file")    
    parser.add_argument("--gpu", type=str, default="0", help="GPU id")
    parser.add_argument("--vis-only", action="store_true")
    parser.add_argument(
        "--use-cgnet",
        action="store_true",
        help="Use foreground segmentation model to filter our background instances or not"
    )
    parser.add_argument(
        "--cgnet-weight-path",
        type=str,
        default="./foreground_segmentation/rgbd_fg.pth",
        help="path to forground segmentation weight"
    )
    parser.add_argument(
        "--anno-path",
        type=str,
        default="./datasets/syntable/validation/uoais_val.json",
        help="path to the uoais_sim annotations"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="./datasets/syntable/validation",
        help="path to the dataset path"
    )
    
    parser.add_argument(
        "--model-weight-name",
        type=str,
        default="model_final.pth",
        help="path to the model"
    )

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    if args.vis_only:
        eval_visible_on_synthetic_data(args)
    else:
        eval_amodal_occ_on_synthetic_data(args)
        eval_visible_on_synthetic_data(args)
    