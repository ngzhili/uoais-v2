# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import cv2
import imageio
import random
import numpy as np
# constants
from detectron2.engine import DefaultPredictor

from adet.config import get_cfg
from adet.utils.visualizer import visualize_pred_amoda_occ
from adet.utils.post_process import detector_postprocess, DefaultPredictor

from utils import *
from foreground_segmentation.model import Context_Guided_Network

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument(
        "--config-file",
        default="configs/R50_rgbdconcat_mlc_occatmask_hom_concat.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
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
        "--dataset-path",
        type=str,
        default="./sample_data",
        help="path to the OSD dataset"
    )
    parser.add_argument(
        "--output-path",
        default="./testing",
        help="path to output folder",
    )
    return parser


if __name__ == "__main__":

    # UOAIS-Net
    args = get_parser().parse_args()
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.defrost()
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = args.confidence_threshold
    predictor = DefaultPredictor(cfg)
    W, H = cfg.INPUT.IMG_SIZE

    # CG-Net (foreground segmentation)
    if args.use_cgnet:
        print("Use foreground segmentation model (CG-Net) to filter out background instances")
        checkpoint = torch.load(os.path.join(args.cgnet_weight_path))
        fg_model = Context_Guided_Network(classes=2, in_channel=4)
        fg_model.load_state_dict(checkpoint['model'])
        fg_model.cuda()
        fg_model.eval()

    # Load output path
    output_path = os.path.join(args.output_path)
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    # Load OSD dataset
    rgb_paths = sorted(glob.glob("{}/image_color/*.png".format(args.dataset_path)))
    depth_paths = sorted(glob.glob("{}/disparity/*.png".format(args.dataset_path)))

    # filter test images by index
    indexes = [0]
    rgb_paths = [rgb_paths[i] for i in indexes]
    depth_paths = [depth_paths[i] for i in indexes]

    for rgb_path, depth_path in zip(rgb_paths, depth_paths):

        rgb_img = cv2.imread(rgb_path)
        rgb_img = cv2.resize(rgb_img, (W, H))
        depth_img = imageio.v2.imread(depth_path)
        depth_img = normalize_depth(depth_img)
        depth_img = cv2.resize(depth_img, (W, H), interpolation=cv2.INTER_NEAREST)
        depth_img = inpaint_depth(depth_img)

        # UOAIS-Net inference
        if cfg.INPUT.DEPTH and cfg.INPUT.DEPTH_ONLY:
            uoais_input = depth_img
        elif cfg.INPUT.DEPTH and not cfg.INPUT.DEPTH_ONLY: 
            uoais_input = np.concatenate([rgb_img, depth_img], -1)        
        outputs = predictor(uoais_input)
        instances = detector_postprocess(outputs['instances'], H, W).to('cpu')

        # CG-Net inference
        if args.use_cgnet:
            fg_rgb_input = standardize_image(cv2.resize(rgb_img, (320, 240)))
            fg_rgb_input = array_to_tensor(fg_rgb_input).unsqueeze(0)
            fg_depth_input = cv2.resize(depth_img, (320, 240)) 
            fg_depth_input = array_to_tensor(fg_depth_input[:,:,0:1]).unsqueeze(0) / 255
            fg_input = torch.cat([fg_rgb_input, fg_depth_input], 1)
            fg_output = fg_model(fg_input.cuda())
            fg_output = fg_output.cpu().data[0].numpy().transpose(1, 2, 0)
            fg_output = np.asarray(np.argmax(fg_output, axis=2), dtype=np.uint8)
            fg_output = cv2.resize(fg_output, (W, H), interpolation=cv2.INTER_NEAREST)

            # save foreground segmentation
            vis_fg = np.zeros_like(rgb_img) 
            vis_fg[:, :, 1] = fg_output*255
            vis_img = cv2.addWeighted(rgb_img, 0.8, vis_fg, 0.2, 0)
            cv2.imwrite(f"{output_path}/fg_seg_"+rgb_path.split("/")[-1], vis_img)

        preds = instances.pred_masks.detach().cpu().numpy() 
        pred_visibles = instances.pred_visible_masks.detach().cpu().numpy() 
        bboxes = instances.pred_boxes.tensor.detach().cpu().numpy() 
        pred_occs = instances.pred_occlusions.detach().cpu().numpy() 
        
        print("\nImage Filename:",rgb_path.split("/")[-1])
        
        print("\n\n======= BEFORE foreground segmentation post processing =======")
        print("\npreds:", preds)
        print("\npred_visibles:", pred_visibles)
        print("\nbboxes:", bboxes)
        print("\npred_occs:", pred_occs)

        print("\npreds:", preds.shape)
        print("\npred_visibles:", pred_visibles.shape)
        print("\nbboxes:", bboxes.shape)
        print("\npred_occs:", pred_occs.shape)
        if args.use_cgnet:
            # reorder predictions for visualization
            #idx_shuf = np.concatenate((np.where(pred_occs==1)[0] , np.where(pred_occs==0)[0] )) 
            #preds, pred_occs, bboxes = preds[idx_shuf], pred_occs[idx_shuf], bboxes[idx_shuf]
            vis_img_org = visualize_pred_amoda_occ(rgb_img, preds, bboxes, pred_occs)

        # filter out the background instances
        if args.use_cgnet:
            remove_idxs = [] # prediction indexes to be removed
            for i, pred_visible in enumerate(pred_visibles):
                iou = np.sum(np.bitwise_and(pred_visible, fg_output)) / np.sum(pred_visible)
                if iou < 0.5: # if iou overlap between predicted object and foreground segmentation < 0.5, remove predicted object
                    remove_idxs.append(i)
            preds = np.delete(preds, remove_idxs, 0)
            pred_visibles = np.delete(pred_visibles, remove_idxs, 0)
            bboxes = np.delete(bboxes, remove_idxs, 0)
            pred_occs = np.delete(pred_occs, remove_idxs, 0)
        

        # reorder predictions for visualization
        idx_shuf = np.concatenate((np.where(pred_occs==1)[0] , np.where(pred_occs==0)[0] )) 
        preds, pred_occs, bboxes = preds[idx_shuf], pred_occs[idx_shuf], bboxes[idx_shuf]
        vis_img = visualize_pred_amoda_occ(rgb_img, preds, bboxes, pred_occs)
        
        
        # visualize foreground segmentation model
        if args.use_cgnet:
            vis_fg = np.zeros_like(rgb_img) 
            vis_fg[:, :, 1] = fg_output*255
            vis_img = cv2.addWeighted(vis_img, 0.8, vis_fg, 0.2, 0)
            vis_img_rgb = cv2.addWeighted(rgb_img, 0.8, vis_fg, 0.2, 0)
            vis_img_all_org = cv2.addWeighted(vis_img_org, 0.8, vis_fg, 0.2, 0)

            vis_all_img_1 = np.hstack([rgb_img, depth_img ])
            vis_all_img_2 = np.hstack([vis_img_org, vis_img_rgb])
            vis_all_img_3 = np.hstack([vis_img_all_org, vis_img])
            vis_all_img = np.vstack([vis_all_img_1, vis_all_img_2,vis_all_img_3])

        else:
            vis_all_img = np.hstack([rgb_img, depth_img, vis_img])



        print("\n\n======= AFTER foreground segmentation post processing =======")
        print("\npreds:", preds)
        print("\npred_visibles:", pred_visibles)
        print("\nbboxes:", bboxes)
        print("\npred_occs:", pred_occs)

        print("\npreds:", preds.shape)
        print("\npred_visibles:", pred_visibles.shape)
        print("\nbboxes:", bboxes.shape)
        print("\npred_occs:", pred_occs.shape)

        cv2.imwrite(f"{output_path}/"+rgb_path.split("/")[-1], vis_all_img)
        #cv2.imshow(rgb_path.split("/")[-1] + "/ Press any key to view the next. / ESC: quit", vis_all_img)
        k = cv2.waitKey(0)
        if k == 27: # esc
            break  
        else:
            cv2.destroyAllWindows()

    
