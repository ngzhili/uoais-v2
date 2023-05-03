import os
import cv2
import glob
import numpy as np
import imageio
import torch

from tqdm import tqdm

from detectron2.engine import DefaultPredictor
from adet.config import get_cfg
from adet.utils.post_process import detector_postprocess, DefaultPredictor

import compute_PRF 
from utils import normalize_depth, inpaint_depth, standardize_image, array_to_tensor
from foreground_segmentation.model import Context_Guided_Network
from termcolor import colored

BACKGROUND_LABEL = 0
BG_LABELS = {}
BG_LABELS["floor"] = [0, 1]
BG_LABELS["table"] = [0, 1, 2]

def compute_occluded_masks(mask1, mask2):
    """Computes occlusions between two sets of masks.
    masks1, masks2: [Height, Width, instances]
    """

    # intersections and union
    mask1_area = np.count_nonzero(mask1)
    mask2_area = np.count_nonzero(mask2)
    intersection_mask = np.logical_and(mask1, mask2)
    intersection = np.count_nonzero(np.logical_and(mask1, mask2))

    iou = intersection/(mask1_area+mask2_area-intersection)

    return iou, intersection_mask.astype(float)

# A(i,j), col j, row i. row i --> col j
def generate_ooam(inst_vis_mask_list,occlusion_mask_list):
    rows = cols = len(inst_vis_mask_list)
    ooam = np.zeros((rows,cols))
    for i in range(0,len(inst_vis_mask_list)):
        visible_mask_i = inst_vis_mask_list[i] # occluder
        for j in range(0,len(occlusion_mask_list)):
            if j != i:
                occluded_mask_j = occlusion_mask_list[j] # occludee
                iou, _ = compute_occluded_masks(visible_mask_i,occluded_mask_j) 
                
                if iou > 0.1: # object i's visible mask is overlapping object j's occluded mask
                    # print(iou)
                    # print(np.count_nonzero(_))
                    ooam[i][j] = 1
    return ooam
    
def eval_visible_on_OSD(args):

    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.defrost()
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, args.model_weight_name)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
    cfg.freeze()
    predictor = DefaultPredictor(cfg)
    W, H = cfg.INPUT.IMG_SIZE
    print("Using model weights from:", args.model_weight_name)
    # foreground segmentation
    if args.use_cgnet:
        print("Use foreground segmentation model (CG-Net) to filter out background instances")
        checkpoint = torch.load(os.path.join(args.cgnet_weight_path))
        fg_model = Context_Guided_Network(classes=2, in_channel=4)
        fg_model.load_state_dict(checkpoint['model'])
        fg_model.cuda()
        fg_model.eval()

    # load dataset
    rgb_paths = sorted(glob.glob("{}/image_color/*.png".format(args.dataset_path)))
    depth_paths = sorted(glob.glob("{}/disparity/*.png".format(args.dataset_path)))
    anno_paths = sorted(glob.glob("{}/annotation/*.png".format(args.dataset_path)))
    assert len(rgb_paths) == len(depth_paths)
    assert len(rgb_paths) == len(anno_paths)
    print(colored("Evaluation on OSD dataset: {} rgbs, {} depths, {} visible masks".format(
                len(rgb_paths), len(depth_paths), len(anno_paths)), "green"))
    
    metrics_all = []
    iou_masks = 0
    num_inst_all = 0 # number of all instances
    num_inst_mat = 0 # number of matched instance

    for i, (rgb_path, depth_path, anno_path) in enumerate(zip(tqdm(rgb_paths), depth_paths, anno_paths)):

        # load rgb and depth
        rgb_img = cv2.imread(rgb_path)
        rgb_img = cv2.resize(rgb_img, (W, H))
        depth_img = imageio.imread(depth_path)
        depth_img = normalize_depth(depth_img)
        depth_img = cv2.resize(depth_img, (W, H), interpolation=cv2.INTER_NEAREST)
        depth_img = inpaint_depth(depth_img)
        
        # UOAIS-Net inference
        if cfg.INPUT.DEPTH and cfg.INPUT.DEPTH_ONLY:
            uoais_input = depth_img
        elif cfg.INPUT.DEPTH and not cfg.INPUT.DEPTH_ONLY: 
            uoais_input = np.concatenate([rgb_img, depth_img], -1)        
        else:
            uoais_input = rgb_img
        # laod GT (annotation) anno: [H, W]
        anno = imageio.imread(anno_path)
        anno = cv2.resize(anno, (W, H), interpolation=cv2.INTER_NEAREST)
        labels_anno = np.unique(anno)
        labels_anno = labels_anno[~np.isin(labels_anno, [BACKGROUND_LABEL])]
        num_inst_all += len(labels_anno)

        # forward (UOAIS)
        outputs = predictor(uoais_input)
        instances = detector_postprocess(outputs['instances'], H, W).to('cpu')

        if cfg.INPUT.AMODAL:
            pred_masks = instances.pred_visible_masks.detach().cpu().numpy()
        else:
            pred_masks = instances.pred_masks.detach().cpu().numpy()
            
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
        
            pred_all = np.zeros_like(anno)
            pred = np.zeros_like(anno)
            for i, mask in enumerate(pred_masks):
                iou = np.sum(np.bitwise_and(mask, fg_output)) / np.sum(mask)
                if iou >= 0.5:
                    pred[mask > False] = i+1
                pred_all[mask > False] = i+1
        else: 
            pred = np.zeros_like(anno)
            for i, mask in enumerate(pred_masks):
                pred[mask > False] = i+1
        
        # evaluate
        metrics, assignments = compute_PRF.multilabel_metrics(pred, anno, return_assign=True)
        metrics_all.append(metrics)

        # compute IoU for all instances
        # print(assignments)
        num_inst_mat += len(assignments)
        assign_visible_pred, assign_visible_gt = 0, 0
        assign_visible_overlap = 0
        for gt_id, pred_id in assignments:
            # count area of visible mask (pred & gt)
            mask_pr = pred == pred_id
            mask_gt = anno == gt_id           
            assign_visible_pred += np.count_nonzero(mask_pr)
            assign_visible_gt += np.count_nonzero(mask_gt)
            # count area of overlap btw. pred & gt
            mask_overlap = np.logical_and(mask_pr, mask_gt)
            assign_visible_overlap += np.count_nonzero(mask_overlap)
        if assign_visible_pred+assign_visible_gt-assign_visible_overlap > 0:
            iou = assign_visible_overlap / (assign_visible_pred+assign_visible_gt-assign_visible_overlap)
        else: iou = 0
        iou_masks += iou
    # compute mIoU for all instances
    miou = iou_masks / len(metrics_all)
    
    # sum the values with same keys
    result = {}
    num = len(metrics_all)
    for metrics in metrics_all:
        for k in metrics.keys():
            result[k] = result.get(k, 0) + metrics[k]
    for k in sorted(result.keys()):
        result[k] /= num

    print('\n')
    print(colored("Visible Metrics for OSD", "green", attrs=["bold"]))
    print(colored("---------------------------------------------", "green"))
    print("    Overlap    |    Boundary")
    print("  P    R    F  |   P    R    F  |  %75 | mIoU")
    print("{:.1f} {:.1f} {:.1f} | {:.1f} {:.1f} {:.1f} | {:.1f} | {:.4f}".format(
        result['Objects Precision']*100, result['Objects Recall']*100, 
        result['Objects F-measure']*100,
        result['Boundary Precision']*100, result['Boundary Recall']*100, 
        result['Boundary F-measure']*100,
        result['obj_detected_075_percentage']*100, miou
    ))
    print(colored("---------------------------------------------", "green"))
    for k in sorted(result.keys()):
        print('%s: %f' % (k, result[k]))
    print('\n')
    import pandas as pd
    if args.use_cgnet:
        csvName = "cg_net_osd_results.csv"
    else:
        csvName = "osd_results.csv"
    result_csv_path = os.path.join(cfg.OUTPUT_DIR,csvName)
    if os.path.isfile(result_csv_path):
        df = pd.read_csv(result_csv_path)

        dict_visible = {'Visible Overlap P': result['Objects Precision']*100, 
                'Visible Overlap R': result['Objects Recall']*100,
                'Visible Overlap F': result['Objects F-measure']*100,
                'Visible Boundary P': result['Boundary Precision']*100,
                'Visible Boundary R': result['Boundary Recall']*100, 
                'Visible Boundary F': result['Boundary F-measure']*100,
                'Visible %75': result['obj_detected_075_percentage']*100,
                'Visible mIoU': miou
                }
        df_vis = pd.DataFrame([dict_visible])
        df = pd.merge(df_vis, df,left_index=True, right_index=True)
        df = df.drop("Unnamed: 0", axis=1)
        df.to_csv(result_csv_path,index=False)
    else:
        print(f"{result_csv_path} does not exists")

def eval_amodal_occ_on_OSD(args):

    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.defrost()
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, args.model_weight_name)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
    cfg.freeze()
    predictor = DefaultPredictor(cfg)
    W, H = cfg.INPUT.IMG_SIZE
    print("Using model weights from:", args.model_weight_name)
    # foreground segmentation
    if args.use_cgnet:
        print("Use foreground segmentation model (CG-Net) to filter out background instances")
        checkpoint = torch.load(os.path.join(args.cgnet_weight_path))
        fg_model = Context_Guided_Network(classes=2, in_channel=4)
        fg_model.load_state_dict(checkpoint['model'])
        fg_model.cuda()
        fg_model.eval()

    # load dataset
    rgb_paths = sorted(glob.glob("{}/image_color/*.png".format(args.dataset_path)))
    depth_paths = sorted(glob.glob("{}/disparity/*.png".format(args.dataset_path)))
    vis_anno_paths = sorted(glob.glob("{}/annotation/*.png".format(args.dataset_path)))

    amodal_anno_paths = sorted(glob.glob("{}/amodal_annotation/*.png".format(args.dataset_path)))
    occlusion_anno_paths = sorted(glob.glob("{}/occlusion_annotation/*.png".format(args.dataset_path)))
    # rgb_paths = rgb_paths[:40]
    # depth_paths = depth_paths[:40]
    # vis_anno_paths = vis_anno_paths[:40]

    assert len(rgb_paths) == len(depth_paths)
    assert len(rgb_paths) == len(vis_anno_paths)
    assert len(amodal_anno_paths) != 0
    assert len(occlusion_anno_paths) != 0
    print(colored("Evaluation on OSD dataset: {} rgbs, {} depths, {} amodal masks, {} occlusion masks".format(
                len(rgb_paths), len(depth_paths), len(amodal_anno_paths), len(occlusion_anno_paths)), "green"))

    metrics_all = [] # amodal mask evaluation
    num_inst_all_pred = 0 # number of all pred instances
    num_inst_all_gt = 0 # number of all GT instances
    num_inst_occ_pred = 0 # number of occluded prediction
    num_inst_occ_mat = 0 # number of occluded and matched
    num_inst_mat = 0 # number of matched instance
    
    mask_ious, occ_ious = 0, 0
    pre_occ, rec_occ, f_occ = 0, 0, 0
    pre_bou, rec_bou, f_bou = 0, 0, 0
    num_correct = 0 # for occlusion classification
    num_occ_over75 = 0 # number of instance IoU>0.75
    occ_over_75_rate = 0 # rate of instance IoU>0.75

    num_correct_occ_ooam = 0 # total correct occlusion prediction
    num_all_occ_ooam = 0 # total occlusion combinations

    # for i, (rgb_path, depth_path) in enumerate(zip(tqdm(rgb_paths), depth_paths)):
    
    for i, (rgb_path, depth_path, anno_path) in enumerate(zip(tqdm(rgb_paths), depth_paths, vis_anno_paths)):
        # print(rgb_path)
        # load rgb and depth
        rgb_img = cv2.imread(rgb_path)
        rgb_img = cv2.resize(rgb_img, (W, H))
        depth_img = imageio.imread(depth_path)
        depth_img = normalize_depth(depth_img)
        depth_img = cv2.resize(depth_img, (W, H), interpolation=cv2.INTER_NEAREST)
        depth_img = inpaint_depth(depth_img)
        
        # UOAIS-Net inference
        if cfg.INPUT.DEPTH and cfg.INPUT.DEPTH_ONLY:
            uoais_input = depth_img
        elif cfg.INPUT.DEPTH and not cfg.INPUT.DEPTH_ONLY: 
            uoais_input = np.concatenate([rgb_img, depth_img], -1)   
        else:
            uoais_input = rgb_img
        
        # load GT (visible masks)
        vis_anno = imageio.imread(anno_path)
        vis_anno = cv2.resize(vis_anno, (W, H), interpolation=cv2.INTER_NEAREST)
        labels_anno = np.unique(vis_anno)
        labels_anno = labels_anno[~np.isin(labels_anno, [BACKGROUND_LABEL])]
        labels_anno = sorted(labels_anno)
        # print(labels_anno)

        inst_vis_mask_list = []
        for label in labels_anno:
            vis_mask = vis_anno.copy()
            vis_mask[np.where(vis_mask == label)] = 1
            vis_mask =vis_mask.astype(np.uint8)
            inst_vis_mask_list.append(vis_mask)

        # load GT (amodal masks)
        img_name = os.path.basename(rgb_path)[:-4]
        annos = [] # [instance, IMG_H, IMG_W]
        filtered_amodal_paths = list(filter(lambda p: img_name + "_" in p, amodal_anno_paths))
        filtered_occlusion_paths = list(filter(lambda p: img_name + "_" in p, occlusion_anno_paths))
        # print(filtered_occlusion_paths)
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
        annos = np.stack(annos)
        num_inst_all_gt += len(filtered_amodal_paths)
        
        # forward (UOAIS)
        outputs = predictor(uoais_input)
        instances = detector_postprocess(outputs['instances'], H, W).to('cpu')
        # print(instances)
        if not args.use_cgnet:
            pred_masks = instances.pred_masks.detach().cpu().numpy()
            preds = [] # mask per each instance
            for i, mask in enumerate(pred_masks):
                pred = np.zeros((H, W))
                pred[mask > False] = i+1
                preds.append(pred)
                num_inst_all_pred += 1
        else:
            fg_rgb_input = standardize_image(cv2.resize(rgb_img, (320, 240)))
            fg_rgb_input = array_to_tensor(fg_rgb_input).unsqueeze(0)
            fg_depth_input = cv2.resize(depth_img, (320, 240)) 
            fg_depth_input = array_to_tensor(fg_depth_input[:,:,0:1]).unsqueeze(0) / 255
            fg_input = torch.cat([fg_rgb_input, fg_depth_input], 1)
            fg_output = fg_model(fg_input.cuda())
            fg_output = fg_output.cpu().data[0].numpy().transpose(1, 2, 0)
            fg_output = np.asarray(np.argmax(fg_output, axis=2), dtype=np.uint8)
            fg_output = cv2.resize(fg_output, (W, H), interpolation=cv2.INTER_NEAREST)

            # filter amodal predictions with foreground mask
            pred_masks = instances.pred_masks.detach().cpu().numpy()
            preds = [] # mask per each instance
            for i, mask in enumerate(pred_masks):
                overlap = np.sum(np.bitwise_and(mask, fg_output)) / np.sum(mask)
                if overlap >= 0.5: # filiter outliers
                    pred = np.zeros((H, W))
                    pred[mask > False] = i+1
                    preds.append(pred)
                    num_inst_all_pred += 1

        if len(preds) > 0:
            preds = np.stack(preds)
        else:
            preds = np.array(preds)
        # evaluate amodal masks
        metrics, assignments = compute_PRF.multilabel_amodal_metrics(preds, annos, return_assign=True)
        metrics_all.append(metrics)

        # count the number of instances
        num_inst_mat += len(assignments)

        if not cfg.INPUT.AMODAL:
            raise ValueError("Cannot support amodal segmentation")

        amodals = instances.pred_masks.detach().cpu().numpy()
        visibles = instances.pred_visible_masks.detach().cpu().numpy()        
        
        # count occluded area of predictions when classified
        all_occ_pred, all_bou_pred = 0, 0
        num_inst_occ_prd_img = 0
        for pred in preds:
            idx = int(pred.max())-1
            amodal = amodals[idx]
            visible = visibles[idx]
            occ = np.bitwise_xor(amodal, visible)
            try:
                cls = instances.pred_occlusions[idx].item()
            except:
                # if area over 5% of amodal mask is not visible
                cls = 1 if np.int64(np.count_nonzero(occ)) / np.int64(np.count_nonzero(amodal)) >= 0.05 else 0                
            if not cls: continue
            num_inst_occ_pred += 1
            num_inst_occ_prd_img += 1
            all_occ_pred += np.int64(np.count_nonzero(occ))
            all_bou_pred += np.sum(compute_PRF.seg2bmap(occ))
        
        occlusion_mask_list = []
        # count occluded area of ground truth
        all_occ_gt, all_bou_gt = 0, 0
        occ_paths = filtered_occlusion_paths
        for occ_path in occ_paths:
            occ = imageio.imread(occ_path)
            occ = cv2.resize(occ, (W, H), interpolation=cv2.INTER_NEAREST)
            occ = occ[:,:] > 0
            all_occ_gt += np.int64(np.count_nonzero(occ))
            all_bou_gt += np.sum(compute_PRF.seg2bmap(occ))
            
            idx = int(os.path.basename(occ_path)[:-4].split("_")[1]) - 1
            # print("idx:",idx)
            # print(np.unique(occ))
            occlusion_mask_list.append(occ)
        # generate occlusion ordering (gt_ooam)
        # print(inst_vis_mask_list)
        # print(occlusion_mask_list)
        
        gt_ooam = generate_ooam(inst_vis_mask_list,occlusion_mask_list)
        # print("gt_ooam:")
        # print(gt_ooam)

        # count area with matched instances
        # assign: [[gt_id, pred_id], ... ]
        assign_amodal_pred, assign_visible_pred, assign_amodal_gt = 0, 0, 0
        assign_occ_pred, assign_occ_gt = 0, 0
        assign_amodal_overlap, assign_occ_overlap = 0, 0
        occ_bou_pre, occ_bou_rec = 0, 0
        num_occ_over75_img = 0

        pred_occ_mask_list_matched = []
        pred_vis_mask_list_matched = []
        # print("assignments:\n",assignments)
        for gt_id, pred_id in assignments:              
            ###############
            # AMODAL MASK #
            ###############
            # count area of masks predictions
            amodal = amodals[int(pred_id)-1] # predicted amodal mask
            visible = visibles[int(pred_id)-1] # predicted visible mask
            assign_amodal_pred += np.count_nonzero(amodal)
            assign_visible_pred += np.count_nonzero(visible)
            # count area of mask GT [ annos: amodal GT ]
            anno = annos[np.where(annos == gt_id)[0][0]] > 0
            assign_amodal_gt += np.count_nonzero(anno)
            # count overlap area of mask btw. pred & GT
            amodal_overlap = np.logical_and(amodal, anno)
            assign_amodal_overlap += np.count_nonzero(amodal_overlap)
                        
            ##################
            # OCCLUSION MASK #
            ##################
            # count area of occlusion prediction
            occ_pred = np.bitwise_xor(amodal, visible) # predicted occlusion mask
            assign_occ_pred += np.count_nonzero(occ_pred)

            pred_occ_mask_list_matched.append(occ_pred)
            pred_vis_mask_list_matched.append(visible)
            ############################
            # OCCLUSION CLASSIFICATION #
            ############################
            # count occlusion classification corrects
            cls_gt = os.path.isfile("{}/occlusion_annotation/{}_{}.png".format(
                                    args.dataset_path, img_name, int(gt_id)))
            
            try:
                cls_pred = instances.pred_occlusions[int(pred_id)-1].item()
            except: # if occ prediction count / amodal prediction count > 0.05, occlusion class = 1
                cls_pred = 1 if np.int64(np.count_nonzero(occ_pred)) / np.int64(np.count_nonzero(amodal)) >= 0.05 else 0
            num_correct += cls_pred==cls_gt
            if cls_pred==cls_gt and cls_pred == 1:
                num_inst_occ_mat += cls_pred         

            ##################
            # OCCLUSION MASK #
            ##################
            # count area of occlusion GT
            occ_path = "{}/occlusion_annotation/{}_{}.png".format(
                                    args.dataset_path, img_name, int(gt_id))
            if not os.path.isfile(occ_path) or not cls_pred: continue
            occ_gt = imageio.imread(occ_path)
            occ_gt = cv2.resize(occ_gt, (W, H), interpolation=cv2.INTER_NEAREST)
            occ_gt = occ_gt[:,:] > 0
            assign_occ_gt += np.count_nonzero(occ_gt)
            # count overlap area of occlusion btw. pred & GT
            occ_overlap = np.logical_and(occ_pred, occ_gt)
            assign_occ_overlap += np.count_nonzero(occ_overlap)
            
            ############################
            # Over75 of OCCLUSION MASK #
            ############################
            pre = np.int64(np.count_nonzero(occ_overlap)) / np.count_nonzero(occ_pred)
            rec = np.int64(np.count_nonzero(occ_overlap)) / np.count_nonzero(occ_gt)
            f = (2 * pre * rec) / (pre + rec) if (pre + rec) > 0 else 0
            if f > 0.75: 
                num_occ_over75_img += 1
                num_occ_over75 += 1
                        
            ###########################
            # OCCLUSION MASK BOUNDARY #
            ###########################
            pre, rec = compute_PRF.boundary_overlap_occ(gt_id, pred_id, occ_pred, occ_gt)
            occ_bou_pre += pre
            occ_bou_rec += rec

        pred_ooam = generate_ooam(pred_vis_mask_list_matched, pred_occ_mask_list_matched)
        # print("pred_ooam:")
        # print(pred_ooam)
        
        # ooam_score = np.sum(pred_ooam == gt_ooam) / gt_ooam.size # individual acc score
        ooam_diagonal_size = len(gt_ooam)
        num_correct_occ_ooam += np.sum(pred_ooam == gt_ooam) - ooam_diagonal_size
        num_all_occ_ooam += gt_ooam.size - ooam_diagonal_size

        #####################################
        # COMPUTE METRICS in a single image #
        #####################################
        # mIoU of amodal mask (only matched instance)
        if assign_amodal_pred+assign_amodal_gt-assign_amodal_overlap > 0:
            iou = assign_amodal_overlap / (assign_amodal_pred+assign_amodal_gt-assign_amodal_overlap)
        else: iou = 0
        mask_ious += iou
        # mIoU of occlusion mask (only matched instance)
        if assign_occ_pred+assign_occ_gt-assign_occ_overlap > 0:
            iou = assign_occ_overlap / (assign_occ_pred+assign_occ_gt-assign_occ_overlap)
        else: iou = 0
        occ_ious +=iou
        
        # number of occluded instances in one image
        num_pred = num_inst_occ_prd_img
        num_gt = len(filtered_occlusion_paths)
        if num_pred == 0 and num_gt > 0:
            pre_occ += 1
            pre_bou += 1
        elif num_pred > 0 and num_gt == 0:
            rec_occ += 1
            rec_bou += 1
        elif num_pred == 0 and num_gt == 0:
            pre_occ += 1
            rec_occ += 1
            f_occ += 1
            pre_bou += 1
            rec_bou += 1
            f_bou += 1
            occ_over_75_rate += 1
        else:
            assert (num_pred > 0) and (num_gt > 0)
            # P, R, F of occlusion mask (all instance)
            pre = assign_occ_overlap / all_occ_pred if all_occ_pred > 0 else 0
            rec = assign_occ_overlap / all_occ_gt if all_occ_gt > 0 else 0
            f = (2 * pre * rec) / (pre + rec) if (pre + rec) > 0 else 0
            pre_occ += pre
            rec_occ += rec
            f_occ += f
            # P, R, F of occlusion boundary (all instance)
            pre = occ_bou_pre / all_bou_pred if all_bou_pred > 0 else 0
            rec = occ_bou_rec / all_bou_gt if all_bou_gt > 0 else 0
            f = (2 * pre * rec) / (pre + rec) if (pre + rec) > 0 else 0
            pre_bou += pre
            rec_bou += rec
            f_bou += f
            occ_over_75_rate += num_occ_over75_img / num_gt
        
    ###############################################
    # get average metirc values among test images #
    ###############################################
    num = len(metrics_all)
    mask_ious /= num 
    occ_ious /= num 
    pre_occ /= num 
    rec_occ /= num 
    f_occ /= num 
    pre_bou /= num 
    rec_bou /= num 
    f_bou /= num 
    occ_over_75_rate /= num
    
    occ_cls_acc = num_correct / num_inst_mat * 100 if num_inst_mat > 0 else 0
    occ_cls_pre = num_correct / num_inst_all_pred * 100 if num_inst_all_pred > 0 else 0
    occ_cls_rec = num_correct / num_inst_all_gt * 100 if num_inst_all_gt > 0 else 0
    occ_cls_f = (2*occ_cls_pre*occ_cls_rec) / (occ_cls_pre+occ_cls_rec) if occ_cls_pre+occ_cls_rec > 0 else 0

    occ_ooam_acc = num_correct_occ_ooam / num_all_occ_ooam * 100 if num_all_occ_ooam > 0 else 0

    # sum the values with same keys
    result = {}
    for metrics in metrics_all:
        for k in metrics.keys():
            result[k] = result.get(k, 0) + metrics[k]
    for k in sorted(result.keys()):
        result[k] /= num

    print('\n')
    print(colored("Amodal Metrics on OSD (for {} instances)".format(num_inst_all_gt), "green", attrs=["bold"]))
    print(colored("---------------------------------------------", "green"))
    print("    Overlap    |    Boundary")
    print("  P    R    F  |   P    R    F  |  %75 | mIoU")
    print("{:.1f} {:.1f} {:.1f} | {:.1f} {:.1f} {:.1f} | {:.1f} | {:.4f}".format(
        result['Objects Precision']*100, result['Objects Recall']*100, 
        result['Objects F-measure']*100,
        result['Boundary Precision']*100, result['Boundary Recall']*100, 
        result['Boundary F-measure']*100,
        result['obj_detected_075_percentage']*100, mask_ious
    ))
    print(colored("---------------------------------------------", "green"))
    for k in sorted(result.keys()):
        print('%s: %f' % (k, result[k]))
    print('\n')
    print(colored("Occlusion Metrics on OSD", "green", attrs=["bold"]))
    print(colored("---------------------------------------------", "green"))
    print("    Overlap    |    Boundary")
    print("  P    R    F  |   P    R    F  |  %75 | mIoU")
    print("{:.1f} {:.1f} {:.1f} | {:.1f} {:.1f} {:.1f} | {:.1f} | {:.4f}".format(
        pre_occ*100, rec_occ*100, f_occ*100, 
        pre_bou*100, rec_bou*100, f_bou*100,
        occ_over_75_rate*100, occ_ious        
    ))
    print(colored("---------------------------------------------", "green"))
    print('\n')
    print(colored("Occlusion Classification on OSD", "green", attrs=["bold"]))
    print(colored("---------------------------------------------", "green"))
    print("  P   R   F   ACC")
    print("{:.1f} {:.1f} {:.1f} {:.1f}".format(
        occ_cls_pre, occ_cls_rec, occ_cls_f, occ_cls_acc        
    ))
    print(colored("---------------------------------------------", "green"))
    print('\n')
    print(colored("Occlusion Order Adjacency Matrix (OOAM) on OSD", "green", attrs=["bold"]))
    print(colored("---------------------------------------------", "green"))
    print("ACC")
    print("{:.1f}".format(
        occ_ooam_acc        
    ))
    print(colored("---------------------------------------------", "green"))
    print('\n')
    
    import pandas as pd
    dict_amodal = {'Amodal Overlap P': result['Objects Precision']*100, 
            'Amodal Overlap R': result['Objects Recall']*100,
            'Amodal Overlap F': result['Objects F-measure']*100,
            'Amodal Boundary P': result['Boundary Precision']*100,
            'Amodal Boundary R': result['Boundary Recall']*100, 
            'Amodal Boundary F': result['Boundary F-measure']*100,
            'Amodal %75': result['obj_detected_075_percentage']*100,
            'Amodal mIoU': mask_ious
            }
    # df = pd.DataFrame([dict_])
    # df.to_csv(os.path.join(cfg.OUTPUT_DIR,"osd_amodal_results.csv"),index=False)

    dict_occ = {'Occlusion Overlap P': pre_occ*100, 
            'Occlusion Overlap R': rec_occ*100,
            'Occlusion Overlap F': f_occ*100,
            'Occlusion Boundary P': pre_bou*100,
            'Occlusion Boundary R': rec_bou*100, 
            'Occlusion Boundary F': f_bou*100,
            'Occlusion %75': occ_over_75_rate*100,
            'Occlusion mIoU': occ_ious   
            }
    # df = pd.DataFrame([dict_])
    # df.to_csv(os.path.join(cfg.OUTPUT_DIR,"osd_occlusion_results.csv"),index=False)
     
    dict_occ_class = {'Occlusion Classification P': occ_cls_pre, 
            'Occlusion Classification R': occ_cls_rec,
            'Occlusion Classification F': occ_cls_f,
            'Occlusion Classification Accuracy': occ_cls_acc
            }
    dict_amodal.update(dict_occ)
    dict_amodal.update(dict_occ_class)
    df = pd.DataFrame([dict_amodal])
    if args.use_cgnet:
        csvName = "cg_net_osd_results.csv"
    else:
        csvName = "osd_results.csv"
    df.to_csv(os.path.join(cfg.OUTPUT_DIR,csvName))

    return

def eval_visible_on_OCID(args):


    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.defrost()
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, args.model_weight_name)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
    cfg.freeze()
    predictor = DefaultPredictor(cfg)
    W, H = cfg.INPUT.IMG_SIZE

    # foreground segmentation
    if args.use_cgnet:
        print("Use foreground segmentation model (CG-Net) to filter out background instances")
        checkpoint = torch.load(os.path.join(args.cgnet_weight_path))
        fg_model = Context_Guided_Network(classes=2, in_channel=4)
        fg_model.load_state_dict(checkpoint['model'])
        fg_model.cuda()
        fg_model.eval()
    
    # load dataset
    image_paths = []
    depth_paths = []
    anno_paths = []
    # load ARID20
    print("... load dataset [ ARID20 ]")
    data_root = args.dataset_path + "/ARID20"
    f_or_t = ["floor", "table"]
    b_or_t = ["bottom", "top"]
    for dir_1 in f_or_t:
        for dir_2 in b_or_t:
            seq_list = sorted(os.listdir(os.path.join(data_root, dir_1, dir_2)))
            for seq in seq_list:
                data_dir = os.path.join(data_root, dir_1, dir_2, seq)
                if not os.path.isdir(data_dir): continue
                data_list = sorted(os.listdir(os.path.join(data_dir, "rgb")))
                for data_name in data_list:
                    image_path = os.path.join(data_root, dir_1, dir_2, seq, "rgb", data_name)
                    image_paths.append(image_path)
                    depth_path = os.path.join(data_root, dir_1, dir_2, seq, "depth", data_name)
                    depth_paths.append(depth_path)
                    anno_path = os.path.join(data_root, dir_1, dir_2, seq, "label", data_name)
                    anno_paths.append(anno_path)
    # load YCB10
    print("... load dataset [ YCB10 ]")
    data_root = args.dataset_path +  "/YCB10"
    f_or_t = ["floor", "table"]
    b_or_t = ["bottom", "top"]
    c_c_m = ["cuboid", "curved", "mixed"]
    for dir_1 in f_or_t:
        for dir_2 in b_or_t:
            for dir_3 in c_c_m:
                seq_list = os.listdir(os.path.join(data_root, dir_1, dir_2, dir_3))
                for seq in seq_list:
                    data_dir = os.path.join(data_root, dir_1, dir_2, dir_3, seq)
                    if not os.path.isdir(data_dir): continue
                    data_list = sorted(os.listdir(os.path.join(data_dir, "rgb")))
                    for data_name in data_list:
                        image_path = os.path.join(data_root, dir_1, dir_2, dir_3, seq, "rgb", data_name)
                        image_paths.append(image_path)
                        depth_path = os.path.join(data_root, dir_1, dir_2, dir_3, seq, "depth", data_name)
                        depth_paths.append(depth_path)
                        anno_path = os.path.join(data_root, dir_1, dir_2, dir_3, seq, "label", data_name)
                        anno_paths.append(anno_path)
    # load ARID10
    print("... load dataset [ ARID10 ]")
    data_root =  args.dataset_path + "/ARID10"
    f_or_t = ["floor", "table"]
    b_or_t = ["bottom", "top"]
    c_c_m = ["box", "curved", "fruits", "mixed", "non-fruits"]
    for dir_1 in f_or_t:
        for dir_2 in b_or_t:
            for dir_3 in c_c_m:
                seq_list = os.listdir(os.path.join(data_root, dir_1, dir_2, dir_3))
                for seq in seq_list:
                    data_dir = os.path.join(data_root, dir_1, dir_2, dir_3, seq)
                    if not os.path.isdir(data_dir): continue
                    data_list = sorted(os.listdir(os.path.join(data_dir, "rgb")))
                    for data_name in data_list:
                        image_path = os.path.join(data_root, dir_1, dir_2, dir_3, seq, "rgb", data_name)
                        image_paths.append(image_path)
                        depth_path = os.path.join(data_root, dir_1, dir_2, dir_3, seq, "depth", data_name)
                        depth_paths.append(depth_path)
                        anno_path = os.path.join(data_root, dir_1, dir_2, dir_3, seq, "label", data_name)
                        anno_paths.append(anno_path)
    assert len(image_paths) == len(depth_paths)
    assert len(image_paths) == len(anno_paths)
    print(colored("Evaluation on OCID dataset: {} rgbs, {} depths, {} visible_masks".format(
                    len(image_paths), len(depth_paths), len(anno_paths)), "green"))
    
    s, e = 0, 10
    # image_paths = image_paths[s:e]
    # depth_paths = depth_paths[s:e]
    # annot_paths = anno_paths[s:e]

    metrics_all = []
    ious_mask = 0
    num_inst_all = 0 # number of all instances
    num_inst_mat = 0 # number of matched instance
    
    for i, (rgb_path, depth_path, anno_path) in enumerate(zip(tqdm(image_paths), depth_paths, anno_paths)):      

        # load rgb and depth
        rgb_img = cv2.imread(rgb_path)
        rgb_img = cv2.resize(rgb_img, (W, H))
        depth_img = imageio.imread(depth_path)
        depth_img = normalize_depth(depth_img)
        depth_img = cv2.resize(depth_img, (W, H), interpolation=cv2.INTER_NEAREST)
        depth_img = inpaint_depth(depth_img)
        
        # UOAIS-Net inference
        if cfg.INPUT.DEPTH and cfg.INPUT.DEPTH_ONLY:
            uoais_input = depth_img
        elif cfg.INPUT.DEPTH and not cfg.INPUT.DEPTH_ONLY: 
            uoais_input = np.concatenate([rgb_img, depth_img], -1)      
        else:
            uoais_input = rgb_img

        # load GT (annotation) anno: [H, W]
        anno = imageio.imread(anno_path)
        anno = cv2.resize(anno, (W, H), interpolation=cv2.INTER_NEAREST)        
        # remove background, table
        floor_table = rgb_path.split("/")[4]
        for label in BG_LABELS[floor_table]:
            anno[anno == label] = 0         
        labels_anno = np.unique(anno)
        labels_anno = labels_anno[~np.isin(labels_anno, [BACKGROUND_LABEL])]
        num_inst_all += len(labels_anno)

        # forward (UOAIS)
        outputs = predictor(uoais_input)
        instances = detector_postprocess(outputs['instances'], H, W).to('cpu')
        if cfg.INPUT.AMODAL:
            pred_masks = instances.pred_visible_masks.detach().cpu().numpy()
        else:
            pred_masks = instances.pred_masks.detach().cpu().numpy()
        
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
        
            pred_all = np.zeros_like(anno)
            pred = np.zeros_like(anno)
            for i, mask in enumerate(pred_masks):
                iou = np.sum(np.bitwise_and(mask, fg_output)) / np.sum(mask)
                if iou >= 0.5:
                    pred[mask > False] = i+1
                pred_all[mask > False] = i+1
        else: 
            pred = np.zeros_like(anno)
            for i, mask in enumerate(pred_masks):
                pred[mask > False] = i+1
            
        # evaluate
        metrics, assignments = compute_PRF.multilabel_metrics(pred, anno, return_assign=True)
        metrics_all.append(metrics)

        # compute IoU for all instances
        # print(assignments)
        num_inst_mat += len(assignments)
        assign_visible_pred, assign_visible_gt = 0, 0
        assign_visible_overlap = 0
        for gt_id, pred_id in assignments:
            # count area of visible mask (pred & gt)
            mask_pr = pred == pred_id
            mask_gt = anno == gt_id           
            assign_visible_pred += np.count_nonzero(mask_pr)
            assign_visible_gt += np.count_nonzero(mask_gt)
            # count area of overlap btw. pred & gt
            mask_overlap = np.logical_and(mask_pr, mask_gt)
            assign_visible_overlap += np.count_nonzero(mask_overlap)
        if (assign_visible_pred+assign_visible_gt-assign_visible_overlap) > 0:
            iou = assign_visible_overlap / (assign_visible_pred+assign_visible_gt-assign_visible_overlap)
        else:
            iou = 0
        ious_mask += iou
    # compute mIoU for all instances
    miou = ious_mask / len(metrics_all)
    
    # sum the values with same keys
    result = {}
    num = len(metrics_all)
    for metrics in metrics_all:
        for k in metrics.keys():
            result[k] = result.get(k, 0) + metrics[k]
    for k in sorted(result.keys()):
        result[k] /= num

    print('\n')
    print(colored("Visible Metrics for OCID", "green", attrs=["bold"]))
    print(colored("---------------------------------------------", "green"))
    print("    Overlap    |    Boundary")
    print("  P    R    F  |   P    R    F  |  %75 | mIoU")
    print("{:.1f} {:.1f} {:.1f} | {:.1f} {:.1f} {:.1f} | {:.1f} | {:.4f}".format(
        result['Objects Precision']*100, result['Objects Recall']*100, 
        result['Objects F-measure']*100,
        result['Boundary Precision']*100, result['Boundary Recall']*100, 
        result['Boundary F-measure']*100,
        result['obj_detected_075_percentage']*100, miou
    ))
    print(colored("---------------------------------------------", "green"))
    for k in sorted(result.keys()):
        print('%s: %f' % (k, result[k]))
    print('\n')

  
    return 





def eval_visible_on_synthetic_data(args):

    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.defrost()
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, args.model_weight_name)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
    cfg.freeze()
    predictor = DefaultPredictor(cfg)
    W, H = cfg.INPUT.IMG_SIZE
    print("Using model weights from:", args.model_weight_name)
    # foreground segmentation
    if args.use_cgnet:
        print("Use foreground segmentation model (CG-Net) to filter out background instances")
        checkpoint = torch.load(os.path.join(args.cgnet_weight_path))
        fg_model = Context_Guided_Network(classes=2, in_channel=4)
        fg_model.load_state_dict(checkpoint['model'])
        fg_model.cuda()
        fg_model.eval()

    # load dataset
    import json
    ann_file = args.anno_path
    with open(ann_file, 'r') as f:
        anno_json = json.load(f)
    # ann0["annotations"]
    # rgb_paths = sorted(glob.glob("{}/image_color/*.png".format(args.dataset_path)))
    # depth_paths = sorted(glob.glob("{}/disparity/*.png".format(args.dataset_path)))
    # anno_paths = sorted(glob.glob("{}/annotation/*.png".format(args.dataset_path)))

    # assert len(rgb_paths) == len(depth_paths)
    # assert len(rgb_paths) == len(anno_paths)
    # print(colored("Evaluation on synthetic dataset: {} rgbs, {} depths, {} visible masks".format(
    #             len(rgb_paths), len(depth_paths), len(anno_paths)), "green"))
    
    metrics_all = []
    iou_masks = 0
    num_inst_all = 0 # number of all instances
    num_inst_mat = 0 # number of matched instance

    # for i, (rgb_path, depth_path, anno_path) in enumerate(zip(tqdm(rgb_paths), depth_paths, anno_paths)):
    curr_anno = []
    cur_img_id = anno_json["annotations"][0]["image_id"]
    import pycocotools.mask as mask_utils
    uniq_img_list = set([i["id"] for i in anno_json["images"]])
    pbar = tqdm(total=len(uniq_img_list))
    for i, ann in enumerate(anno_json["annotations"]):
        
        if i == len(anno_json["annotations"]) - 1:
            curr_anno.append(ann)
            img_info = anno_json["images"][i]
        elif ann["image_id"] == cur_img_id:
            curr_anno.append(ann)
            img_info = anno_json["images"][i]
            continue
        
        rgb_path = os.path.join(args.dataset_path,img_info["file_name"])
        depth_path = os.path.join(args.dataset_path,img_info["depth_file_name"])
        # load rgb and depth
        rgb_img = cv2.imread(rgb_path)
        rgb_img = cv2.resize(rgb_img, (W, H))
        depth_img = imageio.imread(depth_path)
        if "UOAIS-Sim" in args.dataset_path:
            depth_img = normalize_depth(depth_img, min_val=2500, max_val=20000)
        else: # syntable
            depth_img = normalize_depth(depth_img)
        depth_img = cv2.resize(depth_img, (W, H), interpolation=cv2.INTER_NEAREST)
        depth_img = inpaint_depth(depth_img)
        
        # UOAIS-Net inference
        if cfg.INPUT.DEPTH and cfg.INPUT.DEPTH_ONLY:
            uoais_input = depth_img
        elif cfg.INPUT.DEPTH and not cfg.INPUT.DEPTH_ONLY: 
            uoais_input = np.concatenate([rgb_img, depth_img], -1)        
        else:
            uoais_input = rgb_img
        # laod GT (annotation) anno: [H, W]
        labels_anno = []
        anno = np.zeros((H, W))
        for i, current_anno in enumerate(curr_anno):
            rle_mask = current_anno["visible_mask"]
            label_id = i +1
            mask = mask_utils.decode(rle_mask)
            mask[mask==1] = label_id
            anno += mask
            labels_anno.append(label_id)
        num_inst_all += len(labels_anno)

        # anno = imageio.imread(anno_path)
        # anno = cv2.resize(anno, (W, H), interpolation=cv2.INTER_NEAREST)
        # labels_anno = np.unique(anno)
        # labels_anno = labels_anno[~np.isin(labels_anno, [BACKGROUND_LABEL])]
        # num_inst_all += len(labels_anno)

        # forward (UOAIS)
        outputs = predictor(uoais_input)
        instances = detector_postprocess(outputs['instances'], H, W).to('cpu')

        if cfg.INPUT.AMODAL:
            pred_masks = instances.pred_visible_masks.detach().cpu().numpy()
        else:
            pred_masks = instances.pred_masks.detach().cpu().numpy()
            
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
        
            pred_all = np.zeros_like(anno)
            pred = np.zeros_like(anno)
            for i, mask in enumerate(pred_masks):
                iou = np.sum(np.bitwise_and(mask, fg_output)) / np.sum(mask)
                if iou >= 0.5:
                    pred[mask > False] = i+1
                pred_all[mask > False] = i+1
        else: 
            pred = np.zeros_like(anno)
            for i, mask in enumerate(pred_masks):
                pred[mask > False] = i+1
        
        # evaluate
        metrics, assignments = compute_PRF.multilabel_metrics(pred, anno, return_assign=True)
        metrics_all.append(metrics)

        # compute IoU for all instances
        # print(assignments)
        num_inst_mat += len(assignments)
        assign_visible_pred, assign_visible_gt = 0, 0
        assign_visible_overlap = 0
        for gt_id, pred_id in assignments:
            # count area of visible mask (pred & gt)
            mask_pr = pred == pred_id
            mask_gt = anno == gt_id           
            assign_visible_pred += np.count_nonzero(mask_pr)
            assign_visible_gt += np.count_nonzero(mask_gt)
            # count area of overlap btw. pred & gt
            mask_overlap = np.logical_and(mask_pr, mask_gt)
            assign_visible_overlap += np.count_nonzero(mask_overlap)
        if assign_visible_pred+assign_visible_gt-assign_visible_overlap > 0:
            iou = assign_visible_overlap / (assign_visible_pred+assign_visible_gt-assign_visible_overlap)
        else: iou = 0
        iou_masks += iou

        curr_anno = []
        curr_anno.append(ann)
        cur_img_id = ann["image_id"]
        pbar.update(1)
        # if cur_img_id == 10:
        #     break
    # compute mIoU for all instances
    miou = iou_masks / len(metrics_all)
    
    # sum the values with same keys
    result = {}
    num = len(metrics_all)
    for metrics in metrics_all:
        for k in metrics.keys():
            result[k] = result.get(k, 0) + metrics[k]
    for k in sorted(result.keys()):
        result[k] /= num

    print('\n')
    print(colored("Visible Metrics for OSD", "green", attrs=["bold"]))
    print(colored("---------------------------------------------", "green"))
    print("    Overlap    |    Boundary")
    print("  P    R    F  |   P    R    F  |  %75 | mIoU")
    print("{:.1f} {:.1f} {:.1f} | {:.1f} {:.1f} {:.1f} | {:.1f} | {:.4f}".format(
        result['Objects Precision']*100, result['Objects Recall']*100, 
        result['Objects F-measure']*100,
        result['Boundary Precision']*100, result['Boundary Recall']*100, 
        result['Boundary F-measure']*100,
        result['obj_detected_075_percentage']*100, miou
    ))
    print(colored("---------------------------------------------", "green"))
    for k in sorted(result.keys()):
        print('%s: %f' % (k, result[k]))
    print('\n')
    import pandas as pd
    # if args.use_cgnet:
    #     csvName = "cg_net_osd_results.csv"
    # else:
    #     csvName = "osd_results.csv"
    # result_csv_path = os.path.join(cfg.OUTPUT_DIR,csvName)
    # if os.path.isfile(result_csv_path):
    #     df = pd.read_csv(result_csv_path)

    #     dict_visible = {'Visible Overlap P': result['Objects Precision']*100, 
    #             'Visible Overlap R': result['Objects Recall']*100,
    #             'Visible Overlap F': result['Objects F-measure']*100,
    #             'Visible Boundary P': result['Boundary Precision']*100,
    #             'Visible Boundary R': result['Boundary Recall']*100, 
    #             'Visible Boundary F': result['Boundary F-measure']*100,
    #             'Visible %75': result['obj_detected_075_percentage']*100,
    #             'Visible mIoU': miou
    #             }
    #     df_vis = pd.DataFrame([dict_visible])
    #     df = pd.merge(df_vis, df,left_index=True, right_index=True)
    #     df = df.drop("Unnamed: 0", axis=1)
    #     df.to_csv(result_csv_path,index=False)
    # else:
    #     print(f"{result_csv_path} does not exists")

def eval_amodal_occ_on_synthetic_data(args):

    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.defrost()
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, args.model_weight_name)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
    cfg.freeze()
    predictor = DefaultPredictor(cfg)
    W, H = cfg.INPUT.IMG_SIZE
    print("Using model weights from:", args.model_weight_name)
    # foreground segmentation
    if args.use_cgnet:
        print("Use foreground segmentation model (CG-Net) to filter out background instances")
        checkpoint = torch.load(os.path.join(args.cgnet_weight_path))
        fg_model = Context_Guided_Network(classes=2, in_channel=4)
        fg_model.load_state_dict(checkpoint['model'])
        fg_model.cuda()
        fg_model.eval()

    # load dataset
    import json
    ann_file = args.anno_path
    with open(ann_file, 'r') as f:
        anno_json = json.load(f)

    # rgb_paths = sorted(glob.glob("{}/image_color/*.png".format(args.dataset_path)))
    # depth_paths = sorted(glob.glob("{}/disparity/*.png".format(args.dataset_path)))
    # vis_anno_paths = sorted(glob.glob("{}/annotation/*.png".format(args.dataset_path)))

    # amodal_anno_paths = sorted(glob.glob("{}/amodal_annotation/*.png".format(args.dataset_path)))
    # occlusion_anno_paths = sorted(glob.glob("{}/occlusion_annotation/*.png".format(args.dataset_path)))
    # rgb_paths = rgb_paths[:40]
    # depth_paths = depth_paths[:40]
    # vis_anno_paths = vis_anno_paths[:40]

    # assert len(rgb_paths) == len(depth_paths)
    # assert len(rgb_paths) == len(vis_anno_paths)
    # assert len(amodal_anno_paths) != 0
    # assert len(occlusion_anno_paths) != 0
    # print(colored("Evaluation on OSD dataset: {} rgbs, {} depths, {} amodal masks, {} occlusion masks".format(
    #             len(rgb_paths), len(depth_paths), len(amodal_anno_paths), len(occlusion_anno_paths)), "green"))

    metrics_all = [] # amodal mask evaluation
    num_inst_all_pred = 0 # number of all pred instances
    num_inst_all_gt = 0 # number of all GT instances
    num_inst_occ_pred = 0 # number of occluded prediction
    num_inst_occ_mat = 0 # number of occluded and matched
    num_inst_mat = 0 # number of matched instance
    
    mask_ious, occ_ious = 0, 0
    pre_occ, rec_occ, f_occ = 0, 0, 0
    pre_bou, rec_bou, f_bou = 0, 0, 0
    num_correct = 0 # for occlusion classification
    num_occ_over75 = 0 # number of instance IoU>0.75
    occ_over_75_rate = 0 # rate of instance IoU>0.75

    num_correct_occ_ooam = 0 # total correct occlusion prediction
    num_all_occ_ooam = 0 # total occlusion combinations

    # for i, (rgb_path, depth_path) in enumerate(zip(tqdm(rgb_paths), depth_paths)):
    
    # for i, (rgb_path, depth_path, anno_path) in enumerate(zip(tqdm(rgb_paths), depth_paths, vis_anno_paths)):
    curr_anno = []
    cur_img_id = anno_json["annotations"][0]["image_id"]
    import pycocotools.mask as mask_utils
    uniq_img_list = set([i["id"] for i in anno_json["images"]])
    pbar = tqdm(total=len(uniq_img_list))

    for i, ann in enumerate(anno_json["annotations"]):
        
        if i == len(anno_json["annotations"]) - 1:
            curr_anno.append(ann)
            img_info = anno_json["images"][i]
        elif ann["image_id"] == cur_img_id:
            curr_anno.append(ann)
            img_info = anno_json["images"][i]
            continue
        
        rgb_path = os.path.join(args.dataset_path,img_info["file_name"])
        depth_path = os.path.join(args.dataset_path,img_info["depth_file_name"])

        # print(rgb_path)
        # load rgb and depth
        rgb_img = cv2.imread(rgb_path)
        rgb_img = cv2.resize(rgb_img, (W, H))
        depth_img = imageio.imread(depth_path)
        if "UOAIS-Sim" in args.dataset_path:
            depth_img = normalize_depth(depth_img, min_val=2500, max_val=20000)
        else: # syntable
            depth_img = normalize_depth(depth_img)
        depth_img = cv2.resize(depth_img, (W, H), interpolation=cv2.INTER_NEAREST)
        depth_img = inpaint_depth(depth_img)
        
        # UOAIS-Net inference
        if cfg.INPUT.DEPTH and cfg.INPUT.DEPTH_ONLY:
            uoais_input = depth_img
        elif cfg.INPUT.DEPTH and not cfg.INPUT.DEPTH_ONLY: 
            uoais_input = np.concatenate([rgb_img, depth_img], -1)   
        else:
            uoais_input = rgb_img
        
        # load GT (visible masks)
        labels_anno = []
        vis_anno = np.zeros((H, W))
        for i, current_anno in enumerate(curr_anno):
            rle_mask = current_anno["visible_mask"]
            label_id = i + 1
            mask = mask_utils.decode(rle_mask)
            mask[mask==1] = label_id
            vis_anno += mask
            labels_anno.append(label_id)
        labels_anno = sorted(labels_anno)

        # vis_anno = imageio.imread(anno_path)
        # vis_anno = cv2.resize(vis_anno, (W, H), interpolation=cv2.INTER_NEAREST)
        # labels_anno = np.unique(vis_anno)
        # labels_anno = labels_anno[~np.isin(labels_anno, [BACKGROUND_LABEL])]
        # labels_anno = sorted(labels_anno)
        # print(labels_anno)

        inst_vis_mask_list = []
        for label in labels_anno:
            vis_mask = vis_anno.copy()
            vis_mask[np.where(vis_mask == label)] = 1
            vis_mask =vis_mask.astype(np.uint8)
            inst_vis_mask_list.append(vis_mask)

        # load GT (amodal masks)
        img_name = os.path.basename(rgb_path)[:-4]
        annos = [] # [instance, IMG_H, IMG_W]
        # filtered_amodal_paths = list(filter(lambda p: img_name + "_" in p, amodal_anno_paths))
        # filtered_occlusion_paths = list(filter(lambda p: img_name + "_" in p, occlusion_anno_paths))
        # print(filtered_occlusion_paths)
        # vis_anno = np.zeros((H, W))
        for i, current_anno in enumerate(curr_anno):
            rle_mask = current_anno["segmentation"]
            inst_id = i + 1
            anno = mask_utils.decode(rle_mask)
            anno = cv2.resize(anno, (W, H), interpolation=cv2.INTER_NEAREST)
             # fill mask with instance id
            cnd = anno > 0
            anno_mask = np.zeros((H, W))
            anno_mask[cnd] = inst_id
            annos.append(anno_mask)            
        annos = np.stack(annos)
        num_inst_all_gt += len(curr_anno)

        # for anno_path in filtered_amodal_paths:
        #     # get instance id  
        #     inst_id = os.path.basename(anno_path)[:-4].split("_")[-1]
        #     inst_id = int(inst_id)
        #     # load mask image
        #     anno = imageio.imread(anno_path)
        #     anno = cv2.resize(anno, (W, H), interpolation=cv2.INTER_NEAREST)
        #     # fill mask with instance id
        #     cnd = anno > 0
        #     anno_mask = np.zeros((H, W))
        #     anno_mask[cnd] = inst_id
        #     annos.append(anno_mask)            
        # annos = np.stack(annos)
        # num_inst_all_gt += len(filtered_amodal_paths)
        
        # forward (UOAIS)
        outputs = predictor(uoais_input)
        instances = detector_postprocess(outputs['instances'], H, W).to('cpu')
        # print(instances)
        if not args.use_cgnet:
            pred_masks = instances.pred_masks.detach().cpu().numpy()
            preds = [] # mask per each instance
            for i, mask in enumerate(pred_masks):
                pred = np.zeros((H, W))
                pred[mask > False] = i+1
                preds.append(pred)
                num_inst_all_pred += 1
        else:
            fg_rgb_input = standardize_image(cv2.resize(rgb_img, (320, 240)))
            fg_rgb_input = array_to_tensor(fg_rgb_input).unsqueeze(0)
            fg_depth_input = cv2.resize(depth_img, (320, 240)) 
            fg_depth_input = array_to_tensor(fg_depth_input[:,:,0:1]).unsqueeze(0) / 255
            fg_input = torch.cat([fg_rgb_input, fg_depth_input], 1)
            fg_output = fg_model(fg_input.cuda())
            fg_output = fg_output.cpu().data[0].numpy().transpose(1, 2, 0)
            fg_output = np.asarray(np.argmax(fg_output, axis=2), dtype=np.uint8)
            fg_output = cv2.resize(fg_output, (W, H), interpolation=cv2.INTER_NEAREST)

            # filter amodal predictions with foreground mask
            pred_masks = instances.pred_masks.detach().cpu().numpy()
            preds = [] # mask per each instance
            for i, mask in enumerate(pred_masks):
                overlap = np.sum(np.bitwise_and(mask, fg_output)) / np.sum(mask)
                if overlap >= 0.5: # filiter outliers
                    pred = np.zeros((H, W))
                    pred[mask > False] = i+1
                    preds.append(pred)
                    num_inst_all_pred += 1

        if len(preds) > 0:
            preds = np.stack(preds)
        else:
            preds = np.array(preds)
        # evaluate amodal masks
        metrics, assignments = compute_PRF.multilabel_amodal_metrics(preds, annos, return_assign=True)
        metrics_all.append(metrics)

        # count the number of instances
        num_inst_mat += len(assignments)

        if not cfg.INPUT.AMODAL:
            raise ValueError("Cannot support amodal segmentation")

        amodals = instances.pred_masks.detach().cpu().numpy()
        visibles = instances.pred_visible_masks.detach().cpu().numpy()        
        
        # count occluded area of predictions when classified
        all_occ_pred, all_bou_pred = 0, 0
        num_inst_occ_prd_img = 0
        for pred in preds:
            idx = int(pred.max())-1
            amodal = amodals[idx]
            visible = visibles[idx]
            occ = np.bitwise_xor(amodal, visible)
            try:
                cls = instances.pred_occlusions[idx].item()
            except:
                # if area over 5% of amodal mask is not visible
                cls = 1 if np.int64(np.count_nonzero(occ)) / np.int64(np.count_nonzero(amodal)) >= 0.05 else 0                
            if not cls: continue
            num_inst_occ_pred += 1
            num_inst_occ_prd_img += 1
            all_occ_pred += np.int64(np.count_nonzero(occ))
            all_bou_pred += np.sum(compute_PRF.seg2bmap(occ))
        
        occlusion_mask_list = []
        occlusion_mask_list_filtered = []
        # count occluded area of ground truth
        all_occ_gt, all_bou_gt = 0, 0
        # occ_paths = filtered_occlusion_paths
        # for occ_path in occ_paths:
        for i, current_anno in enumerate(curr_anno):
            rle_mask = current_anno["occluded_mask"]
            # inst_id = i + 1
            occ = mask_utils.decode(rle_mask)
            # occ = imageio.imread(occ_path)
            occ = cv2.resize(occ, (W, H), interpolation=cv2.INTER_NEAREST)
            occ = occ[:,:] > 0
            all_occ_gt += np.int64(np.count_nonzero(occ))
            all_bou_gt += np.sum(compute_PRF.seg2bmap(occ))
            
            # idx = int(os.path.basename(occ_path)[:-4].split("_")[1]) - 1
            # print("idx:",idx)
            # print(np.unique(occ))
            occlusion_mask_list.append(occ)
            if int(np.count_nonzero(occ)) > 0:
                occlusion_mask_list_filtered.append(occ)
        # generate occlusion ordering (gt_ooam)
        # print(inst_vis_mask_list)
        # print(occlusion_mask_list)
        
        gt_ooam = generate_ooam(inst_vis_mask_list,occlusion_mask_list)
        # print("gt_ooam:")
        # print(gt_ooam)

        # count area with matched instances
        # assign: [[gt_id, pred_id], ... ]
        assign_amodal_pred, assign_visible_pred, assign_amodal_gt = 0, 0, 0
        assign_occ_pred, assign_occ_gt = 0, 0
        assign_amodal_overlap, assign_occ_overlap = 0, 0
        occ_bou_pre, occ_bou_rec = 0, 0
        num_occ_over75_img = 0

        pred_occ_mask_list_matched = []
        pred_vis_mask_list_matched = []
        # print("assignments:\n",assignments)
        for gt_id, pred_id in assignments:              
            ###############
            # AMODAL MASK #
            ###############
            # count area of masks predictions
            amodal = amodals[int(pred_id)-1] # predicted amodal mask
            visible = visibles[int(pred_id)-1] # predicted visible mask
            assign_amodal_pred += np.count_nonzero(amodal)
            assign_visible_pred += np.count_nonzero(visible)
            # count area of mask GT [ annos: amodal GT ]
            anno = annos[np.where(annos == gt_id)[0][0]] > 0
            assign_amodal_gt += np.count_nonzero(anno)
            # count overlap area of mask btw. pred & GT
            amodal_overlap = np.logical_and(amodal, anno)
            assign_amodal_overlap += np.count_nonzero(amodal_overlap)
                        
            ##################
            # OCCLUSION MASK #
            ##################
            # count area of occlusion prediction
            occ_pred = np.bitwise_xor(amodal, visible) # predicted occlusion mask
            assign_occ_pred += np.count_nonzero(occ_pred)

            pred_occ_mask_list_matched.append(occ_pred)
            pred_vis_mask_list_matched.append(visible)
            ############################
            # OCCLUSION CLASSIFICATION #
            ############################
            # count occlusion classification corrects
            # cls_gt = os.path.isfile("{}/occlusion_annotation/{}_{}.png".format(
            #                         args.dataset_path, img_name, int(gt_id)))
            # if there is occlusion mask, cls_gt = 1, else cls_gt = 0
            cls_gt = 1 if np.int64(np.count_nonzero(occlusion_mask_list[int(gt_id)-1])) > 0 else 0
            
            try:
                cls_pred = instances.pred_occlusions[int(pred_id)-1].item()
            except: # if occ prediction count / amodal prediction count > 0.05, occlusion class = 1
                cls_pred = 1 if np.int64(np.count_nonzero(occ_pred)) / np.int64(np.count_nonzero(amodal)) >= 0.05 else 0
            num_correct += cls_pred==cls_gt
            if cls_pred==cls_gt and cls_pred == 1:
                num_inst_occ_mat += cls_pred         

            ##################
            # OCCLUSION MASK #
            ##################
            # count area of occlusion GT
            # occ_path = "{}/occlusion_annotation/{}_{}.png".format(
            #                         args.dataset_path, img_name, int(gt_id))
            
            # if not os.path.isfile(occ_path) or not cls_pred: continue
            if int(np.count_nonzero(occlusion_mask_list[int(gt_id)-1])) == 0 or not cls_pred: continue # if pred or gt is class 0
            
            # occ_gt = imageio.imread(occ_path)
            occ_gt = occlusion_mask_list[int(gt_id)-1]
            # print(occ_gt)
            # print(np.unique(occ_gt))
            occ_gt = cv2.resize(np.float32(occ_gt), (W, H), interpolation=cv2.INTER_NEAREST)
            occ_gt = occ_gt[:,:] > 0
            assign_occ_gt += np.count_nonzero(occ_gt)
            # count overlap area of occlusion btw. pred & GT
            occ_overlap = np.logical_and(occ_pred, occ_gt)
            assign_occ_overlap += np.count_nonzero(occ_overlap)
            
            ############################
            # Over75 of OCCLUSION MASK #
            ############################
            pre = np.int64(np.count_nonzero(occ_overlap)) / np.count_nonzero(occ_pred)
            rec = np.int64(np.count_nonzero(occ_overlap)) / np.count_nonzero(occ_gt)
            f = (2 * pre * rec) / (pre + rec) if (pre + rec) > 0 else 0
            if f > 0.75: 
                num_occ_over75_img += 1
                num_occ_over75 += 1
                        
            ###########################
            # OCCLUSION MASK BOUNDARY #
            ###########################
            pre, rec = compute_PRF.boundary_overlap_occ(gt_id, pred_id, occ_pred, occ_gt)
            occ_bou_pre += pre
            occ_bou_rec += rec

        
        pred_ooam = generate_ooam(pred_vis_mask_list_matched, pred_occ_mask_list_matched)
        # print("pred_ooam:")
        # print(pred_ooam)
        
        # ooam_score = np.sum(pred_ooam == gt_ooam) / gt_ooam.size # individual acc score
        ooam_diagonal_size = len(gt_ooam)
        num_correct_occ_ooam += np.sum(pred_ooam == gt_ooam) - ooam_diagonal_size
        num_all_occ_ooam += gt_ooam.size - ooam_diagonal_size

        #####################################
        # COMPUTE METRICS in a single image #
        #####################################
        # mIoU of amodal mask (only matched instance)
        if assign_amodal_pred+assign_amodal_gt-assign_amodal_overlap > 0:
            iou = assign_amodal_overlap / (assign_amodal_pred+assign_amodal_gt-assign_amodal_overlap)
        else: iou = 0
        mask_ious += iou
        # mIoU of occlusion mask (only matched instance)
        if assign_occ_pred+assign_occ_gt-assign_occ_overlap > 0:
            iou = assign_occ_overlap / (assign_occ_pred+assign_occ_gt-assign_occ_overlap)
        else: iou = 0
        occ_ious +=iou
        
        # number of occluded instances in one image
        num_pred = num_inst_occ_prd_img
        # num_gt = len(filtered_occlusion_paths)
        num_gt = len(occlusion_mask_list_filtered)
        if num_pred == 0 and num_gt > 0:
            pre_occ += 1
            pre_bou += 1
        elif num_pred > 0 and num_gt == 0:
            rec_occ += 1
            rec_bou += 1
        elif num_pred == 0 and num_gt == 0:
            pre_occ += 1
            rec_occ += 1
            f_occ += 1
            pre_bou += 1
            rec_bou += 1
            f_bou += 1
            occ_over_75_rate += 1
        else:
            assert (num_pred > 0) and (num_gt > 0)
            # P, R, F of occlusion mask (all instance)
            pre = assign_occ_overlap / all_occ_pred if all_occ_pred > 0 else 0
            rec = assign_occ_overlap / all_occ_gt if all_occ_gt > 0 else 0
            f = (2 * pre * rec) / (pre + rec) if (pre + rec) > 0 else 0
            pre_occ += pre
            rec_occ += rec
            f_occ += f
            # P, R, F of occlusion boundary (all instance)
            pre = occ_bou_pre / all_bou_pred if all_bou_pred > 0 else 0
            rec = occ_bou_rec / all_bou_gt if all_bou_gt > 0 else 0
            f = (2 * pre * rec) / (pre + rec) if (pre + rec) > 0 else 0
            pre_bou += pre
            rec_bou += rec
            f_bou += f
            occ_over_75_rate += num_occ_over75_img / num_gt
        
        curr_anno = []
        curr_anno.append(ann)
        cur_img_id = ann["image_id"]
        pbar.update(1)
        # if cur_img_id == 10:
        #     break
    ###############################################
    # get average metirc values among test images #
    ###############################################
    num = len(metrics_all)
    mask_ious /= num 
    occ_ious /= num 
    pre_occ /= num 
    rec_occ /= num 
    f_occ /= num 
    pre_bou /= num 
    rec_bou /= num 
    f_bou /= num 
    occ_over_75_rate /= num
    
    occ_cls_acc = num_correct / num_inst_mat * 100 if num_inst_mat > 0 else 0
    occ_cls_pre = num_correct / num_inst_all_pred * 100 if num_inst_all_pred > 0 else 0
    occ_cls_rec = num_correct / num_inst_all_gt * 100 if num_inst_all_gt > 0 else 0
    occ_cls_f = (2*occ_cls_pre*occ_cls_rec) / (occ_cls_pre+occ_cls_rec) if occ_cls_pre+occ_cls_rec > 0 else 0

    occ_ooam_acc = num_correct_occ_ooam / num_all_occ_ooam * 100 if num_all_occ_ooam > 0 else 0

    # sum the values with same keys
    result = {}
    for metrics in metrics_all:
        for k in metrics.keys():
            result[k] = result.get(k, 0) + metrics[k]
    for k in sorted(result.keys()):
        result[k] /= num

    print('\n')
    print(colored("Amodal Metrics on OSD (for {} instances)".format(num_inst_all_gt), "green", attrs=["bold"]))
    print(colored("---------------------------------------------", "green"))
    print("    Overlap    |    Boundary")
    print("  P    R    F  |   P    R    F  |  %75 | mIoU")
    print("{:.1f} {:.1f} {:.1f} | {:.1f} {:.1f} {:.1f} | {:.1f} | {:.4f}".format(
        result['Objects Precision']*100, result['Objects Recall']*100, 
        result['Objects F-measure']*100,
        result['Boundary Precision']*100, result['Boundary Recall']*100, 
        result['Boundary F-measure']*100,
        result['obj_detected_075_percentage']*100, mask_ious
    ))
    print(colored("---------------------------------------------", "green"))
    for k in sorted(result.keys()):
        print('%s: %f' % (k, result[k]))
    print('\n')
    print(colored("Occlusion Metrics on OSD", "green", attrs=["bold"]))
    print(colored("---------------------------------------------", "green"))
    print("    Overlap    |    Boundary")
    print("  P    R    F  |   P    R    F  |  %75 | mIoU")
    print("{:.1f} {:.1f} {:.1f} | {:.1f} {:.1f} {:.1f} | {:.1f} | {:.4f}".format(
        pre_occ*100, rec_occ*100, f_occ*100, 
        pre_bou*100, rec_bou*100, f_bou*100,
        occ_over_75_rate*100, occ_ious        
    ))
    print(colored("---------------------------------------------", "green"))
    print('\n')
    print(colored("Occlusion Classification on OSD", "green", attrs=["bold"]))
    print(colored("---------------------------------------------", "green"))
    print("  P   R   F   ACC")
    print("{:.1f} {:.1f} {:.1f} {:.1f}".format(
        occ_cls_pre, occ_cls_rec, occ_cls_f, occ_cls_acc        
    ))
    print(colored("---------------------------------------------", "green"))
    print('\n')
    print(colored("Occlusion Order Adjacency Matrix (OOAM) on OSD", "green", attrs=["bold"]))
    print(colored("---------------------------------------------", "green"))
    print("ACC")
    print("{:.1f}".format(
        occ_ooam_acc        
    ))
    print(colored("---------------------------------------------", "green"))
    print('\n')
    
    # import pandas as pd
    # dict_amodal = {'Amodal Overlap P': result['Objects Precision']*100, 
    #         'Amodal Overlap R': result['Objects Recall']*100,
    #         'Amodal Overlap F': result['Objects F-measure']*100,
    #         'Amodal Boundary P': result['Boundary Precision']*100,
    #         'Amodal Boundary R': result['Boundary Recall']*100, 
    #         'Amodal Boundary F': result['Boundary F-measure']*100,
    #         'Amodal %75': result['obj_detected_075_percentage']*100,
    #         'Amodal mIoU': mask_ious
    #         }
    # # df = pd.DataFrame([dict_])
    # # df.to_csv(os.path.join(cfg.OUTPUT_DIR,"osd_amodal_results.csv"),index=False)

    # dict_occ = {'Occlusion Overlap P': pre_occ*100, 
    #         'Occlusion Overlap R': rec_occ*100,
    #         'Occlusion Overlap F': f_occ*100,
    #         'Occlusion Boundary P': pre_bou*100,
    #         'Occlusion Boundary R': rec_bou*100, 
    #         'Occlusion Boundary F': f_bou*100,
    #         'Occlusion %75': occ_over_75_rate*100,
    #         'Occlusion mIoU': occ_ious   
    #         }
    # # df = pd.DataFrame([dict_])
    # # df.to_csv(os.path.join(cfg.OUTPUT_DIR,"osd_occlusion_results.csv"),index=False)
     
    # dict_occ_class = {'Occlusion Classification P': occ_cls_pre, 
    #         'Occlusion Classification R': occ_cls_rec,
    #         'Occlusion Classification F': occ_cls_f,
    #         'Occlusion Classification Accuracy': occ_cls_acc
    #         }
    # dict_amodal.update(dict_occ)
    # dict_amodal.update(dict_occ_class)
    # df = pd.DataFrame([dict_amodal])
    # if args.use_cgnet:
    #     csvName = "cg_net_osd_results.csv"
    # else:
    #     csvName = "osd_results.csv"
    # df.to_csv(os.path.join(cfg.OUTPUT_DIR,csvName))

    return