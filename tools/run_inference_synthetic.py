# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import cv2
import imageio
import random
import numpy as np
from tqdm import tqdm
# constants
from detectron2.engine import DefaultPredictor

from adet.config import get_cfg
from adet.utils.visualizer import visualize_pred_amoda_occ
from adet.utils.post_process import detector_postprocess, DefaultPredictor

from utils import *
from foreground_segmentation.model import Context_Guided_Network

from PIL import Image
import pycocotools.mask as mask_util
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import pyplot as plt

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument("--gpu", type=str, default="0", help="GPU id")
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
        type=str,
        default="./output_inference",
        help="path to the output dir"
    )
    parser.add_argument(
        "--model-weight-path",
        type=str,
        default="model_final.pth",
        help="relative path to the model weights in output dir"
    )
    parser.add_argument(
        "--anno-path",
        type=str,
        default="./datasets/UOAIS-Sim/annotations/coco_anns_uoais_sim_val.json",
        help="path to the uoais_sim annotations"
    )
    return parser


if __name__ == "__main__":

    # UOAIS-Net
    args = get_parser().parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.defrost()
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, args.model_weight_path) # "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = args.confidence_threshold
    predictor = DefaultPredictor(cfg)
    W, H = cfg.INPUT.IMG_SIZE
    output_dir = args.output_path
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        
    # CG-Net (foreground segmentation)
    if args.use_cgnet:
        print("Use foreground segmentation model (CG-Net) to filter out background instances")
        checkpoint = torch.load(os.path.join(args.cgnet_weight_path))
        fg_model = Context_Guided_Network(classes=2, in_channel=4)
        fg_model.load_state_dict(checkpoint['model'])
        fg_model.cuda()
        fg_model.eval()

    # # Load OSD dataset
    # if args.dataset_path in ["./sample_data","./datasets/OSD-0.2-depth"]:
    #     rgb_paths = sorted(glob.glob("{}/image_color/*.png".format(args.dataset_path)))
    #     depth_paths = sorted(glob.glob("{}/disparity/*.png".format(args.dataset_path)))
    #     vis_anno_paths = sorted(glob.glob("{}/annotation/*.png".format(args.dataset_path)))
    #     amodal_anno_paths = sorted(glob.glob("{}/amodal_annotation/*.png".format(args.dataset_path)))
    #     occlusion_anno_paths = sorted(glob.glob("{}/occlusion_annotation/*.png".format(args.dataset_path)))
    # # Load Other Datasets
    # else:
    rgb_paths = sorted(glob.glob("{}/rgb/*.png".format(args.dataset_path)))
    depth_paths = sorted(glob.glob("{}/depth/*.png".format(args.dataset_path)))
    # load dataset
    import json
    ann_file = args.anno_path
    with open(ann_file, 'r') as f:
        anno_json = json.load(f)
    
    # rgb_paths =rgb_paths[:1]
    print(f"Total rgb images: {len(rgb_paths)}, Total depth images: {len(depth_paths)}")

    curr_anno = []
    cur_img_id = anno_json["annotations"][0]["image_id"]
    import pycocotools.mask as mask_utils
    uniq_img_list = set([i["id"] for i in anno_json["images"]])
    pbar = tqdm(total=len(uniq_img_list),position=0, leave=True)

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
        # labels_anno = sorted(labels_anno)

        inst_vis_mask_list = []
        for label in labels_anno:
            vis_mask = vis_anno.copy()
            vis_mask[np.where(vis_mask == label)] = 1
            vis_mask =vis_mask.astype(np.uint8)
            inst_vis_mask_list.append(vis_mask)

        gt_bboxes = []
        for label in labels_anno:
            vis_mask = vis_anno.copy()
            vis_mask[np.where(vis_mask == label)] = 1
            vis_mask =vis_mask.astype(np.uint8)
            
            def get_bbox(mask):
                ys, xs = np.where(mask)
                if len(xs) == 0 or len(ys) == 0:
                    return [0, 0, 0, 0]
                x_min, x_max = xs.min(), xs.max()
                y_min, y_max = ys.min(), ys.max()
                return [int(x_min), int(y_min), int(x_max-x_min), int(y_max-y_min)]
            
            gt_bboxes.append(get_bbox(vis_mask))

        # load GT (amodal masks)
        img_name = os.path.basename(rgb_path)[:-4]
        amodal_annos = [] # [instance, IMG_H, IMG_W]
        for i, current_anno in enumerate(curr_anno):
            rle_mask = current_anno["segmentation"]
            inst_id = i + 1
            anno = mask_utils.decode(rle_mask)
            anno = cv2.resize(anno, (W, H), interpolation=cv2.INTER_NEAREST)
             # fill mask with instance id
            cnd = anno > 0
            anno_mask = np.zeros((H, W))
            anno_mask[cnd] = inst_id
            amodal_annos.append(anno_mask)            
        amodal_annos = np.stack(amodal_annos)

        
        occlusion_mask_list = []
        occlusion_mask_list_filtered = []
        occ_annos = [0 for _ in range(len(curr_anno))]
        for i, current_anno in enumerate(curr_anno):
            rle_mask = current_anno["occluded_mask"]
            # inst_id = i + 1
            occ = mask_utils.decode(rle_mask)
            occ = cv2.resize(occ, (W, H), interpolation=cv2.INTER_NEAREST)
            occ = occ[:,:] > 0
            
            occlusion_mask_list.append(occ)
            if int(np.count_nonzero(occ)) > 0:
                occlusion_mask_list_filtered.append(occ)
                try:
                    occ_annos[i] = 1
                except:
                    print("inst_id-1 is out of range",inst_id-1)
        occ_annos = np.stack(occ_annos)


        curr_anno = []
        curr_anno.append(ann)
        cur_img_id = ann["image_id"]
        pbar.update(1)

       
        # UOAIS-Net inference
        if cfg.INPUT.DEPTH and cfg.INPUT.DEPTH_ONLY:
            uoais_input = depth_img
        elif cfg.INPUT.DEPTH and not cfg.INPUT.DEPTH_ONLY: 
            uoais_input = np.concatenate([rgb_img, depth_img], -1)        
        outputs = predictor(uoais_input)
        instances = detector_postprocess(outputs['instances'], H, W).to('cpu')

        # print("instances:\n",instances)
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

        preds = instances.pred_masks.detach().cpu().numpy() 
        pred_visibles = instances.pred_visible_masks.detach().cpu().numpy() 
        bboxes = instances.pred_boxes.tensor.detach().cpu().numpy() 
        pred_occs = instances.pred_occlusions.detach().cpu().numpy() 
        
        pred_occlusion_masks = instances.pred_occluded_masks.detach().cpu().numpy() 

        # filter out the background instances
        if args.use_cgnet:
            remove_idxs = []
            for i, pred_visible in enumerate(pred_visibles):
                iou = np.sum(np.bitwise_and(pred_visible, fg_output)) / np.sum(pred_visible)
                if iou < 0.5: 
                    remove_idxs.append(i)
            preds = np.delete(preds, remove_idxs, 0)
            pred_visibles = np.delete(pred_visibles, remove_idxs, 0)
            bboxes = np.delete(bboxes, remove_idxs, 0)
            pred_occs = np.delete(pred_occs, remove_idxs, 0)
        
        # reorder predictions for visualization
        idx_shuf = np.concatenate((np.where(pred_occs==1)[0] , np.where(pred_occs==0)[0] )) 
        preds, pred_occs, pred_visibles, pred_occluded_masks, bboxes \
            = preds[idx_shuf], pred_occs[idx_shuf], pred_visibles[idx_shuf], pred_occlusion_masks[idx_shuf], bboxes[idx_shuf]

        
        vis_img = visualize_pred_amoda_occ(rgb_img, preds, bboxes, pred_occs)
        # print("preds:",preds.shape)
        # print("amodal_annos\n",amodal_annos.shape)
        # print("pred_occs",pred_occs)
        # print("pred_occs:\n",pred_occs.shape)
        # print("occ_annos\n",occ_annos.shape)
        # print("bboxes\n",bboxes.shape)
        # print("gt_bboxes\n",np.array(gt_bboxes).shape)
        ground_img = visualize_pred_amoda_occ(rgb_img, amodal_annos, gt_bboxes, occ_annos)
        vis_all_img = np.hstack([rgb_img, depth_img, ground_img, vis_img])
        

        image_name = rgb_path.split("/")[-1][:-4]
        output_dir_img = os.path.join(output_dir,image_name)
        if not os.path.exists(output_dir_img):
            os.makedirs(output_dir_img)

        save_path = os.path.join(output_dir_img,rgb_path.split("/")[-1])
        cv2.imwrite(save_path, vis_all_img)
        
        #cv2.imshow(rgb_path.split("/")[-1] + "/ Press any key to view the next. / ESC: quit", vis_all_img)
        k = cv2.waitKey(0)
        if k == 27: # esc
            break  
        else:
            cv2.destroyAllWindows()
        
        def apply_mask(image, mask):
            # Convert to numpy arrays
            image = np.array(image)
            mask = np.array(mask)
            # Convert grayscale image to RGB
            mask = np.stack((mask,)*3, axis=-1)
            # Multiply arrays
            rgb_result= image*mask

            # First create the image with alpha channel
            rgba = cv2.cvtColor(rgb_result, cv2.COLOR_RGB2RGBA)

            # Then assign the mask to the last channel of the image
            # rgba[:, :, 3] = alpha_data
            # Make image transparent white anywhere it is transparent
            rgba[rgba[...,-1]==0] = [255,255,255,0]

            return rgba
        
        def convert_png(image):
            image = Image.fromarray(np.uint8(image))
            image = image.convert('RGBA')
            # Transparency
            newImage = []
            for item in image.getdata():
                if item[:3] == (0, 0, 0):
                    newImage.append((0, 0, 0, 0))
                else:
                    newImage.append(item)
            image.putdata(newImage)
            return image


        visualise_ooam = False
        verbose = False

        if visualise_ooam:
            """ Generate OOAM, OODAG """

            """ === Generate OOAM === """
            vis_img_list = pred_visibles
            occ_img_list = pred_occluded_masks
            # generate occlusion ordering for current viewport
            rows = cols = len(vis_img_list)
            ooam = np.zeros((rows,cols))
            diagonal_coords_list = []
            # A(i,j), col j, row i. row i --> col j
            for i in range(0,len(vis_img_list)):
                visible_mask_i =  vis_img_list[i] # occluder
                for j in range(0,len(vis_img_list)):
                    if j != i:
                        occluded_mask_j = occ_img_list[j] # occludee
                        intersection = np.count_nonzero(np.logical_and(visible_mask_i,occluded_mask_j))
                        if intersection > 0: # object i's visible mask is overlapping object j's occluded mask
                            ooam[i][j] = 1
                    else:
                        diagonal_coords_list.append((i,j))
            if verbose:
                print(f"\nCalculating Directed Graph for Image:{image_name}")
            # vis_img = cv2.imread(f"{vis_dir}/visuals/{id}.png", cv2.IMREAD_UNCHANGED)
            rows = cols = len(vis_img_list) # number of objects
            obj_rgb_mask_list = []
            for i in range(1,len(vis_img_list)+1):
                visMask = vis_img_list[i-1]
                visible_mask = visMask
                
                rgb_crop = apply_mask(rgb_img, visible_mask)
                rgb_crop = convert_png(rgb_crop)
                
                def bbox(im):
                    a = np.array(im)[:,:,:3]  # keep RGB only
                    m = np.any(a != [0,0,0], axis=2)
                    coords = np.argwhere(m)
                    # print('coords:',coords)
                    if len(coords)!=0:
                        y0, x0, y1, x1 = *np.min(coords, axis=0), *np.max(coords, axis=0)
                    else:
                        y0, x0, y1, x1 = 0,0,0,0
                    return (x0, y0, x1+1, y1+1)

                # print(bbox(rgb_crop))
                # print(rgb_crop)
                # print(rgb_crop.size)
                # print(bbox(rgb_crop))
                # print(len(bbox(rgb_crop)))
                obj_rgb_mask = rgb_crop.crop(bbox(rgb_crop))

                obj_rgb_mask_list.append(obj_rgb_mask) # add obj_rgb_mask


            """ === Generate Directed Graph === """
            # print("Occlusion Order Adjacency Matrix:\n",ooam)

            # show_graph_with_labels(overlap_adjacency_matrix,ax1)
            labels = [i for i in range(1,len(ooam)+1)]
            labels_dict = {}

            
            # for i in range(len(ooam)):
            #     labels_dict.update({i:labels[i]})
            
            rows, cols = np.where(ooam == 1)
            rows += 1
            cols += 1
            edges = zip(rows.tolist(), cols.tolist())
            nodes_list = [i for i in range(1, len(ooam)+1)]

            # Initialise directed graph G
            G = nx.DiGraph()
            G.add_nodes_from(nodes_list)
            G.add_edges_from(edges)
            
            is_planar, embedding = nx.check_planarity(G)
            if is_planar: # planar layout
                pos=nx.planar_layout(G)
            else: # Compute Kamada-Kawai layout
                pos = nx.kamada_kawai_layout(G)
            
            # get start nodes
            start_nodes = [node for (node,degree) in G.in_degree if degree == 0]
            
            # get end nodes
            end_nodes = [node for (node,degree) in G.out_degree if degree == 0]
            for node in end_nodes:
                if node in start_nodes:
                    end_nodes.remove(node)

            # get intermediate notes
            intermediate_nodes = [i for i in nodes_list if i not in (start_nodes) and i not in (end_nodes)]
            if verbose:
                print("Nodes:",G.nodes())
                print("Edges:",G.edges())
                print("start_nodes:",start_nodes)
                print("end_nodes:",end_nodes)
                print("intermediate_nodes:",intermediate_nodes)
                print("(Degree of clustering) Number of Weakly Connected Components:",nx.number_weakly_connected_components(G))

            wcc_list = list(nx.weakly_connected_components(G))
            wcc_len = []
            for component in wcc_list:
                wcc_len.append(len(component))
            if verbose:
                print("(Scene Complexity/Degree of overlapping regions) Sizes of Weakly Connected Components:",wcc_len)
            if not nx.is_directed_acyclic_graph(G): #not G.is_directed():
                if verbose:
                    print("Graph is not directed and contains a cycle!")
                
            else:
                dag_longest_path_length = nx.dag_longest_path_length(G)
                if verbose:
                    print("(Minimum no. of depth layers to order all regions in WCC) Longest directed path of Weakly Connected Components:",dag_longest_path_length)

            node_color_list = []
            node_size_list = []
            for node in nodes_list:
                if node in start_nodes:
                    node_color_list.append('green')
                    node_size_list.append(500)
                elif node in end_nodes:
                    node_color_list.append('yellow')
                    node_size_list.append(300)
                else:
                    node_color_list.append('#1f78b4')
                    node_size_list.append(300)

            options = {
            'node_color': node_color_list,
            # 'node_size': node_size_list,
            'width': 3,
            'arrowstyle': '-|>',
            'arrowsize': 30,
            'node_size':2000,
            'font_size':30
            }
            fig1 = plt.figure(figsize=(16, 10), dpi=80)
            
            # plt.subplot(1,3,1)
            if is_planar: # planar layout
                nx.draw_planar(G,  with_labels = True, arrows=True, **options)
            else: # Compute Kamada-Kawai layout
                nx.draw_kamada_kawai(G, with_labels=True, arrows=True, **options)

            

            dag = nx.is_directed_acyclic_graph(G)
            if verbose:
                print(f"Is Directed Acyclic Graph (DAG)?: {dag}")
            if dag:
                title='Acyclic'
            else:
                title='Cyclic'
            
            import matplotlib.patches as mpatches
            import matplotlib.pyplot as plt
            colors = ["green", "#1f78b4", "yellow"]
            texts = ["Top Layer", "Intermediate Layers", "Bottom Layer"]
            patches = [ plt.plot([],[], marker="o", ms=30, ls="", mec=None, color=colors[i], 
                        label="{:s}".format(texts[i]) )[0]  for i in range(len(texts)) ]
            plt.legend(handles=patches, bbox_to_anchor=(0.5, -0.05), 
                    loc='center', ncol=3, fancybox=True, shadow=True, 
                    facecolor="w", numpoints=1, fontsize=30)
            
                    
            # plt.subplot(1,2,2)
            # plt.imshow(vis_img)       
            # plt.imshow(vis_img)
            
            # plt.title(f"Visible Masks Scene {id}")
            plt.axis('off')
            # plt.show()
            # plt.title(f"Object Index: Directed Occlusion Order Graph ({title})")
            plt.savefig(f"{output_dir_img}/object_index_directed_graph_{image_name}.png",bbox_inches='tight')
            # cv2.imwrite(f"{output_dir_img}/scene_{id}.png", vis_img)
            # plt.show()
            plt.close('all')

            fig2 = plt.figure(figsize=(16, 10), dpi=80)
            # plt.title(f"Object RGB: Directed Occlusion Order Graph ({title})")
            # plt.subplot(1,3,2)
            options = {
            'node_color': "white",
            'width': 2,
            'arrowstyle': '-|>',
            'arrowsize': 20,
            'node_size':2000,
            }

            N = len(G.nodes())
            from math import sqrt

            is_planar, embedding = nx.check_planarity(G)
            if is_planar: # planar layout
                pos=nx.planar_layout(G)
            else: # Compute Kamada-Kawai layout
                pos = nx.kamada_kawai_layout(G)
            
            nx.draw_networkx(G,pos, with_labels= False, arrows=True, **options)
            
            # draw with images on nodes
            # nx.draw_networkx(G,pos,width=3,edge_color="r",alpha=0.6)
            ax=plt.gca()
            fig=plt.gcf()
            trans = ax.transData.transform
            trans2 = fig.transFigure.inverted().transform
            imsize = 0.05 # this is the image size

            node_size_list = []
            for n in G.nodes():
                (x,y) = pos[n]
                xx,yy = trans((x,y)) # figure coordinates
                xa,ya = trans2((xx,yy)) # axes coordinates
                # a = plt.axes([xa-imsize/2.0,ya-imsize/2.0, imsize, imsize ])
                a = plt.axes([xa-imsize/2.0,ya-imsize/2.0, imsize, imsize ])
                a.imshow(obj_rgb_mask_list[n-1])
                a.set_aspect('equal')
                a.axis('off')
            ax.axis('off')

            plt.savefig(f"{output_dir_img}/object_rgb_directed_graph_{image_name}.png",bbox_inches='tight')

            plt.close('all')

            from matplotlib.colors import ListedColormap
            import seaborn as sns
            m = ooam.astype(int)
            unique_chars, matrix = np.unique(m, return_inverse=True)

            # Create a mask based on the coordinates
            # colour_mask = np.zeros_like(matrix)
            # for coord in diagonal_coords_list:
            #     colour_mask[coord] = True

            vmin = np.min(matrix.reshape(m.shape))
            vmax = np.max(matrix.reshape(m.shape))
            off_diag_mask = np.eye(*matrix.reshape(m.shape).shape, dtype=bool)

            fig = plt.figure()

            color_dict = {1: 'darkred', 0: 'white'}

            plt.figure(figsize=(20,20))
            sns.set(font_scale=2)

            sns.heatmap(matrix.reshape(m.shape), annot=m, mask=~off_diag_mask, cmap=["white", "blue"], vmin=vmin, vmax=vmax, cbar=False)
            ax1 = sns.heatmap(matrix.reshape(m.shape), annot=m, annot_kws={'fontsize': 30}, fmt='',
                            linecolor='dodgerblue', lw=5, square=True, clip_on=False,
                            cmap=ListedColormap([color_dict[char] for char in unique_chars]),
                            xticklabels=np.arange(m.shape[1]) + 1, 
                            yticklabels=np.arange(m.shape[0]) + 1, cbar=False)
            
            
            ax1.tick_params(labelrotation=0)
            ax1.tick_params(axis='both', which='major', labelsize=30, labelbottom = False, bottom=False, top = False, labeltop=True)
            plt.xlabel("Occludee", fontsize=40, weight='bold')
            ax1.xaxis.set_ticks_position('top')
            ax1.xaxis.set_label_position('top')
            plt.ylabel("Occluder", fontsize=40, weight='bold')
            plt.tight_layout()
            plt.savefig(f"{output_dir_img}/ooam_{image_name}.png")
            plt.close('all')








# CUDA_VISIBLE_DEVICES=1 python tools/run_inference.py --dataset-path ./sample_data --output-path ./output_sample --config-file configs/R50_rgbdconcat_mlc_occatmask_hom_concat_modified.yaml --model-weight-path model_0034999.pth
# CUDA_VISIBLE_DEVICES=1 python tools/run_inference.py --dataset-path ./datasets/uoais_tabletop_test/data/mono --output-path ./output_inference --config-file configs/R50_rgbdconcat_mlc_occatmask_hom_concat_modified.yaml --model-weight-path model_0034999.pth
# CUDA_VISIBLE_DEVICES=1 python tools/run_inference.py --dataset-path ./datasets/OSD-0.2-depth --output-path ./output_osd_amodal --config-file configs/R50_rgbdconcat_mlc_occatmask_hom_concat_modified.yaml --model-weight-path model_0039999.pth



# python tools/run_inference.py --dataset-path ./datasets/OSD-0.2-depth --output-path ./inference/inference_uoais_sim_tabletop_OSD_amodal --config-file configs/R50_rgbdconcat_mlc_occatmask_hom_concat_uoais_sim_tabletop.yaml --gpu 1
# python tools/run_inference.py --dataset-path ./datasets/OSD-0.2-depth --output-path ./inference/inference_syntable_OSD_amodal --config-file configs/R50_rgbdconcat_mlc_occatmask_hom_concat_syntable_rerun.yaml --gpu 1
# python tools/run_inference.py --dataset-path ./datasets/syntable/validation/data/mono --output-path ./inference/inference_syntable_syntable_validation --config-file configs/R50_rgbdconcat_mlc_occatmask_hom_concat_syntable_rerun.yaml --gpu 1
# python tools/run_inference.py --dataset-path ./datasets/syntable/validation/data/mono --output-path ./inference/inference_uoais_sim_tabletop_syntable_validation  --config-file configs/R50_rgbdconcat_mlc_occatmask_hom_concat_uoais_sim_tabletop.yaml --gpu 2



# python tools/run_inference.py --dataset-path ./datasets/sample_tabletop_rgbd --output-path ./inference/inference_syntable_sample_tabletop_rgbd --config-file configs/R50_rgbdconcat_mlc_occatmask_hom_concat_syntable_rerun.yaml --gpu 1



# python tools/run_inference_synthetic.py --dataset-path ./datasets/syntable/validation/data/mono --anno-path ./datasets/syntable/validation/uoais_val.json --output-path ./inference/inference_uoais_sim_tabletop_syntable_validation_1 --config-file configs/R50_rgbdconcat_mlc_occatmask_hom_concat_uoais_sim_tabletop.yaml --gpu 2


# python tools/run_inference_synthetic.py --dataset-path ./datasets/UOAIS-Sim/val_tabletop --anno-path ./datasets/UOAIS-Sim/annotations/coco_anns_uoais_sim_val_tabletop.json --output-path ./inference/inference_syntable_uoais_sim_validation --config-file configs/R50_rgbdconcat_mlc_occatmask_hom_concat_syntable_rerun.yaml --gpu 2