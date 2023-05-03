import json
import cv2
import os
import numpy as np
import argparse
import glob
from PIL import Image
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import networkx as nx

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

def compute_occluded_masks(mask1, mask2):
    """Computes occlusions between two sets of masks.
    masks1, masks2: [Height, Width, instances]
    """
    # intersections and union
    mask1_area = np.count_nonzero( mask1 )
    mask2_area = np.count_nonzero( mask2 )
    intersection_mask = np.logical_and( mask1, mask2 )
    intersection = np.count_nonzero( np.logical_and( mask1, mask2 ) )
    iou = intersection/(mask1_area+mask2_area-intersection)

    return iou, intersection_mask.astype(float)

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


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/home/ngzhili/uoais/datasets/iccv2",
        help="path to the dataset"
    )
    parser.add_argument(
        "--annotation-path",
        type=str,
        default="annotation_final.json",
        help="path to the annotations"
    )
    return parser

# python /home/ngzhili/uoais/datasets/visualise_annotations.py 

if __name__ == "__main__":

    args = get_parser().parse_args()

    # data_dir = r'/home/ngzhili/uoais/datasets/syntable/train'
    # r'/home/ngzhili/uoais/datasets/iccv2'
    # annotation_json = 'uoais_train_10.json'
    # annotation_json = 'annotation_final.json'
    data_dir = args.dataset_path 
    annotation_json = args.annotation_path
    query_img_id_list = [1]
    query_img_id_list = [33*50 + 26,33*50 + 28] # all images

    # Opening JSON file
    f = open(os.path.join(data_dir,annotation_json))
    # returns JSON object as a dictionary
    data = json.load(f)
    # Iterating through the json list
    print(data.keys())
    print(data['info'])
    print(data['licenses'])
    print(data['categories'])
    #print(data['images'][0])
    #print(data['annotations'][0])
    # Closing file
    f.close()

    def rle2mask(mask_rle, shape=(480,640)):
        '''
        mask_rle: run-length as string formated (start length)
        shape: (width,height) of array to return 
        Returns numpy array, 1 - mask, 0 - background

        '''
        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        return img.reshape(shape).T

    def segmToRLE(segm, img_size):
        h, w = img_size
        if type(segm) == list:
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, h, w)
            rle = maskUtils.merge(rles)
        elif type(segm["counts"]) == list:
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, h, w)
        else:
            # rle
            rle = segm
        return rle

    # Convert 1-channel groundtruth data to visualization image data
    def normalize_greyscale_image(image_data):
        # print(image_data)
        # print(np.unique(image_data))
        image_data = np.reciprocal(image_data)
        image_data[image_data == 0.0] = 1e-5
        image_data = np.clip(image_data, 0, 255)
        image_data -= np.min(image_data)
        if np.max(image_data) > 0:
            image_data /= np.max(image_data)
        # print(image_data)
        # print(np.unique(image_data))
        image_data *= 255
        image_data = image_data.astype(np.uint8)
        return image_data


    referenceDict = {}
    import pycocotools.mask as mask_util
    for i, ann in enumerate(data['annotations']):
        image_id = ann["image_id"]
        ann_id = ann["id"]

        # print(ann_id)
        if image_id not in referenceDict:
            referenceDict.update({image_id:{"rgb":None,"depth":None, "amodal":[], "visible":[],
    "occluded":[],"occluded_rate":[],"category_id":[],"object_name":[],"bbox":[]}})
            # print(referenceDict)
            referenceDict[image_id].update({"rgb":data["images"][i]["file_name"]})
            referenceDict[image_id].update({"depth":data["images"][i]["depth_file_name"]})
            referenceDict[image_id].update({"occlusion_order":data["images"][i]["occlusion_order_file_name"]})
            referenceDict[image_id]["amodal"].append(ann["segmentation"])
            referenceDict[image_id]["visible"].append(ann["visible_mask"])
            referenceDict[image_id]["occluded"].append(ann["occluded_mask"])
            referenceDict[image_id]["occluded_rate"].append(ann["occluded_rate"])
            referenceDict[image_id]["category_id"].append(ann["category_id"])
            referenceDict[image_id]["object_name"].append(ann["object_name"])
            referenceDict[image_id]["bbox"].append(ann["bbox"])

        else:
            # if not (referenceDict[image_id]["rgb"] or referenceDict[image_id]["depth"]):
            #     referenceDict[image_id].update({"rgb":data["images"][i]["file_name"]})
            #     referenceDict[image_id].update({"depth":data["images"][i]["depth_file_name"]})
            referenceDict[image_id]["amodal"].append(ann["segmentation"])
            referenceDict[image_id]["visible"].append(ann["visible_mask"])
            referenceDict[image_id]["occluded"].append(ann["occluded_mask"])
            referenceDict[image_id]["occluded_rate"].append(ann["occluded_rate"])
            referenceDict[image_id]["category_id"].append(ann["category_id"])
            referenceDict[image_id]["object_name"].append(ann["object_name"])
            referenceDict[image_id]["bbox"].append(ann["bbox"])

    import os, shutil
    vis_dir = os.path.join(data_dir,"visualise_dataset")
    if os.path.exists(vis_dir): # remove contents if exist
        for filename in os.listdir(vis_dir):
            file_path = os.path.join(vis_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
    else:
        os.makedirs(vis_dir)

    def get_bbox(mask):
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            return [0, 0, 0, 0]
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        return [int(x_min), int(y_min), int(x_max-x_min), int(y_max-y_min)]

    """ user input """
    if len(query_img_id_list) >0:
        pass
    else:
        query_img_id_list = [i for i in range(1,len(referenceDict)+1)] # visualise all images

    """ Generate Mask,bbox visualisations """
    for id in query_img_id_list:
        if id in referenceDict:
            ann_dic = referenceDict[id]
            vis_dir_img = os.path.join(vis_dir,str(id))
            if not os.path.exists(vis_dir_img):
                os.makedirs(vis_dir_img)
            if not os.path.exists(os.path.join(vis_dir_img,'visible_mask')):
                os.makedirs(os.path.join(vis_dir_img,'visible_mask'))
            if not os.path.exists(os.path.join(vis_dir_img,'amodal_mask')):
                os.makedirs(os.path.join(vis_dir_img,'amodal_mask'))

            rgb_path = os.path.join(data_dir,ann_dic["rgb"])
            rgb_img = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
            rgb_save_path = f"{vis_dir_img}/rgb_{id}.png"
            cv2.imwrite(rgb_save_path, rgb_img)
            # visualise depth image
            depth_path = os.path.join(data_dir,ann_dic["depth"])
            from PIL import Image
            im = Image.open(depth_path)
            im = np.array(im)
            depth_img = Image.fromarray(normalize_greyscale_image(im.astype("float32")))
            file = os.path.join(vis_dir_img,f"depth_{id}.png")
            depth_img.save(file, "PNG")

            # visualise occlusion masks on rgb image
            occ_img_list = ann_dic["occluded"]
            if len(occ_img_list) > 0:
                occ_img = rgb_img.copy()
                overlay = rgb_img.copy()
                combined_mask = np.zeros((occ_img.shape[0],occ_img.shape[1]))
                # iterate through all occlusion masks
                for i, occMask in enumerate(occ_img_list):
                    occluded_mask = mask_util.decode(occMask)
                    if ann_dic["category_id"][i] == 0:
                        occ_img_back = rgb_img.copy()
                        overlay_back = rgb_img.copy()
                        occluded_mask = occluded_mask.astype(bool) # boolean mask
                        overlay_back[occluded_mask] = [0, 0, 255]
                        # print(np.unique(occluded_mask))
                        alpha =0.5                  
                        occ_img_back = cv2.addWeighted(overlay_back, alpha, occ_img_back, 1 - alpha, 0, occ_img_back)      

                        occ_save_path = f"{vis_dir_img}/rgb_occlusion_{id}_background.png"
                        cv2.imwrite(occ_save_path, occ_img_back)
                    else:
                        combined_mask += occluded_mask

                combined_mask = combined_mask.astype(bool) # boolean mask
                overlay[combined_mask] = [0, 0, 255]
                
                alpha =0.5                  
                occ_img = cv2.addWeighted(overlay, alpha, occ_img, 1 - alpha, 0, occ_img)      

                occ_save_path = f"{vis_dir_img}/rgb_occlusion_{id}.png"
                cv2.imwrite(occ_save_path, occ_img)

                combined_mask = combined_mask.astype('uint8')
                occ_save_path = f"{vis_dir_img}/occlusion_mask_{id}.png"
                cv2.imwrite(occ_save_path, combined_mask*255)


                cols = 4
                rows = len(occ_img_list) // cols + 1
                from matplotlib import pyplot as plt
                fig = plt.figure(figsize=(20,10))
                for index, occMask in enumerate(occ_img_list):
                    occ_mask = mask_util.decode(occMask)
                    plt.subplot(rows,cols, index+1)
                    plt.axis('off')
                    plt.title(ann_dic["object_name"][index])
                    plt.imshow(occ_mask)

                plt.tight_layout()
                plt.suptitle(f"Occlusion Masks for {id}.png")      
                plt.savefig(f'{vis_dir_img}/occ_masks_{id}.png')
                plt.close()

            #  visualise visible masks on rgb image
            vis_img_list = ann_dic["visible"]
            amodal_img_list =ann_dic["amodal"]
            if len(vis_img_list) > 0:
                vis_img = rgb_img.copy()
                bbox_vis_img = rgb_img.copy()
                amodal_boundary_img = rgb_img.copy()
                occ_boundary_img = rgb_img.copy()

                # num_colors = np.max(vis_img_list) + 1
                # num_channels = 3
                # color_pixels = random_colours(num_colors, True, num_channels)
                # color_pixels = [[color_pixel[i] * 255 for i in range(num_channels)] for color_pixel in color_pixels]

                color_image = np.zeros(rgb_img.shape, dtype=np.uint8)
                overlay = rgb_img.copy()

                # Create empty color image
                vis_color_mask = np.zeros((rgb_img.shape[0], rgb_img.shape[1], 3), dtype=np.uint8)

                colour_list = []
                for _ in range(len(vis_img_list)):
                    colour = list(np.random.choice(range(256), size=3))
                    colour_list.append(colour)

                # iterate through all occlusion masks
                for i, visMask in enumerate(vis_img_list):
                    amodal_mask = mask_util.decode(amodal_img_list[i])
                    visible_mask =  mask_util.decode(visMask)
                    occlusion_mask = mask_util.decode(occ_img_list[i])
                    bbox = get_bbox(visible_mask)
                    # background
                    if ann_dic["category_id"][i] == 0:
                        vis_img_back = rgb_img.copy()
                        overlay_back = rgb_img.copy()
                        visible_mask = visible_mask.astype(bool) # boolean mask
                        overlay_back[visible_mask] = [0, 0, 255]
                        
                        alpha =0.5                  
                        vis_img_back = cv2.addWeighted(overlay_back, alpha, vis_img_back, 1 - alpha, 0, vis_img_back)      

                        vis_save_path = f"{vis_dir_img}/rgb_visible_mask_{id}_background.png"
                        cv2.imwrite(vis_save_path, vis_img_back)
                    # any object
                    else:
                        vis_combined_mask = visible_mask.astype(bool) # boolean mask      
                        # colour = list(np.random.choice(range(256), size=3))
                        overlay[vis_combined_mask] = colour_list[i]

                        # draw bbox
                        x, y, w, h = list(map(int, bbox))
                        colour_tuple = tuple(int(c) for c in colour_list[i])
                        # draw bbox rectangle
                        cv2.rectangle(bbox_vis_img, (x, y), (x + w, y + h), colour_tuple , 2)

                        obj_id = str(i+1)
                        # Finds space required by the text so that we can put a background with that amount of width.
                        (w_text, h1), _ = cv2.getTextSize(obj_id, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)

                        # Prints the text.    
                        bbox_vis_img = cv2.rectangle(bbox_vis_img, (x, y - 16), (x + w_text, y), colour_tuple, -1)
                        bbox_vis_img = cv2.putText(bbox_vis_img, obj_id, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

                        # fill instance segmentation mask
                        def find_contours(mask):
                            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                            polygons = []

                            for object in contours:
                                coords = []
                                
                                for point in object:
                                    coords_point = []
                                    coords_point.append(int(point[0][0]))
                                    coords_point.append(int(point[0][1]))
                                    coords.append(coords_point)
                                polygons.append(coords)
                            return polygons
                            # cv2.rectangle(bbox_vis_img, (x, y), (x + w, y + h), (255,0,0), 1)

                        # draw amodal polygons
                        polygons = find_contours(amodal_mask)
                        # Using cv2.polylines() method
                        for pts in polygons:
                            pts = np.array(pts)
                            amodal_boundary_img = cv2.polylines(amodal_boundary_img, [pts],
                                                isClosed = True, color=colour_tuple, thickness=2)
                        # draw occlusion polygons
                        occ_polygons = find_contours(occlusion_mask)
                        # Using cv2.polylines() method
                        for pts in occ_polygons:
                            pts = np.array(pts)
                            occ_boundary_img = cv2.polylines(occ_boundary_img, [pts],
                                                isClosed = True, color=colour_tuple, thickness=2)
                        
                        # Assign color to combined colour mask
                        vis_color_mask[visible_mask > 0] = colour_list[i] #(255, 0, 0)  # Blue color for mask pixels

                        # save instance colour mask
                        color_mask_instance = np.zeros((rgb_img.shape[0], rgb_img.shape[1], 3), dtype=np.uint8)
                        color_mask_instance[visible_mask > 0] = colour_list[i]
                        vis_instance_save_path = f"{vis_dir_img}/visible_mask/visible_mask_{id}_{i}.png"
                        cv2.imwrite(vis_instance_save_path,color_mask_instance)

                        amodal_color_mask_instance = np.zeros((rgb_img.shape[0], rgb_img.shape[1], 3), dtype=np.uint8)
                        amodal_color_mask_instance[amodal_mask > 0] = colour_list[i]
                        amodal_instance_save_path = f"{vis_dir_img}/amodal_mask/amodal_mask_{id}_{i}.png"
                        cv2.imwrite(amodal_instance_save_path,amodal_color_mask_instance)

                alpha = 0.5   
                vis_img = cv2.addWeighted(overlay, alpha, vis_img, 1 - alpha, 0, vis_img)        
                vis_save_path = f"{vis_dir_img}/rgb_visible_mask_{id}.png"
                cv2.imwrite(vis_save_path,vis_img)

                vis_bbox_save_path = f"{vis_dir_img}/rgb_visible_bbox_{id}.png"
                cv2.imwrite(vis_bbox_save_path, bbox_vis_img)
                # plt.imshow(bbox_vis_img)
                # plt.show()
                amodal_save_path = f"{vis_dir_img}/rgb_amodal_contour_{id}.png"
                cv2.imwrite(amodal_save_path,amodal_boundary_img)

                occlusion_save_path = f"{vis_dir_img}/rgb_occlusion_contour_{id}.png"
                cv2.imwrite(occlusion_save_path,occ_boundary_img)

                alpha = 0.5   
                vis_mask_occ_boundary_img = cv2.addWeighted(overlay, alpha, occ_boundary_img, 1 - alpha, 0, vis_img)        
                vis_mask_occ_boundary_save_path = f"{vis_dir_img}/rgb_visible_mask_occlusion_boundary_{id}.png"
                cv2.imwrite(vis_mask_occ_boundary_save_path,vis_mask_occ_boundary_img)


                vis_segmentation_save_path = f"{vis_dir_img}/visible_mask/visible_mask_{id}_{i}.png"
                cv2.imwrite(vis_segmentation_save_path,color_mask_instance)

                cols = 4
                rows = len(vis_img_list) // cols + 1
                # print(len(amodal_img_list))
                # print(cols,rows)
                from matplotlib import pyplot as plt
                fig = plt.figure(figsize=(20,10))
                for index, visMask in enumerate(vis_img_list):
                    vis_mask = mask_util.decode(visMask)
                    plt.subplot(rows,cols, index+1)
                    plt.axis('off')
                    plt.title(ann_dic["object_name"][index])
                    plt.imshow(vis_mask)

                plt.tight_layout()
                plt.suptitle(f"Visible Masks for {id}.png")      
                plt.savefig(f'{vis_dir_img}/vis_masks_{id}.png')
                plt.close()

            amodal_img_list = ann_dic["amodal"]
            if len(amodal_img_list) > 0:
                cols = 4
                rows = len(amodal_img_list) // cols + 1
                # print(len(amodal_img_list))
                # print(cols,rows)
                from matplotlib import pyplot as plt
                fig = plt.figure(figsize=(20,10))
                for index, amoMask in enumerate(amodal_img_list):
                    amodal_mask = mask_util.decode(amoMask)
                    plt.subplot(rows,cols, index+1)
                    plt.axis('off')
                    plt.title(ann_dic["object_name"][index])
                    plt.imshow(amodal_mask)

                plt.tight_layout()
                plt.suptitle(f"Amodal Masks for {id}.png")
                # plt.show()        
                plt.savefig(f'{vis_dir_img}/amodal_masks_{id}.png')
                plt.close()     
            

            """ Generate OOAM, OODAG """
            # query_img_id_list = [i for i in range(1,len(referenceDict)+1)]
            # query_img_id_list = [1,8,9]

            # for id in query_img_id_list:
            #     if id in referenceDict:
            # ann_dic = referenceDict[id]
            # vis_dir_img = os.path.join(vis_dir,str(id))
            # if not os.path.exists(vis_dir_img):
            #     os.makedirs(vis_dir_img)
            
            # get rgb_path
            rgb_path = os.path.join(data_dir,ann_dic["rgb"])
            rgb_img = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
            rgb_img = rgb_img[...,::-1]

            # get occlusion order adjacency matrix
            # npy_path = os.path.join(data_dir,ann_dic["occlusion_order"])
            # ooam = np.load(npy_path)

            """ === Generate OOAM === """
            vis_img_list = ann_dic["visible"]
            occ_img_list =ann_dic["occluded"]
            # generate occlusion ordering for current viewport
            rows = cols = len(vis_img_list)
            ooam = np.zeros((rows,cols))
            
            # A(i,j), col j, row i. row i --> col j
            for i in range(0,len(vis_img_list)):
                visible_mask_i =  mask_util.decode(vis_img_list[i]) # occluder
                for j in range(0,len(vis_img_list)):
                    if j != i:
                        occluded_mask_j = mask_util.decode(occ_img_list[j]) # occludee
                        iou, _ = compute_occluded_masks(visible_mask_i,occluded_mask_j) 
                        if iou > 0: # object i's visible mask is overlapping object j's occluded mask
                            ooam[i][j] = 1

            print(f"\nCalculating Directed Graph for Image:{id}")
            # vis_img = cv2.imread(f"{vis_dir}/visuals/{id}.png", cv2.IMREAD_UNCHANGED)
            rows = cols = len(ann_dic["visible"]) # number of objects
            obj_rgb_mask_list = []
            for i in range(1,len(ann_dic["visible"])+1):
                visMask = ann_dic["visible"][i-1]
                visible_mask = mask_util.decode(visMask)
                
                rgb_crop = apply_mask(rgb_img, visible_mask)
                rgb_crop = convert_png(rgb_crop)
                
                def bbox(im):
                    a = np.array(im)[:,:,:3]  # keep RGB only
                    m = np.any(a != [0,0,0], axis=2)
                    coords = np.argwhere(m)
                    y0, x0, y1, x1 = *np.min(coords, axis=0), *np.max(coords, axis=0)
                    return (x0, y0, x1+1, y1+1)

                # print(bbox(rgb_crop))
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
            
            pos=nx.planar_layout(G)
            print("Nodes:",G.nodes())
            print("Edges:",G.edges())

            # get start nodes
            start_nodes = [node for (node,degree) in G.in_degree if degree == 0]
            print("start_nodes:",start_nodes)

            # get end nodes
            end_nodes = [node for (node,degree) in G.out_degree if degree == 0]
            for node in end_nodes:
                if node in start_nodes:
                    end_nodes.remove(node)
            print("end_nodes:",end_nodes)

            # get intermediate notes
            intermediate_nodes = [i for i in nodes_list if i not in (start_nodes) and i not in (end_nodes)]
            print("intermediate_nodes:",intermediate_nodes)
            print("(Degree of clustering) Number of Weakly Connected Components:",nx.number_weakly_connected_components(G))

            wcc_list = list(nx.weakly_connected_components(G))
            wcc_len = []
            for component in wcc_list:
                wcc_len.append(len(component))
            print("(Scene Complexity/Degree of overlapping regions) Sizes of Weakly Connected Components:",wcc_len)
            if not nx.is_directed_acyclic_graph(G): #not G.is_directed():
                print("Graph is not directed and contains a cycle!")
            else:
                dag_longest_path_length = nx.dag_longest_path_length(G)
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
            nx.draw_planar(G,  with_labels = True, arrows=True, **options)

            dag = nx.is_directed_acyclic_graph(G)
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
            plt.savefig(f"{vis_dir_img}/object_index_directed_graph_{id}.png",bbox_inches='tight')
            # cv2.imwrite(f"{output_dir}/scene_{id}.png", vis_img)
            # plt.show()
            plt.close()

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
            pos = nx.planar_layout(G)

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

            plt.savefig(f"{vis_dir_img}/object_rgb_directed_graph_{id}.png",bbox_inches='tight')

            plt.close()

            from matplotlib.colors import ListedColormap
            import seaborn as sns
            m = ooam.astype(int)
            unique_chars, matrix = np.unique(m, return_inverse=True)
            color_dict = {1: 'darkred', 0: 'white'}
            plt.figure(figsize=(20,20))
            sns.set(font_scale=2)
            ax1 = sns.heatmap(matrix.reshape(m.shape), annot=m, annot_kws={'fontsize': 30}, fmt='',
                            linecolor='dodgerblue', lw=5, square=True, clip_on=False,
                            cmap=ListedColormap([color_dict[char] for char in unique_chars]),
                            xticklabels=np.arange(m.shape[1]) + 1, yticklabels=np.arange(m.shape[0]) + 1, cbar=False)
            ax1.tick_params(labelrotation=0)
            ax1.tick_params(axis='both', which='major', labelsize=30, labelbottom = False, bottom=False, top = False, labeltop=True)
            plt.xlabel("Occludee", fontsize=40, weight='bold')
            ax1.xaxis.set_ticks_position('top')
            ax1.xaxis.set_label_position('top')
            plt.ylabel("Occluder", fontsize=40, weight='bold')
            plt.tight_layout()
            plt.savefig(f"{vis_dir_img}/ooam_{id}.png")
            plt.close()



