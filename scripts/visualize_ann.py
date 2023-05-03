# Python program to read
# json file
  
  
import json
import cv2
import numpy as np

# Opening JSON file
f = open('datasets/UOAIS-Sim/annotations/coco_anns_uoais_sim_val.json')


# returns JSON object as 
# a dictionary
data = json.load(f)
  
# Iterating through the json
# list
print(data.keys())

print(data['info'])

print(data['licenses'])
    
print(data['categories'])

#print(data['images'][0])

#print(data['annotations'][0])
# Closing file
f.close()
# print(data['images'])

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


def segmToMask(segm, img_size):
    rle = segmToRLE(segm, img_size)
    m = maskUtils.decode(rle)
    return m


#print(type(data['annotations'][0]["segmentation"]["counts"]))

#img_id = data['images'][0]["id"]
#print(data['images'][0]["file_name"])


output_dir = "/home/ngzhili/uoais/testing/visualize_uoais_sim/val"

query_img_id = [49,53]
import pycocotools.mask as mask_util
for ann in data['annotations']:
    image_id = ann["image_id"]
    if image_id < query_img_id[0]:
        continue

    if image_id > query_img_id[1]:
        break
    
    ann_id = ann["id"]
    # print(image_id)
    segm = ann["segmentation"]
    mask = mask_util.decode(segm)
    # print(mask)
    cv2.imwrite(f"{output_dir}/segmentations/{image_id}_{ann_id}.png", mask*255)

    segm = ann["visible_mask"]
    mask = mask_util.decode(segm)
    cv2.imwrite(f"{output_dir}/visible_masks/{image_id}_{ann_id}.png", mask*255)

    segm = ann["occluded_mask"]
    mask = mask_util.decode(segm)
    cv2.imwrite(f"{output_dir}/occluded_masks/{image_id}_{ann_id}.png", mask*255)


print("[INFO] Visualising masks...")
import glob
import cv2
import numpy as np
import os

dataset_dir = "/home/ngzhili/uoais/datasets/UOAIS-Sim/val"
occlusion_dir = f"{output_dir}/occluded_masks"
visible_dir = f"{output_dir}/visible_masks"

# unique_images = []

# for ann in data['images']:
#     image_id = ann["id"]
#     if image_id not in unique_images:
#         unique_images.append(image_id)
#     # else:
#     #     break

# print(unique_images)

for ann in data['images']:

    image_id = ann["id"]
    if image_id < query_img_id[0]:
        continue
    if image_id > query_img_id[1]:
        break
    img_filename = ann["file_name"]
    img_path = os.path.join(dataset_dir,img_filename)

    rgb_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    occ_img = rgb_img.copy()
    vis_img = rgb_img.copy()

    occ_path_list = glob.glob(f"{occlusion_dir}/{image_id}_*.png")
    vis_path_list = glob.glob(f"{visible_dir}/{image_id}_*.png")

    if len(occ_path_list) > 0:
        for i in range(len(occ_path_list)):
            occ_path = occ_path_list[i]
            occluded_mask = cv2.imread(occ_path, cv2.IMREAD_UNCHANGED)
            # visualize occlusion masks on rgb
            red = np.ones(occluded_mask.shape)
            red = red*255
            occ_img[:,:,0][occluded_mask>0] = red[occluded_mask>0]

    if len(vis_path_list) > 0:
        for i in range(len(vis_path_list)):
            vis_path = vis_path_list[i]
            visible_mask = cv2.imread(vis_path, cv2.IMREAD_UNCHANGED)
            # visualize occlusion masks on rgb
            
            #red = np.ones(visible_mask.shape)

            #ask = visible_mask * 255
            #overlay = cv2.merge((mask, mask, mask))

            # print(mask.shape)
            overlay= np.stack((mask,)*3, -1)
            # print(overlay.shape)
            #overlay = np.stack((mask)*3, -1)
            #overlay[mask==255]=(255, 255, 0)
            rows, cols = np.where(overlay[:,:,1]==255)
            # print(rows,cols)
            colour = np.array([0,255,0])     # green
            overlay[rows,cols,:] = colour
            #print(overlay.shape)
            #print(overlay)
            alpha =0.5
            vis_img = cv2.addWeighted(vis_img,alpha,overlay,1 - alpha,0)


            #red = red*255

            #vis_img[:,:,0][visible_mask>0] = red[visible_mask>0]
    
    save_path = f"{output_dir}/occ_combined/occ_{image_id}.png"
    cv2.imwrite(save_path,occ_img)

    save_path = f"{output_dir}/vis_combined/vis_{image_id}.png"
    cv2.imwrite(save_path,vis_img)

    # save_path = f"{output_dir}/overlay.png"
    # cv2.imwrite(save_path,overlay)