import os
import cv2
img_dir_path = f"/home/ngzhili/uoais/datasets/UOAIS-Sim/val/tabletop/depth"
img_dir_path = f"/home/ngzhili/uoais/datasets/syntable/validation/data/mono/depth"
img_list = os.listdir(img_dir_path)
img_list = img_list[:100]
# print(img_list)
# cols = 4
# rows = len(img_list) //cols + 1
# print(len(img_list))
# print(cols,rows)
# import numpy as np
# np.set_printoptions(threshold=np.inf)
from matplotlib import pyplot as plt

for i in range(len(img_list)):
    img_filename = img_list[i]
    # print(img_filename)
    img_path = os.path.join(img_dir_path,img_filename)
    image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert to RGB   
    # print(image)
    # min_val = 250
    # max_val = 2000

    # image[image < min_val] = min_val
    # image[image > max_val] = max_val

    # depth = (depth - min_val) / (max_val - min_val) * 255
    # depth = np.expand_dims(depth, -1)
    # depth = np.uint8(np.repeat(depth, 3, -1))
    plt.figure()
    plt.imshow(image)
    plt.colorbar(label='Distance to Camera')
    plt.title(f'{img_filename} Depth image')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')
    # plt.show()
    # plt.savefig(f"/home/ngzhili/uoais/datasets/UOAIS-Sim/depth_viz/{img_filename}_depth_viz.png")
    plt.savefig(f"/home/ngzhili/uoais/datasets/syntable/depth_viz/{img_filename}_depth_viz.png")
    plt.close()
        
