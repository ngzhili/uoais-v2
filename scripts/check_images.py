import json
import os
output_dir = "/home/ngzhili/uoais/datasets/syntable/train"
# output_dir = "/home/zhili/Documents/uoais/datasets/syntable/validation"
# concatenate all coco.json checkpoint files to final coco.json
final_json_path = f'{output_dir}/uoais_train.json'
image_set = set()
img_set = set()
arr= []
# concatenate all coco.json checkpoint files to final coco.json
final_json_path = f'{output_dir}/uoais_train.json'
json_files = [os.path.join(output_dir,pos_json) for pos_json in os.listdir(output_dir) if (pos_json.endswith('.json') and os.path.join(output_dir,pos_json) != final_json_path)]
json_files = sorted(json_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
last_index = 0

import glob
cv_img = []
img_list = [img.split('/')[-1] for img in glob.glob("/home/ngzhili/uoais/datasets/syntable/train/data/mono/rgb/*.png")]
img_list = set(img_list)
# print(len(img_list))
# coco_json = {"info":{},"licenses":[],"categories":[],"images":[],"annotations":[]}
for i, file in enumerate(json_files):
    if file != final_json_path:
        # print(file)
        f = open(file)
        data = json.load(f)
        # print(data["images"][0]["id"])   
        for i in range(len(data["images"])):      
            # if data["images"][i]["file_name"] in image_set:
            #     print(file)
            image_set.add(data["images"][i]["file_name"].split('/')[-1])
            
            # arr.append(data["images"][i]["file_name"])
        
        # first_index = data["images"][0]["file_name"].split('/')[-1].strip('.png').split('_')
        # last_index = data["images"][i]["file_name"].split('/')[-1].strip('.png').split('_')
        # print(first_index)
        # print(last_index)

        # check = int(last_index + 1) == int(data["images"][0]["id"])
        # if not check:
        #     print(file)
        # last_index = data["images"][i]["id"]

        # print(int(last_index + 1))
        # print(int(data["images"][0]["id"]))
        # 
        # if i == 0:
        #     coco_json["info"] = data["info"]
        #     coco_json["licenses"] = data["licenses"]
        #     coco_json["categories"] = data["categories"]
        
        # coco_json["images"].extend(data["images"])
        # coco_json["annotations"].extend(data["annotations"])
        f.close()

# with open(final_json_path, 'w') as write_file:
#     json.dump(coco_json, write_file, indent=4)


# with open(final_json_path, 'r') as j:
#     data = json.loads(j.read())
#     # f = open(final_json_path)
#     # data = json.load(f)
#     for i in range(len(data["images"])):
#         # height = data["images"][i]["width"]
#         # data["images"][i]["width"] = data["images"][i]["height"]
#         # data["images"][i]["height"] = height
#         image_set.add(data["images"][i]["file_name"])

#     for i in range(len(data["annotations"])):
#         img_set.add(data["annotations"][i]["image_id"])
# f.close()
print(len(image_set))
print(len(img_list))
print(img_list-image_set)
# print(list(image_set)[0])
# print(list(img_list)[0])
print(len(img_list-image_set))
# print(len(img_set))
import collections
# duplicates = [item for item, count in collections.Counter(arr).items() if count > 1]
# print(len(duplicates))


# with open(final_json_path, 'w') as write_file:
#     json.dump(data, write_file, indent=4)