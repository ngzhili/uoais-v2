import json
import os
output_dir = "datasets/syntable/train"
# output_dir = "/home/zhili/Documents/uoais/datasets/syntable/validation"
# concatenate all coco.json checkpoint files to final coco.json
final_json_path = f'{output_dir}/uoais_train.json'
with open(final_json_path, 'r') as j:
    data = json.loads(j.read())
    # f = open(final_json_path)
    # data = json.load(f)
    for i in range(len(data["images"])):
        height = data["images"][i]["width"]
        data["images"][i]["width"] = data["images"][i]["height"]
        data["images"][i]["height"] = height

    for i in range(len(data["annotations"])):
        height = data["annotations"][i]["width"]
        data["annotations"][i]["width"] = data["annotations"][i]["height"]
        data["annotations"][i]["height"] = height

# f.close()

# import collections
# print([item for item, count in collections.Counter(arr).items() if count > 1])

with open(final_json_path, 'w') as write_file:
    json.dump(data, write_file, indent=4)