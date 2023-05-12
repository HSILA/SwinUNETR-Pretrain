import argparse
import os
import random
import json


parser = argparse.ArgumentParser(description="PyTorch Training")
parser.add_argument(
    "--path",
    default="dataset/dataset0",
    type=str,
    help="path to the images",
)
parser.add_argument(
    "--json",
    default="jsons/dataset0.json",
    type=str,
    help="path to the json output",
)
parser.add_argument(
    "--split",
    action="store_false",
    help="whether to take a validation split or not",
)
parser.add_argument(
    "--ratio",
    default=0.1,
    type=float,
    help="path to the json output",
)
args = parser.parse_args()

def split_list(lst, val_ratio):
    if not 0 < val_ratio < 1:
        raise ValueError("Ratio must be between 0 and 1")
    random.shuffle(lst)
    val_index = int(len(lst) * (1 - val_ratio))

    return lst[:val_index], lst[val_index:]

image_names = os.listdir(args.path)
train_names, val_names = split_list(image_names, args.ratio)

training = []
for image in train_names:
    training.append({"image": f"./{image}"})
validation = []
for image in val_names:
    validation.append({"image": f"./{image}"})
to_json = {"training": training, "validation": validation}

with open(args.json, 'w') as f:
    json.dump(to_json, f)
