import glob
import json
import albumentations as A
import numpy as np
import pandas as pd
from albumentations.pytorch import ToTensorV2

from inception_classifier import EfficientNet_V2

DEBUG = True

import os

import torch.onnx
import cv2
def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()
models = []

model1 = EfficientNet_V2(2) # timm.create_model( 'tf_efficientnetv2_l.in21k_ft_in1k', checkpoint_path = 'weights_class0.pth', pretrained=True, num_classes=2) # D:\\hacks\\train_data_minprirodi\\weights_class0.pth
model1.load_state_dict(torch.load("weights_class0.pth", map_location=torch.device('cpu')))

def get_crop(image: np.array, x: float, y: float, w: float, h: float):
    hg, wg, _ = image.shape
    #x, y, w, h = int(x * wg), int(y * hg), int(w * wg), int(h * hg)
    print(image.shape, x, y, w, h)
    return cv2.resize(image.copy()[y-h+1//2:y+h//2, x-w+1//2:x+w//2], (512, 512), interpolation = cv2.INTER_AREA)

def all_crops(images_path: str, annot_path: str):
    annot = pd.read_csv(annot_path)
    crops = []
    target = []
    for idx, row in annot.iterrows():
        temp_img = cv2.imread(images_path + row["Name"])
        # print(temp_img, *map(float, row["Bbox"].split(",")), images_path + row["Name"])
        crop = get_crop(temp_img, *map(float, row["Bbox"].split(",")))
        crops.append(crop)
        target.append(row["Class"])
    return np.array(crops), np.array(target)


def transform_valid():
    transform = [
        A.Resize(512,512,p=1),
        ToTensorV2(p=1.0)
    ]
    return A.Compose(transform)



lines = ["Name,Bbox,Class\n"]

with open("testdata_detected_animals.json", "r") as read_file:
    data = json.load(read_file)

for annot in data['annotations']: # D:\\test_data\\* "test_data/*"


    print(annot["img_id"])
    image = cv2.imread(annot["img_id"])
    bboxes = annot["bbox"]
    crops = []

    for idx in range(len(bboxes)):
        if annot["confidence"][idx] < 0.5:
            continue
        x1, y1, x2, y2 = bboxes[idx][0], bboxes[idx][1], bboxes[idx][2], bboxes[idx][3]
        w = x2 - x1
        h = y2 - y1
        xc = (x1 + x2) // 2
        yc = (y1 + y2) // 2
        ans_boxes = [xc / image.shape[1], yc / image.shape[0], w / image.shape[1], h / image.shape[0]]
        new_line = annot["img_id"].split("/")[-1] + "," + "\"" + ",".join([str(el) for el in ans_boxes]) + "\""
        print(x1, y1, x2, y2)
        try:
            crop = cv2.resize(image.copy()[y2:y1, x2:x1], (512, 512),
                              interpolation=cv2.INTER_AREA)
        except:
            continue
        crops.append(crop)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        var = variance_of_laplacian(gray)
        im = transform_valid()(image=crop.astype(np.float32))['image']
        print(im.shape)
        pred = (model1(im.unsqueeze(0)))
        if var < 70:
            new_line = new_line + "0"
            continue
        new_line = new_line + "," + str(int(torch.argmax(pred, 1))) + "\n"
        print(new_line)
        if DEBUG: print(new_line)
        lines.append(new_line)

f = open("ans.csv", 'w+')
f.writelines(lines)