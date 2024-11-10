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
from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2 as cv2


def get_res_image(weights_path="groundingdino_swint_ogc.pth", image_path="1833377.jpg"):

    HOME = os.getcwd()

    CONFIG_PATH = os.path.join(HOME, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    WEIGHTS_NAME = weights_path
    WEIGHTS_PATH = "D:\\hacks\\train_data_minprirodi\\" + weights_path
    model = load_model(CONFIG_PATH, WEIGHTS_PATH)
    IMAGE_NAME = image_path
    IMAGE_PATH = os.path.join(HOME, IMAGE_NAME)

    TEXT_PROMPT = "animal or wild animal"
    BOX_TRESHOLD = 0.50
    TEXT_TRESHOLD = 0.45

    image_source, image = load_image(IMAGE_PATH)

    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD,
        device="cpu"
    )

    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)


    cut = []
    good_bboxes = []

    for box in boxes:
        print(box, image_source.shape)
        hg, wg, _ = image_source.shape
        x,y,w,h = int(wg * box[0]), int(hg * box[1]), int(wg * box[2]), int(hg * box[3])
        good_bboxes.append([x,y,w,h])
        cut.append(image_source[y-h//2:y+h//2, x-w//2:x+w//2])



    for i in range(len(cut)):
        cv2.imwrite(f"bbox{i+1}.jpg", cut[i])

    return boxes, annotated_frame.view(), image_source, good_bboxes

def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()
models = []

model1 = EfficientNet_V2(2) # timm.create_model( 'tf_efficientnetv2_l.in21k_ft_in1k', checkpoint_path = 'weights_class0.pth', pretrained=True, num_classes=2) # D:\\hacks\\train_data_minprirodi\\weights_class0.pth
model1.load_state_dict(torch.load("D:\\hacks\\train_data_minprirodi\\weights_class0.pth", map_location=torch.device('cpu')))

def get_crop(image: np.array, x: float, y: float, w: float, h: float):
    hg, wg, _ = image.shape
    #x, y, w, h = int(x * wg), int(y * hg), int(w * wg), int(h * hg)
    img = cv2.resize(image[y-h//2:y+h//2, x-w//2:x+w//2], (512, 512), interpolation = cv2.INTER_AREA)
    return img

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

with open("data_file.json", "w") as write_file:
    json.dump(data, write_file)

for image in glob.glob("D:\\test_data\\*"): # D:\\test_data\\* "test_data/*"


    ans_boxes, _, img, bboxes = get_res_image(image_path=image)
    img = np.array(img)
    crops = []

    for idx in range(len(bboxes)):
        new_line = image.split("/")[0] + "," + "\"" + ",".join([str(el) for el in ans_boxes[idx]]) + "\""
        crop = get_crop(img, *bboxes[idx])
        crops.append(crop)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        var = variance_of_laplacian(gray)
        im = transform_valid()(image=crop.astype(np.float32))['image']
        print(im.shape)
        pred = (model1(im.unsqueeze(0)))
        if var < 70:
            new_line = new_line + "0"
            continue
        new_line = new_line + str(torch.argmax(pred, 1)) + "\n"
        print(new_line)
        if DEBUG: print(new_line)
        lines.append(new_line)

f = open("ans.csv", 'w+')
f.writelines(lines)