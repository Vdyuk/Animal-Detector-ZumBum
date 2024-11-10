import cv2
import pandas as pd
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
import albumentations as A
from inception_classifier import EfficientNet_V2

chkpt = torch.load("weights_class2.pth", map_location=torch.device('cpu'))
model = EfficientNet_V2(2)
model.load_state_dict(chkpt)



def get_crop(image: np.array, x: float, y: float, w: float, h: float):
    hg, wg, _ = image.shape
    x, y, w, h = int(x * wg), int(y * hg), int(w * wg), int(h * hg)
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

# all_crops("data/images/", "data/annotation.csv")#
crops, target = all_crops("D:\\hacks\\train_data_minprirodi\\images\\", "D:\\hacks\\train_data_minprirodi\\annotation.csv")


def transform_valid():
    transform = [
        A.Resize(512,512,p=1),
        ToTensorV2(p=1.0)
    ]
    return A.Compose(transform)

for crop in crops:
    im = transform_valid()(image=crop.astype(np.float32))['image']
    res = model(im.unsqueeze(0))
    print(torch.argmax(res, 1), res)
#print(res, res.shape)