import argparse
import os

import torch.onnx
from torchsummary import summary
from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2 as cv2
import supervision as sv



def get_res_image(target_path, weights_path="groundingdino_swint_ogc.pth", image_path="1833377.jpg"):

    HOME = os.getcwd()

    CONFIG_PATH = os.path.join(HOME, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    WEIGHTS_NAME = weights_path
    WEIGHTS_PATH = "D:\\" + weights_path
    model = load_model(CONFIG_PATH, WEIGHTS_PATH)
    IMAGE_NAME = image_path
    IMAGE_PATH = os.path.join(HOME, IMAGE_NAME)

    TEXT_PROMPT = "animal"
    BOX_TRESHOLD = 0.40
    TEXT_TRESHOLD = 0.25

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

    cv2.imwrite(target_path, annotated_frame.view())

    cut = []

    for box in boxes:
        print(box, image_source.shape)
        hg, wg, _ = image_source.shape
        x,y,w,h = int(wg * box[0]), int(hg * box[1]), int(wg * box[2]), int(hg * box[3])
        cut.append(image_source[y-h//2:y+h//2, x-w//2:x+w//2])



    for i in range(len(cut)):
        cv2.imwrite(f"bbox{i+1}.jpg", cut[i])

    return boxes, annotated_frame.view()

get_res_image("res.jpg")