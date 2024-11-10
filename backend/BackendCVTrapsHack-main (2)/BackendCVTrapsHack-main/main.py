from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import uvicorn
import pandas as pd
import shutil
import os
import onnxruntime as ort
import pathlib
from fastapi.middleware.cors import CORSMiddleware
from utils_clear import image_quality_pipeline, load_model_depth, load_image, detect_blur, get_crop_onnx
from PytorchWildlife.models import detection as pw_detection
import cv2
from cv2 import dnn_superres
from fastapi.staticfiles import StaticFiles
import numpy as np

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/images", StaticFiles(directory="images"), name="images")
app.mount("/images_orig", StaticFiles(directory="images_orig"), name="images_orig")

# Инициализация базы данных пользователей
users_db = {}

model_depth_estimation, transform_depth_estimation, device = load_model_depth()
detection_model = pw_detection.MegaDetectorV6(device=device, pretrained=True, version="yolov9c")

# Super resolution model
super_res_model = dnn_superres.DnnSuperResImpl_create()
path = "FSRCNN_x4.pb"
super_res_model.readModel(path)
super_res_model.setModel("fsrcnn", 4)

onnx_model_path = 'efficient_net_fine_tune1.onnx'
ort_session = ort.InferenceSession(onnx_model_path)
input_name_onnx = ort_session.get_inputs()[0].name

class ProcessResult(BaseModel):
    all_count: int
    count_of_images_with_animals: int
    count_of_animals_on_images: int
    count_of_success: int
    images: List  # Список списков с названием файла и результатом классификации


@app.get("/")
def root():
    return {"message": "Welcome to FastAPI Wild Cam Detection App!"}


@app.post("/upload_and_process/")
def upload_and_process(user_id: str,min_width: int, min_height: int, upscale: bool, images: List[UploadFile] = File(...), paths: List[str] = None):
    allowed_extensions = [".jpg", ".jpeg", ".png"]
    processed_images = []
    user_folder = os.path.join("output", user_id)
    user_bounding_boxes = os.path.join("images", user_id)
    user_images_orig = os.path.join("images_orig", user_id)

    os.makedirs(user_folder, exist_ok=True)
    os.makedirs(user_images_orig, exist_ok=True)
    if os.path.exists(user_bounding_boxes) and os.path.isdir(user_bounding_boxes):
        shutil.rmtree(user_bounding_boxes)
        shutil.rmtree(user_images_orig)

    os.makedirs(user_bounding_boxes, exist_ok=True)

    count_of_success = 0
    count_of_failed = 0
    percent_of_success = 0
    count_of_images_with_animals = 0
    count_of_animals_on_images = 0
    all_count = len(images)
    image_batch = []
    file_info_batch = []

    # Сохраняем изображения в батч
    for i, image in enumerate(images):
        if paths and paths[i]:
            file_name = paths[i]
            file_name_base = image.filename
        else:
            file_name = image.filename
            file_name_base = file_name

        if pathlib.Path(file_name).suffix not in allowed_extensions:
            continue

        file_bytes = image.file.read()
        image_cv, gray_image = load_image(file_bytes)

        processed_image_path_orig_tmp = os.path.join(user_images_orig, f"{file_name_base}")
        os.makedirs(os.path.dirname(processed_image_path_orig_tmp), exist_ok=True)
        cv2.imwrite(processed_image_path_orig_tmp, image_cv)

        image_batch.append(image_cv)
        file_info_batch.append((file_name, file_name_base, image_cv, gray_image))

    # batch_results = detection_model.batch_image_detection(user_images_orig, batch_size=16)
    #
    # if os.path.exists(user_images_orig):
    #     shutil.rmtree(user_images_orig)
    #
    # batch_results_map = {image_name.split("/")[-1]: result for image_name, result in
    #                      zip([info[0] for info in file_info_batch], batch_results)}

    for i, (file_name, file_name_base, image_cv, gray_image) in enumerate(file_info_batch):
        quality_test, _ = image_quality_pipeline(image_cv[30: -55,:], gray_image[30: -55,:],
                                              model_depth_estimation,
                                              transform_depth_estimation, device)

        if quality_test['is_low_contrast'] or quality_test['is_too_bright'] \
            or quality_test['is_too_dark'] or quality_test['is_too_close']:
            classification_result = 0
            continue
        else: # продолжаем детектить
            results = detection_model.single_image_detection(image_cv)

            # for batch detection_model.batch_image_detection(tgt_folder_path, batch_size=16)
            # results = batch_results_map.get(file_name_base)
            # if not results:
            #     continue  # If no results for this file, skip processing

            detections = results['detections']
            has_valid_box = False
            has_valid_not_confident_box = False

            processed_image_path_orig = os.path.join(user_images_orig, f"original_{file_name}")
            os.makedirs(os.path.dirname(processed_image_path_orig), exist_ok=True)
            cv2.imwrite(processed_image_path_orig, image_cv)

            valid_detections = [
                (bbox, conf, class_id) for bbox, conf, class_id in zip(detections.xyxy,
                                                                       detections.confidence,
                                                                       detections.class_id) if (conf > 0.4) and (class_id == 0)
            ]
            if len(valid_detections):
                count_of_images_with_animals += 1
                count_of_animals_on_images += len(valid_detections)

            rectangles = []
            for k, (bbox, conf, _) in enumerate(valid_detections):
                height, width, _ = image_cv.shape
                x1, y1, x2, y2 = bbox
                bbox_width = x2 - x1
                bbox_height = y2 - y1

                if bbox_width >= min_width and bbox_height >= min_height:

                    x_center = (x1 + x2) / 2 / width
                    y_center = (y1 + y2) / 2 / height
                    w_norm = bbox_width / width
                    h_norm = bbox_height / height
                    processed_image = get_crop_onnx(image_cv, x_center, y_center, w_norm, h_norm)

                    outputs = ort_session.run(None, {input_name_onnx: processed_image})
                    if np.argmax(outputs[0]) == 1:
                        has_valid_box = True
                        classification_result = 1
                    else:
                        has_valid_not_confident_box = True
                        classification_result = 0

                    norm_x1, norm_y1 = float(x1 / width), float(y1 / height)
                    norm_x2, norm_y2 = float(x2 / width), float(y2 / height)

                    cropped_image = image_cv[int(y1):int(y2), int(x1):int(x2)]
                    cropped_path = os.path.join(user_folder, f"{k}_{file_name_base}")
                    os.makedirs(os.path.dirname(cropped_path), exist_ok=True)

                    if upscale:
                        if cropped_image.shape[0] <= 200 and cropped_image.shape[1] <= 200:
                            cropped_image = super_res_model.upsample(cropped_image)

                    cv2.imwrite(cropped_path, cropped_image)
                    rectangles.append((x1, y1, x2, y2))

                    if user_id not in users_db:
                        users_db[user_id] = []

                    users_db[user_id].append(
                        [file_name, [norm_x1, norm_y1, norm_x2, norm_y2], classification_result]
                    )
                else:
                    norm_x1, norm_y1 = float(x1 / width), float(y1 / height)
                    norm_x2, norm_y2 = float(x2 / width), float(y2 / height)
                    classification_result = 0
                    users_db[user_id].append(
                        [file_name, [norm_x1, norm_y1, norm_x2, norm_y2], classification_result]
                    )

            # if not valid_detections:
            #     classification_result = 0
            #     users_db[user_id].append(
            #         [file_name, [], classification_result]
            #     )
            #     continue

            if has_valid_box:
                classification_result = 1
                for (x1, y1, x2, y2) in rectangles:
                    cv2.rectangle(image_cv, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

                processed_image_path = os.path.join(user_bounding_boxes, f"{file_name}")
                os.makedirs(os.path.dirname(processed_image_path), exist_ok=True)
                cv2.imwrite(processed_image_path, image_cv)

                processed_images.append(
                    [file_name, classification_result, processed_image_path, processed_image_path_orig]
                )

                continue

            if has_valid_not_confident_box:
                classification_result = 0

                for (x1, y1, x2, y2) in rectangles:
                    cv2.rectangle(image_cv, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)

                processed_image_path = os.path.join(user_bounding_boxes, f"{file_name}")
                os.makedirs(os.path.dirname(processed_image_path), exist_ok=True)
                cv2.imwrite(processed_image_path, image_cv)

                processed_images.append(
                    [file_name, classification_result, processed_image_path, processed_image_path_orig]
                )



    # Подсчет статистики
    if len(processed_images) > 0:
        count_of_success = sum(1 for _, classification, _, _ in processed_images if classification == 1)
        count_of_failed = len(processed_images) - count_of_success
        percent_of_success = (count_of_success / all_count) * 100 if processed_images else 0

    response = ProcessResult(
        all_count=all_count,
        count_of_images_with_animals=count_of_images_with_animals,
        count_of_animals_on_images=count_of_animals_on_images,
        count_of_success=count_of_success,
        images=processed_images
    )
    # всего животных
    # качественных животных
    return response


@app.get("/generate_report/")
def generate_report(user_id: str):
    if user_id not in users_db:
        return {"error": "User not found"}

    images = users_db[user_id]
    report = pd.DataFrame(columns=['Name', 'Bbox','Class'])
    counter = 1

    for file_name, bbox, classification in images:
        bbox_str = ','.join(map(str, bbox))
        report.loc[counter] = [file_name, bbox_str, classification]
        counter += 1

    report.to_excel(f'{user_id}_history.xlsx', index=False)
    headers = {'Content-Disposition': f'attachment; filename="{user_id}_history.xlsx"'}

    return FileResponse(f'{user_id}_history.xlsx', headers=headers)

@app.post("/delete_user_data/")
def delete_user_data(user_id: str):
    if user_id in users_db:
        del users_db[user_id]

        user_folder = os.path.join("output", user_id)
        user_bboxes = os.path.join("images", user_id)
        user_orig = os.path.join("images_orig", user_id)
        zip_file = f"{user_folder}.zip"
        excel_path = f'{user_id}_history.xlsx'

        if os.path.exists(user_folder):
            shutil.rmtree(user_folder)
        if os.path.exists(user_orig):
            shutil.rmtree(user_orig)
        if os.path.exists(user_bboxes):
            shutil.rmtree(user_bboxes)

        if os.path.exists(excel_path):
            os.remove(excel_path)

        if os.path.exists(zip_file):
            os.remove(zip_file)

        return {"message": f"Data, folder, and ZIP file for user {user_id} deleted successfully"}
    else:
        return {"error": "User not found"}

@app.get("/download_processed_images/")
def download_processed_images(user_id: str):
    if user_id in users_db and len(users_db[user_id]) >=1:
        user_folder = os.path.join("images", user_id)
        zip_path = f"{user_id}_processed_files.zip"

        if not os.path.exists(user_folder):
            return {"error": "User folder not found"}

        shutil.make_archive(user_folder, 'zip', user_folder)

        headers = {'Content-Disposition': f'attachment; filename="{zip_path}"'}

        return FileResponse(f"{user_folder}.zip", headers=headers, media_type='application/zip')
    else:
        return {"error": "User not found"}


if __name__ == "__main__":
    uvicorn.run(app, port=8898)