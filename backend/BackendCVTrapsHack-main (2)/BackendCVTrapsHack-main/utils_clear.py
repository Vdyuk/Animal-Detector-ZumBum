import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np


def detect_blur(gray_image, threshold=45.0):
    # Compute the Laplacian of the image and then the variance
    laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
    is_blurry = laplacian_var < threshold
    return is_blurry, laplacian_var

def load_image(file_bytes):
    np_array = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    # Check if image is loaded properly
    if image is None:
        raise ValueError("Image not found or unable to load.")
    # Convert to grayscale for some operations
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray


def estimate_depth(image, model, transform, device):
    # Convert image to NumPy array
    img = np.array(image)

    # Apply transforms and get the input tensor
    input_batch = transform(img).to(device)

    # Prediction
    with torch.no_grad():
        prediction = model(input_batch)

    # Resize and normalize depth map
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=(image.height, image.width),
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    depth_map = prediction.cpu().numpy()
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

    return depth_map


def detect_close_objects_sampled(depth_map, threshold=1, num_samples=10):
    height, width = depth_map.shape
    upper_half = depth_map[:height // 2, :]  # Upper half of the depth map

    # # Generate random sample indices in the upper half
    # y_indices = np.random.randint(0, height // 2, num_samples)
    # x_indices = np.random.randint(0, width, num_samples)

    # # Extract depth values at the sampled points
    # sampled_depths = upper_half[y_indices, x_indices]

    # Check if any sampled depth is below the threshold
    close_points = np.max(upper_half) == threshold
    is_too_close = close_points  # Flag as True if any point is too close

    return is_too_close, np.max(upper_half)


def check_brightness(gray_image, lower_threshold=40, upper_threshold=200):
    mean_brightness = np.mean(gray_image)
    is_too_dark = mean_brightness < lower_threshold
    is_too_bright = mean_brightness > upper_threshold
    return is_too_dark, is_too_bright, mean_brightness


def check_contrast(gray_image, threshold=50):
    min_val, max_val = np.min(gray_image), np.max(gray_image)
    contrast = max_val - min_val
    is_low_contrast = contrast < threshold
    return is_low_contrast, contrast


def load_model_depth(model_type="MiDaS_small"):
    # Load the MiDaS model
    model = torch.hub.load("intel-isl/MiDaS", model_type)
    model.eval()

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load transforms
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    # Select the appropriate transform
    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    return model, transform, device


def image_quality_pipeline(image, gray, model, transform, device):
    results = {}

    # # Blur Detection
    is_blurry, blur_score = detect_blur(gray)
    results['is_blurry'] = is_blurry
    results['blur_score'] = blur_score

    # Depth Estimation
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    depth_map = estimate_depth(pil_image, model, transform, device)
    is_too_close, close_ratio = detect_close_objects_sampled(depth_map)
    results['is_too_close'] = is_too_close
    results['close_ratio'] = close_ratio

    # Brightness Check
    is_too_dark, is_too_bright, mean_brightness = check_brightness(gray)
    results['is_too_dark'] = is_too_dark
    results['is_too_bright'] = is_too_bright
    results['mean_brightness'] = mean_brightness

    # Contrast Check
    is_low_contrast, contrast = check_contrast(gray)
    results['is_low_contrast'] = is_low_contrast
    results['contrast'] = contrast

    return results, depth_map

def get_crop_onnx(image: np.array, x: float, y: float, w: float, h: float):
    hg, wg, _ = image.shape
    x, y, w, h = int(x * wg), int(y * hg), int(w * wg), int(h * hg)

    x_start = max(0, x - w // 2)
    x_end = min(image.shape[1], x + w // 2)
    y_start = max(0, y - h // 2)
    y_end = min(image.shape[0], y + h // 2)

    cropped_image = image[y_start:y_end, x_start:x_end]
    if cropped_image.size > 0:
        img = cv2.resize(cropped_image, (512, 512), interpolation=cv2.INTER_AREA)
    else:
        img = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)

    processed_image = img.astype(np.float32)
    processed_image = np.transpose(processed_image, (2, 0, 1))  # Change to channel-first
    processed_image = np.expand_dims(processed_image, axis=0)  # Add batch dimension
    return processed_image