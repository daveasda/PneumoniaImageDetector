import os
import cv2
import kagglehub
import numpy as np
from matplotlib import pyplot as plt

# 1. SETUP & CONFIGURATION
DATA_DIR = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
TRAIN_DIR = os.path.join(DATA_DIR, "chest_xray", "train")
TARGET_SIZE = (224, 224)  # Global target size


# 2. IMAGE PREPROCESSING HELPER
def resize_with_padding(image, target_size):
    """Preserves aspect ratio and pads remaining edges with black pixels."""
    h, w = image.shape[:2]
    target_h, target_w = target_size
    scale = min(target_h / h, target_w / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h))

    pad_h = target_h - new_h
    pad_w = target_w - new_w
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=0)
    return padded


# 3. CLEAN FILE SCANNER
def get_clean_file_paths(data_dir):
    valid_paths = []
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        for fname in os.listdir(class_path):
            fpath = os.path.join(class_path, fname)

            # Basic validation check
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            if img.shape[0] < 100 or img.shape[1] < 100: continue
            if img.mean() < 5: continue

            valid_paths.append(fpath)

    print(f"Total valid images found: {len(valid_paths)}")
    return valid_paths


train_image_paths = get_clean_file_paths(TRAIN_DIR)


# 4. UPDATED STATS CALCULATOR (Now much faster because images are smaller!)
def compute_train_stats_from_list(image_paths, target_size):
    total_pixels = 0
    pixel_sum = 0.0
    pixel_squared_sum = 0.0

    print("Calculating dataset statistics incrementally...")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    for fpath in image_paths:
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)

        # Apply preprocessing sequence: Enhance -> Resize -> Scale
        img_enhanced = clahe.apply(img)
        img_resized = resize_with_padding(img_enhanced, target_size)
        img_scaled = img_resized.astype(np.float32) / 255.0

        total_pixels += img_scaled.size
        pixel_sum += np.sum(img_scaled)
        pixel_squared_sum += np.sum(img_scaled ** 2)

    mean = pixel_sum / total_pixels
    variance = (pixel_squared_sum / total_pixels) - (mean ** 2)
    std = np.sqrt(variance)

    print(f"Dataset Mean: {mean:.4f}, Std: {std:.4f}")
    return mean, std


# Calculate stats on the fully preprocessed images
train_mean, train_std = compute_train_stats_from_list(train_image_paths, TARGET_SIZE)

# 5. VISUALIZE PREPROCESSING PIPELINE STEP-BY-STEP
sample_path = train_image_paths[0]
original = cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE)

# Run pipeline on sample
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(original)
padded_and_resized = resize_with_padding(enhanced, TARGET_SIZE)

# Final step: Normalize
scaled = padded_and_resized.astype(np.float32) / 255.0
normalized = (scaled - train_mean) / train_std

# Visual Confirmation
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(original, cmap='gray')
axes[0].set_title(f'Original {original.shape}')

axes[1].imshow(enhanced, cmap='gray')
axes[1].set_title('After CLAHE')

axes[2].imshow(padded_and_resized, cmap='gray')
axes[2].set_title(f'Padded & Resized {padded_and_resized.shape}')
plt.show()