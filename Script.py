import os
import cv2
import kagglehub

# Downloads if not cached yet, otherwise just returns the existing local path instantly
DATA_DIR = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
TRAIN_DIR = os.path.join(DATA_DIR, "chest_xray", "train")

# Peek at a sample image
sample_path = os.path.join(TRAIN_DIR, "NORMAL", os.listdir(os.path.join(TRAIN_DIR, "NORMAL"))[0])
sample_image = cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE)

print(f"Image shape: {sample_image.shape}")
print(f"Pixel range: {sample_image.min()} to {sample_image.max()}")
print(f"Data type: {sample_image.dtype}")

print(os.listdir(DATA_DIR))


def validate_dataset(data_dir):
    """Scan a dataset folder and flag common data quality issues."""
    corrupted = []
    too_small = []
    nearly_black = []
    total = 0

    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        for fname in os.listdir(class_path):
            fpath = os.path.join(class_path, fname)
            total += 1
            try:
                img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    corrupted.append(fpath)
                    continue
                if img.shape[0] < 100 or img.shape[1] < 100:
                    too_small.append(fpath)
                if img.mean() < 5:
                    nearly_black.append(fpath)
            except Exception:
                corrupted.append(fpath)

    print(f"Total files scanned: {total}")
    print(f"Corrupted: {len(corrupted)}")
    print(f"Too small: {len(too_small)}")
    print(f"Nearly black: {len(nearly_black)}")
    return corrupted, too_small, nearly_black


validate_dataset(TRAIN_DIR)