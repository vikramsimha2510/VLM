
import os
import random
import socket

# Force IPv4 to prevent hanging on networks with broken IPv6 routing
_orig_getaddrinfo = socket.getaddrinfo
def _ipv4_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    return _orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
socket.getaddrinfo = _ipv4_getaddrinfo

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from datasets import load_dataset
from tqdm import tqdm

# ----------------------------------
# CONFIGURATION
# ----------------------------------

OUTPUT_DIR = "Urban_Normal_25"
IMAGES_PER_CATEGORY = 5
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# ----------------------------------
# LOAD COCO DATASET (STREAMING)
# ----------------------------------

print("Streaming COCO validation dataset...")

dataset = load_dataset(
    "detection-datasets/coco",
    split="val",
    streaming=True
)

print("Dataset stream initialized!")

# ----------------------------------
# DEFINE URBAN CATEGORIES
# ----------------------------------

categories = {
    "buildings": ["building"],
    "roads_traffic": ["car", "truck", "bus", "motorcycle"],
    "vehicles": ["car", "truck", "bus", "motorcycle"],
    "greenery": ["tree", "potted plant"],
    "mixed_urban": ["building", "car", "tree"]
}

# ----------------------------------
# INSPECT DATASET FORMAT
# ----------------------------------

print("\nAvailable columns:", dataset.column_names)

# ----------------------------------
# SAVE 25 IMAGES
# ----------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

total_images = 25
sampled_dataset = dataset.shuffle(seed=RANDOM_SEED, buffer_size=100).take(total_images)

for count, item in enumerate(
    tqdm(sampled_dataset, total=total_images),
    start=1
):
    if count == 1:
        print("\nFirst sample keys:", list(item.keys()))

    image = item["image"].convert("RGB")

    image_path = os.path.join(
        OUTPUT_DIR,
        f"urban_image_{count:02d}.jpg"
    )

    image.save(image_path, quality=95)

print("\nSuccessfully downloaded 25 images!")
print("Saved in:", os.path.abspath(OUTPUT_DIR))
