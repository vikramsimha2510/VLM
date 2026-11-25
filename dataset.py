
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

# Load UCM
dataset = load_dataset(
    "blanchon/UC_Merced",
    split="train"
)

class_names = dataset.features["label"].names

selected_classes = [
    "buildings",
    "denseresidential",
    "mediumresidential",
    "freeway",
    "intersection"
]

random.seed(42)

output_dir = "UCM_50_Images"
os.makedirs(output_dir, exist_ok=True)

for class_name in selected_classes:

    class_id = class_names.index(class_name)

    indices = [
        i for i, label in enumerate(dataset["label"])
        if label == class_id
    ]

    selected = random.sample(indices, 10)

    class_dir = os.path.join(output_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)

    for count, idx in enumerate(selected, 1):

        image = dataset[idx]["image"].convert("RGB")

        image.save(
            os.path.join(
                class_dir,
                f"{class_name}_{count:02d}.jpg"
            )
        )

print("Downloaded and saved 50 images!")
