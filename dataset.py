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

# ---------------- CONFIG ----------------
OUTPUT_DIR     = "DATASETS/Satellite"       # same folder as your existing 50 images
GREEN_CLASSES  = ["forest", "golfcourse"]   # vegetation + open green space
PER_CLASS      = 10
SEED           = 42

# ---------------- LOAD ----------------
dataset = load_dataset("blanchon/UC_Merced", split="train")
class_names = dataset.features["label"].names

missing = [c for c in GREEN_CLASSES if c not in class_names]
if missing:
    raise SystemExit(f"Class not found: {missing}\nAvailable classes: {class_names}")

labels = dataset["label"]
os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(SEED)

# ---------------- SAVE ----------------
for class_name in GREEN_CLASSES:
    class_id = class_names.index(class_name)
    indices = [i for i, lab in enumerate(labels) if lab == class_id]
    selected = random.sample(indices, PER_CLASS)

    for count, idx in enumerate(selected, 1):
        out_path = os.path.join(OUTPUT_DIR, f"{class_name}_{count:02d}.jpg")
        if os.path.exists(out_path):
            continue
        dataset[idx]["image"].convert("RGB").save(out_path)

    print(f"{class_name}: saved {PER_CLASS} images")

total = len([f for f in os.listdir(OUTPUT_DIR) if f.lower().endswith(".jpg")])
print(f"Satellite folder now has {total} images")