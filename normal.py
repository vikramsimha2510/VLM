import os, csv, random, socket, tarfile, urllib.request

# Force IPv4
_orig = socket.getaddrinfo
socket.getaddrinfo = lambda h, p, f=0, t=0, pr=0, fl=0: _orig(h, p, socket.AF_INET, t, pr, fl)

# ---------------- CONFIG ----------------
OUTPUT_DIR = "DATASETS/Normal_Urban"
CACHE_DIR  = "places_cache"
PER_GROUP  = 5            # 6 groups x 5 = 30 images
SEED       = 42

URLS = {
    "cats":     "https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt",
    "filelist": "http://data.csail.mit.edu/places/places365/filelist_places365-standard.tar",
    "val_tar":  "http://data.csail.mit.edu/places/places365/val_256.tar",   # ~500 MB, one time
}

GROUPS = {
    "cars":        ["/p/parking_lot", "/g/gas_station", "/p/parking_garage/outdoor"],
    "road":        ["/h/highway", "/s/street", "/c/crosswalk", "/a/alley"],
    "building":    ["/s/skyscraper", "/o/office_building", "/b/building_facade", "/d/downtown"],
    "house":       ["/h/house", "/r/residential_neighborhood", "/a/apartment_building/outdoor"],
    "supermarket": ["/s/supermarket", "/m/market/outdoor", "/d/department_store"],
    "cafe":        ["/c/coffee_shop", "/r/restaurant", "/r/restaurant_patio", "/f/fastfood_restaurant"],
}

# ---------------- DOWNLOAD ----------------
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def download(url, path):
    if os.path.exists(path):
        return
    print("Downloading:", url)
    def hook(blocks, bsize, total):
        if total > 0 and blocks % 500 == 0:
            print(f"  {min(100, blocks * bsize * 100 // total)}%", end="\r")
    urllib.request.urlretrieve(url, path, hook)
    print("\n  done")

paths = {k: os.path.join(CACHE_DIR, os.path.basename(u)) for k, u in URLS.items()}
for k, u in URLS.items():
    try:
        download(u, paths[k])
    except Exception as e:
        raise SystemExit(f"Could not download {u}\n{e}\n"
                         f"Download it manually from http://places2.csail.mit.edu/download.html "
                         f"and put it in '{CACHE_DIR}/' as '{os.path.basename(u)}'")

# ---------------- GET VAL LABELS FROM FILELIST TAR ----------------
val_txt = os.path.join(CACHE_DIR, "places365_val.txt")
if not os.path.exists(val_txt):
    with tarfile.open(paths["filelist"]) as tf:
        for m in tf:
            if os.path.basename(m.name) == "places365_val.txt":
                with open(val_txt, "wb") as f:
                    f.write(tf.extractfile(m).read())
                break
    if not os.path.exists(val_txt):
        raise SystemExit("places365_val.txt not found inside the filelist tar")

# ---------------- PARSE LABELS ----------------
cat_idx = {}
for line in open(paths["cats"]):
    parts = line.split()
    if len(parts) == 2:
        cat_idx[parts[0]] = int(parts[1])

by_idx = {}
for line in open(val_txt):
    fname, idx = line.split()
    by_idx.setdefault(int(idx), []).append(fname)

# ---------------- PICK IMAGES ----------------
rng = random.Random(SEED)
wanted = {}                       # filename -> (group, places_category)
for g, cats in GROUPS.items():
    pool = {c: rng.sample(by_idx[cat_idx[c]], PER_GROUP) for c in cats}
    for i in range(PER_GROUP):
        c = cats[i % len(cats)]
        wanted[pool[c].pop()] = (g, c)

# ---------------- EXTRACT FROM TAR ----------------
counts = {g: 0 for g in GROUPS}
rows = []
with tarfile.open(paths["val_tar"]) as tf:
    for m in tf:
        base = os.path.basename(m.name)
        if base in wanted:
            g, c = wanted[base]
            counts[g] += 1
            out_name = f"{g}_{counts[g]:02d}.jpg"
            with open(os.path.join(OUTPUT_DIR, out_name), "wb") as f:
                f.write(tf.extractfile(m).read())
            rows.append([out_name, "normal", g, c])
            if len(rows) == len(wanted):
                break

with open(os.path.join(OUTPUT_DIR, "labels_normal.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["image", "type", "category", "places_class"])
    w.writerows(sorted(rows))

print("Saved", len(rows), "images to", os.path.abspath(OUTPUT_DIR))
print(counts)