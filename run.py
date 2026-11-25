# pip install torch open_clip_torch scikit-learn pandas matplotlib pillow
import os, glob, time, socket
import numpy as np, pandas as pd, torch, open_clip
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, roc_auc_score,
                             confusion_matrix, silhouette_score, adjusted_rand_score)

# Force IPv4 (same fix as your other scripts)
_orig = socket.getaddrinfo
socket.getaddrinfo = lambda h, p, f=0, t=0, pr=0, fl=0: _orig(h, p, socket.AF_INET, t, pr, fl)

# ---------------- CONFIG ----------------
NORMAL_DIR = "DATASETS/Normal_Urban"      # change to DATASETS/Normal if you renamed it
SAT_DIR    = "DATASETS/Satellite"
OUT_DIR    = "outputs"
SEED       = 42
os.makedirs(OUT_DIR, exist_ok=True)

NORMAL_PROMPTS = ["a photo taken from the ground level",
                  "a street-level photograph of a city",
                  "a close-up photo of a shop, car or building",
                  "a normal everyday photograph"]
SAT_PROMPTS    = ["a satellite image",
                  "an aerial view from above",
                  "a remote sensing image of a city",
                  "a top-down overhead view of buildings and roads"]

# ---------------- 1. BUILD LABELS ----------------
rows = []
for typ, d in [("normal", NORMAL_DIR), ("satellite", SAT_DIR)]:
    for p in sorted(glob.glob(os.path.join(d, "**", "*"), recursive=True)):
        if p.lower().endswith((".jpg", ".jpeg", ".png")):
            name = os.path.basename(p)
            rows.append({"path": p, "image": name, "type": typ,
                         "category": name.rsplit("_", 1)[0]})
df = pd.DataFrame(rows)
y = (df["type"] == "satellite").astype(int).values       # 0 = normal, 1 = satellite
df.drop(columns="path").to_csv(f"{OUT_DIR}/labels.csv", index=False)
print(df["type"].value_counts().to_string(), "\n")

# ---------------- 2. CLIP EMBEDDINGS ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model = model.to(device).eval()
if device == "cuda":
    model = model.half()
    torch.cuda.reset_peak_memory_stats()

t0, feats = time.time(), []
with torch.no_grad():
    for i in range(0, len(df), 16):
        x = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in df["path"][i:i+16]]).to(device)
        if device == "cuda":
            x = x.half()
        f = model.encode_image(x)
        f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.float().cpu().numpy())
X = np.concatenate(feats)
embed_time = (time.time() - t0) / len(df)
vram = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else 0
np.save(f"{OUT_DIR}/clip_embeddings.npy", X)
print(f"Embedding: {embed_time*1000:.1f} ms/image, peak VRAM {vram:.0f} MB\n")

# ---------------- 3. ZERO-SHOT ROUTER ----------------
def text_emb(prompts):
    with torch.no_grad():
        t = model.encode_text(tokenizer(prompts).to(device))
        t = t / t.norm(dim=-1, keepdim=True)
        t = t.float().mean(0)
        return (t / t.norm()).cpu().numpy()

T = np.stack([text_emb(NORMAL_PROMPTS), text_emb(SAT_PROMPTS)])
logits = 100 * X @ T.T
e = np.exp(logits - logits.max(1, keepdims=True))
probs = e / e.sum(1, keepdims=True)
zs_pred, zs_score = logits.argmax(1), probs[:, 1]

def metrics(y_true, pred, score):
    p, r, f, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
    return {"accuracy": accuracy_score(y_true, pred), "precision": p, "recall": r,
            "f1": f, "roc_auc": roc_auc_score(y_true, score)}

results = {"Zero-shot CLIP": metrics(y, zs_pred, zs_score)}
preds = {"Zero-shot CLIP": zs_pred}

# ---------------- 4. SUPERVISED ROUTERS (5-fold CV) ----------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "SVM (linear)":        make_pipeline(StandardScaler(), SVC(kernel="linear")),
    "SVM (RBF)":           make_pipeline(StandardScaler(), SVC(kernel="rbf")),
    "kNN (k=3)":           KNeighborsClassifier(n_neighbors=3, metric="cosine"),
}
scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
cv_std = {}
for name, m in models.items():
    cv = cross_validate(m, X, y, cv=skf, scoring=scoring)
    results[name] = {k: cv[f"test_{k}"].mean() for k in scoring}
    cv_std[name] = {k: cv[f"test_{k}"].std() for k in scoring}
    preds[name] = cross_val_predict(m, X, y, cv=skf)

table = pd.DataFrame(results).T
table_str = table.copy().astype(object)
for name in table.index:
    for k in scoring:
        sd = cv_std.get(name, {}).get(k)
        table_str.loc[name, k] = f"{table.loc[name, k]:.3f}" + (f" ± {sd:.3f}" if sd is not None else "")
table_str.to_csv(f"{OUT_DIR}/router_results.csv")
print("ROUTER RESULTS (supervised = 5-fold CV mean ± std)")
print(table_str.to_string(), "\n")

# ---------------- 5. CLUSTERING CHECK ----------------
km = KMeans(n_clusters=2, n_init=10, random_state=SEED).fit(X)
print(f"K-Means (k=2): silhouette = {silhouette_score(X, km.labels_):.3f}, "
      f"ARI vs true type = {adjusted_rand_score(y, km.labels_):.3f}\n")

# ---------------- 6. ERROR ANALYSIS ----------------
df["zs_pred"] = np.where(zs_pred == 1, "satellite", "normal")
df["zs_sat_prob"] = zs_score.round(3)
per_cat = (df.assign(correct=(df["zs_pred"] == df["type"]))
             .groupby(["type", "category"])["correct"].mean().round(2))
print("Zero-shot accuracy per category:\n", per_cat.to_string(), "\n")
wrong = df[df["zs_pred"] != df["type"]][["image", "type", "zs_pred", "zs_sat_prob"]]
wrong.to_csv(f"{OUT_DIR}/router_misclassified.csv", index=False)
print("Zero-shot misrouted images:\n", wrong.to_string(index=False) if len(wrong) else "none")

# ---------------- 7. CONFUSION MATRICES ----------------
fig, axes = plt.subplots(1, len(preds), figsize=(4 * len(preds), 4))
for ax, (name, pr) in zip(axes, preds.items()):
    cm = confusion_matrix(y, pr)
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["normal", "satellite"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["normal", "satellite"])
    ax.set_title(name, fontsize=10); ax.set_xlabel("predicted"); ax.set_ylabel("true")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/router_confusion.png", dpi=150)
print(f"\nSaved results to ./{OUT_DIR}/")