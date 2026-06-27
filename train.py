"""
Train a psyllid vs. non-target insect classifier.
Uses MobileNetV2 as a frozen feature extractor + linear head.
Splits by source ID (first 5 chars of filename) to avoid data leakage.
"""

import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
import random
import json
import sys

PSYLLID_DIR = Path("data/psyllid/psillid_images")
OTHER_DIR   = Path("data/other/other_images")
MODEL_DIR   = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

BATCH_SIZE  = 64
EPOCHS      = 25
SEED        = 42
TEST_RATIO  = 0.2
CONF_HIGH   = 0.70   # above this → confident Psyllid
CONF_LOW    = 0.30   # below this → confident Non-target
# between CONF_LOW and CONF_HIGH → Unknown

random.seed(SEED)
torch.manual_seed(SEED)


# ---------- source-based split ----------

def source_id(filename: str) -> str:
    return filename[:5]

def split_by_source(files: list, test_ratio: float):
    sources = sorted(set(source_id(f.name) for f in files))
    random.shuffle(sources)
    n_test = max(1, int(len(sources) * test_ratio))
    test_src  = set(sources[:n_test])
    train_src = set(sources[n_test:])
    train = [f for f in files if source_id(f.name) in train_src]
    test  = [f for f in files if source_id(f.name) in test_src]
    return train, test


psyllid_files = sorted(PSYLLID_DIR.glob("*.jpg"))
other_files   = sorted(OTHER_DIR.glob("*.jpg"))

psyllid_train, psyllid_test = split_by_source(psyllid_files, TEST_RATIO)
other_train,   other_test   = split_by_source(other_files,   TEST_RATIO)

# label 0 = psyllid, label 1 = non-target
train_files = [(f, 0) for f in psyllid_train] + [(f, 1) for f in other_train]
test_files  = [(f, 0) for f in psyllid_test]  + [(f, 1) for f in other_test]

random.shuffle(train_files)
random.shuffle(test_files)

print(f"Train: {len(train_files)}  (psyllid {len(psyllid_train)}, other {len(other_train)})")
print(f"Test:  {len(test_files)}   (psyllid {len(psyllid_test)}, other {len(other_test)})")


# ---------- dataset ----------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

class InsectDataset(Dataset):
    def __init__(self, files):
        self.files = files
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        path, label = self.files[idx]
        img = Image.open(path).convert("RGB")
        return transform(img), label, str(path)


# ---------- feature extractor ----------

print("\nLoading MobileNetV2 (pretrained)...")
backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
backbone.eval()

class FeatureExtractor(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return x.flatten(1)   # (B, 1280)

extractor = FeatureExtractor(backbone)
extractor.eval()


def extract_all(file_list, desc=""):
    dataset = InsectDataset(file_list)
    loader  = DataLoader(dataset, batch_size=32, num_workers=0, shuffle=False)
    feats, labels, paths = [], [], []
    with torch.no_grad():
        for i, (imgs, lbl, pth) in enumerate(loader):
            f = extractor(imgs)
            feats.append(f)
            labels.append(lbl)
            paths.extend(pth)
            done = min((i + 1) * 32, len(dataset))
            print(f"  {desc} {done}/{len(dataset)}", end="\r", flush=True)
    print()
    return torch.cat(feats), torch.cat(labels), paths


print("Extracting train features...")
train_feats, train_labels, train_paths = extract_all(train_files, "train")

print("Extracting test features...")
test_feats, test_labels, test_paths = extract_all(test_files, "test")


# ---------- train linear head ----------

print("\nTraining classifier head...")

n_psyllid = (train_labels == 0).sum().item()
n_other   = (train_labels == 1).sum().item()
# upweight psyllid to compensate for class imbalance
class_weights = torch.tensor([n_other / n_psyllid, 1.0])

classifier = nn.Linear(1280, 2)
optimizer  = torch.optim.Adam(classifier.parameters(), lr=1e-3, weight_decay=1e-4)
criterion  = nn.CrossEntropyLoss(weight=class_weights)

best_acc   = 0.0
best_state = None

for epoch in range(EPOCHS):
    classifier.train()
    perm = torch.randperm(len(train_feats))
    tf   = train_feats[perm]
    tl   = train_labels[perm]

    total_loss = 0.0
    for i in range(0, len(tf), BATCH_SIZE):
        x = tf[i:i + BATCH_SIZE]
        y = tl[i:i + BATCH_SIZE]
        optimizer.zero_grad()
        loss = criterion(classifier(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # evaluate
    classifier.eval()
    with torch.no_grad():
        probs = torch.softmax(classifier(test_feats), dim=1)
        preds = probs.argmax(1)
        acc   = (preds == test_labels).float().mean().item()

        psyllid_mask = test_labels == 0
        recall_p = (preds[psyllid_mask] == 0).float().mean().item() if psyllid_mask.any() else 0.0

    if acc > best_acc:
        best_acc   = acc
        best_state = {k: v.clone() for k, v in classifier.state_dict().items()}

    if (epoch + 1) % 5 == 0:
        print(f"  Epoch {epoch+1:2d}/{EPOCHS}  loss={total_loss:.2f}  "
              f"acc={acc:.3f}  psyllid_recall={recall_p:.3f}")

classifier.load_state_dict(best_state)
print(f"\nBest test accuracy: {best_acc:.3f}")


# ---------- save full model ----------

class InsectClassifier(nn.Module):
    def __init__(self, extractor, head):
        super().__init__()
        self.extractor = extractor
        self.head      = head

    def forward(self, x):
        return self.head(self.extractor(x))

    def predict(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            probs  = torch.softmax(logits, dim=1)
        return probs   # (B, 2) — col 0 = psyllid, col 1 = non-target

full_model = InsectClassifier(extractor, classifier)
torch.save(full_model.state_dict(), MODEL_DIR / "classifier.pth")
print("Model saved -> models/classifier.pth")


# ---------- save test samples for demo ----------

classifier.eval()
with torch.no_grad():
    all_probs = torch.softmax(classifier(test_feats), dim=1).numpy()

samples = []
for i, (path, label) in enumerate(zip(test_paths, test_labels.tolist())):
    samples.append({
        "path":         path,
        "true_label":   label,           # 0=psyllid, 1=other
        "psyllid_prob": float(all_probs[i][0]),
        "other_prob":   float(all_probs[i][1]),
    })

with open(MODEL_DIR / "test_samples.json", "w") as f:
    json.dump(samples, f, indent=2)

# save metadata
meta = {
    "classes":   ["psyllid", "non-target"],
    "best_acc":  best_acc,
    "n_train":   len(train_files),
    "n_test":    len(test_files),
    "conf_high": CONF_HIGH,
    "conf_low":  CONF_LOW,
}
with open(MODEL_DIR / "meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("Test samples saved -> models/test_samples.json")
print("\nAll done. Run: streamlit run app.py")
