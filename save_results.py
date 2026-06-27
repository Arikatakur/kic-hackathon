"""
Generates test_samples.json and meta.json from the already-saved model.
Run this once after train.py if it crashed on the print statement.
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

PSYLLID_DIR = Path("data/psyllid/psillid_images")
OTHER_DIR   = Path("data/other/other_images")
MODEL_DIR   = Path("models")

SEED       = 42
TEST_RATIO = 0.2
CONF_HIGH  = 0.70
CONF_LOW   = 0.30

random.seed(SEED)
torch.manual_seed(SEED)

def source_id(filename): return filename[:5]

def split_by_source(files, test_ratio):
    sources = sorted(set(source_id(f.name) for f in files))
    random.shuffle(sources)
    n_test    = max(1, int(len(sources) * test_ratio))
    test_src  = set(sources[:n_test])
    train_src = set(sources[n_test:])
    return [f for f in files if source_id(f.name) in train_src], \
           [f for f in files if source_id(f.name) in test_src]

psyllid_files = sorted(PSYLLID_DIR.glob("*.jpg"))
other_files   = sorted(OTHER_DIR.glob("*.jpg"))

psyllid_train, psyllid_test = split_by_source(psyllid_files, TEST_RATIO)
other_train,   other_test   = split_by_source(other_files,   TEST_RATIO)

train_files = [(f, 0) for f in psyllid_train] + [(f, 1) for f in other_train]
test_files  = [(f, 0) for f in psyllid_test]  + [(f, 1) for f in other_test]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

class InsectDataset(Dataset):
    def __init__(self, files):
        self.files = files
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        path, label = self.files[idx]
        return transform(Image.open(path).convert("RGB")), label, str(path)

class FeatureExtractor(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.features = backbone.features
        self.pool     = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        return self.pool(self.features(x)).flatten(1)

class InsectClassifier(nn.Module):
    def __init__(self, extractor, head):
        super().__init__()
        self.extractor = extractor
        self.head      = head
    def forward(self, x): return self.head(self.extractor(x))

print("Loading model...")
backbone  = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
extractor = FeatureExtractor(backbone)
head      = nn.Linear(1280, 2)
model     = InsectClassifier(extractor, head)
model.load_state_dict(torch.load(MODEL_DIR / "classifier.pth", map_location="cpu", weights_only=True))
model.eval()

print("Running inference on test set...")
loader = DataLoader(InsectDataset(test_files), batch_size=64, num_workers=0)
samples = []
correct = 0
with torch.no_grad():
    for imgs, labels, paths in loader:
        probs = torch.softmax(model(imgs), dim=1)
        preds = probs.argmax(1)
        correct += (preds == labels).sum().item()
        for i in range(len(paths)):
            samples.append({
                "path":         paths[i],
                "true_label":   int(labels[i]),
                "psyllid_prob": float(probs[i][0]),
                "other_prob":   float(probs[i][1]),
            })

acc = correct / len(test_files)
print(f"Test accuracy: {acc:.3f} over {len(test_files)} images")

with open(MODEL_DIR / "test_samples.json", "w") as f:
    json.dump(samples, f, indent=2)

meta = {
    "classes":   ["psyllid", "non-target"],
    "best_acc":  acc,
    "n_train":   len(train_files),
    "n_test":    len(test_files),
    "conf_high": CONF_HIGH,
    "conf_low":  CONF_LOW,
}
with open(MODEL_DIR / "meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("Saved: models/test_samples.json and models/meta.json")
print("Ready! Run: streamlit run app.py")
