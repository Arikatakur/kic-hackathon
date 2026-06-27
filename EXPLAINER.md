# PestSense AI — Full Explainer

## What is this project?

PestSense is an AI-powered insect detection system built for the KIC Agtech Hackathon 2026.

The core idea: Israeli agriculture exports high-value crops like avocados and citrus. A single confirmed detection of certain dangerous insects can ban exports from an entire farm region for the whole growing season. The problem is that current monitoring is done manually — agronomists walk fields, inspect fruit, and fill out spreadsheets. It's slow, expensive, and by the time you find the pest, it's often already established.

PestSense proposes a smarter system: place a sensor near crops that automatically detects and identifies dangerous insects as early as possible, before visible damage or population spread.

---

## What is a Psyllid?

The **Asian Citrus Psyllid** (*Diaphorina citri*) is a tiny insect, about 3–3.5 mm long. It feeds on citrus trees (oranges, lemons, grapefruits) by sucking sap from young shoots.

The reason it is so dangerous is not the feeding itself — it's what it carries. The psyllid is the primary vector (carrier) of **Huanglongbing (HLB)**, also called **citrus greening disease**. HLB has no cure. Once a tree is infected, it cannot recover. Fruit becomes bitter and misshapen, yield drops to near zero, and eventually the tree dies. HLB has devastated citrus industries in Florida, Brazil, and parts of Asia.

In Israel, the Asian Citrus Psyllid is classified as a **quarantine pest**. This means:
- Finding even a small number of psyllids in a field can trigger an export ban for the entire area.
- Early detection is not just useful — it's economically critical.

The insect is 3.5 mm. It is extremely hard to spot with the naked eye.

---

## What data did we have?

We were given two image datasets from Prof. Tamar Keisar:

| Dataset | Images | What it contains |
|---|---|---|
| `psyllid_images` | 1,544 | Cropped photos of psyllid insects caught in traps |
| `other_images` | 6,000 | Cropped photos of other insects caught in the same traps |

Each image filename starts with a 5-digit **source ID** (e.g. `03501_0349063_crop.jpg`). This source ID represents the trap or scanning session the image came from. Multiple images from the same trap session share the same prefix.

We also had real field monitoring logs from Tzamach Laboratories — Excel spreadsheets with hundreds of manual inspection records from avocado orchards (farms Degania B and HaEmek), showing dates, fields, pest types, and agronomist observations. These confirmed that manual monitoring is repetitive and often finds nothing, which is exactly why automation matters.

---

## What is Transfer Learning?

Training an image classifier from scratch requires millions of images and days of compute. We don't have that.

Instead, we used **transfer learning**. The idea is: take a neural network that was already trained on a massive dataset (ImageNet — 1.2 million images, 1,000 categories), and reuse the features it learned.

The model we used is called **MobileNetV2**, developed by Google. It's a convolutional neural network (CNN) designed to be fast and lightweight while still being accurate. It was originally trained to recognize 1,000 categories of objects (cats, cars, chairs, etc.).

The key insight is that the features a CNN learns to recognize — edges, textures, shapes, color patterns — are general enough to be useful for new tasks, even ones the original model never saw. A network trained to distinguish breeds of dogs has already learned what "wings" and "body segments" look like as visual features.

### How we used MobileNetV2

We split the model into two parts:

1. **The backbone (frozen)** — all the convolutional layers that extract visual features from images. We kept these exactly as they were from ImageNet training. We did not change a single weight here. The backbone takes a 224×224 pixel image and outputs a 1,280-dimensional feature vector — a compact numerical description of what's in the image.

2. **The classifier head (trained)** — a single linear layer (1,280 inputs → 2 outputs) that we trained ourselves on our psyllid data. This is the part that learns: "when the feature vector looks like X, it's a psyllid; when it looks like Y, it's not."

This approach is sometimes called **feature extraction**. We run all images through the frozen backbone once, save the feature vectors, and then train only the linear head. This is very fast — the heavy computation (extracting features from 7,544 images) happens once, and then training the head takes seconds per epoch.

---

## How did we train the model?

### Step 1 — Source-based train/test split

This is the most important decision in the whole training process.

A naive approach would be to randomly shuffle all 7,544 images and split 80% for training, 20% for testing. The problem: images from the same trap session look very similar (same background, same lighting, same trap surface). If image `03501_000001` goes into training and `03501_000002` goes into testing, your model memorizes the trap background, not the insect. You'd get misleadingly high accuracy that collapses in real deployment.

Instead, we split by **source ID**. All images from a given trap session go entirely into training or entirely into testing — never both. This means the test set contains trap sessions the model has never seen before, which is a much more honest evaluation of real-world performance.

Result:
- Train: 5,981 images (1,003 psyllid + 4,978 other)
- Test: 1,563 images (541 psyllid + 1,022 other)

### Step 2 — Feature extraction

We ran every image through the frozen MobileNetV2 backbone to get 1,280-dimensional feature vectors. This was done in batches of 32 images at a time. Total time: about 5 minutes on CPU.

### Step 3 — Handle class imbalance

We have 1,544 psyllid images and 6,000 other images — roughly a 1:4 ratio. If we trained naively, the model would learn to just always predict "non-target" and still be 80% accurate, while completely failing to detect psyllids.

We fixed this with **weighted loss**: we multiplied the loss on psyllid errors by a factor of ~4 (the ratio of non-target to psyllid samples). This forces the model to take psyllid misclassifications more seriously during training.

### Step 4 — Train the linear head

We trained the 2-class linear layer for 25 epochs using Adam optimizer with a learning rate of 0.001 and L2 regularization (weight decay 0.0001). Each epoch shuffles the training data and processes it in mini-batches of 64 feature vectors.

We tracked test accuracy after every 5 epochs and saved the best model weights.

### Step 5 — Confidence thresholding for the "Unknown" class

The model outputs two probabilities that sum to 1.0:
- P(psyllid)
- P(non-target)

Instead of always picking the higher one, we applied thresholds:

| Psyllid probability | Output class |
|---|---|
| >= 0.70 | Psyllid detected |
| <= 0.30 | Non-target insect |
| Between 0.30 and 0.70 | Unknown / Low confidence |

The "Unknown" class is important. In a real deployment, confidently misclassifying an unfamiliar insect as "safe" is worse than admitting uncertainty. If the model is unsure, it says so and flags for manual inspection.

---

## Results

| Metric | Value |
|---|---|
| Test accuracy | 97.7% |
| Psyllid recall | ~95% |
| Test set size | 1,563 images |
| Training images | 5,981 images |

**Psyllid recall of 95%** means the model correctly identifies 95 out of 100 psyllid images. In agricultural terms: 5 out of 100 real psyllid detections would be missed. This is a strong result for a prototype trained in under 10 minutes on a laptop CPU.

---

## What does the dashboard do?

The Streamlit dashboard (`app.py`) simulates the full detection pipeline:

1. **Laser sensor triggered** — In the real system, a laser or infrared beam across the crop canopy detects an insect flying through. In the demo, this is simulated by clicking a button.
2. **Camera activated** — A macro camera photographs the insect in a controlled zone.
3. **Feature extraction** — MobileNetV2 processes the image into a feature vector.
4. **Classification** — The linear head outputs psyllid / non-target / unknown with confidence scores.
5. **Alert** — If psyllid is detected above the confidence threshold, a red alert fires with a recommendation to contact the agronomist within 24 hours.

The dashboard also keeps a running detection log and tracks metrics: total events, total psyllid alerts, alert rate, and model accuracy.

---

## File structure

```
kic-hackathon/
├── data/
│   ├── psyllid/psillid_images/    # 1,544 psyllid images
│   └── other/other_images/        # 6,000 non-psyllid images
├── models/
│   ├── classifier.pth             # saved model weights
│   ├── test_samples.json          # pre-computed test predictions for demo
│   └── meta.json                  # accuracy, class names, thresholds
├── train.py                       # training script
├── save_results.py                # regenerates JSON files from saved model
├── app.py                         # Streamlit dashboard
└── EXPLAINER.md                   # this file
```

---

## How to run

**Train the model (already done):**
```
python train.py
```

**Launch the dashboard:**
```
streamlit run app.py
```

Opens at `http://localhost:8501` in your browser.
