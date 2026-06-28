# 🐛 AgriPestAI — Early Detection of the Asian Citrus Psyllid

> 🏆 **1st place — KIC Agtech Hackathon 2026**
>
> An AI-powered insect-detection system that automatically identifies the **Asian Citrus Psyllid** (*Diaphorina citri*) — a 3.5 mm quarantine pest — from trap images, so farms can catch outbreaks **before** they trigger export bans.
>
> *(Built during the hackathon under the working name “PestSense.”)*

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-MobileNetV2-red) ![Streamlit](https://img.shields.io/badge/UI-Streamlit-ff4b4b) ![Accuracy](https://img.shields.io/badge/Test_Accuracy-97.7%25-success)

---

## Team Members

- Saleem Yousef
- Noor Shama
- Romisa Suliman

---
## 🌍 The Problem

Israeli agriculture exports high-value crops like citrus and avocado. The **Asian Citrus Psyllid** is the primary vector of **Huanglongbing (HLB / citrus greening)** — an incurable disease that has devastated citrus industries in Florida, Brazil, and Asia. Once a tree is infected, it cannot recover.

Because the psyllid is a **quarantine pest** in Israel, finding even a few individuals can trigger an **export ban for an entire region** for the whole season. Yet the insect is only ~3.5 mm — extremely hard to spot — and monitoring is still done **manually** by agronomists walking fields and filling spreadsheets. It's slow, expensive, and usually too late.

**AgriPestAI** replaces that with an automated sensor + AI pipeline that flags dangerous insects as early as possible.

---

## 🔬 How It Works

The end-to-end detection pipeline (simulated in the dashboard):

1. **Sensor trigger** — a laser/IR beam across the crop canopy detects an insect flying through.
2. **Camera capture** — a macro camera photographs the insect in a controlled zone.
3. **Feature extraction** — MobileNetV2 converts the image into a 1,280-dim feature vector.
4. **Classification** — a trained linear head outputs **Psyllid / Non-target / Unknown** with confidence scores.
5. **Alert** — a confident psyllid detection fires a red alert recommending agronomist contact within 24 h.

---

## 🧠 The Model

**Transfer learning** with a frozen **MobileNetV2** (pretrained on ImageNet) as a feature extractor + a lightweight linear classifier head trained on the psyllid data.

Key engineering decisions:

- **Source-based train/test split** — images are split by trap-session ID (the first 5 chars of each filename) so that **no trap session appears in both train and test**. This prevents the model from memorizing trap backgrounds and gives an honest, deployment-realistic accuracy.
- **Class-imbalance handling** — with a ~1:4 psyllid-to-other ratio, the loss is **weighted** to penalize missed psyllids ~4× more, so the model can't cheat by always predicting "non-target."
- **Confidence thresholding with an "Unknown" class** — rather than always taking the top class, predictions between 0.30 and 0.70 are flagged as **Unknown / low-confidence** for manual review. Admitting uncertainty is safer than confidently mislabeling an unfamiliar insect as harmless.

| Psyllid probability | Output |
|---|---|
| ≥ 0.70 | 🔴 Psyllid detected |
| ≤ 0.30 | ✅ Non-target insect |
| 0.30 – 0.70 | ⚠️ Unknown / low confidence |

---

## 📊 Results

| Metric | Value |
|---|---|
| **Test accuracy** | **97.7%** |
| **Psyllid recall** | **~95%** |
| Training images | 5,981 |
| Test images | 1,563 |
| Training time | < 10 min on a laptop CPU |

A **95% psyllid recall** means only ~5 of every 100 real psyllids are missed — a strong result for a prototype trained on CPU in under ten minutes.

*Dataset: 1,544 psyllid + 6,000 other trap images provided by Prof. Tamar Keisar, with field-monitoring logs from Tzamach Laboratories.*

---

## 🛠️ Tech Stack

- **Python** · **PyTorch** / **torchvision** (MobileNetV2)
- **Pillow** for image processing
- **Streamlit** for the live demo dashboard

---

## 📁 Project Structure

    kic-hackathon/
    ├── data/
    │   ├── psyllid/psillid_images/    # 1,544 psyllid images
    │   └── other/other_images/        # 6,000 non-psyllid images
    ├── models/
    │   ├── classifier.pth             # saved model weights
    │   ├── test_samples.json          # pre-computed predictions for the demo
    │   └── meta.json                  # accuracy, class names, thresholds
    ├── train.py                       # training script
    ├── save_results.py                # regenerates JSON files from saved model
    ├── app.py                         # Streamlit dashboard
    ├── EXPLAINER.md                   # full technical write-up
    └── README.md

---

## 🚀 Getting Started

    # 1. Install dependencies
    pip install torch torchvision pillow streamlit

    # 2. (Optional) retrain the model — weights are already included
    python train.py

    # 3. Launch the dashboard
    streamlit run app.py

The dashboard opens at `http://localhost:8501`.

> 📖 For a deep dive into the data, transfer-learning approach, and training process, see **[EXPLAINER.md](EXPLAINER.md)**.

---

## 🙌 Acknowledgements

- **Prof. Tamar Keisar** — psyllid / insect image datasets
- **Tzamach Laboratories** — real field-monitoring logs
- Built for the **KIC Agtech Hackathon 2026** 🏆
