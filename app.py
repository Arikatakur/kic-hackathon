"""
PestSense — Early Agricultural Insect Detection Dashboard
Hackathon demo: Asian Citrus Psyllid detection using multimodal AI.
"""

import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image
from pathlib import Path
import json
import random
import time
import numpy as np

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PestSense AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* global */
    [data-testid="stAppViewContainer"] { background: #0d1117; }
    [data-testid="stSidebar"]          { background: #161b22; }
    h1, h2, h3, h4                     { color: #e6edf3; }
    p, li, label                       { color: #8b949e; }

    /* alert boxes */
    .alert-pest {
        background: linear-gradient(135deg, #b91c1c, #ef4444);
        color: white; padding: 22px 28px; border-radius: 12px;
        font-size: 1.4rem; font-weight: 700; text-align: center;
        box-shadow: 0 0 24px rgba(239,68,68,.4);
        animation: pulse 1.5s infinite;
    }
    .alert-safe {
        background: linear-gradient(135deg, #15803d, #22c55e);
        color: white; padding: 22px 28px; border-radius: 12px;
        font-size: 1.4rem; font-weight: 700; text-align: center;
    }
    .alert-unknown {
        background: linear-gradient(135deg, #b45309, #f59e0b);
        color: white; padding: 22px 28px; border-radius: 12px;
        font-size: 1.4rem; font-weight: 700; text-align: center;
    }
    @keyframes pulse {
        0%   { box-shadow: 0 0 24px rgba(239,68,68,.4); }
        50%  { box-shadow: 0 0 42px rgba(239,68,68,.8); }
        100% { box-shadow: 0 0 24px rgba(239,68,68,.4); }
    }

    /* metric cards */
    .metric-card {
        background: #161b22; border: 1px solid #30363d;
        border-radius: 10px; padding: 18px 22px; text-align: center;
    }
    .metric-val  { font-size: 2.2rem; font-weight: 700; color: #58a6ff; }
    .metric-lbl  { font-size: 0.85rem; color: #8b949e; margin-top: 4px; }

    /* pipeline step */
    .pipeline-step {
        background: #161b22; border: 1px solid #30363d;
        border-radius: 8px; padding: 12px 16px; margin-bottom: 8px;
        color: #8b949e; font-size: 0.9rem;
    }
    .pipeline-step.active { border-color: #58a6ff; color: #e6edf3; }
    .pipeline-step.done   { border-color: #22c55e; color: #22c55e; }

    /* log table */
    .log-row {
        display: flex; gap: 12px; padding: 8px 12px;
        border-bottom: 1px solid #21262d; font-size: 0.85rem; color: #8b949e;
    }
    .log-row .pest  { color: #ef4444; font-weight: 600; }
    .log-row .safe  { color: #22c55e; }
    .log-row .unk   { color: #f59e0b; }
</style>
""", unsafe_allow_html=True)


# ── model ─────────────────────────────────────────────────────────────────────
MODEL_PATH   = Path("models/classifier.pth")
META_PATH    = Path("models/meta.json")
SAMPLES_PATH = Path("models/test_samples.json")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

class FeatureExtractor(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.features = backbone.features
        self.pool     = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return x.flatten(1)

class InsectClassifier(nn.Module):
    def __init__(self, extractor, head):
        super().__init__()
        self.extractor = extractor
        self.head      = head
    def forward(self, x):
        return self.head(self.extractor(x))

@st.cache_resource(show_spinner="Loading AI model...")
def load_model():
    backbone  = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    extractor = FeatureExtractor(backbone)
    head      = nn.Linear(1280, 2)
    model     = InsectClassifier(extractor, head)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()
    return model

@st.cache_data
def load_meta():
    with open(META_PATH) as f:
        return json.load(f)

@st.cache_data
def load_samples():
    with open(SAMPLES_PATH) as f:
        return json.load(f)

def classify(model, img: Image.Image, meta: dict):
    tensor = transform(img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].tolist()
    psyllid_prob = probs[0]
    other_prob   = probs[1]

    conf_high = meta.get("conf_high", 0.70)
    conf_low  = meta.get("conf_low",  0.30)

    if psyllid_prob >= conf_high:
        verdict = "pest"
        label   = "⚠️ Asian Citrus Psyllid Detected"
    elif psyllid_prob <= conf_low:
        verdict = "safe"
        label   = "✅ Non-Target Insect"
    else:
        verdict = "unknown"
        label   = "❓ Unknown / Low Confidence"

    return verdict, label, psyllid_prob, other_prob


# ── session state ─────────────────────────────────────────────────────────────
if "log"         not in st.session_state: st.session_state.log         = []
if "alert_count" not in st.session_state: st.session_state.alert_count = 0
if "scan_count"  not in st.session_state: st.session_state.scan_count  = 0
if "result"      not in st.session_state: st.session_state.result      = None


# ── check if model is ready ───────────────────────────────────────────────────
model_ready = MODEL_PATH.exists() and META_PATH.exists() and SAMPLES_PATH.exists()


# ── header ────────────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 8])
with col_title:
    st.markdown("# 🔬 PestSense AI")
    st.markdown("**Early Agricultural Insect Detection** · Asian Citrus Psyllid Focus · Kinneret Hackathon 2026")

st.divider()


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Detection Settings")
    mode = st.radio("Input mode", ["Demo (test set images)", "Upload image"], index=0)
    st.divider()
    st.markdown("### System Status")

    if model_ready:
        meta = load_meta()
        st.success("AI Model: Online")
        st.markdown(f"- Test accuracy: **{meta['best_acc']:.1%}**")
        st.markdown(f"- Trained on: **{meta['n_train']:,}** images")
        st.markdown(f"- Test set: **{meta['n_test']:,}** images")
    else:
        st.warning("Model not trained yet.\nRun `python train.py` first.")
        meta = {"conf_high": 0.70, "conf_low": 0.30, "best_acc": 0}

    st.divider()
    st.markdown("### About")
    st.markdown(
        "PestSense combines **laser-based flight sensing** with **AI image classification** "
        "to detect the Asian Citrus Psyllid — a 3.5 mm insect and vector of citrus greening disease — "
        "before visible crop symptoms appear."
    )


# ── metrics row ───────────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-val">{st.session_state.scan_count}</div>
        <div class="metric-lbl">Detection Events</div>
    </div>""", unsafe_allow_html=True)
with m2:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-val" style="color:#ef4444">{st.session_state.alert_count}</div>
        <div class="metric-lbl">Psyllid Alerts</div>
    </div>""", unsafe_allow_html=True)
with m3:
    rate = (st.session_state.alert_count / max(st.session_state.scan_count, 1)) * 100
    st.markdown(f"""<div class="metric-card">
        <div class="metric-val" style="color:#f59e0b">{rate:.1f}%</div>
        <div class="metric-lbl">Alert Rate</div>
    </div>""", unsafe_allow_html=True)
with m4:
    acc_val = f"{meta['best_acc']:.1%}" if model_ready else "—"
    st.markdown(f"""<div class="metric-card">
        <div class="metric-val" style="color:#22c55e">{acc_val}</div>
        <div class="metric-lbl">Model Accuracy</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ── main detection panel ──────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.markdown("### Detection Input")

    img_to_classify = None
    sample_info     = None

    if not model_ready:
        st.info("Train the model first: open a terminal and run `python train.py`")

    elif mode == "Demo (test set images)":
        samples = load_samples()

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("⚡ Simulate Detection Event", use_container_width=True, type="primary"):
                sample = random.choice(samples)
                st.session_state.result = sample
                st.session_state.scan_count += 1

        with col_btn2:
            if st.button("🎯 Pick Psyllid Sample", use_container_width=True):
                psyllid_samples = [s for s in samples if s["true_label"] == 0]
                if psyllid_samples:
                    sample = random.choice(psyllid_samples)
                    st.session_state.result = sample
                    st.session_state.scan_count += 1

        if st.session_state.result:
            r = st.session_state.result
            try:
                img_to_classify = Image.open(r["path"]).convert("RGB")
                sample_info     = r
            except FileNotFoundError:
                st.error(f"Image not found: {r['path']}")

    else:  # upload
        uploaded = st.file_uploader(
            "Upload an insect image", type=["jpg", "jpeg", "png"],
            help="Upload a cropped insect image for classification"
        )
        if uploaded:
            img_to_classify = Image.open(uploaded).convert("RGB")
            st.session_state.scan_count += 1

    if img_to_classify:
        st.image(img_to_classify, caption="Captured image", use_container_width=True)


with right_col:
    st.markdown("### AI Analysis")

    if not model_ready:
        st.info("Waiting for model...")

    elif img_to_classify is not None:
        # simulate pipeline steps
        with st.spinner("Processing..."):
            time.sleep(0.3)

        steps = [
            ("Laser sensor triggered",        "done"),
            ("Camera activated & image captured", "done"),
            ("Feature extraction (MobileNetV2)",  "done"),
            ("Psyllid classifier running...",     "done"),
        ]
        for label, state in steps:
            st.markdown(
                f'<div class="pipeline-step {state}">{"✔" if state=="done" else "◐"} {label}</div>',
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # classification
        if sample_info:
            # use pre-computed probs from test set
            psyllid_prob = sample_info["psyllid_prob"]
            other_prob   = sample_info["other_prob"]
            conf_high    = meta.get("conf_high", 0.70)
            conf_low     = meta.get("conf_low",  0.30)
            if psyllid_prob >= conf_high:
                verdict, result_label = "pest", "⚠️ Asian Citrus Psyllid Detected"
            elif psyllid_prob <= conf_low:
                verdict, result_label = "safe", "✅ Non-Target Insect"
            else:
                verdict, result_label = "unknown", "❓ Unknown / Low Confidence"
        else:
            model = load_model()
            verdict, result_label, psyllid_prob, other_prob = classify(model, img_to_classify, meta)

        # alert banner
        css_class = {"pest": "alert-pest", "safe": "alert-safe", "unknown": "alert-unknown"}[verdict]
        st.markdown(f'<div class="{css_class}">{result_label}</div>', unsafe_allow_html=True)

        if verdict == "pest":
            st.session_state.alert_count += 1

        st.markdown("<br>", unsafe_allow_html=True)

        # confidence bars
        st.markdown("**Confidence Breakdown**")
        st.markdown(f"Psyllid probability")
        st.progress(psyllid_prob, text=f"{psyllid_prob:.1%}")
        st.markdown(f"Non-target probability")
        st.progress(other_prob, text=f"{other_prob:.1%}")

        # evidence summary
        st.markdown("<br>**Evidence Summary**", unsafe_allow_html=True)
        confidence_word = "High" if max(psyllid_prob, other_prob) > 0.85 else \
                          "Medium" if max(psyllid_prob, other_prob) > 0.65 else "Low"
        true_label_str = ""
        if sample_info:
            true_label_str = " · Ground truth: " + ("Psyllid" if sample_info["true_label"] == 0 else "Non-target")

        st.markdown(f"""
| Signal         | Value |
|----------------|-------|
| Sensor         | Optical (camera) |
| Confidence     | {confidence_word} ({max(psyllid_prob, other_prob):.1%}) |
| Classification | {"Psyllid" if verdict=="pest" else "Non-target" if verdict=="safe" else "Unknown"} |
| Model          | MobileNetV2 + Linear head |{true_label_str and f"{chr(10)}| True label | {true_label_str.strip(' · Ground truth: ')} |"}
""")

        if verdict == "pest":
            st.error("🚨 Recommended action: Alert agronomist. Schedule inspection within 24h.")
        elif verdict == "unknown":
            st.warning("⚠️ Low confidence. Manual inspection recommended.")

        # add to log
        log_entry = {
            "verdict": verdict,
            "label":   result_label,
            "psyllid": f"{psyllid_prob:.1%}",
        }
        # avoid duplicate entries on re-render
        if not st.session_state.log or st.session_state.log[-1] != log_entry:
            st.session_state.log.append(log_entry)

    else:
        st.markdown(
            '<div style="color:#8b949e; padding: 60px 0; text-align:center;">'
            '👆 Click "Simulate Detection Event" to run the pipeline'
            '</div>', unsafe_allow_html=True
        )


# ── detection log ─────────────────────────────────────────────────────────────
st.divider()
st.markdown("### Detection Log")

if st.session_state.log:
    cols = st.columns([2, 5, 2])
    cols[0].markdown("**Event**")
    cols[1].markdown("**Result**")
    cols[2].markdown("**Psyllid Prob**")

    for i, entry in enumerate(reversed(st.session_state.log[-20:])):
        color = {"pest": "#ef4444", "safe": "#22c55e", "unknown": "#f59e0b"}[entry["verdict"]]
        c1, c2, c3 = st.columns([2, 5, 2])
        c1.markdown(f"Event #{len(st.session_state.log) - i}")
        c2.markdown(f'<span style="color:{color}">{entry["label"]}</span>', unsafe_allow_html=True)
        c3.markdown(entry["psyllid"])
else:
    st.markdown('<span style="color:#8b949e">No events yet.</span>', unsafe_allow_html=True)
