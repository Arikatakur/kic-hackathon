"""
PestSense — Early Agricultural Insect Detection System
Hackathon demo: Asian Citrus Psyllid detection.
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

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PestSense",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🌿</text></svg>",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── SVG icons ─────────────────────────────────────────────────────────────────
def icon(paths, size=16, stroke="#7a9a7a", fill="none", vb="0 0 24 24", sw=1.5):
    return (f'<svg width="{size}" height="{size}" viewBox="{vb}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round" '
            f'stroke-linejoin="round">{paths}</svg>')

ICON_SENSOR = icon(
    '<circle cx="12" cy="12" r="3"/>'
    '<path d="M6.3 6.3a8 8 0 000 11.4M17.7 17.7a8 8 0 000-11.4"/>'
    '<path d="M3.5 3.5a14 14 0 000 17M20.5 20.5a14 14 0 000-17"/>',
    size=15
)
ICON_CAMERA = icon(
    '<path d="M23 19a2 2 0 01-2 2H3a2 2 0 01-2-2V8a2 2 0 012-2h4l2-3h6l2 3h4a2 2 0 012 2z"/>'
    '<circle cx="12" cy="13" r="4"/>',
    size=15
)
ICON_CPU = icon(
    '<rect x="4" y="4" width="16" height="16" rx="2"/>'
    '<rect x="9" y="9" width="6" height="6"/>'
    '<path d="M9 2v2M15 2v2M9 20v2M15 20v2M2 9h2M2 15h2M20 9h2M20 15h2"/>',
    size=15
)
ICON_TARGET = icon(
    '<circle cx="12" cy="12" r="10"/>'
    '<circle cx="12" cy="12" r="6"/>'
    '<circle cx="12" cy="12" r="2"/>',
    size=15
)
ICON_WARN_LG = icon(
    '<path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/>'
    '<line x1="12" y1="9" x2="12" y2="13"/>'
    '<line x1="12" y1="17" x2="12.01" y2="17"/>',
    size=22, stroke="#fff", sw=2
)
ICON_CHECK_LG = icon(
    '<path d="M22 11.08V12a10 10 0 11-5.93-9.14"/>'
    '<polyline points="22 4 12 14.01 9 11.01"/>',
    size=22, stroke="#fff", sw=2
)
ICON_UNKNOWN_LG = icon(
    '<circle cx="12" cy="12" r="10"/>'
    '<path d="M9.09 9a3 3 0 015.83 1c0 2-3 3-3 3"/>'
    '<line x1="12" y1="17" x2="12.01" y2="17"/>',
    size=22, stroke="#fff", sw=2
)
ICON_STEP_DONE = icon(
    '<polyline points="20 6 9 17 4 12"/>',
    size=13, stroke="#3d9a5a", sw=2.5
)
ICON_STEP_PENDING = icon(
    '<circle cx="12" cy="12" r="10"/>',
    size=13, stroke="#2d3d2d", sw=1.5
)
ICON_SCAN = icon(
    '<rect x="3" y="3" width="18" height="18" rx="2"/>'
    '<path d="M3 9h18M3 15h18"/>'
    '<circle cx="12" cy="12" r="2"/>',
    size=15
)
ICON_LEAF = icon(
    '<path d="M11 20A7 7 0 019 6.93c.5-.16 1-.29 1.5-.34"/>'
    '<path d="M15.7 4.9C17.2 6.4 18 8.1 18 10c0 4.4-3.6 8-8 8"/>',
    size=15
)

LOGO_SVG = """<svg width="34" height="34" viewBox="0 0 34 34" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M17 3C9 3 4 10 4 17c0 5.5 3.5 10.5 9 12.5V18l6-6c4-4 5-9.5 1.5-13C19.5 3.5 18.5 3 17 3z"
        fill="#3d9a5a" opacity="0.9"/>
  <circle cx="13" cy="21" r="4" fill="none" stroke="#dce8dc" stroke-width="1.3"/>
  <circle cx="13" cy="21" r="1.5" fill="#dce8dc"/>
  <line x1="13" y1="16" x2="13" y2="18.5" stroke="#dce8dc" stroke-width="1"/>
  <line x1="13" y1="23.5" x2="13" y2="26" stroke="#dce8dc" stroke-width="1"/>
  <line x1="8" y1="21" x2="10.5" y2="21" stroke="#dce8dc" stroke-width="1"/>
  <line x1="15.5" y1="21" x2="18" y2="21" stroke="#dce8dc" stroke-width="1"/>
</svg>"""

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── base ── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', system-ui, -apple-system, sans-serif; }

  [data-testid="stAppViewContainer"] { background: #0c0e0c; }
  [data-testid="stSidebar"]          { background: #111411; border-right: 1px solid #1e2b1e; }
  [data-testid="stSidebarContent"]   { padding: 1.5rem 1rem; }

  /* hide only the footer and deploy button, leave header/toggle alone */
  footer                       { visibility: hidden; }
  [data-testid="stDecoration"] { display: none; }
  [data-testid="stToolbar"]    { display: none !important; }

  /* headings */
  h1, h2, h3, h4, h5 { color: #dce8dc !important; font-weight: 600; letter-spacing: -0.02em; }
  p, li               { color: #7a9a7a; font-size: 0.9rem; }

  /* ── wordmark ── */
  .wordmark {
    display: flex; align-items: center; gap: 10px;
    padding: 0 0 20px 0;
  }
  .wordmark-text {
    font-size: 1.25rem; font-weight: 700; letter-spacing: 0.06em;
    color: #dce8dc; text-transform: uppercase;
  }
  .wordmark-sub {
    font-size: 0.7rem; color: #4a6a4a; letter-spacing: 0.1em;
    text-transform: uppercase; margin-top: 1px;
  }

  /* ── header strip ── */
  .page-header {
    display: flex; align-items: center; gap: 14px;
    padding: 10px 0 18px 0; border-bottom: 1px solid #1e2b1e;
    margin-bottom: 24px;
  }
  .page-header-title { font-size: 1.5rem; font-weight: 700; color: #dce8dc; letter-spacing: -0.03em; }
  .page-header-sub   { font-size: 0.8rem; color: #4a6a4a; letter-spacing: 0.05em; margin-top: 2px; }
  .header-badge {
    background: #1a2e1a; border: 1px solid #2d4a2d;
    color: #3d9a5a; font-size: 0.7rem; font-weight: 600;
    padding: 3px 10px; border-radius: 20px; letter-spacing: 0.08em;
    text-transform: uppercase; white-space: nowrap;
  }

  /* ── metric cards ── */
  .metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 24px; }
  .metric-card {
    background: #111411; border: 1px solid #1e2b1e;
    border-radius: 8px; padding: 16px 18px;
  }
  .metric-card-label {
    font-size: 0.7rem; color: #4a6a4a; text-transform: uppercase;
    letter-spacing: 0.1em; margin-bottom: 8px;
    display: flex; align-items: center; gap: 6px;
  }
  .metric-card-value { font-size: 2rem; font-weight: 700; line-height: 1; color: #dce8dc; }
  .metric-card-value.danger  { color: #c4362a; }
  .metric-card-value.caution { color: #c97c1a; }
  .metric-card-value.success { color: #3d9a5a; }
  .metric-card-delta {
    font-size: 0.72rem; color: #4a6a4a; margin-top: 6px;
  }

  /* ── section headers ── */
  .section-label {
    font-size: 0.7rem; font-weight: 600; color: #4a6a4a;
    text-transform: uppercase; letter-spacing: 0.12em;
    margin-bottom: 12px; display: flex; align-items: center; gap: 8px;
  }
  .section-label::after {
    content: ''; flex: 1; height: 1px; background: #1e2b1e;
  }

  /* ── pipeline steps ── */
  .pipeline { display: flex; flex-direction: column; gap: 6px; margin-bottom: 20px; }
  .pipeline-step {
    display: flex; align-items: center; gap: 10px;
    background: #111411; border: 1px solid #1e2b1e;
    border-radius: 6px; padding: 10px 14px;
    font-size: 0.82rem; color: #4a6a4a;
    transition: border-color 0.2s;
  }
  .pipeline-step.done {
    border-color: #1e3a1e; color: #7a9a7a;
  }
  .pipeline-step .step-num {
    width: 20px; height: 20px; border-radius: 50%;
    background: #1a2e1a; border: 1px solid #2d4a2d;
    color: #3d9a5a; font-size: 0.65rem; font-weight: 700;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
  }
  .pipeline-step.done .step-num {
    background: #1a3a1a; border-color: #3d9a5a;
  }

  /* ── alert banners ── */
  .alert-banner {
    border-radius: 8px; padding: 18px 20px;
    display: flex; align-items: center; gap: 14px;
    margin: 16px 0;
  }
  .alert-banner-text  { font-size: 1rem; font-weight: 600; line-height: 1.3; }
  .alert-banner-sub   { font-size: 0.78rem; opacity: 0.8; margin-top: 3px; font-weight: 400; }

  .alert-pest    { background: #2a0e0e; border: 1px solid #6b1a1a; color: #f0a0a0; }
  .alert-safe    { background: #0e1f0e; border: 1px solid #1e4a1e; color: #90c890; }
  .alert-unknown { background: #231800; border: 1px solid #5a3a00; color: #d4a060; }

  .alert-pest    .alert-banner-icon { color: #e05050; }
  .alert-safe    .alert-banner-icon { color: #50b850; }
  .alert-unknown .alert-banner-icon { color: #c08040; }

  /* ── confidence bars ── */
  .conf-row { margin-bottom: 14px; }
  .conf-label {
    display: flex; justify-content: space-between; align-items: center;
    font-size: 0.78rem; color: #7a9a7a; margin-bottom: 5px;
  }
  .conf-label span { color: #dce8dc; font-weight: 600; font-variant-numeric: tabular-nums; }
  .conf-track {
    height: 6px; background: #1a2e1a; border-radius: 3px; overflow: hidden;
  }
  .conf-fill  { height: 100%; border-radius: 3px; transition: width 0.4s ease; }
  .conf-fill.pest    { background: #c4362a; }
  .conf-fill.safe    { background: #3d9a5a; }
  .conf-fill.neutral { background: #4a6a4a; }

  /* ── evidence table ── */
  .evidence-table { border: 1px solid #1e2b1e; border-radius: 6px; overflow: hidden; font-size: 0.8rem; }
  .ev-row {
    display: flex; border-bottom: 1px solid #1a2b1a;
  }
  .ev-row:last-child { border-bottom: none; }
  .ev-key   { width: 38%; padding: 9px 14px; color: #4a6a4a; background: #0f150f; font-weight: 500; }
  .ev-val   { flex: 1; padding: 9px 14px; color: #dce8dc; background: #111411; }
  .ev-val.highlight { color: #3d9a5a; }
  .ev-val.danger    { color: #c4362a; }

  /* ── action box ── */
  .action-box {
    background: #0f150f; border: 1px solid #1e2b1e; border-left: 3px solid #c4362a;
    border-radius: 6px; padding: 12px 16px; font-size: 0.82rem; color: #9ab09a;
    margin-top: 8px;
  }
  .action-box.caution { border-left-color: #c97c1a; }

  /* ── detection log ── */
  .log-table { border: 1px solid #1e2b1e; border-radius: 8px; overflow: hidden; }
  .log-header {
    display: grid; grid-template-columns: 80px 1fr 100px 100px;
    padding: 8px 16px; background: #0f150f;
    font-size: 0.68rem; font-weight: 600; color: #4a6a4a;
    text-transform: uppercase; letter-spacing: 0.1em;
    border-bottom: 1px solid #1e2b1e;
  }
  .log-row {
    display: grid; grid-template-columns: 80px 1fr 100px 100px;
    padding: 10px 16px; border-bottom: 1px solid #141c14;
    font-size: 0.8rem; align-items: center;
  }
  .log-row:last-child { border-bottom: none; }
  .log-row:hover { background: #111811; }
  .log-id   { color: #4a6a4a; font-variant-numeric: tabular-nums; }
  .log-res  { font-weight: 500; }
  .log-prob { color: #7a9a7a; font-variant-numeric: tabular-nums; }
  .log-badge {
    display: inline-block; font-size: 0.65rem; font-weight: 600;
    padding: 2px 8px; border-radius: 20px; letter-spacing: 0.06em;
    text-transform: uppercase;
  }
  .badge-pest    { background: #2a0e0e; color: #e05050; border: 1px solid #5a1a1a; }
  .badge-safe    { background: #0e1f0e; color: #50b850; border: 1px solid #1e4a1e; }
  .badge-unknown { background: #1a1000; color: #c08040; border: 1px solid #4a3000; }

  /* ── sidebar ── */
  .status-dot {
    display: inline-block; width: 7px; height: 7px;
    border-radius: 50%; margin-right: 6px;
  }
  .status-dot.online  { background: #3d9a5a; box-shadow: 0 0 6px #3d9a5a; }
  .status-dot.offline { background: #c4362a; }

  .stat-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 7px 0; border-bottom: 1px solid #1a2b1a;
    font-size: 0.8rem;
  }
  .stat-row:last-child { border-bottom: none; }
  .stat-key { color: #4a6a4a; }
  .stat-val { color: #dce8dc; font-weight: 500; font-variant-numeric: tabular-nums; }

  /* ── empty state ── */
  .empty-state {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    padding: 52px 24px; border: 1px dashed #1e2b1e; border-radius: 8px;
    color: #3a5a3a; text-align: center; gap: 10px;
  }
  .empty-state-title { font-size: 0.85rem; color: #4a6a4a; font-weight: 500; }
  .empty-state-sub   { font-size: 0.75rem; color: #2d4a2d; }

  /* ── buttons ── */
  [data-testid="stButton"] > button {
    background: #1a2e1a; border: 1px solid #2d4a2d;
    color: #3d9a5a; font-size: 0.82rem; font-weight: 600;
    letter-spacing: 0.04em; border-radius: 6px;
    padding: 8px 16px; transition: all 0.15s;
  }
  [data-testid="stButton"] > button:hover {
    background: #1e3a1e; border-color: #3d9a5a; color: #4db368;
  }
  [data-testid="stButton"][data-key*="primary"] > button,
  [data-testid="stButton"] > button[kind="primary"] {
    background: #1f4a2f; border-color: #3d9a5a; color: #dce8dc;
  }

  /* ── divider ── */
  hr { border-color: #1e2b1e !important; }

  /* image display */
  [data-testid="stImage"] img {
    border: 1px solid #1e2b1e; border-radius: 6px;
  }

  /* sidebar collapse / expand toggle */
  [data-testid="collapsedControl"] {
    background: #1a2e1a !important;
    border: 1px solid #2d4a2d !important;
    border-radius: 0 6px 6px 0 !important;
    color: #3d9a5a !important;
  }
  [data-testid="collapsedControl"]:hover {
    background: #1e3a1e !important;
    border-color: #3d9a5a !important;
  }
  [data-testid="collapsedControl"] svg {
    stroke: #3d9a5a !important;
    fill: none !important;
  }
</style>
""", unsafe_allow_html=True)


# ── model loading ─────────────────────────────────────────────────────────────
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
        return self.pool(self.features(x)).flatten(1)

class InsectClassifier(nn.Module):
    def __init__(self, extractor, head):
        super().__init__()
        self.extractor = extractor
        self.head      = head
    def forward(self, x):
        return self.head(self.extractor(x))

@st.cache_resource(show_spinner="Initialising model...")
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
    with open(META_PATH) as f: return json.load(f)

@st.cache_data
def load_samples():
    with open(SAMPLES_PATH) as f: return json.load(f)

def run_classify(model, img: Image.Image, meta: dict):
    t = transform(img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(t), dim=1)[0].tolist()
    return resolve_verdict(probs[0], probs[1], meta)

def resolve_verdict(p_psyllid, p_other, meta):
    conf_high = meta.get("conf_high", 0.70)
    conf_low  = meta.get("conf_low",  0.30)
    if p_psyllid >= conf_high:
        return "pest",    p_psyllid, p_other
    elif p_psyllid <= conf_low:
        return "safe",    p_psyllid, p_other
    else:
        return "unknown", p_psyllid, p_other


# ── session state ─────────────────────────────────────────────────────────────
for k, v in [("log", []), ("alerts", 0), ("scans", 0), ("result", None)]:
    if k not in st.session_state:
        st.session_state[k] = v

model_ready = MODEL_PATH.exists() and META_PATH.exists() and SAMPLES_PATH.exists()
meta = load_meta() if model_ready else {"conf_high": 0.70, "conf_low": 0.30, "best_acc": 0,
                                         "n_train": 0, "n_test": 0}


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div class="wordmark">
      {LOGO_SVG}
      <div>
        <div class="wordmark-text">PestSense</div>
        <div class="wordmark-sub">Field Monitor v0.1</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Input Mode</div>', unsafe_allow_html=True)
    mode = st.radio("", ["Demo — test set", "Upload image"], index=0, label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">System Status</div>', unsafe_allow_html=True)

    dot   = "online" if model_ready else "offline"
    label = "Online" if model_ready else "Offline — run train.py"
    st.markdown(
        f'<span class="status-dot {dot}"></span>'
        f'<span style="font-size:0.8rem;color:#7a9a7a">Classifier: <strong style="color:#dce8dc">{label}</strong></span>',
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)

    rows = [
        ("Accuracy",     f"{meta['best_acc']:.1%}" if model_ready else "—"),
        ("Train images", f"{meta.get('n_train',0):,}" if model_ready else "—"),
        ("Test images",  f"{meta.get('n_test',0):,}"  if model_ready else "—"),
        ("Architecture", "MobileNetV2"),
        ("Target pest",  "D. citri (psyllid)"),
        ("Threshold",    f"≥{meta.get('conf_high',0.7):.0%}"),
    ]
    st.markdown(
        '<div class="evidence-table">'
        + "".join(f'<div class="ev-row"><div class="ev-key">{k}</div><div class="ev-val">{v}</div></div>'
                  for k, v in rows)
        + '</div>',
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">About</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:0.78rem;line-height:1.6">PestSense detects the '
        '<em>Asian Citrus Psyllid</em> (Diaphorina citri) — a 3.5 mm insect '
        'and primary vector of citrus greening disease — before visible crop '
        'symptoms appear. A laser sensor triggers a macro camera; the AI '
        'classifies the captured image in under one second.</p>',
        unsafe_allow_html=True
    )


# ── page header ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="page-header">
  {LOGO_SVG}
  <div style="flex:1">
    <div class="page-header-title">Crop Pest Monitor</div>
    <div class="page-header-sub">KINNERET HACKATHON 2026 &nbsp;·&nbsp; CITRUS PSYLLID DETECTION</div>
  </div>
  <div class="header-badge">Live Demo</div>
</div>
""", unsafe_allow_html=True)


# ── metrics ───────────────────────────────────────────────────────────────────
rate    = (st.session_state.alerts / max(st.session_state.scans, 1)) * 100
acc_str = f"{meta['best_acc']:.1%}" if model_ready else "—"

st.markdown(f"""
<div class="metric-grid">
  <div class="metric-card">
    <div class="metric-card-label">{ICON_SCAN}&nbsp; Detection Events</div>
    <div class="metric-card-value">{st.session_state.scans}</div>
    <div class="metric-card-delta">session total</div>
  </div>
  <div class="metric-card">
    <div class="metric-card-label">{ICON_WARN_LG[:ICON_WARN_LG.index('stroke="#fff"')]}stroke="#c4362a"{ICON_WARN_LG[ICON_WARN_LG.index('stroke="#fff"')+len('stroke="#fff"'):]}&nbsp; Psyllid Alerts</div>
    <div class="metric-card-value danger">{st.session_state.alerts}</div>
    <div class="metric-card-delta">require intervention</div>
  </div>
  <div class="metric-card">
    <div class="metric-card-label">{ICON_TARGET}&nbsp; Alert Rate</div>
    <div class="metric-card-value caution">{rate:.1f}%</div>
    <div class="metric-card-delta">of events flagged</div>
  </div>
  <div class="metric-card">
    <div class="metric-card-label">{ICON_CPU}&nbsp; Model Accuracy</div>
    <div class="metric-card-value success">{acc_str}</div>
    <div class="metric-card-delta">source-split test set</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── main panel ────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 1], gap="large")

img_to_classify = None
sample_info     = None

with left_col:
    st.markdown('<div class="section-label">Captured Image</div>', unsafe_allow_html=True)

    if not model_ready:
        st.markdown(
            '<div class="empty-state">'
            '<div class="empty-state-title">Model not initialised</div>'
            '<div class="empty-state-sub">Run <code>python train.py</code> then restart</div>'
            '</div>', unsafe_allow_html=True
        )
    elif mode == "Demo — test set":
        samples = load_samples()
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Run Detection", use_container_width=True, type="primary"):
                st.session_state.result = random.choice(samples)
                st.session_state.scans += 1
        with c2:
            if st.button("Force Psyllid Sample", use_container_width=True):
                pool = [s for s in samples if s["true_label"] == 0]
                if pool:
                    st.session_state.result = random.choice(pool)
                    st.session_state.scans += 1

        if st.session_state.result:
            try:
                img_to_classify = Image.open(st.session_state.result["path"]).convert("RGB")
                sample_info     = st.session_state.result
            except FileNotFoundError:
                st.error("Image file not found.")
    else:
        uploaded = st.file_uploader(
            "Upload insect image", type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
        if uploaded:
            img_to_classify = Image.open(uploaded).convert("RGB")
            st.session_state.scans += 1

    if img_to_classify:
        st.image(img_to_classify, caption="", use_container_width=True)
    elif model_ready and not st.session_state.result:
        st.markdown(
            '<div class="empty-state" style="margin-top:12px">'
            '<div class="empty-state-title">No image selected</div>'
            '<div class="empty-state-sub">Click Run Detection to begin</div>'
            '</div>', unsafe_allow_html=True
        )


with right_col:
    st.markdown('<div class="section-label">Analysis Pipeline</div>', unsafe_allow_html=True)

    if not model_ready:
        st.markdown('<div class="empty-state"><div class="empty-state-title">Awaiting model</div></div>',
                    unsafe_allow_html=True)

    elif img_to_classify is not None:
        # pipeline steps
        steps = [
            (ICON_SENSOR, "Optical sensor triggered"),
            (ICON_CAMERA, "Macro camera activated — image captured"),
            (ICON_CPU,    "Feature extraction  (MobileNetV2 backbone)"),
            (ICON_TARGET, "Psyllid classifier — inference complete"),
        ]
        st.markdown('<div class="pipeline">' + "".join(
            f'<div class="pipeline-step done">'
            f'<div class="step-num">{ICON_STEP_DONE}</div>'
            f'<span style="margin-right:8px;opacity:0.6">{ico}</span>{lbl}'
            f'</div>'
            for ico, lbl in steps
        ) + '</div>', unsafe_allow_html=True)

        # classify
        if sample_info:
            verdict, p_psyl, p_other = resolve_verdict(
                sample_info["psyllid_prob"], sample_info["other_prob"], meta
            )
        else:
            model   = load_model()
            verdict, p_psyl, p_other = run_classify(model, img_to_classify, meta)

        # alert banner
        if verdict == "pest":
            st.session_state.alerts += 1
            banner_html = (
                f'<div class="alert-banner alert-pest">'
                f'<div class="alert-banner-icon">{ICON_WARN_LG}</div>'
                f'<div><div class="alert-banner-text">Asian Citrus Psyllid Detected</div>'
                f'<div class="alert-banner-sub">Diaphorina citri — quarantine pest</div></div>'
                f'</div>'
            )
        elif verdict == "safe":
            banner_html = (
                f'<div class="alert-banner alert-safe">'
                f'<div class="alert-banner-icon">{ICON_CHECK_LG}</div>'
                f'<div><div class="alert-banner-text">Non-Target Insect</div>'
                f'<div class="alert-banner-sub">No quarantine species identified</div></div>'
                f'</div>'
            )
        else:
            banner_html = (
                f'<div class="alert-banner alert-unknown">'
                f'<div class="alert-banner-icon">{ICON_UNKNOWN_LG}</div>'
                f'<div><div class="alert-banner-text">Inconclusive — Low Confidence</div>'
                f'<div class="alert-banner-sub">Manual field inspection recommended</div></div>'
                f'</div>'
            )
        st.markdown(banner_html, unsafe_allow_html=True)

        # confidence bars
        conf_word = ("High" if max(p_psyl, p_other) > 0.85
                     else "Medium" if max(p_psyl, p_other) > 0.65 else "Low")

        pest_fill  = "pest"   if verdict == "pest"    else "neutral"
        other_fill = "safe"   if verdict == "safe"    else "neutral"

        st.markdown(f"""
        <div style="margin: 16px 0">
          <div class="conf-row">
            <div class="conf-label">
              Psyllid probability <span>{p_psyl:.1%}</span>
            </div>
            <div class="conf-track">
              <div class="conf-fill {pest_fill}" style="width:{p_psyl*100:.1f}%"></div>
            </div>
          </div>
          <div class="conf-row">
            <div class="conf-label">
              Non-target probability <span>{p_other:.1%}</span>
            </div>
            <div class="conf-track">
              <div class="conf-fill {other_fill}" style="width:{p_other*100:.1f}%"></div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # evidence table
        true_lbl = ""
        if sample_info:
            true_lbl = "Psyllid" if sample_info["true_label"] == 0 else "Non-target"
        verdict_label = {"pest": "Psyllid", "safe": "Non-target", "unknown": "Inconclusive"}[verdict]
        verdict_class = {"pest": "danger",  "safe": "highlight",  "unknown": ""}[verdict]

        rows_ev = [
            ("Sensor",         "Optical (macro camera)", ""),
            ("Confidence",     f"{conf_word} ({max(p_psyl, p_other):.1%})", ""),
            ("Classification", verdict_label, verdict_class),
            ("Model",          "MobileNetV2 + linear head", ""),
        ]
        if true_lbl:
            rows_ev.append(("Ground truth", true_lbl, ""))

        st.markdown(
            '<div class="evidence-table">'
            + "".join(f'<div class="ev-row"><div class="ev-key">{k}</div>'
                      f'<div class="ev-val {c}">{v}</div></div>'
                      for k, v, c in rows_ev)
            + '</div>',
            unsafe_allow_html=True
        )

        # action recommendation
        if verdict == "pest":
            st.markdown(
                '<div class="action-box">'
                '<strong style="color:#e05050">Action required</strong> — Alert agronomist. '
                'Schedule field inspection within 24 hours. '
                'Document GPS location of detection event.'
                '</div>', unsafe_allow_html=True
            )
        elif verdict == "unknown":
            st.markdown(
                '<div class="action-box caution">'
                '<strong style="color:#c08040">Manual review</strong> — Confidence below threshold. '
                'Flag for physical inspection on next field visit.'
                '</div>', unsafe_allow_html=True
            )

        # log
        entry = {"verdict": verdict, "p": f"{p_psyl:.1%}"}
        if not st.session_state.log or st.session_state.log[-1] != entry:
            st.session_state.log.append(entry)

    else:
        st.markdown(
            '<div class="empty-state">'
            '<div class="empty-state-title">Awaiting detection event</div>'
            '<div class="empty-state-sub">Select an image or run demo mode</div>'
            '</div>', unsafe_allow_html=True
        )


# ── detection log ─────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-label">Detection Log</div>', unsafe_allow_html=True)

if st.session_state.log:
    log_rows = ""
    for i, e in enumerate(reversed(st.session_state.log[-25:])):
        n      = len(st.session_state.log) - i
        v      = e["verdict"]
        p      = e["p"]
        badge  = f'<span class="log-badge badge-{v}">{v}</span>'
        labels = {"pest": "Psyllid detected", "safe": "Non-target", "unknown": "Inconclusive"}
        log_rows += (
            f'<div class="log-row">'
            f'<div class="log-id">EVT-{n:04d}</div>'
            f'<div class="log-res">{labels[v]}</div>'
            f'<div>{badge}</div>'
            f'<div class="log-prob">{p}</div>'
            f'</div>'
        )
    st.markdown(
        '<div class="log-table">'
        '<div class="log-header"><span>Event ID</span><span>Result</span>'
        '<span>Status</span><span>Psyllid Prob</span></div>'
        + log_rows + '</div>',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        '<div class="log-table">'
        '<div class="log-header"><span>Event ID</span><span>Result</span>'
        '<span>Status</span><span>Psyllid Prob</span></div>'
        '<div style="padding:24px 16px;color:#2d4a2d;font-size:0.8rem">No events recorded this session.</div>'
        '</div>',
        unsafe_allow_html=True
    )
