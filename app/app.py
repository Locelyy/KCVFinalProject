import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import base64
from io import BytesIO

st.set_page_config(page_title="MammoCNN", layout="centered")

CONFIDENCE_THRESHOLD = 0.90

CLASS_NAMES = [
    "Adenosis",
    "Ductal Carcinoma",
    "Fibroadenoma",
    "Lobular Carcinoma",
    "Mucinous Carcinoma",
    "Papillary Carcinoma",
    "Phyllodes Tumor",
    "Tubular Adenoma"
]

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(160deg, #000000, #0a0f2c, #0f172a);
    color: white;
}

/* Hide default spinner label */
.stSpinner > div { color: rgba(255,255,255,0.5) !important; }

h1, h2, h3, p, span, label, div {
    color: white;
}

.stButton { display: flex; justify-content: center; width: 100%; }
.stButton > button {
    background: rgba(255,255,255,0.08) !important;
    border: 0.5px solid rgba(255,255,255,0.2) !important;
    color: white !important;
    border-radius: 10px !important;
    padding: 10px 28px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    letter-spacing: 0.5px !important;
}
.stButton > button:hover {
    background: rgba(255,255,255,0.14) !important;
}

/* Result badge */
.result-badge {
    background: rgba(99,102,241,0.18);
    border: 0.5px solid rgba(99,102,241,0.4);
    border-radius: 6px;
    padding: 10px 14px;
    margin: 12px 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.result-name { font-size: 16px; font-weight: 500; color: white !important; }
.result-conf { font-size: 13px; color: #a5b4fc !important; }

/* Warning badge */
.warn-badge {
    background: rgba(234,179,8,0.12);
    border: 0.5px solid rgba(234,179,8,0.4);
    border-radius: 6px;
    padding: 14px 16px;
    margin: 12px 0;
    text-align: center;
}
.warn-title { font-size: 14px; font-weight: 500; color: #fde047 !important; margin-bottom: 4px; }
.warn-sub { font-size: 12px; color: rgba(253,224,71,0.65) !important; }

/* Image cards */
.img-card {
    background: rgba(255,255,255,0.05);
    border: 0.5px solid rgba(255,255,255,0.12);
    border-radius: 8px;
    overflow: hidden;
    margin-bottom: 8px;
}
.img-card-label {
    font-size: 11px;
    color: rgba(255,255,255,0.45) !important;
    padding: 8px 10px 4px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

/* Heatmap legend */
.heatmap-legend {
    margin-top: 12px;
    background: rgba(255,255,255,0.04);
    border: 0.5px solid rgba(255,255,255,0.1);
    border-radius: 8px;
    padding: 12px 14px;
}
.legend-title {
    font-size: 11px;
    color: rgba(255,255,255,0.4) !important;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-bottom: 10px;
}
.legend-bar {
    width: 100%;
    height: 8px;
    border-radius: 4px;
    background: linear-gradient(90deg, #000080, #0000ff, #00ffff, #00ff00, #ffff00, #ff8800, #ff0000);
    margin-bottom: 4px;
}
.legend-ends {
    display: flex;
    justify-content: space-between;
    font-size: 10px;
    color: rgba(255,255,255,0.35) !important;
    margin-bottom: 10px;
}
.legend-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 6px;
}
.legend-item {
    display: flex;
    align-items: flex-start;
    gap: 7px;
}
.legend-dot {
    width: 10px;
    height: 10px;
    border-radius: 2px;
    flex-shrink: 0;
    margin-top: 2px;
}
.legend-text {
    font-size: 11px;
    color: rgba(255,255,255,0.5) !important;
    line-height: 1.4;
}

/* Drop zone */
.drop-zone-wrapper {
    border: 1.5px dashed rgba(255,255,255,0.35);
    border-radius: 12px;
    padding: 2.5rem 1.5rem;
    text-align: center;
    margin: 1.5rem 0;
    background: rgba(255,255,255,0.03);
    cursor: pointer;
}
.drop-zone-icon { font-size: 36px; margin-bottom: 10px; opacity: 0.5; }
.drop-zone-label { font-size: 14px; color: rgba(255,255,255,0.7) !important; }
.drop-zone-sub { font-size: 12px; color: rgba(255,255,255,0.35) !important; margin-top: 4px; }
.drop-zone-types { font-size: 11px; color: rgba(255,255,255,0.25) !important; margin-top: 8px; letter-spacing: 0.5px; }

/* Preview box */
.preview-filename {
    font-size: 11px;
    color: rgba(255,255,255,0.35) !important;
    padding: 6px 10px;
    letter-spacing: 0.3px;
}

/* Progress bar */
.analyzing-label {
    font-size: 12px;
    color: rgba(255,255,255,0.5) !important;
    text-align: center;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.stProgress > div > div > div {
    background: linear-gradient(90deg, #6366f1, #818cf8, #a5b4fc) !important;
}
.stProgress > div > div {
    background: rgba(255,255,255,0.1) !important;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 8)
    model.load_state_dict(torch.load(
        "models/best_resnet50_all_mag.pth",
        map_location=torch.device('cpu')
    ))
    model.eval()
    return model


@st.cache_resource
def load_efficientnet_model():
    model = models.efficientnet_v2_s(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 8)
    model.load_state_dict(torch.load("models/best_efficientnet_v2_all_mag.pth", map_location=torch.device('cpu')))
    model.eval()
    return model


@st.cache_resource
def load_densenet_model():
    model = models.densenet121(weights=None)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, 8)
    model.load_state_dict(torch.load("models/best_densenet121_cutmix_all_mag.pth", map_location=torch.device('cpu')))
    model.eval()
    return model


def make_circular_progress(conf_val, model_name, pred_name, color="#6366f1"):
    conf_pct = int(conf_val * 100)
    return f"""<div style="display:flex; flex-direction:column; align-items:center; gap:8px;">
<span style="font-size:12px; color:rgba(255,255,255,0.7); font-weight:500; text-align:center;">{model_name}</span>
<svg viewBox="0 0 36 36" style="width:70px;height:70px;">
<path style="stroke:rgba(255,255,255,0.1);stroke-width:3;fill:none;" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
<path style="stroke:{color};stroke-width:3;stroke-dasharray:{conf_pct}, 100;fill:none;stroke-linecap:round;" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
<text x="18" y="20.35" style="fill:white;font-size:9px;font-weight:600;text-anchor:middle;">{conf_pct}%</text>
</svg>
<span style="font-size:11px; font-weight:600; color:{color}; text-align:center;">{pred_name}</span>
</div>"""


def generate_gradcam(model, input_tensor, target_class_idx):
    target_layer = model.layer4[-1]
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    model.zero_grad()
    output = model(input_tensor)
    loss = output[0, target_class_idx]
    loss.backward()

    h1.remove()
    h2.remove()

    grad = gradients[0].detach()
    act = activations[0].detach()
    weights = torch.mean(grad, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * act, dim=1).squeeze()
    cam = F.relu(cam)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.numpy()


def overlay_heatmap(original_image, cam):
    img = np.array(original_image.resize((224, 224)))
    heatmap = cv2.resize(cam, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(img_bgr, 0.6, heatmap, 0.4, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


def pil_to_base64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def numpy_to_base64(arr: np.ndarray) -> str:
    img = Image.fromarray(arr.astype(np.uint8))
    return pil_to_base64(img)


# ── Header ──────────────────────────────────────────────────────────────────
st.markdown("<h1 style='text-align:center;font-size:28px;font-weight:500;letter-spacing:-0.5px;'>HistopathAI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;font-size:13px;color:rgba(255,255,255,0.5);'>Breast cancer histopathology image classification · ResNet50 · EfficientNetV2 · DenseNet121 + Grad-CAM</p>", unsafe_allow_html=True)

# ── File uploader ────────────────────────────────────────────────────────
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

uploaded_file = st.file_uploader(
    "Upload histopathology image",
    type=["jpg", "jpeg", "png"]
)

# ── Image uploaded ───────────────────────────────────────────────────────────
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    img_b64 = pil_to_base64(image)

    # Preview box replacing drop zone
    st.markdown(f"""
    <div style="border-radius:12px;overflow:hidden;border:0.5px solid rgba(255,255,255,0.12);
                background:rgba(255,255,255,0.05);margin:1.5rem 0;position:relative;text-align:center;">
        <img src="data:image/png;base64,{img_b64}"
             style="max-width:100%;height:auto;display:block;margin:0 auto;">
        <p class="preview-filename">{uploaded_file.name}</p>
    </div>
    """, unsafe_allow_html=True)

    col_btn1, col_btn2, col_btn3 = st.columns([1.5, 1, 1.5])
    with col_btn2:
        predict_clicked = st.button("PREDICT", use_container_width=True)

    if predict_clicked:
        model = load_model()

        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        input_tensor = preprocess(image).unsqueeze(0)

        # Analyzing label + progress bar
        st.markdown('<p class="analyzing-label">Analyzing</p>', unsafe_allow_html=True)
        progress = st.progress(0)

        progress.progress(15)
        output = model(input_tensor)
        progress.progress(50)
        probs = torch.softmax(output, dim=1)[0]
        conf, pred_idx = torch.max(probs, 0)
        conf_value = conf.item()
        progress.progress(75)

        # ── Confidence threshold check ───────────────────────────────────
        if conf_value < CONFIDENCE_THRESHOLD:
            progress.progress(100)
            st.markdown(f"""
            <div class="warn-badge">
                <p class="warn-title">⚠️ Not a breast disease histopathology image</p>
                <p class="warn-sub">Confidence too low ({conf_value*100:.2f}%) — please upload a valid breast tissue histopathology image.</p>
            </div>
            """, unsafe_allow_html=True)

        else:
            cam = generate_gradcam(model, input_tensor, pred_idx)
            progress.progress(80)
            overlay = overlay_heatmap(image, cam)
            progress.progress(90)
            
            # Run comparison models
            eff_model = load_efficientnet_model()
            with torch.no_grad():
                eff_out = eff_model(input_tensor)
            eff_probs = torch.softmax(eff_out, dim=1)[0]
            conf_eff, pred_idx_eff = torch.max(eff_probs, 0)
            
            dense_model = load_densenet_model()
            with torch.no_grad():
                dense_out = dense_model(input_tensor)
            dense_probs = torch.softmax(dense_out, dim=1)[0]
            conf_dense, pred_idx_dense = torch.max(dense_probs, 0)

            progress.progress(100)

            # Result badge
            st.markdown(f"""
            <div class="result-badge">
                <span class="result-name">{CLASS_NAMES[pred_idx]}</span>
                <span class="result-conf">{conf_value*100:.2f}% confidence</span>
            </div>
            """, unsafe_allow_html=True)

            # Original + Grad-CAM via standard Streamlit columns for fullsize capability
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<p style="font-size:12px;color:rgba(255,255,255,0.7);text-transform:uppercase;margin-bottom:8px;font-weight:500;">Original</p>', unsafe_allow_html=True)
                st.image(image.resize((224, 224)), use_container_width=True)
            with col2:
                st.markdown('<p style="font-size:12px;color:rgba(255,255,255,0.7);text-transform:uppercase;margin-bottom:8px;font-weight:500;">Grad-CAM</p>', unsafe_allow_html=True)
                st.image(overlay, use_container_width=True)

            # Horizontal Heatmap Legend
            st.markdown("""
            <div style="background:rgba(255,255,255,0.04); border:0.5px solid rgba(255,255,255,0.1); border-radius:8px; padding:12px 16px; margin-top:8px; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
                <span style="font-size:11px; color:rgba(255,255,255,0.4); text-transform:uppercase; letter-spacing:0.5px; font-weight:600;">Heatmap Guide</span>
                <div style="display:flex; gap:16px; flex-wrap:wrap;">
                    <div style="display:flex; align-items:center; gap:6px;">
                        <div style="width:12px;height:12px;background:#ff2a00;border-radius:2px;"></div>
                        <span style="font-size:12px;color:rgba(255,255,255,0.7);">Red (Strong)</span>
                    </div>
                    <div style="display:flex; align-items:center; gap:6px;">
                        <div style="width:12px;height:12px;background:#ffcc00;border-radius:2px;"></div>
                        <span style="font-size:12px;color:rgba(255,255,255,0.7);">Yellow (Mid)</span>
                    </div>
                    <div style="display:flex; align-items:center; gap:6px;">
                        <div style="width:12px;height:12px;background:#00cc88;border-radius:2px;"></div>
                        <span style="font-size:12px;color:rgba(255,255,255,0.7);">Green (Low)</span>
                    </div>
                    <div style="display:flex; align-items:center; gap:6px;">
                        <div style="width:12px;height:12px;background:#0033aa;border-radius:2px;"></div>
                        <span style="font-size:12px;color:rgba(255,255,255,0.7);">Blue (Ignored)</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            res_circle = make_circular_progress(conf_value, "ResNet50", CLASS_NAMES[pred_idx], "#3b82f6")
            eff_circle = make_circular_progress(conf_eff.item(), "EfficientNet-V2", CLASS_NAMES[pred_idx_eff], "#10b981")
            den_circle = make_circular_progress(conf_dense.item(), "DenseNet121", CLASS_NAMES[pred_idx_dense], "#8b5cf6")
            
            st.markdown("""<h3 style="font-size:14px; font-weight:500; margin-top:24px; margin-bottom:12px; color:rgba(255,255,255,0.8); text-align:center;">Model Comparison</h3>""", unsafe_allow_html=True)
            
            mod_col1, mod_col2, mod_col3 = st.columns(3)
            with mod_col1:
                st.markdown(res_circle, unsafe_allow_html=True)
                with st.expander("All probabilities"):
                    for i, name in enumerate(CLASS_NAMES):
                        st.markdown(f"<div style='font-size:11px;display:flex;justify-content:space-between;color:rgba(255,255,255,0.8);margin-bottom:4px;'><span>{name}</span><span style='font-weight:600;'>{probs[i].item()*100:.1f}%</span></div>", unsafe_allow_html=True)
            with mod_col2:
                st.markdown(eff_circle, unsafe_allow_html=True)
                with st.expander("All probabilities"):
                    for i, name in enumerate(CLASS_NAMES):
                        st.markdown(f"<div style='font-size:11px;display:flex;justify-content:space-between;color:rgba(255,255,255,0.8);margin-bottom:4px;'><span>{name}</span><span style='font-weight:600;'>{eff_probs[i].item()*100:.1f}%</span></div>", unsafe_allow_html=True)
            with mod_col3:
                st.markdown(den_circle, unsafe_allow_html=True)
                with st.expander("All probabilities"):
                    for i, name in enumerate(CLASS_NAMES):
                        st.markdown(f"<div style='font-size:11px;display:flex;justify-content:space-between;color:rgba(255,255,255,0.8);margin-bottom:4px;'><span>{name}</span><span style='font-weight:600;'>{dense_probs[i].item()*100:.1f}%</span></div>", unsafe_allow_html=True)