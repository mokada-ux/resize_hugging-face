import streamlit as st
from huggingface_hub import InferenceClient
import base64
import io
import time
from PIL import Image

# --- 1. ページ設定 (必ず一番上に書くルール) ---
st.set_page_config(page_title="AI背景拡張", layout="wide")

# --- 2. タイトルと説明 ---
st.title("🎨 AI広告画像メーカー (自動リトライ版)")
st.markdown("画像をドロップすると、指定したサイズに合わせてAIが背景を拡張します。")

# --- 3. 設定とSecrets読み込み ---
MODEL_ID = "stabilityai/stable-diffusion-2-inpainting"

try:
    api_token = st.secrets["HF_TOKEN"]
except Exception:
    st.error("⚠️ 設定エラー: APIトークンが見つかりません。")
    st.stop()

# --- 4. 画像アップロード場所 (ここが消えていました！) ---
uploaded_file = st.file_uploader("👇 ここに画像をドラッグ＆ドロップしてください", type=['jpg', 'png', 'jpeg'])

# --- 5. 便利関数とAI処理関数 ---
def image_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def ai_expand(api_token, image, target_w, target_h):
    # エラー対策: 生成サイズ調整
    gen_w, gen_h = target_w, target_h
    scale_factor = 1.0

    if target_w > 1024 or target_h > 1024:
        scale_factor = 0.6
        gen_w = int(target_w * scale_factor)
        gen_h = int(target_h * scale_factor)
    
    # キャンバス作成
    orig_w, orig_h = image.size
    scale = min(gen_w / orig_w, gen_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    resized_img = image.resize((new_w, new_h), Image.LANCZOS)
    
    background = Image.new("RGB", (gen_w, gen_h), (0, 0, 0))
    paste_x = (gen_w - new_w) // 2
    paste_y = (gen_h
