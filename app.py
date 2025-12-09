import streamlit as st
from huggingface_hub import InferenceClient
import base64
import io
import time  # 待機用に時間を操るライブラリを追加
from PIL import Image

# --- 設定 ---
# モデル: StabilityAI Stable Diffusion 2 Inpainting
MODEL_ID = "stabilityai/stable-diffusion-2-inpainting"

# --- ページ設定 ---
st.set_page_config(page_title="AI背景拡張", layout="wide")
st.title("🎨 AI広告画像メーカー (自動リトライ版)")
st.markdown("画像をドロップすると、指定したサイズに合わせてAIが背景を拡張します。")

# --- SecretsからAPIキーを読み込み ---
try:
    api_token = st.secrets["HF_TOKEN"]
except Exception:
    st.error("⚠️ 設定エラー: APIトークンが見つかりません。")
    st.stop()

# --- 便利関数 ---
def image_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# --- 関数: AIによる背景拡張 ---
def ai_expand(api_token, image, target_w, target_h):
    # エラー対策: 生成サイズが大きすぎると無料枠では落ちるので
    # 最大1024px以下に抑えて生成し、最後に本来のサイズに戻す
    gen_w, gen_h = target_w, target_h
    scale_factor = 1.0

    if target_w > 1024 or target_h > 1024:
        scale_factor = 0.6 # 少し画質を落として成功率を上げる
        gen_w = int(target_w * scale_factor)
        gen_h = int(target_h * scale_factor)
    
    # 1. キャンバス作成
    orig_w, orig_h = image.size
    scale = min(gen_w / orig_w, gen_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    resized_img = image.resize((new_w, new_h), Image.LANCZOS)
    
    background = Image.new("RGB", (gen_w, gen_h), (0, 0, 0))
    paste_x = (gen_w - new_w) // 2
    paste_y = (gen_h - new_h) // 2
    background.paste(resized_img, (paste_x, paste_y))
    
    # 2. マスク作成
    mask = Image.new("L", (gen_w, gen_h), 255) 
    mask_keep = Image.new("L", (new_w, new_h), 0)
    mask.paste(mask_keep, (paste_x, paste_y))
    
    # 3. APIリクエスト準備
    client = InferenceClient(token=api_token)
    payload = {
        "inputs": "high quality background, seamless extension, photorealistic, 4k, cinematic lighting, no text",
        "parameters": {
            "negative_prompt": "text, watermark, low quality, distorted, blurry, ugly, bad anatomy, frame, borders",
            "num_inference_steps": 25,
            "guidance_scale": 7.5,
            "image": image_to_base64(background),
            "mask_image": image_to_base64(mask)
        }
    }

    # ★ここから改良点: 粘り強くリトライするループ★
    max_retries = 3  # 最大3回挑戦する
