import streamlit as st
import requests
import base64
import io
from PIL import Image

# --- 設定 ---
# APIのURL (Hugging Faceの新URL)
API_URL = "https://router.huggingface.co/models/runwayml/stable-diffusion-inpainting"

# --- ページ設定 ---
st.set_page_config(page_title="AI背景拡張", layout="wide")
st.title("🎨 AI広告画像メーカー (背景拡張)")
st.markdown("""
画像をアップロードすると、足りない背景をAIが自動で描き足します。
Hugging Faceの無料APIを使用しています。
""")

# --- SecretsからAPIキーを読み込み ---
try:
    api_token = st.secrets["HF_TOKEN"]
except Exception:
    st.error("⚠️ 設定エラー: APIトークンが見つかりません。Streamlit CloudのSettings > Secrets に 'HF_TOKEN' を設定してください。")
    st.stop()

# --- 便利関数: 画像をBase64(文字)に変換 ---
def image_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# --- 関数: AIによる背景拡張 ---
def ai_expand(api_token, image, target_w, target_h):
    orig_w, orig_h = image.size
    
    # 1. キャンバス作成（リサイズして中央配置）
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    resized_img = image.resize((new_w, new_h), Image.LANCZOS)
    
    background = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    background.paste(resized_img, (paste_x, paste_y))
    
    # 2. マスク作成（白=描き直す、黒=残す）
    mask = Image.new("L", (target_w, target_h), 255) 
    mask_keep = Image.new("L", (new_w, new_h), 0)
    mask.paste(mask_keep, (paste_x, paste_y))
    
    # 3. APIリクエスト
    headers = {"Authorization": f"Bearer {api_token}"}
    
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

    # ★ここがエラーの原因だった箇所です（tryとexceptはセットである必要があります）
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        
        # エラーチェック
        if response.status_code != 200:
            st.error(f"APIエラー: {response.text}")
            return None
            
        # 画像データを読み込む
        image_bytes = response.content
        return Image.open(io.BytesIO(image_bytes))

    except Exception as e:
        st.error(f"通信エラー: {e}")
        return None

# --- メイン処理 ---
uploaded_file = st.file_uploader("画像をアップロード", type=['jpg', 'png'])

if uploaded_file:
    input_image = Image.open(uploaded_file).convert("RGB")
    st.image(input_image, caption="元の画像", width=300)
    st.divider()
    
    if st.button("🚀 AI生成開始 (約20〜30秒かかります)"):
        targets = [
            (512, 512, "正方形"), 
            (768, 432, "横長"), 
            (600, 400, "バナー")
        ]
        
        cols = st.columns(3)
        
        for i, (w, h, label) in enumerate(targets):
            with cols[i]:
                st.write(f"⏳ {label}...")
                result_img = ai_expand(api_token, input_image, w, h)
                
                if result_img:
                    st.image(result_img, use_container_width=True)
                    buf = io.BytesIO()
                    result_img.save(buf, format="JPEG", quality=95)
                    st.download_button(
                        label="保存",
                        data=buf.getvalue(),
                        file_name=f"ai_bg_{w}x{h}.jpg",
                        mime="image/jpeg"
                    )
