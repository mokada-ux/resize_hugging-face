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
    paste_y = (gen_h - new_h) // 2
    background.paste(resized_img, (paste_x, paste_y))
    
    # マスク作成
    mask = Image.new("L", (gen_w, gen_h), 255) 
    mask_keep = Image.new("L", (new_w, new_h), 0)
    mask.paste(mask_keep, (paste_x, paste_y))
    
    # APIリクエスト準備
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

    # リトライ処理
    max_retries = 3
    for attempt in range(max_retries):
        try:
            image_bytes = client.post(json=payload, model=MODEL_ID)
            generated_img = Image.open(io.BytesIO(image_bytes))

            if scale_factor != 1.0:
                generated_img = generated_img.resize((target_w, target_h), Image.LANCZOS)
                
            return generated_img

        except Exception as e:
            print(f"試行 {attempt+1}/{max_retries} 失敗: {e}")
            if attempt < max_retries - 1:
                time.sleep(3)
            else:
                return None
    return None

# --- 6. 画像がアップロードされた後の処理 ---
if uploaded_file:
    # 画像読み込み
    input_image = Image.open(uploaded_file).convert("RGB")
    
    # 元画像をサイドバーに表示（メイン画面をすっきりさせるため）
    st.sidebar.image(input_image, caption="元の画像", use_container_width=True)
    st.sidebar.info("🚀 生成を開始しました！")
    
    st.divider()
    
    # 生成ターゲット
    targets = [
        (1080, 1080, "正方形 (Instagram)"), 
        (1920, 1080, "横長 (YouTube/Web)"), 
        (600, 400, "バナー (広告)")
    ]
    
    cols = st.columns(len(targets))
    progress_bar = st.progress(0)
    
    for i, (w, h, label) in enumerate(targets):
        with cols[i]:
            status_text = st.empty()
            status_text.info(f"⏳ {label} 生成中...")
            
            result_img = ai_expand(api_token, input_image, w, h)
            
            if result_img:
                status_text.empty()
                st.image(result_img, use_container_width=True)
                
                buf = io.BytesIO()
                result_img.save(buf, format="JPEG", quality=95)
                st.download_button(
                    label="保存",
                    data=buf.getvalue(),
                    file_name=f"ai_bg_{w}x{h}.jpg",
                    mime="image/jpeg",
                    key=f"btn_{i}"
                )
            else:
                status_text.error("混雑中 (再試行してください)")
        
        progress_bar.progress((i + 1) / len(targets))

    st.success("🎉 すべて完了しました！")
