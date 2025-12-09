import streamlit as st
from huggingface_hub import InferenceClient
from PIL import Image
import io

# --- 設定 ---
MODEL_ID = "runwayml/stable-diffusion-inpainting"

# --- ページ設定 ---
st.set_page_config(page_title="AI背景拡張", layout="wide")
st.title("🎨 AI広告画像メーカー (背景拡張)")
st.markdown("""
画像をアップロードすると、足りない背景をAIが自動で描き足します。
Hugging Faceの無料APIを使用しています。
""")

# --- 【変更点】SecretsからAPIキーを自動読み込み ---
try:
    # Streamlit CloudのSecretsから取得
    api_token = st.secrets["HF_TOKEN"]
except Exception:
    st.error("⚠️ 設定エラー: APIトークンが見つかりません。Streamlit CloudのSettings > Secrets に 'HF_TOKEN' を設定してください。")
    st.stop() # キーがない場合はここで停止

# --- 関数: AIによる背景拡張 ---
def ai_expand(api_token, image, target_w, target_h):
    client = InferenceClient(token=api_token)
    orig_w, orig_h = image.size
    
    # 1. キャンバス作成
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    resized_img = image.resize((new_w, new_h), Image.LANCZOS)
    
    background = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    background.paste(resized_img, (paste_x, paste_y))
    
    # 2. マスク作成
    mask = Image.new("L", (target_w, target_h), 255)
    mask_keep = Image.new("L", (new_w, new_h), 0)
    mask.paste(mask_keep, (paste_x, paste_y))
    
    # 3. 生成リクエスト
    prompt = "high quality background, seamless extension, photorealistic, 4k, cinematic lighting, no text"
    negative_prompt = "text, watermark, low quality, distorted, blurry, ugly, bad anatomy, frame, borders"

    try:
        output_image = client.text_to_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=background,
            mask_image=mask,
            model=MODEL_ID,
            height=target_h,
            width=target_w,
            num_inference_steps=25,
            guidance_scale=7.5,
        )
        return output_image
    except Exception as e:
        st.error(f"生成エラー: {e}")
        return None

# --- メイン処理 ---
uploaded_file = st.file_uploader("画像をアップロード", type=['jpg', 'png'])

if uploaded_file:
    input_image = Image.open(uploaded_file).convert("RGB")
    
    st.divider()
    
    if st.button("🚀 AI生成開始"):
        # 安全のためサイズは控えめに
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
