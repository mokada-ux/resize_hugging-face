import streamlit as st
from huggingface_hub import InferenceClient
from PIL import Image
import io

# --- 設定 ---
# 使用するモデル: 無料で安定しているInpaintingモデル（SD 1.5ベース）
MODEL_ID = "runwayml/stable-diffusion-inpainting"

# --- ページ設定 ---
st.set_page_config(page_title="AI背景拡張", layout="wide")
st.title("🎨 AI広告画像メーカー (背景拡張)")
st.markdown("""
画像をアップロードすると、足りない背景をAIが自動で描き足して
広告用のサイズ（正方形・横長・バナー）を作成します。
**Hugging Faceの無料APIを使用しています。**
""")

# --- サイドバー: APIキー入力 ---
# セキュリティのため、パスワードのように入力させます
api_token = st.sidebar.text_input("Hugging Face Token", type="password", help="Hugging Faceで取得したToken(Write権限)を入力してください")

# --- 関数: AIによる背景拡張 ---
def ai_expand(api_token, image, target_w, target_h):
    # クライアントの準備
    client = InferenceClient(token=api_token)
    
    orig_w, orig_h = image.size
    
    # 1. キャンバスの作成（リサイズして中央配置）
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    resized_img = image.resize((new_w, new_h), Image.LANCZOS)
    
    # ベース画像（背景黒）
    background = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    background.paste(resized_img, (paste_x, paste_y))
    
    # 2. マスク作成（白=描き直す、黒=残す）
    mask = Image.new("L", (target_w, target_h), 255) # 全体を描き直す設定
    mask_keep = Image.new("L", (new_w, new_h), 0)    # 元画像部分は守る
    mask.paste(mask_keep, (paste_x, paste_y))
    
    # 3. APIリクエスト
    # プロンプト: 高品質な背景、シームレスな拡張
    prompt = "high quality background, seamless extension, photorealistic, 4k, cinematic lighting, no text"
    negative_prompt = "text, watermark, low quality, distorted, blurry, ugly, bad anatomy, frame, borders"

    try:
        # Inpainting実行
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
        st.error(f"エラーが発生しました: {e}")
        return None

# --- メイン処理 ---
uploaded_file = st.file_uploader("画像をアップロード", type=['jpg', 'png'])

if uploaded_file and api_token:
    input_image = Image.open(uploaded_file).convert("RGB")
    
    st.divider()
    st.subheader("生成プレビュー")
    
    # 生成ボタン
    if st.button("🚀 AI生成開始 (少し時間がかかります)"):
        
        # 3つのサイズを定義 (APIの仕様上、8の倍数が安全です)
        targets = [
            (512, 512, "正方形 (Instagram)"), 
            (768, 432, "横長 (Web/YouTube)"), # 1920x1080の比率に近い小型版
            (600, 400, "バナー (Web広告)")
        ]
        
        cols = st.columns(3)
        
        for i, (w, h, label) in enumerate(targets):
            with cols[i]:
                st.write(f"⏳ {label} 生成中...")
                
                # AI処理実行
                # ※無料APIは大きいサイズ(1000px以上)だとエラーになりやすいので、
                # 小さめに作って必要なら後で拡大するのがコツです。
                result_img = ai_expand(api_token, input_image, w, h)
                
                if result_img:
                    st.image(result_img, caption=f"{label}", use_container_width=True)
                    
                    # ダウンロードボタン
                    buf = io.BytesIO()
                    result_img.save(buf, format="JPEG", quality=95)
                    st.download_button(
                        label="保存",
                        data=buf.getvalue(),
                        file_name=f"ai_bg_{w}x{h}.jpg",
                        mime="image/jpeg"
                    )
    
elif not api_token:
    st.warning("👈 左のサイドバーにHugging Faceのトークンを入力してください")