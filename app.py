import streamlit as st
from huggingface_hub import InferenceClient
import base64
import io
import time
from PIL import Image

# --- 設定 ---
# 複数のInpaintingモデルをリストアップ（優先順）
# 1つが混んでいても、他が空いていれば成功します
MODEL_CANDIDATES = [
    "stabilityai/stable-diffusion-2-inpainting",       # 最新・高画質
    "runwayml/stable-diffusion-inpainting",            # 定番・軽量
    "kandinsky-community/kandinsky-2-2-decoder-inpaint", # 別アーキテクチャ
]

# --- ページ設定 ---
st.set_page_config(page_title="AI背景拡張", layout="wide")
st.title("🎨 AI広告画像メーカー (混雑回避版)")
st.markdown("画像をドロップすると、空いているAIサーバーを探して背景を自動生成します。")

# --- Secrets読み込み ---
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
    # エラー対策: 生成サイズ調整 (最大800px程度に抑えて成功率アップ)
    gen_w, gen_h = target_w, target_h
    scale_factor = 1.0

    if target_w > 800 or target_h > 800:
        scale_factor = 0.6
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
    
    # Base64変換
    bg_b64 = image_to_base64(background)
    mask_b64 = image_to_base64(mask)

    # 3. APIリクエスト（モデルを順番に試す総力戦）
    client = InferenceClient(token=api_token)
    
    payload = {
        "inputs": "high quality background, seamless extension, photorealistic, 4k, cinematic lighting, no text",
        "parameters": {
            "negative_prompt": "text, watermark, low quality, distorted, blurry, ugly, bad anatomy, frame, borders",
            "num_inference_steps": 25,
            "guidance_scale": 7.5,
            "image": bg_b64,
            "mask_image": mask_b64
        }
    }

    # モデルリストを順番に試す
    for model_id in MODEL_CANDIDATES:
        # 各モデルにつき2回ずつリトライ
        for attempt in range(2):
            try:
                # 生データをPOST送信
                # URLをrouterに変更して自動ルーティングさせる
                image_bytes = client.post(json=payload, model=model_id)
                generated_img = Image.open(io.BytesIO(image_bytes))

                # サイズを戻す
                if scale_factor != 1.0:
                    generated_img = generated_img.resize((target_w, target_h), Image.LANCZOS)
                
                return generated_img, model_id # 成功したら画像とモデル名を返す

            except Exception as e:
                # 失敗したら少し待って次へ
                time.sleep(2)
                continue
    
    return None, None

# --- メイン処理 ---
uploaded_file = st.file_uploader("👇 ここに画像をドラッグ＆ドロップしてください", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    input_image = Image.open(uploaded_file).convert("RGB")
    
    st.sidebar.image(input_image, caption="元の画像", use_container_width=True)
    st.sidebar.info("🚀 生成を開始しました！")
    st.divider()
    
    targets = [
        (1080, 1080, "正方形 (Instagram)"), 
        (1920, 1080, "横長 (Web)"), 
        (600, 400, "バナー (広告)")
    ]
    
    cols = st.columns(len(targets))
    progress_bar = st.progress(0)
    
    for i, (w, h, label) in enumerate(targets):
        with cols[i]:
            status_text = st.empty()
            status_text.info(f"⏳ {label} 生成中...")
            
            result_img, used_model = ai_expand(api_token, input_image, w, h)
            
            if result_img:
                status_text.success(f"✅ 完了 (AI: {used_model.split('/')[0]})")
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
                status_text.error("全サーバー混雑中。時間を空けてください。")
        
        progress_bar.progress((i + 1) / len(targets))

    st.success("🎉 処理が終了しました")
