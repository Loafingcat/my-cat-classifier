"""🐱 내 고양이 감별기 — Streamlit Dashboard"""

import streamlit as st
import numpy as np
from pathlib import Path
from PIL import Image

IMG_SIZE = 64

# ─────────────────────────────────────────────
# 페이지 설정
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="내 고양이 감별기",
    page_icon="🐱",
    layout="centered",
)

# ─────────────────────────────────────────────
# 커스텀 CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Gabarito:wght@400;700;900&family=Noto+Sans+KR:wght@400;500;700;900&display=swap');

/* 전체 배경 */
.stApp {
    background: linear-gradient(160deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
}

/* 메인 타이틀 */
.main-title {
    font-family: 'Gabarito', 'Noto Sans KR', sans-serif;
    font-size: 3.2rem;
    font-weight: 900;
    text-align: center;
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 50%, #fda085 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
    letter-spacing: -1px;
}

.subtitle {
    font-family: 'Noto Sans KR', sans-serif;
    text-align: center;
    color: #888;
    font-size: 1.05rem;
    margin-top: 4px;
    margin-bottom: 2rem;
}

/* 결과 카드 */
.result-card {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    backdrop-filter: blur(10px);
    margin-top: 1.5rem;
}

.result-card.mine {
    border-color: rgba(72, 219, 147, 0.4);
    box-shadow: 0 0 40px rgba(72, 219, 147, 0.1);
}

.result-card.other {
    border-color: rgba(245, 87, 108, 0.4);
    box-shadow: 0 0 40px rgba(245, 87, 108, 0.1);
}

.result-emoji {
    font-size: 4rem;
    margin-bottom: 0.5rem;
}

.result-label {
    font-family: 'Gabarito', 'Noto Sans KR', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}

.result-label.mine { color: #48db93; }
.result-label.other { color: #f5576c; }

.confidence {
    font-family: 'Gabarito', sans-serif;
    font-size: 3rem;
    font-weight: 900;
    color: #fff;
    margin: 0.5rem 0;
}

/* 프로그레스 바 */
.prob-bar-bg {
    width: 100%;
    height: 12px;
    background: rgba(255,255,255,0.08);
    border-radius: 6px;
    overflow: hidden;
    margin: 1rem auto;
    max-width: 400px;
}

.prob-bar-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.8s ease;
}

.prob-bar-fill.mine {
    background: linear-gradient(90deg, #48db93, #3bb078);
}

.prob-bar-fill.other {
    background: linear-gradient(90deg, #f5576c, #c9303e);
}

/* 업로드 영역 */
.upload-zone {
    background: rgba(255,255,255,0.02);
    border: 2px dashed rgba(255,255,255,0.15);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
}

/* 모델 상태 뱃지 */
.status-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-family: 'Noto Sans KR', sans-serif;
    font-weight: 500;
}

.status-badge.ready {
    background: rgba(72, 219, 147, 0.15);
    color: #48db93;
    border: 1px solid rgba(72, 219, 147, 0.3);
}

.status-badge.missing {
    background: rgba(245, 87, 108, 0.15);
    color: #f5576c;
    border: 1px solid rgba(245, 87, 108, 0.3);
}

/* 사이드바 */
section[data-testid="stSidebar"] {
    background: rgba(10, 10, 10, 0.95);
}

/* 모드 선택 라디오 */
div[data-testid="stRadio"] label {
    color: #ccc !important;
}

/* 파일 업로더 */
section[data-testid="stFileUploader"] {
    border: 2px dashed rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 0.5rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 모델 로드 (캐싱)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model(model_path: str):
    """학습된 Keras 모델 로드 (한 번만 로드 후 캐싱)"""
    import tensorflow as tf
    return tf.keras.models.load_model(model_path)


def preprocess_image(uploaded_file, channels: int):
    """업로드된 이미지를 모델 입력 형태로 전처리"""
    img = Image.open(uploaded_file)

    if channels == 1:
        img = img.convert("L")
    else:
        img = img.convert("RGB")

    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized).astype("float32") / 255.0
    img_array = img_array.reshape(1, IMG_SIZE, IMG_SIZE, channels)

    return img, img_array


# ─────────────────────────────────────────────
# 헤더
# ─────────────────────────────────────────────
st.markdown('<div class="main-title">🐱 내 고양이 감별기</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">사진을 올리면 우리 집 고양이인지 알려드려요</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 사이드바 설정
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ 설정")

    mode = st.radio(
        "모델 선택",
        ["흑백 (1채널)", "컬러 (3채널)"],
        help="01 노트북 = 흑백, 03 노트북 = 컬러",
    )

    if mode == "흑백 (1채널)":
        model_path = "models/best_cat_model.keras"
        channels = 1
    else:
        model_path = "models/best_cat_color.keras"
        channels = 3

    threshold = st.slider(
        "판별 기준 (threshold)",
        min_value=0.3,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="이 값보다 높으면 '내 고양이'로 판별",
    )

    st.markdown("---")

    # 모델 상태 확인
    st.markdown("### 📦 모델 상태")
    if Path(model_path).exists():
        st.markdown(
            '<span class="status-badge ready">✅ 모델 로드 가능</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-badge missing">❌ 모델 파일 없음</span>',
            unsafe_allow_html=True,
        )
        st.caption(f"`{model_path}`를 찾을 수 없습니다. 노트북에서 학습을 먼저 실행하세요.")

    st.markdown("---")
    st.markdown("### 📊 모델 정보")
    st.caption(f"입력 크기: {IMG_SIZE}×{IMG_SIZE}×{channels}")
    st.caption(f"구조: Conv→Pool ×2 + Dense + Dropout")
    st.caption(f"프레임워크: TensorFlow / Keras")

# ─────────────────────────────────────────────
# 메인: 이미지 업로드
# ─────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "고양이 사진을 올려주세요",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
    help="JPG, PNG, WebP 등 일반 이미지 파일",
)

# ─────────────────────────────────────────────
# 예측 실행
# ─────────────────────────────────────────────
if uploaded_file is not None:
    if not Path(model_path).exists():
        st.error(f"⚠️ 모델 파일(`{model_path}`)이 없습니다. 노트북에서 학습을 먼저 실행해주세요.")
        st.stop()

    # 모델 로드
    model = load_model(model_path)

    # 전처리 & 예측
    original_img, img_array = preprocess_image(uploaded_file, channels)
    prob = float(model.predict(img_array, verbose=0)[0][0])

    is_my_cat = prob >= threshold
    confidence = prob if is_my_cat else (1 - prob)

    # 레이아웃: 이미지 + 결과
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.image(original_img, caption="업로드된 사진", use_container_width=True)

    with col2:
        cls = "mine" if is_my_cat else "other"
        emoji = "😻" if is_my_cat else "🙀"
        label = "우리 고양이!" if is_my_cat else "다른 고양이"

        st.markdown(f"""
        <div class="result-card {cls}">
            <div class="result-emoji">{emoji}</div>
            <div class="result-label {cls}">{label}</div>
            <div class="confidence">{confidence:.1%}</div>
            <div style="color:#888; font-size:0.9rem; margin-bottom:0.8rem;">확신도</div>
            <div class="prob-bar-bg">
                <div class="prob-bar-fill {cls}" style="width:{confidence*100:.0f}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 상세 확률
    with st.expander("📈 상세 확률 보기"):
        st.markdown(f"""
        | 클래스 | 확률 |
        |--------|------|
        | 내 고양이 (`my_cat`) | `{prob:.4f}` ({prob:.2%}) |
        | 다른 고양이 (`other_cats`) | `{1-prob:.4f}` ({1-prob:.2%}) |
        | 판별 기준 (threshold) | `{threshold}` |
        """)

        st.caption(
            f"모델 출력값 {prob:.4f}이 threshold {threshold}보다 "
            f"{'높으므로 → 내 고양이' if is_my_cat else '낮으므로 → 다른 고양이'}로 분류"
        )

else:
    # 업로드 전 안내
    st.markdown("""
    <div style="
        text-align:center;
        padding:3rem 2rem;
        color:#666;
        border:2px dashed rgba(255,255,255,0.08);
        border-radius:20px;
        margin-top:1rem;
    ">
        <div style="font-size:3rem; margin-bottom:0.5rem;">📸</div>
        <div style="font-size:1.1rem;">위에서 고양이 사진을 업로드하면</div>
        <div style="font-size:1.1rem;">우리 집 고양이인지 감별해드려요</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 푸터
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center; color:#555; font-size:0.8rem;">'
    'CNN (Conv→Pool×2 + Dense + Dropout) · TensorFlow/Keras · Day 4 프로젝트'
    '</div>',
    unsafe_allow_html=True,
)