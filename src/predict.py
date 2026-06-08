"""학습된 모델로 새 고양이 사진 예측 (CLI)"""

import sys
import numpy as np
from pathlib import Path
from PIL import Image

IMG_SIZE = 64


def predict(image_path: str, mode: str = "gray"):
    import tensorflow as tf
    from tensorflow import keras

    # 모델 경로 선택
    if mode == "color":
        model_path = Path("models/best_cat_color.keras")
        channels = 3
        convert_mode = "RGB"
    else:
        model_path = Path("models/best_cat_model.keras")
        channels = 1
        convert_mode = "L"

    if not model_path.exists():
        print(f"❌ 모델 파일이 없습니다: {model_path}")
        print("   먼저 노트북에서 학습을 실행하세요.")
        sys.exit(1)

    model = keras.models.load_model(model_path)

    # 이미지 전처리
    img = Image.open(image_path).convert(convert_mode)
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img).astype("float32") / 255.0
    img_array = img_array.reshape(1, IMG_SIZE, IMG_SIZE, channels)

    # 예측
    prob = model.predict(img_array, verbose=0)[0][0]

    if prob > 0.5:
        print(f"🐱 내 고양이 맞아! (확신도: {prob:.1%})")
    else:
        print(f"😿 내 고양이 아님    (확신도: {1-prob:.1%})")

    return prob


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: uv run python src/predict.py <이미지경로> [gray|color]")
        print("예시:   uv run python src/predict.py test_images/photo.jpg color")
        sys.exit(1)

    img_path = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "gray"
    predict(img_path, mode)