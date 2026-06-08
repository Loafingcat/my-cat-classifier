# 🐱 내 고양이 감별기 (My Cat Classifier)

CNN(합성곱 신경망)을 이용해 **내 고양이와 다른 고양이를 구분**하는 이진 분류 모델입니다.
학습된 모델을 Streamlit 대시보드에서 바로 테스트할 수 있습니다.

## 주요 기술

- **모델 구조**: Conv2D→MaxPooling2D ×2 + Dense + Dropout (Keras Sequential)
- **전이 없는 순수 CNN**: 사전훈련 모델 없이 직접 설계
- **데이터 증강**: reshape, 정규화, 흑백/컬러 변환
- **콜백**: ModelCheckpoint (최고 모델 자동 저장) + EarlyStopping (조기 종료)
- **대시보드**: Streamlit 기반 웹 UI에서 사진 업로드 → 즉시 판별

## 프로젝트 구조

```
my-cat-classifier/
├── app.py                         # Streamlit 대시보드
├── notebooks/
│   ├── 01_cat_classifier_cnn.ipynb   # 메인: 흑백 CNN + 이론 정리
│   ├── 02_filter_experiment.ipynb    # Conv2D(32) vs (64) 비교 실험
│   └── 03_color_extension.ipynb      # CHALLENGE: 3채널 컬러 CNN
├── src/
│   ├── split_data.py                 # raw → train/val 자동 분할
│   └── predict.py                    # CLI 예측
├── data/                             # .gitignore 처리됨
│   ├── raw/my_cat/                   # 내 고양이 사진 원본
│   ├── raw/other_cats/               # 다른 고양이 사진 원본
│   ├── train/                        # 학습용 (80%)
│   └── val/                          # 검증용 (20%)
└── models/                           # 학습된 모델 (.keras)
```

## 실행 방법

```bash
# 1. 환경 설정
uv venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv pip install tensorflow numpy matplotlib scikit-learn pillow streamlit ipykernel

# 2. 사진 준비
#    data/raw/my_cat/      ← 내 고양이 30~50장
#    data/raw/other_cats/  ← 다른 고양이 100~200장

# 3. 데이터 분할
python src/split_data.py

# 4. 노트북 학습 (VSCode에서 순서대로 실행)
#    01 → 02 → 03

# 5. 대시보드
streamlit run app.py
```

## 환경

- Python 3.12
- TensorFlow / Keras
- Streamlit