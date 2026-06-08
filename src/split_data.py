"""data/raw 폴더의 사진을 train/val로 자동 분할 (80:20)"""

import shutil
import random
from pathlib import Path


def split_data(raw_dir: str = "data/raw", output_dir: str = "data", ratio: float = 0.8):
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)

    for category in ["my_cat", "other_cats"]:
        source = raw_path / category
        if not source.exists():
            print(f"⚠️  {source} 폴더가 없습니다. 사진을 넣어주세요.")
            continue

        images = [
            f for f in source.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp")
        ]

        if len(images) == 0:
            print(f"⚠️  {source}에 이미지가 없습니다.")
            continue

        random.shuffle(images)
        split_idx = int(len(images) * ratio)
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]

        # 기존 파일 정리 후 복사
        for split_name, img_list in [("train", train_imgs), ("val", val_imgs)]:
            dest = output_path / split_name / category
            if dest.exists():
                shutil.rmtree(dest)
            dest.mkdir(parents=True, exist_ok=True)

            for img in img_list:
                shutil.copy2(img, dest / img.name)

        print(f"[{category}] 전체 {len(images)}장 → 학습 {len(train_imgs)}장 / 검증 {len(val_imgs)}장")


if __name__ == "__main__":
    split_data()
    print("\n✅ 분할 완료! data/train, data/val 폴더를 확인하세요.")