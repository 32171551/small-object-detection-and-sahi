import cv2
import os
import json
from PIL import Image
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

model = r"D:\small object detection\250711_yolov8-p2_aihub\best.pt" #사용할 모델
video_path = r"D:\small object detection\test_video\DJI_0804_0001_30m_1.mp4" #사용할 영상
result_path = r"D:\small object detection\test_video\frames" #저장 위치


# 모델 로딩
detection_model = AutoDetectionModel.from_pretrained(
    model_type='ultralytics',  # or 'mmdet', 'torchvision', etc.
    model_path=model,
    confidence_threshold=0.5,
    device='cuda:0'
)

# 동영상 열기
cap = cv2.VideoCapture(video_path)

# 저장 경로
os.makedirs(result_path, exist_ok=True)

# 결과 누적 리스트
all_coco_annotations = []
all_coco_predictions = []

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    result = get_sliced_prediction(
        image=pil_image,
        detection_model=detection_model,
        slice_height=256,
        slice_width=256,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2
    )

    result.export_visuals(
        export_dir=result_path,
        file_name=f"frame_{frame_idx:05d}"
    )

    # COCO Annotation
    anns = result.to_coco_annotations()
    for ann in anns:
        ann["image_id"] = frame_idx
    all_coco_annotations.extend(anns)

    # COCO Prediction
    preds = result.to_coco_predictions(image_id=frame_idx)
    all_coco_predictions.extend(preds)

    frame_idx += 1

cap.release()

#with open("outputs/coco_annotations.json", "w") as f:
#    json.dump(all_coco_annotations, f)

file_path = os.path.join(result_path, "coco_predictions.json")
with open(file_path, "w") as f:
    json.dump(all_coco_predictions, f)
