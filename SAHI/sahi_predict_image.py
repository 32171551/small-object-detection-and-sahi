import os, json
from sahi import AutoDetectionModel
from sahi.predict import predict

model = r"D:\small_object_detection\250825_yolov11-p2-SeaSee-x\weights\best.pt"
image_dir = r"D:\small_object_detection\MOT_sample_image" #추론에 사용할 폴더 이름
result_dir = r"D:\small_object_detection\runs\sahi-predict\MOT 데이터 추론" #결과 저장 위치

os.makedirs(result_dir, exist_ok=True)

detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=model,
    confidence_threshold=0.3,
    device="cuda:0"   # 여기서 GPU/CPU 선택
)

result = predict(
    detection_model=detection_model,
    model_path=model,
    source=image_dir,
    slice_height=256,
    slice_width=256,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
    return_dict=True,
    project=result_dir,
    name="250825_yolov11-p2-SeaSee-x", #저장할 폴더 이름
    device="cuda:0"
)

print(result.keys())
# 보통 'images', 'coco_prediction_list', 'count' 같은 키를 포함합니다.
all_coco_predictions = result.get("coco_prediction_list", [])

out_json = os.path.join(result_dir, "coco_predictions.json")
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(all_coco_predictions, f, ensure_ascii=False, indent=2)
