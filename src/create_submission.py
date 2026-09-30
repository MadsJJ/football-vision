import pandas as pd
from ultralytics import RTDETR

MODEL_PATH = "models/best.pt"
VIDEO_PATH = "data/RBK-AALESUND/aalesund.mp4"

model = RTDETR(MODEL_PATH)
results = model.track(
    source=VIDEO_PATH,
    tracker="botsort.yaml",
    persist=True,
    # conf=0.25,
    stream=True,
)

track_output = []
for frame_idx, result in enumerate(results):
    if result.boxes is None or result.boxes.id is None:
        continue
    height, width = result.orig_shape[:2]
    boxes = result.boxes.xyxy.cpu().numpy()
    ids = result.boxes.id.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    for track_id, box, cls in zip(ids, boxes, classes):
        x1, y1, x2, y2 = box
        x1 /= width
        y1 /= height
        x2 /= width
        y2 /= height
        track_output.append([frame_idx, int(track_id), int(cls), x1, y1, x2, y2])

# Sort by frame number and track id
track_output.sort(key=lambda row: (row[0], row[1]))

# Save to MOT-format CSV
track_df = pd.DataFrame(
    track_output,
    columns=["frame_id", "track_id", "class_id", "x1", "y1", "x2", "y2"]
)
track_df.to_csv("tracker_output.csv", index=False, header=True)
print(f"Tracker output saved as tracker_output.csv ({len(track_output)} rows)")
