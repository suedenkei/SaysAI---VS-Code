from pathlib import Path
import csv
import cv2
import numpy as np
import mediapipe as mp

manifest_csv = Path(r"C:\SAYSAI\02_clean_data\manifests\wlasl_10_gloss_subset.csv")
model_path = Path(r"C:\SAYSAI\models\hand_landmarker.task")
output_dir = Path(r"C:\SAYSAI\02_clean_data\landmarks")
summary_csv = output_dir / "landmark_manifest.csv"

output_dir.mkdir(parents=True, exist_ok=True)

if not manifest_csv.exists():
    raise FileNotFoundError(f"Manifest not found: {manifest_csv}")

if not model_path.exists():
    raise FileNotFoundError(f"Model file not found: {model_path}")

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(model_path)),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3,
)

def empty_hand():
    return np.zeros(63, dtype=np.float32)

def normalize_hand(flat_coords):
    arr = np.array(flat_coords, dtype=np.float32).reshape(21, 3)
    wrist = arr[0].copy()
    arr = arr - wrist

    scale = np.linalg.norm(arr[9])
    if scale < 1e-6:
        scale = np.linalg.norm(arr[12])
    if scale < 1e-6:
        scale = 1.0

    arr = arr / scale
    return arr.reshape(-1).astype(np.float32)

def hand_to_flat(hand_landmarks):
    coords = []
    for lm in hand_landmarks:
        coords.extend([lm.x, lm.y, lm.z])
    return normalize_hand(coords)

def result_to_feature(result):
    left = empty_hand()
    right = empty_hand()

    if result.hand_landmarks and result.handedness:
        for hand_landmarks, handedness_list in zip(result.hand_landmarks, result.handedness):
            coords = hand_to_flat(hand_landmarks)

            label = ""
            if handedness_list and len(handedness_list) > 0:
                label = handedness_list[0].category_name.upper()

            if label == "LEFT":
                left = coords
            elif label == "RIGHT":
                right = coords
            else:
                if np.count_nonzero(left) == 0:
                    left = coords
                else:
                    right = coords

    return np.concatenate([left, right]).astype(np.float32)

def process_video(landmarker, video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    features = []
    detected_frames = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Use fixed 25 FPS timestamps: 0, 40, 80, 120, ...
        timestamp_ms = frame_idx * 40

        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        feat = result_to_feature(result)

        if np.count_nonzero(feat) > 0:
            detected_frames += 1

        features.append(feat)
        frame_idx += 1

    cap.release()

    if len(features) == 0:
        raise RuntimeError(f"No readable frames in video: {video_path}")

    arr = np.stack(features).astype(np.float32)
    detection_rate = detected_frames / len(features)

    return arr, len(features), detection_rate

def is_usable(row):
    return str(row.get("usable", "")).strip() in {"1", "True", "true"}

with open(manifest_csv, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

usable_rows = [r for r in rows if is_usable(r)]

if not usable_rows:
    raise RuntimeError("No usable rows found in manifest.")

print(f"Manifest rows: {len(rows)}")
print(f"Usable rows: {len(usable_rows)}")

summary_rows = []

for idx, row in enumerate(usable_rows, start=1):
    video_id = row["video_id"]
    gloss = row["normalized_gloss"] if "normalized_gloss" in row else row["gloss"]
    split = row["split"]
    video_path = Path(row["video_path"])

    if not video_path.exists():
        print(f"[{idx}/{len(usable_rows)}] MISSING FILE -> {video_id}")
        summary_rows.append({
            "video_id": video_id,
            "gloss": gloss,
            "split": split,
            "video_path": str(video_path),
            "landmark_path": "",
            "num_frames": 0,
            "feature_dim": 126,
            "detection_rate": 0.0,
            "status": "missing_video_file",
        })
        continue

    out_path = output_dir / f"{video_id}.npy"

    try:
        # Create a fresh landmarker per video so timestamps can restart at 0
        with HandLandmarker.create_from_options(options) as landmarker:
            arr, num_frames, detection_rate = process_video(landmarker, video_path)

        np.save(out_path, arr)

        print(f"[{idx}/{len(usable_rows)}] OK -> {video_id} | frames={num_frames} | detect={detection_rate:.2f}")

        summary_rows.append({
            "video_id": video_id,
            "gloss": gloss,
            "split": split,
            "video_path": str(video_path),
            "landmark_path": str(out_path),
            "num_frames": num_frames,
            "feature_dim": arr.shape[1],
            "detection_rate": round(float(detection_rate), 4),
            "status": "ok",
        })

    except Exception as e:
        print(f"[{idx}/{len(usable_rows)}] ERROR -> {video_id} | {e}")
        summary_rows.append({
            "video_id": video_id,
            "gloss": gloss,
            "split": split,
            "video_path": str(video_path),
            "landmark_path": "",
            "num_frames": 0,
            "feature_dim": 126,
            "detection_rate": 0.0,
            "status": f"error: {e}",
        })

with open(summary_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "video_id",
            "gloss",
            "split",
            "video_path",
            "landmark_path",
            "num_frames",
            "feature_dim",
            "detection_rate",
            "status",
        ],
    )
    writer.writeheader()
    writer.writerows(summary_rows)

ok_rows = [r for r in summary_rows if r["status"] == "ok"]

print("\n=== DONE ===")
print(f"Saved summary CSV: {summary_csv}")
print(f"Successful landmark files: {len(ok_rows)} / {len(summary_rows)}")

if ok_rows:
    rates = [float(r["detection_rate"]) for r in ok_rows]
    print(f"Average detection rate: {sum(rates)/len(rates):.3f}")