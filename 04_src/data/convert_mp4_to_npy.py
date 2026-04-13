import cv2
import csv
import numpy as np
from pathlib import Path

# ── config ──────────────────────────────────────────────
RAW_DIR = Path(r"C:\SAYSAI\01_raw_data\custom_mp4")
SAVE_DIR = Path(r"C:\SAYSAI\02_clean_data\landmarks")
MANIFEST = SAVE_DIR / "landmark_manifest.csv"
MODEL_PATH = r"C:\SAYSAI\models\hand_landmarker.task"

TARGET_FRAMES = 30
MIN_DETECTION_RATE = 0.60
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
# ────────────────────────────────────────────────────────

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

SAVE_DIR.mkdir(parents=True, exist_ok=True)

base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3,
    running_mode=RunningMode.IMAGE,
)
detector = HandLandmarker.create_from_options(options)


def folder_to_gloss(folder_name: str) -> str:
    return folder_name.replace("_", " ").upper().strip()


def get_next_video_id() -> int:
    existing_ids = []
    if MANIFEST.exists():
        with open(MANIFEST, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    existing_ids.append(int(row["video_id"]))
                except Exception:
                    pass
    return max(existing_ids, default=100000) + 1


def load_existing_video_paths():
    existing = set()
    if MANIFEST.exists():
        with open(MANIFEST, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                vp = row.get("video_path", "").strip()
                if vp:
                    existing.add(vp.lower())
    return existing


def sample_frames_evenly(frames, target_frames=30):
    if len(frames) == 0:
        return []

    if len(frames) == target_frames:
        return frames

    if len(frames) > target_frames:
        idxs = np.linspace(0, len(frames) - 1, num=target_frames).astype(int)
        return [frames[i] for i in idxs]

    # if fewer than target_frames, repeat last frame
    idxs = np.linspace(0, len(frames) - 1, num=target_frames)
    idxs = np.round(idxs).astype(int)
    return [frames[i] for i in idxs]


def extract_landmarks(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    left = np.zeros(63, dtype=np.float32)
    right = np.zeros(63, dtype=np.float32)
    detected_any = False

    if result.hand_landmarks and result.handedness:
        for lm_list, hd_list in zip(result.hand_landmarks, result.handedness):
            label = hd_list[0].category_name.upper()

            wx = lm_list[0].x
            wy = lm_list[0].y
            wz = lm_list[0].z

            coords = []
            for p in lm_list:
                coords += [p.x - wx, p.y - wy, p.z - wz]

            arr = np.array(coords, dtype=np.float32)
            scale = np.linalg.norm(arr[27:30])
            if scale > 1e-6:
                arr /= scale

            if np.abs(arr).sum() > 0:
                detected_any = True

            if label == "LEFT":
                left = arr
            else:
                right = arr

    merged = np.concatenate([left, right])
    return merged, detected_any


def read_all_frames(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    frames = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)

    cap.release()
    return frames


def get_split(index: int, total: int) -> str:
    r = index / total
    if r < 0.6:
        return "train"
    elif r < 0.8:
        return "val"
    return "test"


def append_manifest(video_id, gloss, split, video_path, landmark_path, num_frames, detection_rate, status):
    file_exists = MANIFEST.exists()
    with open(MANIFEST, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "video_id", "gloss", "split", "video_path", "landmark_path",
                "num_frames", "feature_dim", "detection_rate", "status"
            ])

        writer.writerow([
            video_id,
            gloss,
            split,
            str(video_path),
            str(landmark_path),
            num_frames,
            126,
            f"{detection_rate:.4f}",
            status,
        ])


def convert_video(video_path: Path, video_id: int):
    frames = read_all_frames(video_path)
    if len(frames) == 0:
        print(f"SKIP no frames: {video_path}")
        return None

    sampled = sample_frames_evenly(frames, TARGET_FRAMES)

    features = []
    detected_count = 0

    for frame in sampled:
        feat, detected = extract_landmarks(frame)
        features.append(feat)
        if detected:
            detected_count += 1

    arr = np.array(features, dtype=np.float32)
    detection_rate = detected_count / max(len(sampled), 1)

    if detection_rate < MIN_DETECTION_RATE:
        print(f"SKIP weak detection ({detection_rate:.2f}): {video_path}")
        return None

    save_path = SAVE_DIR / f"{video_id}.npy"
    np.save(save_path, arr)

    print(
        f"SAVED {video_path.name} -> {save_path.name} "
        f"shape={arr.shape} detection_rate={detection_rate:.2f}"
    )
    return save_path, detection_rate


def main():
    if not RAW_DIR.exists():
        print(f"RAW_DIR not found: {RAW_DIR}")
        return

    existing_video_paths = load_existing_video_paths()
    video_id = get_next_video_id()

    gloss_dirs = [d for d in RAW_DIR.iterdir() if d.is_dir()]
    gloss_dirs = sorted(gloss_dirs, key=lambda p: p.name.lower())

    if not gloss_dirs:
        print("No gloss folders found.")
        return

    total_saved = 0
    total_skipped = 0

    for gloss_dir in gloss_dirs:
        gloss = folder_to_gloss(gloss_dir.name)

        video_files = [
            p for p in gloss_dir.iterdir()
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS
        ]
        video_files = sorted(video_files, key=lambda p: p.name.lower())

        if not video_files:
            continue

        print(f"\n=== GLOSS: {gloss} ===")
        print(f"Found {len(video_files)} video(s)")

        for i, video_path in enumerate(video_files):
            if str(video_path).lower() in existing_video_paths:
                print(f"SKIP already in manifest: {video_path.name}")
                total_skipped += 1
                continue

            split = get_split(i, len(video_files))
            result = convert_video(video_path, video_id)

            if result is None:
                total_skipped += 1
                continue

            landmark_path, detection_rate = result
            append_manifest(
                video_id=video_id,
                gloss=gloss,
                split=split,
                video_path=video_path,
                landmark_path=landmark_path,
                num_frames=TARGET_FRAMES,
                detection_rate=detection_rate,
                status="ok",
            )

            video_id += 1
            total_saved += 1

    print("\n=== DONE ===")
    print(f"Saved samples: {total_saved}")
    print(f"Skipped samples: {total_skipped}")
    print("\nNext:")
    print(r"1. python C:\SAYSAI\04_src\data\build_dataset_arrays.py")
    print(r"2. python C:\SAYSAI\04_src\training\train_cnn_transformer.py")


if __name__ == "__main__":
    main()