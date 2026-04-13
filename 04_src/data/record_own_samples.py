import cv2
import numpy as np
from pathlib import Path
import time
import csv

# ── config ──────────────────────────────────────────────
SIGNS_TO_RECORD = [
    ("DRINK", 15),
    ("YES", 15),
    ("EAT", 15),
    ("BAD", 15),
]
SEQUENCE_LENGTH = 30
MIN_DETECTION_RATE = 0.60
SAVE_DIR = Path(r"C:\SAYSAI\02_clean_data\landmarks")
MANIFEST = SAVE_DIR / "landmark_manifest.csv"
MODEL_PATH = r"C:\SAYSAI\models\hand_landmarker.task"
# ────────────────────────────────────────────────────────

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

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
    return max(existing_ids, default=99000) + 1


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


def get_split(sample_idx: int, total: int) -> str:
    r = sample_idx / total
    if r < 0.6:
        return "train"
    elif r < 0.8:
        return "val"
    return "test"


def append_manifest(video_id, sign, split, save_path, num_frames, detection_rate, status):
    with open(MANIFEST, "a", encoding="utf-8") as f:
        f.write(
            f"{video_id},{sign},{split},"
            f"recorded,{save_path},{num_frames},126,"
            f"{detection_rate:.4f},{status}\n"
        )


def record_sign(sign: str, sample_idx: int, total_samples: int, video_id: int):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return None

    frames = []
    detected_frames = 0
    state = "READY"
    countdown = 3
    start_time = time.time()

    print(f"\n--- {sign} sample {sample_idx + 1}/{total_samples} ---")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        elapsed = time.time() - start_time

        if state == "READY":
            remaining = max(0, countdown - int(elapsed))
            cv2.putText(
                frame, f"GET READY: {sign}", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3
            )
            cv2.putText(
                frame, f"Starting in {remaining}s", (30, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2
            )
            cv2.putText(
                frame, "Press Q to quit", (30, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
            )

            if elapsed >= countdown:
                state = "RECORDING"
                start_time = time.time()
                frames = []
                detected_frames = 0

        elif state == "RECORDING":
            lm, detected = extract_landmarks(frame)
            frames.append(lm)
            if detected:
                detected_frames += 1

            progress = len(frames)
            det_rate = detected_frames / max(progress, 1)

            cv2.putText(
                frame, f"RECORDING {sign}", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3
            )
            cv2.putText(
                frame, f"Frames: {progress}/{SEQUENCE_LENGTH}", (30, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2
            )
            cv2.putText(
                frame, f"Detection rate: {det_rate:.2f}", (30, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2
            )

            if len(frames) >= SEQUENCE_LENGTH:
                state = "DONE"

        elif state == "DONE":
            det_rate = detected_frames / max(len(frames), 1)
            if det_rate >= MIN_DETECTION_RATE:
                msg = f"GOOD ({det_rate:.2f}) - SPACE save / R retry"
                color = (0, 255, 0)
            else:
                msg = f"WEAK ({det_rate:.2f}) - R retry"
                color = (0, 0, 255)

            cv2.putText(
                frame, msg, (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
            )
            cv2.putText(
                frame, "Q quit", (30, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
            )

        cv2.imshow("Record Own Samples", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            return None

        if state == "DONE":
            det_rate = detected_frames / max(len(frames), 1)

            if key == ord("r"):
                state = "READY"
                start_time = time.time()
                frames = []
                detected_frames = 0

            elif key == ord(" "):
                if det_rate < MIN_DETECTION_RATE:
                    print(f"Skipped weak sample for {sign} (detection_rate={det_rate:.2f})")
                    state = "READY"
                    start_time = time.time()
                    frames = []
                    detected_frames = 0
                    continue

                arr = np.array(frames, dtype=np.float32)
                save_path = SAVE_DIR / f"{video_id}.npy"
                np.save(save_path, arr)
                print(
                    f"Saved: {save_path} shape={arr.shape} detection_rate={det_rate:.2f}"
                )

                cap.release()
                cv2.destroyAllWindows()
                return save_path, det_rate

    cap.release()
    cv2.destroyAllWindows()
    return None


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    video_id = get_next_video_id()

    print("\n=== RECORDING TARGET GLOSSES ===")
    for sign, count in SIGNS_TO_RECORD:
        print(f"  {sign}: {count} samples")
    print(f"  sequence length: {SEQUENCE_LENGTH}")
    print(f"  minimum detection rate: {MIN_DETECTION_RATE:.2f}")

    for sign, total_samples in SIGNS_TO_RECORD:
        print(f"\n{'=' * 42}")
        print(f"SIGN: {sign}")
        print(f"You will record {total_samples} good samples")
        print("SPACE = save good sample")
        print("R = retry current sample")
        print("Q = quit")
        print(f"{'=' * 42}")
        input("Press ENTER when ready... ")

        i = 0
        while i < total_samples:
            split = get_split(i, total_samples)
            result = record_sign(sign, i, total_samples, video_id)

            if result is None:
                print("Stopped early.")
                return

            save_path, detection_rate = result
            append_manifest(
                video_id=video_id,
                sign=sign,
                split=split,
                save_path=save_path,
                num_frames=SEQUENCE_LENGTH,
                detection_rate=detection_rate,
                status="ok",
            )

            video_id += 1
            i += 1

    print("\nRecording complete.")
    print("Next:")
    print(r"1. python C:\SAYSAI\04_src\data\build_dataset_arrays.py")
    print(r"2. python C:\SAYSAI\04_src\training\train_cnn_transformer.py")
    print(r"3. export new sign_model.tflite")
    print(r"4. copy new sign_model.tflite into Android assets\models")


if __name__ == "__main__":
    main()