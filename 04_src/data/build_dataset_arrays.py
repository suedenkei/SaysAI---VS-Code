from pathlib import Path
import csv
import json
import numpy as np

manifest_csv = Path(r"C:\SAYSAI\02_clean_data\landmarks\landmark_manifest.csv")
output_dir = Path(r"C:\SAYSAI\03_processed_data")
sequence_length = 30
feature_dim = 126

output_dir.mkdir(parents=True, exist_ok=True)

if not manifest_csv.exists():
    raise FileNotFoundError(f"Manifest not found: {manifest_csv}")

def fix_length(arr, target_len=30):
    """
    arr shape: (num_frames, feature_dim)
    Output shape: (target_len, feature_dim)
    """
    num_frames = arr.shape[0]

    if num_frames == target_len:
        return arr.astype(np.float32)

    if num_frames > target_len:
        # sample evenly across the sequence
        indices = np.linspace(0, num_frames - 1, target_len).astype(int)
        return arr[indices].astype(np.float32)

    # num_frames < target_len -> pad with last frame
    pad_count = target_len - num_frames
    last_frame = arr[-1:]
    pad = np.repeat(last_frame, pad_count, axis=0)
    return np.concatenate([arr, pad], axis=0).astype(np.float32)

with open(manifest_csv, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

ok_rows = [r for r in rows if r["status"] == "ok"]

if not ok_rows:
    raise RuntimeError("No successful landmark rows found.")

# Build label map from successful rows only
all_glosses = sorted(set(r["gloss"] for r in ok_rows))
label_to_index = {gloss: idx for idx, gloss in enumerate(all_glosses)}
index_to_label = {idx: gloss for gloss, idx in label_to_index.items()}

splits = {
    "train": {"X": [], "y": []},
    "val": {"X": [], "y": []},
    "test": {"X": [], "y": []},
}

for row in ok_rows:
    split = row["split"].strip().lower()
    gloss = row["gloss"]
    landmark_path = Path(row["landmark_path"])

    if split not in splits:
        continue

    if not landmark_path.exists():
        continue

    arr = np.load(landmark_path)

    if arr.ndim != 2:
        continue

    if arr.shape[1] != feature_dim:
        continue

    arr_fixed = fix_length(arr, target_len=sequence_length)
    label_idx = label_to_index[gloss]

    splits[split]["X"].append(arr_fixed)
    splits[split]["y"].append(label_idx)

# Convert to numpy arrays
for split_name in splits:
    X_list = splits[split_name]["X"]
    y_list = splits[split_name]["y"]

    if len(X_list) == 0:
        X = np.empty((0, sequence_length, feature_dim), dtype=np.float32)
        y = np.empty((0,), dtype=np.int64)
    else:
        X = np.stack(X_list).astype(np.float32)
        y = np.array(y_list, dtype=np.int64)

    np.save(output_dir / f"{split_name}_X.npy", X)
    np.save(output_dir / f"{split_name}_y.npy", y)

# Save label map
with open(output_dir / "label_map.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "label_to_index": label_to_index,
            "index_to_label": {str(k): v for k, v in index_to_label.items()},
            "sequence_length": sequence_length,
            "feature_dim": feature_dim,
        },
        f,
        indent=2,
        ensure_ascii=False,
    )

print("=== DATASET BUILD COMPLETE ===")
for split_name in splits:
    x_path = output_dir / f"{split_name}_X.npy"
    y_path = output_dir / f"{split_name}_y.npy"
    X = np.load(x_path)
    y = np.load(y_path)
    print(f"{split_name}: X shape = {X.shape}, y shape = {y.shape}")

print(f"Saved label map to: {output_dir / 'label_map.json'}")
print("Classes:")
for gloss, idx in label_to_index.items():
    print(f"  {idx}: {gloss}")