from pathlib import Path
import json
import csv
from collections import Counter

# =========================
# PATHS
# =========================


json_path = Path(r"C:\SAYSAI\01_raw_data\wlasl_10\start_kit\WLASL_v0.3.json")
videos_dir = Path(r"C:\SAYSAI\01_raw_data\wlasl_10\start_kit\videos")
output_csv = Path(r"C:\SAYSAI\02_clean_data\manifests\wlasl_10_gloss_subset.csv")
# =========================
# TARGET GLOSSES
# =========================
target_glosses = {
    "HELLO",
    "YES",
    "NO",
    "THANK YOU",
    "PLEASE",
    "HELP",
    "GOOD",
    "BAD",
    "EAT",
    "DRINK",
}

def normalize_gloss(text: str) -> str:
    return text.strip().upper().replace("-", " ").replace("_", " ")

def possible_video_paths(video_id: str):
    return [
        videos_dir / f"{video_id}.mp4",
        videos_dir / f"{video_id}.mkv",
        videos_dir / f"{video_id}.webm",
        videos_dir / f"{video_id}.avi",
    ]

# =========================
# CHECK PATHS
# =========================
if not json_path.exists():
    raise FileNotFoundError(f"JSON file not found: {json_path}")

if not videos_dir.exists():
    raise FileNotFoundError(f"Videos folder not found: {videos_dir}")

output_csv.parent.mkdir(parents=True, exist_ok=True)

print("Using JSON:", json_path)
print("Using videos folder:", videos_dir)
print("Saving CSV to:", output_csv)

# =========================
# LOAD JSON
# =========================
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# =========================
# FIND MATCHING GLOSSES
# =========================
normalized_targets = {normalize_gloss(g) for g in target_glosses}

all_dataset_glosses = sorted({item["gloss"] for item in data})
matched_dataset_glosses = []

print("\n=== MATCHED GLOSSES IN DATASET ===")
for gloss in all_dataset_glosses:
    if normalize_gloss(gloss) in normalized_targets:
        matched_dataset_glosses.append(gloss)
        print(gloss)

if not matched_dataset_glosses:
    raise ValueError("No matching glosses found in the WLASL JSON. Check the gloss names.")

matched_dataset_glosses_set = set(matched_dataset_glosses)

# =========================
# BUILD SUBSET ROWS
# =========================
rows = []

for item in data:
    gloss = item.get("gloss", "").strip()

    if gloss not in matched_dataset_glosses_set:
        continue

    instances = item.get("instances", [])

    for inst in instances:
        video_id = str(inst.get("video_id", "")).strip()
        split = str(inst.get("split", "")).strip()
        signer_id = str(inst.get("signer_id", "")).strip()
        frame_start = inst.get("frame_start", "")
        frame_end = inst.get("frame_end", "")
        fps = inst.get("fps", "")
        url = str(inst.get("url", "")).strip()

        found_path = None
        for p in possible_video_paths(video_id):
            if p.exists():
                found_path = str(p)
                break

        usable = 1 if found_path else 0
        reason = "" if found_path else "missing_video_file"

        rows.append({
            "gloss": gloss,
            "normalized_gloss": normalize_gloss(gloss),
            "video_id": video_id,
            "split": split,
            "signer_id": signer_id,
            "frame_start": frame_start,
            "frame_end": frame_end,
            "fps": fps,
            "video_path": found_path if found_path else "",
            "usable": usable,
            "reason": reason,
            "url": url,
        })

# =========================
# SAVE CSV
# =========================
with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "gloss",
            "normalized_gloss",
            "video_id",
            "split",
            "signer_id",
            "frame_start",
            "frame_end",
            "fps",
            "video_path",
            "usable",
            "reason",
            "url",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

# =========================
# SUMMARY
# =========================
print("\n=== SUMMARY ===")
print(f"Total rows saved: {len(rows)}")

usable_rows = [r for r in rows if r["usable"] == 1]
print(f"Usable rows: {len(usable_rows)}")
print(f"Missing video rows: {len(rows) - len(usable_rows)}")

counts_by_gloss = Counter(r["normalized_gloss"] for r in usable_rows)
print("\n=== USABLE SAMPLES PER GLOSS ===")
for gloss in sorted(counts_by_gloss):
    print(f"{gloss}: {counts_by_gloss[gloss]}")

print(f"\nDone. CSV saved at:\n{output_csv}")