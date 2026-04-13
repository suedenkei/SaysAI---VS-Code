import json
from pathlib import Path

source_json = Path(r"C:\SAYSAI\01_raw_data\wlasl\WLASL-master\start_kit\WLASL_v0.3.json")
output_json = Path(r"C:\SAYSAI\01_raw_data\wlasl_10\start_kit\WLASL_v0.3.json")

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

def normalize(text: str) -> str:
    return text.strip().upper().replace("-", " ").replace("_", " ")

if not source_json.exists():
    raise FileNotFoundError(f"Source JSON not found: {source_json}")

with open(source_json, "r", encoding="utf-8") as f:
    data = json.load(f)

normalized_targets = {normalize(g) for g in target_glosses}

matched_entries = []
matched_names = []

for entry in data:
    gloss = entry.get("gloss", "")
    if normalize(gloss) in normalized_targets:
        matched_entries.append(entry)
        matched_names.append(gloss)

with open(output_json, "w", encoding="utf-8") as f:
    json.dump(matched_entries, f, ensure_ascii=False, indent=2)

print("Saved filtered JSON to:", output_json)
print("Matched dataset gloss names:")
for name in sorted(set(matched_names)):
    print(" -", name)

print("\nTotal matched gloss classes:", len(set(matched_names)))
print("Total matched top-level entries:", len(matched_entries))

print("\nInstances per gloss:")
for entry in matched_entries:
    print(f"{entry['gloss']}: {len(entry.get('instances', []))}")