import json
import re

def clean_ground_truth(entry: str) -> dict:
    """Remove <s_custom> tags and convert to JSON dict."""
    cleaned = re.sub(r"</?s_custom>", "", entry).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print("JSON decode error:", e)
        return {}

def convert_jsonl_to_json(input_file: str, output_file: str):
    """
    Read a JSONL file with 'ground_truth' fields wrapped in <s_custom> tags
    and save cleaned JSON objects into a JSON file.
    """
    cleaned_data = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                gt = clean_ground_truth(record["ground_truth"])
                cleaned_data.append({
                    "image": record["image"],
                    "ground_truth": gt
                })

    # Save as JSON array
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Converted {input_file} → {output_file}")


# Run conversion for train.jsonl → train.json
convert_jsonl_to_json("train.jsonl", "train.json")
