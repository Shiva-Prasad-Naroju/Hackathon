import os
import json
import re

def clean_text(text):
    return re.sub(r'[^\x00-\x7F]+', ' ', text).strip()

def structure_data(raw_data):
    result = []
    headings = raw_data['headings']
    paragraphs = raw_data['paragraphs']
    url = raw_data['url']

    current_topic = None
    current_subtopic = None
    paragraph_index = 0
    buffer = []

    for i, heading in enumerate(headings):
        tag = heading["tag"]
        text = clean_text(heading["text"])

        if tag == "h2":
            if buffer:
                result.append({
                    "topic": current_topic,
                    "subtopic": current_subtopic,
                    "content": " ".join(buffer).strip(),
                    "source_url": url
                })
                buffer = []
            current_topic = text
            current_subtopic = None

        elif tag == "h3":
            if buffer:
                result.append({
                    "topic": current_topic,
                    "subtopic": current_subtopic,
                    "content": " ".join(buffer).strip(),
                    "source_url": url
                })
                buffer = []
            current_subtopic = text

        next_heading_texts = [clean_text(h["text"]) for h in headings[i+1:]]
        while paragraph_index < len(paragraphs):
            para = clean_text(paragraphs[paragraph_index])
            paragraph_index += 1
            if any(para.startswith(htext) for htext in next_heading_texts):
                paragraph_index -= 1
                break
            if para:
                buffer.append(para)

    if buffer:
        result.append({
            "topic": current_topic,
            "subtopic": current_subtopic,
            "content": " ".join(buffer).strip(),
            "source_url": url
        })

    return result

# === LOOP THROUGH ALL JSON FILES IN `data/` ===
input_folder = "data"
output_file = "data/structured_data.jsonl"

with open(output_file, "w", encoding="utf-8") as out_f:
    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):
            path = os.path.join(input_folder, filename)
            with open(path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
                structured_entries = structure_data(raw_data)
                for item in structured_entries:
                    out_f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✅ All structured data written to: {output_file}")
