import os
import json
from typing import List, Dict
from pathlib import Path

# === Config ===
INPUT_FOLDER = "data"
OUTPUT_FOLDER = "structured_data"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def extract_topic_and_subtopics(headings: List[Dict]) -> List[Dict]:
    topic = None
    blocks = []
    current_block = None

    for h in headings:
        tag, text = h['tag'], h['text'].strip()

        if tag == 'h1' or tag == 'h2':
            if not topic:
                topic = text
            else:
                if current_block:
                    blocks.append(current_block)
                current_block = {'subtopic': text, 'paragraphs': []}
        elif tag == 'h3' and current_block:
            # Treat h3 as more granular subtopic inside h2
            current_block['paragraphs'].append(f"## {text}")

    if current_block:
        blocks.append(current_block)

    return topic, blocks

def split_paragraphs(paragraphs: List[str], blocks: List[Dict]) -> List[Dict]:
    para_iter = iter(paragraphs)
    for block in blocks:
        while True:
            try:
                para = next(para_iter)
                if "{{" in para:
                    continue  # Skip component placeholders
                if len(block['paragraphs']) < 6:  # heuristic to group 4–6 paragraphs per subtopic
                    block['paragraphs'].append(para.strip())
                else:
                    break
            except StopIteration:
                break
    return blocks

def process_file(file_path: str) -> List[Dict]:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    url = data.get("url", "")
    title = data.get("title", "")
    headings = data.get("headings", [])
    paragraphs = data.get("paragraphs", [])

    topic, blocks = extract_topic_and_subtopics(headings)
    blocks = split_paragraphs(paragraphs, blocks)

    structured = []
    for i, block in enumerate(blocks):
        structured.append({
            "url": url,
            "title": title,
            "topic": topic,
            "subtopic": block['subtopic'],
            "content": "\n".join(block['paragraphs']),
            "metadata": {
                "source": Path(file_path).name,
                "chunk_id": f"{i+1:03d}",
                "topic": topic,
                "subtopic": block['subtopic']
            }
        })
    return structured

def process_all():
    all_structured = []
    for filename in os.listdir(INPUT_FOLDER):
        if filename.endswith(".json"):
            file_path = os.path.join(INPUT_FOLDER, filename)
            structured_blocks = process_file(file_path)
            output_file = os.path.join(OUTPUT_FOLDER, filename)
            with open(output_file, "w", encoding="utf-8") as out:
                json.dump(structured_blocks, out, indent=2, ensure_ascii=False)
            all_structured.extend(structured_blocks)

    # Optionally write all in one file
    with open(os.path.join(OUTPUT_FOLDER, "all_structured.json"), "w", encoding="utf-8") as out:
        json.dump(all_structured, out, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    process_all()
