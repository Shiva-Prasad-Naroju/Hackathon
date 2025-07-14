import requests
from bs4 import BeautifulSoup
import json
import os

# URL = "https://www.plumhq.com/starter-guides/starter-guide-to-hyderabad"

# URL = "https://www.plumhq.com/starter-guides/starter-guide-to-bangalore"

# URL = "https://www.plumhq.com/starter-guides/starter-guide-to-delhi"

URL = "https://www.plumhq.com/starter-guides/starter-guide-to-chennai"

def scrape_plum_guide(url=URL):
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    result = {
    "url": url,
    "title": soup.title.string.strip() if soup.title and soup.title.string else "",
    "headings": [],
    "paragraphs": [],
    "lists": [],
    "images": [],
    "meta_description": "",
}


    # Meta Description
    meta = soup.find("meta", attrs={"name": "description"})
    if meta:
        result["meta_description"] = meta.get("content", "").strip()

    # Headings (h1 to h4)
    for tag in soup.find_all(["h1", "h2", "h3", "h4"]):
        result["headings"].append({
            "tag": tag.name,
            "text": tag.get_text(strip=True)
        })

    # Paragraphs
    for p in soup.find_all("p"):
        text = p.get_text(strip=True)
        if text:
            result["paragraphs"].append(text)

    # Lists (ul and ol)
    for lst in soup.find_all(["ul", "ol"]):
        items = [li.get_text(strip=True) for li in lst.find_all("li")]
        if items:
            result["lists"].append({
                "type": lst.name,
                "items": items
            })

    # Images
    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src")
        if src:
            if src.startswith("//"):
                src = "https:" + src
            result["images"].append({
                "src": src,
                "alt": img.get("alt", "").strip()
            })

    return result


if __name__ == "__main__":
    data = scrape_plum_guide()
    os.makedirs("data", exist_ok=True)
    with open("data/chennai.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print("✅ Scraped data saved to data/chennai.json")
