from fastapi import FastAPI
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
import re
import json
import google.generativeai as genai

# --- Configure Gemini ---
genai.configure(api_key="AIzaSyAr8TJOiua07BEox_KrlgQ6QoZXJaKhL2I")

app = FastAPI()


# === Health Check ===
@app.get("/health")
def health_check():
    return {"status": "healthy"}


# === Request Body ===
class QueryInput(BaseModel):
    query: str


# === Scrape SHL Product Catalog ===
def scrape_shl_catalog():
    base_url = "https://www.shl.com"
    catalog_url = f"{base_url}/solutions/products/product-catalog/"
    
    response = requests.get(catalog_url)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.text, "html.parser")
    products = soup.select(".product")

    assessments = []
    for product in products:
        try:
            title_tag = product.select_one(".product-title")
            desc_tag = product.select_one(".product-description")
            duration_tag = product.select_one(".product-info-duration")
            remote = "Yes" if "remote" in product.text.lower() else "No"
            adaptive = "Yes" if "adaptive" in product.text.lower() else "No"
            test_type = re.findall(r"(Knowledge & Skills|Cognitive|Personality & Behaviour|Competencies|Language)", product.text, re.I)

            title = title_tag.get_text(strip=True)
            url = base_url + title_tag.get("href", "#")
            desc = desc_tag.get_text(strip=True) if desc_tag else ""
            duration_match = re.search(r"(\d+)\s*min", duration_tag.get_text(strip=True)) if duration_tag else None
            duration = int(duration_match.group(1)) if duration_match else 0

            assessments.append({
                "url": url,
                "adaptive_support": adaptive,
                "description": desc,
                "duration": duration,
                "remote_support": remote,
                "test_type": list(set(test_type)) or ["Unknown"]
            })
        except Exception:
            continue
    return assessments


# === Use Gemini to extract keywords from query ===
def extract_keywords_with_gemini(query):
    prompt = (
        "You are an intelligent assistant. From the following job description or query, extract key topics or skills to match assessment types.\n"
        "Return a simple Python list of keywords (no explanations).\n\n"
        f"Query: {query}\n\n"
        "Example response: [\"Java\", \"collaboration\", \"cognitive\", \"backend\"]"
    )
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(prompt)
        match = re.search(r'\[.*\]', response.text.strip())
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    return []


# === Match keywords to catalog items ===
def match_keywords_to_catalog(keywords, catalog):
    matched = []
    for item in catalog:
        combined_text = f"{item['description']} {' '.join(item['test_type'])}".lower()
        if any(kw.lower() in combined_text for kw in keywords):
            matched.append(item)
    return matched[:10]


# === Main Recommender Endpoint ===
@app.post("/recommend")
def recommend_assessments(data: QueryInput):
    try:
        print("\n=== Incoming Query ===")
        print(data.query)

        catalog = scrape_shl_catalog()
        print(f"✅ Scraped {len(catalog)} assessments from SHL")

        keywords = extract_keywords_with_gemini(data.query)
        print(f"🧠 Extracted keywords: {keywords}")

        matched = match_keywords_to_catalog(keywords, catalog)
        print(f"🔍 Found {len(matched)} matching assessments from scraped catalog")

        # ✅ If we have valid matches from scraping, use them
        if matched:
            return {"recommended_assessments": matched}

        # ❌ Otherwise fallback to raw Gemini-generated JSON
        print("⚠️ Falling back to Gemini for direct JSON recommendation")

        fallback_prompt = (
            "You are a helpful assistant. Based on the following job description, recommend up to 10 relevant SHL assessments.\n\n"
            f"{data.query.strip()}\n\n"
            "Your response MUST be a valid JSON list. Each object in the list should have these exact keys:\n"
            "- url (must link to SHL’s catalog)\n"
            "- remote_support (Yes/No)\n"
            "- adaptive_support (Yes/No)\n"
            "- description\n"
            "- duration (in minutes, as integer)\n"
            "- test_type (array of strings)\n\n"
            "Respond ONLY in valid JSON format like this:\n"
            '[{"url": "...", "remote_support": "Yes", "adaptive_support": "No", "description": "...", "duration": 30, "test_type": ["Cognitive"]}]'
        )

        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(fallback_prompt)

        # Extract JSON list from Gemini's output
        match = re.search(r'\[\s*{.*?}\s*]', response.text.strip(), re.DOTALL)
        fallback_data = json.loads(match.group()) if match else []

        print(f"✅ Fallback Gemini returned {len(fallback_data)} assessments")
        return {"recommended_assessments": fallback_data}

    except Exception as e:
        print("❌ ERROR:", str(e))
        return {"error": str(e)}
