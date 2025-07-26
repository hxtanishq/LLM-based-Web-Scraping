import requests
from bs4 import BeautifulSoup
import json
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

load_dotenv()

def scraper():
    
    session = requests.Session()
    session.headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    url = "https://search.earth911.com"
    params = {'what': 'Electronics', 'where': '10001', 'max_distance': '100'}
    try:
        response = session.get(url, params=params)
        response.encoding = 'utf-8-sig'
        response.raise_for_status()
        html = response.text
        print("HTML fetched successfully.")
    except requests.RequestException as e:
        print(f"Error fetching URL: {e}")
        return []

    soup = BeautifulSoup(html, 'html.parser')
    for tag in soup(['script', 'style']):
        tag.decompose()
    text = soup.get_text(separator='\n', strip=True)
    text = text[:8000]  

    materials = {
        "Electronics": [
            "Computers, Laptops, Tablets",
            "Monitors, TVs (CRT & Flat Screen)",
            "Cell Phones, Smartphones",
            "Printers, Copiers, Fax Machines",
            "Audio/Video Equipment",
            "Gaming Consoles",
            "Small Appliances (Microwaves, Toasters, etc.)",
            "Computer Peripherals (Keyboards, Mice, Cables, etc.)"
        ],
        "Batteries": [
            "Alkaline Batteries",
            "Lithium-ion Batteries",
            "Lead-Acid Batteries",
            "Rechargeable Batteries",
            "Button Cell Batteries",
            "Car Batteries"
        ],
    }
    prompt = f"""Extract recycling facility info from the following text. 
    Return only a valid JSON array of facilities in this exact format:
    [
        {{
            "business_name": "name",
            "last_update_date": "dd-mm-yyyy",
            "street_address": "full address",
            "materials_category": ["Electronics"],
            "materials_accepted": ["Computers, Laptops, Tablets"]
        }}
    ]
    Rules:
    - Return only a valid JSON array, no additional text, comments, or explanations.
    - Ensure all property names and string values are enclosed in double quotes.
    - Escape special characters (e.g., quotes, backslashes) correctly.
    - If no date is available, use an empty string for "last_update_date".
    - Do not include trailing commas or invalid JSON syntax.
    - If no facilities are found, return an empty array [].
    Electronics categories: {materials['Electronics']}
    Batteries categories: {materials['Batteries']}
    Text: {text}
    """
    print("Prompt created successfully.")

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in the .env file.")
    try:
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")
        reply = llm.invoke([HumanMessage(content=prompt)]).content.strip()
        print("LLM response:", reply)
    except Exception as e:
        print(f"Error invoking LLM: {e}")
        return []

    try:
        
        results = json.loads(reply)
        if not isinstance(results, list):
            print("Error: LLM response is not a JSON array")
            return []
        print("JSON parsed successfully.")
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        print(f"Raw LLM response: {reply}")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        print(f"Raw LLM response: {reply}")
        return []

    return results[:3]

if __name__ == "__main__":
    facilities = scraper()
    print(json.dumps(facilities, indent=2))
    with open("data.json", "w", encoding='utf-8') as f:
        json.dump(facilities, f, indent=2, ensure_ascii=False)