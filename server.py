import streamlit as st
import os
import zipfile
import json
import sqlite3
import pinecone
import openai
from tqdm import tqdm
from langchain.agents import initialize_agent
from langchain.tools import Tool
import torchvision
import transformers
from sentence_transformers import SentenceTransformer
from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import requests
from geopy.distance import geodesic

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")


server = Flask(__name__)



# === CONFIGURATION ===
DATA_PATH = "C:/Users/joshu/project-aieng-interactive-travel-planner/data/"  # Path where files are stored
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")  
DB_FILE = "puerto_rico_data.db"  # SQLite database file
openai.api_key = OPENAI_API_KEY 


# === Function: Fetch Wikipedia Image ===
def get_wikipedia_image(place_name):
    """
    Fetch the main image of a place from Wikipedia.
    """
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{place_name.replace(' ', '_')}"
    response = requests.get(url).json()
    return response.get("thumbnail", {}).get("source", "No image available")

# === Function: Generate Chat Response ===

def get_chat_response(text):
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)  # Use the new client structure
        
        response = client.chat.completions.create(  # ✅ Updated API
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a travel assistant for Puerto Rico. Provide travel recommendations."},
                {"role": "user", "content": text}
            ]
        )
        
        return response.choices[0].message.content  # ✅ Extract correct response format
    
    except Exception as e:
        print(f"❌ OpenAI API Error: {e}")
        return "Sorry, I couldn't process that request."


# === FUNCTION: Extract ZIP Files ===
def extract_zip(zip_path, extract_to):
    """Extracts a zip file into a directory."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

# === FUNCTION: Process Text Files ===
def process_text_files(directory, user_language="en"):
    """Reads and processes all .txt files in a directory, with multilingual support and intelligent agents."""
    structured_data = []
    
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()

                # === Create an agent with tools ===
                tools = [
                    Tool(name="Location Finder", func=rank_appropriate_locations, description="Finds the best travel spots."),
                    Tool(name="Weather Finder", func=find_weather_forecast, description="Checks weather for a given location."),
                    Tool(name="Distance Calculator", func=compute_distance_to_list, description="Calculates distances between locations."),
                    Tool(name="Multilingual Translator", func=translate_summary, description="Translates text into different languages.")
                ]
                
                agent = initialize_agent(tools, llm=openai.ChatCompletion, agent_type="zero-shot-react-description")
                
                # === Use Agent to Summarize ===
                response = agent.run(f"Summarize this location as a tourist guide in {user_language}: {text}")
                summary = response if isinstance(response, str) else response["choices"][0]["message"]["content"]

                # === Extract possible coordinates ===
                coordinates = extract_coordinates(text)

                # === Store structured data ===
                structured_data.append({
                    "name": filename.replace(".txt", "").replace("_", " ").title(),
                    "summary": summary,
                    "coordinates": coordinates,
                    "raw_text": text
                })
    
    return structured_data

# === FUNCTION: Extract Coordinates from Text ===
def extract_coordinates(text):
    """Extracts latitude and longitude if present in text (rudimentary regex approach)."""
    import re
    match = re.search(r"(-?\d{1,2}\.\d+),\s*(-?\d{1,3}\.\d+)", text)
    if match:
        return {"latitude": float(match.group(1)), "longitude": float(match.group(2))}
    return None


def translate_summary(summary, target_lang="es"):
    """Translates text into the target language."""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": f"Translate this into {target_lang}:"},
                  {"role": "user", "content": summary}]
    )
    return response["choices"][0]["message"]["content"]



# === FUNCTION: Store in JSON ===
def save_to_json(data, filename):
    """Saves structured data to a JSON file."""
    with open(filename, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)

# === FUNCTION: Store in SQLite ===
def save_to_sqlite(data, db_file, target_lang="es"):
    """Stores structured data in an SQLite database with multilingual support."""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS locations (
                      id INTEGER PRIMARY KEY AUTOINCREMENT,
                      name TEXT,
                      summary TEXT,
                      summary_translated TEXT,
                      latitude REAL,
                      longitude REAL,
                      raw_text TEXT)''')

    for entry in data:
        # Translate the summary before saving
        translated_summary = translate_summary(entry["summary"], target_lang)

        cursor.execute('''INSERT INTO locations (name, summary, summary_translated, latitude, longitude, raw_text)
                          VALUES (?, ?, ?, ?, ?, ?)''',
                       (entry["name"], entry["summary"], translated_summary,
                        entry["coordinates"]["latitude"] if entry["coordinates"] else None,
                        entry["coordinates"]["longitude"] if entry["coordinates"] else None,
                        entry["raw_text"]))
    
    conn.commit()
    conn.close()



# === FUNCTION: Store in Pinecone ===
from pinecone import Pinecone
from pinecone import Pinecone, ServerlessSpec

def save_to_pinecone(data, index_name="puerto-rico-locations", target_lang="es"):
    """Stores summarized data into Pinecone for vector-based retrieval, including translations."""
    
    # Initialize Pinecone instance
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Check if the index exists, create it if not
    existing_indexes = pc.list_indexes().names()
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name, 
            dimension=1536, 
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    # Get index instance
    index = pc.Index(index_name)

    # Upload data to Pinecone
    for entry in tqdm(data, desc="Uploading to Pinecone"):
        # Generate embeddings for both original and translated summaries
        embedding_response = openai.Embedding.create(model="text-embedding-ada-002", input=entry["summary"])
        embedding_translated_response = openai.Embedding.create(model="text-embedding-ada-002", input=entry["summary_translated"])

        embedding = embedding_response["data"][0]["embedding"]
        embedding_translated = embedding_translated_response["data"][0]["embedding"]

        # Upsert into Pinecone
        index.upsert([
            (entry["name"], embedding, {"summary": entry["summary"], "language": "en"}),
            (entry["name"], embedding_translated, {"summary": entry["summary_translated"], "language": target_lang})
        ])

    print("✅ Data successfully uploaded to Pinecone!")



# === MAIN EXECUTION ===
if __name__ == "__main__":
    # Define extraction directories
    landmarks_dir = os.path.join(DATA_PATH, "landmarks")
    municipalities_dir = os.path.join(DATA_PATH, "municipalities")
    news_dir = os.path.join(DATA_PATH, "news")

    # Extract all zip files
    extract_zip(os.path.join(DATA_PATH, "landmarks.zip"), landmarks_dir)
    extract_zip(os.path.join(DATA_PATH, "municipalities.zip"), municipalities_dir)
    extract_zip(os.path.join(DATA_PATH, "elmundo_chunked_es_page1_15years.zip"), news_dir)

    # Process the text files
    landmarks_data = process_text_files(landmarks_dir)
    municipalities_data = process_text_files(municipalities_dir)
    news_data = process_text_files(news_dir)


# Merge all data
all_data = landmarks_data + municipalities_data + news_data

    # Save to JSON
save_to_json(all_data, "puerto-rico-data.json")

    # Save to SQLite
save_to_sqlite(all_data, DB_FILE)

    # Save to Pinecone
save_to_pinecone(all_data)

print("✅ Data Extraction, Processing, and Storage Completed!")



# Initialize multilingual embedding model
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


from typing import List, Dict


# === FUNCTION: Chunk Text ===
def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """Splits text into overlapping chunks."""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    
    return chunks

# === FUNCTION: Process News Files ===
def process_news_files(directory: str) -> List[Dict]:
    """Reads and processes all .txt files in the directory."""
    structured_data = []
    
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
                chunks = chunk_text(text)
                
                for chunk in chunks:
                    embedding = embedding_model.encode(chunk).tolist()
                    
                    structured_data.append({
                        "news_id": filename.replace(".txt", ""),
                        "text": chunk,
                        "embedding": embedding
                    })
    return structured_data



# Initialize Pinecone Client
from pinecone import Pinecone

INDEX_NAME = "puerto-rico-news" 


def save_to_pinecone(data: List[Dict]):
    """Stores news embeddings into Pinecone."""
    
    # Initialize Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Check if index exists
    existing_indexes = pc.list_indexes().names()
    if INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,  # 384 for MiniLM embeddings
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")  # ✅ FIXED: Added spec
        )

    # Get index instance
    index = pc.Index(INDEX_NAME)

    # Upload data in batches to avoid API rate limits
    batch_size = 100
    for i in tqdm(range(0, len(data), batch_size), desc="Uploading to Pinecone"):
        batch = data[i:i + batch_size]
        index.upsert([(entry["news_id"], entry["embedding"], {"text": entry["text"]}) for entry in batch])
    
    print("✅ Data successfully uploaded to Pinecone!")



pc = Pinecone(api_key=PINECONE_API_KEY)

# === FUNCTION: Query Pinecone for Relevant News ===
def retrieve_relevant_news(query: str, top_k: int = 5):
    """Retrieves the most relevant news chunks based on a user query."""
    
    index = pc.Index(INDEX_NAME)
    query_embedding = embedding_model.encode(query).tolist()
    
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    
    return [(match["id"], match["metadata"]["text"]) for match in results["matches"]]

# === FUNCTION: Retrieve News for Landmarks ===
def retrieve_news_for_landmark(landmark_name: str):
    """Finds news articles related to a given landmark."""
    return retrieve_relevant_news(f"Historical news about {landmark_name}")

# === MAIN EXECUTION ===
if __name__ == "__main__":
    news_data = process_news_files(os.path.join(DATA_PATH, "news"))
    save_to_pinecone(news_data)
    
    print("✅ News processing and indexing complete!")
    
    # Example query
    user_query = "History of San Juan Cathedral"
    results = retrieve_relevant_news(user_query)
    
    print("🔍 Relevant News Snippets:")
    for news_id, text in results:
        print(f"- {text[:200]}...")



# Store user preferences
user_preferences = {
    "dates": None,
    "interests": [],
    "locked_locations": [],
}

# === FUNCTION: Ask for Travel Dates ===
def ask_for_travel_dates():
    """Prompts the user for travel dates."""
    user_preferences["dates"] = input("Enter your travel dates (YYYY-MM-DD to YYYY-MM-DD): ")
    print(f"📅 Travel Dates Set: {user_preferences['dates']}")

# === FUNCTION: Ask for Interests ===
def ask_for_interests():
    """Prompts the user for their interests."""
    interests = input("What are your interests? (e.g., history, nature, food): ")
    user_preferences["interests"] = interests.lower().split(", ")
    print(f"✅ Interests Recorded: {user_preferences['interests']}")

# === FUNCTION: Retrieve and Rank Locations ===
def rank_appropriate_locations():
    """Retrieves and ranks locations based on user interests."""
    print("🔍 Retrieving top locations for your interests...")
    ranked_locations = [
        {"name": "El Yunque National Forest", "category": "nature", "score": 0.9},
        {"name": "San Juan Old Town", "category": "history", "score": 0.85},
        {"name": "Bioluminescent Bay", "category": "nature", "score": 0.8},
    ]
    ranked_locations.sort(key=lambda x: x["score"], reverse=True)
    return ranked_locations

openai.api_key = OPENAI_API_KEY

# === FUNCTION: Answer Questions about Locations ===
def find_info_on_location(location_name: str):
    """Uses OpenAI's new API to retrieve information about a location."""
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Provide concise travel information about locations in Puerto Rico."},
            {"role": "user", "content": f"Tell me about {location_name}"},
        ]
    )
    return response.choices[0].message.content

# === FUNCTION: Check Weather Conditions ===
def find_weather_forecast(location: str):
    """Stub function to check weather forecast (to be implemented with real API)."""
    return f"🌦️ The weather in {location} is expected to be sunny with occasional showers."

# === FUNCTION: Add Location to Visit List ===
def add_location_to_visit_list(location: str):
    """Adds a selected location to the locked itinerary."""
    if location not in user_preferences["locked_locations"]:
        user_preferences["locked_locations"].append(location)
        print(f"✅ {location} added to your visit list.")
    else:
        print(f"⚠️ {location} is already in your itinerary.")

# === FUNCTION: Compute Travel Distances ===
def compute_distance_to_list(new_location: str):
    """Computes distance between new location and all locked locations."""
    locations = {
        "San Juan": (18.4663, -66.1057),
        "El Yunque National Forest": (18.2957, -65.8006),
        "Bioluminescent Bay": (18.309, -65.3031),
    }
    if new_location not in locations:
        print("❌ Location not found in database.")
        return
    
    new_coords = locations[new_location]
    for loc in user_preferences["locked_locations"]:
        if loc in locations:
            dist = geodesic(new_coords, locations[loc]).km  # ✅ Corrected
            print(f"🚗 Distance from {new_location} to {loc}: {dist:.2f} km")


# === FUNCTION: Finalize Travel List ===
def finalize_travel_list():
    """Displays the finalized travel itinerary."""
    print("\n📌 Final Travel Itinerary:")
    for loc in user_preferences["locked_locations"]:
        print(f"- {loc}")
    print("🎒 Have a great trip!")



# === MAIN CHATBOT LOOP ===
def chatbot_loop():
    print("👋 Welcome to the Puerto Rico Travel Planner!")
    ask_for_travel_dates()
    ask_for_interests()
    
    ranked_locations = rank_appropriate_locations()
    print("\n🏝️ Recommended Locations:")
    for loc in ranked_locations:
        print(f"- {loc['name']} ({loc['category']})")
    
    while True:
        action = input("\nWhat would you like to do? (ask, add, weather, distance, finalize, exit): ").lower()
        
        if action == "ask":
            location = input("Enter a location name: ")
            print(find_info_on_location(location))
        
        elif action == "add":
            location = input("Enter a location to add: ")
            add_location_to_visit_list(location)
        
        elif action == "weather":
            location = input("Enter a location: ")
            print(find_weather_forecast(location))
        
        elif action == "distance":
            location = input("Enter a location: ")
            compute_distance_to_list(location)
        
        elif action == "finalize":
            finalize_travel_list()
            break
        
        elif action == "exit":
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Try again.")



# === Route: Serve Chat UI ===
@server.route("/")
def index():
    return render_template('chat.html')

# === Route: Handle Chat Requests ===
@server.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg", "")

    if not msg:
        return jsonify({"response": "Error: No message received", "image_url": None}), 400

    response_text = get_chat_response(msg)  

    # Check if user asks about El Yunque and return custom image
    if "el yunque" in msg:
        image_url = r"C:\Users\joshu\project-aieng-interactive-travel-planner\static\el_yunque.jpeg"  

    else:
        image_url = None  # No image for other queries

    image_url = get_wikipedia_image(msg)  

    print(f"✅ Chatbot Response: {response_text}, Image: {image_url}")  # Debugging

    return jsonify({"response": response_text, "image_url": image_url})


if __name__ == '__main__':
    server.run()
