from openai import OpenAI
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
import re
import json

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
CARS_DB_FILE = "cars_database.json"

client = OpenAI(api_key=OPENAI_API_KEY)
model = SentenceTransformer("all-MiniLM-L6-v2")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)


def load_cars_database():
    """Load the cars database to get namespace info"""
    if os.path.exists(CARS_DB_FILE):
        with open(CARS_DB_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"manufacturers": {}}


def get_namespace_for_car(manufacturer: str, model_name: str):
    """Get the namespace for a specific car model"""
    db = load_cars_database()
    manufacturer_lower = manufacturer.lower()
    
    if manufacturer_lower in db["manufacturers"]:
        for car in db["manufacturers"][manufacturer_lower]:
            if car["model"].lower() == model_name.lower():
                return car["namespace"]
    
   
    return f"{manufacturer.lower().replace(' ', '-')}-{model_name.lower().replace(' ', '-')}"


def query_pinecone_for_answer(query: str, manufacturer: str, model_name: str, mode: str = "owner"):
    """Search Pinecone in the specific car's namespace and generate an answer."""
    
    # Get the namespace for this specific car
    namespace = get_namespace_for_car(manufacturer, model_name)
    print(f"🔍 Searching in namespace: {namespace}")
    
    # Generate query embedding
    query_embedding = model.encode(query).tolist()

    # Query Pinecone with namespace filter
    results = index.query(
        vector=query_embedding, 
        top_k=5, 
        namespace=namespace,  
        include_metadata=True
    )
    
    if not results.get("matches"):
        return {
            "answer_text": f"No relevant information found in the {manufacturer} {model_name} manual. Please try rephrasing your question.",
            "images": [],
            "tables": []
        }

    # Gather context from top matches
    context_chunks = []
    all_images = []
    all_tables = []
    
    for match in results["matches"][:3]:  
        text = match["metadata"]["text"]
        
        # Extract image and table markers
        images = re.findall(r'\[IMAGE:\s*(.*?)\]', text)
        tables = re.findall(r'\[TABLE:\s*(.*?)\]', text)
        
        all_images.extend(images)
        all_tables.extend(tables)
        
        # Clean text
        clean_text = re.sub(r'\[IMAGE:\s*.*?\]', '', text)
        clean_text = re.sub(r'\[TABLE:\s*.*?\]', '', clean_text).strip()
        
        context_chunks.append(clean_text)
    
    combined_context = "\n\n".join(context_chunks)

    # Adjust prompt based on mode
    if mode == "mechanic":
        system_prompt = """You are a professional automotive technician assistant. 
        Provide detailed technical information including specifications, diagnostic procedures, 
        and repair instructions. Use technical terminology appropriately."""
    else:  # owner mode
        system_prompt = """You are a helpful car owner's assistant. 
        Explain things in simple, easy-to-understand language. 
        Focus on practical advice and safety. Avoid overly technical jargon."""

    prompt = f"""
Use the following context from the {manufacturer} {model_name} owner's manual to answer the question.

Context:
{combined_context}

Question:
{query}

Provide a clear, helpful answer based on the manual content.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
    )

    return {
        "answer_text": response.choices[0].message.content.strip(),
        "images": list(set(all_images)), 
        "tables": list(set(all_tables))
    }


def upload_pdf_to_pinecone(pdf_bytes, filename: str):
    """Stub for PDF upload — replace with actual logic."""
    return {"status": "success", "filename": filename}
# # basic.py
# from openai import OpenAI
# from pinecone import Pinecone
# from sentence_transformers import SentenceTransformer
# import os
# from dotenv import load_dotenv
# import re

# load_dotenv()

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# client = OpenAI(api_key=OPENAI_API_KEY)
# model = SentenceTransformer("all-MiniLM-L6-v2")
# pc = Pinecone(api_key=PINECONE_API_KEY)
# index = pc.Index(PINECONE_INDEX_NAME)



# def query_pinecone_for_answer(query: str, mode: str = "default"):
#     """Search Pinecone and generate an answer."""
#     query_embedding = model.encode(query).tolist()

#     results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
#     if not results.get("matches"):
#         return "No relevant information found."

#     top_match = results["matches"][0]["metadata"]["text"]

#     # Detect and remove image/table markers
#     image_paths = re.findall(r'\[IMAGE:\s*(.*?)\]', top_match)
#     table_paths = re.findall(r'\[TABLE:\s*(.*?)\]', top_match)
#     clean_text = re.sub(r'\[IMAGE:\s*.*?\]', '', top_match)
#     clean_text = re.sub(r'\[TABLE:\s*.*?\]', '', clean_text).strip()

#     prompt = f"""
# Use the following context to answer the question.

# Context:
# {clean_text}

# Question:
# {query}

# Answer:
# """

#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[
#             {"role": "system", "content": "Answer factually and concisely using the given context."},
#             {"role": "user", "content": prompt}
#         ],
#         temperature=0.3,
#     )

#     return {
#         "answer_text": response.choices[0].message.content.strip(),
#         "images": image_paths,
#         "tables": table_paths
#     }


# def upload_pdf_to_pinecone(pdf_bytes, filename: str):
#     """Stub for PDF upload — replace with actual logic."""
#     # Here you would parse the PDF, extract text, images, tables, etc.
#     # For now, just return a dummy message.
#     return {"status": "success", "filename": filename}