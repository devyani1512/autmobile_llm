# basic.py
from openai import OpenAI
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
import re

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

client = OpenAI(api_key=OPENAI_API_KEY)
model = SentenceTransformer("all-MiniLM-L6-v2")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)



def query_pinecone_for_answer(query: str, mode: str = "default"):
    """Search Pinecone and generate an answer."""
    query_embedding = model.encode(query).tolist()

    results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
    if not results.get("matches"):
        return "No relevant information found."

    top_match = results["matches"][0]["metadata"]["text"]

    # Detect and remove image/table markers
    image_paths = re.findall(r'\[IMAGE:\s*(.*?)\]', top_match)
    table_paths = re.findall(r'\[TABLE:\s*(.*?)\]', top_match)
    clean_text = re.sub(r'\[IMAGE:\s*.*?\]', '', top_match)
    clean_text = re.sub(r'\[TABLE:\s*.*?\]', '', clean_text).strip()

    prompt = f"""
Use the following context to answer the question.

Context:
{clean_text}

Question:
{query}

Answer:
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer factually and concisely using the given context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
    )

    return {
        "answer_text": response.choices[0].message.content.strip(),
        "images": image_paths,
        "tables": table_paths
    }


def upload_pdf_to_pinecone(pdf_bytes, filename: str):
    """Stub for PDF upload — replace with actual logic."""
    # Here you would parse the PDF, extract text, images, tables, etc.
    # For now, just return a dummy message.
    return {"status": "success", "filename": filename}
