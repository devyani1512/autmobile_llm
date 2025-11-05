# embed2.py - Upload car manuals with manufacturer + model
from dotenv import load_dotenv
import os
import torch
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
import pinecone
import json

load_dotenv()

# ===========================
# CONFIG
# ===========================
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")  # use env variable name
INDEX_NAME = "automobile-manuals-multicar"
EMBED_DIM = 384
OUTPUT_DIR = "pdf_assets"
CARS_DB_FILE = "cars_database.json"

# ===========================
# SETUP
# ===========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

# Create index if not exists
if INDEX_NAME not in [i["name"] for i in pc.list_indexes()]:
    print(f"Creating NEW index '{INDEX_NAME}'...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBED_DIM,
        metric="cosine",
        spec=pinecone.ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print("✅ Index created!")

index = pc.Index(INDEX_NAME)

# ===========================
# CARS DATABASE
# ===========================
def load_cars_database():
    if os.path.exists(CARS_DB_FILE):
        with open(CARS_DB_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"manufacturers": {}}

def save_cars_database(db):
    with open(CARS_DB_FILE, 'w', encoding='utf-8') as f:
        json.dump(db, f, indent=2, ensure_ascii=False)

def add_car_to_database(manufacturer, model, year, namespace):
    db = load_cars_database()
    if manufacturer not in db["manufacturers"]:
        db["manufacturers"][manufacturer] = []
    existing = [m for m in db["manufacturers"][manufacturer] if m["model"] == model]
    if not existing:
        db["manufacturers"][manufacturer].append({
            "model": model,
            "year": year,
            "namespace": namespace
        })
        print(f"✅ Added {manufacturer} {model} to database")
    else:
        print(f"⚠️  {manufacturer} {model} already exists in database")
    save_cars_database(db)

# ===========================
# PDF EXTRACTION
# ===========================
def extract_pdf_text(pdf_path, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    pages_text = []

    for i, page in enumerate(doc):
        content = []
        page_num = i + 1
        page_prefix = f"[page:{page_num}]"

        blocks = page.get_text("blocks")
        for b in blocks:
            block_type = b[6]
            if block_type == 0:
                text = b[4].strip()
                if text:
                    content.append(text)

        page_content = "\n".join(content)
        pages_text.append(f"{page_prefix}\n{page_content}")

    doc.close()
    return "\n\n".join(pages_text)

# ===========================
# CHUNKING
# ===========================
def chunk_text(text, max_words=300, overlap=50):
    """Split text into overlapping chunks for embeddings."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + max_words
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += max_words - overlap
    return chunks

# ===========================
# UPLOAD TO PINECONE
# ===========================
def upload_manual(pdf_path, manufacturer, model, year):
    """Upload a car manual to Pinecone."""
    namespace = f"{manufacturer.lower().replace(' ', '-')}-{model.lower().replace(' ', '-')}"
    
    print(f"\n{'='*60}")
    print(f"🚗 Uploading: {manufacturer} {model} ({year})")
    print(f"📦 Namespace: {namespace}")
    print(f"{'='*60}")
    
    # Extract text
    print("📄 Extracting text from PDF...")
    text = extract_pdf_text(pdf_path)
    
    # Chunk text
    print("✂️  Chunking text...")
    chunks = chunk_text(text, max_words=300, overlap=50)
    print(f"   Created {len(chunks)} chunks")
    
    # Generate embeddings
    print("🧠 Generating embeddings...")
    embeddings = embed_model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)

    # Prepare vectors
    print("📦 Preparing vectors...")
    vectors = []
    for i, chunk in enumerate(chunks):
        vid = f"{namespace}-chunk-{i}"
        meta = {
            "manufacturer": manufacturer,
            "model": model,
            "year": year,
            "namespace": namespace,
            "text": chunk
        }
        vectors.append((vid, embeddings[i].tolist(), meta))

    # Upload to Pinecone
    print("☁️  Uploading to Pinecone...")
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch, namespace=namespace)
        print(f"   Uploaded batch {i//batch_size + 1}/{(len(vectors)-1)//batch_size + 1}")

    print(f"✅ SUCCESS! Uploaded {len(vectors)} chunks to namespace '{namespace}'")
    
    # Add to database
    add_car_to_database(manufacturer, model, year, namespace)
    return namespace

# ===========================
# MAIN
# ===========================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚗 AUTOMOBILE MANUAL UPLOADER")
    print("="*60)
    
    pdf_path = input("\n📄 Enter PDF file path: ").strip().strip('"')
    manufacturer = input("🏭 Enter manufacturer (e.g., Toyota, Maruti Suzuki): ").strip()
    model = input("🚙 Enter model (e.g., Corolla Hybrid, Swift): ").strip()
    year = input("📅 Enter year (e.g., 2022): ").strip()
    
    if os.path.exists(pdf_path):
        namespace = upload_manual(pdf_path, manufacturer, model, year)
        print(f"\n{'='*60}")
        print("✅ UPLOAD COMPLETE!")
        print(f"{'='*60}")
        print(f"Manufacturer: {manufacturer}")
        print(f"Model: {model}")
        print(f"Year: {year}")
        print(f"Namespace: {namespace}")
        print(f"\nDatabase saved to: {CARS_DB_FILE}")
    else:
        print(f"❌ ERROR: File not found: {pdf_path}")
