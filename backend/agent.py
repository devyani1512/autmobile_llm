import os
import re
import torch
import streamlit as st
import numpy as np
import pandas as pd
import fitz  # PyMuPDF
import pdfplumber
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import pinecone
from PIL import Image

# ===========================
# CONFIGURATION
# ===========================
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "combined-manuals-index"
EMBED_DIM = 384
PDF_PATH = "CPC_Service_Manual_MEGA_AC_Rev3.pdf"
MANUAL_NAME = "CPC_MANUAL"

ASSET_DIR = "pdf_assets"
TABLE_DIR = "pdf_tables"
os.makedirs(ASSET_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

# ===========================
# DEVICE (CPU/GPU)
# ===========================
device = "cuda" if torch.cuda.is_available() else "cpu"

# ===========================
# EMBEDDING MODEL
# ===========================
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

# ===========================
# INITIALIZE PINECONE
# ===========================
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
if INDEX_NAME not in [i["name"] for i in pc.list_indexes()]:
    st.write(f"Creating index '{INDEX_NAME}'...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBED_DIM,
        metric="cosine",
        spec=pinecone.ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)

# ===========================
# PDF EXTRACTION (TEXT + IMAGES + TABLES)
# ===========================
def extract_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    all_pages_text = []

    for i, page in enumerate(doc):
        blocks = page.get_text("blocks")
        content = []
        image_index = 0

        # Extract images
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            img_ext = ".jpeg" if pix.n < 5 else ".png"
            img_path = os.path.join(ASSET_DIR, f"page_{i+1}_img_{img_index}{img_ext}")
            pix.save(img_path)
            pix = None
            content.append(f"[IMAGE: {img_path}]")

        # Extract text blocks
        for b in blocks:
            if b[6] == 0:
                content.append(b[4].strip())
            elif b[6] == 1:
                content.append(f"[IMAGE: page_{i+1}_img_{image_index}]")
                image_index += 1
            elif b[6] == 2:
                content.append(f"[TABLE: page_{i+1}_table_detected]")

        page_text = "\n".join([c for c in content if c])
        all_pages_text.append(f"[page:{i+1}]\n{page_text}")

    # Extract tables with pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            for t_idx, table in enumerate(tables):
                df = pd.DataFrame(table)
                table_path = os.path.join(TABLE_DIR, f"page_{i+1}_table_{t_idx}.csv")
                df.to_csv(table_path, index=False)
                all_pages_text[i] += f"\n[TABLE: {table_path}]"

    return "\n".join(all_pages_text)

# ===========================
# HYBRID CHUNKING
# ===========================
def split_by_headings(text):
    sections = []
    current_section = []
    current_title = "Untitled"

    for line in text.splitlines():
        line_stripped = line.strip()
        if line_stripped.isupper() or line_stripped[:3].replace('.', '').isdigit():
            if current_section:
                sections.append((current_title, "\n".join(current_section)))
                current_section = []
            current_title = line_stripped
        current_section.append(line)

    if current_section:
        sections.append((current_title, "\n".join(current_section)))

    return sections


def hybrid_chunking(text, max_words=300, overlap=50):
    sections = split_by_headings(text)
    final_chunks = []

    for title, section_text in sections:
        words = section_text.split()
        if len(words) <= max_words:
            final_chunks.append((title, section_text))
        else:
            i = 0
            while i < len(words):
                chunk_words = words[i:i+max_words]
                chunk_text = " ".join(chunk_words)
                final_chunks.append((title, chunk_text))
                i += max_words - overlap

    return final_chunks

# ===========================
# UPLOAD TO PINECONE
# ===========================
def upload_pdf_to_pinecone(pdf_path, manual_name, chunk_words=300):
    st.write(f"Extracting from {pdf_path} ...")
    text = extract_pdf_text(pdf_path)
    chunks = hybrid_chunking(text, max_words=chunk_words, overlap=50)

    st.write(f"Encoding {len(chunks)} chunks ...")
    embeddings = embed_model.encode(
        [c[1] for c in chunks],
        convert_to_numpy=True,
        show_progress_bar=True
    )

    st.write(f"Uploading to Pinecone index '{INDEX_NAME}' ...")
    vectors = []
    for i, (title, chunk_text) in enumerate(chunks):
        vid = f"{manual_name}-chunk-{i}"
        meta = {
            "manual": manual_name,
            "title": title,
            "text": chunk_text
        }
        vectors.append((vid, embeddings[i].tolist(), meta))

    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[i:i+batch_size])

    st.success(f"✅ Uploaded {len(vectors)} chunks from '{manual_name}' to Pinecone.")


# ===========================
# STREAMLIT UI
# ===========================
# ===========================
# IMAGE DISPLAY FUNCTION
# ===========================
def display_answer_with_images(answer_text):
    image_matches = re.findall(r'\[IMAGE:\s*(.*?)\]', answer_text)
    table_matches = re.findall(r'\[TABLE:\s*(.*?)\]', answer_text)

    # Clean up text before showing
    clean_text = re.sub(r'\[IMAGE:\s*.*?\]', '', answer_text)
    clean_text = re.sub(r'\[TABLE:\s*.*?\]', '', clean_text)
    st.markdown(clean_text)

    # Show images
    for img_path in image_matches:
        if os.path.exists(img_path):
            st.image(img_path, caption=os.path.basename(img_path))
        else:
            st.warning(f"Missing image: {img_path}")

    # Show tables
    for table_path in table_matches:
        if os.path.exists(table_path):
            df = pd.read_csv(table_path)
            st.dataframe(df)
        else:
            st.warning(f"Missing table: {table_path}")
st.title("📘 Manual QA with Image Support")

mode = st.radio("Choose Mode", ["Upload PDF to Pinecone", "Ask a Question"])

# --- Upload mode ---
if mode == "Upload PDF to Pinecone":
    pdf_file = st.file_uploader("Upload your PDF manual", type=["pdf"])
    if pdf_file:
        with open(PDF_PATH, "wb") as f:
            f.write(pdf_file.read())
        if st.button("Process and Upload to Pinecone"):
            upload_pdf_to_pinecone(PDF_PATH, MANUAL_NAME)

# --- Question mode ---
else:
    query = st.text_input("Ask a question about the manual:")
    if st.button("Search") and query.strip():
        q_embed = embed_model.encode([query], convert_to_numpy=True)
        results = index.query(
            vector=q_embed[0].tolist(),  # Convert NumPy array to plain list
            top_k=3,
            include_metadata=True
        )


        if results and "matches" in results and len(results["matches"]) > 0:
            best_match = results["matches"][0]["metadata"]["text"]
            st.subheader("Answer:")
            display_answer_with_images(best_match)
        else:
            st.warning("No matching information found.")



