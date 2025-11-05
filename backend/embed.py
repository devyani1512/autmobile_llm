# # embed_pdf.py
# # Install dependencies if not already done:
# # pip install sentence-transformers pinecone-client pypdf numpy tqdm

# from dotenv import load_dotenv
# import os
# from sentence_transformers import SentenceTransformer
# from PyPDF2 import PdfReader
# import pinecone
# import numpy as np

# load_dotenv()

# # -------------------------
# # CONFIG
# # -------------------------
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# INDEX_NAME = "cpc-manual-index"
# EMBED_DIM = 384
# PDF_PATH = "CPC_Service_Manual_MEGA_AC_Rev3.pdf"

# # Embedding model
# embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# # Initialize Pinecone
# pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

# # Auto-reset index
# if INDEX_NAME in [i["name"] for i in pc.list_indexes()]:
#     print(f"Deleting existing index '{INDEX_NAME}'...")
#     pc.delete_index(INDEX_NAME)

# # Recreate index
# pc.create_index(
#     name=INDEX_NAME,
#     dimension=EMBED_DIM,
#     metric="cosine",
#     spec=pinecone.ServerlessSpec(cloud="aws", region="us-east-1")
# )
# index = pc.Index(INDEX_NAME)

# # -------------------------
# # PDF extraction + chunking
# # -------------------------
# def extract_pdf_text(pdf_path):
#     reader = PdfReader(pdf_path)
#     text = []
#     for i, page in enumerate(reader.pages):
#         page_text = page.extract_text()
#         if page_text:
#             text.append(f"[page:{i+1}]\n{page_text.strip()}")
#     return "\n\n".join(text)

# def chunk_text(text, max_words=300, overlap=50):
#     words = text.split()
#     chunks = []
#     i = 0
#     while i < len(words):
#         chunk_words = words[i:i+max_words]
#         chunks.append(" ".join(chunk_words))
#         i += max_words - overlap
#     return chunks

# # -------------------------
# # Upload chunks to Pinecone
# # -------------------------
# def upload_pdf_to_pinecone(pdf_path, chunk_words=300):
#     text = extract_pdf_text(pdf_path)
#     chunks = chunk_text(text, max_words=chunk_words, overlap=50)
#     embeddings = embed_model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)

#     vectors = []
#     for i, emb in enumerate(embeddings):
#         vid = f"chunk-{i}"
#         meta = {"text": chunks[i]}
#         vectors.append((vid, emb.tolist(), meta))

#     # Upsert in batches
#     batch_size = 100
#     for i in range(0, len(vectors), batch_size):
#         index.upsert(vectors=vectors[i:i+batch_size])

#     print(f" Uploaded {len(vectors)} chunks to Pinecone.")

# # -------------------------
# # Run embedding
# # -------------------------
# if __name__ == "__main__":
#     upload_pdf_to_pinecone(PDF_PATH)

# ===========================
# embed_pdf.py (Enhanced)
# ===========================

from dotenv import load_dotenv
import os
import torch
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
import pinecone
import numpy as np
import pdfplumber
from PIL import Image
import io

# ===========================
# CONFIG
# ===========================
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "combined-manuals-index"
EMBED_DIM = 384
PDF_PATH = "CPC_Service_Manual_MEGA_AC_Rev3.pdf"
MANUAL_NAME = "CPC_MANUAL"  # <-- set per PDF
OUTPUT_DIR = "pdf_assets"   # Folder for extracted images/tables

# ===========================
# DEVICE (CPU/GPU)
# ===========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Embedding model
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

# ===========================
# PINECONE SETUP
# ===========================
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

# Create index if not exists
if INDEX_NAME not in [i["name"] for i in pc.list_indexes()]:
    print(f"Creating index '{INDEX_NAME}'...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBED_DIM,
        metric="cosine",
        spec=pinecone.ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)

# ===========================
# PDF EXTRACTION (with tables & images)
# ===========================
def extract_pdf_text(pdf_path, output_dir=OUTPUT_DIR):
    """
    Extract text, images, and tables from the PDF.
    Saves extracted images and tables into 'output_dir',
    and inserts placeholders in the text like:
        [IMAGE: page3_img1.png]
        [TABLE: page3_table1.txt]
    """
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    pages_text = []

    for i, page in enumerate(doc):
        content = []
        page_num = i + 1
        page_prefix = f"[page:{page_num}]"

        # --- Extract text and detect block types ---
        blocks = page.get_text("blocks")
        for b in blocks:
            block_type = b[6]

            # Text blocks
            if block_type == 0:
                text = b[4].strip()
                if text:
                    content.append(text)

            # Image blocks
            elif block_type == 1:
                for img_index, img in enumerate(page.get_images(full=True)):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    image_filename = f"page{page_num}_img{img_index + 1}.{image_ext}"
                    image_path = os.path.join(output_dir, image_filename)

                    # Save the image
                    with open(image_path, "wb") as f:
                        f.write(image_bytes)

                    content.append(f"[IMAGE: {image_filename}]")

            # Vector drawings (lines/tables)
            elif block_type == 2:
                content.append("[VECTOR GRAPHICS FOUND]")

        # --- Extract tables using pdfplumber ---
        with pdfplumber.open(pdf_path) as pdf:
            if page_num - 1 < len(pdf.pages):
                plumber_page = pdf.pages[page_num - 1]
                try:
                    tables = plumber_page.extract_tables()
                    if tables:
                        for t_index, table in enumerate(tables):
                            table_filename = f"page{page_num}_table{t_index + 1}.txt"
                            table_path = os.path.join(output_dir, table_filename)

                            # Save table as text file
                            with open(table_path, "w", encoding="utf-8") as f:
                                for row in table:
                                    f.write("\t".join([cell if cell else "" for cell in row]) + "\n")

                            content.append(f"[TABLE: {table_filename}]")
                except Exception as e:
                    print(f"⚠️ Error extracting table on page {page_num}: {e}")

        # Combine page text
        page_content = "\n".join([c for c in content if c])
        pages_text.append(f"{page_prefix}\n{page_content}")

    doc.close()
    return "\n\n".join(pages_text)


# ===========================
# CHUNKING LOGIC
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
                chunk_words = words[i:i + max_words]
                chunk_text = " ".join(chunk_words)
                final_chunks.append((title, chunk_text))
                i += max_words - overlap

    return final_chunks


# ===========================
# UPLOAD TO PINECONE
# ===========================
def upload_pdf_to_pinecone(pdf_path, manual_name, chunk_words=300):
    text = extract_pdf_text(pdf_path)
    chunks = hybrid_chunking(text, max_words=chunk_words, overlap=50)

    embeddings = embed_model.encode(
        [c[1] for c in chunks],
        convert_to_numpy=True,
        show_progress_bar=True
    )

    vectors = []
    for i, (title, chunk_text) in enumerate(chunks):
        vid = f"{manual_name}-chunk-{i}"

        # Extract image/table references for metadata
        images = [part.split(": ")[1] for part in chunk_text.split() if part.startswith("[IMAGE:")]
        print("Problematic chunk:", chunk_text)
    
    
        #tables = [part.split(": ")[1] for part in chunk_text.split() if part.startswith("[TABLE:")]
    tables = []
    for part in chunk_text.split():
        if part.startswith("[TABLE:") and part.endswith("]"):
            table_ref = part[len("[TABLE:"): -1].strip()
            tables.append(table_ref)
   
        meta = {
            "manual": manual_name,
            "title": title,
            "text": chunk_text,
            "images": images,
            "tables": tables
        }

        vectors.append((vid, embeddings[i].tolist(), meta))

    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[i:i + batch_size])

    print(f"✅ Uploaded {len(vectors)} chunks from '{manual_name}' to Pinecone.")


# ===========================
# RUN
# ===========================
if __name__ == "__main__":
    upload_pdf_to_pinecone(PDF_PATH, MANUAL_NAME)


# ===========================
# TO DELETE INDEX (optional)
# ===========================
# import pinecone
# pc = pinecone.Pinecone(api_key="YOUR_PINECONE_KEY")
# pc.delete_index("combined-manuals-index")
