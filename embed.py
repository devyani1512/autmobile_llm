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


from dotenv import load_dotenv
import os
import torch
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
import pinecone
import numpy as np

load_dotenv()

# CONFIG
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") 
INDEX_NAME = "combined-manuals-index"
EMBED_DIM = 384
PDF_PATH = "CPC_Service_Manual_MEGA_AC_Rev3.pdf"
MANUAL_NAME = "CPC_MANUAL"   # <-- set per PDF


# DEVICE (CPU/GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Embedding model
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

# Initialize Pinecone
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

# PDF extraction (with table/image markers)
def extract_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    pages_text = []

    for i, page in enumerate(doc):
        blocks = page.get_text("blocks")  # block-level content
        content = []

        for b in blocks:
            if b[6] == 0:  # text block
                content.append(b[4].strip())
            elif b[6] == 1:  # image block
                content.append("[IMAGE FOUND]")
            elif b[6] == 2:  # drawing/vector like tables,images
                content.append("[TABLE FOUND]")

        # Merge blocks in reading order
        page_content = "\n".join([c for c in content if c])
        pages_text.append(f"[page:{i+1}]\n{page_content}")

    return "\n".join(pages_text)


# Hybrid chunking

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


# Upload to Pinecone

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
        meta = {
            "manual": manual_name,
            "title": title,
            "text": chunk_text
        }
        vectors.append((vid, embeddings[i].tolist(), meta))

    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[i:i+batch_size])

    print(f" Uploaded {len(vectors)} chunks from '{manual_name}' to Pinecone.")


# Run

if __name__ == "__main__":
    upload_pdf_to_pinecone(PDF_PATH, MANUAL_NAME)



#to delete
# import pinecone

# pc = pinecone.Pinecone(api_key="YOUR_PINECONE_KEY")
# pc.delete_index("combined-manuals-index")   # or "cpc-manual-index"
