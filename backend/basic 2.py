# # search_pdf.py
# # pip install sentence-transformers pinecone-client numpy openai

# from dotenv import load_dotenv
# import os
# from sentence_transformers import SentenceTransformer, CrossEncoder
# import numpy as np
# from openai import OpenAI
# import pinecone

# load_dotenv()

# # -------------------------
# # CONFIG
# # -------------------------
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# INDEX_NAME = "cpc-manual-index"

# # Embedding + reranker
# embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2") 
# #BAAI/bge-base-automobile-matryoshka to be figured out

# # OpenAI client
# client = OpenAI(api_key=OPENAI_API_KEY)

# # Connect to Pinecone index
# pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
# index = pc.Index(INDEX_NAME)

# # -------------------------
# # Retrieval and answer
# # -------------------------
# def chunk_retrieve(query, top_k=10, rerank_top_k=5):
#     # 1) Embed query
#     q_emb = embed_model.encode([query], convert_to_numpy=True)[0].tolist()

#     # 2) Pinecone recall
#     res = index.query(vector=q_emb, top_k=top_k, include_metadata=True)
#     matches = res.get("matches", [])

#     candidates = [m["metadata"]["text"] for m in matches if "metadata" in m]
#     if not candidates:
#         return "No relevant information found in the manual."

#     # 3) Rerank with BGE-automobile
#     pairs = [[query, c] for c in candidates]
#     scores = cross_encoder.predict(pairs)
#     top_indices = np.argsort(scores)[::-1][:rerank_top_k]
#     top_contexts = [candidates[i] for i in top_indices]

#     # 4) Build OpenAI prompt
#     context_text = "\n\n---\n\n".join(top_contexts)
#     prompt = f"""You are a helpful automobile service assistant. 
# Use ONLY the information from the CONTEXT (CPC MEGA AC Service Manual). 
# If the answer is not in the context, reply: "The manual does not contain this information."

# Make your explanation clear, step-by-step, and human-friendly. 
# Avoid hallucinations.

# CONTEXT:
# {context_text}

# QUESTION:
# {query}

# ANSWER:"""

#     # 5) Query OpenAI
#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0
#     )

#     return response.choices[0].message.content.strip()

# # -------------------------
# # Example usage
# # -------------------------
# if __name__ == "__main__":
#     user_query = "which condition affect battery charging?"
#     answer = chunk_retrieve(user_query)
#     print("\n=== ANSWER ===\n")
#     print(answer)



# search_pdf.py
# pip install sentence-transformers pinecone-client numpy transformers
