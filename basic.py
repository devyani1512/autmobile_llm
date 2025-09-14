# from dotenv import load_dotenv
# import os
# from sentence_transformers import SentenceTransformer, CrossEncoder
# import numpy as np
# from openai import OpenAI
# import pinecone
# from flask import Flask, render_template, request

# # Load .env variables
# load_dotenv()

# # CONFIG
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# INDEX_NAME = "combined-manuals-index"

# # Embedding + reranker
# embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# # OpenAI client
# client = OpenAI(api_key=OPENAI_API_KEY)

# # Connect to Pinecone
# pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
# index = pc.Index(INDEX_NAME)

# # Flask app
# app = Flask(__name__)


# def chunk_retrieve(query, top_k=10, rerank_top_k=5):
#     # 1) Embed query
#     q_emb = embed_model.encode([query], convert_to_numpy=True)[0].tolist()

#     # 2) Pinecone recall
#     res = index.query(vector=q_emb, top_k=top_k, include_metadata=True)
#     matches = res.get("matches", [])

#     candidates, sources = [], []
#     for m in matches:
#         meta = m.get("metadata", {})
#         if "text" in meta:
#             candidates.append(meta["text"])
#             sources.append(meta.get("manual", "Unknown"))

#     if not candidates:
#         return [], [], "No relevant information found in the manuals."

#     # 3) Rerank
#     pairs = [[query, c] for c in candidates]
#     scores = cross_encoder.predict(pairs)
#     top_indices = np.argsort(scores)[::-1][:rerank_top_k]

#     top_contexts = [candidates[i] for i in top_indices]
#     top_sources = [sources[i] for i in top_indices]

#     # 4) Build OpenAI prompt
#     context_text = "\n\n---\n\n".join(
#         [f"[{src}] {txt}" for src, txt in zip(top_sources, top_contexts)]
#     )
#     prompt = f"""You are a helpful automobile service assistant.
# Use ONLY the information from the CONTEXT.
# If the answer is not in the context, reply: "The manuals do not contain this information."

# CONTEXT:
# {context_text}

# QUESTION:
# {query}

# ANSWER:"""

#     # 5) Query OpenAI
#     response = client.chat.completions.create(
#         model="gpt-4o-mini",  # you can switch to "gpt-4o" if needed
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0
#     )

#     #  FIX: correct way to access response text
#     final_answer = response.choices[0].message.content.strip()


#     return list(zip(candidates, sources)), list(zip(top_contexts, top_sources)), final_answer


# @app.route("/", methods=["GET", "POST"])
# def home():
#     retrieved, reranked, final_answer = [], [], ""
#     query = ""
#     if request.method == "POST":
#         query = request.form.get("query")
#         if query:
#             retrieved, reranked, final_answer = chunk_retrieve(query)

#     return render_template(
#         "basic.html",
#         query=query,
#         retrieved=retrieved,
#         reranked=reranked,
#         final_answer=final_answer
#     )


# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5600, debug=True)





# search_pdf_flan.py
# pip install sentence-transformers pinecone-client numpy transformers torch python-dotenv

# from dotenv import load_dotenv
# import os
# import torch
# from sentence_transformers import SentenceTransformer, CrossEncoder
# import numpy as np
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# import pinecone

# load_dotenv()

# # -------------------------
# # CONFIG
# # -------------------------
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# INDEX_NAME = "combined-manuals-index"

# # Device (GPU if available)
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using device: {device}")

# # Embedding + reranker
# embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
# cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)

# # Hugging Face LLM (Flan-T5)
# llm_model = "google/flan-t5-base"
# tokenizer = AutoTokenizer.from_pretrained(llm_model)
# model = AutoModelForSeq2SeqLM.from_pretrained(llm_model).to(device)

# # Connect to Pinecone
# pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
# index = pc.Index(INDEX_NAME)


# # -------------------------
# # Retrieval + Answering
# # -------------------------
# def chunk_retrieve(query, top_k=10, rerank_top_k=5, log_file="answers.log"):
#     # 1) Embed query
#     q_emb = embed_model.encode([query], convert_to_numpy=True)[0].tolist()

#     # 2) Pinecone recall
#     res = index.query(vector=q_emb, top_k=top_k, include_metadata=True)
#     matches = res.get("matches", [])

#     # Collect candidates
#     candidates = []
#     sources = []
#     for m in matches:
#         if "metadata" in m:
#             candidates.append(m["metadata"]["text"])
#             sources.append(m["metadata"].get("manual", "Unknown"))

#     if not candidates:
#         return "No relevant information found in the manuals."

#     # 3) Rerank with cross-encoder
#     pairs = [[query, c] for c in candidates]
#     scores = cross_encoder.predict(pairs)
#     top_indices = np.argsort(scores)[::-1][:rerank_top_k]
#     top_contexts = [candidates[i] for i in top_indices]
#     top_sources = [sources[i] for i in top_indices]

#     # 4) Build prompt for Flan-T5
#     context_text = "\n\n---\n\n".join(
#         [f"[{src}] {txt}" for src, txt in zip(top_sources, top_contexts)]
#     )

#     prompt = f"""You are a helpful automobile service assistant. 
# Use ONLY the information from the CONTEXT (CPC + Mega AC Service Manuals). 
# If the answer is not in the context, reply: "The manuals do not contain this information."

# Make your explanation clear, step-by-step, and human-friendly. 
# Avoid hallucinations.

# CONTEXT:
# {context_text}

# QUESTION:
# {query}

# ANSWER:"""

#     # 5) Run Flan-T5
#     inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
#     outputs = model.generate(**inputs, max_new_tokens=300, do_sample=False)
#     final_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

#     # 6) Save log
#     with open(log_file, "a", encoding="utf-8") as f:
#         f.write("\n============================\n")
#         f.write(f"QUERY: {query}\n")
#         f.write("\n Retrieved Chunks:\n")
#         for c, s in zip(candidates, sources):
#             f.write(f"- [{s}] {c[:300]}...\n")
#         f.write("\n Reranked Chunks:\n")
#         for c, s in zip(top_contexts, top_sources):
#             f.write(f"- [{s}] {c}\n")
#         f.write("\n Final Answer:\n")
#         f.write(final_answer + "\n")

#     return final_answer


# # -------------------------
# # Example usage
# # -------------------------
# if __name__ == "__main__":
#     user_query = "which condition affect battery charging?"
#     answer = chunk_retrieve(user_query)
#     print("\n=== ANSWER ===\n")
#     print(answer)
#     print("\n(Saved in answers.log )")
from dotenv import load_dotenv
import os
import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
from openai import OpenAI
import pinecone

# NEW: Evaluation imports
from evaluation import load_qa_from_pdf, evaluate, find_closest_question  

# Load env vars
load_dotenv()

# CONFIG
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = "combined-manuals-index"

# Embedding + reranker
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Connect to Pinecone index
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Load gold Q&A dataset (from your PDF)
qa_dict = load_qa_from_pdf("Electric Vehicle Q&A (Based on Training Manual).pdf")


# ---------------- Retrieval + Answer ---------------- #
def chunk_retrieve(query, top_k=10, rerank_top_k=5, log_file="answers.log"):
    q_emb = embed_model.encode([query], convert_to_numpy=True)[0].tolist()
    res = index.query(vector=q_emb, top_k=top_k, include_metadata=True)
    matches = res.get("matches", [])

    candidates, sources = [], []
    for m in matches:
        if "metadata" in m:
            candidates.append(m["metadata"]["text"])
            sources.append(m["metadata"].get("manual", "Unknown"))

    if not candidates:
        return "No relevant information found in the manuals.", [], []

    pairs = [[query, c] for c in candidates]
    scores = cross_encoder.predict(pairs)
    top_indices = np.argsort(scores)[::-1][:rerank_top_k]
    top_contexts = [candidates[i] for i in top_indices]
    top_sources = [sources[i] for i in top_indices]

    context_text = "\n\n---\n\n".join(
        [f"[{src}] {txt}" for src, txt in zip(top_sources, top_contexts)]
    )

    prompt = f"""You are a helpful automobile service assistant. 
Use ONLY the information from the CONTEXT (CPC + Mega AC Service Manuals). 
If the answer is not in the context, reply: "The manuals do not contain this information."

Make your explanation clear, step-by-step, and human-friendly. 
Avoid hallucinations.

CONTEXT:
{context_text}

QUESTION:
{query}

ANSWER:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    final_answer = response.choices[0].message.content.strip()

    return final_answer, list(zip(candidates, sources)), list(zip(top_contexts, top_sources))


#  STREAMLIT APP
st.set_page_config(page_title="Automobile Manual Assistant", layout="wide")

st.title(" Automobile Service Manual Assistant + Evaluation")
st.write("Ask a question based on CPC + Mega AC Service Manuals.")

query = st.text_input("Enter your question:")

if st.button("Search & Analyze"):
    if query.strip():
        with st.spinner("Searching manuals..."):
            bot_answer, retrieved, reranked = chunk_retrieve(query)

        st.subheader("📖 Final Answer (Bot)")
        st.write(bot_answer)

        #  Use fuzzy match to find closest gold question
        matched_q, gold_answer = find_closest_question(query, qa_dict)

        if gold_answer:
            st.subheader(" Gold Answer (from PDF)")
            st.write(gold_answer)
            st.caption(f"(Matched with: {matched_q})")

            bleu, rouge_scores, bert_f1, cosine_sim = evaluate(bot_answer, gold_answer)

            st.subheader(" Evaluation Metrics")
            st.write(f"**BLEU Score:** {bleu:.3f}")
            st.write(f"**ROUGE-1 F1:** {rouge_scores['rouge1'].fmeasure:.3f}")
            st.write(f"**ROUGE-L F1:** {rouge_scores['rougeL'].fmeasure:.3f}")
            st.write(f"**BERTScore F1:** {bert_f1:.3f}")
            st.write(f"**Cosine Similarity:** {cosine_sim:.3f}")

        else:
            st.warning("No sufficiently similar gold answer found in dataset for this question.")

        with st.expander(" Retrieved Chunks"):
            for c, s in retrieved:
                st.markdown(f"**[{s}]** {c}")

        with st.expander(" Reranked Results"):
            for c, s in reranked:
                st.markdown(f"**[{s}]** {c}")
    else:
        st.warning("Please enter a question.")
