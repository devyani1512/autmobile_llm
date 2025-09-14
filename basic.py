from dotenv import load_dotenv
import os
import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
from openai import OpenAI
import pinecone
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd

# Evaluation 
from evaluation import load_qa_from_pdf, evaluate, find_closest_question

#  CONFIG 
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = "combined-manuals-index"

#  MODELS 
# Rerankers / bi-encoders
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
bi_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    # "MPNet": SentenceTransformer("sentence-transformers/all-mpnet-base-v2"), for gpu


# Embedding model 
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Flan-T5
device = "cuda" if torch.cuda.is_available() else "cpu"
flan_model_id = "google/flan-t5-base"
flan_tokenizer = AutoTokenizer.from_pretrained(flan_model_id)
flan_model = AutoModelForSeq2SeqLM.from_pretrained(flan_model_id).to(device)

# Pinecone index
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# ground truth
qa_dict = load_qa_from_pdf("Electric Vehicle Q&A (Based on Training Manual).pdf")

#  HELPERS 
def ask_openai(prompt, model="gpt-4o-mini"):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()

def ask_flan(prompt):
    inputs = flan_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    outputs = flan_model.generate(**inputs, max_new_tokens=300, do_sample=False)
    return flan_tokenizer.decode(outputs[0], skip_special_tokens=True)

def build_prompt(query, context_text):
    return f"""You are a helpful automobile service assistant. 
Use ONLY the information from the CONTEXT (the provided service manuals) and give the most relevant and crisp answer. 
If the answer is not in the context, reply: "The manuals do not contain this information."

Make your explanation clear, step-by-step, and human-friendly. Avoid hallucinations.

CONTEXT:
{context_text}

QUESTION:
{query}

ANSWER:"""

#  RETRIEVAL + RERANK 
def get_candidates_from_index(query, top_k=10):
    # query via embed_model 
    q_emb = embed_model.encode([query], convert_to_numpy=True)[0].tolist()
    res = index.query(vector=q_emb, top_k=top_k, include_metadata=True)
    matches = res.get("matches", [])

    candidates, sources = [], []
    for m in matches:
        meta = m.get("metadata", {})
        text = meta.get("text") or ""
        src = meta.get("manual", "Unknown")
        candidates.append(text)
        sources.append(src)
    return candidates, sources

def compute_reranks(query, candidates):
    # CrossEncoder scores
    cross_scores = []
    if len(candidates) > 0:
        pairs = [[query, c] for c in candidates]
        cross_scores = cross_encoder.predict(pairs)
    else:
        cross_scores = np.array([])

    # BiEncoder scores (dot product)
    if len(candidates) > 0:
        q_emb_bi = bi_encoder.encode([query], convert_to_numpy=True)[0]
        doc_embs_bi = bi_encoder.encode(candidates, convert_to_numpy=True)
        bi_scores = np.dot(doc_embs_bi, q_emb_bi)
    else:
        bi_scores = np.array([])

    return cross_scores, bi_scores

#  STREAMLIT APP 
st.set_page_config(page_title="Automobile Assistant (Single Table + Reranks)", layout="wide")
st.title(" Automobile Service Manual Assistant — Single Table + Reranked Chunks")

query = st.text_input("Enter your question:")
top_k = st.number_input("Pinecone top_k", value=10, min_value=1, max_value=50, step=1)
rerank_top_k = st.number_input("Rerank top_k to build context", value=5, min_value=1, max_value=20, step=1)

if st.button("Search & Compare"):
    if not query.strip():
        st.warning("Enter a question first.")
    else:
        # Step A: get initial candidates from embedding query
        candidates, sources = get_candidates_from_index(query, top_k=top_k)
        if not candidates:
            st.error("No candidates returned from index.")
        else:
            # Embedding top1 (first item from Pinecone results)
            embed_top1_text = candidates[0]
            embed_top1_src = sources[0]

            # Step B: compute rerank scores for both encoders once
            cross_scores, bi_scores = compute_reranks(query, candidates)

            # get top-1 from cross encoder and bi encoder
            if len(candidates) > 0:
                cross_top_idx = int(np.argmax(cross_scores)) if len(cross_scores) else 0
                bi_top_idx = int(np.argmax(bi_scores)) if len(bi_scores) else 0
                cross_top_text, cross_top_src = candidates[cross_top_idx], sources[cross_top_idx]
                bi_top_text, bi_top_src = candidates[bi_top_idx], sources[bi_top_idx]
            else:
                cross_top_text = bi_top_text = ""
                cross_top_src = bi_top_src = "Unknown"

            # For each retriever (CrossEncoder / BiEncoder) build (top rerank_top_k)
            all_results = []
            contexts_for_display = {}

            for retriever_name in ("CrossEncoder", "BiEncoder"):
                # choose which scores to use for forming the context_text
                scores_use = cross_scores if retriever_name == "CrossEncoder" else bi_scores
                if len(scores_use) == 0:
                    # fallback use Pinecone order
                    top_indices = np.arange(min(len(candidates), rerank_top_k))
                else:
                    top_indices = np.argsort(scores_use)[::-1][:rerank_top_k]

                top_contexts = [(candidates[i], sources[i]) for i in top_indices]
                # join them into the prompt context
                context_text = "\n\n---\n\n".join([f"[{src}] {txt}" for txt, src in top_contexts])
                contexts_for_display[retriever_name] = {
                    "top_contexts": top_contexts,
                    "context_text": context_text
                }

                # Ask LLMs (using the retriever-specific context)
                prompt = build_prompt(query, context_text)
                with st.spinner(f"Querying LLMs with {retriever_name} context..."):
                    try:
                        answer_gpt4o = ask_openai(prompt, "gpt-4o")
                    except Exception as e:
                        answer_gpt4o = f"OpenAI error: {e}"
                    try:
                        answer_gpt4o_mini = ask_openai(prompt, "gpt-4o-mini")
                    except Exception as e:
                        answer_gpt4o_mini = f"OpenAI error: {e}"
                    try:
                        answer_flan = ask_flan(prompt)
                    except Exception as e:
                        answer_flan = f"Flan error: {e}"

                # Ground truth match
                _, gold_answer = find_closest_question(query, qa_dict)

                # Evaluate each model and append to results (row per retriever+LLM)
                for model_name, bot_answer in [
                    ("GPT-4o", answer_gpt4o),
                    ("GPT-4o-mini", answer_gpt4o_mini),
                    ("Flan-T5", answer_flan),
                ]:
                    if gold_answer:
                        bleu, rouge_scores, bert_f1, cosine_sim = evaluate(bot_answer, gold_answer)
                        row = {
                            "Retriever": retriever_name,
                            "Model": model_name,
                            "Answer": bot_answer,
                            "BLEU": round(bleu, 3),
                            "ROUGE-1 F1": round(rouge_scores["rouge1"].fmeasure, 3),
                            "ROUGE-L F1": round(rouge_scores["rougeL"].fmeasure, 3),
                            "BERTScore F1": round(bert_f1, 3),
                            "Cosine Sim": round(cosine_sim, 3),
                            # Context columns (same for all rows for this retriever)
                            "Cross_Reranked_Top1": cross_top_text,
                            "Cross_Reranked_Src": cross_top_src,
                            "Bi_Reranked_Top1": bi_top_text,
                            "Bi_Reranked_Src": bi_top_src,
                            "Embedding_Retrieved_Top1": embed_top1_text,
                            "Embedding_Retrieved_Src": embed_top1_src,
                            "Gold": gold_answer
                        }
                    else:
                        # If no gold found, still record the answer and leave metrics blank
                        row = {
                            "Retriever": retriever_name,
                            "Model": model_name,
                            "Answer": bot_answer,
                            "BLEU": None,
                            "ROUGE-1 F1": None,
                            "ROUGE-L F1": None,
                            "BERTScore F1": None,
                            "Cosine Sim": None,
                            "Cross_Reranked_Top1": cross_top_text,
                            "Cross_Reranked_Src": cross_top_src,
                            "Bi_Reranked_Top1": bi_top_text,
                            "Bi_Reranked_Src": bi_top_src,
                            "Embedding_Retrieved_Top1": embed_top1_text,
                            "Embedding_Retrieved_Src": embed_top1_src,
                            "Gold": None
                        }
                    all_results.append(row)

            # ---------- Display single table (leaderboard) ----------
            if all_results:
                df = pd.DataFrame(all_results)

                st.markdown("##  Single Table: All Retrievers × LLMs")
                # show a trimmed table but include long text columns; use st.dataframe for scrolling
                cols_to_show = [
                    "Retriever", "Model", "BLEU", "ROUGE-1 F1", "ROUGE-L F1",
                    "BERTScore F1", "Cosine Sim", "Answer", 
                    "Cross_Reranked_Top1", "Bi_Reranked_Top1", "Embedding_Retrieved_Top1"
                ]
                st.dataframe(df[cols_to_show], use_container_width=True, height=400)

                # Also show Gold answer (if exists) above the contexts
                golds = df["Gold"].dropna().unique()
                if len(golds) > 0:
                    st.markdown("###  Ground Truth (reference)")
                    st.info(str(golds[0]))

                # ---------- Show the three contexts once per retriever ----------
                st.markdown("##  Reranked / Retrieved Chunks (shown once)")
                for retriever_name, ctx_info in contexts_for_display.items():
                    st.markdown(f"### Retriever: {retriever_name}")
                    st.markdown("- **CrossEncoder top1 (global rerank)**")
                    st.write(cross_top_text)
                    st.markdown("- **BiEncoder top1 (global rerank)**")
                    st.write(bi_top_text)
                    st.markdown("- **Embedding top1 (Pinecone initial)**")
                    st.write(embed_top1_text)

            else:
                st.warning("No results to display.")

