from dotenv import load_dotenv
import os
import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
from openai import OpenAI
import pinecone
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd  # for leaderboard table

# NEW: Evaluation imports
from evaluation import load_qa_from_pdf, evaluate, find_closest_question  

# CONFIG  #
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = "combined-manuals-index"

# Embedding + reranker
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Hugging Face Flan-T5
device = "cuda" if torch.cuda.is_available() else "cpu"
flan_model_id = "google/flan-t5-base"
flan_tokenizer = AutoTokenizer.from_pretrained(flan_model_id)
flan_model = AutoModelForSeq2SeqLM.from_pretrained(flan_model_id).to(device)

# Connect to Pinecone index
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Load gold Q&A dataset (from PDF)
qa_dict = load_qa_from_pdf("Electric Vehicle Q&A (Based on Training Manual).pdf")


#  Retrieval 
def build_context(query, top_k=10, rerank_top_k=5):
    q_emb = embed_model.encode([query], convert_to_numpy=True)[0].tolist()
    res = index.query(vector=q_emb, top_k=top_k, include_metadata=True)
    matches = res.get("matches", [])

    candidates, sources = [], []
    for m in matches:
        if "metadata" in m:
            candidates.append(m["metadata"]["text"])
            sources.append(m["metadata"].get("manual", "Unknown"))

    if not candidates:
        return None, [], []

    pairs = [[query, c] for c in candidates]
    scores = cross_encoder.predict(pairs)
    top_indices = np.argsort(scores)[::-1][:rerank_top_k]
    top_contexts = [candidates[i] for i in top_indices]
    top_sources = [sources[i] for i in top_indices]

    context_text = "\n\n---\n\n".join(
        [f"[{src}] {txt}" for src, txt in zip(top_sources, top_contexts)]
    )
    return context_text, list(zip(candidates, sources)), list(zip(top_contexts, top_sources))


def build_prompt(query, context_text):
    return f"""You are a helpful automobile service assistant. 
Use ONLY the information from the CONTEXT (CPC + Mega AC Service Manuals) and give the most relevant and crisp answer. 
If the answer is not in the context, reply: "The manuals do not contain this information."

Make your explanation clear, step-by-step, and human-friendly. 
Avoid hallucinations.

CONTEXT:
{context_text}

QUESTION:
{query}

ANSWER:"""


#  LLM Wrappers 
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


# STREAMLIT APP 
st.set_page_config(page_title="Automobile Manual Assistant (Multi-LLM Leaderboard)", layout="wide")

st.title(" Automobile Service Manual Assistant + Multi-LLM Leaderboard")
st.write("Compare GPT-4o, GPT-4o-mini, and Flan-T5 on CPC + Mega AC Service Manuals.")

query = st.text_input("Enter your question:")

if st.button("Search & Compare"):
    if query.strip():
        with st.spinner("Retrieving context..."):
            context_text, retrieved, reranked = build_context(query)

        if not context_text:
            st.error("No relevant context found in manuals.")
        else:
            prompt = build_prompt(query, context_text)

            with st.spinner("Querying LLMs..."):
                answers = {
                    "GPT-4o": ask_openai(prompt, "gpt-4o"),
                    "GPT-4o-mini": ask_openai(prompt, "gpt-4o-mini"),
                    "Flan-T5": ask_flan(prompt),
                }

            # Ground truth
            matched_q, gold_answer = find_closest_question(query, qa_dict)

            st.subheader(" Ground Truth (from PDF)")
            if gold_answer:
                st.write(gold_answer)
                st.caption(f"(Matched with: {matched_q})")
            else:
                st.warning("No sufficiently similar ground truth found in dataset.")

            #  Leaderboard 
            if gold_answer:
                results = []
                for model_name, bot_answer in answers.items():
                    bleu, rouge_scores, bert_f1, cosine_sim = evaluate(bot_answer, gold_answer)
                    results.append({
                        "Model": model_name,
                        "BLEU": round(bleu, 3),
                        "ROUGE-1 F1": round(rouge_scores['rouge1'].fmeasure, 3),
                        "ROUGE-L F1": round(rouge_scores['rougeL'].fmeasure, 3),
                        "BERTScore F1": round(bert_f1, 3),
                        "Cosine Sim": round(cosine_sim, 3),
                    })

                df = pd.DataFrame(results)
                st.subheader(" Model Performance Leaderboard")
                st.dataframe(df, use_container_width=True)

            #  Show full answers 
            st.subheader(" Model Answers")
            for model_name, bot_answer in answers.items():
                st.markdown(f"### {model_name}")
                st.write(bot_answer)
                st.markdown("---")

            # Retrieved chunks
            with st.expander(" Retrieved Chunks"):
                for c, s in retrieved:
                    st.markdown(f"**[{s}]** {c}")

            with st.expander(" Reranked Results"):
                for c, s in reranked:
                    st.markdown(f"**[{s}]** {c}")
    else:
        st.warning("Please enter a question.")
