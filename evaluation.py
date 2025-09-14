import pdfplumber
import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from rapidfuzz import process


from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


#  Preload models at startup

bert_model_type = "roberta-base"  # lighter than roberta-large
print(f"🔄 Preloading BERTScore model ({bert_model_type})...")
_ = bert_score(["test"], ["test"], lang="en", model_type=bert_model_type, rescale_with_baseline=True)

print("🔄 Preloading SentenceTransformer (MiniLM)...")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("✅ Models loaded and ready!")


# Load Q&A pairs from PDF
def load_qa_from_pdf(pdf_path):
    qa_dict = {}
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"

    lines = text.split("\n")
    current_q, current_a = None, None

    for line in lines:
        line = line.strip()
        if re.match(r"^Q\d+\.", line):
            if current_q and current_a:
                qa_dict[current_q] = current_a
            current_q = re.sub(r"^Q\d+\.\s*", "", line).strip()
            current_a = ""
        elif re.match(r"^A\d+\.", line):
            current_a = re.sub(r"^A\d+\.\s*", "", line).strip()
        elif current_a is not None:
            current_a += " " + line

    if current_q and current_a:
        qa_dict[current_q] = current_a

    return qa_dict



# Fuzzy match for closest question

def find_closest_question(user_query, qa_dict, threshold=75):
    """
    Finds the closest matching question from qa_dict for a given user query.
    Returns (matched_question, gold_answer) if found, else (None, None).
    """
    questions = list(qa_dict.keys())
    match = process.extractOne(user_query, questions)
    if match:
        q, score, _ = match
        if score >= threshold:
            return q, qa_dict[q]
    return None, None



# Evaluate one Q/A

def evaluate(bot_answer, gold_answer):
    if not gold_answer:
        return None, None, None, None

    # BLEU
    smoothie = SmoothingFunction().method4
    bleu = sentence_bleu([gold_answer.split()], bot_answer.split(), smoothing_function=smoothie)

    # ROUGE
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    rouge_scores = scorer.score(gold_answer, bot_answer)

    # BERTScore (using preloaded roberta-base)
    P, R, F1 = bert_score(
        [bot_answer], [gold_answer],
        lang="en",
        model_type=bert_model_type,
        rescale_with_baseline=True
    )
    bertscore_f1 = F1.mean().item()

    # Cosine Similarity (using MiniLM embeddings)
    emb_gold = embedder.encode([gold_answer], convert_to_numpy=True)
    emb_bot = embedder.encode([bot_answer], convert_to_numpy=True)
    cosine_sim = cosine_similarity(emb_gold, emb_bot)[0][0]

    return bleu, rouge_scores, bertscore_f1, cosine_sim
