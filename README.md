# AI-Powered Technical Assistant for Modern Vehicles

An AI-powered technical assistant that helps vehicle owners and professional mechanics interact with automotive service manuals through natural language. The system combines a Retrieval-Augmented Generation (RAG) pipeline with a role-based frontend, and includes a comparative research benchmark of embedding models, rerankers, and LLMs for automotive question answering.

## Overview

Modern vehicles are increasingly complex, and the service manuals that document them are long, technical, and difficult to search. Traditional keyword search fails on natural-language queries and cannot connect answers to relevant diagrams or tables.

This project builds a dual-purpose system:

- A practical AI assistant that lets vehicle owners and mechanics query service manuals in plain language.
- A research framework that benchmarks embedding models, cross-encoder rerankers, and LLMs to identify the most accurate and efficient configuration for domain-specific automotive retrieval.

## Key Features

### Vehicle Owner Mode
- Ask Technical Questions: free-text queries answered with simplified, user-friendly explanations.
- Explore Your Vehicle: conceptual, component-level information.
- Emergency Support: pre-curated guidance for scenarios such as flat tires, engine overheating, brake failure, and dead batteries, with follow-up conversation support.

### Professional Mechanic Mode
- Workshop Assistant: in-depth diagnostic reasoning and technical specifications.
- Service Report Generation: structured reports from vehicle info (VIN, make, model, year), problem description, diagnostic findings, parts, labor, and estimated cost, with download, print, and email options.

### Backend Intelligence Layer
- PDF ingestion and preprocessing (text, tables, and images) from automotive service manuals.
- Hybrid chunking based on document headings and length-based overlapping windows.
- Multi-model embedding generation (MiniLM, E5, BGE, and others) for semantic representation comparison.
- Vector storage and retrieval using Pinecone.
- Cross-encoder reranking to prioritize the most relevant retrieved chunks.
- Answer generation using multiple LLMs, grounded strictly in retrieved manual content.
- Automated evaluation pipeline using BLEU, ROUGE, BERTScore, and cosine similarity against gold-standard answers.

## System Architecture

The system follows a modular RAG pipeline:

1. **PDF Processing** – Extracts text, tables, and images from uploaded manuals using PyMuPDF (fitz) and pdfplumber; tables and images are marked with placeholder tags in the text.
2. **Hybrid Chunking** – Splits content by headings and numbered sections, with overlapping word-count windows for long sections, preserving context.
3. **Embedding Generation** – Encodes chunks using SentenceTransformer models (primarily `all-MiniLM-L6-v2`) into dense vectors.
4. **Vector Storage** – Stores embeddings in Pinecone, organized by a shared index or per-vehicle namespace.
5. **Query and Retrieval** – Embeds the user query, retrieves top-k candidate chunks via Pinecone similarity search.
6. **Reranking** – Applies cross-encoder models to reorder candidates and select the most relevant context.
7. **Answer Generation** – Combines the query and reranked context into a prompt sent to an LLM (e.g., GPT-4o, GPT-4o-mini, Flan-T5), configured with low temperature to reduce hallucination.
8. **Evaluation** – Compares generated answers against gold-standard references using BLEU, ROUGE, BERTScore, and cosine similarity, displayed on a results leaderboard.

## Tech Stack

| Component | Technology |
|---|---|
| Embeddings | sentence-transformers (MiniLM, MPNet, and others) |
| Vector Database | Pinecone |
| PDF Extraction | PyMuPDF (fitz), pdfplumber |
| Data Handling | pandas |
| Frontend | React |
| Answer Generation | OpenAI GPT-4o / GPT-4o-mini, Flan-T5 |
| Fuzzy Matching | rapidfuzz |
| Evaluation | nltk, rouge_score, bert_score, scikit-learn |

## Model Comparison Summary

A comparative benchmark was run across six automotive query types (simple fact, procedural, diagnostic, specification, comparison, and complex multi-section queries) to compare `MiniLM` against `MiniLM_Large`.

| Metric | MiniLM | MiniLM_Large |
|---|---|---|
| Avg. Retrieval Time | ~381 ms | ~472 ms |
| Avg. Similarity Score | 0.48 | 0.35 |
| Fastest Query | 286 ms | 380 ms |
| Slowest Query | 738 ms | 818 ms |

**Result:** MiniLM was selected as the preferred embedding model. It was consistently faster (approximately 23–25 percent), produced higher similarity scores, performed more consistently across all query types, and is computationally lighter — making it well suited for real-time automotive RAG applications.

Six retriever–LLM combinations (bi-encoder and cross-encoder retrievers paired with GPT-4o, GPT-4o-mini, and Flan-T5) were also evaluated on a sample diagnostic query using BLEU, ROUGE-1/L, BERTScore, and cosine similarity. Results indicated that answer quality was more sensitive to context specificity than to the choice of retriever or generator model alone.

## Evaluation Methodology

The evaluation engine:
1. Loads question/answer pairs from a reference document.
2. Matches user queries to the closest known question using fuzzy matching (rapidfuzz).
3. Compares the generated answer to the gold-standard answer using BLEU, ROUGE-1, ROUGE-L, BERTScore, and cosine similarity.
4. Displays results, including the retrieved context and ground truth, on a leaderboard for comparison.

## Risk Analysis and Mitigation

| Risk | Mitigation |
|---|---|
| Incorrect or incomplete PDF extraction | Multiple extraction strategies (fitz + pdfplumber), with error logging |
| Inaccurate retrieval due to weak embeddings | Strong yet efficient embedding model, heading-based chunking, overlapping context |
| AI hallucination | Strict grounding prompts, low generation temperature (0.3) |
| Slow response times | Preloaded models, Pinecone vector search |
| Data inconsistency across manuals | Namespaced indexing and a vehicle database keyed by manufacturer/model/year |

## Known Limitations

- Extraction accuracy depends on the quality and formatting of the uploaded manual; poorly scanned PDFs reduce reliability.
- Evaluation requires properly formatted Q/A reference documents (e.g., `Q1./A1.` patterns).
- The system currently displays images and tables but does not perform deep visual reasoning over diagrams.
- Manuals for different vehicle models must be indexed and managed separately to avoid cross-contamination of results.

## Future Work

- Full integration of all frontend flows (Explore Your Vehicle, Emergency Support, Service Reports) with the backend RAG pipeline.
- Visual understanding of diagrams and dashboard icons using vision models.
- Voice-based interaction (speech-to-text and text-to-speech) for hands-free use.
- Feedback-driven improvement (RLHF-style) using user ratings on answer quality.
- Reinforcement learning for retrieval optimization based on user interaction patterns.

## Project Information

- **Title:** AI-Powered Technical Assistant for Modern Vehicles
- **Institution:** Jaypee Institute of Information Technology, Noida
- **Department:** Computer Science and Information Technology
- **Program:** B.Tech, 5th Semester, 3rd Year
- **Supervisor:** Prof. Anuja Arora
- **Contributors:** Devyani Sharma, Anushka Tayal, Shambhavi Tripathi
- **Date:** November 2025

## References

Key references include Sentence-BERT (Reimers and Gurevych, 2019), Pinecone vector database documentation, PyMuPDF and pdfplumber documentation, OpenAI GPT-4o model documentation, and BERTScore (Zhang et al., ICLR 2020). A complete IEEE-format reference list is included in the project report.
