# ⚖️ LegalMind — Smart Legal Consultant

LegalMind is an AI-powered legal assistant designed to help users search and understand **Bahraini legal information** through natural-language questions.

The project combines OCR, multilingual embeddings, vector search, RAG, and an LLM to provide relevant and source-based legal answers.

## 👥 Team

- Zainab Abdulwahab
- Fatima Khalifa
- Fatima Shamlooh

---

## 🚀 How It Works

LegalMind follows a five-step pipeline:

### 1. Data Collection
Legal information is collected from trusted and approved sources.

For scanned legal books and documents, **MacOCR** is used to extract the Arabic text.

### 2. Vector Embedding
The extracted text is cleaned and divided into smaller chunks.

A **multilingual embedding model** converts the legal text into vectors, allowing the system to identify semantically similar legal content.

### 3. RAG Retrieval
When a user asks a question, the system searches the vector database and retrieves the most relevant legal passages.

Article numbers, dates, and source information are preserved during retrieval.

### 4. LLM Generation
The retrieved legal context is provided to the **ChatGroq LLM**, which generates a clear and grounded answer based on the available sources.

### 5. Chatbot
The final answer is displayed through an **Arabic Streamlit chatbot**, together with the relevant legal sources whenever available.

---

## 🧠 Technology Stack

| Component | Technology |
|---|---|
| OCR | MacOCR |
| Text Processing | Arabic Text Processing |
| Embeddings | Multilingual Embedding Model |
| Retrieval | Vector Similarity Search |
| Architecture | RAG |
| LLM | ChatGroq |
| Interface | Streamlit |

---

## 🔄 Pipeline

```text
Legal Sources
     ↓
OCR & Data Collection
     ↓
Cleaning & Chunking
     ↓
Multilingual Embeddings
     ↓
Vector Database
     ↓
RAG Retrieval
     ↓
ChatGroq LLM
     ↓
Arabic Answer + Sources
```

---

## 🛡️ Key Features

- 🔎 Semantic legal search
- 🇧🇭 Focused on Bahraini legal information
- 🤖 AI-powered question answering
- 📚 Source-based responses
- 🔗 Legal article and source references
- 🛡️ RAG-based approach to reduce hallucinations
- 💬 Arabic chatbot interface

---

## 🎯 Project Goal

The goal of LegalMind is to make legal research **faster, easier, and more accessible** by helping users find relevant Bahraini legal information without manually searching through large numbers of legal documents.

> **Note:** LegalMind is an AI-assisted legal information tool and should not be considered a substitute for professional legal advice.

