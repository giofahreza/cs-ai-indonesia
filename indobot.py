# chatbot_api.py

from fastapi import FastAPI, Request
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import re

# ----------------- Init FastAPI App ------------------
app = FastAPI(title="Indonesian FAQ Chatbot API")

# ----------------- Utilities ------------------
def normalize(text):
    return re.sub(r'[^\w\s]', '', text.lower().strip())

# ----------------- Load Model and Vector DB ------------------
chroma_client = chromadb.PersistentClient(path="./chroma_data")
collection = chroma_client.get_or_create_collection(
    name="indonesian_faq",
    metadata={"distance_metric": "cosine"}
)

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

faqs = [
    {"id": "1", "question": "Bagaimana cara reset password?", "answer": "Klik tombol 'Lupa Password' di halaman login."},
    {"id": "2", "question": "Jam operasional bisnis Anda?", "answer": "Kami buka Senin sampai Jumat, pukul 08.00 sampai 17.00."},
    {"id": "2a", "question": "Buka jam berapa?", "answer": "Kami buka Senin sampai Jumat, pukul 08.00 sampai 17.00."},
    {"id": "2b", "question": "Jam kerja kantor Anda?", "answer": "Kami buka Senin sampai Jumat, pukul 08.00 sampai 17.00."},
    {"id": "2c", "question": "Hari dan jam operasional Anda?", "answer": "Kami buka Senin sampai Jumat, pukul 08.00 sampai 17.00."},
    {"id": "3", "question": "Bagaimana cara menghubungi layanan pelanggan?", "answer": "Anda bisa menghubungi kami melalui WhatsApp di nomor 0812-3456-7890."}
]

# Add to ChromaDB if not already added
existing = collection.count()
if existing == 0:
    questions = [normalize(f["question"]) for f in faqs]
    embeddings = model.encode(questions, normalize_embeddings=True).tolist()

    collection.add(
        documents=[f["answer"] for f in faqs],
        metadatas=[{"question": f["question"]} for f in faqs],
        ids=[f["id"] for f in faqs],
        embeddings=embeddings
    )

# ----------------- Search Logic ------------------
def search_answer(user_input: str):
    user_input = normalize(user_input)
    user_emb = model.encode([user_input], normalize_embeddings=True)

    result = collection.query(
        query_embeddings=user_emb.tolist(),
        n_results=1,
        include=["distances", "metadatas", "documents"]
    )

    # Manual similarity calculation
    faq_questions = [f["question"] for f in faqs]
    faq_embs = model.encode(faq_questions, normalize_embeddings=True)
    cos_sim = cosine_similarity(user_emb, faq_embs)[0]
    
    # Find best match
    best_idx = np.argmax(cos_sim)
    best_score = cos_sim[best_idx]

    # print(f"\nChromaDB matched question: '{metadatas[0][0]['question']}', raw distance: {result['distances'][0][0]:.4f}")
    
    # print("\nManual similarity scores:")
    # for i, q in enumerate(faq_questions):
    #     print(f"Q: {q}")
    #     print(f" Cosine similarity: {cos_sim[i]:.4f}, Euclidean distance: {np.linalg.norm(user_emb - faq_embs[i]):.4f}")

    THRESHOLD = 0.65  # You can tune this
    if best_score >= THRESHOLD:
        return faqs[best_idx]["answer"]
    else:
        return "Maaf, saya belum bisa menjawab pertanyaan itu."

# ----------------- Request Model ------------------
class ChatRequest(BaseModel):
    message: str

# ----------------- API Route ------------------
@app.post("/chat")
def chatbot_reply(data: ChatRequest):
    response = search_answer(data.message)
    return {"reply": response}
