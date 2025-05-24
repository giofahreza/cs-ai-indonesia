# chatbot_api.py

from fastapi import FastAPI, Request
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import re
from fastapi.responses import JSONResponse
import json

with open("faqs.json") as f:
    faqs = json.load(f)

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
        return {
            "code": 200,
            "status": True,
            "matched_question": faqs[best_idx]["question"],
            "answer": faqs[best_idx]["answer"],
            "similarity_score": float(best_score)  # <-- convert to Python float
        }
    else:
        return {
            "code": 404,
            "status": False,
            "matched_question": None,
            "answer": "Maaf, saya tidak dapat menemukan jawaban untuk pertanyaan Anda.",
            "similarity_score": float(best_score)  # <-- convert to Python float
        }

# ----------------- Request Model ------------------
class ChatRequest(BaseModel):
    message: str

# ----------------- API Route ------------------
@app.post("/chat")
def chatbot_reply(data: ChatRequest):
    response = search_answer(data.message)
    return JSONResponse(content=response, status_code=response["code"])

# ----------------- Run the app ------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8082)
