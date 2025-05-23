from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import re

def normalize(text):
    return re.sub(r'[^\w\s]', '', text.lower().strip())

# ✅ Fix ChromaDB client initialization
chroma_client = chromadb.PersistentClient(path="./chroma_data")
collection = chroma_client.get_or_create_collection(
    name="indonesian_faq",
    metadata={"distance_metric": "cosine"}  # or check your chromadb docs for correct param
)

# Load embedding model that supports Indonesian
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Your FAQ data (in Bahasa Indonesia)
faqs = [
    {"id": "1", "question": "Bagaimana cara reset password?", "answer": "Klik tombol 'Lupa Password' di halaman login."},
    {"id": "2", "question": "Jam operasional bisnis Anda?", "answer": "Kami buka Senin sampai Jumat, pukul 08.00 sampai 17.00."},
    {"id": "2a", "question": "Buka jam berapa?", "answer": "Kami buka Senin sampai Jumat, pukul 08.00 sampai 17.00."},
    {"id": "2b", "question": "Jam kerja kantor Anda?", "answer": "Kami buka Senin sampai Jumat, pukul 08.00 sampai 17.00."},
    {"id": "2c", "question": "Hari dan jam operasional Anda?", "answer": "Kami buka Senin sampai Jumat, pukul 08.00 sampai 17.00."},
    {"id": "3", "question": "Bagaimana cara menghubungi layanan pelanggan?", "answer": "Anda bisa menghubungi kami melalui WhatsApp di nomor 0812-3456-7890."}
]

# Prepare and store embeddings
questions = [normalize(f["question"]) for f in faqs]
# questions = [f["question"] for f in faqs]
embeddings = model.encode(questions, normalize_embeddings=True).tolist()

collection.add(
    documents=[f["answer"] for f in faqs],
    metadatas=[{"question": f["question"]} for f in faqs],
    ids=[f["id"] for f in faqs],
    embeddings=embeddings
)

def manual_similarity_check(user_input, faq_questions, model):
    # Encode and normalize
    user_emb = model.encode([user_input], normalize_embeddings=True)
    faq_embs = model.encode(faq_questions, normalize_embeddings=True)
    
    # Cosine similarity (user vs all FAQs)
    cos_sim = cosine_similarity(user_emb, faq_embs)[0]  # 1D array
    # Euclidean distance
    euc_dist = euclidean_distances(user_emb, faq_embs)[0]
    
    print("\nManual similarity scores:")
    for i, q in enumerate(faq_questions):
        print(f"Q: {q}")
        print(f" Cosine similarity: {cos_sim[i]:.4f}, Euclidean distance: {euc_dist[i]:.4f}")
    return cos_sim, euc_dist

# Function to search answer
def search_answer(user_input):
    user_emb = model.encode([user_input], normalize_embeddings=True)
    
    # Query ChromaDB
    result = collection.query(query_embeddings=user_emb.tolist(), n_results=1, include=["distances", "metadatas", "documents"])
    
    documents = result.get('documents')
    metadatas = result.get('metadatas')

    # Manual similarity calculation
    faq_questions = [f["question"] for f in faqs]
    faq_embs = model.encode(faq_questions, normalize_embeddings=True)
    cos_sim = cosine_similarity(user_emb, faq_embs)[0]
    
    # Find best match
    best_idx = np.argmax(cos_sim)
    best_score = cos_sim[best_idx]

    print(f"\nChromaDB matched question: '{metadatas[0][0]['question']}', raw distance: {result['distances'][0][0]:.4f}")
    
    print("\nManual similarity scores:")
    for i, q in enumerate(faq_questions):
        print(f"Q: {q}")
        print(f" Cosine similarity: {cos_sim[i]:.4f}, Euclidean distance: {np.linalg.norm(user_emb - faq_embs[i]):.4f}")

    THRESHOLD = 0.65  # You can tune this
    if best_score >= THRESHOLD:
        return faqs[best_idx]["answer"]
    else:
        return "Maaf, saya belum bisa menjawab pertanyaan itu."

# Test example
if __name__ == "__main__":
    while True:
        user_input = input("\nPertanyaan Anda: ")
        user_input = normalize(user_input)
        if user_input.lower() in ['exit', 'keluar']:
            break
        print("Jawaban Bot:", search_answer(user_input))
