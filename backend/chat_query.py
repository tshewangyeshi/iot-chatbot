import faiss
import pickle
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai
import os

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def embed_query(query):
    result = genai.embed_content(
        model="models/embedding-001",
        content=query,
        task_type="retrieval_query"
    )
    return np.array(result["embedding"], dtype="float32")

def retrieve(query, k=5):
    index = faiss.read_index("embeddings/vector.index")
    with open("embeddings/texts.pkl", "rb") as f:
        texts = pickle.load(f)

    q_vec = embed_query(query).reshape(1, -1)
    _, I = index.search(q_vec, k)
    return [texts[i] for i in I[0]]

def ask_gemini(query, retrieved_chunks):
    normalized = query.strip().lower()

    if normalized in ["hi", "hello", "hey"]:
        return (
            "👋 Hello! I’m GAO Bot — your assistant for IoT, sensors, asset tracking, and SOPs at GAO Tek.\n\n"
            "Here’s how I can help you today:\n\n"
            "1️⃣ Enquire about IoT devices for a specific use case\n"
            "2️⃣ Learn about device compatibility, protocols, or setup steps\n"
            "3️⃣ Get a vendor-neutral BOM & system diagram\n"
            "4️⃣ Understand applications of GAO Tek products in smart factories\n"
            "5️⃣ Others (please specify your question)\n\n"
            "Just type the number or your specific query to get started!"
        )

    mapped_prompts = {
        "1": "What IoT devices can I use for a specific use case like predictive maintenance or energy monitoring?",
        "2": "What protocols or platforms are compatible with GAO Tek devices, and how do I set them up?",
        "3": "Can you help me build a vendor-neutral bill of materials (BOM) and system diagram for an IoT project?",
        "4": "How are GAO Tek IoT devices used in smart factories or industrial automation?",
        "5": "I have a different question not listed above. Please help!"
    }

    if normalized in mapped_prompts:
        query = mapped_prompts[normalized]
        retrieved_chunks = retrieve(query)

    context = "\n\n".join(retrieved_chunks)

    if not context.strip():
        return "Sorry, I couldn’t find relevant info. Can you ask about our IoT products, systems, or SOPs?"

    prompt = f"""
You are GAO Bot – an expert assistant for GAO Tek Inc., helping users understand IoT products, sensor systems, protocols, and use cases based on the context below.

Only answer using this context:
{context}

User question: {query}

Answer clearly, helpfully, and professionally. Do NOT introduce yourself. Focus only on the user’s query.
"""

    model = genai.GenerativeModel("gemini-2.5-pro")
    response = model.generate_content(prompt)
    return response.text.strip()
