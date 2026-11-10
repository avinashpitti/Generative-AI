import os
from dotenv import load_dotenv
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load Environment & Init Groq
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env")

client = Groq(api_key=api_key)

# 2. Document Ingestion & Chunking
# In RAG, we break the raw text into chunks (here, line-by-line / sentence-level)
with open("knowledge.txt", "r", encoding="utf-8") as f:
    raw_lines = f.readlines()

chunks = [line.strip() for line in raw_lines if line.strip()]

print(f"Loaded {len(chunks)} knowledge chunks from knowledge.txt.\n")

# 3. Vectorization (Embedding Generation)
# Convert text chunks into numerical vectors so mathematical comparisons can be made
vectorizer = TfidfVectorizer()
chunk_vectors = vectorizer.fit_transform(chunks)

def retrieve_relevant_chunk(query, top_k=2):
    """Finds the most semantically relevant text chunk matching the query."""
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, chunk_vectors).flatten()
    
    # Sort by similarity score descending
    top_indices = similarities.argsort()[-top_k:][::-1]
    retrieved_texts = [chunks[i] for i in top_indices if similarities[i] > 0.0]
    
    return "\n".join(retrieved_texts)

print("=" * 60)
print("Stage 3: RAG (Retrieval-Augmented Generation)")
print("Ask questions about 'Project Nebula' (knowledge.txt)")
print("Type 'exit' to quit.")
print("=" * 60 + "\n")

# 4. Interactive RAG Query Loop
while True:
    query = input("Ask a question: ").strip()

    if not query:
        continue
    if query.lower() == "exit":
        print("Goodbye!")
        break

    # Step A: Retrieve context
    context = retrieve_relevant_chunk(query)
    
    if not context:
        context = "No relevant context found in document."

    print(f"\n[Retrieved Context from File]:\n{context}\n")

    # Step B: Ground the LLM with an Augmented Prompt
    system_prompt = (
        "You are an accurate corporate assistant. Answer the user's question "
        "STRICTLY using only the provided context. If the answer is not contained "
        "in the context, state clearly: 'I do not have enough information to answer that.'\n"
        f"Context:\n{context}"
    )

    try:
        # Step C: Generate response (Temperature=0 for absolute fidelity)
        response = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.0  # Zero hallucination mode
        )

        answer = response.choices[0].message.content
        print(f"AI Answer: {answer}\n")
        print("-" * 60)

    except Exception as e:
        print(f"Error: {e}\n")