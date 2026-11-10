import os
import glob
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Environment & API Setup
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env")

client = Groq(api_key=api_key)

# 2. Document Loaders
def extract_text_from_pdf(pdf_path):
    """Extracts raw text page-by-page from a PDF."""
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            full_text += page_text + "\n"
    return full_text

def chunk_text(text, chunk_size=400, overlap=50):
    """
    Slices long text into fixed-size chunks with overlapping boundaries.
    Prevents context from being chopped off at chunk seams.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# 3. Document Ingestion Pipeline
all_chunks = []

# Ingest all .txt and .pdf files found in this directory
for txt_file in glob.glob("*.txt"):
    with open(txt_file, "r", encoding="utf-8") as f:
        all_chunks.extend(chunk_text(f.read()))

for pdf_file in glob.glob("*.pdf"):
    print(f"Loading PDF: {pdf_file}")
    extracted = extract_text_from_pdf(pdf_file)
    all_chunks.extend(chunk_text(extracted))

if not all_chunks:
    raise ValueError("No .txt or .pdf files found to build the knowledge base!")

print(f"Total knowledge chunks indexed: {len(all_chunks)}\n")

# 4. Vectorizer Indexing
vectorizer = TfidfVectorizer(stop_words='english')
chunk_vectors = vectorizer.fit_transform(all_chunks)

def retrieve_context(query, top_k=3):
    """Vector search: queries chunks and scores via cosine similarity."""
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, chunk_vectors).flatten()
    
    top_indices = similarities.argsort()[-top_k:][::-1]
    relevant = [all_chunks[i] for i in top_indices if similarities[i] > 0.05]
    return relevant

# 5. Interactive Chat Loop
print("=" * 60)
print("Stage 3: Advanced Document RAG (PDF & TXT)")
print("Type 'exit' to quit.")
print("=" * 60 + "\n")

while True:
    query = input("Ask a question about your documents: ").strip()
    if not query:
        continue
    if query.lower() == "exit":
        print("Exiting RAG.")
        break

    matched_chunks = retrieve_context(query, top_k=3)
    
    if not matched_chunks:
        context_block = "No relevant context found in documents."
    else:
        context_block = "\n---\n".join(matched_chunks)

    # Augmented System Prompt
    system_prompt = (
        "You are an expert document analysis assistant. Answer the user's "
        "question using ONLY the provided document context below.\n"
        "Rules:\n"
        "1. If the context does not contain the answer, say: 'I cannot find that in the documents.'\n"
        "2. Do not invent or assume facts outside the context.\n\n"
        f"Context:\n{context_block}"
    )

    try:
        response = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.0
        )
        print(f"\nAI Answer:\n{response.choices[0].message.content}\n")
        print("=" * 60)

    except Exception as e:
        print(f"Inference error: {e}")