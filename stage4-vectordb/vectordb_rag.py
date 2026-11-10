import os
import glob
from dotenv import load_dotenv
import chromadb
from groq import Groq
from pypdf import PdfReader

# 1. Setup API and ChromaDB Client
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env")

groq_client = Groq(api_key=api_key)

# Persistent ChromaDB client saves embeddings to disk inside ./chroma_db
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="company_knowledge")

# 2. Helpers for Text Extraction & Chunking
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            full_text += page_text + "\n"
    return full_text

def chunk_text(text, chunk_size=400, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# 3. Index Documents into Vector DB
def index_documents():
    # Only index if the collection is currently empty
    if collection.count() > 0:
        print(f"Vector DB already has {collection.count()} chunks indexed.")
        return

    docs_to_add = []
    metadatas = []
    ids = []
    doc_counter = 0

    # Ingest text files
    for txt_file in glob.glob("*.txt"):
        with open(txt_file, "r", encoding="utf-8") as f:
            chunks = chunk_text(f.read())
            for idx, c in enumerate(chunks):
                docs_to_add.append(c)
                metadatas.append({"source": txt_file, "chunk_id": idx})
                ids.append(f"txt_{doc_counter}")
                doc_counter += 1

    # Ingest PDF files
    for pdf_file in glob.glob("*.pdf"):
        chunks = chunk_text(extract_text_from_pdf(pdf_file))
        for idx, c in enumerate(chunks):
            docs_to_add.append(c)
            metadatas.append({"source": pdf_file, "chunk_id": idx})
            ids.append(f"pdf_{doc_counter}")
            doc_counter += 1

    if docs_to_add:
        print(f"Generating semantic embeddings and storing {len(docs_to_add)} chunks into ChromaDB...")
        collection.add(documents=docs_to_add, metadatas=metadatas, ids=ids)
        print("Indexing complete!\n")
    else:
        print("No documents found to index.")

# 4. Ingestion Step
index_documents()

# 5. Semantic Query Loop
print("=" * 60)
print("Stage 4: Production Vector DB (ChromaDB + Semantic Embeddings)")
print("Type 'exit' to quit.")
print("=" * 60 + "\n")

while True:
    user_query = input("Ask a question: ").strip()
    if not user_query:
        continue
    if user_query.lower() == "exit":
        break

    # Vector search: queries ChromaDB using semantic similarity
    query_results = collection.query(
        query_texts=[user_query],
        n_results=2
    )

    retrieved_docs = query_results["documents"][0]
    sources = [m["source"] for m in query_results["metadatas"][0]]

    context_block = "\n---\n".join(retrieved_docs) if retrieved_docs else "No context found."

    system_prompt = (
        "You are an expert enterprise assistant. Answer the user's question "
        "using ONLY the retrieved document chunks below. If the information is not "
        "present, say 'I cannot find that in the documents.'\n\n"
        f"Context:\n{context_block}"
    )

    try:
        response = groq_client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            temperature=0.0
        )
        print(f"\nAI Answer:\n{response.choices[0].message.content}")
        print(f"(Sources used: {', '.join(set(sources))})\n")
        print("-" * 60)

    except Exception as e:
        print(f"Error during inference: {e}")