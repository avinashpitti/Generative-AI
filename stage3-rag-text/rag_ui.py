import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Page Configuration
st.set_page_config(page_title="DocuChat RAG", page_icon="📄", layout="wide")
st.title("📄 DocuChat: Chat with your Documents (RAG)")

# 2. Environment & Groq Client Setup
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("GROQ_API_KEY is missing from your .env file!")
    st.stop()

client = Groq(api_key=api_key)

# 3. Helper Functions: PDF Parsing & Text Chunking
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

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

# 4. Sidebar: File Upload & Indexing
with st.sidebar:
    st.header("📂 Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files", 
        type=["pdf", "txt"], 
        accept_multiple_files=True
    )
    
    if st.button("Process & Index Documents"):
        if not uploaded_files:
            st.warning("Please upload at least one document first.")
        else:
            with st.spinner("Parsing and chunking documents..."):
                all_chunks = []
                for file in uploaded_files:
                    if file.name.endswith(".pdf"):
                        raw_text = extract_text_from_pdf(file)
                    else:
                        raw_text = file.read().decode("utf-8")
                    
                    chunks = chunk_text(raw_text)
                    all_chunks.extend(chunks)

                if all_chunks:
                    # Fit TF-IDF Vectorizer
                    vectorizer = TfidfVectorizer(stop_words="english")
                    chunk_vectors = vectorizer.fit_transform(all_chunks)

                    # Store in session state so it persists across user chats
                    st.session_state.chunks = all_chunks
                    st.session_state.vectorizer = vectorizer
                    st.session_state.chunk_vectors = chunk_vectors
                    st.success(f"Successfully indexed {len(all_chunks)} chunks!")
                else:
                    st.error("No readable text found in uploaded files.")

# 5. Initialize Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display previous messages
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 6. Chat Input & Retrieval-Augmented Generation
user_query = st.chat_input("Ask a question about your uploaded documents...")

if user_query:
    # Check if documents have been indexed
    if "vectorizer" not in st.session_state or "chunks" not in st.session_state:
        st.warning("Please upload and index documents in the sidebar before asking questions.")
    else:
        # Display user message
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Retrieval Step (Top-k = 3)
        query_vector = st.session_state.vectorizer.transform([user_query])
        similarities = cosine_similarity(query_vector, st.session_state.chunk_vectors).flatten()
        top_indices = similarities.argsort()[-3:][::-1]
        
        retrieved_chunks = [
            st.session_state.chunks[i] for i in top_indices if similarities[i] > 0.05
        ]

        context_block = "\n---\n".join(retrieved_chunks) if retrieved_chunks else "No relevant context found."

        # Augmented Prompt
        system_prompt = (
            "You are an expert document assistant. Answer the user's question STRICTLY "
            "using only the provided context below. If the answer is not present in the context, "
            "say 'I cannot find that information in the provided documents.'\n\n"
            f"Context:\n{context_block}"
        )

        # Groq LLM Inference
        with st.chat_message("assistant"):
            with st.spinner("Analyzing context and drafting answer..."):
                try:
                    response = client.chat.completions.create(
                        model="qwen/qwen3.8-27b",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_query}
                        ],
                        temperature=0.0
                    )
                    answer = response.choices[0].message.content
                    st.markdown(answer)

                    # Show retrieved context in an expandable box for transparency
                    with st.expander("🔍 View Retrieved Context"):
                        st.write(context_block)

                    st.session_state.chat_history.append({"role": "assistant", "content": answer})

                except Exception as e:
                    st.error(f"Inference error: {e}")