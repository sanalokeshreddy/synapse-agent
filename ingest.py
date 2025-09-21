# ingest.py (Final Version)
import os
import faiss
import re
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- 1. Load and Clean PDFs ---
def load_and_clean_pdfs(folder_path='documents'):
    all_text = ""
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            filepath = os.path.join(folder_path, filename)
            reader = PdfReader(filepath)
            print(f"Reading and cleaning {filename}...")
            for page in reader.pages:
                page_text = page.extract_text()
                # Clean up the text: replace single newlines with spaces, but keep double newlines
                page_text = re.sub(r'(?<!\n)\n(?!\n)', ' ', page_text)
                all_text += page_text + "\n"
    return all_text

# --- 2. Create Smart Chunks ---
def create_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""] # More robust separators
    )
    chunks = text_splitter.split_text(text)
    return chunks

# --- 3. Create and Save Index ---
def create_faiss_index(chunks, model_name='all-MiniLM-L6-v2'):
    print("Loading sentence transformer model...")
    model = SentenceTransformer(model_name)
    
    print("Creating embeddings...")
    embeddings = model.encode(chunks, show_progress_bar=True)
    
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    print("Saving FAISS index and chunks...")
    faiss.write_index(index, 'index.faiss')
    with open('chunks.txt', 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(f"{chunk.strip()}\n") # Clean up whitespace
            
    print("Ingestion complete!")

# --- Main Execution ---
if __name__ == "__main__":
    cleaned_text = load_and_clean_pdfs()
    text_chunks = create_chunks(cleaned_text)
    create_faiss_index(text_chunks)