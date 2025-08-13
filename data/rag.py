import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS

def ingest_data():
    # Load data from file
    with open("data/course_info.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Split text into chunks of 800 characters with 100 overlap
    splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_text(raw_text)

    # Load embeddings model
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create embeddings for chunks
    embeddings = embedding_model.embed_documents(chunks)

    # Create or load FAISS index
    if os.path.exists("faiss_index"):
        vector_store = FAISS.load_local("faiss_index", embedding_model)
    else:
        vector_store = FAISS.from_texts(chunks, embedding_model)
        vector_store.save_local("faiss_index")

    print(f"Ingested {len(chunks)} chunks into FAISS index.")

if __name__ == "__main__":
    ingest_data()
