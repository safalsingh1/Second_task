from datetime import datetime
import os
import pickle
import pandas as pd
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from database.vector_store import VectorStore
from timescale_vector.client import uuid_from_time

embedding_file = "embeddings.pkl"

def create_embeddings(splits):
    """Create and save embeddings."""
    if os.path.exists(embedding_file):
        print("Removing old embeddings file...")
        os.remove(embedding_file)

    model_name = "BAAI/bge-small-en"
    hf_embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = FAISS.from_documents(splits, hf_embeddings)

    with open(embedding_file, "wb") as f:
        pickle.dump(vectorstore, f)

    return vectorstore

def load_embeddings():
    """Load saved embeddings."""
    if os.path.exists(embedding_file):
        with open(embedding_file, "rb") as f:
            return pickle.load(f)
    else:
        raise FileNotFoundError("No embeddings file found.")

# Initialize VectorStore
vec = VectorStore()

# Drop the existing table if it exists
with vec.vec_client.connect() as conn:
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(f"DROP TABLE IF EXISTS {vec.vector_settings.table_name}")
        conn.commit()

# Read the CSV file
df = pd.read_csv("data/faq_dataset.csv", sep=";")
df.head()

# Prepare data for insertion
def prepare_record(row):
    """Prepare a record for insertion into the vector store."""
    content = f"Question: {row['question']}\nAnswer: {row['answer']}"
    embedding = vec.get_embedding(content)  # Embedding with 384 dimensions
    print(f"Generated embedding for content: {content}\nEmbedding: {embedding}\n")
    return pd.Series(
        {
            "id": str(uuid_from_time(datetime.now())),
            "metadata": {
                "category": row["category"],
                "created_at": datetime.now().isoformat(),
            },
            "contents": content,
            "embedding": embedding,  # This should be a list of 384 floats
        }
    )

# Prepare records and insert them
records_df = df.apply(prepare_record, axis=1)

# Create tables and insert data
vec.create_tables()
vec.create_index()  # DiskAnnIndex
vec.upsert(records_df)
