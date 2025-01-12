from datetime import datetime
import os
import pandas as pd
from database.vector_store import VectorStore
import uuid

vec = VectorStore()
vec.setup_database()
vec.create_tables()

df = pd.read_csv("data/faq_dataset.csv", sep=";")

def prepare_record(row):
    content = f"Question: {row['question']}\nAnswer: {row['answer']}"
    embedding = vec.get_embedding(content)
    return pd.Series({
        "id": str(uuid.uuid4()),
        "metadata": {
            "category": row["category"],
            "created_at": datetime.now().isoformat()
        },
        "contents": content,
        "embedding": embedding
    })

records_df = df.apply(prepare_record, axis=1)
vec.upsert(records_df)
vec.create_index()
