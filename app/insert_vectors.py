# %%

import logging
from datetime import datetime
import pandas as pd
from database.vector_store import VectorStore
from timescale_vector.client import uuid_from_time

# Initialize VectorStore
vec = VectorStore()

# Delete all existing embeddings
vec.delete(delete_all=True)
logging.info("Deleted all existing embeddings")

# Read the CSV file
df = pd.read_csv("data/faq_dataset.csv", sep=";")

# Prepare the data with proper metadata
processed_data = []
for _, row in df.iterrows():
    # Create metadata dictionary
    metadata = {
        "category": row["category"],
        "question": row["question"],
        "created_at": datetime.now().isoformat(),
    }
    
    # Generate embedding for the question (since we'll search by questions)
    embedding = vec.get_embedding(row["question"])
    
    # Create record with UUID based on current time
    record = {
        "id": uuid_from_time(datetime.now()),
        "metadata": metadata,
        "content": row["answer"],  # Store the answer as the main content
        "embedding": embedding
    }
    processed_data.append(record)

# Convert to DataFrame and insert
insert_df = pd.DataFrame(processed_data)
vec.upsert(insert_df)

# Create the index
vec.create_index()

logging.info(f"Successfully inserted {len(processed_data)} FAQ entries")

# %%
