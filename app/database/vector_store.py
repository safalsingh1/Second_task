import logging
import time
from typing import Any, List, Optional, Tuple, Union
from datetime import datetime

import pandas as pd
import psycopg2
from psycopg2.extras import Json
from config.settings import get_settings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings


class VectorStore:
    """A class for managing vector operations and database interactions."""

    def __init__(self):
        """Initialize the VectorStore with settings."""
        self.settings = get_settings()
        self.embedding_model = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-small-en",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        self.vector_settings = self.settings.vector_store
        self.conn_params = {
            'dbname': 'postgres',
            'user': 'postgres',
            'password': 'password',
            'host': 'localhost',
            'port': 5432
        }
        self.table_name = 'vector_store'

    def get_connection(self):
        """Create database connection."""
        return psycopg2.connect(**self.conn_params)

    def setup_database(self):
        """Initialize database with required extensions"""
        conn = self.get_connection()
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")
        conn.close()

    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for the given text.

        Args:
            text: The input text to generate an embedding for.

        Returns:
            A list of floats representing the embedding.
        """
        text = text.replace("\n", " ")
        start_time = time.time()
        embedding = self.embedding_model.embed_documents([text])[0]
        elapsed_time = time.time() - start_time
        logging.info(f"Embedding generated in {elapsed_time:.3f} seconds")
        return embedding

    def create_tables(self) -> None:
        """Create the necessary tables in the database with the correct number of dimensions."""
        conn = self.get_connection()
        with conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id UUID PRIMARY KEY,
                    metadata JSONB,
                    contents TEXT,
                    embedding vector(384)
                )
            """)
            conn.commit()
        conn.close()

    def create_index(self) -> None:
        """Create the Ivfflat index to speed up similarity search"""
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.table_name}_embedding 
                    ON {self.table_name} 
                    USING ivfflat (embedding vector_cosine_ops)
                """)
                conn.commit()
        finally:
            conn.close()

    def _index_exists(self, index_name: str) -> bool:
        """Check if the index exists in the database."""
        query = f"""
        SELECT EXISTS (
            SELECT 1
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relname = '{index_name}'
            AND c.relkind = 'i'
        );
        """
        with self.vec_client.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                return cur.fetchone()[0]

    def drop_index(self) -> None:
        """Drop the StreamingDiskANN index in the database"""
        self.vec_client.drop_embedding_index()

    def upsert(self, df: pd.DataFrame) -> None:
        """
        Insert or update records in the database from a pandas DataFrame.

        Args:
            df: A pandas DataFrame containing the data to insert or update.
                Expected columns: id, metadata, contents, embedding
        """
        conn = self.get_connection()
        with conn.cursor() as cur:
            for _, row in df.iterrows():
                cur.execute(f"""
                    INSERT INTO {self.table_name} 
                    (id, metadata, contents, embedding) 
                    VALUES (%s, %s, %s, %s)
                """, (row.id, Json(row.metadata), row.contents, row.embedding))
            conn.commit()
        conn.close()

    def search(
        self,
        query_text: str,
        limit: int = 5,
        metadata_filter: dict = None,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        return_dataframe: bool = True,
    ) -> Union[List[Tuple[Any, ...]], pd.DataFrame]:
        """
        Query the vector database for similar embeddings based on input text.
        """
        query_embedding = self.get_embedding(query_text)
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                query = f"""
                SELECT id, metadata, contents, 
                       1 - (embedding <=> %s::vector) as similarity
                FROM {self.vector_settings.table_name}
                WHERE 1=1
                """
                params = [query_embedding]
                
                if metadata_filter:
                    query += " AND metadata @> %s::jsonb"
                    params.append(Json(metadata_filter))
                
                query += " ORDER BY similarity DESC LIMIT %s"
                params.append(limit)
                
                cur.execute(query, params)
                results = cur.fetchall()
                
                if return_dataframe:
                    return self._create_dataframe_from_results(results)
                return results

    def _create_dataframe_from_results(
        self,
        results: List[Tuple[Any, ...]],
    ) -> pd.DataFrame:
        """
        Create a pandas DataFrame from the search results.
        """
        return pd.DataFrame(results, columns=['id', 'metadata', 'contents', 'similarity'])

    def delete(
        self,
        ids: List[str] = None,
        metadata_filter: dict = None,
        delete_all: bool = False,
    ) -> None:
        """Delete records from the vector database."""
        if sum(bool(x) for x in (ids, metadata_filter, delete_all)) != 1:
            raise ValueError(
                "Provide exactly one of: ids, metadata_filter, or delete_all"
            )

        if delete_all:
            self.vec_client.delete_all()
            logging.info(f"Deleted all records from {self.vector_settings.table_name}")
        elif ids:
            self.vec_client.delete_by_ids(ids)
            logging.info(
                f"Deleted {len(ids)} records from {self.vector_settings.table_name}"
            )
        elif metadata_filter:
            self.vec_client.delete_by_metadata(metadata_filter)
            logging.info(
                f"Deleted records matching metadata filter from {self.vector_settings.table_name}"
            )
