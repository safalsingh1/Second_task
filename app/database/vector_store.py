import logging
import time
from typing import Any, List, Optional, Tuple, Union
from datetime import datetime

import pandas as pd
from config.settings import get_settings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from timescale_vector import client


class VectorStore:
    """A class for managing vector operations and database interactions."""

    def __init__(self):
        """Initialize the VectorStore with settings and Timescale Vector client."""
        self.settings = get_settings()
        self.embedding_model = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-small-en",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        self.vector_settings = self.settings.vector_store
        
        # Added num_dimensions argument to specify the embedding dimensions
        self.vec_client = client.Sync(
            self.settings.database.service_url,
            self.vector_settings.table_name,
            num_dimensions=384,  # Number of dimensions for BGAI/bge-small-en embeddings
            time_partition_interval=self.vector_settings.time_partition_interval,
        )

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
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {self.vector_settings.table_name} (
            id UUID PRIMARY KEY,
            metadata JSONB,
            contents TEXT,
            embedding VECTOR({self.vector_settings.embedding_dimensions})
        );
        """
        with self.vec_client.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(create_table_query)
                conn.commit()

    def create_index(self) -> None:
        """Create the Ivfflat index to speed up similarity search"""
        index_name = f"{self.vector_settings.table_name}_embedding_idx"
        if not self._index_exists(index_name):
            self.vec_client.create_embedding_index(client.IvfflatIndex())
        else:
            logging.info(f"Index {index_name} already exists")

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
        records = df.to_records(index=False)
        self.vec_client.upsert(list(records))
        logging.info(
            f"Inserted {len(df)} records into {self.vector_settings.table_name}"
        )

    def search(
        self,
        query_text: str,
        limit: int = 5,
        metadata_filter: Union[dict, List[dict]] = None,
        predicates: Optional[client.Predicates] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        return_dataframe: bool = True,
    ) -> Union[List[Tuple[Any, ...]], pd.DataFrame]:
        """
        Query the vector database for similar embeddings based on input text.
        """
        query_embedding = self.get_embedding(query_text)

        start_time = time.time()

        search_args = {
            "limit": limit,
        }

        if metadata_filter:
            search_args["filter"] = metadata_filter

        if predicates:
            search_args["predicates"] = predicates

        if time_range:
            start_date, end_date = time_range
            search_args["uuid_time_filter"] = client.UUIDTimeRange(start_date, end_date)

        results = self.vec_client.search(query_embedding, **search_args)
        elapsed_time = time.time() - start_time

        logging.info(f"Vector search completed in {elapsed_time:.3f} seconds")

        if return_dataframe:
            return self._create_dataframe_from_results(results)
        else:
            return results

    def _create_dataframe_from_results(
        self,
        results: List[Tuple[Any, ...]],
    ) -> pd.DataFrame:
        """
        Create a pandas DataFrame from the search results.
        """
        df = pd.DataFrame(
            results, columns=["id", "metadata", "content", "embedding", "distance"]
        )

        df = pd.concat(
            [df.drop(["metadata"], axis=1), df["metadata"].apply(pd.Series)], axis=1
        )

        df["id"] = df["id"].astype(str)

        return df

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
