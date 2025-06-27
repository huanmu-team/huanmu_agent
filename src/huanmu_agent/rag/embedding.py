from typing import List
from google.genai import types
from pymilvus import model

from huanmu_agent.rag import GOOGLE_AI_STUDIO_API_KEY


def embedding_docs(documents: List[str]):
    """
    Generates embeddings for a list of documents using GeminiEmbeddingFunction.

    What does it do:
        This function creates an embedding configuration for retrieval queries,
        initializes the Gemini embedding function, and encodes the input documents
        into embeddings. It wraps the process in a try-except block to handle errors.

    When to use it:
        Use this function when you need to convert a list of text documents into
        vector embeddings for tasks such as semantic search or retrieval.

    How to use it:
        Call embedding_docs(documents) with a list of strings (documents).
        Example:
            embeddings = embedding_docs(["doc1 text", "doc2 text"])

    Args:
        documents (List[str]): List of text documents to embed.

    Returns:
        List[np.array]: List of embedding vectors for the input documents.

    Raises:
        Exception: If any error occurs during embedding generation.
    """
    try:
        retrieval_query_config = types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=768
        )

        gemini_ef = model.dense.GeminiEmbeddingFunction(
            model_name="gemini-embedding-exp-03-07",
            api_key=GOOGLE_AI_STUDIO_API_KEY,
            config=retrieval_query_config
        )
        docs_embeddings = gemini_ef.encode_documents(documents)
        return docs_embeddings
    except Exception as e:
        print(f"Error generating document embeddings: {e}")
        raise

def embedding_query(query: str):
    try:
        retrieval_query_config = types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=768
        )

        gemini_ef = model.dense.GeminiEmbeddingFunction(
            model_name="gemini-embedding-exp-03-07",
            api_key=GOOGLE_AI_STUDIO_API_KEY,
            config=retrieval_query_config
        )
        query_embeddings = gemini_ef.encode_queries(query)
        return query_embeddings
    except Exception as e:
        print(f"Error generating query embeddings: {e}")
        raise