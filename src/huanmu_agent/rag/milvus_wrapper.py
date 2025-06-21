from langchain_google_vertexai import VertexAIEmbeddings
from langchain_milvus import Milvus

from huanmu_agent.utils.rag_utils import load_and_chunk_word_document


class MilvusConnection:
    """
    Manages a reusable connection to a Milvus vector store.

    What does it do:
        This class creates and holds a single instance of a Milvus vector store,
        preventing the need to reconnect for every operation. It follows a
        singleton-like pattern where the vector store is initialized once.

    When to use it:
        Use this class when you need to perform multiple operations on a Milvus
        database (e.g., adding documents, searching) and want to avoid the
        overhead of establishing a new connection each time. It's suitable
        for applications where the Milvus URI is constant.

    How to use it:
        1. Instantiate the class once in your application's startup code:
           `milvus_conn = MilvusConnection(uri="your_milvus_uri.db")`
        2. Access the vector store instance via the `get_vector_store()` method:
           `vector_store = milvus_conn.get_vector_store()`
        3. Use the `vector_store` object for Milvus operations.
    """
    _vector_store: Milvus | None = None

    def __init__(self, uri: str = "milvus_example.db"):
        """
        Initializes the Milvus connection if it doesn't already exist.
        Args:
            uri (str): The URI for the Milvus database.
        """
        if MilvusConnection._vector_store is None:
            try:
                embeddings = VertexAIEmbeddings(model="text-multilingual-embedding-002")
                MilvusConnection._vector_store = Milvus(
                    embedding_function=embeddings,
                    connection_args={"uri": uri},
                    index_params={
                        "index_type": "AUTOINDEX",
                        "metric_type": "L2",
                        "params": {},
                    }
                )
            except Exception as e:
                print(f"Failed to initialize Milvus connection: {e}")
                raise

    def get_vector_store(self) -> Milvus | None:
        """
        Retrieves the active Milvus vector store instance.

        What does it do:
            Returns the singleton instance of the Milvus vector store that is
            managed by this class.

        When to use it:
            Call this method whenever you need to interact with the Milvus
            vector store.

        How to use it:
            `vector_store = milvus_connection.get_vector_store()`

        Returns:
            The Milvus vector store instance, or None if it's not initialized.
        """
        return MilvusConnection._vector_store

milvus_connection = MilvusConnection()

def store_doc_to_milvus(file_path: str):
    """
    Stores chunked documents from a Word file into a Milvus vector store.

    What does it do:
        Loads and chunks a Word document, generates embeddings, and stores them in Milvus.
        Handles exceptions during the process and reports errors.

    When to use it:
        Use when you need to index a document into Milvus for semantic search or retrieval.

    How to use it:
        Call store_doc_to_milvus(file_path) with the path to your .docx file.
        It will use the globally managed Milvus connection.

    Args:
        file_path (str): Path to the Word document.

    Raises:
        Exception: If any error occurs during document loading, embedding, or storage.
    """
    try:
        chunked_docs = load_and_chunk_word_document(file_path)
        vector_store = milvus_connection.get_vector_store()
        if not vector_store:
            raise Exception("Milvus connection not established.")
        vector_store.add_documents(chunked_docs)
    except Exception as e:
        print(f"Error storing document to Milvus: {e}")
        raise

def query_milvus(query: str, k: int = 2):
    """
    Performs a similarity search on the Milvus vector store.

    What does it do:
        Executes a similarity search for a given query string against the
        documents stored in Milvus and returns the top k matching results.

    When to use it:
        Use this function when you need to retrieve documents from Milvus that
        are semantically similar to a given query text.

    How to use it:
        `results = query_milvus("your search query", k=5)`
        The function will print the results and also return them.

    Args:
        query (str): The text to search for.
        k (int): The number of similar documents to return.

    Returns:
        A list of matching documents.

    Raises:
        Exception: If the Milvus connection is not established.
    """
    try:
        vector_store = milvus_connection.get_vector_store()
        if not vector_store:
            raise Exception("Milvus connection not established.")

        results = vector_store.similarity_search(query, k=k)

        print(f"Query: '{query}', k={k}")
        for res in results:
            print(f"* {res.page_content} [{res.metadata}]")

        return results
    except Exception as e:
        print(f"Error querying Milvus: {e}")
        raise

def get_retriever(k: int = 4):
    """
    Creates a retriever from the Milvus vector store.

    What does it do:
        This function returns a LangChain Retriever instance from the global
        Milvus vector store. The retriever is configured for similarity search.

    When to use it:
        Use this function to get a retriever for RAG (Retrieval-Augmented
        Generation) chains where you need to fetch documents based on a query.

    How to use it:
        `retriever = get_retriever(k=5)`
        `docs = retriever.invoke("my query")`

    Args:
        k (int): The number of documents to retrieve.

    Returns:
        A LangChain retriever instance.

    Raises:
        Exception: If the Milvus connection is not established.
    """
    vector_store = milvus_connection.get_vector_store()
    if not vector_store:
        raise Exception("Milvus connection not established.")
    return vector_store.as_retriever(search_kwargs={'k': k})

# embeddings = VertexAIEmbeddings(model="text-multilingual-embedding-002")

# URI = "./milvus_example.db"

# vector_store = Milvus(
#     embedding_function=embeddings,
#     connection_args={"uri": URI},
#     index_params={
#         "index_type": "AUTOINDEX",
#         "metric_type": "L2",
#         "params": {},
#     },
# )

# uuids = [str(uuid4()) for _ in range(len(documents))]

# vector_store.add_documents(documents=documents, ids=uuids)

# results = vector_store.similarity_search(
#     "what is Langchain",
#     k=2,
#     expr='source == "tweet"',
# )
# for res in results:
#     print(f"* {res.page_content} [{res.metadata}]")
    
# retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 1})
# retriever.invoke("Stealing from the bank is a crime", filter={"source": "news"})
