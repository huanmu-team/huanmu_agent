from typing import List

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
import requests


def load_and_chunk_word_document(
    file_path: str,
    chunk_size: int = 200,
    chunk_overlap: int = 0,
    **unstructured_kwargs,
) -> List[Document]:
    """
    Loads and chunks a Word document for vector store embedding.

    This function takes a path to a Word document, uses the
    UnstructuredWordDocumentLoader to load its content, and then splits the
    loaded content into smaller chunks for easier processing and embedding in
    a vector store.

    When to use it:
    - Use this function when you need to preprocess a Word document before
      adding it to a RAG (Retrieval-Augmented Generation) system.
    - It's ideal for breaking down large documents into manageable pieces that
      can be easily vectorized and indexed.

    How to use it:
    - Simply call the function with the path to your .docx file.
    - You can optionally specify the chunk size and overlap to control how
      the document is split.
    - Additional arguments for the UnstructuredWordDocumentLoader can also be
      passed as keyword arguments.

    Example:
        chunked_docs = load_and_chunk_word_document("path/to/your/document.docx")
        # or with custom settings
        chunked_docs = load_and_chunk_word_document(
            "path/to/your/document.docx",
            chunk_size=500,
            chunk_overlap=50
        )

    Args:
        file_path (str): The path to the Word document.
        chunk_size (int): The maximum size of each text chunk.
        chunk_overlap (int): The number of characters to overlap between chunks.
        **unstructured_kwargs: Additional arguments for UnstructuredWordDocumentLoader.

    Returns:
        List[Document]: A list of chunked documents, ready for embedding.
    """
    loader = UnstructuredWordDocumentLoader(file_path, **unstructured_kwargs)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunked_docs = text_splitter.split_documents(docs)

    return chunked_docs
  
def download_doc(file_url: str) -> str:
    """
    Downloads a doc file from the given URL and saves it to the project root directory.

    Args:
        file_url (str): The URL of the doc file to download.

    Returns:
        str: The path to the saved file.

    What does it do:
        Downloads a file from a URL and saves it locally, returning the filename. Handles exceptions and reports errors.

    When to use it:
        Use when you need to programmatically download a document from a URL and save it to your local environment, with error handling.

    How to use it:
        Call pre_process_doc(url) with a valid file URL. It returns the saved filename or raises an exception if download fails.
    """
    try:
        doc_response = requests.get(file_url)
        doc_response.raise_for_status()
        # split url to get the filename
        filename = file_url.split("/")[-1]
        with open(filename, "wb") as f:
            f.write(doc_response.content)
        return filename
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file from {file_url}: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error while saving file: {e}")
        raise
