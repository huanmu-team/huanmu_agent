import operator
import os
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END

from huanmu_agent.rag.embedding import embedding_docs
from huanmu_agent.rag.milvus_wrapper import store_doc_to_milvus
from huanmu_agent.utils.rag_utils import download_doc, load_and_chunk_word_document
from huanmu_agent.rag import milvus_client
class GraphStateInput(TypedDict):
    file_url: str

class GraphState(GraphStateInput):
    """
    Represents the state of our graph.

    Attributes:
        file_url: The URL of the document to process.
        local_path: The local path where the document is saved.
        error: A string to hold any error messages that occur.
        messages: A list of messages to track the conversation history.
    """
    file_url: str
    error: str
    messages: Annotated[List[BaseMessage], operator.add]

def ingest_doc_node(state: GraphState):
    """
    Downloads the document from the URL provided in the state.

    Args:
        state: The current graph state.

    Returns:
        A dictionary with the local path of the downloaded file or an error.
    """
    print("---DOWNLOADING DOCUMENT---")
    try:
        file_url = state['file_url']
        local_path = download_doc(file_url)
        print(f"---DOCUMENT DOWNLOADED to {local_path}---")
        chunked_docs = load_and_chunk_word_document(local_path)
        page_contents = [doc.page_content for doc in chunked_docs]
        doc_vectors = embedding_docs(page_contents)
        data = [
        {
            "vector": doc_vectors[i],
            "text": doc.page_content,
            "source": doc.metadata.get("source", local_path),
        }
        for i, doc in enumerate(chunked_docs)
    ]

        res = milvus_client.insert(
            collection_name="company_info_primary_key",
            data=data,      
        )
        # delete the local fileq
        os.remove(local_path)
        # judge if the document is removed
        if not os.path.exists(local_path):
            print(f"---DOCUMENT REMOVED SUCCESSFULLY---")
            return {"error": None, "messages": [f"---DOCUMENT REMOVED SUCCESSFULLY---"]}
        else:
            print(f"---DOCUMENT NOT REMOVED---")
            return {"error": "DOCUMENT NOT REMOVED", "messages": [f"---DOCUMENT NOT REMOVED---"]}
    except Exception as e:
        print(f"---ERROR DOWNLOADING DOCUMENT: {e}---")
        return {"error": str(e)}

# Define the workflow
workflow = StateGraph(GraphState, input=GraphStateInput)

# Add the nodes
workflow.add_node("ingest_doc_node", ingest_doc_node)

# Set the entrypoint
workflow.set_entry_point("ingest_doc_node")

# Compile the workflow
doc_ingestion_workflow = workflow.compile()
