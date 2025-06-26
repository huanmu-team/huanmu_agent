import operator
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END

from huanmu_agent.rag.milvus_wrapper import store_doc_to_milvus
from huanmu_agent.utils.rag_utils import download_doc

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
    local_path: str
    error: str
    messages: Annotated[List[BaseMessage], operator.add]

def download_doc_node(state: GraphState):
    """
    Downloads the document from the URL provided in the state.

    Args:
        state: The current graph state.

    Returns:
        A dictionary with the local path of the downloaded file or an error.
    """
    try:
        print("---DOWNLOADING DOCUMENT---")
        file_url = state['file_url']
        local_path = download_doc(file_url)
        print(f"---DOCUMENT DOWNLOADED to {local_path}---")
        return {"local_path": local_path, "error": None}
    except Exception as e:
        print(f"---ERROR DOWNLOADING DOCUMENT: {e}---")
        return {"error": str(e)}

def store_in_milvus_node(state: GraphState):
    """
    Stores the downloaded document into the Milvus vector store.

    Args:
        state: The current graph state.

    Returns:
        A dictionary indicating success or an error.
    """
    try:
        print("---STORING DOCUMENT IN MILVUS---")
        local_path = state['local_path']
        store_doc_to_milvus(local_path)
        print("---DOCUMENT STORED SUCCESSFULLY---")
        return {"error": None}
    except Exception as e:
        print(f"---ERROR STORING DOCUMENT: {e}---")
        return {"error": str(e)}

def should_continue(state: GraphState):
    """
    Determines whether the workflow should continue to the next step or end.

    Args:
        state: The current graph state.

    Returns:
        "continue" if there is no error, otherwise "end".
    """
    if state.get("error"):
        return "end"
    else:
        return "continue"

# Define the workflow
workflow = StateGraph(GraphState)

# Add the nodes
workflow.add_node("download_doc", download_doc_node)
workflow.add_node("store_in_milvus", store_in_milvus_node)

# Set the entrypoint
workflow.set_entry_point("download_doc")

# Add conditional edges
workflow.add_conditional_edges(
    "download_doc",
    should_continue,
    {
        "continue": "store_in_milvus",
        "end": END
    }
)
workflow.add_edge("store_in_milvus", END)

# Compile the workflow
app = workflow.compile()
