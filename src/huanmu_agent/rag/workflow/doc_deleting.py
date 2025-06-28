import operator
from typing import TypedDict, Annotated, List

from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END

from huanmu_agent.rag import milvus_client


class GraphStateInput(TypedDict):
    file_name: str


class GraphState(GraphStateInput):
    """
    Represents the state of our graph for document deletion.

    Attributes:
        file_name: The name of the document to be deleted.
        error: A string to hold any error messages that occur.
        messages: A list of messages to track the conversation history.
    """
    file_name: str
    error: str
    messages: Annotated[List[BaseMessage], operator.add]


def delete_doc_node(state: GraphState):
    """
    Deletes documents from Milvus based on the source file name.

    Args:
        state: The current graph state.

    Returns:
        A dictionary with an error status or success message.
    """
    print("---DELETING DOCUMENT FROM MILVUS---")
    try:
        file_name = state['file_name']
        filter_expression = f"source == '{file_name}'"
        
        milvus_client.delete(
            collection_name="company_info_primary_key",
            filter=filter_expression,
        )
        success_message = f"---DOCUMENT {file_name} DELETED SUCCESSFULLY---"
        print(success_message)
        return {"error": None, "messages": [success_message]}

    except Exception as e:
        error_message = f"---ERROR DELETING DOCUMENT: {e}---"
        print(error_message)
        return {"error": str(e)}


# Define the workflow
workflow = StateGraph(GraphState, input=GraphStateInput)

# Add the nodes
workflow.add_node("delete_doc_node", delete_doc_node)

# Set the entrypoint
workflow.set_entry_point("delete_doc_node")

# Compile the workflow
doc_deleting_workflow = workflow.compile()