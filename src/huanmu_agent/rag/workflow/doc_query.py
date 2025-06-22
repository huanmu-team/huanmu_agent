import operator
from typing import TypedDict, Annotated, List

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import StateGraph, END

from huanmu_agent.rag.milvus_wrapper import get_retriever
from huanmu_agent.utils.langchain_utils import load_chat_model


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        keys: A dictionary containing the query and other details.
        documents: A list of retrieved documents.
        generation: The generated response from the LLM.
        messages: A list of messages to track the conversation history.
    """q
    query: str
    index_text: str
    messages: Annotated[List[BaseMessage], operator.add]

def retrieve_node(state: GraphState):
    """
    Retrieves documents from the vector store.

    Args:
        state (GraphState): The current graph state.

    Returns:
        A dictionary with the retrieved documents.
    """
    print("---RETRIEVING DOCUMENTS---")
    keys = state['keys']
    query = keys['query']
    retriever = get_retriever()
    documents = retriever.invoke(query)
    print("---DOCUMENTS RETRIEVED---")
    return {"documents": documents}

def rag_chain_node(state: GraphState):
    """
    Generates a response using a RAG chain.

    Args:
        state (GraphState): The current graph state.

    Returns:
        A dictionary with the generated response.
    """
    print("---GENERATING RESPONSE---")
    keys = state['keys']
    query = keys['query']
    documents = state['documents']

    # RAG prompt
    prompt = PromptTemplate(
        template="""You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Keep the answer concise.

    Question: {question}
    Context: {context}
    Answer:""",
        input_variables=["question", "context"],
    )

    # LLM
    llm = load_chat_model('google/gemini-pro', temperature=0)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # RAG chain
    rag_chain = (
        {"context": lambda x: format_docs(documents), "question": lambda x: query}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Generate response
    generation = rag_chain.invoke("")
    print("---RESPONSE GENERATED---")
    return {"generation": generation}


def should_continue(state: GraphState):
    """
    Determines whether to continue to generation.

    Args:
        state (GraphState): The current graph state.

    Returns:
        "continue" if documents are found, otherwise "end".
    """
    if not state['documents']:
        return "end"
    else:
        return "continue"

# Define the workflow
workflow = StateGraph(GraphState)

# Add the nodes
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("rag_chain", rag_chain_node)

# Set the entrypoint
workflow.set_entry_point("retrieve")

# Add conditional edges
workflow.add_conditional_edges(
    "retrieve",
    should_continue,
    {
        "continue": "rag_chain",
        "end": END,
    },
)

# Add an edge from RAG chain to the end
workflow.add_edge("rag_chain", END)

# Compile the workflow
app = workflow.compile()
