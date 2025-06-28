import json
from typing import List, Dict, Any, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START, END
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langchain_community.vectorstores import FAISS
from huanmu_agent.configuration import Configuration
from huanmu_agent.rag.local_vectorstore import init_local_vectorstore
from pydantic import BaseModel, Field
from constant import GOOGLE_GEMINI_FLASH_MODEL
import re
from langchain_openai import ChatOpenAI

class ProductResearchState(TypedDict):
    """产品研究代理状态"""
    messages: List[BaseMessage]

class TargetAudience(BaseModel):
    age_range: str = Field(description="适用年龄段")
    skin_type: str = Field(description="适用肤质")
    concerns: List[str] = Field(description="主要痛点")

class ProductAnalysisReport(BaseModel):
    """产品分析报告的结构化输出。"""
    product_type: str = Field(description="产品类型")
    core_benefits: List[str] = Field(description="核心功效")
    key_ingredients: List[str] = Field(description="关键成分")
    target_audience: TargetAudience
    usage_scenarios: List[str] = Field(description="使用场景")
    unique_selling_points: List[str] = Field(description="产品优势")
    marketing_angles: List[str] = Field(description="营销角度")

PRODUCT_RESEARCH_PROMPT = """你是一个专业的产品研究专家。你的任务是：
1. 分析用户在最后一条消息中的需求。
2. 调用一个或多个工具 (`search_milvus_products`, `search_local_products`) 来检索相关产品信息。
3. 收集到足够的信息后，调用 `analyze_products` 工具来总结产品的核心特点和优势。
4. 提供最终的JSON格式产品分析报告作为你的最终答案。请直接输出由 `analyze_products` 工具返回的JSON内容。

请确保信息准确、专业，为后续的营销内容创作提供可靠的基础。"""

# --- LLM and Tools Definition ---

config = Configuration.from_context()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

@tool
def search_milvus_products(query: str) -> List[Dict[str, Any]]:
    """从Milvus向量数据库中检索与查询相关的产品信息。"""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="nomic-ai/nomic-embed-text-v1.5",
            model_kwargs={"trust_remote_code": True},
            encode_kwargs={"normalize_embeddings": True}
        )
        vector_store = Milvus(
            embedding_function=embeddings,
            collection_name=config.milvus_collection_name,
            connection_args={"uri": config.milvus_uri, "user": config.milvus_user, "password": config.milvus_password, "secure": True}
        )
        results = vector_store.similarity_search(query, k=2)
        return [{"content": doc.page_content, "source": "milvus", **doc.metadata} for doc in results]
    except Exception as e:
        return [{"error": f"Milvus检索失败: {e}"}]

@tool
def search_local_products(query: str) -> List[Dict[str, Any]]:
    """从本地产品文档的向量存储中检索相关内容。"""
    try:
        vectorstore = init_local_vectorstore()
        results = vectorstore.similarity_search_with_score(query, k=2)
        return [{"content": doc.page_content, "source": "local", "score": score, **doc.metadata} for doc, score in results]
    except Exception as e:
        return [{"error": f"本地文档检索失败: {e}"}]

@tool
def analyze_products(products: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    分析一个包含多个产品信息的字典列表，并生成一个结构化的JSON分析报告。
    """
    system_prompt = "作为产品分析专家，请对提供的产品信息进行专业分析，并根据提供的JSON schema格式化你的输出。"
    analyzer_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2).with_structured_output(ProductAnalysisReport)
    
    product_content = "\n\n".join([f"来源: {p.get('source', 'N/A')}\n内容: {p.get('content', '')}" for p in products])
    user_prompt = f"请分析以下产品信息：\n\n{product_content}"
    
    response = analyzer_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])
    
    return response.dict()


# --- Agent and Graph Definition ---

tools = [search_milvus_products, search_local_products, analyze_products]

# 1. Create the agent without a prompt
product_research_agent = create_react_agent(llm, tools)

def agent_node(state: ProductResearchState):
    """
    包装节点，用于在调用代理前注入系统提示。
    """
    # 2. Add the system prompt to the state's messages
    messages = [SystemMessage(content=PRODUCT_RESEARCH_PROMPT)] + state["messages"]
    
    # 3. Invoke the agent with the modified state
    result = product_research_agent.invoke({"messages": messages})
        
    return {"messages": result["messages"]}

# 4. Build the graph with the wrapper node
workflow = StateGraph(ProductResearchState)
workflow.add_node("agent", agent_node)
workflow.set_entry_point("agent")
workflow.set_finish_point("agent")
product_research_graph = workflow.compile() 