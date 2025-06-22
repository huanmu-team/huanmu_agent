from huanmu_agent.rag.milvus_wrapper import pre_process_doc
from huanmu_agent.utils.rag_utils import load_and_chunk_word_document

if __name__ == "__main__":
    pre_process_doc("https://www.consumerclone.com:9090/tenant-0/wechat/aifile/0/1750340857276_langgraph_介绍.docx")
    # print(load_and_chunk_word_document("langgraph_介绍.docx"))