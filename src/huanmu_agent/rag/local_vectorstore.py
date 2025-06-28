import os
from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

_local_vectorstore: Optional[VectorStore] = None


def init_local_vectorstore() -> VectorStore:
    """初始化本地向量存储"""
    global _local_vectorstore

    if _local_vectorstore is not None:
        return _local_vectorstore

    try:
        # 1. 加载本地文档
        # The original path was relative to the agent file.
        # New path is relative to this file's location in `src/huanmu_agent/rag/`
        current_dir = os.path.dirname(os.path.abspath(__file__))
        product_dir = os.path.abspath(
            os.path.join(current_dir, "..", "content_generation", "product_file")
        )

        print(f"正在从目录加载文档: {product_dir}")

        if not os.path.exists(product_dir):
            raise ValueError(f"文档目录不存在: {product_dir}")

        files = os.listdir(product_dir)
        print(f"目录中的文件: {files}")

        docs = []

        pdf_path = os.path.join(product_dir, "美容产品_combined.pdf")
        if os.path.exists(pdf_path):
            from langchain_community.document_loaders import PyPDFLoader

            pdf_loader = PyPDFLoader(pdf_path)
            pdf_docs = pdf_loader.load()
            print(f"从PDF加载了 {len(pdf_docs)} 页")
            # 只处理前2页作为测试
            pdf_docs = pdf_docs[:2]
            print(f"选取前 {len(pdf_docs)} 页进行测试")
            docs.extend(pdf_docs)

        print(f"总共加载了 {len(docs)} 个文档")

        # 2. 分割文档
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2500,  # 进一步增加块大小
            chunk_overlap=200,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?"],
            length_function=len,
            is_separator_regex=False,
        )
        splits = text_splitter.split_documents(docs)
        print(f"文档分割完成，共 {len(splits)} 个片段")

        # 打印几个片段示例
        print("\n片段示例:")
        for i, split in enumerate(splits[:2]):  # 只显示2个示例
            print(f"\n片段 {i + 1}:")
            print(f"内容长度: {len(split.page_content)}")
            print(f"内容预览: {split.page_content[:100]}")  # 减少预览长度
            print(f"元数据: {split.metadata}")

        # 3. 为每个文档添加元数据
        for doc in splits:
            doc.metadata["source"] = "local"
            doc.metadata["file_path"] = doc.metadata.get("source", "")
            doc.metadata["chunk_size"] = len(doc.page_content)

        # 4. 创建向量存储
        embeddings = HuggingFaceEmbeddings(
            model_name="nomic-ai/nomic-embed-text-v1.5",
            model_kwargs={"trust_remote_code": True},
            encode_kwargs={"normalize_embeddings": True},
        )

        # 批量处理
        batch_size = 8  # 使用更大的批次大小
        all_splits = None

        for i in range(0, len(splits), batch_size):
            batch = splits[i : i + batch_size]
            print(f"\n处理批次 {i//batch_size + 1}/{(len(splits)-1)//batch_size + 1}")

            try:
                vectorstore_batch = FAISS.from_documents(
                    batch, embeddings, distance_strategy="COSINE"
                )

                if all_splits is None:
                    all_splits = vectorstore_batch
                else:
                    all_splits.merge_from(vectorstore_batch)

                print(f"批次处理完成，当前已处理 {i + len(batch)} 个片段")
            except Exception as e:
                print(f"处理批次时出错: {e}")
                continue

        if all_splits is None:
            raise ValueError("向量存储初始化失败：所有批次处理都失败了")

        _local_vectorstore = all_splits
        print("\n向量存储创建完成")
        return _local_vectorstore

    except Exception as e:
        print(f"初始化本地向量存储失败: {str(e)}")
        print(f"错误类型: {type(e)}")
        import traceback

        print(f"错误堆栈: {traceback.format_exc()}")
        raise  # 重新抛出异常，因为这是初始化过程中的关键错误 