import os

def load_prompt(name: str) -> str:
    """
    从 prompts/ 目录加载一个prompt文本文件。硬文末拼接，而不是自然融合，与自定义agent仍有差距
    对应需求中的人设调整部分。

    Args:
        name (str): prompt的名称，对应文件名（不含扩展名）。
                    例如, name='state_evaluator' 将加载 'state_evaluator.txt'。

    Returns:
        str: prompt文件的内容。
        
    Raises:
        FileNotFoundError: 如果对应的prompt文件不存在。
    """
    # 获取当前文件所在的目录
    prompts_dir = os.path.dirname(__file__)
    # 构建prompt文件的完整路径
    prompt_path = os.path.join(prompts_dir, f"{name}.txt")
    
    # 读取具体 prompt 内容
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt_body = f.read()
    except FileNotFoundError:
        print(f"Error: Prompt file not found at {prompt_path}")
        raise

    # 如果存在全局上下文，自动拼接
    base_path = os.path.join(prompts_dir, "base_context.txt")
    if os.path.exists(base_path):
        try:
            with open(base_path, 'r', encoding='utf-8') as f:
                base_context = f.read().strip()
        except Exception:
            base_context = ""
        if base_context:
            # 用两个换行分开，避免格式混乱
            return f"{base_context}\n\n{prompt_body}"

    return prompt_body 