import re


def extract_xml(text: str, tag: str) -> str:
    """
    从给定的文本中提取指定XML标签的内容。
    这个函数是解析大语言模型返回的结构化响应的关键工具。

    工作原理:
    - 使用正则表达式 `re.search` 来查找模式 `<tag>(.*?)</tag>`。
    - `re.DOTALL` 标志允许 `.` 匹配包括换行符在内的任意字符，
      这对于提取可能包含多行内容的XML标签至关重要。
    - 如果找到匹配项，`match.group(1)`会返回第一个捕获组的内容，
      也就是开始和结束标签之间的所有文本。

    Args:
        text (str): 包含XML的文本。
        tag (str): 要提取内容的XML标签名。

    Returns:
        str: 指定XML标签的内容，如果未找到标签则返回空字符串。
    """
    # 使用正则表达式搜索指定标签对之间的内容
    match = re.search(f'<{tag}>(.*?)</{tag}>', text, re.DOTALL)
    # 如果找到匹配项，返回捕获的内容，否则返回空字符串
    return match.group(1) if match else ""


