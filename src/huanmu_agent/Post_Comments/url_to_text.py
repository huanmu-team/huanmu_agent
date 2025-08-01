from langchain_core.messages import HumanMessage
from typing import List
import asyncio
import base64
import requests
import re
from constant import GOOGLE_GEMINI_FLASH_MODEL

# 图片处理函数
async def process_images_to_descriptions(urls: List[str], llm) -> List[str]:
    """
    处理多个图片URL，返回图片描述列表

    Args:
        urls: 图片URL列表
        llm: 语言模型实例

    Returns:
        图片描述列表，如果处理失败则返回错误信息
    """
    if not urls:
        return []

    descriptions = []

    for url in urls:
        print(f"[DEBUG] 处理图片URL: {url}")
        # 检查是否为图片格式
        if not any(ext in url.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']):
            print(f"[DEBUG] URL不是图片格式: {url}")
            descriptions.append(f"URL不是图片格式: {url}")
            continue

        try:
            # 下载图片 - 使用线程池避免阻塞事件循环
            print(f"[DEBUG] 开始下载图片: {url}")
            
            def download_image(url):
                """同步下载图片的函数，将在线程池中运行"""
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                return response
            
            # 在线程池中执行同步的网络请求
            response = await asyncio.to_thread(download_image, url)
            print(f"[DEBUG] 图片下载成功，状态码: {response.status_code}")

            # 转换为base64
            image_data = base64.b64encode(response.content).decode('utf-8')
            print(f"[DEBUG] 图片转换为base64，长度: {len(image_data)}")

            # 使用视觉模型分析图片
            print("[DEBUG] 开始调用视觉模型分析图片...")
            vision_message = HumanMessage(
                content=[
                    {"type": "text", "text": "请一句话简洁描述这张图片的内容："},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                    }
                ]
            )

            # 调用视觉模型 - 统一使用同步方法并在线程池中执行
            try:
                vision_result = await asyncio.to_thread(llm.invoke, [vision_message])
                print(f"[DEBUG] 视觉分析结果: {vision_result.content}")
                descriptions.append(vision_result.content)
            except Exception as vision_error:
                print(f"[ERROR] 视觉模型调用失败: {vision_error}")
                descriptions.append(f"图片分析失败: {str(vision_error)}")

        except Exception as e:
            print(f"[DEBUG] 图片处理异常: {str(e)}")
            descriptions.append(f"图片处理失败: {str(e)}")

    return descriptions