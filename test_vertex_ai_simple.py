"""测试Vertex AI Gemini模型调用"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_vertexai import ChatVertexAI

def test_vertex_ai():
    """测试Vertex AI Gemini模型调用"""
    try:
        print("🧪 测试Vertex AI Gemini模型...")
        
        # 初始化模型
        model = ChatVertexAI(
            model="gemini-2.5-flash-preview-05-20",
            temperature=0.7,
        )
        
        print("✅ 模型初始化成功")
        
        # 创建消息
        messages = [
            SystemMessage(content="你是一个营销内容创作专家，帮助用户生成高质量的营销文案。"),
            HumanMessage(content="为美妆产品水光针创作一个简短的营销标题。")
        ]
        
        print("📤 发送消息...")
        
        # 调用模型
        response = model.invoke(messages)
        
        print("✅ 调用成功！")
        print(f"📝 响应: {response.content}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_vertex_ai() 