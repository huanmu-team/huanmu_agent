from huanmu_agent.content_generation.marketing_orchestrator import orchestrator_graph, OrchestratorState
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import json

load_dotenv()

def test_marketing_orchestrator():
    """测试营销编排器代理的完整工作流程"""
    print("\n=== 测试营销编排器 (OpenAI版) ===")
    print("这个测试将验证编排器是否能够按顺序调用所有子代理并生成最终海报：")
    print("1. 产品研究")
    print("2. 小红书研究")
    print("3. 内容生成")
    print("4. 海报生成")
    
    # 初始化状态
    initial_state = {
        "messages": [
            HumanMessage(content="帮我为一款'星辰系列'美白精华液生成营销文案和海报，主打夜间修复和提亮肤色，适合都市熬夜人群。")
        ],
    }
    
    # 运行营销代理
    try:
        print("\n=== 开始执行 ===")
        result = orchestrator_graph.invoke(initial_state)
        
        print("\n=== 执行完毕，最终状态: ===")
        # A more robust way to pretty print, handling non-serializable objects
        final_state_str = json.dumps(result, indent=2, ensure_ascii=False, default=lambda o: f"<non-serializable: {type(o).__name__}>")
        print(final_state_str)
            
        # Explicitly check for the final poster URL
        final_url = result.get("final_poster_url")
        print(f"\n\n=== 最终生成的海报URL ===")
        print(final_url or "未生成海报URL。")
                    
    except Exception as e:
        print(f"\n测试过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_marketing_orchestrator() 