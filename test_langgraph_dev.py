"""测试LangGraph dev服务器的脚本"""

import requests
import json

# LangGraph dev服务器地址
BASE_URL = "http://127.0.0.1:2024"

def test_content_generation_workflow():
    """测试完整的内容生成工作流"""
    print("🚀 测试完整的三代理内容生成工作流...")
    
    url = f"{BASE_URL}/runs/stream"
    payload = {
        "assistant_id": "content_generation_workflow",
        "input": {
            "product_id": "smart_watch_001",
            "user_segment": "young_professionals", 
            "style": "professional",
            "messages": []
        }
    }
    
    try:
        response = requests.post(url, json=payload, stream=True)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ 工作流启动成功!")
            print("📄 响应内容:")
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        print(json.dumps(data, indent=2, ensure_ascii=False))
                    except:
                        print(line.decode('utf-8'))
        else:
            print(f"❌ 请求失败: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到LangGraph dev服务器，请确保langgraph dev正在运行")
    except Exception as e:
        print(f"❌ 错误: {e}")

def test_data_collection_agent():
    """测试数据收集代理"""
    print("\n🔍 测试数据收集代理...")
    
    url = f"{BASE_URL}/runs/stream"
    payload = {
        "assistant_id": "data_collection_agent",
        "input": {
            "product_id": "smart_watch_001",
            "user_segment": "young_professionals",
            "messages": []
        }
    }
    
    try:
        response = requests.post(url, json=payload)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ 数据收集代理响应成功!")
            try:
                result = response.json()
                print("📊 收集的数据:")
                print(json.dumps(result, indent=2, ensure_ascii=False))
            except:
                print("📄 响应:", response.text)
        else:
            print(f"❌ 请求失败: {response.text}")
            
    except Exception as e:
        print(f"❌ 错误: {e}")

def test_product_optimization_agent():
    """测试产品优化代理"""
    print("\n🔧 测试产品优化代理...")
    
    # 首先需要一些基础数据
    mock_data = {
        "holiday_info": {"holidays": ["双十一"], "season": "秋季"},
        "user_profile": {"demographics": "25-35岁职场人士"},
        "product_info": {"name": "智能手表Pro", "features": ["健康监测", "长续航"]},
        "market_feedback": {"trending_topics": ["智能健康", "便携设计"]},
        "messages": []
    }
    
    url = f"{BASE_URL}/runs/stream"
    payload = {
        "assistant_id": "product_optimization_agent",
        "input": mock_data
    }
    
    try:
        response = requests.post(url, json=payload)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ 产品优化代理响应成功!")
            try:
                result = response.json()
                print("💡 优化建议:")
                print(json.dumps(result, indent=2, ensure_ascii=False))
            except:
                print("📄 响应:", response.text)
        else:
            print(f"❌ 请求失败: {response.text}")
            
    except Exception as e:
        print(f"❌ 错误: {e}")

def test_content_creation_agent():
    """测试内容创作代理"""
    print("\n🎨 测试内容创作代理...")
    
    # 模拟前两步的数据
    mock_data = {
        "holiday_info": {"holidays": ["双十一"], "season": "秋季"},
        "user_profile": {"demographics": "25-35岁职场人士"},
        "product_info": {"name": "智能手表Pro", "features": ["健康监测", "长续航"]},
        "market_feedback": {"trending_topics": ["智能健康", "便携设计"]},
        "product_optimization": {
            "suggestions": [{"description": "突出健康监测功能", "priority": "high"}],
            "summary": "强化健康主题营销"
        },
        "messages": []
    }
    
    url = f"{BASE_URL}/runs/stream"
    payload = {
        "assistant_id": "content_creation_agent",
        "input": mock_data
    }
    
    try:
        response = requests.post(url, json=payload)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ 内容创作代理响应成功!")
            try:
                result = response.json()
                print("✍️ 生成的内容:")
                print(json.dumps(result, indent=2, ensure_ascii=False))
            except:
                print("📄 响应:", response.text)
        else:
            print(f"❌ 请求失败: {response.text}")
            
    except Exception as e:
        print(f"❌ 错误: {e}")

def list_available_graphs():
    """列出可用的graphs"""
    print("\n📋 列出所有可用的graphs...")
    
    url = f"{BASE_URL}/assistants"
    
    try:
        response = requests.get(url)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            assistants = response.json()
            print("✅ 可用的graphs:")
            for assistant in assistants:
                print(f"  - {assistant.get('assistant_id', 'unknown')}")
        else:
            print(f"❌ 请求失败: {response.text}")
            
    except Exception as e:
        print(f"❌ 错误: {e}")

if __name__ == "__main__":
    print("🧪 LangGraph Dev 测试工具")
    print("=" * 50)
    
    # 首先列出可用的graphs
    list_available_graphs()
    
    # 测试各个代理
    test_data_collection_agent()
    test_product_optimization_agent() 
    test_content_creation_agent()
    
    # 最后测试完整工作流
    test_content_generation_workflow()
    
    print("\n🎉 测试完成!") 