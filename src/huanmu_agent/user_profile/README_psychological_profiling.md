# 心理侧写智能体 (Psychological Profiling Agent)

## 概述

心理侧写智能体是一个基于心理学理论的用户分析工具，能够通过对话内容深度分析用户的心理特征，为高情商对话提供专业支撑。

## 主要功能

### 🧠 深度心理分析
- **认知风格分析**: 理性vs感性、系统性vs直觉性思维模式
- **情感模式识别**: 情绪表达方式、情感稳定性、情感需求
- **行为倾向预测**: 决策风格、风险偏好、行动模式
- **心理防御机制**: 应对压力和冲突的心理策略

### 💬 沟通洞察
- **沟通偏好分析**: 信息处理方式、交流节奏、表达风格
- **依恋风格识别**: 基于依恋理论的人际关系模式
- **情商特征评估**: 自我认知、社交技能、情感管理能力

### 🎯 高情商对话支撑
- **个性化互动建议**: 基于心理特征的沟通策略
- **情感需求识别**: 核心心理需求和成长机会
- **脆弱性感知**: 潜在敏感点和应避免的话题

## 分析维度

### 核心心理维度 (15个)

1. **认知风格** - 思维处理方式和决策模式
2. **情感模式** - 情绪表达和情感调节特征  
3. **行为倾向** - 行动风格和习惯模式
4. **心理防御机制** - 应对压力的心理策略
5. **沟通偏好** - 交流方式和信息处理偏好
6. **依恋风格** - 人际关系和信任模式
7. **压力应对** - 面对挑战时的反应模式
8. **内在动机** - 驱动行为的深层动力
9. **自我概念** - 自我认知和价值观
10. **人际模式** - 社交互动和关系建立方式
11. **情商特征** - 情感智力的各个方面
12. **心理需求** - 核心心理满足需求
13. **潜在脆弱性** - 心理敏感点和风险因素
14. **成长机会** - 个人发展的潜在方向
15. **建议互动方式** - 具体的沟通和互动策略

## 使用方法

### 基础用法

```python
from psychological_profiling_agent import psychological_profiling_graph
from langchain_core.messages import HumanMessage, AIMessage

# 准备对话历史
messages = [
    HumanMessage(content="用户的对话内容..."),
    AIMessage(content="助手的回复..."),
    # ... 更多对话
]

# 调用心理侧写分析
result = await psychological_profiling_graph.ainvoke({
    "messages": messages
})

# 获取分析结果
profile = result["structured_response"]
print(f"认知风格: {profile.cognitive_style}")
print(f"沟通偏好: {profile.communication_preferences}")
```

### 高级应用

```python
from psychological_profiling_agent import (
    format_psychological_profile,
    get_key_psychological_insights
)

# 格式化完整报告
formatted_report = format_psychological_profile(profile)
print(formatted_report)

# 提取关键洞察
key_insights = get_key_psychological_insights(profile)
for insight_type, content in key_insights.items():
    print(f"{insight_type}: {content}")
```

## 应用场景

### 🏥 医美咨询场景
- **谨慎型用户**: 强调安全性、提供详细信息、给予充足考虑时间
- **冲动型用户**: 突出效果、简化流程、快速响应需求
- **理性型用户**: 提供数据支撑、详细对比、性价比分析

### 💼 销售对话优化
- **情感导向**: 强调体验和感受
- **逻辑导向**: 提供数据和证据
- **关系导向**: 建立信任和连接

### 🤝 客户服务提升
- **识别情绪状态**: 及时调整服务策略
- **预判需求**: 主动提供个性化服务
- **冲突预防**: 避免触及敏感话题

## 心理学理论基础

### 核心理论框架
- **大五人格理论**: 开放性、责任心、外向性、宜人性、神经质
- **依恋理论**: 安全型、焦虑型、回避型、混乱型依恋
- **认知行为理论**: 思维模式与行为的关联
- **情绪智力理论**: 自我认知、自我管理、社会认知、关系管理

### 分析方法
- **内容分析**: 词汇选择、表达方式、主题偏好
- **语言模式识别**: 句式结构、情感色彩、逻辑关系
- **互动风格观察**: 回应速度、信息量、主动性
- **价值观推导**: 决策标准、优先级、原则体现

## 技术架构

### 核心组件
```
psychological_profiling_agent.py
├── PsychologicalProfilingStructure     # 心理侧写数据结构
├── psychological_profiling_agent       # 核心分析智能体
├── psychological_profiling_graph       # LangGraph工作流
└── 辅助函数
    ├── format_psychological_profile()  # 格式化输出
    └── get_key_psychological_insights() # 关键洞察提取
```

## 集成到LangGraph

该智能体已集成到项目的 `langgraph.json` 配置中：

```json
{
  "graphs": {
    "psychological_profiling_agent": "./src/huanmu_agent/user_profile/psychological_profiling_agent.py:psychological_profiling_graph"
  }
}
```

可通过 `langgraph dev` 运行并访问该智能体。

## 注意事项

### ⚠️ 使用限制
1. **非诊断工具**: 仅用于沟通优化，不可作为心理健康诊断
2. **样本依赖**: 分析质量取决于对话内容的丰富程度
3. **文化适配**: 主要基于中文语境和文化背景
4. **隐私保护**: 确保用户对话内容的安全性

### 🔒 伦理考量
- 尊重用户隐私和个人边界
- 避免标签化和刻板印象
- 关注积极心理特质
- 提供建设性而非评判性分析

## 开发扩展

### 自定义分析维度
可以通过修改 `PsychologicalProfilingStructure` 来添加新的分析维度：

```python
class CustomPsychologicalProfile(PsychologicalProfilingStructure):
    cultural_background: str = Field(description="文化背景分析", default="")
    professional_traits: str = Field(description="职业特征", default="")
```

### 集成其他智能体
心理侧写结果可以作为其他智能体的输入：

```python
# 在其他智能体中使用心理侧写结果
psychological_insights = get_key_psychological_insights(profile)
enhanced_prompt = f"基于用户心理特征: {psychological_insights}, 请..."
```

## 版本信息

- **版本**: 1.0.0
- **更新日期**: 2024年
- **兼容性**: Python 3.8+, LangGraph, LangChain
- **模型支持**: Google Gemini Pro 