# 朋友圈评论 Agent 使用说明

## 概述
`comment_analysis_graph` 是一个智能朋友圈评论生成器，能够根据输入的文字内容和图片，生成合适的朋友圈评论。

## 输入格式
```python
{
    "context": "朋友圈文字内容",  # 必填
    "urls": ["图片URL1", "图片URL2"]  # 可选，支持多张图片
}
```

## 输出行为说明

### 🔴 返回空字符串 `""` 的情况

Agent 在以下情况下会选择"沉默"，返回空字符串：

#### 1. 输入为空
```python
# 示例
{"context": "", "urls": []}
{"context": None}
# 返回：""
```

#### 2. 敏感内容 - 涉政/宗教/暴力/歧视
```python
# 示例
{"context": "政府真是太腐败了"}
{"context": "某某宗教都是骗子"}
{"context": "我要杀了他"}
{"context": "黑人都很懒"}
# 返回：""
```

#### 3. 负面情绪爆发 - 辱骂/发泄
```python
# 示例
{"context": "去死吧"}
{"context": "我恨这个世界"}
{"context": "都是垃圾"}
{"context": "xxx是傻逼"}
# 返回：""
```

#### 4. 商业广告
```python
# 示例
{"context": "微商代理，月入过万，点击链接了解详情"}
{"context": "买房投资，稳赚不赔"}
# 返回：""
```

#### 5. 过于严肃/庄重的内容
```python
# 示例
{"context": "今天参加了XXX同志的追悼会"}
{"context": "深切缅怀革命先烈"}
# 返回：""
```

#### 6. 内容不清晰
```python
# 示例
{"context": "asdfgh"}
{"context": "。。。。。。"}
{"context": "emmmmm"}
# 返回：""
```

### ✅ 正常输出评论的情况

Agent 会为以下类型的内容生成温暖、贴心的评论：

#### 1. 日常生活分享
```python
# 输入示例
{"context": "今天天气真好，出门散步了"}
# 可能输出："心情都变好了" / "享受美好时光"
```

#### 2. 美食分享
```python
# 输入示例
{"context": "今天做了红烧肉"}
# 可能输出："看起来好香啊" / "手艺真棒"
```

#### 3. 开心喜悦
```python
# 输入示例
{"context": "升职加薪啦！"}
# 可能输出："恭喜恭喜" / "为你开心"
```

#### 4. 困难求助
```python
# 输入示例
{"context": "最近压力好大"}
# 可能输出："抱抱，会好起来的" / "需要帮忙随时找我"
```

#### 5. 成就展示
```python
# 输入示例
{"context": "终于拿到驾照了"}
# 可能输出："太棒了" / "恭喜你"
```

#### 6. 风景照片
```python
# 输入示例
{"context": "海边日落", "urls": ["sunset.jpg"]}
# 可能输出："好美的地方" / "太漂亮了"
```

## 评论特点

- **字数控制**：3-25字，简洁有温度
- **情感导向**：积极正面，维护人际关系
- **个性化**：可配置Agent的姓名、性别、性格
- **多媒体支持**：能理解图片内容并结合文字生成评论

## 配置参数

```python
config = {
    "configurable": {
        "agent_name": "小七",      # Agent姓名
        "agent_gender": "女",      # 性别
        "agent_personality": "热情" # 性格特点
    }
}
```

## 注意事项

1. **输入安全**：Agent 会主动识别并拒绝处理不当内容
2. **隐私保护**：不会对过于私人或敏感的内容发表评论
3. **情感智能**：能识别情绪状态，选择合适的回应方式
4. **社交礼仪**：遵循社交媒体的基本礼貌和边界感

## 使用示例

```python
# 正常使用
result = comment_analysis_graph.invoke(
    {"context": "今天心情特别好"}, 
    config={"configurable": {"agent_name": "小美", "agent_personality": "温柔"}}
)
# 输出：{"structured_response": "看到你开心我也很开心"}

# 敏感内容
result = comment_analysis_graph.invoke(
    {"context": "政治相关敏感内容"}, 
    config=config
)
# 输出：{"structured_response": ""}
``` 