"""医美行业用户画像变量定义
包含所有预定义的标签选项，用于用户画像生成
必须与profile_agent.py中的PROFILE_SYSTEM_PROMPT结构保持一致
"""

profile_variables = {
    "social_profile": {
        "occupation": [
            "医生", "护士", "教师", "公务员", "程序员", "设计师", "产品经理", "律师", "法务", "金融从业者",
            "销售", "市场营销", "运营", "人力资源", "行政", "自由职业者", "企业主", "个体户",
            "全职妈妈", "全职爸爸", "学生", "研究生", "退休人员", "艺术工作者", "主播", "博主", "演员", "空乘", "客服",
            "建筑师", "工程师", "物流人员", "电商从业者", "美容师", "健身教练", "医药代表", "宠物行业从业者",
            "保险顾问", "房地产中介", "司机", "服务员", "厨师", "技工"
        ],
        "age": ["18-25岁", "26-35岁", "36-45岁", "46-55岁", "56岁以上"],
        "region": ["一线城市", "新一线城市", "二线城市", "三线城市", "四线及以下城市"],
        "lifestyle": ["健身习惯", "瑜伽练习", "熬夜", "素食", "低碳饮食", "吸烟", "饮酒", "规律作息"],
        "family_status": ["已婚", "未婚", "离异", "有子女", "无子女", "与父母同住", "独居"],
        "emotion": ["积极", "消极", "中立", "犹豫", "信任", "怀疑", "焦虑", "期待", "满意", "不满意"]
    },
    "personality_traits": {
        "character": ["果断型", "谨慎型", "冲动型", "犹豫型", "理性型", "感性型", "外向型", "内向型"],
        "values": ["健康意识强", "环保主义", "效率优先", "品质至上", "性价比优先", "时尚潮流", "个性化"],
        "aesthetic_style": ["自然风格", "精致妆容", "时尚前卫", "简约大方", "华丽夸张", "甜美可爱", "中性风格"]
    },
    "consumption_profile": {
        "ability": ["高消费能力", "中高消费能力", "中等消费", "中低消费", "预算有限"],
        "willingness": ["价格敏感", "品质优先", "冲动消费", "理性消费", "品牌忠诚", "尝鲜型"],
        "preferences": ["国际奢侈品牌", "国际大众品牌", "国产高端品牌", "国产平价品牌", "小众设计师品牌", "医美机构自有品牌"]
    },
    "product_intent": {
        "current_use": [
            "玻尿酸", "肉毒素", "胶原蛋白", "少女针", "童颜针", "溶脂针",
            "热玛吉", "超声刀", "热拉提", "Fotona 4D", "黄金微针", "光子嫩肤",
            "皮秒", "超皮秒", "点阵激光", "黑金DPL", "激光脱毛",
            "双眼皮手术", "开眼角", "眼袋去除", "隆鼻", "鼻综合", "隆胸",
            "脂肪填充", "面部吸脂", "身体吸脂", "手臂吸脂", "大腿吸脂",
            "腹部吸脂", "腰腹环吸", "私密整形", "处女膜修复", "阴道紧缩",
            "水光针", "微针", "果酸换肤", "小气泡", "黑脸娃娃", "白瓷娃娃",
            "线雕", "超声炮", "冷冻溶脂"
        ],
        "potential_needs": [
            "瘦脸需求", "V脸塑造", "面部轮廓", "面部年轻化", "抗衰老",
            "皮肤美白", "祛斑", "祛痘", "祛痘印", "祛红血丝", "收缩毛孔",
            "改善肤质", "皮肤紧致", "法令纹填充", "泪沟填充", "苹果肌填充",
            "双眼皮", "开眼角", "祛眼袋", "祛黑眼圈", "眼部年轻化",
            "隆鼻", "鼻头缩小", "鼻翼缩小", "鼻综合", "驼峰鼻矫正",
            "丰唇", "唇形调整", "唇色改善", "M唇塑造",
            "抽脂", "身体塑形", "丰胸", "乳房提升", "乳晕缩小",
            "瘦腿", "瘦手臂", "马甲线塑造", "翘臀塑造",
            "毛发移植", "植发", "牙齿美白", "牙齿矫正", "疤痕修复",
            "妊娠纹修复", "私密整形"
        ],
        "decision_factors": [
            "医生推荐", "朋友口碑", "价格因素", "效果持久", 
            "安全性", "恢复期短", "机构知名度", "医生资质",
            "案例效果", "术后服务", "隐私保护", "分期付款"
        ],
        "purchase_intent_score": ["0-10分对购买意向评分"]
    },
    "customer_lifecycle": {
        "stage": ["潜在客户期", "兴趣客户期", "意向客户期", "决策期", "成交客户期", "复购期", "忠诚客户期", "休眠期", "流失客户期"],
        "value": ["超高价值客户", "高价值客户", "中价值客户", "普通客户", "低频客户", "一次性客户"],
        "retention_strategy": ["客户关怀", "促销活动", "专属服务", "会员权益", "个性化推荐", "定期回访"]
    }
}

# 数据完整性验证
required_sections = ["social_profile", "personality_traits", "consumption_profile",
                    "product_intent", "customer_lifecycle"]
for section in required_sections:
    if section not in profile_variables:
        raise ValueError(f"Missing required section: {section}")
    
if __name__ == "__main__":
    print("Profile variables loaded successfully")
    print(f"Total categories: {len(profile_variables)}")
    for category, options in profile_variables.items():
        print(f"{category}: {len(options)} sub-categories")