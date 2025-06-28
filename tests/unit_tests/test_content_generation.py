"""Tests for content generation module."""

import pytest
from huanmu_agent.configuration import Configuration
from huanmu_agent.content_generation import (
    ProductRefinementAgent,
    PosterCopyAgent,
    PosterDesignAgent
)

@pytest.fixture
def config():
    return Configuration()

@pytest.fixture
def sample_product():
    return {
        "name": "抗皱精华液",
        "key_features": ["深层滋养", "抗衰老", "天然成分"],
        "unique_selling_points": ["专利配方", "临床验证"]
    }

@pytest.fixture
def sample_user_profile():
    return {
        "demographics": "25-35岁女性，都市白领",
        "preferences": ["天然成分", "便捷使用"]
    }

@pytest.fixture
def sample_market_feedback():
    return {
        "positive": ["效果好", "包装精美"],
        "negative": ["价格偏高", "吸收稍慢"]
    }

@pytest.mark.asyncio
async def test_product_refinement(config, sample_product, sample_user_profile, sample_market_feedback):
    agent = ProductRefinementAgent(config)
    result = await agent.run(
        product_info=sample_product,
        user_profile=sample_user_profile,
        market_feedback=sample_market_feedback
    )
    assert isinstance(result, str)
    assert len(result) > 0
    print(f"\nProduct refinement suggestions:\n{result}")

@pytest.mark.asyncio
async def test_poster_copy(config, sample_product, sample_user_profile):
    agent = PosterCopyAgent(config)
    result = await agent.generate_copy(
        occasion="双十一促销",
        product_info=sample_product,
        user_profile=sample_user_profile,
        style="fun"
    )
    assert isinstance(result, str)
    assert len(result) > 0
    print(f"\nPoster copy:\n{result}")

@pytest.mark.asyncio
async def test_poster_design(config):
    agent = PosterDesignAgent(config)
    result = await agent.generate_design(
        copy_keywords="双十一特惠 抗皱精华液 限时折扣",
        style_description="节日促销风格，红色主题"
    )
    assert isinstance(result, str)
    assert result.startswith(("http", "data:image"))
    print(f"\nPoster design URL:\n{result}")
