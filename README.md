# HuanMu Agent

A custom Reasoning and Action agent built using LangGraph, designed for sales, user profile management and WeChat integration.

## Project Structure

```
.
├── .gitignore
├── constant.py
├── Makefile
├── pyproject.toml
├── README.md
└── src/
    ├── huanmu_agent/
    │   ├── __init__.py
    │   ├── configuration.py - Core agent configuration
    │   ├── graph.py - Main agent workflow definitions
    │   ├── prompts.py - System and user prompts
    │   ├── state.py - Agent state management
    │   ├── tools.py - Core agent tools
    │   ├── utils.py - Utility functions
    │   ├── sales/
    │   │   ├── sale_advice_agent.py - Sales recommendation agent
    │   │   └── tools.py - Sales-specific tools
    │   ├── user_profile/
    │   │   ├── profile_agent.py - User profile management agent
    │   │   └── profile_variables.py - Profile variables definitions
    │   └── wechat/
    │       ├── configuration.py - WeChat integration settings
    │       └── moment_agent.py - WeChat Moments posting agent
```

## Features

### Core Capabilities
- Flexible agent workflows using LangGraph
- State management for conversation tracking
- Custom tool integration system

### Sales Module
- Sales advice generation
- Product recommendation tools
- Customer interaction analysis

### User Profile Module
- User preference tracking
- Profile variable management
- Personalized content generation

### WeChat Module
- WeChat Moments posting automation
- Social media integration
- Content scheduling

### Content Generation Module
- Product refinement suggestions
- Marketing poster copy generation
- Visual poster design creation

## Getting Started

### Prerequisites
- Python 3.10+
- Poetry (recommended)

### Installation
```bash
pip install -e ".[dev]"
```

### Configuration
1. Copy environment file:
```bash
cp .env.example .env
```

2. Set required API keys:
```
OPENAI_API_KEY=your-key-here
ANTHROPIC_API_KEY=your-key-here
```

## Usage Examples

### Running the Sales Agent
```python
from src.huanmu_agent.sales.sale_advice_agent import SalesAgent

agent = SalesAgent()
response = agent.run("What products should I recommend to this customer?")
print(response)
```

### Using Content Generation
```python
from src.huanmu_agent.content_generation import (
    ProductRefinementAgent,
    PosterCopyAgent,
    PosterDesignAgent
)
from src.huanmu_agent.configuration import Configuration

config = Configuration()
refinement_agent = ProductRefinementAgent(config)
copy_agent = PosterCopyAgent(config)
design_agent = PosterDesignAgent(config)

# Generate product suggestions
suggestions = await refinement_agent.run(
    product_info={...},
    user_profile={...},
    market_feedback={...}
)

# Create marketing copy
copy = await copy_agent.generate_copy(
    occasion="Summer Sale",
    product_info={...},
    user_profile={...},
    style="fun"
)

# Generate poster design
design_url = await design_agent.generate_design(
    copy_keywords=copy,
    style_description="Bright summer colors"
)
```

### Posting to WeChat Moments
```python
from src.huanmu_agent.wechat.moment_agent import WeChatMomentAgent

agent = WeChatMomentAgent()
agent.post_moment("Check out our new product line!")
```

## Development

### Testing
```bash
make test
```

### Linting
```bash
make lint
```

### Formatting
```bash
make format
```

## Configuration Options

Key configuration files:
- `src/huanmu_agent/configuration.py` - Core agent settings
- `.env` - Environment variables
- `src/huanmu_agent/wechat/configuration.py` - WeChat integration settings

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

Please ensure all tests pass and code is properly formatted before submitting.
