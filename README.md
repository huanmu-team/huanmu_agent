# HuanMu Agent

A custom Reasoning and Action agent built using LangGraph, designed for sales and user profile management.

## Project Structure

```
.
├── .gitignore
├── constant.py
├── langgraph.json
├── Makefile
├── novel_danis_vertexai_credential.json
├── pyproject.toml
├── README.md
└── src/
    ├── huanmu_agent/
    │   ├── __init__.py
    │   ├── configuration.py
    │   ├── graph.py
    │   ├── prompts.py
    │   ├── state.py
    │   ├── tools.py
    │   ├── utils.py
    │   ├── sales/
    │   │   ├── sale_advice_agent.py
    │   │   └── tools.py
    │   ├── user_profile/
    │   │   └── profile_agent.py
    │   └── wechat/
    │       ├── configuration.py
    │       └── moment_agent.py
```

## Features

- Sales advice agent with custom tools
- User profile management capabilities
- Built on LangGraph for flexible agent workflows

## Getting Started

1. Install dependencies:
```bash
pip install -e ".[dev]"
```

2. Create a `.env` file:
```bash
cp .env.example .env
```

3. Configure API keys in `.env`:
```
OPENAI_API_KEY=your-key
ANTHROPIC_API_KEY=your-key
```

## Development

Run tests:
```bash
make test
```

Run linters:
```bash
make lint
```

Format code:
```bash
make format
```

## Configuration

The agent can be configured via:
- `src/huanmu_agent/configuration.py` - Core settings
- `.env` file - API keys and environment variables

## Customization

1. Add new tools in `src/huanmu_agent/tools.py` or module-specific tools files
2. Modify agent logic in `src/huanmu_agent/graph.py`
3. Update prompts in `src/huanmu_agent/prompts.py`
