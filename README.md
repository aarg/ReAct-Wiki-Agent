# Multi-Tool AI Agent

A ReAct (Reasoning and Acting) pattern AI agent that orchestrates multiple tools to answer questions. Built with LangGraph for workflow orchestration and Claude for reasoning. Features calculator, datetime, date calculations, and web search capabilities.

## Key Features

- **ReAct Pattern**: Alternates between reasoning and tool execution for transparent decision-making
- **Multiple Tools**: Calculator, DateTime, DateCalculator, and Web Search
- **Observability**: LangSmith integration for tracing and debugging
- **Explainable AI**: Every decision is logged with reasoning
- **Loop Prevention**: Automatic detection and prevention of infinite loops
- **Clean Architecture**: Modular design with tool registry pattern

## Quick Start

1. **Setup Environment**
```bash
# Clone the repository
cd /Users/rg/Code/llm/wiki

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

2. **Configure API Keys**
Create a `.env` file with your API keys:
```
# Required
ANTHROPIC_API_KEY=your_api_key_here

# Optional for web search
GOOGLE_API_KEY=your_google_api_key
GOOGLE_SEARCH_ENGINE_ID=your_search_engine_id

# Optional for observability
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=wiki-agent
```

3. **Run the Agent**
```bash
# Ask a question
python wiki.py "What is 25 * 4 + 10?"
python wiki.py "What day will it be in 7 days?"
python wiki.py "How many days until Thanksgiving?"
python wiki.py "Who is the current president of France?"

# With verbose output
python wiki.py -v "What is machine learning?"
```

## Architecture

The agent implements the ReAct (Reasoning and Acting) pattern:

```
┌─────────────┐
│   Question  │
└──────┬──────┘
       │
       ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Reasoning  │────▶│ Tool Execute │────▶│  Reasoning  │
│    Node     │     │     Node     │     │    Node     │
└─────────────┘     └──────────────┘     └──────┬──────┘
       ▲                                         │
       │                                         ▼
       └─────────────────────────────────┬──────────────┐
                                         │ Final Answer │
                                         └──────────────┘
```

### Available Tools

1. **Calculator** (`calculator`)
   - Performs mathematical calculations
   - Supports basic arithmetic and math functions
   - Example: "What is 25 * 4 + 10?"

2. **DateTime** (`datetime`)
   - Returns current date, time, and day of week
   - Supports custom formatting
   - Example: "What day is today?"

3. **DateCalculator** (`datecalculator`)
   - Calculates future/past dates
   - Computes days between dates
   - Example: "What day will it be in 7 days?" or "How many days until Christmas?"

4. **Web Search** (`web_search`)
   - Searches the web for current information
   - Uses Google Custom Search API
   - Example: "Who is the current president of France?"

## How It Works

1. **User asks a question** → Agent receives the query
2. **Reasoning phase** → Claude analyzes the question and available tools
3. **Tool selection** → Agent selects appropriate tool based on the question
4. **Tool execution** → Selected tool runs and returns results
5. **Observation** → Agent analyzes tool output
6. **Decision** → Either call another tool or provide final answer
7. **Final answer** → Synthesized response based on tool observations

## Observability with LangSmith

When LangSmith is configured, you get:
- Visual trace of the entire ReAct flow
- Tool execution timeline
- Token usage and costs
- Error tracking and debugging

See [docs/OBSERVABILITY.md](docs/OBSERVABILITY.md) for setup instructions.

## Project Structure

```
wiki/
├── wiki.py                # Main entry point
├── src/
│   ├── workflow.py        # LangGraph ReAct workflow
│   ├── state.py           # Agent state definition
│   ├── tools.py           # Tool implementations
│   ├── claude_client.py   # Claude LLM integration
│   └── google_search.py   # Web search implementation
├── docs/
│   ├── AI_CONCEPTS.md     # AI/ML concepts documentation
│   ├── AI_CHALLENGES.md   # Challenges and solutions
│   ├── OBSERVABILITY.md   # LangSmith setup guide
│   └── walkthrough.md     # Detailed code walkthrough
└── requirements.txt       # Python dependencies
```

## Examples

### Math Calculation
```bash
$ python wiki.py "What is 25 * 4 + 10?"
🤔 Question: What is 25 * 4 + 10?

🤖 MULTI-TOOL AGENT
🤔 STARTING REASONING

💭 This is a mathematical calculation...
🔧 Tool: calculator
   Input: {'expression': '25 * 4 + 10'}
   Output: Result: 110

✅ Answer: 25 * 4 + 10 = 110
```

### Date Calculation
```bash
$ python wiki.py "How many days until Thanksgiving?"
🤔 Question: How many days until Thanksgiving?

🤖 MULTI-TOOL AGENT
🤔 STARTING REASONING

💭 I need to find when Thanksgiving is...
🔧 Tool: web_search
   Input: {'query': 'Thanksgiving 2024 date'}
   Output: Found information from Wikipedia...

💭 Thanksgiving 2024 is November 28...
🔧 Tool: datecalculator
   Input: {'target_date': 'November 27, 2025'}
   Output: Days until November 27, 2025: 145 days

✅ Answer: There are 145 days until Thanksgiving 2025.
```

## Development

### Testing

```bash
# Run basic test suite
python test_agent.py

# Run enhanced test suite (tests all tools)
python test_agent_enhanced.py

# Test specific tools
python -c "from src.tools import ToolRegistry; print(ToolRegistry().list_tools())"
```

### Code Quality
```bash
# Format code
python -m black src/

# Type checking
python -m mypy src/
```

## Documentation

- **[AI_CONCEPTS.md](docs/AI_CONCEPTS.md)**: ReAct pattern, prompt engineering, tool orchestration
- **[AI_CHALLENGES.md](docs/AI_CHALLENGES.md)**: Challenges encountered and solutions
- **[OBSERVABILITY.md](docs/OBSERVABILITY.md)**: LangSmith integration guide
- **[walkthrough.md](docs/walkthrough.md)**: Detailed code walkthrough

## Future Enhancements

- Additional tools (weather, news, calculations)
- Multi-turn conversations with memory
- Streaming responses
- Tool result caching
- Custom tool creation interface

## License

This project is for educational purposes. Please ensure you comply with API terms of service for Anthropic, Google, and LangSmith.