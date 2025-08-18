# Agentic Insurance Data Migration Platform

A modern, LLM-agnostic platform for migrating disability and absence book of business from legacy mainframe systems to modern platforms using Google's A2A framework, LangGraph, and MCP.

## 🚀 Features

- **A2A Framework Integration**: Leverages Google's Agent-to-Agent framework for intelligent agent coordination
- **LLM Agnostic**: Support for OpenAI, AWS Bedrock, Anthropic, and Google Vertex AI
- **LangGraph Workflows**: Complex multi-agent workflows with state management
- **MCP Integration**: Model Context Protocol for standardized API interactions
- **Field Mapping**: Dynamic field mapping from configuration files
- **Comprehensive Validation**: LLM-powered data validation and business rule checking
- **Rich Logging**: LangSmith integration for LLM call tracking and debugging

## 🏗️ Architecture

The platform uses a multi-agent architecture with the following components:

### Core Agents
1. **File Reader Agent** - Reads and parses mainframe files
2. **Validation Agent** - Validates data integrity and business rules
3. **Mapping Agent** - Handles field transformations and mappings
4. **Transformation Agent** - Converts data to target format
5. **API Integration Agent** - Manages MCP-based API calls
6. **Orchestration Agent** - Coordinates the entire workflow

### Key Technologies
- **A2A Framework**: Google's Agent-to-Agent coordination
- **LangGraph**: Multi-agent workflow orchestration
- **LangChain**: LLM integration and prompt management
- **LangSmith**: LLM call logging and debugging
- **MCP**: Model Context Protocol for API interactions

## 📦 Installation

```bash
# Clone the repository
git clone <repository-url>
cd learning-agenticAI

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration
```

## ⚙️ Configuration

### Environment Variables

```bash
# LLM Provider Configuration
LLM_PROVIDER=openai  # openai, bedrock, anthropic, vertexai
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
AWS_REGION=us-east-1

# LangSmith Configuration
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=your_project_name
LANGCHAIN_TRACING_V2=true

# MCP Configuration
MCP_SERVER_URL=http://localhost:3000
MCP_API_KEY=your_mcp_key
```

### Field Mapping Configuration

Create mapping files in `config/mappings/`:

```yaml
# config/mappings/disability_mapping.yaml
source_format: csv
target_format: json
record_type: disability
rules:
  - source_field: policy_number
    target_field: policyId
    transformation_type: direct
    validation:
      required: true
      pattern: "^[A-Z0-9]{6,12}$"
  
  - source_field: effective_date
    target_field: effectiveDate
    transformation_type: date_format
    source_format: "%Y-%m-%d"
    target_format: "ISO8601"
    validation:
      required: true
      future_date: false
```

## 🚀 Usage

### Basic Migration

```bash
# Run migration with default configuration
python main.py migrate data/input/disability_data.csv

# Run with custom mapping
python main.py migrate data/input/disability_data.csv --mapping config/mappings/custom_mapping.yaml

# Run in dry-run mode
python main.py migrate data/input/disability_data.csv --dry-run
```

### Advanced Usage

```python
from migration_platform import MigrationPlatform
from config import LLMConfig, MCPConfig

# Initialize platform
platform = MigrationPlatform(
    llm_config=LLMConfig(provider="openai", model="gpt-4"),
    mcp_config=MCPConfig(server_url="http://localhost:3000")
)

# Run migration
result = await platform.migrate(
    input_file="data/input/disability_data.csv",
    mapping_file="config/mappings/disability_mapping.yaml",
    target_system="modern_insurance_platform"
)

print(f"Migration completed: {result.success_rate}% success rate")
```

### CLI Commands

```bash
# Validate configuration
python main.py validate config/mappings/disability_mapping.yaml

# Check platform status
python main.py status

# Run tests
python main.py test

# View logs in LangSmith
python main.py logs --project insurance-migration
```

## 🔧 Development

### Project Structure

```
├── agents/                 # A2A agents
│   ├── base_agent.py      # Base agent class
│   ├── file_reader.py     # File reading agent
│   ├── validator.py       # Validation agent
│   ├── mapper.py          # Mapping agent
│   ├── transformer.py     # Transformation agent
│   ├── api_integration.py # API integration agent
│   └── orchestrator.py    # Orchestration agent
├── workflows/             # LangGraph workflows
│   ├── migration_graph.py # Main migration workflow
│   └── validation_graph.py # Validation workflow
├── llm/                   # LLM provider abstractions
│   ├── providers.py       # LLM provider factory
│   └── prompts.py         # Prompt templates
├── mcp/                   # MCP integration
│   ├── client.py          # MCP client
│   └── tools.py           # MCP tools
├── config/                # Configuration
│   ├── settings.py        # Platform settings
│   └── mappings/          # Field mapping files
├── data/                  # Sample data
├── tests/                 # Test suite
└── main.py               # CLI entry point
```

### Adding New LLM Providers

```python
from llm.providers import LLMProvider

class CustomLLMProvider(LLMProvider):
    def __init__(self, config):
        super().__init__(config)
    
    async def generate(self, prompt: str) -> str:
        # Implement your LLM provider logic
        pass
    
    async def generate_structured(self, prompt: str, schema: dict) -> dict:
        # Implement structured generation
        pass
```

### Creating Custom MCP Tools

```python
from mcp.tools import MCPTool

class InsuranceAPITool(MCPTool):
    def __init__(self):
        super().__init__("insurance_api")
    
    async def create_policy(self, policy_data: dict) -> dict:
        # Implement policy creation logic
        pass
    
    async def update_policy(self, policy_id: str, updates: dict) -> dict:
        # Implement policy update logic
        pass
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test category
pytest tests/test_agents.py

# Run with coverage
pytest --cov=migration_platform

# Run integration tests
pytest tests/integration/
```

## 📊 Monitoring

### LangSmith Integration

The platform integrates with LangSmith for comprehensive LLM call monitoring:

- **Trace LLM calls** in real-time
- **Debug agent interactions**
- **Monitor performance metrics**
- **Analyze prompt effectiveness**

### Logging

```python
import structlog

logger = structlog.get_logger()
logger.info("Migration started", 
           input_file="data.csv", 
           mapping_file="mapping.yaml")
```

## 🔒 Security

- **API Key Management**: Secure environment variable handling
- **Data Encryption**: End-to-end encryption for sensitive data
- **Access Control**: Role-based access control for API endpoints
- **Audit Logging**: Comprehensive audit trails for all operations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the documentation
- Review LangSmith traces for debugging

## 🔄 Roadmap

- [ ] Support for additional file formats (XML, EDI)
- [ ] Real-time migration monitoring dashboard
- [ ] Advanced business rule engine
- [ ] Multi-tenant support
- [ ] Performance optimization for large datasets
- [ ] Integration with additional MCP servers
