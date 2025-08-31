# Migration-Accelerator

A modern, LLM-agnostic platform for accelerating data migration projects using Agentic AI, Google's A2A framework, LangGraph, and MCP. This platform leverages intelligent agents to automate complex migration workflows across various domains and data formats.

## 🚀 Features

- **Agentic AI Architecture**: Multi-agent system that autonomously handles complex migration tasks
- **A2A Framework Integration**: Leverages Google's Agent-to-Agent framework for intelligent agent coordination
- **LLM Agnostic**: Support for OpenAI, AWS Bedrock, Anthropic, and Google Vertex AI
- **LangGraph Workflows**: Complex multi-agent workflows with state management
- **MCP Integration**: Model Context Protocol for standardized API interactions
- **Dynamic Field Mapping**: AI-powered field mapping and transformation from configuration files
- **Intelligent Validation**: LLM-powered data validation and business rule checking with batch processing
- **Rich Logging**: LangSmith integration for LLM call tracking and debugging
- **Workflow Visualization**: Interactive graph generation showing migration workflow structure
- **Performance Tracking**: Comprehensive timing and performance metrics
- **Domain Agnostic**: Works across various industries and data formats

## 🏗️ Architecture

The platform uses an Agentic AI architecture with intelligent agents that autonomously handle migration tasks:

### Core Agents
1. **File Reader Agent** - Intelligently reads and parses various file formats
2. **Validation Agent** - AI-powered data integrity and business rule validation
3. **Mapping Agent** - Autonomous field mapping and transformation discovery
4. **Transformation Agent** - Intelligent data format conversion and optimization
5. **API Integration Agent** - Manages MCP-based API calls with error recovery
6. **Orchestration Agent** - Coordinates the entire workflow with adaptive decision making

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
cd Migration-Accelerators

# Install dependencies
pip install -r requirements.txt

# Install Graphviz (required for workflow visualization)
# Windows:
winget install graphviz

# macOS:
brew install graphviz

# Ubuntu/Debian:
sudo apt-get install graphviz

# Set up environment variables
cp env.example .env
# Edit .env with your configuration
```

### Graphviz Setup (Windows)

For Windows users, run the setup script to configure Graphviz:

```powershell
# Run as Administrator
.\setup_graphviz.ps1
```

This script will:
- Check if Graphviz is installed
- Add Graphviz to your PATH
- Test the installation
- Provide usage instructions

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
python main.py migrate data/input/sample_disability_data.csv

# Run with custom mapping
python main.py migrate data/input/sample_disability_data.csv --mapping config/mappings/custom_mapping.yaml

# Run in dry-run mode
python main.py migrate data/input/sample_disability_data.csv --dry-run
```

### Advanced Usage

```python
from workflows.migration_graph import MigrationWorkflow
from config.settings import LLMConfig, MCPConfig

# Initialize platform
workflow = MigrationWorkflow(
    llm_config=LLMConfig(provider="openai", model="gpt-4"),
    mcp_config=MCPConfig(server_url="http://localhost:3000")
)

# Run migration
result = await workflow.run(
    file_path="data/input/sample_disability_data.csv",
    mapping_config=mapping_config,
    record_type="disability"
)

print(f"Migration completed: {result['migration_summary']['success']}")
```

### CLI Commands

```bash
# Validate configuration
python main.py validate config/mappings/disability_mapping.yaml

# Check platform status
python main.py status

# Run tests
python main.py test

# Generate workflow graph
python main.py graph

# Generate custom graph
python main.py graph --output custom_workflow.png

# View logs in LangSmith
python main.py logs --project migration-accelerators
```

### Workflow Visualization

The platform includes powerful graph visualization capabilities:

```bash
# Generate and display workflow graph
python main.py graph

# Generate graph during migration
python main.py migrate data/input/sample_disability_data.csv --dry-run
```

**Graph Features:**
- **Visual Workflow**: See the complete migration workflow structure
- **Node Relationships**: Understand how agents interact and data flows
- **Error Paths**: Visualize error handling and recovery paths
- **PNG Export**: Save graphs as high-quality images
- **Console Display**: View workflow structure in terminal

**Generated Files:**
- `migration_workflow_graph.png` - Main workflow graph
- `data/output/migration_workflow_graph.png` - Graph generated during migration

### Performance Tracking

The platform provides comprehensive performance monitoring:

```bash
# Run migration with timing information
python main.py migrate data/input/sample_disability_data.csv --dry-run
```

**Performance Metrics:**
- **Total Duration**: Complete end-to-end migration time
- **Step Timing**: Individual agent processing times
- **Start/End Times**: Precise timestamps for migration execution
- **Progress Tracking**: Real-time progress monitoring
- **Resource Usage**: Memory and processing efficiency metrics

**Sample Output:**
```
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Metric             ┃ Value                      ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Success            │ ✓ Yes                      │
│ Records Processed  │ 10                         │
│ Progress           │ 100.0%                     │
│ Completed Steps    │ 5                          │
│ Total Duration     │ 0:00:20                    │
│ Duration (seconds) │ 20.03s                     │
│ Start Time         │ 2025-08-31T19:02:02.299400 │
│ End Time           │ 2025-08-31T19:02:22.333727 │
└────────────────────┴────────────────────────────┘
```

## 🔧 Development

### Project Structure

```
├── agents/                 # A2A agents
│   ├── base_agent.py      # Base agent class
│   ├── file_reader.py     # File reading agent
│   ├── validation.py      # Validation agent with batch processing
│   ├── mapping.py         # Mapping agent
│   ├── transformation.py  # Transformation agent
│   ├── api_integration.py # API integration agent
│   └── orchestration.py   # Orchestration agent
├── workflows/             # LangGraph workflows
│   └── migration_graph.py # Main migration workflow with graph generation
├── llm/                   # LLM provider abstractions
│   ├── providers.py       # LLM provider factory with LangSmith integration
│   └── prompts.py         # Prompt templates including batch validation
├── mcp_tools/             # MCP integration (renamed from mcp/)
│   └── client.py          # MCP client
├── config/                # Configuration
│   ├── settings.py        # Platform settings with timing support
│   ├── mappings.py        # Mapping utilities
│   └── mappings/          # Field mapping files
├── data/                  # Sample data and outputs
│   ├── input/             # Input data files
│   └── output/            # Generated output files and graphs
├── tests/                 # Test suite
├── main.py               # CLI entry point with graph commands
├── example.py            # Usage examples
├── setup_graphviz.ps1    # Graphviz setup script for Windows
├── requirements.txt      # Dependencies including graphviz
└── README.md             # This file
```

### Adding New LLM Providers

```python
from llm.providers import BaseLLMProvider

class CustomLLMProvider(BaseLLMProvider):
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
from mcp.client import MCPTool

class DataAPITool(MCPTool):
    def __init__(self):
        super().__init__("data_api")
    
    async def create_record(self, record_data: dict) -> dict:
        # Implement record creation logic
        pass
    
    async def update_record(self, record_id: str, updates: dict) -> dict:
        # Implement record update logic
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
pytest tests/test_integration.py
```

## 📊 Monitoring

### LangSmith Integration

The platform integrates with LangSmith for comprehensive LLM call monitoring:

- **Trace LLM calls** in real-time
- **Debug agent interactions**
- **Monitor performance metrics**
- **Analyze prompt effectiveness**

### Batch Processing Optimization

The platform includes intelligent batch processing to optimize LLM usage:

- **Batch Validation**: Process multiple records in single LLM calls
- **Configurable Batch Size**: Adjust batch size based on model limits
- **Reduced API Calls**: Minimize throttling and improve performance
- **Cost Optimization**: Lower API costs through efficient batching

**Example Configuration:**
```python
# In agents/validation.py
batch_size = 5  # Process 5 records at a time
```

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

- [x] **Workflow Visualization** - Interactive graph generation showing migration workflow structure
- [x] **Performance Tracking** - Comprehensive timing and performance metrics
- [x] **Batch Processing Optimization** - Intelligent batch processing to optimize LLM usage
- [x] **LangSmith Integration** - Full LLM call tracking and debugging
- [x] **Graphviz Integration** - Visual workflow representation with PNG export
- [ ] Support for additional file formats (XML, EDI, Parquet, Avro)
- [ ] Real-time migration monitoring dashboard
- [ ] Advanced Agentic AI decision making capabilities
- [ ] Multi-tenant support with agent isolation
- [ ] Performance optimization for large datasets
- [ ] Integration with additional MCP servers
- [ ] Autonomous schema discovery and mapping
- [ ] Cross-domain migration templates

## 🎯 Use Cases

### Insurance Data Migration
- **Disability Insurance**: Migrate disability policy data from mainframe to modern systems
- **Absence Management**: Transfer absence records and leave management data
- **Group Policies**: Handle group insurance policy migrations
- **Claims Processing**: Migrate claims data with validation and transformation

### General Data Migration
- **Customer Data**: Migrate customer information across systems
- **Product Catalogs**: Transfer product data with complex relationships
- **Financial Records**: Handle sensitive financial data migrations
- **HR Systems**: Migrate employee and organizational data

## 🚀 Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Install Graphviz**:
   ```bash
   # Windows
   winget install graphviz
   
   # macOS
   brew install graphviz
   
   # Ubuntu/Debian
   sudo apt-get install graphviz
   ```

3. **Configure Environment**:
   ```bash
   cp env.example .env
   # Edit .env with your API keys
   ```

4. **Run Sample Migration**:
   ```bash
   python main.py migrate data/input/sample_disability_data.csv --dry-run
   ```

5. **Generate Workflow Graph**:
   ```bash
   python main.py graph
   ```

6. **Check Status**:
   ```bash
   python main.py status
   ```

7. **Run Tests**:
   ```bash
   python main.py test
   ```

### Output Files

After running a migration, you'll find:

- **`data/output/disability_policy_YYYYMMDD_HHMMSS.json`** - Transformed data in JSON format
- **`data/output/migration_workflow_graph.png`** - Visual workflow graph
- **`migration_workflow_graph.png`** - Standalone workflow graph (when using `graph` command)

## 📈 Performance

- **Concurrent Processing**: Handles large datasets with parallel processing
- **Memory Efficient**: Streams data to avoid memory issues
- **Error Recovery**: Automatic retry and error handling
- **Progress Tracking**: Real-time progress monitoring
- **Scalable Architecture**: Designed for enterprise-scale migrations

---

**Migration-Accelerators** - Accelerating data migration with the power of Agentic AI 🚀
