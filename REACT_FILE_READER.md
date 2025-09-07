# LangGraph File Reader Agent with MCP Integration

## Overview

The LangGraph File Reader Agent implements a **StateGraph workflow** using LangGraph for intelligent file reading operations. It integrates **MCP (Model Context Protocol)** client for file operations, **memory checkpointing** for conversation state persistence, and follows **LangGraph best practices** for tool-based architecture.

## Architecture

```
FileReaderAgent (Main Interface)
├── StateGraph Workflow (LangGraph)
│   ├── Agent Node (LLM Processing)
│   ├── Tools Node (MCP Operations)
│   └── Conditional Edges (Tool Routing)
├── Memory Checkpointer (State Persistence)
│   ├── MemorySaver (In-memory)
│   └── SqliteSaver (Persistent)
├── MCPFileOperations (MCP Client)
│   ├── detect_file_type
│   ├── read_file_content
│   ├── parse_structured_file
│   └── validate_json_content
├── Tool Implementations
│   ├── detect_file_type_tool
│   ├── read_file_content_tool
│   ├── parse_structured_file_tool
│   └── validate_json_content_tool
└── State Management
    ├── FileReaderState (TypedDict)
    ├── Thread-based conversations
    └── Conversation history tracking
```

## Key Features

### 🏗️ LangGraph StateGraph Implementation
- **Workflow Orchestration**: StateGraph manages the file processing pipeline
- **Agent Node**: Handles LLM reasoning and decision making
- **Tools Node**: Executes MCP operations for file handling
- **Conditional Edges**: Routes between agent and tools based on LLM responses

### 💾 Memory Checkpointing
- **Conversation Persistence**: Maintains state across multiple interactions
- **Thread Management**: Separate conversation threads with unique IDs
- **History Tracking**: Full conversation history retrieval and management
- **Flexible Storage**: Memory or SQLite-based checkpointing options

### 🔧 MCP Server Tools
- **File Type Detection**: Intelligent detection of file types, extensions, and MIME types
- **Content Reading**: Multi-encoding support with automatic detection
- **Structured Parsing**: CSV, JSON, XML parsing with structure analysis
- **JSON Validation**: Format validation with schema support
- **File Analysis**: Comprehensive metadata and structure analysis

### 🎯 Intelligent Analysis
- **Context-Aware**: Adapts analysis based on file type and requirements
- **Error Handling**: Graceful degradation and detailed error reporting
- **Recommendations**: Provides processing recommendations based on analysis
- **Backward Compatibility**: Works with existing FileReaderAgent interface

## Components

### 1. ReactFileReaderAgent (`agents/react_file_reader.py`)

The main React agent that orchestrates file analysis using LangGraph's create_react_agent.

**Key Methods:**
- `initialize()`: Sets up LLM provider and React agent
- `process()`: Main processing method using React pattern
- `_process_with_react_agent()`: Uses LangGraph React agent
- `_process_manual_workflow()`: Fallback manual workflow

### 2. MCPFileTools (`agents/react_file_reader.py`)

Tool implementations for LangGraph integration.

**Available Tools:**
```python
@tool
async def detect_file_extension(file_path: str) -> str:
    """Detect and analyze file extension and type."""

@tool  
async def read_file_content(file_path: str, encoding: str = "auto", max_size_mb: int = 10) -> str:
    """Read file content with encoding detection."""

@tool
async def parse_structured_file(file_path: str, file_type: str = "auto", sample_rows: int = 5) -> str:
    """Parse structured files (CSV, JSON, XML) into structured data."""

@tool
async def validate_json_format(json_content: str, schema: Optional[Dict] = None) -> str:
    """Validate JSON format and structure."""
```

### 3. FileReadingMCPServer (`mcp_tools/file_mcp_server.py`)

Full MCP server implementation for file operations.

**Server Tools:**
- `detect_file_extension`: File type analysis
- `read_file_content`: Content reading with encoding detection
- `parse_structured_file`: Structured data parsing
- `validate_json_format`: JSON validation
- `analyze_file_structure`: Comprehensive file analysis

### 4. Enhanced FileReaderAgent (`agents/file_reader.py`)

Updated main agent with React integration and fallback support.

**Features:**
- Automatic React agent initialization
- Graceful fallback to traditional LLM approach
- Result enhancement for backward compatibility
- Comprehensive error handling

## Usage Examples

### Basic Usage

```python
from agents.file_reader import FileReaderAgent
from config.settings import get_llm_config, get_mcp_config

# Initialize agent
agent = FileReaderAgent(get_llm_config(), get_mcp_config())
await agent.initialize()

# Analyze file
result = await agent.process("data/sample.csv")
```

### Advanced Analysis

```python
# Specify analysis requirements
context = {
    "analysis_requests": ["full_analysis", "structure_analysis", "validation"]
}

result = await agent.process("data/sample.json", context)
```

### Direct React Agent

```python
from agents.react_file_reader import ReactFileReaderAgent

# Use React agent directly
react_agent = ReactFileReaderAgent(llm_config, mcp_config)
await react_agent.initialize()

analysis_request = {
    "file_path": "data/sample.csv",
    "analysis_requests": ["structure_analysis", "data_validation"]
}

result = await react_agent.process(analysis_request)
```

### MCP Tools Directly

```python
from agents.react_file_reader import MCPFileTools

# Use individual tools
detection = await MCPFileTools.detect_file_extension("data/file.csv")
content = await MCPFileTools.read_file_content("data/file.csv")
parsed = await MCPFileTools.parse_structured_file("data/file.csv", "csv", 10)
validation = await MCPFileTools.validate_json_format('{"test": "data"}')
```

## Response Format

### React Agent Response

```json
{
  "file_path": "data/sample.csv",
  "analysis_timestamp": "2024-01-15T10:30:00",
  "file_info": {
    "success": true,
    "file_info": {
      "name": "sample.csv",
      "extension": ".csv",
      "exists": true,
      "size_mb": 0.15
    },
    "type_info": {
      "category": "structured_data",
      "format_type": "csv"
    }
  },
  "structured_data": {
    "success": true,
    "file_type": "csv",
    "structure": {
      "headers": ["id", "name", "status"],
      "total_rows": 100,
      "sample_rows": 5
    },
    "sample_data": [
      {"id": "1", "name": "John", "status": "active"},
      {"id": "2", "name": "Jane", "status": "pending"}
    ]
  },
  "summary": {
    "file_analyzed": true,
    "file_type": "csv",
    "data_summary": {
      "type": "csv",
      "columns": 3,
      "rows": 100
    }
  },
  "recommendations": [
    "CSV file ready for field mapping and transformation",
    "File is structured csv - suitable for data processing"
  ]
}
```

## Configuration

### Requirements

Add to `requirements.txt`:
```
langgraph>=0.6.0
langchain>=0.3.0
langchain-core>=0.3.0
mcp>=1.0.0
mcp-tools>=0.1.0
```

### Environment Variables

```bash
# LLM Configuration
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=your_key_here

# MCP Configuration  
MCP_SERVER_URL=http://localhost:3000
MCP_API_KEY=your_mcp_key
```

## Testing

Run the test script:

```bash
python test_react_file_reader.py
```

The test script demonstrates:
- Direct React agent usage
- Enhanced FileReaderAgent functionality
- Individual MCP tool testing
- Various file types and analysis contexts

## Error Handling

The system provides multiple layers of error handling:

1. **React Agent Level**: LangGraph handles tool execution errors
2. **MCP Tools Level**: Individual tool error handling and validation
3. **Agent Level**: Graceful fallback to traditional LLM approach
4. **Result Enhancement**: Backward compatibility transformation

## Benefits

### For Developers
- **Extensible**: Easy to add new MCP tools for additional file operations
- **Testable**: Individual tools can be tested independently
- **Maintainable**: Clear separation of concerns between reasoning and actions

### For Users
- **Intelligent**: Agent reasons about file requirements and chooses appropriate tools
- **Robust**: Multiple fallback mechanisms ensure reliable operation
- **Comprehensive**: Detailed analysis with actionable recommendations
- **Compatible**: Works with existing migration workflows

## Future Enhancements

- **Custom Schema Validation**: Support for custom JSON schemas
- **Binary File Support**: Analysis of binary file formats
- **Cloud File Support**: Integration with cloud storage providers
- **Streaming Analysis**: Support for large file streaming analysis
- **Caching**: Intelligent caching of analysis results
- **Multi-file Analysis**: Batch analysis of multiple files

## Contributing

When adding new file operations:

1. Add tool implementation to `MCPFileTools`
2. Register tool in `FileReadingMCPServer`
3. Update React agent prompts if needed
4. Add tests to `test_react_file_reader.py`
5. Update documentation

## See Also

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [MCP Protocol Specification](https://modelcontextprotocol.io/)
- [ReAct Pattern Paper](https://react-lm.github.io/)
