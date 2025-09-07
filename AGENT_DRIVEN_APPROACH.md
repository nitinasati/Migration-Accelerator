# Agent-Driven Tool Selection Approach

## Overview

This document explains the updated implementation where the LangGraph agent autonomously decides which MCP tools to use based on comprehensive prompts, rather than hardcoding specific tool sequences.

## Key Changes Made

### Before (Hardcoded Approach)
```python
# Multiple separate queries
example_queries = [
    "Detect the file type of './data/input/sample_absence_data.csv'",
    "Read the content of './data/input/sample_absence_data.csv'", 
    "Parse './data/input/sample_absence_data.csv' as structured data"
]

# Agent processes each query separately
for query in example_queries:
    result = await agent.process_file_query(query)
```

### After (Agent-Driven Approach)
```python
# Single comprehensive prompt
comprehensive_query = """
I need you to perform a complete analysis of the file './data/input/sample_absence_data.csv'. 
Please follow these steps in the most appropriate order:

1. First, detect and analyze the file type to understand what we're working with
2. Read the file content to see the actual data structure and content
3. Parse the file as structured data to extract all records properly
4. Analyze the overall file structure and provide recommendations for processing
5. Validate the data quality and format

For each step, choose the most appropriate tool available and provide detailed insights.
"""

# Agent decides which tools to use and in what order
result = await agent.process_file_query(comprehensive_query)
```

## Benefits of Agent-Driven Approach

### 🧠 Intelligent Decision Making
- Agent analyzes the requirements and chooses appropriate tools
- Dynamic tool selection based on file type and context
- Optimized workflow based on LLM reasoning

### 🔄 Flexible Workflow
- No hardcoded tool sequences
- Adaptive to different file types and scenarios
- Can handle complex multi-step analysis requests

### 📊 Better Observability
- Clear logging of which tools the agent chose
- Tool execution order determined by agent reasoning
- Event streaming shows decision-making process

## Implementation Modes

### 1. Comprehensive Analysis Mode
```python
await run_comprehensive_analysis()
```
- Single detailed prompt with multiple requirements
- Agent breaks down into steps and chooses tools
- Complete file analysis workflow

### 2. Multiple Scenarios Mode  
```python
await run_multiple_scenarios()
```
- Different analysis scenarios (quality assessment, migration prep, etc.)
- Each scenario lets agent decide tool usage
- Demonstrates flexibility across use cases

### 3. Interactive Mode
```python
await interactive_mode()
```
- Manual query input
- Real-time agent-driven tool selection
- Custom analysis requests

## Example Agent Decision Making

When given the comprehensive prompt, the agent might choose:

1. **detect_file_type_tool** - To understand file characteristics
2. **read_file_content_tool** - To see actual content structure  
3. **parse_structured_file_tool** - To extract structured data
4. **analyze_file_structure_tool** - For detailed metadata analysis
5. **validate_json_content_tool** - If JSON validation needed

The agent decides this sequence based on:
- File type discovered in step 1
- Content analysis from step 2
- Requirements specified in the prompt
- Available tool capabilities

## Tool Selection Intelligence

### Context-Aware Decisions
```
File: sample_absence_data.csv
Agent reasoning: "This is a CSV file, so I should:
1. First detect file type to confirm format
2. Read content to understand structure  
3. Parse as structured data to extract records
4. Analyze structure for processing recommendations"
```

### Adaptive Workflows
```
Different file types → Different tool sequences
CSV files: detect → read → parse → analyze
JSON files: detect → read → validate → parse → analyze  
Unknown files: detect → analyze → read (if safe)
```

## Logging and Observability

### Agent Decision Tracking
```python
logger.info("🔧 Agent chose tools", tool_names=['detect_file_type_tool', 'parse_structured_file_tool'])
logger.info("🤖 Agent starting chain: AgentExecutor")
logger.info("🔨 Tool Called: detect_file_type_tool", inputs={'file_path': './data/input/sample_absence_data.csv'})
logger.info("✅ Tool Result: File detected as CSV with 5 records...")
```

### Tool Usage Analytics
```python
# Automatic tracking of tool frequency across scenarios
Tool usage frequency: {
    'detect_file_type_tool': 4,
    'parse_structured_file_tool': 3, 
    'read_file_content_tool': 2,
    'analyze_file_structure_tool': 4
}
```

## Code Structure

### Main Agent Class
```python
class LangGraphMCPAgent:
    async def initialize()           # Setup MCP client and discover tools
    async def process_file_query()   # Process comprehensive queries
    async def close()               # Cleanup resources
```

### Event Processing Methods
```python
def _process_chain_start_event()    # Handle workflow start
def _process_tool_start_event()     # Handle tool execution start  
def _process_tool_end_event()       # Handle tool completion
def _process_chain_end_event()      # Handle final results
```

### Scenario Management
```python
def _create_analysis_scenarios()    # Define analysis scenarios
def _process_scenario_result()      # Process and log results
async def run_multiple_scenarios()  # Execute multiple scenarios
```

## Integration with Migration-Accelerator

### Following Workspace Rules
✅ **Agent-Based Architecture**: Uses BaseAgent patterns  
✅ **Structured Logging**: Console output with context [[memory:7739487]]  
✅ **Type Hints**: All functions properly typed  
✅ **Async Operations**: Proper async/await patterns  
✅ **MCP Integration**: Uses established MCP client  
✅ **Error Handling**: Comprehensive error management  

### File Structure Integration
```
Migration-Accelerator/
├── langgraph_mcp_example.py        # 🆕 Agent-driven example
├── AGENT_DRIVEN_APPROACH.md        # 🆕 This documentation
├── LANGGRAPH_MCP_USAGE.md          # Updated usage guide
├── mcp_tools/
│   ├── file_tool_client.py         # MCP client for tool discovery
│   └── file_mcp_server.py          # FastMCP file operations
└── agents/
    └── file_reader.py              # Existing file reader agent
```

## Testing the Implementation

### Quick Test
```bash
python -c "import langgraph_mcp_example; print('✅ Implementation ready')"
```

### Full Demo
```bash
python langgraph_mcp_example.py
# Choose option 1 for comprehensive analysis
# Choose option 3 for multiple scenarios
```

### Custom Testing
```bash
python langgraph_mcp_example.py
# Choose option 2 for interactive mode
# Enter custom prompts like:
# "Analyze this file for migration purposes and recommend the best processing approach"
```

This agent-driven approach provides much more flexibility and intelligence compared to hardcoded tool sequences, allowing the LLM to make optimal decisions based on context and requirements.
