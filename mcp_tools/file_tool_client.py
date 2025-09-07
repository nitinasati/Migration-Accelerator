"""
Simplified MCP Client for LangGraph.
✅ No hardcoded tool names
✅ Uses MCP + LangGraph's tool creation helpers
✅ Minimal boilerplate, schema passed as metadata
"""

import json
from typing import Any, Dict, List, Optional
import structlog

try:
    import mcp
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except ImportError:
    print("⚠️ MCP packages not available. Install with: pip install mcp")
    mcp = None
    ClientSession = None

try:
    from langchain_core.tools import tool
except ImportError:
    tool = None

logger = structlog.get_logger(__name__)


class MCPFileClient:
    """Lightweight MCP client that dynamically registers MCP tools for LangGraph."""

    def __init__(self, server_path: Optional[str] = None):
        self.server_path = server_path or "mcp_tools/file_mcp_server.py"
        self._available_tools = []

    async def initialize(self):
        """Discover MCP tools once and store metadata."""
        if not mcp:
            logger.warning("MCP not available")
            return

        params = StdioServerParameters(command="python", args=[self.server_path])
        async with stdio_client(params) as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()
                tools_result = await session.list_tools()
                self._available_tools = tools_result.tools
                logger.info("Discovered MCP tools", count=len(self._available_tools))

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Generic tool caller for any MCP tool."""
        params = StdioServerParameters(command="python", args=[self.server_path])
        async with stdio_client(params) as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments)

                if not result.content:
                    return json.dumps({"success": False, "error": "No content returned"})

                for c in result.content:
                    if hasattr(c, "text"):
                        return c.text
                return str(result.content[0])

    def create_langchain_tools(self) -> List[Any]:
        """Create simple LangChain tools with @tool decorator."""
        if not tool or not self._available_tools:
            return []

        tools = []
        for t in self._available_tools:
            # Create simple wrapper function with closure
            def make_tool_fn(tool_name: str):
                async def _tool_fn(input_args) -> str:
                    """Dynamically generated MCP tool wrapper."""
                    try:
                        # Handle both dict and kwargs formats
                        if isinstance(input_args, dict):
                            kwargs = input_args
                        else:
                            # Fallback for other formats
                            kwargs = input_args.__dict__ if hasattr(input_args, '__dict__') else {}
                        
                        # Call the MCP tool directly
                        return await self.call_tool(tool_name, kwargs)
                    except Exception as e:
                        error_msg = f"Error calling MCP tool: {str(e)}"
                        logger.error("Tool execution error", tool_name=tool_name, error=str(e))
                        return error_msg
                return _tool_fn

            # Use @tool decorator
            tool_fn = make_tool_fn(t.name)
            langchain_tool = tool(tool_fn)
            langchain_tool.name = t.name
            langchain_tool.description = t.description or "MCP-provided tool"
            
            tools.append(langchain_tool)
        
        return tools

    async def close(self):
        """Gracefully close MCP client session."""
        # Using short-lived connections, no persistent session to close
        pass

async def create_mcp_file_client(server_path: Optional[str] = None) -> MCPFileClient:
    client = MCPFileClient(server_path)
    await client.initialize()
    return client



