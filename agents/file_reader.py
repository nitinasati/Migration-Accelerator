"""
File Reader Agent for the Migration-Accelerators platform.

This agent uses LangGraph with proper state management, memory checkpointing,
and MCP (Model Context Protocol) for intelligent file reading operations.

Architecture:
- StateGraph workflow for file processing pipeline
- Memory checkpointer for conversation state persistence
- MCP client integration for file operations
- Tool-based architecture following LangGraph patterns
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, Annotated
from datetime import datetime
import structlog

# LangChain tool imports (for MCP integration)
try:
    from langchain_core.tools import tool
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
except ImportError:
    print("⚠️ LangChain core packages not available")
    tool = None

# MCP imports
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp_tools.file_tool_client import create_mcp_file_client
    from mcp_tools.mcp_utils import process_agent_stream
except ImportError:
    print("⚠️ MCP packages not available. Install with: pip install mcp")
    ClientSession = None
    create_mcp_file_client = None
    process_agent_stream = None

# LangGraph imports
try:
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent
except ImportError:
    print("⚠️ LangGraph packages not available. Install with: pip install langgraph langchain-openai")
    ChatOpenAI = None
    create_react_agent = None

from .base_agent import BaseAgent, AgentResult
from config.settings import LLMConfig, MCPConfig, get_llm_config
from llm.providers import BaseLLMProvider, LLMProviderFactory
from llm.prompts import get_system_prompt, get_prompt

# Constants
AGENT_INIT_ERROR = "Failed to initialize agent"

# Import MCP client
try:
    from mcp_tools.file_tool_client import MCPFileClient
except ImportError:
    print("⚠️ MCP client not available")
    MCPFileClient = type(None)  # Use type(None) for type hints when import fails


# State definitions for LangGraph
class FileReaderState(TypedDict):
    """State for the file reader workflow."""
    messages: Annotated[List[BaseMessage], "Messages in the conversation"]
    file_path: str
    file_content: Optional[str]
    file_info: Optional[Dict[str, Any]]
    analysis_results: Optional[Dict[str, Any]]
    structured_data: Optional[List[Dict[str, Any]]]
    validation_results: Optional[Dict[str, Any]]
    processing_status: str
    errors: List[str]
    warnings: List[str]
    current_step: str
    completed_steps: List[str]
    thread_id: Optional[str]


# MCPFileOperations class removed - all file operations now handled by MCP server


class FileReaderAgent(BaseAgent):
    """
    Simplified File Reader Agent that directly calls MCP tools.
    
    This agent:
    - Uses MCP client for file operations
    - Directly calls tools without complex workflow orchestration
    - Provides clean and simple file parsing functionality
    """
    
    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        mcp_config: Optional[MCPConfig] = None
    ):
        super().__init__("file_reader", llm_config, mcp_config)
        self.llm_provider: Optional[BaseLLMProvider] = None
        self.mcp_file_client: Optional[MCPFileClient] = None
        self.tools = []
    
    async def initialize(self) -> None:
        """Initialize the file reader agent."""
        await super().start()
        
        # Initialize LLM provider
        if self.llm_config:
            self.llm_provider = LLMProviderFactory.create(self.llm_config, self.agent_name)
            await self.llm_provider.initialize()
            self.logger.info("LLM provider initialized")
        
        # Note: MCPFileOperations removed - all operations now go through MCP server
        
        # Initialize MCP file client for tool integration
        if MCPFileClient:
            self.mcp_file_client = MCPFileClient()
            await self.mcp_file_client.initialize()
            self.logger.info("MCP file client initialized")
        else:
            self.logger.warning("MCP file client not available, using fallback tools")
        
        # Setup tools
        self._setup_tools()
        
        self.logger.info("File reader agent initialized")
    
    
    def _setup_tools(self):
        """Setup tools for the workflow using MCP client."""
        
        if self.mcp_file_client:
            # Use MCP client tools - these are the only tools we need
            # All tool implementations are in the MCP server
            self.tools = self.mcp_file_client.create_langchain_tools()
            self.logger.info(f"Initialized {len(self.tools)} tools from MCP server via client")
        else:
            # No tools available if MCP client is not working
            self.tools = []
            self.logger.warning("No MCP client available - agent will run without tools")
            self.logger.warning("Agent will not be able to perform file operations")
    
    
    async def _process_with_llm(self, file_path: str) -> dict:
        """Parse structured files using LangGraph MCP agent with intelligent tool selection."""
        # Create and use a temporary LangGraph agent
        agent = await self._create_langgraph_agent()
        if not agent:
            return {"success": False, "error": AGENT_INIT_ERROR}
        
        try:
            # Option 1: Use LangGraph with tools (current approach)
            # Get the optimized prompt from prompts.py
            comprehensive_query = get_prompt("file_reader_langgraph_mcp_workflow", file_path=file_path)
            
            self.logger.info("Running structured file parsing with LangGraph agent")
            
            # Process query with agent
            result = await self._execute_agent_query(agent["agent"], comprehensive_query)
            
            # The agent should return structured JSON directly (validated by parse_json_output_tool)
            if result.get("success") and result.get("final_output"):
                try:
                    # The agent's final output should be valid JSON after using parse_json_output_tool
                    final_output = result["final_output"].strip()
                    if final_output.startswith('{') and final_output.endswith('}'):
                        structured_data = json.loads(final_output)
                        result["structured_data"] = structured_data
                        result["file_path"] = file_path
                        return result
                except json.JSONDecodeError:
                    self.logger.error("Agent output is not valid JSON despite using parse_json_output_tool")
            
            return {"success": False, "error": "Agent did not return valid structured JSON"}
            
        finally:
            # Always close the client
            if agent and agent.get("client"):
                await agent["client"].close()
    
    async def _create_langgraph_agent(self) -> Optional[Dict[str, Any]]:
        """Create a LangGraph agent with MCP tools following the optimized pattern."""
        try:
            # Check if required imports are available
            if not create_mcp_file_client or not ChatOpenAI or not create_react_agent:
                self.logger.error("Required LangGraph/MCP dependencies not available")
                return None
            
            # 1️⃣ Create MCP client and discover tools
            self.logger.info("Initializing MCP file client")
            client = await create_mcp_file_client()
            
            if not client:
                self.logger.error("Failed to create MCP client")
                return None
            
            # Get available tools from MCP server
            available_tools = client.create_langchain_tools()
            
            if not available_tools:
                self.logger.error("No tools discovered from MCP server")
                await client.close()
                return None

            self.logger.info("Discovered tools from MCP", 
                           tool_names=[t.name for t in available_tools],
                           tool_count=len(available_tools))

            # 2️⃣ Create LLM using the framework's provider abstraction
            llm_config = get_llm_config()
            llm_provider = LLMProviderFactory.create(llm_config, "file_reader_agent")
            
            if not llm_provider:
                self.logger.error("Failed to create LLM provider")
                await client.close()
                return None
                
            await llm_provider.initialize()
            
            # Get the underlying LangChain client for LangGraph integration
            if hasattr(llm_provider, '_client') and llm_provider._client:
                llm = llm_provider._client
                self.logger.info("LLM provider initialized successfully", 
                               provider=llm_config.provider.value,
                               model=llm_config.model)
            else:
                self.logger.error("LLM provider client not available")
                return None

            # 3️⃣ Create LangGraph agent with discovered tools
            agent = create_react_agent(llm, available_tools)
            
            self.logger.info("LangGraph agent created successfully", 
                           provider=llm_config.provider.value,
                           model=llm_config.model, 
                           tools_count=len(available_tools))
            
            return {
                "agent": agent,
                "client": client,
                "tools": available_tools
            }
            
        except Exception as e:
            self.logger.error("Failed to create LangGraph MCP agent", error=str(e))
            return None
    
    async def _execute_agent_query(self, agent, query: str) -> dict:
        """Execute a query with the LangGraph agent and track events using common MCP utilities."""
        if not process_agent_stream:
            self.logger.error("MCP utilities not available")
            return {"success": False, "error": "MCP utilities not available"}
            
        try:
            # Use the common utility function for agent stream processing
            result = await process_agent_stream(agent, query, self.logger)
            
            # Add additional metadata for compatibility
            if result.get("success"):
                result["query"] = query
                result["tools_used"] = len([e for e in result.get("events", []) if e.get("type") == "tool_start"])
            
            return result
            
        except Exception as e:
            self.logger.error("Error processing file query", query=query, error=str(e))
            return {"success": False, "error": str(e), "query": query}
    
    async def process(self, data: Any, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """
        Process file reading request using LLM with MCP tools for intelligent decision making.
        
        Args:
            data: File path or file data
            context: Additional context (optional)
            
        Returns:
            AgentResult: Processing result with file analysis
        """
        try:
            self.logger.info("Starting LLM-driven file reading process", data_type=type(data).__name__)
            
            # Validate input
            if not self.validate_input(data):
                return AgentResult(
                    success=False,
                    errors=["Invalid input data for file reading"]
                )
            
            # Prepare file path
            if isinstance(data, str):
                file_path = data
            elif isinstance(data, dict):
                file_path = data.get("file_path", "")
            else:
                file_path = str(data)
            
            if not file_path:
                return AgentResult(
                    success=False,
                    errors=["No file path provided"]
                )
            
            # Use LLM with MCP tools to intelligently process the file
            processing_result = await self._process_with_llm(file_path)
            
            if not processing_result.get("success"):
                return AgentResult(
                    success=False,
                    errors=[processing_result.get("error", "Unknown processing error")]
                )
            
            # Extract structured data from the LLM output
            structured_data = processing_result.get("structured_data", {})
            file_data = structured_data.get("file_data", []) if structured_data else []
            
            # Create successful result
            metadata = {
                "file_path": file_path,
                "file_info": processing_result.get("structure", {}),
                "analysis_results": processing_result,
                "agent": self.agent_name,
                "processing_method": "llm_with_mcp_tools"
            }
            
            result = AgentResult(
                success=True,
                data=file_data,  # This is what the workflow expects
                errors=[],
                warnings=[],
                metadata=metadata
            )
            
            self.logger.info(
                "LLM-driven file reading completed successfully",
                file_path=file_path,
                records_count=len(file_data) if isinstance(file_data, list) else 0
            )
            
            return result
            
        except Exception as e:
            return await self.handle_error(e, {"data_type": type(data).__name__, "context": context})
    
    
    async def close(self):
        """Close the agent and cleanup resources."""
        try:
            # Close MCP file client
            if self.mcp_file_client:
                await self.mcp_file_client.close()
            
            # Close base agent
            await super().close()
            
            self.logger.info("FileReaderAgent closed successfully")
            
        except Exception as e:
            self.logger.error("Error closing FileReaderAgent", error=str(e))
