"""
Simple File Reader Agent for Migration Accelerator

This agent handles file reading operations using direct LLM prompting.
It can read various file formats and convert them to structured JSON data.
"""

import json
import os
from typing import Any, Dict, List, Optional
import structlog
import dotenv
import aiofiles

# Load environment variables
dotenv.load_dotenv()

from agents.base_agent import BaseAgent, AgentResult
from config.settings import LLMConfig
from llm.prompts import FILE_CONVERSION_BASE, FILE_CONVERSION_WITH_LAYOUT
from llm.llm_provider import LLMProviderFactory

logger = structlog.get_logger(__name__)


class SimpleFileReaderAgent(BaseAgent):
    """Simple file reader agent that uses LLM prompting for file conversion."""
    
    def __init__(self, llm_config: Optional[LLMConfig] = None):
        super().__init__("simple_file_reader", llm_config)
        self.llm_provider = None
        
    async def initialize(self) -> None:
        """Initialize the file reader agent."""
        await super().start()
        
        # Initialize LLM provider using the configuration
        if self.llm_config:
            self.llm_provider = LLMProviderFactory.create(self.llm_config, self.agent_name)
            await self.llm_provider.initialize()
            self.logger.info("LLM provider initialized for simple file reader agent")
        else:
            raise ValueError("LLM configuration is required for SimpleFileReaderAgent")
    
    async def read_file(self, file_path: str, layout_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Read and convert a file to JSON using LLM prompting.
        
        Args:
            file_path: Path to the file to read
            layout_file: Optional layout file for mainframe .dat files
            
        Returns:
            Dict containing the converted JSON data and metadata
        """
        if not self.llm_provider:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        
        try:
            # Validate file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Get file extension
            ext = os.path.splitext(file_path)[1].lower()
            
            # Read the main data file
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                file_content = await f.read()
            
            # Determine conversion strategy
            if ext == ".json":
                return await self._read_json_file(file_path, file_content)
            else:
                # Use common LLM conversion for all other file types
                return await self._convert_file_with_llm(file_path, file_content, ext, layout_file)
                
        except Exception as e:
            self.logger.error("Error reading file", file_path=file_path, error=str(e))
            raise
    
    async def _convert_file_with_llm(self, file_path: str, file_content: str, ext: str, layout_file: Optional[str] = None) -> Dict[str, Any]:
        """Convert file to JSON using LLM with appropriate prompt based on file type."""
        
        # Create prompt based on file type and layout availability
        if layout_file:
            # Mainframe file with layout - use specialized prompt
            async with aiofiles.open(layout_file, "r", encoding="utf-8") as f:
                layout_content = await f.read()
            
            prompt = FILE_CONVERSION_WITH_LAYOUT.format(
                layout_content=layout_content,
                file_content=file_content
            )
            file_type = "mainframe"
            
        else:
            # Non-mainframe files (CSV, TXT, etc.) - use base prompt
            prompt = FILE_CONVERSION_BASE.format(
                file_content=file_content
            )
            file_type = ext[1:] if ext.startswith('.') else ext
        
        return await self._call_llm(prompt, file_type)
    
    async def _read_json_file(self, file_path: str, file_content: str) -> Dict[str, Any]:
        """Read existing JSON file."""
        try:
            data = json.loads(file_content)
            return {
                "source_file": file_path,
                "content_type": "json",
                "data": data,
                "total_records": len(data) if isinstance(data, list) else 1,
                "success": True
            }
        except json.JSONDecodeError as e:
            return {
                "source_file": file_path,
                "content_type": "json",
                "error": f"Invalid JSON: {str(e)}",
                "success": False
            }
    
    async def _call_llm(self, prompt: str, file_type: str) -> Dict[str, Any]:
        """Call LLM API to convert file content using the configured provider."""
        try:
            if not self.llm_provider:
                raise ValueError("LLM provider not initialized")
            
            # Prepare system message
            system_message = f"You are a {file_type} file conversion expert. You MUST respond with only valid JSON. No explanations, no markdown formatting, no code blocks. Start with {{ and end with }}."
            
            # Call the LLM provider
            json_text = await self.llm_provider.generate(
                prompt=prompt,
                system_message=system_message,
                temperature=0  # deterministic output
            )
            
            # Try parsing JSON to validate
            try:
                json_data = json.loads(json_text)
                json_data["success"] = True
                return json_data
            except json.JSONDecodeError:
                return {
                    "success": False,
                    "error": "LLM did not return valid JSON",
                    "raw_output": json_text
                }
                
        except Exception as e:
            self.logger.error("Error calling LLM", error=str(e))
            return {
                "success": False,
                "error": f"LLM call failed: {str(e)}"
            }
    
    async def process(self, data: Any, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """
        Process data using the file reader agent.
        
        Args:
            data: File path or file data to process
            context: Additional context (e.g., layout_file)
            
        Returns:
            AgentResult: Processing result
        """
        try:
            # Handle different input types
            if isinstance(data, str):
                # data is a file path
                file_path = data
                layout_file = context.get("layout_file") if context else None
                result = await self.read_file(file_path, layout_file)
            else:
                # data is already file content
                result = {"success": False, "error": "Direct file content processing not supported"}
            
            if result.get("success", False):
                return AgentResult(
                    success=True,
                    data=result,
                    metadata={"file_path": data if isinstance(data, str) else "unknown"}
                )
            else:
                return AgentResult(
                    success=False,
                    errors=[result.get("error", "Unknown error")],
                    data=result
                )
                
        except Exception as e:
            self.logger.error("Error in process method", error=str(e))
            return AgentResult(
                success=False,
                errors=[str(e)]
            )

    async def close(self) -> None:
        """Close the file reader agent."""
        self.logger.info("Simple file reader agent closed")
        await super().close()
