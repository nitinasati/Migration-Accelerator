"""
File Reader Agent for the Migration-Accelerators platform.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import structlog

from .base_agent import BaseAgent, AgentResult
from config.settings import LLMConfig, MCPConfig
from llm.providers import BaseLLMProvider, LLMProviderFactory
from llm.prompts import get_prompt, get_system_prompt


class FileReaderAgent(BaseAgent):
    """
    File Reader Agent for processing mainframe and legacy system files using LLM.
    
    This agent uses LLM to:
    - Automatically detect file format
    - Parse various file formats (CSV, Excel, XML, JSON, Fixed-width)
    - Handle encoding and formatting issues
    - Extract structured data from files
    - Provide intelligent file analysis
    """
    
    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        mcp_config: Optional[MCPConfig] = None
    ):
        super().__init__("file_reader", llm_config, mcp_config)
        self.llm_provider: Optional[BaseLLMProvider] = None
    
    async def initialize(self) -> None:
        """Initialize the file reader agent."""
        await super().start()
        
        if self.llm_config:
            self.llm_provider = LLMProviderFactory.create(self.llm_config, self.agent_name)
            await self.llm_provider.initialize()
            self.logger.info("LLM provider initialized for file reader")
        
        self.logger.info("File reader agent initialized")
    
    async def process(self, data: Any, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """
        Process file reading request using LLM.
        
        Args:
            data: File path or file data
            context: Additional context (file format, encoding, etc.)
            
        Returns:
            AgentResult: Processing result with file data
        """
        try:
            self.logger.info("Starting file reading process", data_type=type(data).__name__)
            
            # Validate input
            if not self.validate_input(data):
                return AgentResult(
                    success=False,
                    errors=["Invalid input data for file reading"]
                )
            
            # Check if LLM provider is available
            if not self.llm_provider:
                return AgentResult(
                    success=False,
                    errors=["LLM provider not available for file reading"]
                )
            
            # Determine if data is file path or file content
            if isinstance(data, str) and Path(data).exists():
                file_path = data
                file_content = await self._read_file_content(file_path)
            else:
                file_path = context.get("file_path") if context else None
                file_content = str(data)
            
            # Use LLM to read and parse the file
            result = await self._read_file_with_llm(file_path, file_content, context)
            
            self.logger.info(
                "File reading completed",
                file_path=file_path,
                records_count=len(result.data) if isinstance(result.data, list) else 1
            )
            
            return result
            
        except Exception as e:
            return await self.handle_error(e, {"data_type": type(data).__name__, "context": context})
    
    async def _read_file_content(self, file_path: str) -> str:
        """
        Read file content from disk.
        
        Args:
            file_path: Path to the file
            
        Returns:
            str: File content
        """
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'ascii']
            
            for encoding in encodings:
                try:
                    import aiofiles
                    async with aiofiles.open(file_path, 'r', encoding=encoding) as file:
                        content = await file.read()
                    self.logger.info("File content read successfully", encoding=encoding)
                    return content
                except UnicodeDecodeError:
                    continue
            
            # If all encodings fail, read as binary and decode with errors='replace'
            import aiofiles
            async with aiofiles.open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                content = await file.read()
            self.logger.warning("File read with encoding errors replaced")
            return content
            
        except Exception as e:
            self.logger.error("Error reading file content", error=str(e))
            raise
    
    async def _read_file_with_llm(self, file_path: Optional[str], file_content: str, context: Optional[Dict[str, Any]]) -> AgentResult:
        """
        Use LLM to read and parse file content.
        
        Args:
            file_path: Path to the file
            file_content: Raw file content
            context: Additional context
            
        Returns:
            AgentResult: Parsed file data
        """
        try:
            # Detect file format from extension
            file_format = self._detect_format_from_extension(file_path)
            
            # Create file reading prompt
            prompt = get_prompt(
                "file_reader_read_file",
                file_path=file_path or "unknown",
                file_format=file_format,
                file_content=file_content
            )
            
            system_prompt = get_system_prompt("file_reader")
            
            # Get LLM response
            response = await self.llm_provider.generate(prompt, system_prompt)
            
            # Parse LLM response
            result = self._parse_llm_response(response)
            
            # Add metadata
            result.metadata.update({
                "file_format": file_format,
                "file_path": file_path,
                "agent": self.agent_name,
                "llm_processed": True
            })
            
            return result
            
        except Exception as e:
            self.logger.error("LLM file reading failed", error=str(e))
            return AgentResult(
                success=False,
                errors=[f"LLM file reading error: {str(e)}"]
            )
    
    def _detect_format_from_extension(self, file_path: Optional[str]) -> str:
        """
        Detect file format from file extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            str: Detected file format
        """
        if not file_path:
            return "csv"  # Default to CSV if no path provided
        
        # Extract file extension
        import os
        file_extension = os.path.splitext(file_path)[1].lower()
        
        # Map extensions to formats
        extension_map = {
            '.csv': 'csv',
            '.xlsx': 'excel',
            '.xls': 'excel',
            '.json': 'json',
            '.xml': 'xml',
            '.txt': 'fixed_width',  # Assume fixed-width for .txt files
            '.dat': 'fixed_width',  # Common for mainframe data files
        }
        
        detected_format = extension_map.get(file_extension, 'csv')
        self.logger.info("Format detected from extension", file_path=file_path, format=detected_format)
        
        return detected_format
    
    def _parse_llm_response(self, response: str) -> AgentResult:
        """
        Parse LLM response to extract structured data.
        
        Args:
            response: LLM response string
            
        Returns:
            AgentResult: Parsed data result
        """
        try:
            # Try to parse as JSON
            if response.strip().startswith('{'):
                parsed_response = json.loads(response)
                
                if parsed_response.get("success", False):
                    return AgentResult(
                        success=True,
                        data=parsed_response.get("data", []),
                        metadata=parsed_response.get("metadata", {})
                    )
                else:
                    return AgentResult(
                        success=False,
                        errors=[parsed_response.get("error", "Unknown error from LLM")]
                    )
            
            # If not JSON, try to extract JSON from the response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                parsed_response = json.loads(json_str)
                
                if parsed_response.get("success", False):
                    return AgentResult(
                        success=True,
                        data=parsed_response.get("data", []),
                        metadata=parsed_response.get("metadata", {})
                    )
            
            # If no valid JSON found, return error
            return AgentResult(
                success=False,
                errors=["Could not parse LLM response as valid JSON"]
            )
            
        except json.JSONDecodeError as e:
            self.logger.error("JSON parsing error", error=str(e))
            return AgentResult(
                success=False,
                errors=[f"JSON parsing error: {str(e)}"]
            )
        except Exception as e:
            self.logger.error("Error parsing LLM response", error=str(e))
            return AgentResult(
                success=False,
                errors=[f"Response parsing error: {str(e)}"]
            )
    

    

