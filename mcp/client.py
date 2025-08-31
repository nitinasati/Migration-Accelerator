"""
MCP (Model Context Protocol) client for the Migration-Accelerators platform.
"""

import asyncio
import json
from typing import Any, Dict, List, Optional, Union
import httpx
import structlog

from config.settings import MCPConfig


class MCPTool:
    """Base class for MCP tools."""
    
    def __init__(self, tool_name: str, description: str = ""):
        self.tool_name = tool_name
        self.description = description
        self.logger = structlog.get_logger().bind(tool=tool_name)
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the tool with given parameters.
        
        Args:
            parameters: Tool parameters
            
        Returns:
            Dict[str, Any]: Tool execution result
        """
        raise NotImplementedError("Tool execution must be implemented by subclasses")


class MCPClient:
    """MCP client for API interactions."""
    
    def __init__(self, config: MCPConfig):
        self.config = config
        self.logger = structlog.get_logger().bind(component="mcp_client")
        self.tools: Dict[str, MCPTool] = {}
        self._client: Optional[httpx.AsyncClient] = None
    
    async def initialize(self) -> None:
        """Initialize the MCP client."""
        try:
            self._client = httpx.AsyncClient(
                timeout=self.config.timeout,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.config.api_key}" if self.config.api_key else ""
                }
            )
            
            # Register default tools
            await self._register_default_tools()
            
            self.logger.info("MCP client initialized", server_url=self.config.server_url)
            
        except Exception as e:
            self.logger.error("Failed to initialize MCP client", error=str(e))
            raise
    
    async def _register_default_tools(self) -> None:
        """Register default MCP tools."""
        # API Call Tool
        api_tool = APICallTool()
        self.register_tool(api_tool)
        
        # Data Validation Tool
        validation_tool = DataValidationTool()
        self.register_tool(validation_tool)
        
        # File Processing Tool
        file_tool = FileProcessingTool()
        self.register_tool(file_tool)
        
        self.logger.info("Default MCP tools registered", tool_count=len(self.tools))
    
    def register_tool(self, tool: MCPTool) -> None:
        """
        Register an MCP tool.
        
        Args:
            tool: Tool to register
        """
        self.tools[tool.tool_name] = tool
        self.logger.info("MCP tool registered", tool_name=tool.tool_name)
    
    async def call_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        retries: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Call an MCP tool.
        
        Args:
            tool_name: Name of the tool to call
            parameters: Tool parameters
            retries: Number of retries (uses config default if None)
            
        Returns:
            Dict[str, Any]: Tool execution result
        """
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        tool = self.tools[tool_name]
        max_retries = retries or self.config.max_retries
        
        for attempt in range(max_retries + 1):
            try:
                self.logger.info(
                    "Calling MCP tool",
                    tool_name=tool_name,
                    attempt=attempt + 1,
                    parameters=parameters
                )
                
                result = await tool.execute(parameters)
                
                self.logger.info(
                    "MCP tool call successful",
                    tool_name=tool_name,
                    result_keys=list(result.keys()) if isinstance(result, dict) else None
                )
                
                return result
                
            except Exception as e:
                if attempt == max_retries:
                    self.logger.error(
                        "MCP tool call failed after all retries",
                        tool_name=tool_name,
                        error=str(e)
                    )
                    raise
                
                delay = 2 ** attempt
                self.logger.warning(
                    "MCP tool call failed, retrying",
                    tool_name=tool_name,
                    attempt=attempt + 1,
                    delay=delay,
                    error=str(e)
                )
                
                await asyncio.sleep(delay)
    
    async def make_api_call(
        self,
        method: str,
        url: str,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make an API call using MCP.
        
        Args:
            method: HTTP method
            url: API endpoint URL
            data: Request data
            headers: Additional headers
            params: Query parameters
            
        Returns:
            Dict[str, Any]: API response
        """
        parameters = {
            "method": method.upper(),
            "url": url,
            "data": data,
            "headers": headers,
            "params": params
        }
        
        return await self.call_tool("api_call", parameters)
    
    async def validate_data(
        self,
        data: Any,
        schema: Dict[str, Any],
        validation_rules: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Validate data using MCP.
        
        Args:
            data: Data to validate
            schema: Validation schema
            validation_rules: Additional validation rules
            
        Returns:
            Dict[str, Any]: Validation result
        """
        parameters = {
            "data": data,
            "schema": schema,
            "validation_rules": validation_rules
        }
        
        return await self.call_tool("data_validation", parameters)
    
    async def process_file(
        self,
        file_path: str,
        operation: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process file using MCP.
        
        Args:
            file_path: Path to the file
            operation: Operation to perform
            parameters: Additional parameters
            
        Returns:
            Dict[str, Any]: Processing result
        """
        params = {
            "file_path": file_path,
            "operation": operation,
            "parameters": parameters or {}
        }
        
        return await self.call_tool("file_processing", params)
    
    async def health_check(self) -> bool:
        """
        Check MCP client health.
        
        Returns:
            bool: True if healthy, False otherwise
        """
        try:
            if not self._client:
                return False
            
            # Try to make a simple API call
            response = await self._client.get(f"{self.config.server_url}/health")
            return response.status_code == 200
            
        except Exception as e:
            self.logger.error("MCP health check failed", error=str(e))
            return False
    
    async def close(self) -> None:
        """Close the MCP client."""
        if self._client:
            await self._client.aclose()
            self.logger.info("MCP client closed")


class APICallTool(MCPTool):
    """Tool for making API calls."""
    
    def __init__(self):
        super().__init__("api_call", "Make HTTP API calls")
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute API call."""
        try:
            method = parameters.get("method", "GET").upper()
            url = parameters["url"]
            data = parameters.get("data")
            headers = parameters.get("headers", {})
            params = parameters.get("params")
            
            async with httpx.AsyncClient() as client:
                response = await client.request(
                    method=method,
                    url=url,
                    json=data,
                    headers=headers,
                    params=params
                )
                
                return {
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "data": response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text,
                    "success": 200 <= response.status_code < 300
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "status_code": 500
            }


class DataValidationTool(MCPTool):
    """Tool for data validation."""
    
    def __init__(self):
        super().__init__("data_validation", "Validate data against schema and rules")
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data validation."""
        try:
            data = parameters["data"]
            schema = parameters["schema"]
            validation_rules = parameters.get("validation_rules", {})
            
            errors = []
            warnings = []
            
            # Basic validation logic
            if isinstance(data, list):
                for i, record in enumerate(data):
                    record_errors = self._validate_record(record, schema, validation_rules)
                    if record_errors:
                        errors.extend([f"Record {i}: {error}" for error in record_errors])
            else:
                errors = self._validate_record(data, schema, validation_rules)
            
            return {
                "success": len(errors) == 0,
                "errors": errors,
                "warnings": warnings,
                "validated_count": len(data) if isinstance(data, list) else 1
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "errors": [str(e)]
            }
    
    def _validate_record(self, record: Dict[str, Any], schema: Dict[str, Any], rules: Dict[str, Any]) -> List[str]:
        """Validate a single record."""
        errors = []
        
        # Check required fields
        required_fields = schema.get("required", [])
        for field in required_fields:
            if field not in record or record[field] is None or record[field] == "":
                errors.append(f"Required field '{field}' is missing or empty")
        
        # Check field types and patterns
        properties = schema.get("properties", {})
        for field, value in record.items():
            if field in properties:
                field_schema = properties[field]
                
                # Type validation
                expected_type = field_schema.get("type")
                if expected_type == "string" and not isinstance(value, str):
                    errors.append(f"Field '{field}' should be string, got {type(value).__name__}")
                elif expected_type == "number" and not isinstance(value, (int, float)):
                    errors.append(f"Field '{field}' should be number, got {type(value).__name__}")
                elif expected_type == "integer" and not isinstance(value, int):
                    errors.append(f"Field '{field}' should be integer, got {type(value).__name__}")
                
                # Pattern validation
                pattern = field_schema.get("pattern")
                if pattern and isinstance(value, str):
                    import re
                    if not re.match(pattern, value):
                        errors.append(f"Field '{field}' does not match pattern '{pattern}'")
                
                # Range validation
                if isinstance(value, (int, float)):
                    min_val = field_schema.get("minimum")
                    max_val = field_schema.get("maximum")
                    if min_val is not None and value < min_val:
                        errors.append(f"Field '{field}' value {value} is below minimum {min_val}")
                    if max_val is not None and value > max_val:
                        errors.append(f"Field '{field}' value {value} is above maximum {max_val}")
        
        return errors


class FileProcessingTool(MCPTool):
    """Tool for file processing operations."""
    
    def __init__(self):
        super().__init__("file_processing", "Process files for migration")
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute file processing."""
        try:
            file_path = parameters["file_path"]
            operation = parameters["operation"]
            op_params = parameters.get("parameters", {})
            
            if operation == "read":
                return await self._read_file(file_path, op_params)
            elif operation == "write":
                return await self._write_file(file_path, op_params)
            elif operation == "validate":
                return await self._validate_file(file_path, op_params)
            else:
                return {
                    "success": False,
                    "error": f"Unknown operation: {operation}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _read_file(self, file_path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Read file operation."""
        try:
            import aiofiles
            
            async with aiofiles.open(file_path, 'r', encoding=params.get("encoding", "utf-8")) as file:
                content = await file.read()
            
            return {
                "success": True,
                "content": content,
                "file_path": file_path
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _write_file(self, file_path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Write file operation."""
        try:
            import aiofiles
            
            content = params.get("content", "")
            encoding = params.get("encoding", "utf-8")
            
            async with aiofiles.open(file_path, 'w', encoding=encoding) as file:
                await file.write(content)
            
            return {
                "success": True,
                "file_path": file_path,
                "bytes_written": len(content.encode(encoding))
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _validate_file(self, file_path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate file operation."""
        try:
            from pathlib import Path
            
            path = Path(file_path)
            
            if not path.exists():
                return {
                    "success": False,
                    "error": "File does not exist"
                }
            
            return {
                "success": True,
                "file_path": file_path,
                "exists": True,
                "size": path.stat().st_size,
                "is_file": path.is_file(),
                "is_readable": path.stat().st_mode & 0o444
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


class MCPToolManager:
    """Manager for MCP tools and client."""
    
    def __init__(self, config: MCPConfig):
        self.config = config
        self.client = MCPClient(config)
        self.logger = structlog.get_logger().bind(component="mcp_manager")
    
    async def initialize(self) -> None:
        """Initialize the MCP tool manager."""
        await self.client.initialize()
        self.logger.info("MCP tool manager initialized")
    
    async def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Call an MCP tool."""
        return await self.client.call_tool(tool_name, parameters)
    
    async def make_api_call(
        self,
        method: str,
        url: str,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Make an API call."""
        return await self.client.make_api_call(method, url, data, headers)
    
    async def health_check(self) -> bool:
        """Check MCP health."""
        return await self.client.health_check()
    
    async def close(self) -> None:
        """Close the MCP manager."""
        await self.client.close()
        self.logger.info("MCP tool manager closed")
