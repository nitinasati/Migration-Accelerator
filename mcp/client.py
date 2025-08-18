"""
MCP (Model Context Protocol) client for API interactions.
"""

import asyncio
import json
import httpx
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod

from config.settings import MCPConfig


class MCPClient:
    """MCP client for interacting with MCP servers."""
    
    def __init__(self, config: MCPConfig):
        self.config = config
        self.base_url = config.server_url
        self.api_key = config.api_key
        self.timeout = config.timeout
        self.max_retries = config.max_retries
        
        # Initialize HTTP client
        self.client = httpx.AsyncClient(
            timeout=self.timeout,
            headers=self._get_headers()
        )
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        return headers
    
    async def initialize(self) -> Dict[str, Any]:
        """Initialize MCP connection."""
        try:
            response = await self.client.post(
                f"{self.base_url}/initialize",
                json={"protocol_version": "2024-11-05"}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise Exception(f"MCP initialization failed: {str(e)}")
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available MCP tools."""
        try:
            response = await self.client.get(f"{self.base_url}/tools")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise Exception(f"Failed to list MCP tools: {str(e)}")
    
    async def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Call an MCP tool."""
        try:
            payload = {
                "tool": tool_name,
                "parameters": parameters
            }
            
            response = await self.client.post(
                f"{self.base_url}/call",
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise Exception(f"MCP tool call failed: {str(e)}")
    
    async def call_tool_with_retry(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Call an MCP tool with retry logic."""
        for attempt in range(self.max_retries):
            try:
                return await self.call_tool(tool_name, parameters)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise e
                
                # Exponential backoff
                await asyncio.sleep(2 ** attempt)
        
        raise Exception("Max retries exceeded")
    
    async def close(self):
        """Close the MCP client."""
        await self.client.aclose()


class MCPTool(ABC):
    """Base class for MCP tools."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
    
    @abstractmethod
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool with given parameters."""
        pass
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.get_parameters_schema()
        }
    
    @abstractmethod
    def get_parameters_schema(self) -> Dict[str, Any]:
        """Get parameters schema."""
        pass


class InsuranceAPITool(MCPTool):
    """Insurance API tool for MCP."""
    
    def __init__(self):
        super().__init__(
            name="insurance_api",
            description="Insurance API operations for policy management"
        )
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute insurance API operation."""
        operation = parameters.get("operation")
        
        if operation == "create_policy":
            return await self._create_policy(parameters)
        elif operation == "update_policy":
            return await self._update_policy(parameters)
        elif operation == "get_policy":
            return await self._get_policy(parameters)
        elif operation == "delete_policy":
            return await self._delete_policy(parameters)
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    async def _create_policy(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new policy."""
        policy_data = parameters.get("policy_data", {})
        
        # Simulate API call
        await asyncio.sleep(0.1)  # Simulate network delay
        
        return {
            "success": True,
            "policy_id": f"POL{len(policy_data.get('policyId', ''))}",
            "message": "Policy created successfully",
            "data": policy_data
        }
    
    async def _update_policy(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing policy."""
        policy_id = parameters.get("policy_id")
        updates = parameters.get("updates", {})
        
        # Simulate API call
        await asyncio.sleep(0.1)
        
        return {
            "success": True,
            "policy_id": policy_id,
            "message": "Policy updated successfully",
            "data": updates
        }
    
    async def _get_policy(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Get policy details."""
        policy_id = parameters.get("policy_id")
        
        # Simulate API call
        await asyncio.sleep(0.1)
        
        return {
            "success": True,
            "policy_id": policy_id,
            "data": {
                "policyId": policy_id,
                "status": "ACTIVE",
                "effectiveDate": "2023-01-01",
                "benefitAmount": 50000.0
            }
        }
    
    async def _delete_policy(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Delete a policy."""
        policy_id = parameters.get("policy_id")
        
        # Simulate API call
        await asyncio.sleep(0.1)
        
        return {
            "success": True,
            "policy_id": policy_id,
            "message": "Policy deleted successfully"
        }
    
    def get_parameters_schema(self) -> Dict[str, Any]:
        """Get parameters schema."""
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["create_policy", "update_policy", "get_policy", "delete_policy"],
                    "description": "The operation to perform"
                },
                "policy_id": {
                    "type": "string",
                    "description": "Policy ID for get, update, and delete operations"
                },
                "policy_data": {
                    "type": "object",
                    "description": "Policy data for create operation"
                },
                "updates": {
                    "type": "object",
                    "description": "Updates for update operation"
                }
            },
            "required": ["operation"]
        }


class DataValidationTool(MCPTool):
    """Data validation tool for MCP."""
    
    def __init__(self):
        super().__init__(
            name="data_validation",
            description="Data validation and quality assessment"
        )
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data validation."""
        data = parameters.get("data", {})
        validation_rules = parameters.get("validation_rules", {})
        
        # Perform validation
        validation_result = await self._validate_data(data, validation_rules)
        
        return {
            "success": True,
            "validation_result": validation_result
        }
    
    async def _validate_data(self, data: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data against rules."""
        errors = []
        warnings = []
        
        for field, rule in rules.items():
            if field in data:
                value = data[field]
                
                # Check required
                if rule.get("required", False) and not value:
                    errors.append(f"Field '{field}' is required")
                
                # Check pattern
                if "pattern" in rule and value:
                    import re
                    if not re.match(rule["pattern"], str(value)):
                        errors.append(f"Field '{field}' does not match pattern")
                
                # Check min/max length
                if "min_length" in rule and len(str(value)) < rule["min_length"]:
                    errors.append(f"Field '{field}' is too short")
                
                if "max_length" in rule and len(str(value)) > rule["max_length"]:
                    errors.append(f"Field '{field}' is too long")
                
                # Check min/max value
                if "min_value" in rule and value is not None:
                    try:
                        if float(value) < rule["min_value"]:
                            errors.append(f"Field '{field}' value is too low")
                    except (ValueError, TypeError):
                        pass
                
                if "max_value" in rule and value is not None:
                    try:
                        if float(value) > rule["max_value"]:
                            errors.append(f"Field '{field}' value is too high")
                    except (ValueError, TypeError):
                        pass
        
        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "score": max(0, 1 - len(errors) / len(rules)) if rules else 1.0
        }
    
    def get_parameters_schema(self) -> Dict[str, Any]:
        """Get parameters schema."""
        return {
            "type": "object",
            "properties": {
                "data": {
                    "type": "object",
                    "description": "Data to validate"
                },
                "validation_rules": {
                    "type": "object",
                    "description": "Validation rules to apply"
                }
            },
            "required": ["data", "validation_rules"]
        }


class MCPToolManager:
    """Manager for MCP tools."""
    
    def __init__(self, config: MCPConfig):
        self.config = config
        self.client = MCPClient(config)
        self.tools = {}
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register default tools."""
        self.register_tool(InsuranceAPITool())
        self.register_tool(DataValidationTool())
    
    def register_tool(self, tool: MCPTool):
        """Register a new tool."""
        self.tools[tool.name] = tool
    
    async def initialize(self):
        """Initialize MCP connection."""
        await self.client.initialize()
    
    async def list_available_tools(self) -> List[Dict[str, Any]]:
        """List all available tools."""
        tool_schemas = []
        
        # Local tools
        for tool in self.tools.values():
            tool_schemas.append(tool.get_schema())
        
        # Remote tools (from MCP server)
        try:
            remote_tools = await self.client.list_tools()
            tool_schemas.extend(remote_tools)
        except Exception as e:
            print(f"Warning: Could not fetch remote tools: {e}")
        
        return tool_schemas
    
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool."""
        # Check if it's a local tool
        if tool_name in self.tools:
            tool = self.tools[tool_name]
            return await tool.execute(parameters)
        
        # Try remote tool
        try:
            return await self.client.call_tool_with_retry(tool_name, parameters)
        except Exception as e:
            raise Exception(f"Tool execution failed: {str(e)}")
    
    async def close(self):
        """Close the tool manager."""
        await self.client.close()
