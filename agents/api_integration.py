"""
API Integration Agent for the Migration-Accelerators platform.
"""

import asyncio
from typing import Any, Dict, List, Optional, Union
import structlog

from .base_agent import BaseAgent, AgentResult
from config.settings import LLMConfig, MCPConfig
from llm.providers import BaseLLMProvider, LLMProviderFactory
from llm.prompts import get_prompt, get_system_prompt
from mcp_tools.client import MCPToolManager


class APIIntegrationAgent(BaseAgent):
    """
    API Integration Agent for making API calls to target systems.
    
    This agent handles:
    - Making API calls to target systems
    - Handling authentication and authorization
    - Managing request/response processing
    - Implementing retry logic and error handling
    - Monitoring API performance and health
    """
    
    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        mcp_config: Optional[MCPConfig] = None
    ):
        super().__init__("api_integration", llm_config, mcp_config)
        self.llm_provider: Optional[BaseLLMProvider] = None
        self.mcp_manager: Optional[MCPToolManager] = None
        self.api_endpoints: Dict[str, Dict[str, Any]] = {}
        self.authentication_cache: Dict[str, Any] = {}
    
    async def initialize(self) -> None:
        """Initialize the API integration agent."""
        await super().start()
        
        if self.llm_config:
            try:
                self.llm_provider = LLMProviderFactory.create(self.llm_config, self.agent_name)
                await self.llm_provider.initialize()
                self.logger.info("LLM provider initialized for API integration agent")
            except ImportError as e:
                self.logger.warning(f"LLM provider not available: {e}")
                self.logger.info("Running without LLM enhancements")
                self.llm_provider = None
            except Exception as e:
                self.logger.error(f"Failed to initialize LLM provider: {e}")
                self.logger.info("Running without LLM enhancements")
                self.llm_provider = None
        
        if self.mcp_config:
            self.mcp_manager = MCPToolManager(self.mcp_config)
            await self.mcp_manager.initialize()
            self.logger.info("MCP manager initialized for API integration agent")
        
        # Initialize default API endpoints
        await self._initialize_default_endpoints()
        
        self.logger.info("API integration agent initialized")
    
    async def _initialize_default_endpoints(self) -> None:
        """Initialize default API endpoints for insurance systems."""
        self.api_endpoints = {
            "disability_policy": {
                "url": "/api/v1/policies/disability",
                "method": "POST",
                "content_type": "application/json",
                "authentication": "bearer_token"
            },
            "absence_record": {
                "url": "/api/v1/absences",
                "method": "POST",
                "content_type": "application/json",
                "authentication": "bearer_token"
            },
            "group_policy": {
                "url": "/api/v1/policies/group",
                "method": "POST",
                "content_type": "application/json",
                "authentication": "bearer_token"
            },
            "employee": {
                "url": "/api/v1/employees",
                "method": "POST",
                "content_type": "application/json",
                "authentication": "bearer_token"
            },
            "claim": {
                "url": "/api/v1/claims",
                "method": "POST",
                "content_type": "application/json",
                "authentication": "bearer_token"
            }
        }
    
    async def process(self, data: Any, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """
        Process API integration request.
        
        Args:
            data: Data to send to API (record or list of records)
            context: API context (endpoint, authentication, batch_size, etc.)
            
        Returns:
            AgentResult: API integration result
        """
        try:
            self.logger.info("Starting API integration process", data_type=type(data).__name__)
            
            # Validate input
            if not await self.validate_input(data):
                return AgentResult(
                    success=False,
                    errors=["Invalid input data for API integration"]
                )
            
            # Get API context
            endpoint_name = context.get("endpoint", "disability_policy") if context else "disability_policy"
            base_url = context.get("base_url", "https://api.insurance-system.com") if context else "https://api.insurance-system.com"
            batch_size = context.get("batch_size", 10) if context else 10
            authentication = context.get("authentication", {}) if context else {}
            output_mode = context.get("output_mode", "api") if context else "api"  # "api" or "file"
            output_dir = context.get("output_dir", "data/output") if context else "data/output"
            
            # Get endpoint configuration
            endpoint_config = self.api_endpoints.get(endpoint_name)
            if not endpoint_config:
                return AgentResult(
                    success=False,
                    errors=[f"Unknown endpoint: {endpoint_name}"]
                )
            
            # Process data based on output mode
            if output_mode == "file":
                # Write to JSON file instead of making API calls
                api_results = await self._write_to_file(
                    data, endpoint_name, output_dir, context
                )
            else:
                # Make API calls (original behavior)
                if isinstance(data, list):
                    api_results = await self._process_batch(
                        data, endpoint_config, base_url, batch_size, authentication
                    )
                else:
                    api_results = await self._process_single_record(
                        data, endpoint_config, base_url, authentication
                    )
            
            # Analyze results
            success_count = sum(1 for result in api_results if result.get("success", False))
            error_count = len(api_results) - success_count
            
            success = error_count == 0
            
            self.logger.info(
                "API integration completed",
                success=success,
                total_records=len(data) if isinstance(data, list) else 1,
                success_count=success_count,
                error_count=error_count
            )
            
            return AgentResult(
                success=success,
                data={
                    "api_results": api_results,
                    "summary": {
                        "total_records": len(data) if isinstance(data, list) else 1,
                        "success_count": success_count,
                        "error_count": error_count,
                        "success_rate": (success_count / len(api_results)) * 100 if api_results else 0
                    }
                },
                errors=[result.get("error") for result in api_results if not result.get("success", False)],
                metadata={
                    "endpoint": endpoint_name,
                    "base_url": base_url,
                    "batch_size": batch_size,
                    "agent": self.agent_name
                }
            )
            
        except Exception as e:
            return await self.handle_error(e, {"data_type": type(data).__name__, "context": context})
    
    async def _process_batch(
        self,
        records: List[Dict[str, Any]],
        endpoint_config: Dict[str, Any],
        base_url: str,
        batch_size: int,
        authentication: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Process a batch of records."""
        api_results = []
        
        # Process in batches
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            
            try:
                batch_result = await self._process_batch_chunk(
                    batch, endpoint_config, base_url, authentication
                )
                api_results.extend(batch_result)
                
                self.logger.info("Batch processed", batch_index=i // batch_size, records_count=len(batch))
                
                # Add delay between batches to avoid rate limiting
                if i + batch_size < len(records):
                    await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error("Batch processing failed", batch_index=i // batch_size, error=str(e))
                # Add error results for this batch
                for record in batch:
                    api_results.append({
                        "success": False,
                        "error": str(e),
                        "record": record
                    })
        
        return api_results
    
    async def _process_batch_chunk(
        self,
        batch: List[Dict[str, Any]],
        endpoint_config: Dict[str, Any],
        base_url: str,
        authentication: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Process a single batch chunk."""
        results = []
        
        # Prepare batch request
        batch_data = {
            "records": batch,
            "batch_id": f"batch_{asyncio.get_event_loop().time()}",
            "total_records": len(batch)
        }
        
        # Make API call
        api_result = await self._make_api_call(
            endpoint_config, base_url, batch_data, authentication
        )
        
        # Process batch response
        if api_result.get("success", False):
            response_data = api_result.get("data", {})
            individual_results = response_data.get("results", [])
            
            # Ensure we have results for each record
            for i, record in enumerate(batch):
                if i < len(individual_results):
                    results.append(individual_results[i])
                else:
                    results.append({
                        "success": True,
                        "record": record,
                        "message": "Processed in batch"
                    })
        else:
            # All records in batch failed
            for record in batch:
                results.append({
                    "success": False,
                    "error": api_result.get("error", "Batch processing failed"),
                    "record": record
                })
        
        return results
    
    async def _process_single_record(
        self,
        record: Dict[str, Any],
        endpoint_config: Dict[str, Any],
        base_url: str,
        authentication: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Process a single record."""
        api_result = await self._make_api_call(
            endpoint_config, base_url, record, authentication
        )
        
        return [api_result]
    
    async def _make_api_call(
        self,
        endpoint_config: Dict[str, Any],
        base_url: str,
        data: Dict[str, Any],
        authentication: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Make an API call using MCP."""
        try:
            # Prepare request
            url = f"{base_url}{endpoint_config['url']}"
            method = endpoint_config.get("method", "POST")
            content_type = endpoint_config.get("content_type", "application/json")
            
            # Prepare headers
            headers = {
                "Content-Type": content_type,
                "Accept": "application/json"
            }
            
            # Add authentication
            auth_type = endpoint_config.get("authentication", "bearer_token")
            if auth_type == "bearer_token" and authentication.get("token"):
                headers["Authorization"] = f"Bearer {authentication['token']}"
            elif auth_type == "api_key" and authentication.get("api_key"):
                headers["X-API-Key"] = authentication["api_key"]
            
            # Make API call using MCP
            if self.mcp_manager:
                response = await self.mcp_manager.make_api_call(
                    method=method,
                    url=url,
                    data=data,
                    headers=headers
                )
                
                return {
                    "success": response.get("success", False),
                    "status_code": response.get("status_code"),
                    "data": response.get("data"),
                    "error": response.get("error"),
                    "headers": response.get("headers", {})
                }
            else:
                return {
                    "success": False,
                    "error": "MCP manager not available"
                }
                
        except Exception as e:
            self.logger.error("API call failed", error=str(e))
            return {
                "success": False,
                "error": str(e)
            }
    
    async def authenticate(self, auth_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Authenticate with the target system.
        
        Args:
            auth_config: Authentication configuration
            
        Returns:
            Dict[str, Any]: Authentication result
        """
        try:
            auth_type = auth_config.get("type", "bearer_token")
            
            if auth_type == "bearer_token":
                return await self._authenticate_bearer_token(auth_config)
            elif auth_type == "api_key":
                return await self._authenticate_api_key(auth_config)
            elif auth_type == "oauth2":
                return await self._authenticate_oauth2(auth_config)
            else:
                return {
                    "success": False,
                    "error": f"Unsupported authentication type: {auth_type}"
                }
                
        except Exception as e:
            self.logger.error("Authentication failed", error=str(e))
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _authenticate_bearer_token(self, auth_config: Dict[str, Any]) -> Dict[str, Any]:
        """Authenticate using bearer token."""
        try:
            username = auth_config.get("username")
            password = auth_config.get("password")
            token_url = auth_config.get("token_url")
            
            if not all([username, password, token_url]):
                return {
                    "success": False,
                    "error": "Missing required authentication parameters"
                }
            
            # Make authentication request
            auth_data = {
                "username": username,
                "password": password,
                "grant_type": "password"
            }
            
            if self.mcp_manager:
                response = await self.mcp_manager.make_api_call(
                    method="POST",
                    url=token_url,
                    data=auth_data,
                    headers={"Content-Type": "application/x-www-form-urlencoded"}
                )
                
                if response.get("success", False):
                    token_data = response.get("data", {})
                    access_token = token_data.get("access_token")
                    
                    if access_token:
                        # Cache token
                        self.authentication_cache["bearer_token"] = {
                            "token": access_token,
                            "expires_at": token_data.get("expires_at"),
                            "token_type": token_data.get("token_type", "Bearer")
                        }
                        
                        return {
                            "success": True,
                            "token": access_token,
                            "expires_at": token_data.get("expires_at")
                        }
                
                return {
                    "success": False,
                    "error": "Failed to obtain access token"
                }
            else:
                return {
                    "success": False,
                    "error": "MCP manager not available"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _authenticate_api_key(self, auth_config: Dict[str, Any]) -> Dict[str, Any]:
        """Authenticate using API key."""
        api_key = auth_config.get("api_key")
        
        if not api_key:
            return {
                "success": False,
                "error": "API key not provided"
            }
        
        # Cache API key
        self.authentication_cache["api_key"] = {
            "api_key": api_key
        }
        
        return {
            "success": True,
            "api_key": api_key
        }
    
    async def _authenticate_oauth2(self, auth_config: Dict[str, Any]) -> Dict[str, Any]:
        """Authenticate using OAuth2."""
        # This is a simplified OAuth2 implementation
        # In a real system, you would implement the full OAuth2 flow
        
        client_id = auth_config.get("client_id")
        client_secret = auth_config.get("client_secret")
        token_url = auth_config.get("token_url")
        
        if not all([client_id, client_secret, token_url]):
            return {
                "success": False,
                "error": "Missing OAuth2 configuration"
            }
        
        # Make OAuth2 token request
        oauth_data = {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret
        }
        
        if self.mcp_manager:
            response = await self.mcp_manager.make_api_call(
                method="POST",
                url=token_url,
                data=oauth_data,
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
            
            if response.get("success", False):
                token_data = response.get("data", {})
                access_token = token_data.get("access_token")
                
                if access_token:
                    # Cache token
                    self.authentication_cache["oauth2"] = {
                        "token": access_token,
                        "expires_at": token_data.get("expires_at"),
                        "token_type": token_data.get("token_type", "Bearer")
                    }
                    
                    return {
                        "success": True,
                        "token": access_token,
                        "expires_at": token_data.get("expires_at")
                    }
            
            return {
                "success": False,
                "error": "OAuth2 authentication failed"
            }
        else:
            return {
                "success": False,
                "error": "MCP manager not available"
            }
    
    async def health_check(self) -> bool:
        """Check API health."""
        try:
            if not self.mcp_manager:
                return False
            
            return await self.mcp_manager.health_check()
            
        except Exception as e:
            self.logger.error("API health check failed", error=str(e))
            return False
    
    async def get_api_status(self, base_url: str) -> Dict[str, Any]:
        """
        Get API status information.
        
        Args:
            base_url: Base URL of the API
            
        Returns:
            Dict[str, Any]: API status information
        """
        try:
            if not self.mcp_manager:
                return {
                    "success": False,
                    "error": "MCP manager not available"
                }
            
            # Make health check request
            response = await self.mcp_manager.make_api_call(
                method="GET",
                url=f"{base_url}/health",
                headers={"Accept": "application/json"}
            )
            
            return {
                "success": response.get("success", False),
                "status_code": response.get("status_code"),
                "data": response.get("data"),
                "error": response.get("error")
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def register_endpoint(
        self,
        name: str,
        url: str,
        method: str = "POST",
        content_type: str = "application/json",
        authentication: str = "bearer_token"
    ) -> None:
        """
        Register a new API endpoint.
        
        Args:
            name: Endpoint name
            url: Endpoint URL
            method: HTTP method
            content_type: Content type
            authentication: Authentication type
        """
        self.api_endpoints[name] = {
            "url": url,
            "method": method,
            "content_type": content_type,
            "authentication": authentication
        }
        
        self.logger.info("API endpoint registered", endpoint_name=name, url=url)
    
    async def _write_to_file(
        self,
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
        endpoint_name: str,
        output_dir: str,
        context: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Write data to JSON file instead of making API calls.
        
        Args:
            data: Data to write (single record or list of records)
            endpoint_name: Name of the endpoint (used for filename)
            output_dir: Directory to write the file to
            context: Additional context information
            
        Returns:
            List[Dict[str, Any]]: List of results indicating success/failure
        """
        try:
            import json
            import os
            from datetime import datetime
            
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{endpoint_name}_{timestamp}.json"
            filepath = os.path.join(output_dir, filename)
            
            # Prepare data for writing
            if isinstance(data, str):
                # If data is already a JSON string, parse it first
                try:
                    json_data = json.loads(data)
                except json.JSONDecodeError:
                    json_data = {"raw_data": data}
            else:
                json_data = data
            
            # Add metadata
            output_data = {
                "metadata": {
                    "endpoint": endpoint_name,
                    "timestamp": datetime.now().isoformat(),
                    "record_count": len(json_data) if isinstance(json_data, list) else 1,
                    "output_mode": "file",
                    "file_path": filepath
                },
                "data": json_data
            }
            
            # Write to file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(
                "Data written to file successfully",
                filepath=filepath,
                record_count=len(json_data) if isinstance(json_data, list) else 1
            )
            
            # Return success result
            return [{
                "success": True,
                "file_path": filepath,
                "record_count": len(json_data) if isinstance(json_data, list) else 1,
                "message": f"Data successfully written to {filepath}"
            }]
            
        except Exception as e:
            self.logger.error("Failed to write data to file", error=str(e))
            return [{
                "success": False,
                "error": str(e),
                "message": f"Failed to write data to file: {str(e)}"
            }]
    
    async def close(self) -> None:
        """Close the API integration agent."""
        if self.mcp_manager:
            await self.mcp_manager.close()
        
        await super().stop()
        self.logger.info("API integration agent closed")
