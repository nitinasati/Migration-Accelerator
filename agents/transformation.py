"""
Transformation Agent for the Migration-Accelerators platform.
"""

import json
from typing import Any, Dict, List, Optional
import structlog

from .base_agent import BaseAgent, AgentResult
from config.settings import LLMConfig, MCPConfig
from llm.providers import BaseLLMProvider, LLMProviderFactory
from llm.prompts import get_prompt, get_system_prompt


class TransformationAgent(BaseAgent):
    """
    Transformation Agent for LLM-based data transformation.
    
    This agent handles:
    - LLM-powered data transformation using mapping configurations
    - Intelligent field mapping and data conversion
    - Business logic application
    - Data structure optimization
    """
    
    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        mcp_config: Optional[MCPConfig] = None
    ):
        super().__init__("transformation", llm_config, mcp_config)
        self.llm_provider: Optional[BaseLLMProvider] = None
    
    async def initialize(self) -> None:
        """Initialize the transformation agent."""
        await super().start()
        
        if self.llm_config:
            self.llm_provider = LLMProviderFactory.create(self.llm_config, self.agent_name)
            await self.llm_provider.initialize()
            self.logger.info("LLM provider initialized for transformation agent")
        
        self.logger.info("Transformation agent initialized")
    
    async def process(self, data: Any, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """
        Process transformation request using LLM.
        
        Args:
            data: Data to transform
            context: Transformation context (mapping_config, record_type, etc.)
            
        Returns:
            AgentResult: Transformation result
        """
        try:
            self.logger.info("Starting transformation process", data_type=type(data).__name__)
            
            # Validate input
            if not await self.validate_input(data):
                return AgentResult(
                    success=False,
                    errors=["Invalid input data for transformation"]
                )
            
            # Check if LLM provider is available
            if not self.llm_provider:
                return AgentResult(
                    success=False,
                    errors=["LLM provider not available for transformation"]
                )
            
            # Get transformation context
            mapping_config = context.get("mapping_config") if context else None
            record_type = context.get("record_type", "unknown") if context else "unknown"
            
            if not mapping_config:
                return AgentResult(
                    success=False,
                    errors=["No mapping configuration provided for transformation"]
                )
            
            # Use LLM to transform data
            result = await self._transform_data_with_llm(data, mapping_config, record_type)
            
            self.logger.info(
                "Transformation completed",
                success=result.success,
                record_type=record_type,
                errors_count=len(result.errors) if result.errors else 0
            )
            
            return result
            
        except Exception as e:
            return await self.handle_error(e, {"data_type": type(data).__name__, "context": context})
    
    async def _transform_data_with_llm(
        self,
        data: List[Dict[str, Any]],
        mapping_config: Dict[str, Any],
        record_type: str
    ) -> AgentResult:
        """
        Use LLM to transform data using mapping configuration.
        
        Args:
            data: Source data to transform
            mapping_config: Mapping configuration to apply
            record_type: Type of record being transformed
            
        Returns:
            AgentResult with transformed data
        """
        try:
            # Create transformation prompt
            prompt = get_prompt(
                "mapping_transform_with_llm",
                source_data=json.dumps(data, indent=2),
                mapping_config=json.dumps(mapping_config, indent=2),
                record_type=record_type
            )
            
            system_prompt = get_system_prompt("mapping")
            
            # Get LLM response
            response = await self.llm_provider.generate(prompt, system_prompt)
            
            # Parse LLM response
            result = self._parse_transformation_response(response)
            
            # Add metadata
            result.metadata.update({
                "record_type": record_type,
                "agent": self.agent_name,
                "llm_processed": True
            })
            
            return result
            
        except Exception as e:
            self.logger.error("LLM data transformation failed", error=str(e))
            return AgentResult(
                success=False,
                errors=[f"LLM transformation error: {str(e)}"]
            )
    
    def _parse_transformation_response(self, response: str) -> AgentResult:
        """
        Parse LLM response for data transformation.
        
        Args:
            response: LLM response string
            
        Returns:
            AgentResult with parsed data
        """
        try:
            # Try to parse as JSON
            if response.strip().startswith('{'):
                parsed_response = json.loads(response)
                
                if parsed_response.get("success", False):
                    return AgentResult(
                        success=True,
                        data=parsed_response.get("transformed_data", []),
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
                        data=parsed_response.get("transformed_data", []),
                        metadata=parsed_response.get("metadata", {})
                    )
            
            # If no valid JSON found, return error
            return AgentResult(
                success=False,
                errors=["Could not parse LLM transformation response as valid JSON"]
            )
            
        except json.JSONDecodeError as e:
            self.logger.error("JSON parsing error in transformation response", error=str(e))
            return AgentResult(
                success=False,
                errors=[f"JSON parsing error: {str(e)}"]
            )
        except Exception as e:
            self.logger.error("Error parsing transformation response", error=str(e))
            return AgentResult(
                success=False,
                errors=[f"Response parsing error: {str(e)}"]
            )
