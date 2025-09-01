"""
Mapping Agent for the Migration-Accelerators platform.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
import structlog

from .base_agent import BaseAgent, AgentResult
from config.settings import LLMConfig, MCPConfig
from llm.providers import BaseLLMProvider, LLMProviderFactory
from llm.prompts import get_prompt, get_system_prompt


class MappingAgent(BaseAgent):
    """
    LLM-powered Mapping Agent for intelligent field transformation and data mapping.
    
    This agent uses LLM to:
    - Analyze input data attributes to determine record type
    - Select appropriate mapping configuration files
    - Apply intelligent field transformations
    - Handle data type conversions and business logic
    - Generate properly structured output data
    """
    
    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        mcp_config: Optional[MCPConfig] = None
    ):
        super().__init__("mapping", llm_config, mcp_config)
        self.llm_provider: Optional[BaseLLMProvider] = None
        self.mapping_configs: Dict[str, Any] = {}
    
    async def initialize(self) -> None:
        """Initialize the mapping agent."""
        await super().start()
        
        if self.llm_config:
            self.llm_provider = LLMProviderFactory.create(self.llm_config, self.agent_name)
            await self.llm_provider.initialize()
            self.logger.info("LLM provider initialized for mapping agent")
        
        # Load available mapping configurations
        await self._load_mapping_configs()
        
        self.logger.info("Mapping agent initialized")
    
    async def process(self, data: Any, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """
        Process mapping request using LLM intelligence.
        
        Args:
            data: List of records to map
            context: Additional context (optional mapping config, record type, etc.)
            
        Returns:
            AgentResult: Mapping result with transformed data
        """
        try:
            self.logger.info("Starting mapping process", data_type=type(data).__name__)
            
            # Validate input
            if not self.validate_input(data):
                return AgentResult(
                    success=False,
                    errors=["Invalid input data for mapping"]
                )
            
            # Check if LLM provider is available
            if not self.llm_provider:
                return AgentResult(
                    success=False,
                    errors=["LLM provider not available for intelligent mapping"]
                )
            
            # Use LLM to select appropriate mapping configuration
            selected_mapping = await self._select_mapping_config(data)
            if not selected_mapping:
                return AgentResult(
                    success=False,
                    errors=["Could not determine appropriate mapping configuration"]
                )
            
            # Get the actual mapping configuration
            mapping_name = selected_mapping.get("selected_mapping")
            mapping_config = self.mapping_configs.get(mapping_name)
            
            if not mapping_config:
                return AgentResult(
                    success=False,
                    errors=[f"Mapping configuration not found: {mapping_name}"]
                )
            
            # Create result with mapping configuration for the transformation step
            result = AgentResult(
                success=True,
                data=data,  # Return original data for transformation step
                metadata={
                    "mapping_config": mapping_config,
                    "record_type": selected_mapping.get("record_type", "unknown"),
                    "selected_mapping": mapping_name,
                    "agent": self.agent_name,
                    "llm_processed": True
                }
            )
            
            self.logger.info(
                "Mapping configuration selected",
                record_type=selected_mapping.get("record_type", "unknown"),
                selected_mapping=mapping_name,
                source_records=len(data) if isinstance(data, list) else 1
            )
            
            return result
            
        except Exception as e:
            return await self.handle_error(e, {"data_type": type(data).__name__, "context": context})
    
    async def _load_mapping_configs(self) -> None:
        """Load all available mapping configuration files."""
        try:
            mapping_dir = "config/mappings"
            if not os.path.exists(mapping_dir):
                self.logger.warning("Mapping directory not found", path=mapping_dir)
                return
            
            for filename in os.listdir(mapping_dir):
                if filename.endswith('.yaml') or filename.endswith('.yml'):
                    file_path = os.path.join(mapping_dir, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as file:
                            import yaml
                            config = yaml.safe_load(file)
                            config_name = filename.replace('.yaml', '').replace('.yml', '')
                            self.mapping_configs[config_name] = config
                            self.logger.info("Loaded mapping config", name=config_name)
                    except Exception as e:
                        self.logger.error("Error loading mapping config", file=filename, error=str(e))
            
            self.logger.info("Mapping configurations loaded", count=len(self.mapping_configs))
            
        except Exception as e:
            self.logger.error("Error loading mapping configurations", error=str(e))
    
    async def _select_mapping_config(self, data: Any) -> Optional[Dict[str, Any]]:
        """
        Use LLM to select the most appropriate mapping configuration.
        
        Args:
            data: Input data to analyze
            
        Returns:
            Dict containing selected mapping information
        """
        try:
            # Prepare data sample for analysis
            if isinstance(data, list) and len(data) > 0:
                data_sample = data[:2]  # Use first 2 records for analysis
            else:
                data_sample = [data] if data else []
            
            # Prepare available mappings list
            available_mappings = list(self.mapping_configs.keys())
            
            # Create mapping selection prompt
            prompt = get_prompt(
                "mapping_select_config",
                data_sample=json.dumps(data_sample, indent=2),
                available_mappings=json.dumps(available_mappings, indent=2)
            )
            
            system_prompt = get_system_prompt("mapping")
            
            # Get LLM response
            response = await self.llm_provider.generate(prompt, system_prompt)
            
            # Parse LLM response
            selected_mapping = await self._parse_mapping_selection(response)
            
            if selected_mapping and selected_mapping.get("selected_mapping") in self.mapping_configs:
                self.logger.info(
                    "Mapping configuration selected",
                    selected=selected_mapping.get("selected_mapping"),
                    record_type=selected_mapping.get("record_type"),
                    confidence=selected_mapping.get("confidence")
                )
                return selected_mapping
            else:
                self.logger.warning("Invalid mapping selection from LLM", response=response)
                return None
                
        except Exception as e:
            self.logger.error("Error selecting mapping configuration", error=str(e))
            return None
    
    async def _parse_mapping_selection(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Parse LLM response for mapping selection.
        
        Args:
            response: LLM response string
            
        Returns:
            Parsed mapping selection or None
        """
        try:
            # Try to parse as JSON
            if response.strip().startswith('{'):
                return json.loads(response)
            
            # If not JSON, try to extract JSON from the response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            
            return None
            
        except json.JSONDecodeError as e:
            self.logger.error("JSON parsing error in mapping selection", error=str(e))
            return None
        except Exception as e:
            self.logger.error("Error parsing mapping selection", error=str(e))
            return None
    
