"""
Orchestration Agent for the Migration-Accelerators platform.
"""

from typing import Any, Dict, List, Optional, Union
import structlog

from .base_agent import BaseAgent, AgentResult
from .file_reader import FileReaderAgent
from .validation import ValidationAgent
from .mapping import MappingAgent
from .transformation import TransformationAgent
from .api_integration import APIIntegrationAgent
from config.settings import LLMConfig, MCPConfig, FieldMappingConfig
from llm.providers import BaseLLMProvider, LLMProviderFactory
from llm.prompts import get_prompt, get_system_prompt


class OrchestrationAgent(BaseAgent):
    """
    Orchestration Agent that coordinates the entire migration workflow.
    
    This agent handles:
    - Coordinating agent interactions
    - Managing workflow state and progress
    - Handling error recovery and retry logic
    - Monitoring overall system health
    - Providing status updates and reporting
    """
    
    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        mcp_config: Optional[MCPConfig] = None
    ):
        super().__init__("orchestration", llm_config, mcp_config)
        self.llm_provider: Optional[BaseLLMProvider] = None
        
        # Initialize sub-agents
        self.file_reader = FileReaderAgent(llm_config, mcp_config)
        self.validator = ValidationAgent(llm_config, mcp_config)
        self.mapper = MappingAgent(llm_config, mcp_config)
        self.transformer = TransformationAgent(llm_config, mcp_config)
        self.api_integrator = APIIntegrationAgent(llm_config, mcp_config)
        
        # Workflow state
        self.workflow_state = {
            "current_step": None,
            "completed_steps": [],
            "errors": [],
            "data_status": {},
            "progress": 0.0
        }
    
    async def initialize(self) -> None:
        """Initialize the orchestration agent and all sub-agents."""
        await super().start()
        
        if self.llm_config:
            self.llm_provider = LLMProviderFactory.create(self.llm_config, self.agent_name)
            await self.llm_provider.initialize()
            self.logger.info("LLM provider initialized for orchestration agent")
        
        # Initialize all sub-agents
        await self.file_reader.initialize()
        await self.validator.initialize()
        await self.mapper.initialize()
        await self.transformer.initialize()
        await self.api_integrator.initialize()
        
        self.logger.info("Orchestration agent and all sub-agents initialized")
    
    async def process(self, data: Any, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """
        Process complete migration workflow.
        
        Args:
            data: Migration request data (file path, mapping config, etc.)
            context: Migration context (target system, authentication, etc.)
            
        Returns:
            AgentResult: Complete migration result
        """
        try:
            self.logger.info("Starting complete migration workflow")
            
            # Reset workflow state
            self.workflow_state = {
                "current_step": "initialization",
                "completed_steps": [],
                "errors": [],
                "data_status": {},
                "progress": 0.0
            }
            
            # Extract migration parameters
            file_path = data.get("file_path") if isinstance(data, dict) else str(data)
            mapping_config = context.get("mapping_config") if context else None
            record_type = context.get("record_type", "disability") if context else "disability"
            target_system = context.get("target_system", {}) if context else {}
            
            if not file_path:
                return AgentResult(
                    success=False,
                    errors=["File path is required for migration"]
                )
            
            # Step 1: Read file
            self.workflow_state["current_step"] = "file_reading"
            file_result = await self._read_file(file_path, context)
            
            if not file_result.success:
                return await self._handle_workflow_error("file_reading", file_result.errors)
            
            self.workflow_state["completed_steps"].append("file_reading")
            self.workflow_state["progress"] = 20.0
            self.workflow_state["data_status"]["file_data"] = file_result.data
            
            # Step 2: Validate data
            self.workflow_state["current_step"] = "validation"
            validation_result = await self._validate_data(file_result.data, mapping_config, record_type)
            
            if not validation_result.success:
                return await self._handle_workflow_error("validation", validation_result.errors)
            
            self.workflow_state["completed_steps"].append("validation")
            self.workflow_state["progress"] = 40.0
            self.workflow_state["data_status"]["validated_data"] = validation_result.data
            
            # Step 3: Map data
            self.workflow_state["current_step"] = "mapping"
            mapping_result = await self._map_data(validation_result.data, mapping_config, record_type)
            
            if not mapping_result.success:
                return await self._handle_workflow_error("mapping", mapping_result.errors)
            
            self.workflow_state["completed_steps"].append("mapping")
            self.workflow_state["progress"] = 60.0
            self.workflow_state["data_status"]["mapped_data"] = mapping_result.data
            
            # Step 4: Transform data
            self.workflow_state["current_step"] = "transformation"
            transformation_result = await self._transform_data(mapping_result.data, mapping_config)
            
            if not transformation_result.success:
                return await self._handle_workflow_error("transformation", transformation_result.errors)
            
            self.workflow_state["completed_steps"].append("transformation")
            self.workflow_state["progress"] = 80.0
            self.workflow_state["data_status"]["transformed_data"] = transformation_result.data
            
            # Step 5: API integration
            self.workflow_state["current_step"] = "api_integration"
            api_result = await self._integrate_with_api(transformation_result.data, target_system)
            
            if not api_result.success:
                return await self._handle_workflow_error("api_integration", api_result.errors)
            
            self.workflow_state["completed_steps"].append("api_integration")
            self.workflow_state["progress"] = 100.0
            self.workflow_state["current_step"] = "completed"
            
            # Create final result
            final_result = await self._create_final_result(
                file_result, validation_result, mapping_result, transformation_result, api_result
            )
            
            self.logger.info("Migration workflow completed successfully")
            
            return AgentResult(
                success=True,
                data=final_result,
                metadata={
                    "workflow_state": self.workflow_state,
                    "agent": self.agent_name
                }
            )
            
        except Exception as e:
            return await self.handle_error(e, {"data": data, "context": context})
    
    async def _read_file(self, file_path: str, context: Optional[Dict[str, Any]]) -> AgentResult:
        """Read file using file reader agent."""
        try:
            self.logger.info("Reading file", file_path=file_path)
            
            result = await self.file_reader.process(file_path, context)
            
            if result.success:
                self.logger.info("File read successfully", records_count=len(result.data) if isinstance(result.data, list) else 1)
            else:
                self.logger.error("File reading failed", errors=result.errors)
            
            return result
            
        except Exception as e:
            self.logger.error("Error in file reading step", error=str(e))
            return AgentResult(
                success=False,
                errors=[f"File reading error: {str(e)}"]
            )
    
    async def _validate_data(
        self,
        data: Any,
        mapping_config: Optional[FieldMappingConfig],
        record_type: str
    ) -> AgentResult:
        """Validate data using validation agent."""
        try:
            self.logger.info("Validating data", record_type=record_type)
            
            # Prepare validation context
            validation_context = {
                "record_type": record_type,
                "validation_rules": mapping_config.rules if mapping_config else [],
                "schema": None  # Could be derived from mapping config
            }
            
            result = await self.validator.process(data, validation_context)
            
            if result.success:
                self.logger.info("Data validation successful")
            else:
                self.logger.warning("Data validation found issues", errors=result.errors, warnings=result.warnings)
            
            return result
            
        except Exception as e:
            self.logger.error("Error in validation step", error=str(e))
            return AgentResult(
                success=False,
                errors=[f"Validation error: {str(e)}"]
            )
    
    async def _map_data(
        self,
        data: Any,
        mapping_config: Optional[FieldMappingConfig],
        record_type: str
    ) -> AgentResult:
        """Map data using mapping agent."""
        try:
            self.logger.info("Mapping data", record_type=record_type)
            
            if not mapping_config:
                return AgentResult(
                    success=False,
                    errors=["Mapping configuration is required"]
                )
            
            # Prepare mapping context
            mapping_context = {
                "mapping_rules": mapping_config.rules,
                "record_type": record_type
            }
            
            result = await self.mapper.process(data, mapping_context)
            
            if result.success:
                self.logger.info("Data mapping successful")
            else:
                self.logger.error("Data mapping failed", errors=result.errors)
            
            return result
            
        except Exception as e:
            self.logger.error("Error in mapping step", error=str(e))
            return AgentResult(
                success=False,
                errors=[f"Mapping error: {str(e)}"]
            )
    
    async def _transform_data(
        self,
        data: Any,
        mapping_config: Optional[FieldMappingConfig]
    ) -> AgentResult:
        """Transform data using transformation agent."""
        try:
            self.logger.info("Transforming data")
            
            # Prepare transformation context
            transformation_context = {
                "source_format": mapping_config.source_format if mapping_config else "json",
                "target_format": mapping_config.target_format if mapping_config else "json",
                "target_schema": None  # Could be derived from mapping config
            }
            
            result = await self.transformer.process(data, transformation_context)
            
            if result.success:
                self.logger.info("Data transformation successful")
            else:
                self.logger.error("Data transformation failed", errors=result.errors)
            
            return result
            
        except Exception as e:
            self.logger.error("Error in transformation step", error=str(e))
            return AgentResult(
                success=False,
                errors=[f"Transformation error: {str(e)}"]
            )
    
    async def _integrate_with_api(
        self,
        data: Any,
        target_system: Dict[str, Any]
    ) -> AgentResult:
        """Integrate with target system using API integration agent."""
        try:
            self.logger.info("Integrating with target system")
            
            # Prepare API context
            api_context = {
                "endpoint": target_system.get("endpoint", "disability_policy"),
                "base_url": target_system.get("base_url", "https://api.insurance-system.com"),
                "authentication": target_system.get("authentication", {}),
                "batch_size": target_system.get("batch_size", 10)
            }
            
            result = await self.api_integrator.process(data, api_context)
            
            if result.success:
                self.logger.info("API integration successful")
            else:
                self.logger.error("API integration failed", errors=result.errors)
            
            return result
            
        except Exception as e:
            self.logger.error("Error in API integration step", error=str(e))
            return AgentResult(
                success=False,
                errors=[f"API integration error: {str(e)}"]
            )
    
    async def _create_final_result(
        self,
        file_result: AgentResult,
        validation_result: AgentResult,
        mapping_result: AgentResult,
        transformation_result: AgentResult,
        api_result: AgentResult
    ) -> Dict[str, Any]:
        """Create final migration result."""
        return {
            "migration_summary": {
                "total_records_processed": len(file_result.data) if isinstance(file_result.data, list) else 1,
                "validation_summary": validation_result.data.get("summary", {}) if validation_result.data else {},
                "mapping_summary": mapping_result.metadata.get("transformation_summary", {}) if mapping_result.metadata else {},
                "api_summary": api_result.data.get("summary", {}) if api_result.data else {},
                "workflow_progress": self.workflow_state["progress"],
                "completed_steps": self.workflow_state["completed_steps"]
            },
            "data_pipeline": {
                "file_data": file_result.data,
                "validated_data": validation_result.data,
                "mapped_data": mapping_result.data,
                "transformed_data": transformation_result.data,
                "api_results": api_result.data
            },
            "metadata": {
                "file_metadata": file_result.metadata,
                "validation_metadata": validation_result.metadata,
                "mapping_metadata": mapping_result.metadata,
                "transformation_metadata": transformation_result.metadata,
                "api_metadata": api_result.metadata
            }
        }
    
    async def _handle_workflow_error(self, step: str, errors: List[str]) -> AgentResult:
        """Handle workflow error and decide next action."""
        self.workflow_state["errors"].extend(errors)
        
        # Use LLM to decide next action if available
        if self.llm_provider:
            next_action = await self._decide_next_action(step, errors)
            
            if next_action == "retry":
                self.logger.info("Retrying failed step", step=step)
                # Implement retry logic
                return AgentResult(
                    success=False,
                    errors=errors,
                    metadata={"retry_recommended": True, "failed_step": step}
                )
            elif next_action == "skip":
                self.logger.info("Skipping failed step", step=step)
                # Implement skip logic
                return AgentResult(
                    success=False,
                    errors=errors,
                    metadata={"skip_recommended": True, "failed_step": step}
                )
        
        return AgentResult(
            success=False,
            errors=errors,
            metadata={"failed_step": step, "workflow_state": self.workflow_state}
        )
    
    async def _decide_next_action(self, step: str, errors: List[str]) -> str:
        """Use LLM to decide next action for failed step."""
        try:
            prompt = get_prompt(
                "orchestration_decide_next_step",
                current_state=step,
                completed_steps=self.workflow_state["completed_steps"],
                errors=errors,
                data_status=self.workflow_state["data_status"]
            )
            
            system_prompt = get_system_prompt("orchestration")
            
            response = await self.llm_provider.generate(prompt, system_prompt)
            
            # Parse response to determine action
            response_lower = response.lower()
            if "retry" in response_lower:
                return "retry"
            elif "skip" in response_lower:
                return "skip"
            elif "pause" in response_lower:
                return "pause"
            else:
                return "complete"
                
        except Exception as e:
            self.logger.error("Error deciding next action", error=str(e))
            return "complete"
    
    async def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status."""
        return {
            "workflow_state": self.workflow_state,
            "agent_status": {
                "file_reader": await self.file_reader.health_check(),
                "validator": await self.validator.health_check(),
                "mapper": await self.mapper.health_check(),
                "transformer": await self.transformer.health_check(),
                "api_integrator": await self.api_integrator.health_check()
            }
        }
    
    async def pause_workflow(self) -> None:
        """Pause the workflow."""
        self.workflow_state["current_step"] = "paused"
        self.logger.info("Workflow paused")
    
    async def resume_workflow(self) -> None:
        """Resume the workflow."""
        if self.workflow_state["current_step"] == "paused":
            # Determine next step based on completed steps
            if "api_integration" not in self.workflow_state["completed_steps"]:
                self.workflow_state["current_step"] = "api_integration"
            elif "transformation" not in self.workflow_state["completed_steps"]:
                self.workflow_state["current_step"] = "transformation"
            elif "mapping" not in self.workflow_state["completed_steps"]:
                self.workflow_state["current_step"] = "mapping"
            elif "validation" not in self.workflow_state["completed_steps"]:
                self.workflow_state["current_step"] = "validation"
            else:
                self.workflow_state["current_step"] = "file_reading"
            
            self.logger.info("Workflow resumed", current_step=self.workflow_state["current_step"])
    
    async def cancel_workflow(self) -> None:
        """Cancel the workflow."""
        self.workflow_state["current_step"] = "cancelled"
        self.logger.info("Workflow cancelled")
    
    async def close(self) -> None:
        """Close the orchestration agent and all sub-agents."""
        await self.file_reader.close()
        await self.validator.close()
        await self.mapper.close()
        await self.transformer.close()
        await self.api_integrator.close()
        
        await super().stop()
        self.logger.info("Orchestration agent and all sub-agents closed")
