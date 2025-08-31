"""
LangGraph migration workflow for the Migration-Accelerators platform.
"""

from typing import Any, Dict, List, Optional, TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import structlog

from agents.file_reader import FileReaderAgent
from agents.validation import ValidationAgent
from agents.mapping import MappingAgent
from agents.transformation import TransformationAgent
from agents.api_integration import APIIntegrationAgent
from config.settings import LLMConfig, MCPConfig, FieldMappingConfig


class MigrationState(TypedDict):
    """State for the migration workflow."""
    
    # Input data
    file_path: str
    mapping_config: Optional[FieldMappingConfig]
    record_type: str
    target_system: Dict[str, Any]
    
    # Workflow data
    file_data: Optional[List[Dict[str, Any]]]
    validated_data: Optional[List[Dict[str, Any]]]
    mapped_data: Optional[List[Dict[str, Any]]]
    transformed_data: Optional[Any]
    api_results: Optional[List[Dict[str, Any]]]
    
    # Workflow state
    current_step: str
    completed_steps: List[str]
    errors: List[str]
    warnings: List[str]
    progress: float
    
    # Agent configurations
    llm_config: Optional[LLMConfig]
    mcp_config: Optional[MCPConfig]
    
    # Results
    final_result: Optional[Dict[str, Any]]
    success: bool


class MigrationWorkflow:
    """LangGraph-based migration workflow."""
    
    def __init__(self, llm_config: Optional[LLMConfig] = None, mcp_config: Optional[MCPConfig] = None):
        self.llm_config = llm_config
        self.mcp_config = mcp_config
        self.logger = structlog.get_logger().bind(component="migration_workflow")
        
        # Initialize agents
        self.file_reader = FileReaderAgent(llm_config, mcp_config)
        self.validator = ValidationAgent(llm_config, mcp_config)
        self.mapper = MappingAgent(llm_config, mcp_config)
        self.transformer = TransformationAgent(llm_config, mcp_config)
        self.api_integrator = APIIntegrationAgent(llm_config, mcp_config)
        
        # Build workflow graph
        self.graph = self._build_workflow_graph()
    
    def _build_workflow_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        
        # Create state graph
        workflow = StateGraph(MigrationState)
        
        # Add nodes
        workflow.add_node("initialize", self._initialize_workflow)
        workflow.add_node("read_file", self._read_file_node)
        workflow.add_node("validate_data", self._validate_data_node)
        workflow.add_node("map_data", self._map_data_node)
        workflow.add_node("transform_data", self._transform_data_node)
        workflow.add_node("integrate_api", self._integrate_api_node)
        workflow.add_node("finalize", self._finalize_workflow)
        workflow.add_node("handle_error", self._handle_error_node)
        
        # Add edges
        workflow.set_entry_point("initialize")
        
        workflow.add_edge("initialize", "read_file")
        workflow.add_edge("read_file", "validate_data")
        workflow.add_edge("validate_data", "map_data")
        workflow.add_edge("map_data", "transform_data")
        workflow.add_edge("transform_data", "integrate_api")
        workflow.add_edge("integrate_api", "finalize")
        workflow.add_edge("finalize", END)
        
        # Add conditional edges for error handling
        workflow.add_conditional_edges(
            "read_file",
            self._should_continue,
            {
                "continue": "validate_data",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "validate_data",
            self._should_continue,
            {
                "continue": "map_data",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "map_data",
            self._should_continue,
            {
                "continue": "transform_data",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "transform_data",
            self._should_continue,
            {
                "continue": "integrate_api",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "integrate_api",
            self._should_continue,
            {
                "continue": "finalize",
                "error": "handle_error"
            }
        )
        
        workflow.add_edge("handle_error", END)
        
        return workflow.compile()
    
    async def _initialize_workflow(self, state: MigrationState) -> MigrationState:
        """Initialize the workflow."""
        self.logger.info("Initializing migration workflow")
        
        # Initialize agents
        await self.file_reader.initialize()
        await self.validator.initialize()
        await self.mapper.initialize()
        await self.transformer.initialize()
        await self.api_integrator.initialize()
        
        # Update state
        state["current_step"] = "initialization"
        state["completed_steps"] = []
        state["errors"] = []
        state["warnings"] = []
        state["progress"] = 0.0
        state["success"] = False
        
        self.logger.info("Workflow initialized")
        return state
    
    async def _read_file_node(self, state: MigrationState) -> MigrationState:
        """Read file node."""
        try:
            self.logger.info("Reading file", file_path=state["file_path"])
            
            # Prepare context
            context = {
                "file_format": state.get("file_format"),
                "encoding": state.get("encoding"),
                "delimiter": state.get("delimiter")
            }
            
            # Read file
            result = await self.file_reader.process(state["file_path"], context)
            
            if result.success:
                state["file_data"] = result.data
                state["completed_steps"].append("read_file")
                state["progress"] = 20.0
                state["current_step"] = "file_read"
                
                self.logger.info("File read successfully", records_count=len(result.data) if isinstance(result.data, list) else 1)
            else:
                state["errors"].extend(result.errors)
                state["current_step"] = "file_read_error"
                
                self.logger.error("File reading failed", errors=result.errors)
            
            return state
            
        except Exception as e:
            self.logger.error("Error in read file node", error=str(e))
            state["errors"].append(f"File reading error: {str(e)}")
            state["current_step"] = "file_read_error"
            return state
    
    async def _validate_data_node(self, state: MigrationState) -> MigrationState:
        """Validate data node."""
        try:
            self.logger.info("Validating data")
            
            if not state["file_data"]:
                state["errors"].append("No file data to validate")
                state["current_step"] = "validation_error"
                return state
            
            # Prepare validation context
            validation_context = {
                "record_type": state["record_type"],
                "validation_rules": state["mapping_config"].rules if state["mapping_config"] else [],
                "schema": None
            }
            
            # Validate data
            result = await self.validator.process(state["file_data"], validation_context)
            
            if result.success:
                state["validated_data"] = result.data
                state["completed_steps"].append("validate_data")
                state["progress"] = 40.0
                state["current_step"] = "data_validated"
                
                if result.warnings:
                    state["warnings"].extend(result.warnings)
                
                self.logger.info("Data validation successful")
            else:
                state["errors"].extend(result.errors)
                state["current_step"] = "validation_error"
                
                self.logger.error("Data validation failed", errors=result.errors)
            
            return state
            
        except Exception as e:
            self.logger.error("Error in validate data node", error=str(e))
            state["errors"].append(f"Data validation error: {str(e)}")
            state["current_step"] = "validation_error"
            return state
    
    async def _map_data_node(self, state: MigrationState) -> MigrationState:
        """Map data node."""
        try:
            self.logger.info("Mapping data")
            
            if not state["validated_data"]:
                state["errors"].append("No validated data to map")
                state["current_step"] = "mapping_error"
                return state
            
            if not state["mapping_config"]:
                state["errors"].append("No mapping configuration provided")
                state["current_step"] = "mapping_error"
                return state
            
            # Prepare mapping context
            mapping_context = {
                "mapping_rules": state["mapping_config"].rules,
                "record_type": state["record_type"]
            }
            
            # Map data
            result = await self.mapper.process(state["validated_data"], mapping_context)
            
            if result.success:
                state["mapped_data"] = result.data
                state["completed_steps"].append("map_data")
                state["progress"] = 60.0
                state["current_step"] = "data_mapped"
                
                if result.warnings:
                    state["warnings"].extend(result.warnings)
                
                self.logger.info("Data mapping successful")
            else:
                state["errors"].extend(result.errors)
                state["current_step"] = "mapping_error"
                
                self.logger.error("Data mapping failed", errors=result.errors)
            
            return state
            
        except Exception as e:
            self.logger.error("Error in map data node", error=str(e))
            state["errors"].append(f"Data mapping error: {str(e)}")
            state["current_step"] = "mapping_error"
            return state
    
    async def _transform_data_node(self, state: MigrationState) -> MigrationState:
        """Transform data node."""
        try:
            self.logger.info("Transforming data")
            
            if not state["mapped_data"]:
                state["errors"].append("No mapped data to transform")
                state["current_step"] = "transformation_error"
                return state
            
            if not state["mapping_config"]:
                state["errors"].append("No mapping configuration for transformation")
                state["current_step"] = "transformation_error"
                return state
            
            # Prepare transformation context
            transformation_context = {
                "source_format": state["mapping_config"].source_format,
                "target_format": state["mapping_config"].target_format,
                "target_schema": None
            }
            
            # Transform data
            result = await self.transformer.process(state["mapped_data"], transformation_context)
            
            if result.success:
                state["transformed_data"] = result.data
                state["completed_steps"].append("transform_data")
                state["progress"] = 80.0
                state["current_step"] = "data_transformed"
                
                if result.warnings:
                    state["warnings"].extend(result.warnings)
                
                self.logger.info("Data transformation successful")
            else:
                state["errors"].extend(result.errors)
                state["current_step"] = "transformation_error"
                
                self.logger.error("Data transformation failed", errors=result.errors)
            
            return state
            
        except Exception as e:
            self.logger.error("Error in transform data node", error=str(e))
            state["errors"].append(f"Data transformation error: {str(e)}")
            state["current_step"] = "transformation_error"
            return state
    
    async def _integrate_api_node(self, state: MigrationState) -> MigrationState:
        """Integrate with API node."""
        try:
            self.logger.info("Integrating with target system")
            
            if not state["transformed_data"]:
                state["errors"].append("No transformed data for API integration")
                state["current_step"] = "api_integration_error"
                return state
            
            # Prepare API context
            api_context = {
                "endpoint": state["target_system"].get("endpoint", "disability_policy"),
                "base_url": state["target_system"].get("base_url", "https://api.insurance-system.com"),
                "authentication": state["target_system"].get("authentication", {}),
                "batch_size": state["target_system"].get("batch_size", 10)
            }
            
            # Integrate with API
            result = await self.api_integrator.process(state["transformed_data"], api_context)
            
            if result.success:
                state["api_results"] = result.data
                state["completed_steps"].append("integrate_api")
                state["progress"] = 100.0
                state["current_step"] = "api_integrated"
                
                if result.warnings:
                    state["warnings"].extend(result.warnings)
                
                self.logger.info("API integration successful")
            else:
                state["errors"].extend(result.errors)
                state["current_step"] = "api_integration_error"
                
                self.logger.error("API integration failed", errors=result.errors)
            
            return state
            
        except Exception as e:
            self.logger.error("Error in integrate API node", error=str(e))
            state["errors"].append(f"API integration error: {str(e)}")
            state["current_step"] = "api_integration_error"
            return state
    
    async def _finalize_workflow(self, state: MigrationState) -> MigrationState:
        """Finalize the workflow."""
        try:
            self.logger.info("Finalizing migration workflow")
            
            # Create final result
            final_result = {
                "migration_summary": {
                    "total_records_processed": len(state["file_data"]) if state["file_data"] else 0,
                    "completed_steps": state["completed_steps"],
                    "progress": state["progress"],
                    "success": len(state["errors"]) == 0
                },
                "data_pipeline": {
                    "file_data": state["file_data"],
                    "validated_data": state["validated_data"],
                    "mapped_data": state["mapped_data"],
                    "transformed_data": state["transformed_data"],
                    "api_results": state["api_results"]
                },
                "errors": state["errors"],
                "warnings": state["warnings"]
            }
            
            state["final_result"] = final_result
            state["success"] = len(state["errors"]) == 0
            state["current_step"] = "completed"
            
            self.logger.info("Migration workflow finalized", success=state["success"])
            
            return state
            
        except Exception as e:
            self.logger.error("Error in finalize workflow", error=str(e))
            state["errors"].append(f"Workflow finalization error: {str(e)}")
            state["success"] = False
            return state
    
    async def _handle_error_node(self, state: MigrationState) -> MigrationState:
        """Handle error node."""
        self.logger.error("Handling workflow error", errors=state["errors"])
        
        # Create error result
        error_result = {
            "migration_summary": {
                "total_records_processed": len(state["file_data"]) if state["file_data"] else 0,
                "completed_steps": state["completed_steps"],
                "progress": state["progress"],
                "success": False,
                "failed_step": state["current_step"]
            },
            "errors": state["errors"],
            "warnings": state["warnings"]
        }
        
        state["final_result"] = error_result
        state["success"] = False
        state["current_step"] = "error"
        
        return state
    
    def _should_continue(self, state: MigrationState) -> str:
        """Determine if workflow should continue or handle error."""
        if state["errors"]:
            return "error"
        else:
            return "continue"
    
    async def run(
        self,
        file_path: str,
        mapping_config: FieldMappingConfig,
        record_type: str = "disability",
        target_system: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run the migration workflow.
        
        Args:
            file_path: Path to the input file
            mapping_config: Field mapping configuration
            record_type: Type of records to process
            target_system: Target system configuration
            
        Returns:
            Dict[str, Any]: Migration result
        """
        try:
            self.logger.info("Starting migration workflow", file_path=file_path, record_type=record_type)
            
            # Prepare initial state
            initial_state: MigrationState = {
                "file_path": file_path,
                "mapping_config": mapping_config,
                "record_type": record_type,
                "target_system": target_system or {},
                "file_data": None,
                "validated_data": None,
                "mapped_data": None,
                "transformed_data": None,
                "api_results": None,
                "current_step": "initialization",
                "completed_steps": [],
                "errors": [],
                "warnings": [],
                "progress": 0.0,
                "llm_config": self.llm_config,
                "mcp_config": self.mcp_config,
                "final_result": None,
                "success": False
            }
            
            # Run workflow
            final_state = await self.graph.ainvoke(initial_state)
            
            self.logger.info("Migration workflow completed", success=final_state["success"])
            
            return final_state["final_result"]
            
        except Exception as e:
            self.logger.error("Migration workflow failed", error=str(e))
            return {
                "migration_summary": {
                    "success": False,
                    "error": str(e)
                },
                "errors": [str(e)]
            }
    
    async def close(self) -> None:
        """Close the workflow and all agents."""
        await self.file_reader.close()
        await self.validator.close()
        await self.mapper.close()
        await self.transformer.close()
        await self.api_integrator.close()
        
        self.logger.info("Migration workflow closed")
