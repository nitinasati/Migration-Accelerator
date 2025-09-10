"""
LangGraph migration workflow for the Migration-Accelerators platform.
"""

# Standard library imports
import operator
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, TypedDict, Annotated

# Third-party imports
import structlog
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Local imports
from agents.api_integration import APIIntegrationAgent
from agents.simple_file_reader import SimpleFileReaderAgent
from agents.mapping import MappingAgent
from agents.transformation import TransformationAgent
from config.database import get_database_config
from config.settings import LLMConfig, MCPConfig
from llm.llm_provider import initialize_langsmith
from memory.postgres_memory import PostgresMemoryManager


class MigrationState(TypedDict):
    """
    State container for the migration workflow.
    
    This TypedDict defines the complete state structure that flows through
    the LangGraph workflow. It contains all input parameters, intermediate
    data transformations, workflow progress tracking, and final results.
    
    Attributes:
        file_path: Path to the source file to be migrated
        mapping_config: Field mapping configuration for data transformation
        record_type: Type of records being processed (e.g., 'disability', 'absence')
        target_system: Configuration for the target system API integration
        
        file_data: Raw data read from the source file

        mapped_data: Data after field mapping transformation
        transformed_data: Final transformed data ready for target system
        api_results: Results from API calls to target system
        
        current_step: Current workflow step being executed
        completed_steps: List of successfully completed workflow steps
        errors: List of error messages encountered during processing
        warnings: List of warning messages for non-critical issues
        progress: Overall workflow progress percentage (0.0 to 100.0)
        
        llm_config: LLM provider configuration for AI-enhanced processing
        mcp_config: MCP client configuration for API interactions
        
        final_result: Complete migration result with summary and pipeline data
        success: Boolean indicating overall migration success
    """
    
    # Input data
    file_path: str
    mapping_config: Optional[Any]  # Will be selected by LLM agent
    record_type: str
    target_system: Dict[str, Any]
    layout_file: Optional[str]  # Optional layout file for mainframe files
    run_id: Optional[str]  # Database run ID for state persistence
    
    # Workflow data
    file_data: Optional[List[Dict[str, Any]]]
    mapped_data: Optional[List[Dict[str, Any]]]
    mapping_metadata: Optional[Dict[str, Any]]
    transformed_data: Optional[Any]
    api_results: Optional[List[Dict[str, Any]]]
    
    # Workflow state
    current_step: str
    completed_steps: List[str]
    errors: List[str]
    warnings: List[str]
    progress: float
    
    # Timing information
    start_time: Optional[float]
    end_time: Optional[float]
    step_timings: Optional[Dict[str, float]]
    
    # Agent configurations
    llm_config: Optional[LLMConfig]
    mcp_config: Optional[MCPConfig]
    
    # Results
    final_result: Optional[Dict[str, Any]]
    success: bool


class MigrationWorkflow:
    """
    LangGraph-based migration workflow orchestrator.
    
    This class orchestrates the complete data migration process using LangGraph
    for workflow management. It coordinates multiple specialized agents to
    perform file reading, data validation, field mapping, transformation,
    and API integration in a structured, error-handled workflow.
    
    The workflow follows a sequential pipeline:
    1. Initialize all agents and workflow state
    2. Read and parse source file data
    3. Validate data against schema and business rules
    4. Apply field mapping transformations
    5. Transform data to target format
    6. Integrate with target system via API calls
    7. Finalize and return comprehensive results
    
    Features:
    - Error handling with conditional workflow routing
    - Progress tracking and state management
    - Comprehensive logging and monitoring
    - Support for multiple data formats and record types
    - LLM-enhanced processing capabilities
    - MCP-based API integration
    """
    
    def __init__(self, llm_config: Optional[LLMConfig] = None, mcp_config: Optional[MCPConfig] = None):
        """
        Initialize the migration workflow.
        
        Args:
            llm_config: LLM provider configuration for AI-enhanced processing
            mcp_config: MCP client configuration for API interactions
        """
        self.llm_config = llm_config
        self.mcp_config = mcp_config
        self.logger = structlog.get_logger().bind(component="migration_workflow")
        
        # Initialize database and memory
        self.db_config = get_database_config()
        self.memory_manager: Optional[PostgresMemoryManager] = None
        
        # Initialize LangSmith tracing
        initialize_langsmith()
        
        # Initialize specialized agents for each workflow step
        # Initialize simple file reader agent
        self.file_reader = SimpleFileReaderAgent(llm_config)
        self.mapper = MappingAgent(llm_config, mcp_config)
        self.transformer = TransformationAgent(llm_config, mcp_config)
        self.api_integrator = APIIntegrationAgent(llm_config, mcp_config)
        
        # Build the LangGraph workflow structure
        self.graph = self._build_workflow_graph()
    
    def _build_workflow_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow structure.
        
        Creates a state graph with nodes for each workflow step and conditional
        edges for error handling. The workflow follows a linear progression
        with error handling at each step.
        
        Workflow Structure:
        - Linear progression: initialize → read_file → map_data → transform_data → integrate_api → finalize
        - Error handling: Each step can route to handle_error if errors occur
        - Conditional routing: Uses _should_continue to determine next step
        
        Returns:
            StateGraph: Compiled LangGraph workflow ready for execution
        """
        
        # Create state graph with MigrationState as the state type
        workflow = StateGraph(MigrationState)
        
        # Add workflow nodes - each represents a processing step
        workflow.add_node("initialize", self._initialize_workflow)
        workflow.add_node("read_file", self._read_file_node)
        workflow.add_node("map_data", self._map_data_node)
        workflow.add_node("transform_data", self._transform_data_node)
        workflow.add_node("integrate_api", self._integrate_api_node)
        workflow.add_node("finalize", self._finalize_workflow)
        workflow.add_node("handle_error", self._handle_error_node)
        
        # Set the entry point for the workflow
        workflow.set_entry_point("initialize")
        
        # Add conditional edges for error handling at each step
        # If errors occur, route to handle_error; otherwise continue to next step
        workflow.add_conditional_edges(
            "initialize",
            self._should_continue,
            {
                "continue": "read_file",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "read_file",
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
        
        # Finalize always leads to workflow end
        workflow.add_edge("finalize", END)
        
        # Error handling always leads to workflow end
        workflow.add_edge("handle_error", END)
        
        # Compile the workflow for execution
        compiled_workflow = workflow.compile()
        return compiled_workflow
    

    
    async def _initialize_workflow(self, state: MigrationState) -> MigrationState:
        """
        Initialize the migration workflow and all agents.
        
        This is the first step in the workflow that prepares all agents
        for processing and initializes the workflow state tracking.
        
        Args:
            state: Current workflow state
            
        Returns:
            MigrationState: Updated state with initialization complete
        """
        self.logger.info("Initializing migration workflow")
        
        # Initialize memory manager
        self.memory_manager = PostgresMemoryManager(self.db_config)
        await self.memory_manager.initialize()
        
        # Create migration run in database
        memory_store = self.memory_manager.get_memory_store()
        run_id = await memory_store.create_migration_run(
            file_path=state["file_path"],
            record_type=state["record_type"]
        )
        
        # Store run ID in state for reference
        state["run_id"] = run_id
        
        # Initialize all specialized agents for the workflow steps
        await self.file_reader.initialize()
        await self.mapper.initialize()
        await self.transformer.initialize()
        await self.api_integrator.initialize()
        
        # Initialize workflow state tracking variables
        state["current_step"] = "initialization"
        state["completed_steps"] = []
        state["errors"] = []
        state["warnings"] = []
        state["progress"] = 0.0
        state["success"] = False
        
        # Save state to database
        await memory_store.save_state(
            step_name="initialization",
            state=state,
            metadata={"progress": 0.0, "step": "initialization"}
        )
        
        self.logger.info("Workflow initialized", run_id=run_id)
        return state
    
    async def _read_file_node(self, state: MigrationState) -> MigrationState:
        """
        Read and parse the source file data.
        
        This node uses the FileReaderAgent to read the source file and parse
        it into structured data. It handles various file formats (CSV, Excel, JSON, etc.)
        and extracts the raw data for further processing.
        
        Args:
            state: Current workflow state containing file_path
            
        Returns:
            MigrationState: Updated state with file_data populated or errors recorded
        """
        try:
            self.logger.info("Reading file", file_path=state["file_path"])
            
            # Use SimpleFileReaderAgent to read and parse the file
            layout_file = state.get("layout_file")  # Optional layout file for mainframe files
            result = await self.file_reader.read_file(state["file_path"], layout_file)
            
            if result.get("success", False):
                # Store parsed file data in workflow state
                state["file_data"] = result
                state["completed_steps"].append("read_file")
                state["progress"] = 25.0
                state["current_step"] = "file_read"
                
                # Save state to database
                if self.memory_manager:
                    memory_store = self.memory_manager.get_memory_store()
                    # Extract record count from the result - handle different LLM output formats
                    records_count = result.get("total_records", 0)
                    if records_count == 0:
                        # Try different possible field names for the data array
                        data_array = result.get("data", []) or result.get("members", []) or result.get("records", [])
                        records_count = len(data_array) if isinstance(data_array, list) else 1
                    await memory_store.save_state(
                        step_name="read_file",
                        state=state,
                        metadata={"progress": 25.0, "step": "read_file", "records_count": records_count}
                    )
                
                # Extract record count for logging
                records_count = result.get("total_records", 0)
                if records_count == 0:
                    # Try different possible field names for the data array
                    data_array = result.get("data", []) or result.get("members", []) or result.get("records", [])
                    records_count = len(data_array) if isinstance(data_array, list) else 1
                self.logger.info("File read successfully", records_count=records_count)
            else:
                # Record errors and update state for error handling
                error_msg = result.get("error", "Unknown error")
                state["errors"].append(error_msg)
                state["current_step"] = "file_read_error"
                
                # Save error state to database
                if self.memory_manager:
                    memory_store = self.memory_manager.get_memory_store()
                    await memory_store.save_state(
                        step_name="read_file_error",
                        state=state,
                        metadata={"progress": 25.0, "step": "read_file_error", "errors": [error_msg]}
                    )
                
                self.logger.error("File reading failed", errors=[error_msg])
            
            return state
            
        except Exception as e:
            # Handle unexpected errors during file reading
            self.logger.error("Error in read file node", error=str(e))
            state["errors"].append(f"File reading error: {str(e)}")
            state["current_step"] = "file_read_error"
            return state
    

    async def _map_data_node(self, state: MigrationState) -> MigrationState:
        """
        Apply intelligent field mapping transformations using LLM.
        
        This node uses the LLM-powered MappingAgent to analyze data attributes,
        select appropriate mapping configurations, and transform data intelligently.
        
        Args:
            state: Current workflow state containing file_data
            
        Returns:
            MigrationState: Updated state with mapped_data or mapping errors
        """
        try:
            self.logger.info("Mapping data")
            
            # Validate prerequisites for mapping
            if not state["file_data"]:
                state["errors"].append("No file data to map")
                state["current_step"] = "mapping_error"
                return state
            
            # Use LLM-powered MappingAgent for intelligent mapping
            # The agent will automatically select appropriate mapping configuration
            result = await self.mapper.process(state["file_data"], {})
            
            if result.success:
                # Store mapped data and update progress
                state["mapped_data"] = result.data
                state["completed_steps"].append("map_data")
                state["progress"] = 50.0
                state["current_step"] = "data_mapped"
                
                # Store mapping metadata for transformation step
                if result.metadata:
                    state["mapping_metadata"] = result.metadata
                
                # Collect any warnings from mapping
                if result.warnings:
                    state["warnings"].extend(result.warnings)
                
                # Save state to database
                if self.memory_manager:
                    memory_store = self.memory_manager.get_memory_store()
                    await memory_store.save_state(
                        step_name="map_data",
                        state=state,
                        metadata={"progress": 50.0, "step": "map_data", "mapping_metadata": result.metadata}
                    )
                
                self.logger.info("Data mapping successful")
            else:
                # Record mapping errors
                state["errors"].extend(result.errors)
                state["current_step"] = "mapping_error"
                
                # Save error state to database
                if self.memory_manager:
                    memory_store = self.memory_manager.get_memory_store()
                    await memory_store.save_state(
                        step_name="map_data_error",
                        state=state,
                        metadata={"progress": 50.0, "step": "map_data_error", "errors": result.errors}
                    )
                
                self.logger.error("Data mapping failed", errors=result.errors)
            
            return state
            
        except Exception as e:
            # Handle unexpected errors during mapping
            self.logger.error("Error in map data node", error=str(e))
            state["errors"].append(f"Data mapping error: {str(e)}")
            state["current_step"] = "mapping_error"
            return state
    
    async def _transform_data_node(self, state: MigrationState) -> MigrationState:
        """
        Transform mapped data to target format.
        
        This node uses the TransformationAgent to convert the mapped data
        from source format to target format (e.g., CSV to JSON). It handles
        format-specific transformations and prepares data for API integration.
        
        Args:
            state: Current workflow state containing mapped_data and mapping_config
            
        Returns:
            MigrationState: Updated state with transformed_data or transformation errors
        """
        try:
            self.logger.info("Transforming data")
            
            # Validate prerequisites for transformation
            if not state["mapped_data"]:
                state["errors"].append("No mapped data to transform")
                state["current_step"] = "transformation_error"
                return state
            
            # Get mapping configuration from mapping agent's metadata
            mapping_metadata = state.get("mapping_metadata", {})
            mapping_config = mapping_metadata.get("mapping_config")
            record_type = mapping_metadata.get("record_type", "unknown")
            
            if not mapping_config:
                state["errors"].append("No mapping configuration for transformation")
                state["current_step"] = "transformation_error"
                return state
            
            # Prepare transformation context with mapping configuration
            transformation_context = {
                "mapping_config": mapping_config,
                "record_type": record_type
            }
            
            # Use TransformationAgent for LLM-based transformation
            result = await self.transformer.process(state["mapped_data"], transformation_context)

            if result.success:
                # Store transformed data and update progress
                state["transformed_data"] = result.data
                state["completed_steps"].append("transform_data")
                state["progress"] = 75.0
                state["current_step"] = "data_transformed"
                
                # Collect any warnings from transformation
                if result.warnings:
                    state["warnings"].extend(result.warnings)
                
                # Save state to database
                if self.memory_manager:
                    memory_store = self.memory_manager.get_memory_store()
                    await memory_store.save_state(
                        step_name="transform_data",
                        state=state,
                        metadata={"progress": 75.0, "step": "transform_data"}
                    )
                
                self.logger.info("Data transformation successful")
            else:
                # Record transformation errors
                state["errors"].extend(result.errors)
                state["current_step"] = "transformation_error"
                
                # Save error state to database
                if self.memory_manager:
                    memory_store = self.memory_manager.get_memory_store()
                    await memory_store.save_state(
                        step_name="transform_data_error",
                        state=state,
                        metadata={"progress": 75.0, "step": "transform_data_error", "errors": result.errors}
                    )
                
                self.logger.error("Data transformation failed", errors=result.errors)
            
            return state
            
        except Exception as e:
            # Handle unexpected errors during transformation
            self.logger.error("Error in transform data node", error=str(e))
            state["errors"].append(f"Data transformation error: {str(e)}")
            state["current_step"] = "transformation_error"
            return state
    
    async def _integrate_api_node(self, state: MigrationState) -> MigrationState:
        """
        Integrate with target system via API calls.
        
        This node uses the APIIntegrationAgent to send the transformed data
        to the target system via REST API calls. It handles authentication,
        batching, retries, and response processing.
        
        Args:
            state: Current workflow state containing transformed_data and target_system config
            
        Returns:
            MigrationState: Updated state with api_results or integration errors
        """
        try:
            self.logger.info("Integrating with target system")
            
            # Validate prerequisites for API integration
            if not state["transformed_data"]:
                state["errors"].append("No transformed data for API integration")
                state["current_step"] = "api_integration_error"
                return state
            
            # Prepare API context with target system configuration
            api_context = {
                "endpoint": state["target_system"].get("endpoint", "disability_policy"),
                "base_url": state["target_system"].get("base_url", "https://api.insurance-system.com"),
                "authentication": state["target_system"].get("authentication", {}),
                "batch_size": state["target_system"].get("batch_size", 10),
                "output_mode": "file",  # Use file output instead of API calls
                "output_dir": "data/output"  # Output directory for JSON files
            }
            
            # Use APIIntegrationAgent to send data to target system
            result = await self.api_integrator.process(state["transformed_data"], api_context)
            
            if result.success:
                # Store API results and mark workflow as complete
                state["api_results"] = result.data
                state["completed_steps"].append("integrate_api")
                state["progress"] = 100.0
                state["current_step"] = "api_integrated"
                
                # Collect any warnings from API integration
                if result.warnings:
                    state["warnings"].extend(result.warnings)
                
                # Save state to database
                if self.memory_manager:
                    memory_store = self.memory_manager.get_memory_store()
                    await memory_store.save_state(
                        step_name="integrate_api",
                        state=state,
                        metadata={"progress": 100.0, "step": "integrate_api"}
                    )
                
                self.logger.info("API integration successful")
            else:
                # Record API integration errors
                state["errors"].extend(result.errors)
                state["current_step"] = "api_integration_error"
                
                # Save error state to database
                if self.memory_manager:
                    memory_store = self.memory_manager.get_memory_store()
                    await memory_store.save_state(
                        step_name="integrate_api_error",
                        state=state,
                        metadata={"progress": 100.0, "step": "integrate_api_error", "errors": result.errors}
                    )
                
                self.logger.error("API integration failed", errors=result.errors)
            
            return state
            
        except Exception as e:
            # Handle unexpected errors during API integration
            self.logger.error("Error in integrate API node", error=str(e))
            state["errors"].append(f"API integration error: {str(e)}")
            state["current_step"] = "api_integration_error"
            return state
    
    async def _finalize_workflow(self, state: MigrationState) -> MigrationState:
        """
        Finalize the migration workflow and create comprehensive results.
        
        This node compiles all workflow data, progress, and results into a
        comprehensive final result structure. It determines overall success
        and prepares the complete migration report.
        
        Args:
            state: Current workflow state with all processing results
            
        Returns:
            MigrationState: Updated state with final_result and success status
        """
        try:
            self.logger.info("Finalizing migration workflow")
            
            # Create comprehensive final result structure
            # Extract record count from file_data - handle different LLM output formats
            file_data = state.get("file_data", {})
            if isinstance(file_data, dict):
                records_count = file_data.get("total_records", 0)
                if records_count == 0:
                    # Try different possible field names for the data array
                    data_array = file_data.get("data", []) or file_data.get("members", []) or file_data.get("records", [])
                    records_count = len(data_array) if isinstance(data_array, list) else 1
            else:
                records_count = len(file_data) if isinstance(file_data, list) else 0
            
            final_result = {
                "migration_summary": {
                    "total_records_processed": records_count,
                    "completed_steps": state["completed_steps"],
                    "progress": state["progress"],
                    "success": len(state["errors"]) == 0
                },
                "data_pipeline": {
                    "file_data": state["file_data"],

                    "mapped_data": state["mapped_data"],
                    "transformed_data": state["transformed_data"],
                    "api_results": state["api_results"]
                },
                "errors": state["errors"],
                "warnings": state["warnings"]
            }
            
            # Update state with final results
            state["final_result"] = final_result
            state["success"] = len(state["errors"]) == 0
            state["current_step"] = "completed"
            
            # Log final status for debugging
            self.logger.info(
                "Final status determined", 
                success=state["success"], 
                error_count=len(state["errors"]), 
                errors=state["errors"][:3] if state["errors"] else []  # Log first 3 errors for debugging
            )
            
            # Update migration run status in database
            if self.memory_manager and state.get("run_id"):
                memory_store = self.memory_manager.get_memory_store()
                
                # Calculate total duration
                start_time = state.get("start_time")
                end_time = datetime.now()
                total_duration = None
                if start_time:
                    if isinstance(start_time, str):
                        start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                    elif isinstance(start_time, float):
                        # Convert float timestamp to datetime
                        start_time = datetime.fromtimestamp(start_time)
                    total_duration = (end_time - start_time).total_seconds()
                
                # Update run status
                await memory_store.update_migration_run_status(
                    run_id=state["run_id"],
                    status="completed" if state["success"] else "failed",
                    success=state["success"],
                    total_duration=total_duration,
                    error_message="; ".join(state["errors"]) if state["errors"] else None
                )
                
                # Save final state to database
                await memory_store.save_state(
                    step_name="finalize",
                    state=state,
                    metadata={"progress": 100.0, "step": "finalize", "success": state["success"]}
                )
                
                # Save checkpoint for potential resume
                await memory_store.save_checkpoint(
                    checkpoint_name="final",
                    state=state,
                    metadata={"final_state": True, "success": state["success"]}
                )
            
            self.logger.info("Migration workflow finalized", success=state["success"])
            
            return state
            
        except Exception as e:
            # Handle unexpected errors during finalization
            self.logger.error("Error in finalize workflow", error=str(e))
            state["errors"].append(f"Workflow finalization error: {str(e)}")
            state["success"] = False
            return state
    
    async def _handle_error_node(self, state: MigrationState) -> MigrationState:
        """
        Handle workflow errors and create error result.
        
        This node is called when any step in the workflow encounters errors.
        It creates a comprehensive error result with partial progress information
        and detailed error reporting.
        
        Args:
            state: Current workflow state with errors recorded
            
        Returns:
            MigrationState: Updated state with error result and failure status
        """
        self.logger.error("Handling workflow error", errors=state["errors"])
        
        # Create comprehensive error result with partial progress
        # Extract record count from file_data - handle different LLM output formats
        file_data = state.get("file_data", {})
        if isinstance(file_data, dict):
            records_count = file_data.get("total_records", 0)
            if records_count == 0:
                # Try different possible field names for the data array
                data_array = file_data.get("data", []) or file_data.get("members", []) or file_data.get("records", [])
                records_count = len(data_array) if isinstance(data_array, list) else 1
        else:
            records_count = len(file_data) if isinstance(file_data, list) else 0
        
        error_result = {
            "migration_summary": {
                "total_records_processed": records_count,
                "completed_steps": state["completed_steps"],
                "progress": state["progress"],
                "success": False,
                "failed_step": state["current_step"]
            },
            "errors": state["errors"],
            "warnings": state["warnings"]
        }
        
        # Update state with error result
        state["final_result"] = error_result
        state["success"] = False
        state["current_step"] = "error"
        
        return state
    
    def _should_continue(self, state: MigrationState) -> str:
        """
        Determine workflow routing based on error state.
        
        This function is used by LangGraph's conditional edges to determine
        whether the workflow should continue to the next step or route to
        error handling.
        
        Args:
            state: Current workflow state
            
        Returns:
            str: "continue" if no errors, "error" if errors exist
        """
        if state["errors"]:
            return "error"
        else:
            return "continue"
    
    async def run(
        self,
        file_path: str,
        record_type: str = "disability",
        target_system: Optional[Dict[str, Any]] = None,
        layout_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute the complete migration workflow.
        
        This is the main entry point for running a data migration. It initializes
        the workflow state, executes the LangGraph workflow, and returns comprehensive
        results including success status, processed data, and any errors or warnings.
        
        Args:
            file_path: Path to the source file to be migrated
            record_type: Type of records being processed (e.g., 'disability', 'absence')
            target_system: Target system configuration for API integration
            layout_file: Optional layout file for mainframe fixed-width files
            
        Returns:
            Dict[str, Any]: Complete migration result with:
                - migration_summary: Overall success, progress, and statistics
                - data_pipeline: All intermediate data transformations
                - errors: List of error messages
                - warnings: List of warning messages
        """
        try:
            # Record start time
            start_time = time.time()
            self.logger.info("Starting migration workflow", file_path=file_path, record_type=record_type)
            
            # Prepare initial workflow state with all required parameters
            initial_state: MigrationState = {
                "file_path": file_path,
                "mapping_config": None,  # Mapping config will be selected by LLM agent
                "record_type": record_type,
                "target_system": target_system or {},
                "layout_file": layout_file,  # Optional layout file for mainframe files
                "run_id": None,  # Will be set during initialization
                "file_data": None,
                "mapped_data": None,
                "mapping_metadata": None,
                "transformed_data": None,
                "api_results": None,
                "current_step": "initialization",
                "completed_steps": [],
                "errors": [],
                "warnings": [],
                "progress": 0.0,
                "start_time": start_time,
                "end_time": None,
                "step_timings": {},
                "llm_config": self.llm_config,
                "mcp_config": self.mcp_config,
                "final_result": None,
                "success": False
            }
            
            # Execute the complete LangGraph workflow
            final_state = await self.graph.ainvoke(initial_state)
            
            # Record end time and calculate total duration
            end_time = time.time()
            total_duration = end_time - start_time
            
            # Update final state with timing information
            final_state["end_time"] = end_time
            if "final_result" in final_state and final_state["final_result"]:
                final_state["final_result"]["migration_summary"]["total_duration_seconds"] = total_duration
                final_state["final_result"]["migration_summary"]["total_duration_formatted"] = str(timedelta(seconds=int(total_duration)))
                final_state["final_result"]["migration_summary"]["start_time"] = datetime.fromtimestamp(start_time).isoformat()
                final_state["final_result"]["migration_summary"]["end_time"] = datetime.fromtimestamp(end_time).isoformat()
            
            self.logger.info(
                "Migration workflow completed", 
                success=final_state["success"],
                total_duration_seconds=total_duration,
                total_duration_formatted=str(timedelta(seconds=int(total_duration)))
            )
            
            # Return the comprehensive final result
            return final_state["final_result"]
            
        except Exception as e:
            # Handle unexpected workflow failures
            self.logger.error("Migration workflow failed", error=str(e))
            return {
                "migration_summary": {
                    "success": False,
                    "error": str(e)
                },
                "errors": [str(e)]
            }
    
    async def close(self) -> None:
        """
        Close the workflow and clean up all agents.
        
        This method properly closes all initialized agents and cleans up
        resources. It should be called after workflow completion to ensure
        proper resource management.
        """
        # Close memory manager
        if self.memory_manager:
            await self.memory_manager.close()
        
        # Close all specialized agents in the workflow
        await self.file_reader.close()
        await self.mapper.close()
        await self.transformer.close()
        await self.api_integrator.close()
        
        self.logger.info("Migration workflow closed")
