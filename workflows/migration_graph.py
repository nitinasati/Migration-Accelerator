"""
LangGraph workflow for the migration process.
"""

import asyncio
import json
import time
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from dataclasses import dataclass

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage

from agents.base_agent import BaseAgent, AgentRole, AgentStatus
from agents.file_reader import FileReaderAgent
from config.settings import LLMConfig, MCPConfig, FieldMappingConfig


@dataclass
class MigrationState:
    """State for the migration workflow."""
    # Input data
    file_path: str
    mapping_config: Optional[FieldMappingConfig] = None
    record_type: str = "unknown"
    
    # Processing data
    records: List[Dict[str, Any]] = None
    validation_results: List[Dict[str, Any]] = None
    transformed_records: List[Dict[str, Any]] = None
    api_results: List[Dict[str, Any]] = None
    
    # Workflow state
    current_step: str = "start"
    errors: List[str] = None
    warnings: List[str] = None
    
    # Metrics
    start_time: float = None
    end_time: float = None
    total_records: int = 0
    successful_records: int = 0
    failed_records: int = 0
    
    def __post_init__(self):
        if self.records is None:
            self.records = []
        if self.validation_results is None:
            self.validation_results = []
        if self.transformed_records is None:
            self.transformed_records = []
        if self.api_results is None:
            self.api_results = []
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.start_time is None:
            self.start_time = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "file_path": self.file_path,
            "record_type": self.record_type,
            "current_step": self.current_step,
            "total_records": self.total_records,
            "successful_records": self.successful_records,
            "failed_records": self.failed_records,
            "errors": self.errors,
            "warnings": self.warnings,
            "duration_seconds": self.end_time - self.start_time if self.end_time else None
        }


class MigrationWorkflow:
    """Main migration workflow using LangGraph."""
    
    def __init__(
        self,
        llm_config: LLMConfig,
        mcp_config: MCPConfig,
        agents: Optional[Dict[str, BaseAgent]] = None
    ):
        self.llm_config = llm_config
        self.mcp_config = mcp_config
        self.agents = agents or {}
        
        # Initialize workflow graph
        self.graph = self._create_workflow_graph()
        
        # Initialize checkpoint memory
        self.memory = MemorySaver()
    
    def _create_workflow_graph(self) -> StateGraph:
        """Create the migration workflow graph."""
        
        # Create state graph
        workflow = StateGraph(MigrationState)
        
        # Add nodes
        workflow.add_node("file_reader", self._file_reader_node)
        workflow.add_node("validator", self._validator_node)
        workflow.add_node("mapper", self._mapper_node)
        workflow.add_node("transformer", self._transformer_node)
        workflow.add_node("api_integrator", self._api_integrator_node)
        workflow.add_node("error_handler", self._error_handler_node)
        
        # Add edges
        workflow.add_edge("file_reader", "validator")
        workflow.add_edge("validator", "mapper")
        workflow.add_edge("mapper", "transformer")
        workflow.add_edge("transformer", "api_integrator")
        workflow.add_edge("api_integrator", END)
        
        # Add conditional edges for error handling
        workflow.add_conditional_edges(
            "file_reader",
            self._should_continue,
            {
                "continue": "validator",
                "error": "error_handler"
            }
        )
        
        workflow.add_conditional_edges(
            "validator",
            self._should_continue,
            {
                "continue": "mapper",
                "error": "error_handler"
            }
        )
        
        workflow.add_conditional_edges(
            "mapper",
            self._should_continue,
            {
                "continue": "transformer",
                "error": "error_handler"
            }
        )
        
        workflow.add_conditional_edges(
            "transformer",
            self._should_continue,
            {
                "continue": "api_integrator",
                "error": "error_handler"
            }
        )
        
        workflow.add_conditional_edges(
            "api_integrator",
            self._should_continue,
            {
                "continue": END,
                "error": "error_handler"
            }
        )
        
        # Set entry point
        workflow.set_entry_point("file_reader")
        
        return workflow.compile(checkpointer=self.memory)
    
    async def _file_reader_node(self, state: MigrationState) -> MigrationState:
        """File reader node."""
        try:
            state.current_step = "file_reader"
            
            # Get or create file reader agent
            file_reader = self.agents.get("file_reader")
            if not file_reader:
                from llm.providers import LLMProviderFactory
                from mcp.client import MCPToolManager
                
                llm_provider = LLMProviderFactory.create(self.llm_config)
                mcp_manager = MCPToolManager(self.mcp_config)
                
                file_reader = FileReaderAgent(
                    llm_provider=llm_provider,
                    mcp_manager=mcp_manager
                )
                self.agents["file_reader"] = file_reader
            
            # Execute file reading
            result = await file_reader.execute({
                "file_path": state.file_path,
                "record_type": state.record_type
            })
            
            # Update state
            state.records = result.get("records", [])
            state.total_records = len(state.records)
            state.record_type = result.get("record_type", state.record_type)
            
            # Add any warnings
            if result.get("file_analysis", {}).get("issues"):
                state.warnings.extend(result["file_analysis"]["issues"])
            
            return state
            
        except Exception as e:
            state.errors.append(f"File reading failed: {str(e)}")
            return state
    
    async def _validator_node(self, state: MigrationState) -> MigrationState:
        """Validator node."""
        try:
            state.current_step = "validator"
            
            # Get or create validator agent
            validator = self.agents.get("validator")
            if not validator:
                from agents.validator import ValidationAgent
                from llm.providers import LLMProviderFactory
                from mcp.client import MCPToolManager
                
                llm_provider = LLMProviderFactory.create(self.llm_config)
                mcp_manager = MCPToolManager(self.mcp_config)
                
                validator = ValidationAgent(
                    llm_provider=llm_provider,
                    mcp_manager=mcp_manager
                )
                self.agents["validator"] = validator
            
            # Execute validation
            validation_results = []
            for record in state.records:
                result = await validator.execute({
                    "record": record,
                    "record_type": state.record_type,
                    "validation_rules": state.mapping_config.rules if state.mapping_config else []
                })
                validation_results.append(result)
            
            # Update state
            state.validation_results = validation_results
            
            # Count successful/failed records
            for result in validation_results:
                if result.get("is_valid", False):
                    state.successful_records += 1
                else:
                    state.failed_records += 1
                    if result.get("errors"):
                        state.errors.extend(result["errors"])
            
            return state
            
        except Exception as e:
            state.errors.append(f"Validation failed: {str(e)}")
            return state
    
    async def _mapper_node(self, state: MigrationState) -> MigrationState:
        """Mapper node."""
        try:
            state.current_step = "mapper"
            
            # Get or create mapper agent
            mapper = self.agents.get("mapper")
            if not mapper:
                from agents.mapper import MappingAgent
                from llm.providers import LLMProviderFactory
                from mcp.client import MCPToolManager
                
                llm_provider = LLMProviderFactory.create(self.llm_config)
                mcp_manager = MCPToolManager(self.mcp_config)
                
                mapper = MappingAgent(
                    llm_provider=llm_provider,
                    mcp_manager=mcp_manager
                )
                self.agents["mapper"] = mapper
            
            # Execute mapping
            result = await mapper.execute({
                "records": state.records,
                "mapping_config": state.mapping_config,
                "record_type": state.record_type
            })
            
            # Update state
            state.mapping_config = result.get("mapping_config", state.mapping_config)
            
            return state
            
        except Exception as e:
            state.errors.append(f"Mapping failed: {str(e)}")
            return state
    
    async def _transformer_node(self, state: MigrationState) -> MigrationState:
        """Transformer node."""
        try:
            state.current_step = "transformer"
            
            # Get or create transformer agent
            transformer = self.agents.get("transformer")
            if not transformer:
                from agents.transformer import TransformationAgent
                from llm.providers import LLMProviderFactory
                from mcp.client import MCPToolManager
                
                llm_provider = LLMProviderFactory.create(self.llm_config)
                mcp_manager = MCPToolManager(self.mcp_config)
                
                transformer = TransformationAgent(
                    llm_provider=llm_provider,
                    mcp_manager=mcp_manager
                )
                self.agents["transformer"] = transformer
            
            # Execute transformation
            transformed_records = []
            for record in state.records:
                result = await transformer.execute({
                    "record": record,
                    "mapping_config": state.mapping_config,
                    "record_type": state.record_type
                })
                transformed_records.append(result.get("transformed_record", record))
            
            # Update state
            state.transformed_records = transformed_records
            
            return state
            
        except Exception as e:
            state.errors.append(f"Transformation failed: {str(e)}")
            return state
    
    async def _api_integrator_node(self, state: MigrationState) -> MigrationState:
        """API integrator node."""
        try:
            state.current_step = "api_integrator"
            
            # Get or create API integrator agent
            api_integrator = self.agents.get("api_integrator")
            if not api_integrator:
                from agents.api_integrator import APIIntegrationAgent
                from llm.providers import LLMProviderFactory
                from mcp.client import MCPToolManager
                
                llm_provider = LLMProviderFactory.create(self.llm_config)
                mcp_manager = MCPToolManager(self.mcp_config)
                
                api_integrator = APIIntegrationAgent(
                    llm_provider=llm_provider,
                    mcp_manager=mcp_manager
                )
                self.agents["api_integrator"] = api_integrator
            
            # Execute API integration
            api_results = []
            for record in state.transformed_records:
                result = await api_integrator.execute({
                    "record": record,
                    "record_type": state.record_type,
                    "operation": "create"
                })
                api_results.append(result)
            
            # Update state
            state.api_results = api_results
            state.end_time = time.time()
            
            return state
            
        except Exception as e:
            state.errors.append(f"API integration failed: {str(e)}")
            return state
    
    async def _error_handler_node(self, state: MigrationState) -> MigrationState:
        """Error handler node."""
        state.current_step = "error_handler"
        state.end_time = time.time()
        
        # Log errors
        for error in state.errors:
            print(f"ERROR: {error}")
        
        return state
    
    def _should_continue(self, state: MigrationState) -> str:
        """Determine if workflow should continue or handle errors."""
        if state.errors:
            return "error"
        return "continue"
    
    async def run(self, file_path: str, mapping_config: Optional[FieldMappingConfig] = None, record_type: str = "unknown") -> Dict[str, Any]:
        """Run the migration workflow."""
        
        # Create initial state
        initial_state = MigrationState(
            file_path=file_path,
            mapping_config=mapping_config,
            record_type=record_type
        )
        
        # Run the workflow
        try:
            final_state = await self.graph.ainvoke(initial_state)
            
            # Return results
            return {
                "success": len(final_state.errors) == 0,
                "state": final_state.to_dict(),
                "summary": {
                    "total_records": final_state.total_records,
                    "successful_records": final_state.successful_records,
                    "failed_records": final_state.failed_records,
                    "success_rate": final_state.successful_records / final_state.total_records if final_state.total_records > 0 else 0,
                    "duration_seconds": final_state.end_time - final_state.start_time if final_state.end_time else None,
                    "errors": final_state.errors,
                    "warnings": final_state.warnings
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "state": initial_state.to_dict()
            }
    
    async def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents."""
        status = {}
        for name, agent in self.agents.items():
            status[name] = agent.get_status()
        return status
    
    async def cleanup(self):
        """Cleanup resources."""
        for agent in self.agents.values():
            await agent.cleanup()
