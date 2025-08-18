"""
Base agent class for A2A (Agent-to-Agent) framework.
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum

import structlog
from pydantic import BaseModel

from llm.providers import LLMProviderBase
from mcp.client import MCPToolManager
from config.settings import LLMConfig, MCPConfig


class AgentStatus(str, Enum):
    """Agent status enumeration."""
    IDLE = "idle"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    WAITING = "waiting"


class AgentRole(str, Enum):
    """Agent role enumeration."""
    FILE_READER = "file_reader"
    VALIDATOR = "validator"
    MAPPER = "mapper"
    TRANSFORMER = "transformer"
    API_INTEGRATOR = "api_integrator"
    ORCHESTRATOR = "orchestrator"


@dataclass
class AgentMetrics:
    """Agent performance metrics."""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration_seconds: Optional[float] = None
    records_processed: int = 0
    errors_count: int = 0
    warnings_count: int = 0
    llm_calls: int = 0
    mcp_calls: int = 0
    
    def finish(self):
        """Mark the agent as finished and calculate duration."""
        self.end_time = time.time()
        self.duration_seconds = self.end_time - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
            "records_processed": self.records_processed,
            "errors_count": self.errors_count,
            "warnings_count": self.warnings_count,
            "llm_calls": self.llm_calls,
            "mcp_calls": self.mcp_calls
        }


class AgentMessage(BaseModel):
    """Message for inter-agent communication."""
    sender: str
    recipient: str
    message_type: str
    payload: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    correlation_id: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True


class BaseAgent(ABC):
    """Base class for all agents in the A2A framework."""
    
    def __init__(
        self,
        name: str,
        role: AgentRole,
        llm_provider: Optional[LLMProviderBase] = None,
        mcp_manager: Optional[MCPToolManager] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.role = role
        self.llm_provider = llm_provider
        self.mcp_manager = mcp_manager
        self.config = config or {}
        
        # State management
        self.status = AgentStatus.IDLE
        self.metrics = AgentMetrics()
        self.logger = structlog.get_logger(name)
        
        # Message queue for inter-agent communication
        self.message_queue: List[AgentMessage] = []
        self.dependencies: List[str] = []
        self.next_agents: List[str] = []
    
    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data and return results."""
        pass
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent with proper error handling and metrics."""
        try:
            self.status = AgentStatus.PROCESSING
            self.logger.info("Agent started", agent_name=self.name, role=self.role.value)
            
            # Process the data
            result = await self.process(input_data)
            
            # Update metrics
            self.metrics.finish()
            self.status = AgentStatus.COMPLETED
            
            self.logger.info(
                "Agent completed",
                agent_name=self.name,
                duration=self.metrics.duration_seconds,
                records_processed=self.metrics.records_processed
            )
            
            return result
            
        except Exception as e:
            self.status = AgentStatus.FAILED
            self.metrics.finish()
            
            self.logger.error(
                "Agent failed",
                agent_name=self.name,
                error=str(e),
                duration=self.metrics.duration_seconds
            )
            
            raise
    
    async def coordinate(self, task_description: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate with other agents using A2A framework."""
        if not self.llm_provider:
            raise ValueError("LLM provider required for coordination")
        
        prompt = f"""
You are a {self.role.value} agent in a multi-agent migration system. Your task is:

{task_description}

Context:
{json.dumps(context, indent=2)}

Please coordinate with other agents and provide your response in JSON format:
{{
    "status": "ready|processing|completed|failed",
    "result": "your_result_here",
    "next_agent": "next_agent_name",
    "dependencies": ["dependency1", "dependency2"],
    "estimated_duration": "estimated_time_in_seconds"
}}
"""
        
        try:
            response = await self.llm_provider.generate(prompt)
            self.metrics.llm_calls += 1
            
            # Parse JSON response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {"status": "failed", "error": "Invalid JSON response"}
                
        except Exception as e:
            self.logger.error("Coordination failed", error=str(e))
            return {"status": "failed", "error": str(e)}
    
    async def handoff(self, next_agent: str, handoff_data: Dict[str, Any], task_context: Dict[str, Any]) -> Dict[str, Any]:
        """Handoff to the next agent."""
        if not self.llm_provider:
            raise ValueError("LLM provider required for handoff")
        
        prompt = f"""
Agent handoff from {self.name} to {next_agent}:

Task Context:
{json.dumps(task_context, indent=2)}

Handoff Data:
{json.dumps(handoff_data, indent=2)}

Please acknowledge the handoff and provide your plan in JSON format:
{{
    "acknowledged": true/false,
    "plan": "your_execution_plan",
    "estimated_duration": "estimated_time_in_seconds",
    "dependencies": ["dependency1", "dependency2"],
    "risks": ["risk1", "risk2"]
}}
"""
        
        try:
            response = await self.llm_provider.generate(prompt)
            self.metrics.llm_calls += 1
            
            # Parse JSON response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {"acknowledged": False, "error": "Invalid JSON response"}
                
        except Exception as e:
            self.logger.error("Handoff failed", error=str(e))
            return {"acknowledged": False, "error": str(e)}
    
    def send_message(self, recipient: str, message_type: str, payload: Dict[str, Any], correlation_id: Optional[str] = None):
        """Send a message to another agent."""
        message = AgentMessage(
            sender=self.name,
            recipient=recipient,
            message_type=message_type,
            payload=payload,
            correlation_id=correlation_id
        )
        
        self.message_queue.append(message)
        self.logger.info("Message sent", recipient=recipient, message_type=message_type)
    
    def receive_message(self, sender: str, message_type: str, payload: Dict[str, Any], correlation_id: Optional[str] = None):
        """Receive a message from another agent."""
        message = AgentMessage(
            sender=sender,
            recipient=self.name,
            message_type=message_type,
            payload=payload,
            correlation_id=correlation_id
        )
        
        self.message_queue.append(message)
        self.logger.info("Message received", sender=sender, message_type=message_type)
    
    async def call_mcp_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Call an MCP tool."""
        if not self.mcp_manager:
            raise ValueError("MCP manager required for tool calls")
        
        try:
            result = await self.mcp_manager.execute_tool(tool_name, parameters)
            self.metrics.mcp_calls += 1
            return result
        except Exception as e:
            self.logger.error("MCP tool call failed", tool_name=tool_name, error=str(e))
            raise
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        return {
            "name": self.name,
            "role": self.role.value,
            "status": self.status.value,
            "metrics": self.metrics.to_dict(),
            "dependencies": self.dependencies,
            "next_agents": self.next_agents,
            "message_queue_length": len(self.message_queue)
        }
    
    def add_dependency(self, agent_name: str):
        """Add a dependency on another agent."""
        if agent_name not in self.dependencies:
            self.dependencies.append(agent_name)
    
    def add_next_agent(self, agent_name: str):
        """Add the next agent in the workflow."""
        if agent_name not in self.next_agents:
            self.next_agents.append(agent_name)
    
    async def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data."""
        # Base implementation - can be overridden by subclasses
        return True
    
    async def cleanup(self):
        """Cleanup resources."""
        self.status = AgentStatus.IDLE
        self.message_queue.clear()
        self.logger.info("Agent cleanup completed", agent_name=self.name)
    
    def __str__(self) -> str:
        return f"{self.name} ({self.role.value})"
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name='{self.name}' role='{self.role.value}' status='{self.status.value}'>"
