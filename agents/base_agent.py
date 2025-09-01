"""
Base agent class for the Migration-Accelerators platform.
"""

import asyncio
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import structlog

from config.settings import LLMConfig, MCPConfig


class AgentMessage:
    """Message structure for agent communication."""
    
    def __init__(
        self,
        sender: str,
        recipient: str,
        message_type: str,
        data: Any,
        metadata: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ):
        self.id = str(uuid.uuid4())
        self.sender = sender
        self.recipient = recipient
        self.message_type = message_type
        self.data = data
        self.metadata = metadata or {}
        self.correlation_id = correlation_id or str(uuid.uuid4())
        from datetime import timezone
        self.timestamp = datetime.now(timezone.utc)
        self.status = "pending"


class AgentResult:
    """Result structure for agent operations."""
    
    def __init__(
        self,
        success: bool,
        data: Any = None,
        errors: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.success = success
        self.data = data
        self.errors = errors or []
        self.warnings = warnings or []
        self.metadata = metadata or {}
        from datetime import timezone
        self.timestamp = datetime.now(timezone.utc)


class BaseAgent(ABC):
    """
    Base class for all agents in the Migration-Accelerators platform.
    
    This class provides common functionality for all agents including:
    - Logging and monitoring
    - Error handling and retry logic
    - Message passing between agents
    - Configuration management
    """
    
    def __init__(
        self,
        agent_name: str,
        llm_config: Optional[LLMConfig] = None,
        mcp_config: Optional[MCPConfig] = None
    ):
        self.agent_name = agent_name
        self.llm_config = llm_config
        self.mcp_config = mcp_config
        self.logger = structlog.get_logger().bind(agent=agent_name)
        self.message_queue: List[AgentMessage] = []
        self.is_running = False
        self.correlation_id = str(uuid.uuid4())
        
        self.logger.info("Agent initialized", agent_name=agent_name)
    
    @abstractmethod
    async def process(self, data: Any, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """
        Main processing method that must be implemented by subclasses.
        
        Args:
            data: Input data to process
            context: Additional context information
            
        Returns:
            AgentResult: Processing result
        """
        pass
    
    def validate_input(self, data: Any) -> bool:
        """
        Validate input data before processing.
        
        Args:
            data: Input data to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            if data is None:
                self.logger.warning("Input data is None")
                return False
            
            if isinstance(data, (list, dict)) and len(data) == 0:
                self.logger.warning("Input data is empty")
                return False
            
            return True
        except Exception as e:
            self.logger.error("Error validating input", error=str(e))
            return False
    
    async def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """
        Handle errors gracefully with proper logging and error recovery.
        
        Args:
            error: Exception that occurred
            context: Additional context information
            
        Returns:
            AgentResult: Error result
        """
        error_msg = f"Error in {self.agent_name}: {str(error)}"
        self.logger.error(
            "Agent error occurred",
            error=str(error),
            error_type=type(error).__name__,
            context=context
        )
        
        return AgentResult(
            success=False,
            errors=[error_msg],
            metadata={"error_type": type(error).__name__, "context": context}
        )
    
    async def retry_with_backoff(
        self,
        func,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute function with exponential backoff retry logic.
        
        Args:
            func: Function to execute
            max_retries: Maximum number of retries
            base_delay: Base delay in seconds
            max_delay: Maximum delay in seconds
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Any: Function result
            
        Raises:
            Exception: Last exception if all retries fail
        """
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt == max_retries:
                    self.logger.error(
                        "Max retries exceeded",
                        function=func.__name__,
                        attempts=attempt + 1,
                        error=str(e)
                    )
                    break
                
                delay = min(base_delay * (2 ** attempt), max_delay)
                self.logger.warning(
                    "Retry attempt failed, retrying",
                    function=func.__name__,
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    delay=delay,
                    error=str(e)
                )
                
                await asyncio.sleep(delay)
        
        raise last_exception
    
    def send_message(
        self,
        recipient: str,
        message_type: str,
        data: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AgentMessage:
        """
        Send a message to another agent.
        
        Args:
            recipient: Recipient agent name
            message_type: Type of message
            data: Message data
            metadata: Additional metadata
            
        Returns:
            AgentMessage: Created message
        """
        message = AgentMessage(
            sender=self.agent_name,
            recipient=recipient,
            message_type=message_type,
            data=data,
            metadata=metadata,
            correlation_id=self.correlation_id
        )
        
        self.logger.info(
            "Message sent",
            recipient=recipient,
            message_type=message_type,
            message_id=message.id
        )
        
        return message
    
    def receive_message(self, message: AgentMessage) -> None:
        """
        Receive a message from another agent.
        
        Args:
            message: Received message
        """
        self.message_queue.append(message)
        self.logger.info(
            "Message received",
            sender=message.sender,
            message_type=message.message_type,
            message_id=message.id
        )
    
    async def process_messages(self) -> List[AgentResult]:
        """
        Process all pending messages.
        
        Returns:
            List[AgentResult]: Results from processing messages
        """
        results = []
        
        while self.message_queue:
            message = self.message_queue.pop(0)
            try:
                result = await self._handle_message(message)
                results.append(result)
            except Exception as e:
                error_result = await self.handle_error(e, {"message_id": message.id})
                results.append(error_result)
        
        return results
    
    async def _handle_message(self, message: AgentMessage) -> AgentResult:
        """
        Handle a specific message.
        
        Args:
            message: Message to handle
            
        Returns:
            AgentResult: Processing result
        """
        self.logger.info(
            "Processing message",
            message_id=message.id,
            message_type=message.message_type
        )
        
        # Default message handling - can be overridden by subclasses
        return AgentResult(
            success=True,
            data=message.data,
            metadata={"message_id": message.id, "message_type": message.message_type}
        )
    
    async def start(self) -> None:
        """Start the agent."""
        self.is_running = True
        self.logger.info("Agent started")
    
    async def stop(self) -> None:
        """Stop the agent."""
        self.is_running = False
        self.logger.info("Agent stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get agent status information.
        
        Returns:
            Dict[str, Any]: Status information
        """
        return {
            "agent_name": self.agent_name,
            "is_running": self.is_running,
            "correlation_id": self.correlation_id,
            "pending_messages": len(self.message_queue),
            "llm_configured": self.llm_config is not None,
            "mcp_configured": self.mcp_config is not None
        }
    
    async def health_check(self) -> bool:
        """
        Perform health check for the agent.
        
        Returns:
            bool: True if healthy, False otherwise
        """
        try:
            # Basic health check - can be overridden by subclasses
            return self.is_running and not any(
                msg.status == "error" for msg in self.message_queue
            )
        except Exception as e:
            self.logger.error("Health check failed", error=str(e))
            return False
    
    def __str__(self) -> str:
        """String representation of the agent."""
        return f"{self.__class__.__name__}(name={self.agent_name})"
    
    async def close(self) -> None:
        """Close the agent and clean up resources."""
        if self.is_running:
            await self.stop()
        self.logger.info("Agent closed", agent=self.agent_name)
    
    def __repr__(self) -> str:
        """Detailed string representation of the agent."""
        return (
            f"{self.__class__.__name__}("
            f"name={self.agent_name}, "
            f"running={self.is_running}, "
            f"messages={len(self.message_queue)})"
        )
