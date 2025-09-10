"""
Memory management utilities for LangGraph agents.

This module provides common memory management functionality that can be
shared across all agents in the Migration-Accelerator platform.
"""

import uuid
from typing import Any, Dict, List, Optional
import structlog

try:
    from langgraph.checkpoint.base import BaseCheckpointSaver
except ImportError:
    BaseCheckpointSaver = None


class MemoryManager:
    """
    Common memory management functionality for LangGraph agents.
    
    This class provides methods for managing conversation history,
    thread IDs, and checkpointer state that can be used by any agent.
    """
    
    def __init__(
        self, 
        checkpointer: Optional[BaseCheckpointSaver] = None,
        thread_id: Optional[str] = None,
        agent_name: str = "agent"
    ):
        """
        Initialize memory manager.
        
        Args:
            checkpointer: LangGraph checkpointer instance
            thread_id: Memory thread ID (generates one if not provided)
            agent_name: Name of the agent for logging
        """
        self.checkpointer = checkpointer
        self.thread_id = thread_id or f"{agent_name}_{uuid.uuid4().hex[:8]}"
        self.agent_name = agent_name
        self.logger = structlog.get_logger().bind(agent=agent_name)
    
    def get_thread_id(self) -> str:
        """Get the current memory thread ID."""
        return self.thread_id
    
    def set_thread_id(self, thread_id: str) -> None:
        """Set a new memory thread ID for conversation continuity."""
        old_thread_id = self.thread_id
        self.thread_id = thread_id
        self.logger.info("Memory thread ID updated", 
                        old_thread_id=old_thread_id,
                        new_thread_id=thread_id)
    
    def is_memory_enabled(self) -> bool:
        """Check if memory is enabled (checkpointer available)."""
        return self.checkpointer is not None
    
    async def get_conversation_history(self) -> Optional[List[Dict[str, Any]]]:
        """Get the conversation history for the current thread."""
        if not self.checkpointer:
            self.logger.warning("No memory checkpointer available")
            return None
        
        try:
            # Get the latest checkpoint for this thread
            config = {"configurable": {"thread_id": self.thread_id}}
            checkpoint = await self.checkpointer.aget(config)
            
            if checkpoint and hasattr(checkpoint, 'channel_values'):
                messages = checkpoint.channel_values.get('messages', [])
                return [
                    {"role": msg.type, "content": msg.content} 
                    for msg in messages 
                    if hasattr(msg, 'content')
                ]
            
            return []
        except Exception as e:
            self.logger.error("Failed to get conversation history", error=str(e))
            return None
    
    async def clear_memory(self) -> bool:
        """Clear the memory for the current thread."""
        if not self.checkpointer:
            self.logger.warning("No memory checkpointer available")
            return False
        
        try:
            # Note: PostgresSaver might not have a delete method, 
            # so we'll create a new thread ID to effectively clear memory
            old_thread_id = self.thread_id
            self.thread_id = f"{self.agent_name}_{uuid.uuid4().hex[:8]}"
            
            self.logger.info("Memory cleared by creating new thread", 
                            old_thread_id=old_thread_id,
                            new_thread_id=self.thread_id)
            return True
        except Exception as e:
            self.logger.error("Failed to clear memory", error=str(e))
            return False
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics for the current thread."""
        if not self.checkpointer:
            return {
                "memory_enabled": False, 
                "message": "No memory checkpointer available",
                "thread_id": self.thread_id
            }
        
        try:
            history = await self.get_conversation_history()
            return {
                "memory_enabled": True,
                "thread_id": self.thread_id,
                "message_count": len(history) if history else 0,
                "checkpointer_type": type(self.checkpointer).__name__,
                "agent_name": self.agent_name
            }
        except Exception as e:
            return {
                "memory_enabled": True,
                "thread_id": self.thread_id,
                "error": str(e),
                "checkpointer_type": type(self.checkpointer).__name__,
                "agent_name": self.agent_name
            }
    
    def get_config_for_agent(self) -> Dict[str, Any]:
        """Get configuration dict for LangGraph agent execution."""
        if not self.checkpointer:
            return {}
        
        return {"configurable": {"thread_id": self.thread_id}}
    
    async def get_last_message(self) -> Optional[Dict[str, Any]]:
        """Get the last message from conversation history."""
        history = await self.get_conversation_history()
        if history and len(history) > 0:
            return history[-1]
        return None
    
    async def get_message_count(self) -> int:
        """Get the total number of messages in conversation history."""
        history = await self.get_conversation_history()
        return len(history) if history else 0
    
    def create_new_session(self) -> str:
        """Create a new memory session with a new thread ID."""
        old_thread_id = self.thread_id
        self.thread_id = f"{self.agent_name}_{uuid.uuid4().hex[:8]}"
        
        self.logger.info("Created new memory session", 
                        old_thread_id=old_thread_id,
                        new_thread_id=self.thread_id)
        
        return self.thread_id


def create_memory_manager(
    checkpointer: Optional[BaseCheckpointSaver] = None,
    thread_id: Optional[str] = None,
    agent_name: str = "agent"
) -> MemoryManager:
    """
    Factory function to create a memory manager instance.
    
    Args:
        checkpointer: LangGraph checkpointer instance
        thread_id: Memory thread ID (generates one if not provided)
        agent_name: Name of the agent for logging
        
    Returns:
        MemoryManager: Configured memory manager instance
    """
    return MemoryManager(checkpointer, thread_id, agent_name)


def get_memory_config(memory_manager: MemoryManager) -> Dict[str, Any]:
    """
    Get LangGraph configuration for memory-enabled execution.
    
    Args:
        memory_manager: Memory manager instance
        
    Returns:
        Dict with configuration for LangGraph agent execution
    """
    return memory_manager.get_config_for_agent()


async def log_memory_info(memory_manager: MemoryManager, logger: structlog.BoundLogger) -> None:
    """
    Log memory information for debugging.
    
    Args:
        memory_manager: Memory manager instance
        logger: Structured logger instance
    """
    stats = await memory_manager.get_memory_stats()
    logger.info("Memory status", **stats)
