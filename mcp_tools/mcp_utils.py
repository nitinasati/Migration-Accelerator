"""
MCP utilities for common event processing and logging functionality.

This module provides reusable functions for processing LangGraph/MCP events
that can be used across agents, MCP clients, and servers.
"""

import structlog
from typing import Dict, Any, List, Optional


def process_chain_start_event(event: Dict[str, Any], events: List[Dict[str, Any]], logger: structlog.BoundLogger) -> None:
    """Process chain start event."""
    try:
        event_data = event.get('data', {})
        if isinstance(event_data, dict) and 'input' in event_data:
            logger.info("🚀 Agent execution started")
            events.append({"type": "chain_start", "data": event_data})
    except Exception as e:
        logger.error("Error processing chain start event", error=str(e))


def process_tool_start_event(event: Dict[str, Any], events: List[Dict[str, Any]], logger: structlog.BoundLogger) -> None:
    """Process tool start event and log tool execution."""
    try:
        event_data = event.get('data', {})
        if isinstance(event_data, dict) and 'input' in event_data:
            tool_name = event.get('name', 'unknown_tool')
            logger.info("🔧 Tool execution started", tool_name=tool_name)
            events.append({"type": "tool_start", "name": tool_name, "input": event_data.get('input')})
    except Exception as e:
        logger.error("Error processing tool start event", error=str(e))


def process_tool_end_event(event: Dict[str, Any], events: List[Dict[str, Any]], logger: structlog.BoundLogger) -> None:
    """Process tool end event and log tool completion."""
    try:
        event_data = event.get('data', {})
        if isinstance(event_data, dict):
            tool_name = event.get('name', 'unknown_tool')
            logger.info("✅ Tool execution completed", tool_name=tool_name)
            events.append({"type": "tool_end", "name": tool_name, "output": event_data.get('output')})
    except Exception as e:
        logger.error("Error processing tool end event", error=str(e))


def process_chain_end_event(event: Dict[str, Any], events: List[Dict[str, Any]], logger: structlog.BoundLogger) -> Optional[str]:
    """Process chain end event and return final output."""
    try:
        event_data = event.get('data', {})
        if not isinstance(event_data, dict):
            # This is normal for some LangGraph events, skip silently
            return None

        output_data = event_data.get('output', {})
        if not isinstance(output_data, dict):
            # Different event types have different output formats, this is expected
            return None

        messages = output_data.get('messages', [])
        if messages and isinstance(messages, list):
            final_message = messages[-1]
            if hasattr(final_message, 'content'):
                final_output = final_message.content
                logger.info("🏁 Final Output received")
                events.append({"type": "chain_end", "final_output": final_output})
                return final_output

        return None

    except Exception as e:
        logger.error("Error processing chain end event", error=str(e), event=event)
        return None


def process_langgraph_event(event: Dict[str, Any], events: List[Dict[str, Any]], logger: structlog.BoundLogger) -> Optional[str]:
    """
    Process a single LangGraph event and route to appropriate handler.
    
    Args:
        event: The event dictionary from LangGraph stream
        events: List to append processed events to
        logger: Structured logger instance
        
    Returns:
        Optional[str]: Final output if this is a chain end event, None otherwise
    """
    event_type = event.get('event')
    
    if event_type == 'on_chain_start':
        process_chain_start_event(event, events, logger)
    elif event_type == 'on_tool_start':
        process_tool_start_event(event, events, logger)
    elif event_type == 'on_tool_end':
        process_tool_end_event(event, events, logger)
    elif event_type == 'on_chain_end':
        return process_chain_end_event(event, events, logger)
    
    return None


async def process_agent_stream(agent, query: str, logger: structlog.BoundLogger) -> Dict[str, Any]:
    """
    Process an agent query with event streaming and return results.
    
    Args:
        agent: The LangGraph agent instance
        query: The query string to send to the agent
        logger: Structured logger instance
        
    Returns:
        Dict containing success status, events, and final output
    """
    events = []
    final_output = None
    
    try:
        logger.info("Starting agent query execution", query_length=len(query))
        
        async for event in agent.astream_events({"messages": [{"role": "user", "content": query}]}, version="v1"):
            result = process_langgraph_event(event, events, logger)
            if result is not None:
                final_output = result
        
        if final_output:
            logger.info("Agent execution completed successfully", events_count=len(events))
            return {
                "success": True,
                "events": events,
                "final_output": final_output
            }
        else:
            logger.warning("Agent execution completed but no final output received")
            return {
                "success": False,
                "error": "No final output received from agent",
                "events": events
            }
            
    except Exception as e:
        logger.error("Agent execution failed", error=str(e))
        return {
            "success": False,
            "error": f"Agent execution failed: {str(e)}",
            "events": events
        }
