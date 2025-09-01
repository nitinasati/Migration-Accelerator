"""
PostgreSQL-based memory store for LangGraph state persistence.
"""

import json
import uuid
from typing import Any, Dict, List, Optional, Sequence
from datetime import datetime
import asyncpg

from config.database import DatabaseConfig, MigrationRun, WorkflowState, WorkflowCheckpoint


class PostgresMemoryStore:
    """
    PostgreSQL-based memory store for LangGraph state persistence.
    
    This class provides persistent storage of workflow states in PostgreSQL.
    """
    
    def __init__(self, config: DatabaseConfig):
        """Initialize the PostgreSQL memory store."""
        self.config = config
        self._pool: Optional[asyncpg.Pool] = None
        self._current_run_id: Optional[str] = None
        self._step_counter = 0
    
    async def initialize(self) -> None:
        """Initialize the database connection pool."""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                host=self.config.host,
                port=self.config.port,
                user=self.config.username,
                password=self.config.password,
                database=self.config.database,
                min_size=1,
                max_size=10
            )
    
    async def close(self) -> None:
        """Close the database connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
    
    async def create_migration_run(
        self, 
        file_path: str, 
        record_type: str
    ) -> str:
        """Create a new migration run and return its ID."""
        await self.initialize()
        
        run_id = str(uuid.uuid4())
        self._current_run_id = run_id
        self._step_counter = 0
        
        async with self._pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO migration_runs (id, file_path, record_type, status)
                VALUES ($1, $2, $3, $4)
            """, run_id, file_path, record_type, "running")
        
        return run_id
    
    async def update_migration_run_status(
        self, 
        run_id: str, 
        status: str, 
        success: Optional[bool] = None,
        total_duration: Optional[float] = None,
        error_message: Optional[str] = None
    ) -> None:
        """Update the status of a migration run."""
        await self.initialize()
        
        async with self._pool.acquire() as conn:
            if status == "completed":
                await conn.execute("""
                    UPDATE migration_runs 
                    SET status = $1, success = $2, total_duration = $3, 
                        completed_at = NOW(), updated_at = NOW()
                    WHERE id = $4
                """, status, success, total_duration, run_id)
            elif status == "failed":
                await conn.execute("""
                    UPDATE migration_runs 
                    SET status = $1, success = $2, error_message = $3, 
                        updated_at = NOW()
                    WHERE id = $4
                """, status, success, error_message, run_id)
            else:
                await conn.execute("""
                    UPDATE migration_runs 
                    SET status = $1, updated_at = NOW()
                    WHERE id = $4
                """, status, run_id)
    
    async def save_state(
        self,
        step_name: str,
        state: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save a workflow state to the database."""
        if not self._current_run_id:
            raise ValueError("No migration run created. Call create_migration_run first.")

        await self.initialize()

        state_id = str(uuid.uuid4())
        self._step_counter += 1

        # Serialize the state data
        state_data = self._serialize_state(state)
        
        # Serialize metadata if provided
        metadata_json = None
        if metadata:
            metadata_json = self._serialize_state(metadata)

        async with self._pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO workflow_states (id, run_id, step_name, state_data, step_order, metadata)
                VALUES ($1, $2, $3, $4, $5, $6)
            """, state_id, self._current_run_id, step_name, state_data, self._step_counter, metadata_json)

        return state_id
    
    async def save_checkpoint(
        self, 
        checkpoint_name: str, 
        state: Dict[str, Any], 
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save a workflow checkpoint to the database."""
        if not self._current_run_id:
            raise ValueError("No migration run created. Call create_migration_run first.")
        
        await self.initialize()
        
        checkpoint_id = str(uuid.uuid4())
        
        # Serialize the state data
        state_data = self._serialize_state(state)
        
        # Serialize metadata if provided
        metadata_json = None
        if metadata:
            metadata_json = self._serialize_state(metadata)
        
        async with self._pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO workflow_checkpoints (id, run_id, checkpoint_name, state_data, metadata)
                VALUES ($1, $2, $3, $4, $5)
            """, checkpoint_id, self._current_run_id, checkpoint_name, state_data, metadata_json)
        
        return checkpoint_id
    
    async def get_checkpoint(self, checkpoint_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve a workflow checkpoint from the database."""
        if not self._current_run_id:
            return None
        
        await self.initialize()
        
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT state_data 
                FROM workflow_checkpoints 
                WHERE run_id = $1 AND checkpoint_name = $2
                ORDER BY created_at DESC 
                LIMIT 1
            """, self._current_run_id, checkpoint_name)
        
        if row:
            return self._deserialize_state(row['state_data'])
        return None
    
    async def get_latest_state(self) -> Optional[Dict[str, Any]]:
        """Get the latest workflow state for the current run."""
        if not self._current_run_id:
            return None
        
        await self.initialize()
        
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT state_data 
                FROM workflow_states 
                WHERE run_id = $1
                ORDER BY step_order DESC 
                LIMIT 1
            """, self._current_run_id)
        
        if row:
            return self._deserialize_state(row['state_data'])
        return None
    
    async def get_run_history(self, run_id: str) -> List[Dict[str, Any]]:
        """Get the complete history of states for a migration run."""
        await self.initialize()
        
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT step_name, state_data, step_order, created_at, metadata
                FROM workflow_states 
                WHERE run_id = $1
                ORDER BY step_order ASC
            """, run_id)
        
        history = []
        for row in rows:
            history.append({
                "step_name": row['step_name'],
                "state_data": self._deserialize_state(row['state_data']),
                "step_order": row['step_order'],
                "created_at": row['created_at'].isoformat(),
                "metadata": row['metadata']
            })
        
        return history
    
    async def get_run_summary(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get a summary of a migration run."""
        await self.initialize()
        
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT id, file_path, record_type, status, created_at, updated_at,
                       completed_at, total_duration, success, error_message
                FROM migration_runs 
                WHERE id = $1
            """, run_id)
        
        if row:
            return {
                "id": str(row['id']),
                "file_path": row['file_path'],
                "record_type": row['record_type'],
                "status": row['status'],
                "created_at": row['created_at'].isoformat(),
                "updated_at": row['updated_at'].isoformat(),
                "completed_at": row['completed_at'].isoformat() if row['completed_at'] else None,
                "total_duration": row['total_duration'],
                "success": row['success'],
                "error_message": row['error_message']
            }
        return None
    
    def _serialize_state(self, state: Dict[str, Any]) -> str:
        """Serialize state data for database storage."""
        # Convert any non-serializable objects to strings
        serializable_state = {}
        for key, value in state.items():
            try:
                # Test if the value is JSON serializable
                json.dumps(value)
                serializable_state[key] = value
            except (TypeError, ValueError):
                # Convert non-serializable objects to string representation
                serializable_state[key] = str(value)
        
        return json.dumps(serializable_state, default=str)
    
    def _deserialize_state(self, state_data: str) -> Dict[str, Any]:
        """Deserialize state data from database storage."""
        try:
            return json.loads(state_data)
        except json.JSONDecodeError:
            return {"error": "Failed to deserialize state data", "raw_data": state_data}
    



class PostgresMemoryManager:
    """Manager class for PostgreSQL memory operations."""
    
    def __init__(self, config: DatabaseConfig):
        """Initialize the memory manager."""
        self.config = config
        self.memory_store: Optional[PostgresMemoryStore] = None
    
    async def initialize(self) -> None:
        """Initialize the memory store."""
        self.memory_store = PostgresMemoryStore(self.config)
        await self.memory_store.initialize()
    
    async def close(self) -> None:
        """Close the memory manager."""
        if self.memory_store:
            await self.memory_store.close()
    
    def get_memory_store(self) -> PostgresMemoryStore:
        """Get the memory store instance."""
        if not self.memory_store:
            raise RuntimeError("Memory manager not initialized. Call initialize() first.")
        return self.memory_store
