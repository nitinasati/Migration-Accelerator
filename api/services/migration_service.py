"""
Async migration service for handling background migration tasks.
"""

import asyncio
import uuid
import logging
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from pathlib import Path

from models.requests import MigrationTriggerRequest
from models.responses import MigrationStatus, MigrationTriggerResponse, MigrationStatusResponse, MigrationProgress


# Configure logging
logger = logging.getLogger(__name__)

# Global storage for running migrations (in production, use Redis/DB)
_running_migrations: Dict[str, Dict[str, Any]] = {}
_migration_lock = asyncio.Lock()


class MigrationService:
    """Service for managing async migrations."""
    
    def __init__(self):
        self.logger = logger
    
    async def trigger_migration(self, request: MigrationTriggerRequest) -> MigrationTriggerResponse:
        """
        Trigger a new migration asynchronously.
        
        Args:
            request: Migration trigger request
            
        Returns:
            MigrationTriggerResponse: Response with run ID and status
        """
        # Generate unique run ID
        run_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc)
        
        # Validate input file exists
        if not Path(request.file_path).exists():
            raise FileNotFoundError(f"Input file not found: {request.file_path}")
        
        # Initialize migration state
        migration_state = {
            "run_id": run_id,
            "status": MigrationStatus.PENDING,
            "created_at": created_at,
            "started_at": None,
            "completed_at": None,
            "file_path": request.file_path,
            "record_type": request.record_type,
            "output_mode": request.output_mode,
            "output_dir": request.output_dir,
            "batch_size": request.batch_size,
            "authentication": request.authentication,
            "metadata": request.metadata or {},
            "progress": {
                "total_records": 0,
                "processed_records": 0,
                "success_count": 0,
                "error_count": 0,
                "progress_percentage": 0.0,
                "current_step": "initializing"
            },
            "error_message": None,
            "duration": None
        }
        
        # Store migration state
        async with _migration_lock:
            _running_migrations[run_id] = migration_state
        
        # Start migration in background
        migration_task = asyncio.create_task(self._execute_migration(run_id, request))
        # Store task reference to prevent garbage collection
        migration_state["_task"] = migration_task
        
        self.logger.info(f"Migration triggered successfully - run_id={run_id}, file_path={request.file_path}, record_type={request.record_type}")
        
        return MigrationTriggerResponse(
            run_id=run_id,
            status=MigrationStatus.PENDING,
            message="Migration queued successfully",
            created_at=created_at,
            estimated_duration="5-10 minutes"
        )
    
    async def get_migration_status(self, run_id: str, include_details: bool = False) -> Optional[MigrationStatusResponse]:
        """
        Get the status of a migration.
        
        Args:
            run_id: Migration run ID
            include_details: Include detailed progress information
            
        Returns:
            MigrationStatusResponse: Migration status and progress
        """
        async with _migration_lock:
            migration_state = _running_migrations.get(run_id)
        
        if not migration_state:
            return None
        
        progress = None
        if include_details:
            progress_data = migration_state["progress"]
            progress = MigrationProgress(
                total_records=progress_data["total_records"],
                processed_records=progress_data["processed_records"],
                success_count=progress_data["success_count"],
                error_count=progress_data["error_count"],
                progress_percentage=progress_data["progress_percentage"],
                current_step=progress_data["current_step"]
            )
        
        return MigrationStatusResponse(
            run_id=run_id,
            status=migration_state["status"],
            progress=progress,
            created_at=migration_state["created_at"],
            started_at=migration_state["started_at"],
            completed_at=migration_state["completed_at"],
            duration=migration_state["duration"],
            error_message=migration_state["error_message"],
            file_path=migration_state["file_path"],
            record_type=migration_state["record_type"],
            metadata=migration_state["metadata"]
        )
    
    async def list_running_migrations(self) -> Dict[str, Dict[str, Any]]:
        """Get all running migrations."""
        async with _migration_lock:
            return _running_migrations.copy()
    
    async def cancel_migration(self, run_id: str) -> bool:
        """
        Cancel a running migration.
        
        Args:
            run_id: Migration run ID
            
        Returns:
            bool: True if cancelled successfully
        """
        async with _migration_lock:
            migration_state = _running_migrations.get(run_id)
            if migration_state and migration_state["status"] in [MigrationStatus.PENDING, MigrationStatus.RUNNING]:
                migration_state["status"] = MigrationStatus.CANCELLED
                migration_state["completed_at"] = datetime.now(timezone.utc)
                if migration_state["started_at"]:
                    migration_state["duration"] = (migration_state["completed_at"] - migration_state["started_at"]).total_seconds()
                return True
        return False
    
    async def _execute_migration(self, run_id: str, request: MigrationTriggerRequest):
        """
        Execute the migration in the background.
        
        Args:
            run_id: Migration run ID
            request: Migration request
        """
        start_time = time.time()
        
        try:
            # Update status to running
            async with _migration_lock:
                migration_state = _running_migrations[run_id]
                migration_state["status"] = MigrationStatus.RUNNING
                migration_state["started_at"] = datetime.now(timezone.utc)
                migration_state["progress"]["current_step"] = "starting"
            
            self.logger.info(f"Starting migration execution - run_id={run_id}")
            
            # Import the workflow here to avoid circular imports
            import sys
            import os
            
            # Get the project root directory (parent of api directory)
            current_dir = os.path.dirname(os.path.abspath(__file__))  # services directory
            api_dir = os.path.dirname(current_dir)  # api directory
            project_root = os.path.dirname(api_dir)  # project root
            
            # Add project root to Python path
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            
            self.logger.info(f"Added to Python path: {project_root}")
            
            try:
                from workflows.migration_graph import MigrationWorkflow
                from config.settings import settings, get_llm_config, get_mcp_config
                
                # Initialize workflow
                llm_config = get_llm_config()
                mcp_config = get_mcp_config()
                workflow = MigrationWorkflow(llm_config, mcp_config)
                
            except ImportError as e:
                self.logger.error(f"Failed to import workflow modules: {e}")
                self.logger.error(f"Current Python path: {sys.path}")
                self.logger.error(f"Project root: {project_root}")
                self.logger.error(f"Contents of project root: {os.listdir(project_root) if os.path.exists(project_root) else 'Directory not found'}")
                raise ImportError(f"Could not import workflow modules. Project structure issue: {e}")
            except Exception as e:
                self.logger.error(f"Failed to initialize workflow: {e}")
                raise
            
            # Update progress
            await self._update_progress(run_id, 0, 0, 0, 0, "workflow_initialized")
            
            # Execute migration workflow
            result = await workflow.run(
                file_path=request.file_path,
                record_type=request.record_type.value
            )
            
            # Update final progress
            if result.get("migration_summary", {}).get("success", False):
                total_records = result.get("migration_summary", {}).get("total_records_processed", 0)
                await self._update_progress(run_id, total_records, total_records, total_records, 0, "completed")
                
                # Update status to completed
                async with _migration_lock:
                    migration_state = _running_migrations[run_id]
                    migration_state["status"] = MigrationStatus.COMPLETED
                    migration_state["completed_at"] = datetime.now(timezone.utc)
                    migration_state["duration"] = time.time() - start_time
                    migration_state["metadata"]["result"] = result
                
                self.logger.info(f"Migration completed successfully - run_id={run_id}, duration={time.time() - start_time:.2f}s")
            else:
                # Migration failed
                errors = result.get("errors", ["Unknown error"])
                await self._update_progress(run_id, 0, 0, 0, 1, "failed")
                
                async with _migration_lock:
                    migration_state = _running_migrations[run_id]
                    migration_state["status"] = MigrationStatus.FAILED
                    migration_state["completed_at"] = datetime.now(timezone.utc)
                    migration_state["duration"] = time.time() - start_time
                    migration_state["error_message"] = "; ".join(errors)
                
                self.logger.error(f"Migration failed - run_id={run_id}, errors={errors}")
            
            # Close workflow
            await workflow.close()
            
        except Exception as e:
            # Handle migration error
            async with _migration_lock:
                migration_state = _running_migrations[run_id]
                migration_state["status"] = MigrationStatus.FAILED
                migration_state["completed_at"] = datetime.now(timezone.utc)
                migration_state["duration"] = time.time() - start_time
                migration_state["error_message"] = str(e)
            
            self.logger.error(f"Migration execution failed - run_id={run_id}, error={str(e)}", exc_info=True)
    
    async def _update_progress(self, run_id: str, total: int, processed: int, success: int, errors: int, step: str):
        """Update migration progress."""
        progress_percentage = (processed / total * 100) if total > 0 else 0
        
        async with _migration_lock:
            migration_state = _running_migrations.get(run_id)
            if migration_state:
                migration_state["progress"].update({
                    "total_records": total,
                    "processed_records": processed,
                    "success_count": success,
                    "error_count": errors,
                    "progress_percentage": progress_percentage,
                    "current_step": step
                })


# Global service instance
migration_service = MigrationService()
