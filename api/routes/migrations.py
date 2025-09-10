"""
Migration-related API routes.
"""

import time
import logging
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from typing import List, Optional
from config.database import MigrationRun, WorkflowState, WorkflowCheckpoint, SearchFilters
from api.models.requests import MigrationTriggerRequest, MigrationStatusRequest
from api.models.responses import MigrationTriggerResponse, MigrationStatusResponse, ErrorResponse
from api.database import (
    get_migration_runs, get_migration_run, get_workflow_states,
    get_workflow_checkpoints, search_migration_runs, get_migration_stats
)
from api.services.migration_service import migration_service

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/", response_model=List[MigrationRun])
def list_migrations(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """List all migration runs."""
    start_time = time.time()
    logger.info(f"Starting list_migrations API call - limit={limit}, offset={offset}")
    
    try:
        migrations = get_migration_runs(limit=limit, offset=offset)
        
        execution_time = time.time() - start_time
        logger.info(f"list_migrations API call completed successfully - execution_time_ms={round(execution_time * 1000, 2)}, records_returned={len(migrations)}, limit={limit}, offset={offset}")
        
        return migrations
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"list_migrations API call failed - error={str(e)}, error_type={type(e).__name__}, execution_time_ms={round(execution_time * 1000, 2)}, limit={limit}, offset={offset}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch migrations: {str(e)}")

@router.get("/stats")
def get_stats():
    """Get migration statistics."""
    start_time = time.time()
    logger.info("Starting get_stats API call")
    
    try:
        stats = get_migration_stats()
        logger.info(f"Stats: {stats}")
        execution_time = time.time() - start_time
        logger.info(f"get_stats API call completed successfully - execution_time_ms={round(execution_time * 1000, 2)}, stats={stats}")
        
        return stats
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"get_stats API call failed - error={str(e)}, error_type={type(e).__name__}, execution_time_ms={round(execution_time * 1000, 2)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch stats: {str(e)}")

@router.get("/search/", response_model=List[MigrationRun])
def search_migrations(
    record_type: Optional[str] = Query(None, description="Filter by record type"),
    file_path: Optional[str] = Query(None, description="Filter by file path"),
    status: Optional[str] = Query(None, description="Filter by status")
):
    """Search migration runs based on filters."""
    start_time = time.time()
    logger.info(f"Starting search_migrations API call - record_type={record_type}, file_path={file_path}, status={status}")
    
    try:
        filters = SearchFilters(
            record_type=record_type,
            file_path=file_path,
            status=status
        )
        results = search_migration_runs(filters)
        
        execution_time = time.time() - start_time
        logger.info(f"search_migrations API call completed successfully - execution_time_ms={round(execution_time * 1000, 2)}, results_count={len(results)}, filters={filters.dict() if hasattr(filters, 'dict') else str(filters)}")
        
        return results
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"search_migrations API call failed - error={str(e)}, error_type={type(e).__name__}, execution_time_ms={round(execution_time * 1000, 2)}, filters=record_type={record_type}, file_path={file_path}, status={status}")
        raise HTTPException(status_code=500, detail=f"Failed to search migrations: {str(e)}")


@router.get("/running", response_model=dict)
async def list_running_migrations():
    """
    List all currently running migrations.
    
    Returns a dictionary of all migrations that are currently in progress,
    indexed by their run IDs.
    """
    start_time = time.time()
    logger.info("Starting list_running_migrations API call")
    
    try:
        running_migrations = await migration_service.list_running_migrations()
        
        execution_time = time.time() - start_time
        logger.info(f"list_running_migrations API call completed successfully - execution_time_ms={round(execution_time * 1000, 2)}, count={len(running_migrations)}")
        
        return {
            "running_migrations": running_migrations,
            "count": len(running_migrations)
        }
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"list_running_migrations API call failed - error={str(e)}, error_type={type(e).__name__}, execution_time_ms={round(execution_time * 1000, 2)}")
        raise HTTPException(status_code=500, detail=f"Failed to list running migrations: {str(e)}")


@router.get("/{run_id}", response_model=MigrationRun)
def get_migration(run_id: str):
    """Get a specific migration run by ID."""
    start_time = time.time()
    logger.info(f"Starting get_migration API call - run_id={run_id}")
    
    try:
        migration = get_migration_run(run_id)
        if not migration:
            execution_time = time.time() - start_time
            logger.warning(f"get_migration API call - migration not found - execution_time_ms={round(execution_time * 1000, 2)}, run_id={run_id}")
            raise HTTPException(status_code=404, detail="Migration run not found")
        
        execution_time = time.time() - start_time
        logger.info(f"get_migration API call completed successfully - execution_time_ms={round(execution_time * 1000, 2)}, run_id={run_id}, migration_found=True")
        
        return migration
        
    except HTTPException:
        raise
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"get_migration API call failed - error={str(e)}, error_type={type(e).__name__}, execution_time_ms={round(execution_time * 1000, 2)}, run_id={run_id}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch migration: {str(e)}")

@router.get("/{run_id}/states", response_model=List[WorkflowState])
def get_workflow_states_for_run(run_id: str):
    """Get workflow states for a specific migration run."""
    start_time = time.time()
    logger.info(f"Starting get_workflow_states_for_run API call - run_id={run_id}")
    
    try:
        states = get_workflow_states(run_id)
        
        execution_time = time.time() - start_time
        logger.info(f"get_workflow_states_for_run API call completed successfully - execution_time_ms={round(execution_time * 1000, 2)}, run_id={run_id}, states_count={len(states)}")
        
        return states
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"get_workflow_states_for_run API call failed - error={str(e)}, error_type={type(e).__name__}, execution_time_ms={round(execution_time * 1000, 2)}, run_id={run_id}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch workflow states: {str(e)}")

@router.get("/{run_id}/checkpoints", response_model=List[WorkflowCheckpoint])
def get_workflow_checkpoints_for_run(run_id: str):
    """Get workflow checkpoints for a specific migration run."""
    start_time = time.time()
    logger.info(f"Starting get_workflow_checkpoints_for_run API call - run_id={run_id}")
    
    try:
        checkpoints = get_workflow_checkpoints(run_id)
        
        execution_time = time.time() - start_time
        logger.info(f"get_workflow_checkpoints_for_run API call completed successfully - execution_time_ms={round(execution_time * 1000, 2)}, run_id={run_id}, checkpoints_count={len(checkpoints)}")
        
        return checkpoints
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"get_workflow_checkpoints_for_run API call failed - error={str(e)}, error_type={type(e).__name__}, execution_time_ms={round(execution_time * 1000, 2)}, run_id={run_id}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch workflow checkpoints: {str(e)}")


# === ASYNC MIGRATION ENDPOINTS ===

@router.post("/trigger", response_model=MigrationTriggerResponse)
async def trigger_migration(request: MigrationTriggerRequest):
    """
    Trigger a new migration asynchronously.
    
    This endpoint starts a migration in the background and immediately returns a run ID
    that can be used to check the migration status.
    """
    start_time = time.time()
    logger.info(f"Starting trigger_migration API call - file_path={request.file_path}, record_type={request.record_type}")
    
    try:
        response = await migration_service.trigger_migration(request)
        
        execution_time = time.time() - start_time
        logger.info(f"trigger_migration API call completed successfully - execution_time_ms={round(execution_time * 1000, 2)}, run_id={response.run_id}, file_path={request.file_path}")
        
        return response
        
    except FileNotFoundError as e:
        execution_time = time.time() - start_time
        logger.error(f"trigger_migration API call failed - file not found - execution_time_ms={round(execution_time * 1000, 2)}, file_path={request.file_path}, error={str(e)}")
        raise HTTPException(status_code=404, detail=f"Input file not found: {request.file_path}")
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"trigger_migration API call failed - error={str(e)}, error_type={type(e).__name__}, execution_time_ms={round(execution_time * 1000, 2)}, file_path={request.file_path}")
        raise HTTPException(status_code=500, detail=f"Failed to trigger migration: {str(e)}")


@router.get("/status/{run_id}", response_model=MigrationStatusResponse)
async def get_migration_status(
    run_id: str,
    include_details: bool = Query(default=False, description="Include detailed progress information"),
    include_logs: bool = Query(default=False, description="Include execution logs")
):
    """
    Get the status of a running or completed migration.
    
    Returns detailed information about the migration progress, including:
    - Current status (pending, running, completed, failed, cancelled)
    - Progress information (if include_details=True)
    - Execution logs (if include_logs=True)
    """
    start_time = time.time()
    logger.info(f"Starting get_migration_status API call - run_id={run_id}, include_details={include_details}")
    
    try:
        status_response = await migration_service.get_migration_status(run_id, include_details)
        
        if not status_response:
            execution_time = time.time() - start_time
            logger.warning(f"get_migration_status API call - migration not found - execution_time_ms={round(execution_time * 1000, 2)}, run_id={run_id}")
            raise HTTPException(status_code=404, detail=f"Migration run not found: {run_id}")
        
        execution_time = time.time() - start_time
        logger.info(f"get_migration_status API call completed successfully - execution_time_ms={round(execution_time * 1000, 2)}, run_id={run_id}, status={status_response.status}")
        
        return status_response
        
    except HTTPException:
        raise
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"get_migration_status API call failed - error={str(e)}, error_type={type(e).__name__}, execution_time_ms={round(execution_time * 1000, 2)}, run_id={run_id}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch migration status: {str(e)}")

@router.post("/cancel/{run_id}")
async def cancel_migration(run_id: str):
    """
    Cancel a running migration.
    
    Attempts to cancel a migration that is currently pending or running.
    Returns success status and updated migration information.
    """
    start_time = time.time()
    logger.info(f"Starting cancel_migration API call - run_id={run_id}")
    
    try:
        cancelled = await migration_service.cancel_migration(run_id)
        
        if not cancelled:
            execution_time = time.time() - start_time
            logger.warning(f"cancel_migration API call - migration not found or not cancellable - execution_time_ms={round(execution_time * 1000, 2)}, run_id={run_id}")
            raise HTTPException(status_code=404, detail=f"Migration run not found or not cancellable: {run_id}")
        
        execution_time = time.time() - start_time
        logger.info(f"cancel_migration API call completed successfully - execution_time_ms={round(execution_time * 1000, 2)}, run_id={run_id}")
        
        return {
            "message": f"Migration {run_id} cancelled successfully",
            "run_id": run_id,
            "cancelled": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"cancel_migration API call failed - error={str(e)}, error_type={type(e).__name__}, execution_time_ms={round(execution_time * 1000, 2)}, run_id={run_id}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel migration: {str(e)}")