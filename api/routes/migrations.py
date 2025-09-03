"""
Migration-related API routes.
"""

import time
import logging
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from models.database import MigrationRun, WorkflowState, WorkflowCheckpoint, SearchFilters
from database import (
    get_migration_runs, get_migration_run, get_workflow_states,
    get_workflow_checkpoints, search_migration_runs, get_migration_stats
)

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