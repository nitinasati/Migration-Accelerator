"""
Migration-related API routes.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from models.database import MigrationRun, WorkflowState, WorkflowCheckpoint, SearchFilters
from database import (
    get_migration_runs, get_migration_run, get_workflow_states,
    get_workflow_checkpoints, search_migration_runs, get_migration_stats
)

router = APIRouter()

@router.get("/", response_model=List[MigrationRun])
def list_migrations(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """List all migration runs."""
    try:
        migrations = get_migration_runs(limit=limit, offset=offset)
        return migrations
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch migrations: {str(e)}")

@router.get("/stats")
def get_stats():
    """Get migration statistics."""
    try:
        stats = get_migration_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch stats: {str(e)}")

@router.get("/search/", response_model=List[MigrationRun])
def search_migrations(
    record_type: Optional[str] = Query(None, description="Filter by record type"),
    file_path: Optional[str] = Query(None, description="Filter by file path"),
    status: Optional[str] = Query(None, description="Filter by status")
):
    """Search migration runs based on filters."""
    try:
        filters = SearchFilters(
            record_type=record_type,
            file_path=file_path,
            status=status
        )
        results = search_migration_runs(filters)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search migrations: {str(e)}")

@router.get("/{run_id}", response_model=MigrationRun)
def get_migration(run_id: str):
    """Get a specific migration run by ID."""
    try:
        migration = get_migration_run(run_id)
        if not migration:
            raise HTTPException(status_code=404, detail="Migration run not found")
        return migration
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch migration: {str(e)}")

@router.get("/{run_id}/states", response_model=List[WorkflowState])
def get_workflow_states_for_run(run_id: str):
    """Get workflow states for a specific migration run."""
    try:
        states = get_workflow_states(run_id)
        return states
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch workflow states: {str(e)}")

@router.get("/{run_id}/checkpoints", response_model=List[WorkflowCheckpoint])
def get_workflow_checkpoints_for_run(run_id: str):
    """Get workflow checkpoints for a specific migration run."""
    try:
        checkpoints = get_workflow_checkpoints(run_id)
        return checkpoints
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch workflow checkpoints: {str(e)}")