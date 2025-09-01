"""
Database connection and operations for the Migration-Accelerators API.
"""

import json
import os
from typing import List, Optional, Dict, Any
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool
from dotenv import load_dotenv

from models.database import (
    MigrationRun, WorkflowState, WorkflowCheckpoint, 
    MigrationRunSummary, SearchFilters
)

# Load environment variables
load_dotenv()

# Database connection pool
_pool: Optional[SimpleConnectionPool] = None

def get_pool() -> SimpleConnectionPool:
    """Get database connection pool."""
    global _pool
    if _pool is None:
        _pool = SimpleConnectionPool(
            minconn=1,
            maxconn=10,
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            database=os.getenv("DB_NAME", "migration"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "")
        )
    return _pool

def close_pool() -> None:
    """Close database connection pool."""
    global _pool
    if _pool:
        _pool.closeall()
        _pool = None

def _get_schema_prefix() -> str:
    """Get schema prefix for table names."""
    schema = os.getenv("DB_SCHEMA", "public")
    return f"{schema}." if schema != "public" else ""

def get_migration_runs(limit: int = 100, offset: int = 0) -> List[MigrationRun]:
    """Get list of migration runs."""
    try:
        pool = get_pool()
        conn = pool.getconn()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Set search path if schema is specified
        schema = os.getenv("DB_SCHEMA")
        if schema and schema != "public":
            cursor.execute(f"SET search_path TO {schema}, public")
        
        query = f"""
            SELECT * FROM {_get_schema_prefix()}migration_runs 
            ORDER BY created_at DESC 
            LIMIT %s OFFSET %s
        """
        cursor.execute(query, (limit, offset))
        rows = cursor.fetchall()
        
        migration_runs = []
        for row in rows:
            # Convert datetime objects to strings for JSON serialization
            row_dict = dict(row)
            if 'created_at' in row_dict and row_dict['created_at']:
                row_dict['created_at'] = row_dict['created_at'].isoformat()
            if 'updated_at' in row_dict and row_dict['updated_at']:
                row_dict['updated_at'] = row_dict['updated_at'].isoformat()
            if 'completed_at' in row_dict and row_dict['completed_at']:
                row_dict['completed_at'] = row_dict['completed_at'].isoformat()
            
            migration_runs.append(MigrationRun(**row_dict))
        
        cursor.close()
        pool.putconn(conn)
        return migration_runs
        
    except Exception as e:
        print(f"Error getting migration runs: {e}")
        return []

def get_migration_run(run_id: str) -> Optional[MigrationRun]:
    """Get a specific migration run by ID."""
    try:
        pool = get_pool()
        conn = pool.getconn()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Set search path if schema is specified
        schema = os.getenv("DB_SCHEMA")
        if schema and schema != "public":
            cursor.execute(f"SET search_path TO {schema}, public")
        
        query = f"SELECT * FROM {_get_schema_prefix()}migration_runs WHERE id = %s"
        cursor.execute(query, (run_id,))
        row = cursor.fetchone()
        
        cursor.close()
        pool.putconn(conn)
        
        if row:
            # Convert datetime objects to strings for JSON serialization
            row_dict = dict(row)
            if 'created_at' in row_dict and row_dict['created_at']:
                row_dict['created_at'] = row_dict['created_at'].isoformat()
            if 'updated_at' in row_dict and row_dict['updated_at']:
                row_dict['updated_at'] = row_dict['updated_at'].isoformat()
            if 'completed_at' in row_dict and row_dict['completed_at']:
                row_dict['completed_at'] = row_dict['completed_at'].isoformat()
            
            return MigrationRun(**row_dict)
        return None
        
    except Exception as e:
        print(f"Error getting migration run: {e}")
        return None

def get_workflow_states(run_id: str) -> List[WorkflowState]:
    """Get workflow states for a specific migration run."""
    try:
        pool = get_pool()
        conn = pool.getconn()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Set search path if schema is specified
        schema = os.getenv("DB_SCHEMA")
        if schema and schema != "public":
            cursor.execute(f"SET search_path TO {schema}, public")
        
        query = f"""
            SELECT * FROM {_get_schema_prefix()}workflow_states 
            WHERE run_id = %s 
            ORDER BY step_order ASC
        """
        cursor.execute(query, (run_id,))
        rows = cursor.fetchall()
        
        workflow_states = []
        for row in rows:
            # Convert datetime objects to strings for JSON serialization
            row_dict = dict(row)
            if 'created_at' in row_dict and row_dict['created_at']:
                row_dict['created_at'] = row_dict['created_at'].isoformat()
            
            workflow_states.append(WorkflowState(**row_dict))
        
        cursor.close()
        pool.putconn(conn)
        return workflow_states
        
    except Exception as e:
        print(f"Error getting workflow states: {e}")
        return []

def get_workflow_checkpoints(run_id: str) -> List[WorkflowCheckpoint]:
    """Get workflow checkpoints for a specific migration run."""
    try:
        pool = get_pool()
        conn = pool.getconn()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Set search path if schema is specified
        schema = os.getenv("DB_SCHEMA")
        if schema and schema != "public":
            cursor.execute(f"SET search_path TO {schema}, public")
        
        query = f"""
            SELECT * FROM {_get_schema_prefix()}workflow_checkpoints 
            WHERE run_id = %s 
            ORDER BY created_at ASC
        """
        cursor.execute(query, (run_id,))
        rows = cursor.fetchall()
        
        workflow_checkpoints = []
        for row in rows:
            # Convert datetime objects to strings for JSON serialization
            row_dict = dict(row)
            if 'created_at' in row_dict and row_dict['created_at']:
                row_dict['created_at'] = row_dict['created_at'].isoformat()
            
            workflow_checkpoints.append(WorkflowCheckpoint(**row_dict))
        
        cursor.close()
        pool.putconn(conn)
        return workflow_checkpoints
        
    except Exception as e:
        print(f"Error getting workflow checkpoints: {e}")
        return []

def get_migration_stats() -> Dict[str, Any]:
    """Get migration statistics."""
    try:
        pool = get_pool()
        conn = pool.getconn()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Set search path if schema is specified
        schema = os.getenv("DB_SCHEMA")
        if schema and schema != "public":
            cursor.execute(f"SET search_path TO {schema}, public")
        
        # Total migrations
        cursor.execute(f"SELECT COUNT(*) FROM {_get_schema_prefix()}migration_runs")
        total_migrations = cursor.fetchone()['count']
        
        # Status breakdown
        cursor.execute(f"""
            SELECT status, COUNT(*) as count 
            FROM {_get_schema_prefix()}migration_runs 
            GROUP BY status
        """)
        status_breakdown = {row['status']: row['count'] for row in cursor.fetchall()}
        
        # Recent activity
        cursor.execute(f"""
            SELECT COUNT(*) FROM {_get_schema_prefix()}migration_runs 
            WHERE created_at >= NOW() - INTERVAL '24 hours'
        """)
        recent_24h = cursor.fetchone()['count']
        
        cursor.close()
        pool.putconn(conn)
        
        return {
            "total_migrations": total_migrations,
            "status_breakdown": status_breakdown,
            "recent_24h": recent_24h
        }
        
    except Exception as e:
        print(f"Error getting migration stats: {e}")
        return {}

def search_migration_runs(filters: SearchFilters) -> List[MigrationRun]:
    """Search migration runs based on filters."""
    try:
        pool = get_pool()
        conn = pool.getconn()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Set search path if schema is specified
        schema = os.getenv("DB_SCHEMA")
        if schema and schema != "public":
            cursor.execute(f"SET search_path TO {schema}, public")
        
        # Build dynamic query
        query_parts = [f"SELECT * FROM {_get_schema_prefix()}migration_runs WHERE 1=1"]
        params = []
        
        if filters.record_type:
            query_parts.append("AND record_type ILIKE %s")
            params.append(f"%{filters.record_type}%")
        
        if filters.file_path:
            query_parts.append("AND file_path ILIKE %s")
            params.append(f"%{filters.file_path}%")
        
        if filters.status:
            query_parts.append("AND status = %s")
            params.append(filters.status)
        
        query_parts.append("ORDER BY created_at DESC")
        
        query = " ".join(query_parts)
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        migration_runs = []
        for row in rows:
            # Convert datetime objects to strings for JSON serialization
            row_dict = dict(row)
            if 'created_at' in row_dict and row_dict['created_at']:
                row_dict['created_at'] = row_dict['created_at'].isoformat()
            if 'updated_at' in row_dict and row_dict['updated_at']:
                row_dict['updated_at'] = row_dict['updated_at'].isoformat()
            if 'completed_at' in row_dict and row_dict['completed_at']:
                row_dict['completed_at'] = row_dict['completed_at'].isoformat()
            
            migration_runs.append(MigrationRun(**row_dict))
        
        cursor.close()
        pool.putconn(conn)
        return migration_runs
        
    except Exception as e:
        print(f"Error searching migration runs: {e}")
        return []