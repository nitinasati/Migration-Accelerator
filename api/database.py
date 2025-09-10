"""
Database connection and operations for the Migration-Accelerators API.

This module provides comprehensive logging for all database operations including:
- Before and after database calls with timing information
- Connection pool management logging
- Query execution details and parameters
- Error logging with context information
- Performance metrics (execution time in milliseconds)

Logging Levels:
- INFO: High-level operation start/completion and connection pool events
- DEBUG: Detailed query execution, parameters, and connection details
- ERROR: Failed operations with error context and timing

Environment Variables:
- DB_DEBUG: Set to "true" to enable DEBUG level logging for database operations
"""

import json
import os
import time
import logging
import threading
from typing import List, Optional, Dict, Any

# Defer heavy imports until needed
# import psycopg2
# from psycopg2.extras import RealDictCursor
# from psycopg2.pool import ThreadedConnectionPool

# Defer dotenv loading until needed
# from dotenv import load_dotenv

from config.database import (
    MigrationRun, WorkflowState, WorkflowCheckpoint, 
    SearchFilters
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set debug level for database operations if needed
if os.getenv("DB_DEBUG", "false").lower() == "true":
    logger.setLevel(logging.DEBUG)

# Database connection pool with thread safety
_pool = None
_pool_lock = threading.Lock()
_pool_initialized = False

# Schema prefix loaded once from environment
_schema_prefix: str = ""

# Flag to track if environment has been loaded
_env_loaded = False

def _load_environment():
    """Load environment variables only when needed."""
    global _env_loaded
    if not _env_loaded:
        try:
            from dotenv import load_dotenv
            load_dotenv('../.env')
            _env_loaded = True
            logger.debug("Environment variables loaded")
        except Exception as e:
            logger.warning(f"Failed to load environment variables: {e}")
            _env_loaded = True  # Mark as loaded to prevent retries

def _get_psycopg2_imports():
    """Get psycopg2 imports when needed."""
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
        from psycopg2.pool import ThreadedConnectionPool
        return psycopg2, RealDictCursor, ThreadedConnectionPool
    except ImportError as e:
        logger.error(f"psycopg2 not available: {e}")
        raise

def get_pool():
    """Get database connection pool with thread safety."""
    global _pool, _pool_initialized
    
    if _pool is None or not _pool_initialized:
        with _pool_lock:
            if _pool is None or not _pool_initialized:
                logger.info("Inside get_pool: Creating new database connection pool")
                _create_pool()
                logger.info("Inside get_pool: Database connection pool created successfully")
    
    return _pool

def _create_pool() -> None:
    """Create a new database connection pool."""
    global _pool, _pool_initialized
    
    try:
        logger.info("Creating new database connection pool")
        
        # Load environment and imports only when needed
        _load_environment()
        _, RealDictCursor, ThreadedConnectionPool = _get_psycopg2_imports()
        
        # Close existing pool if any
        if _pool:
            try:
                _pool.closeall()
            except Exception as e:
                logger.warning(f"Error closing existing pool: {e}")
        
        # Get database configuration from settings
        import sys
        import os
        
        # Add project root to Python path if not already there
        current_dir = os.path.dirname(os.path.abspath(__file__))  # api directory
        project_root = os.path.dirname(current_dir)  # project root
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        from config.settings import settings
        
        # Create new pool with optimized settings
        _pool = ThreadedConnectionPool(
            minconn=5,       # Reduced from 10 to 5
            maxconn=15,      # Reduced from 20 to 15
            host=os.getenv("DB_HOST", settings.db_host),
            port=int(os.getenv("DB_PORT", str(settings.db_port))),
            database=os.getenv("DB_NAME", settings.db_name),
            user=os.getenv("DB_USER", settings.db_user),
            password=os.getenv("DB_PASSWORD", settings.db_password or ""),
            # Connection settings
            connect_timeout=3,      # Reduced from 5 to 3 seconds
            # PostgreSQL keepalive settings
            keepalives=3,           # Reduced from 5 to 3
            keepalives_idle=20,     # Reduced from 30 to 20 seconds
            keepalives_interval=5,  # Reduced from 10 to 5 seconds
            keepalives_count=2,     # Reduced from 3 to 2
            # Performance settings
            options="-c statement_timeout=5000 -c idle_in_transaction_session_timeout=15000"
        )
        
        _pool_initialized = True
        logger.info(f"Database connection pool created successfully - pool_size={_pool.maxconn}")
        
        # Warm up the pool (reduced connections)
        _warm_up_pool()
        
    except Exception as e:
        logger.error(f"Failed to create connection pool: {e}")
        _pool = None
        _pool_initialized = False
        raise

def _warm_up_pool() -> None:
    """Warm up the connection pool by testing connections."""
    try:
        if not _pool:
            return
            
        logger.info("Warming up database connection pool")
        
        # Test fewer connections to reduce startup time
        test_connections = []
        for i in range(min(2, _pool.minconn)):  # Reduced from 3 to 2
            try:
                conn = _pool.getconn()
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.close()
                test_connections.append(conn)
                logger.debug(f"Connection {i+1} tested successfully")
            except Exception as e:
                logger.warning(f"Failed to test connection {i+1}: {e}")
                if conn:
                    _pool.putconn(conn)
        
        # Return test connections to pool
        for conn in test_connections:
            _pool.putconn(conn)
        
        logger.info("Database connection pool warmed up successfully")
        
    except Exception as e:
        logger.warning(f"Failed to warm up connection pool: {e}")

def close_pool() -> None:
    """Close database connection pool."""
    global _pool, _pool_initialized
    
    with _pool_lock:
        if _pool:
            try:
                logger.info("Closing database connection pool")
                _pool.closeall()
                _pool = None
                _pool_initialized = False
                logger.info("Database connection pool closed successfully")
            except Exception as e:
                logger.error(f"Error closing connection pool: {e}")

def get_pool_status() -> Dict[str, Any]:
    """Get connection pool status."""
    logger.info("Inside get_pool_status: Getting connection pool status")
    if _pool is None or not _pool_initialized:
        logger.info("Inside get_pool_status: Pool is not initialized")
        return {"status": "not_initialized"}
    
    try:
        return {
            "status": "active",
            "min_connections": _pool.minconn,
            "max_connections": _pool.maxconn,
            "current_connections": len(_pool._used) + len(_pool._pool),
            "used_connections": len(_pool._used),
            "available_connections": len(_pool._pool)
        }
    except Exception as e:
        logger.warning(f"Error getting pool status: {e}")
        return {"status": "error", "error": str(e)}

def check_connection_health() -> bool:
    """Check if connections in the pool are still healthy."""
    try:
        if not _pool or not _pool_initialized:
            return False
        
        # Test a connection from the pool
        conn = _pool.getconn()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            cursor.close()
            return result[0] == 1
        finally:
            _pool.putconn(conn)
            
    except Exception as e:
        logger.warning(f"Connection health check failed: {e}")
        return False

def refresh_expired_connections() -> None:
    """Refresh expired connections in the pool."""
    global _pool, _pool_initialized
    
    with _pool_lock:
        try:
            logger.info("Refreshing expired database connections")
            
            # Close existing pool
            if _pool:
                _pool.closeall()
                _pool = None
                _pool_initialized = False
            
            # Create new pool
            _create_pool()
            
            logger.info("Database connection pool refreshed successfully")
            
        except Exception as e:
            logger.error(f"Failed to refresh connection pool: {e}")

def _load_schema_prefix() -> None:
    """Load schema prefix from environment variables."""
    global _schema_prefix
    if not _env_loaded:
        _load_environment()
    schema = os.getenv("DB_SCHEMA", "public")
    _schema_prefix = f"{schema}." if schema != "public" else ""
    logger.debug(f"Schema prefix loaded: '{_schema_prefix}'")

def _get_schema_prefix() -> str:
    """Get schema prefix for table names."""
    if not _schema_prefix:
        _load_schema_prefix()
    return _schema_prefix

def _execute_query(query: str, params: tuple = None, operation: str = "unknown") -> List[Dict[str, Any]]:
    """Execute a database query with proper connection management."""
    start_time = time.time()
    conn = None

    logger.info("Inside _execute_query: Executing query")
    try:
        pool = get_pool()
        conn = pool.getconn()
        
        # Get RealDictCursor when needed
        _, RealDictCursor, _ = _get_psycopg2_imports()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        logger.debug(f"Database connection acquired - connection_id={id(conn)}")
        
        # Set search path if schema is specified
        schema = os.getenv("DB_SCHEMA")
        if schema and schema != "public":
            logger.debug(f"Setting search path - schema={schema}")
            cursor.execute(f"SET search_path TO {schema}, public")
        
        # Execute query
        logger.debug(f"Executing query - operation={operation}, query={query}, params={params}")
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        logger.debug(f"Query executed successfully - operation={operation}, rows_returned={len(rows)}")
        
        # Convert to list of dictionaries
        result = [dict(row) for row in rows]
        
        # Convert datetime objects to strings for JSON serialization
        for row_dict in result:
            for key, value in row_dict.items():
                if hasattr(value, 'isoformat'):  # Check if it's a datetime object
                    row_dict[key] = value.isoformat()
        
        cursor.close()
        
        execution_time = time.time() - start_time
        logger.info(f"Database query completed successfully: {operation} - execution_time_ms={round(execution_time * 1000, 2)}, rows_returned={len(result)}")
        
        return result
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Database query failed: {operation} - error={str(e)}, error_type={type(e).__name__}, execution_time_ms={round(execution_time * 1000, 2)}")
        raise
        
    finally:
        if conn:
            try:
                pool.putconn(conn)
            except Exception as e:
                logger.warning(f"Error returning connection to pool: {e}")

def get_migration_runs(limit: int = 100, offset: int = 0) -> List[MigrationRun]:
    """Get list of migration runs."""
    try:
        query = f"""
            SELECT * FROM {_get_schema_prefix()}migration_runs 
            ORDER BY created_at DESC 
            LIMIT %s OFFSET %s
        """
        
        rows = _execute_query(query, (limit, offset), "get_migration_runs")
        
        migration_runs = []
        for row in rows:
            migration_runs.append(MigrationRun(**row))
        
        return migration_runs
        
    except Exception as e:
        logger.error(f"Error in get_migration_runs: {e}")
        return []

def get_migration_run(run_id: str) -> Optional[MigrationRun]:
    """Get a specific migration run by ID."""
    try:
        query = f"SELECT * FROM {_get_schema_prefix()}migration_runs WHERE id = %s"
        rows = _execute_query(query, (run_id,), "get_migration_run")
        
        if rows:
            return MigrationRun(**rows[0])
        return None
        
    except Exception as e:
        logger.error(f"Error in get_migration_run: {e}")
        return None

def get_workflow_states(run_id: str) -> List[WorkflowState]:
    """Get workflow states for a specific migration run."""
    try:
        query = f"""
            SELECT * FROM {_get_schema_prefix()}workflow_states 
            WHERE run_id = %s 
            ORDER BY step_order ASC
        """
        
        rows = _execute_query(query, (run_id,), "get_workflow_states")
        
        workflow_states = []
        for row in rows:
            workflow_states.append(WorkflowState(**row))
        
        return workflow_states
        
    except Exception as e:
        logger.error(f"Error in get_workflow_states: {e}")
        return []

def get_workflow_checkpoints(run_id: str) -> List[WorkflowCheckpoint]:
    """Get workflow checkpoints for a specific migration run."""
    try:
        query = f"""
            SELECT * FROM {_get_schema_prefix()}workflow_checkpoints 
            WHERE run_id = %s 
            ORDER BY created_at ASC
        """
        
        rows = _execute_query(query, (run_id,), "get_workflow_checkpoints")
        
        workflow_checkpoints = []
        for row in rows:
            workflow_checkpoints.append(WorkflowCheckpoint(**row))
        
        return workflow_checkpoints
        
    except Exception as e:
        logger.error(f"Error in get_workflow_checkpoints: {e}")
        return []

def get_migration_stats() -> Dict[str, Any]:
    """Get migration statistics."""
    logger.info("Inside get_migration_stats: Getting migration statistics")
    try:
        # Single optimized query for all statistics
        stats_query = f"""
            SELECT 
                status,
                COUNT(*) as status_count,
                COUNT(CASE WHEN created_at >= NOW() - INTERVAL '24 hours' THEN 1 END) as recent_count
            FROM {_get_schema_prefix()}migration_runs 
            GROUP BY status
        """
        
        rows = _execute_query(stats_query, None, "get_migration_stats")
        
        # Process results
        total_migrations = 0
        status_breakdown = {}
        recent_24h = 0
        
        for row in rows:
            status_count = row['status_count']
            total_migrations += status_count
            status_breakdown[row['status']] = status_count
            recent_24h += row['recent_count']
        
        logger.info(f"Stats processed - total_migrations={total_migrations}, status_breakdown={status_breakdown}, recent_24h={recent_24h}")
        
        return {
            "total_migrations": total_migrations,
            "status_breakdown": status_breakdown,
            "recent_24h": recent_24h
        }
        
    except Exception as e:
        logger.error(f"Error in get_migration_stats: {e}")
        return {}

def search_migration_runs(filters: SearchFilters) -> List[MigrationRun]:
    """Search migration runs based on filters."""
    try:
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
        
        rows = _execute_query(query, tuple(params), "search_migration_runs")
        
        migration_runs = []
        for row in rows:
            migration_runs.append(MigrationRun(**row))
        
        return migration_runs
        
    except Exception as e:
        logger.error(f"Error in search_migration_runs: {e}")
        return []

# Schema prefix will be loaded when first needed
