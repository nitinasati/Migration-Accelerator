#!/usr/bin/env python3
"""
Database schema creation script for LangGraph state persistence.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.database import get_database_config
import asyncpg


async def create_tables():
    """Create all necessary database tables."""
    
    config = get_database_config()
    
    # SQL statements for table creation
    create_migration_runs_table = """
    CREATE TABLE IF NOT EXISTS migration_runs (
        id UUID PRIMARY KEY,
        file_path VARCHAR(500) NOT NULL,
        record_type VARCHAR(100) NOT NULL,
        status VARCHAR(50) NOT NULL DEFAULT 'running',
        created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
        completed_at TIMESTAMP WITH TIME ZONE,
        total_duration DOUBLE PRECISION,
        success BOOLEAN,
        error_message TEXT,
        
        -- Indexes for better query performance
        CONSTRAINT idx_migration_runs_status CHECK (status IN ('running', 'completed', 'failed', 'paused')),
        CONSTRAINT idx_migration_runs_created_at CHECK (created_at IS NOT NULL)
    );
    
    -- Create indexes
    CREATE INDEX IF NOT EXISTS idx_migration_runs_status ON migration_runs(status);
    CREATE INDEX IF NOT EXISTS idx_migration_runs_created_at ON migration_runs(created_at);
    CREATE INDEX IF NOT EXISTS idx_migration_runs_file_path ON migration_runs(file_path);
    CREATE INDEX IF NOT EXISTS idx_migration_runs_record_type ON migration_runs(record_type);
    """
    
    create_workflow_states_table = """
    CREATE TABLE IF NOT EXISTS workflow_states (
        id UUID PRIMARY KEY,
        run_id UUID NOT NULL REFERENCES migration_runs(id) ON DELETE CASCADE,
        step_name VARCHAR(100) NOT NULL,
        state_data JSONB NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
        step_order INTEGER NOT NULL,
        metadata JSONB,
        
        -- Constraints
        CONSTRAINT idx_workflow_states_step_order CHECK (step_order >= 0),
        CONSTRAINT idx_workflow_states_step_name CHECK (step_name IS NOT NULL AND step_name != '')
    );
    
    -- Create indexes
    CREATE INDEX IF NOT EXISTS idx_workflow_states_run_id ON workflow_states(run_id);
    CREATE INDEX IF NOT EXISTS idx_workflow_states_step_name ON workflow_states(step_name);
    CREATE INDEX IF NOT EXISTS idx_workflow_states_step_order ON workflow_states(step_order);
    CREATE INDEX IF NOT EXISTS idx_workflow_states_created_at ON workflow_states(created_at);
    CREATE INDEX IF NOT EXISTS idx_workflow_states_state_data ON workflow_states USING GIN (state_data);
    """
    
    create_workflow_checkpoints_table = """
    CREATE TABLE IF NOT EXISTS workflow_checkpoints (
        id UUID PRIMARY KEY,
        run_id UUID NOT NULL REFERENCES migration_runs(id) ON DELETE CASCADE,
        checkpoint_name VARCHAR(100) NOT NULL,
        state_data JSONB NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
        metadata JSONB,
        
        -- Constraints
        CONSTRAINT idx_workflow_checkpoints_name CHECK (checkpoint_name IS NOT NULL AND checkpoint_name != '')
    );
    
    -- Create indexes
    CREATE INDEX IF NOT EXISTS idx_workflow_checkpoints_run_id ON workflow_checkpoints(run_id);
    CREATE INDEX IF NOT EXISTS idx_workflow_checkpoints_name ON workflow_checkpoints(checkpoint_name);
    CREATE INDEX IF NOT EXISTS idx_workflow_checkpoints_created_at ON workflow_checkpoints(created_at);
    CREATE INDEX IF NOT EXISTS idx_workflow_checkpoints_state_data ON workflow_checkpoints USING GIN (state_data);
    """
    
    create_workflow_metadata_table = """
    CREATE TABLE IF NOT EXISTS workflow_metadata (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        run_id UUID NOT NULL REFERENCES migration_runs(id) ON DELETE CASCADE,
        key VARCHAR(200) NOT NULL,
        value JSONB NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
        
        -- Ensure unique key per run
        UNIQUE(run_id, key)
    );
    
    -- Create indexes
    CREATE INDEX IF NOT EXISTS idx_workflow_metadata_run_id ON workflow_metadata(run_id);
    CREATE INDEX IF NOT EXISTS idx_workflow_metadata_key ON workflow_metadata(key);
    CREATE INDEX IF NOT EXISTS idx_workflow_metadata_value ON workflow_metadata USING GIN (value);
    """
    
    # Create a function to update the updated_at timestamp
    create_updated_at_function = """
    CREATE OR REPLACE FUNCTION update_updated_at_column()
    RETURNS TRIGGER AS $$
    BEGIN
        NEW.updated_at = NOW();
        RETURN NEW;
    END;
    $$ language 'plpgsql';
    """
    
    # Create triggers for updated_at
    create_triggers = """
    -- Trigger for migration_runs table
    DROP TRIGGER IF EXISTS update_migration_runs_updated_at ON migration_runs;
    CREATE TRIGGER update_migration_runs_updated_at
        BEFORE UPDATE ON migration_runs
        FOR EACH ROW
        EXECUTE FUNCTION update_updated_at_column();
    
    -- Trigger for workflow_metadata table
    DROP TRIGGER IF EXISTS update_workflow_metadata_updated_at ON workflow_metadata;
    CREATE TRIGGER update_workflow_metadata_updated_at
        BEFORE UPDATE ON workflow_metadata
        FOR EACH ROW
        EXECUTE FUNCTION update_updated_at_column();
    """
    
    try:
        print(f"Connecting to PostgreSQL at {config.host}:{config.port}/{config.database}...")
        
        # Connect to PostgreSQL
        conn = await asyncpg.connect(
            host=config.host,
            port=config.port,
            user=config.username,
            password=config.password,
            database=config.database
        )
        
        print("Connected to PostgreSQL successfully!")
        
        # Create tables
        print("\nCreating database tables...")
        
        await conn.execute(create_migration_runs_table)
        print("Created migration_runs table")
        
        await conn.execute(create_workflow_states_table)
        print("Created workflow_states table")
        
        await conn.execute(create_workflow_checkpoints_table)
        print("Created workflow_checkpoints table")
        
        await conn.execute(create_workflow_metadata_table)
        print("Created workflow_metadata table")
        
        # Create function and triggers
        await conn.execute(create_updated_at_function)
        print("Created update_updated_at function")
        
        await conn.execute(create_triggers)
        print("Created triggers for updated_at columns")
        
        # Verify tables were created
        tables = await conn.fetch("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN ('migration_runs', 'workflow_states', 'workflow_checkpoints', 'workflow_metadata')
            ORDER BY table_name
        """)
        
        print(f"\nSuccessfully created {len(tables)} tables:")
        for table in tables:
            print(f"  - {table['table_name']}")
        
        await conn.close()
        print("\nDatabase setup completed successfully!")
        
    except Exception as e:
        print(f"Error creating database tables: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(create_tables())
