# Async Migration API

This document describes the asynchronous migration API that allows you to trigger migrations via REST API and monitor their progress using run IDs.

## Overview

The async migration API provides the following capabilities:

- ✅ **Trigger migrations asynchronously** - Get immediate run ID
- ✅ **Monitor progress in real-time** - Poll for status and progress
- ✅ **Background execution** - No blocking while migration runs
- ✅ **Status tracking** - Detailed progress and execution information
- ✅ **Error handling** - Comprehensive error reporting and recovery

## Endpoints

### 1. Trigger Migration

**POST** `/api/v1/migrations/trigger`

Starts a new migration asynchronously and returns a run ID immediately.

**Request Body:**
```json
{
  "file_path": "data/input/sample_disability_data.csv",
  "record_type": "disability",
  "output_mode": "api",
  "output_dir": "data/output",
  "batch_size": 10,
  "authentication": {
    "type": "bearer_token",
    "token": "your-api-token"
  },
  "metadata": {
    "description": "Disability records migration",
    "source_system": "legacy_mainframe"
  }
}
```

**Response:**
```json
{
  "run_id": "abc123-def456-ghi789",
  "status": "pending",
  "message": "Migration queued successfully",
  "created_at": "2025-01-27T10:30:00Z",
  "estimated_duration": "5-10 minutes"
}
```

### 2. Check Migration Status

**GET** `/api/v1/migrations/status/{run_id}?include_details=true`

Gets the current status and progress of a migration.

**Response:**
```json
{
  "run_id": "abc123-def456-ghi789",
  "status": "running",
  "progress": {
    "total_records": 1000,
    "processed_records": 250,
    "success_count": 240,
    "error_count": 10,
    "progress_percentage": 25.0,
    "current_step": "data_transformation"
  },
  "created_at": "2025-01-27T10:30:00Z",
  "started_at": "2025-01-27T10:30:05Z",
  "completed_at": null,
  "duration": null,
  "error_message": null,
  "file_path": "data/input/sample_disability_data.csv",
  "record_type": "disability",
  "metadata": {
    "description": "Disability records migration"
  }
}
```

### 3. List Running Migrations

**GET** `/api/v1/migrations/running`

Lists all currently running migrations.

**Response:**
```json
{
  "running_migrations": {
    "abc123-def456-ghi789": {
      "status": "running",
      "created_at": "2025-01-27T10:30:00Z",
      "file_path": "data/input/sample_disability_data.csv",
      "record_type": "disability"
    }
  },
  "count": 1
}
```

### 4. Cancel Migration

**POST** `/api/v1/migrations/cancel/{run_id}`

Cancels a running or pending migration.

**Response:**
```json
{
  "message": "Migration abc123-def456-ghi789 cancelled successfully",
  "run_id": "abc123-def456-ghi789",
  "cancelled": true
}
```

## Migration Statuses

| Status | Description |
|--------|-------------|
| `pending` | Migration is queued and waiting to start |
| `running` | Migration is currently executing |
| `completed` | Migration finished successfully |
| `failed` | Migration failed with errors |
| `cancelled` | Migration was cancelled by user |

## Migration Steps

The migration goes through several steps:

1. **initializing** - Setting up the migration
2. **workflow_initialized** - Workflow components ready
3. **file_reading** - Reading and parsing input file
4. **data_mapping** - Mapping data to target format
5. **data_transformation** - Transforming data
6. **api_integration** - Making API calls or writing files
7. **completed** - Migration finished

## Record Types

Supported record types:

- `disability` - Disability insurance records
- `absence` - Absence management records
- `group_policy` - Group insurance policies
- `employee` - Employee records
- `claim` - Insurance claims

## Usage Examples

### Python Example

```python
import asyncio
import aiohttp

async def trigger_and_monitor_migration():
    base_url = "http://127.0.0.1:8000/api/v1/migrations"
    
    # 1. Trigger migration
    async with aiohttp.ClientSession() as session:
        migration_request = {
            "file_path": "data/input/sample_disability_data.csv",
            "record_type": "disability",
            "output_mode": "api",
            "batch_size": 10
        }
        
        async with session.post(f"{base_url}/trigger", json=migration_request) as response:
            result = await response.json()
            run_id = result["run_id"]
            print(f"Migration started: {run_id}")
        
        # 2. Monitor progress
        while True:
            async with session.get(f"{base_url}/status/{run_id}?include_details=true") as response:
                status = await response.json()
                print(f"Status: {status['status']}")
                
                if status["progress"]:
                    progress = status["progress"]
                    print(f"Progress: {progress['progress_percentage']:.1f}%")
                
                if status["status"] in ["completed", "failed", "cancelled"]:
                    break
                
                await asyncio.sleep(2)

# Run the example
asyncio.run(trigger_and_monitor_migration())
```

### cURL Examples

```bash
# 1. Trigger migration
curl -X POST "http://127.0.0.1:8000/api/v1/migrations/trigger" \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "data/input/sample_disability_data.csv",
    "record_type": "disability",
    "output_mode": "api",
    "batch_size": 10
  }'

# 2. Check status (replace {run_id} with actual ID)
curl "http://127.0.0.1:8000/api/v1/migrations/status/{run_id}?include_details=true"

# 3. List running migrations
curl "http://127.0.0.1:8000/api/v1/migrations/running"

# 4. Cancel migration (replace {run_id} with actual ID)
curl -X POST "http://127.0.0.1:8000/api/v1/migrations/cancel/{run_id}"
```

## Testing

Run the test script to verify the API:

```bash
# Test full migration workflow
cd api
python test_async_migration.py

# Test just endpoint connectivity
python test_async_migration.py endpoints
```

## Error Handling

The API uses standard HTTP status codes:

- `200` - Success
- `404` - Migration run not found
- `422` - Validation error (invalid request data)
- `500` - Internal server error

Error responses include detailed information:

```json
{
  "detail": "Migration run not found: invalid-run-id"
}
```

## Integration with Existing Dashboard

The dashboard will automatically show migrations triggered via API in the recent migrations table. The auto-refresh functionality will pick up API-triggered migrations and display their progress in real-time.

## Production Considerations

For production deployment:

1. **Persistence**: Replace in-memory storage with Redis or database
2. **Scalability**: Use task queues (Celery, RQ) for better scaling
3. **Security**: Add authentication and authorization
4. **Monitoring**: Add detailed metrics and alerting
5. **Rate Limiting**: Implement rate limiting for API endpoints
