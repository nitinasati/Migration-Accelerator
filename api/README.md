# Migration-Accelerators API

RESTful API for managing and monitoring data migration workflows.

## Features

- **Migration Management**: CRUD operations for migration runs
- **Workflow States**: Retrieve workflow execution states
- **Search & Filtering**: Advanced search capabilities
- **Statistics**: Real-time migration statistics
- **Async Database**: PostgreSQL with asyncpg
- **OpenAPI Documentation**: Auto-generated API docs

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set environment variables in `.env` file:
```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=migration
DB_USER=postgres
DB_PASSWORD=your_password
DB_SCHEMA=public
```

3. Run the API:
```bash
python main.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Base URLs
- **Health Check**: `GET /health`
- **API Info**: `GET /`
- **Documentation**: `GET /docs` (Swagger UI)

### Migration Endpoints
- **List Migrations**: `GET /api/v1/migrations`
- **Get Migration**: `GET /api/v1/migrations/{run_id}`
- **Get Workflow States**: `GET /api/v1/migrations/{run_id}/states`
- **Get Checkpoints**: `GET /api/v1/migrations/{run_id}/checkpoints`
- **Search Migrations**: `GET /api/v1/migrations/search?q={query}`
- **Get Statistics**: `GET /api/v1/migrations/stats`

### Query Parameters

#### List Migrations
- `record_type`: Filter by record type
- `file_path`: Filter by file path (partial match)
- `status`: Filter by status
- `limit`: Maximum results (default: 50)
- `offset`: Pagination offset (default: 0)

#### Search
- `q`: Search term (searches across multiple fields)
- `limit`: Maximum results (default: 50)
- `offset`: Pagination offset (default: 0)

## Database Schema

The API connects to PostgreSQL tables:
- `migration_runs`: Main migration records
- `workflow_states`: Individual workflow execution states
- `workflow_checkpoints`: Workflow checkpoints

## Response Format

All API responses follow a consistent JSON format:

```json
{
  "id": "uuid",
  "file_path": "string",
  "record_type": "string",
  "status": "string",
  "created_at": "datetime",
  "updated_at": "datetime",
  "completed_at": "datetime|null",
  "total_duration": "float|null",
  "success": "boolean|null",
  "error_message": "string|null"
}
```

## Error Handling

The API returns appropriate HTTP status codes:
- `200`: Success
- `400`: Bad Request
- `404`: Not Found
- `500`: Internal Server Error

Error responses include a `detail` field with error information.

## Development

### Running in Development Mode
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### API Documentation
- **Swagger UI**: `/docs`
- **ReDoc**: `/redoc`
- **OpenAPI Schema**: `/openapi.json`

## Technologies Used

- **Framework**: FastAPI
- **Database**: PostgreSQL with asyncpg
- **Validation**: Pydantic
- **Documentation**: OpenAPI/Swagger
- **Async Support**: Python asyncio
