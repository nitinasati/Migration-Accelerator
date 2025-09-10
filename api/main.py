"""
FastAPI application for the Migration-Accelerators API.
"""

from contextlib import asynccontextmanager
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes.migrations import router as migrations_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for FastAPI."""
    # Startup - Initialize database pool asynchronously
    startup_task = None
    try:
        # Start database initialization in background
        startup_task = asyncio.create_task(_initialize_database_async())
        print("🚀 Starting database initialization in background...")
    except Exception as e:
        print(f"⚠️ Failed to start database initialization: {e}")
    
    yield
    
    # Cancel startup task if still running
    if startup_task and not startup_task.done():
        startup_task.cancel()
        try:
            await startup_task
        except asyncio.CancelledError:
            pass
    
    # Shutdown
    try:
        from api.database import close_pool
        close_pool()
        print("✅ Database connection pool closed on shutdown")
    except Exception as e:
        print(f"⚠️ Failed to close database pool: {e}")

async def _initialize_database_async():
    """Initialize database connection pool asynchronously."""
    try:
        # Small delay to let FastAPI start up first
        await asyncio.sleep(0.1)
        
        from api.database import get_pool
        # This will create and warm up the pool
        pool = get_pool()
        print("✅ Database connection pool initialized and warmed up on startup")
    except Exception as e:
        print(f"⚠️ Failed to initialize database pool: {e}")

# Create FastAPI app
app = FastAPI(
    title="Migration-Accelerators API",
    description="API for managing migration workflows and data",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include migration routes
app.include_router(migrations_router, prefix="/api/v1/migrations", tags=["migrations"])

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Migration-Accelerators API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "migration-accelerators-api"}

@app.get("/health/db")
async def database_health_check():
    """Database health check endpoint."""
    try:
        from api.database import get_pool_status, check_connection_health
        pool_status = get_pool_status()
        connection_healthy = check_connection_health()
        
        return {
            "status": "healthy" if connection_healthy else "unhealthy",
            "service": "migration-accelerators-api",
            "database": "connected" if connection_healthy else "disconnected",
            "pool_status": pool_status
        }
    except Exception as e:
        return {
            "status": "unhealthy", 
            "service": "migration-accelerators-api",
            "database": "error",
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="127.0.0.1",  # Force IPv4 instead of localhost
        port=8000,
        log_level="info"
    )