"""
FastAPI application for the Migration-Accelerators API.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.migrations import router as migrations_router

# Create FastAPI app
app = FastAPI(
    title="Migration-Accelerators API",
    description="API for managing migration workflows and data",
    version="1.0.0"
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)