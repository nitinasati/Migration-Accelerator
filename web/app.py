"""
Flask web application for the Migration-Accelerators dashboard.
"""

import os
import requests
import time
import logging
from flask import Flask, render_template, request, jsonify, redirect, url_for
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key')

# API base URL
API_BASE_URL = os.getenv('API_BASE_URL', 'http://127.0.0.1:8000')

def format_duration(duration):
    """Format duration in seconds to human readable format."""
    if duration is None:
        return "N/A"
    
    try:
        duration = float(duration)
        if duration < 60:
            return f"{duration:.1f}s"
        elif duration < 3600:
            minutes = duration / 60
            return f"{minutes:.1f}m"
        else:
            hours = duration / 3600
            return f"{hours:.1f}h"
    except (ValueError, TypeError):
        return "N/A"

def format_datetime(datetime_str):
    """Format datetime string to readable format."""
    if not datetime_str:
        return "N/A"
    
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return datetime_str

def format_file_path(file_path):
    """Format file path for display."""
    if not file_path:
        return "N/A"
    
    # Show only filename if path is long
    if len(file_path) > 50:
        return f"...{file_path[-47:]}"
    return file_path

def status_badge(status):
    """Convert status to Bootstrap badge class."""
    status_map = {
        'running': 'warning',
        'completed': 'success',
        'failed': 'danger',
        'paused': 'info'
    }
    return status_map.get(status.lower(), 'secondary')

# Register custom filters
app.jinja_env.filters['status_badge'] = status_badge
app.jinja_env.filters['format_duration'] = format_duration
app.jinja_env.filters['format_datetime'] = format_datetime
app.jinja_env.filters['datetime'] = format_datetime  # Alias for datetime filter
app.jinja_env.filters['format_file_path'] = format_file_path

# Make helper functions available in templates
app.jinja_env.globals.update({
    'format_duration': format_duration,
    'format_datetime': format_datetime,
    'format_file_path': format_file_path,
    'status_badge': status_badge
})

# Add request timing middleware
@app.before_request
def before_request():
    request.start_time = time.time()
    logger.info(f"Request started: {request.method} {request.path}")
    
    # Debug routing for API calls
    if request.path.startswith('/api/'):
        logger.info(f"DEBUG: API request path: {request.path}")
        logger.info(f"DEBUG: Available routes: {[str(rule) for rule in app.url_map.iter_rules()]}")

@app.after_request
def after_request(response):
    if hasattr(request, 'start_time'):
        total_time = time.time() - request.start_time
        logger.info(f"Request completed: {request.method} {request.path} - Total time: {total_time:.3f}s - Status: {response.status_code}")
    return response

@app.route('/')
def index():
    """Dashboard home page."""
    try:
        # Get migration statistics
        logger.info(f"Starting API call to: {API_BASE_URL}/api/v1/migrations/stats")
        start_time = time.time()
        response = requests.get(f"{API_BASE_URL}/api/v1/migrations/stats", timeout=5)
        api_time = time.time() - start_time
        logger.info(f"API call completed in {api_time:.3f}s - Status: {response.status_code}")
        
        if response.status_code == 200:
            stats = response.json()
            # Ensure we have default values if any keys are missing
            stats = {
                'total_migrations': stats.get('total_migrations', 0),
                'status_breakdown': stats.get('status_breakdown', {}),
                'recent_24h': stats.get('recent_24h', 0)
            }
            logger.info(f"Stats API response: {stats}")
        else:
            stats = {
                'total_migrations': 0,
                'status_breakdown': {},
                'recent_24h': 0
            }
            logger.warning(f"Stats API failed with status: {response.status_code}")
        
        # Get recent migrations
        logger.info(f"Starting API call to: {API_BASE_URL}/api/v1/migrations/?limit=5")
        start_time = time.time()
        response = requests.get(f"{API_BASE_URL}/api/v1/migrations/?limit=5", timeout=5)
        api_time = time.time() - start_time
        logger.info(f"Recent migrations API call completed in {api_time:.3f}s - Status: {response.status_code}")
        
        if response.status_code == 200:
            recent_migrations = response.json()
            logger.info(f"Recent migrations count: {len(recent_migrations)}")
        else:
            recent_migrations = []
            logger.warning(f"Recent migrations API failed with status: {response.status_code}")
        
        total_time = time.time() - start_time
        logger.info(f"Index route completed - Total time: {total_time:.3f}s")
        return render_template('index.html', stats=stats, recent_migrations=recent_migrations)
    
    except Exception as e:
        logger.error(f"Error in index route: {str(e)}", exc_info=True)
        return render_template('index.html', 
            stats={
                'total_migrations': 0,
                'status_breakdown': {},
                'recent_24h': 0
            }, 
            recent_migrations=[], 
            error=str(e)
        )

@app.route('/migrations')
def migrations():
    """Migrations list page."""
    try:
        # Get query parameters
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        # Get filter parameters
        record_type = request.args.get('record_type', '')
        file_path = request.args.get('file_path', '')
        status = request.args.get('status', '')
        
        # Create filters object for template
        filters = {
            'record_type': record_type,
            'file_path': file_path,
            'status': status
        }
        
        # Apply filters to API call if any filters are set
        if any([record_type, file_path, status]):
            # Use search endpoint with filters
            search_params = {}
            if record_type:
                search_params['record_type'] = record_type
            if file_path:
                search_params['file_path'] = file_path
            if status:
                search_params['status'] = status
            
            logger.info(f"Starting filtered search API call with params: {search_params}")
            start_time = time.time()
            response = requests.get(f"{API_BASE_URL}/api/v1/migrations/search/", params=search_params, timeout=5)
            api_time = time.time() - start_time
            logger.info(f"Filtered search API call completed in {api_time:.3f}s - Status: {response.status_code}")
        else:
            # Get all migrations without filters
            logger.info(f"Starting migrations API call - limit: {limit}, offset: {offset}")
            start_time = time.time()
            response = requests.get(f"{API_BASE_URL}/api/v1/migrations/?limit={limit}&offset={offset}", timeout=5)
            api_time = time.time() - start_time
            logger.info(f"Migrations API call completed in {api_time:.3f}s - Status: {response.status_code}")
        
        if response.status_code == 200:
            migrations = response.json()
            logger.info(f"Migrations retrieved: {len(migrations)} records")
        else:
            migrations = []
            logger.warning(f"Migrations API failed with status: {response.status_code}")
        
        return render_template(
            'migrations.html',
            migrations=migrations,
            limit=limit,
            offset=offset,
            filters=filters
        )
    
    except Exception as e:
        return render_template(
            'migrations.html', 
            migrations=[], 
            error=str(e),
            filters={},
            limit=50,
            offset=0
        )

@app.route('/migrations/<run_id>')
def migration_detail(run_id):
    """Migration detail page."""
    try:
        route_start_time = time.time()
        logger.info(f"Starting migration detail route for run_id: {run_id}")
        
        # Get migration details
        logger.info(f"Starting API call to get migration details: {run_id}")
        start_time = time.time()
        response = requests.get(f"{API_BASE_URL}/api/v1/migrations/{run_id}", timeout=5)
        api_time = time.time() - start_time
        logger.info(f"Migration details API call completed in {api_time:.3f}s - Status: {response.status_code}")
        
        if response.status_code != 200:
            logger.warning(f"Migration details API failed for run_id: {run_id}")
            return render_template(
                'migration_detail.html', 
                error="Migration not found",
                migration_run=None,
                workflow_states=[],
                workflow_checkpoints=[]
            )
        
        migration_run = response.json()
        logger.info(f"Migration details retrieved successfully for run_id: {run_id}")
        
        # Get workflow states
        logger.info(f"Starting API call to get workflow states for run_id: {run_id}")
        start_time = time.time()
        response = requests.get(f"{API_BASE_URL}/api/v1/migrations/{run_id}/states", timeout=5)
        api_time = time.time() - start_time
        logger.info(f"Workflow states API call completed in {api_time:.3f}s - Status: {response.status_code}")
        
        if response.status_code == 200:
            workflow_states = response.json()
            logger.info(f"Workflow states retrieved: {len(workflow_states)} records")
        else:
            workflow_states = []
            logger.warning(f"Workflow states API failed with status: {response.status_code}")
        
        # Get workflow checkpoints
        logger.info(f"Starting API call to get workflow checkpoints for run_id: {run_id}")
        start_time = time.time()
        response = requests.get(f"{API_BASE_URL}/api/v1/migrations/{run_id}/checkpoints", timeout=5)
        api_time = time.time() - start_time
        logger.info(f"Workflow checkpoints API call completed in {api_time:.3f}s - Status: {response.status_code}")
        
        if response.status_code == 200:
            workflow_checkpoints = response.json()
            logger.info(f"Workflow checkpoints retrieved: {len(workflow_checkpoints)} records")
        else:
            workflow_checkpoints = []
            logger.warning(f"Workflow checkpoints API failed with status: {response.status_code}")
        
        total_route_time = time.time() - route_start_time
        logger.info(f"Migration detail route completed in {total_route_time:.3f}s")
        
        return render_template(
            'migration_detail.html',
            migration_run=migration_run,
            workflow_states=workflow_states,
            workflow_checkpoints=workflow_checkpoints
        )
    
    except Exception as e:
        return render_template(
            'migration_detail.html', 
            error=str(e),
            migration_run=None,
            workflow_states=[],
            workflow_checkpoints=[]
        )

@app.route('/search')
def search():
    """Search page."""
    try:
        # Get search parameters
        record_type = request.args.get('record_type', '')
        file_path = request.args.get('file_path', '')
        status = request.args.get('status', '')
        
        if any([record_type, file_path, status]):
            # Perform search
            params = {}
            if record_type:
                params['record_type'] = record_type
            if file_path:
                params['file_path'] = file_path
            if status:
                params['status'] = status
            
            logger.info(f"Starting search API call with params: {params}")
            start_time = time.time()
            response = requests.get(f"{API_BASE_URL}/api/v1/migrations/search/", params=params, timeout=5)
            api_time = time.time() - start_time
            logger.info(f"Search API call completed in {api_time:.3f}s - Status: {response.status_code}")
            
            if response.status_code == 200:
                results = response.json()
                logger.info(f"Search results retrieved: {len(results)} records")
            else:
                results = []
                logger.warning(f"Search API failed with status: {response.status_code}")
        else:
            results = []
            logger.info("No search parameters provided, returning empty results")
        
        return render_template(
            'search.html',
            results=results,
            record_type=record_type,
            file_path=file_path,
            status=status
        )
    
    except Exception as e:
        return render_template('search.html', results=[], error=str(e))

# API proxy endpoints for AJAX calls
@app.route('/api/stats')
def api_stats():
    """API proxy for statistics."""
    try:
        logger.info("API proxy: Starting stats call")
        start_time = time.time()
        response = requests.get(f"{API_BASE_URL}/api/v1/migrations/stats", timeout=5)
        api_time = time.time() - start_time
        logger.info(f"API proxy: Stats call completed in {api_time:.3f}s - Status: {response.status_code}")
        
        return jsonify(response.json()), response.status_code
    
    except Exception as e:
        logger.error(f"API proxy: Error in stats call: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/migrations')
def api_migrations():
    """API proxy for migrations."""
    try:
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        logger.info(f"API proxy: Starting migrations call - limit: {limit}, offset: {offset}")
        start_time = time.time()
        response = requests.get(f"{API_BASE_URL}/api/v1/migrations/?limit={limit}&offset={offset}", timeout=5)
        api_time = time.time() - start_time
        logger.info(f"API proxy: Migrations call completed in {api_time:.3f}s - Status: {response.status_code}")
        
        return jsonify(response.json()), response.status_code
    
    except Exception as e:
        logger.error(f"API proxy: Error in migrations call: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/migrations/<run_id>/states')
def api_workflow_states(run_id):
    """API proxy for workflow states."""
    try:
        logger.info(f"API proxy: Starting workflow states call for run_id: {run_id}")
        start_time = time.time()
        response = requests.get(f"{API_BASE_URL}/api/v1/migrations/{run_id}/states", timeout=5)
        api_time = time.time() - start_time
        logger.info(f"API proxy: Workflow states call completed in {api_time:.3f}s - Status: {response.status_code}")
        
        return jsonify(response.json()), response.status_code
    
    except Exception as e:
        logger.error(f"API proxy: Error in workflow states call for run_id {run_id}: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/migrations/<run_id>/checkpoints')
def api_workflow_checkpoints(run_id):
    """API proxy for workflow checkpoints."""
    try:
        logger.info(f"API proxy: Starting workflow checkpoints call for run_id: {run_id}")
        start_time = time.time()
        response = requests.get(f"{API_BASE_URL}/api/v1/migrations/{run_id}/checkpoints", timeout=5)
        api_time = time.time() - start_time
        logger.info(f"API proxy: Workflow checkpoints call completed in {api_time:.3f}s - Status: {response.status_code}")
        
        return jsonify(response.json()), response.status_code
    
    except Exception as e:
        logger.error(f"API proxy: Error in workflow checkpoints call for run_id {run_id}: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# This route must come LAST and be more specific to avoid conflicts
@app.route('/api/migrations/<path:run_id>')
def api_migration(run_id):
    """API proxy for specific migration."""
    # Check if this is actually a stats call that got misrouted
    if run_id == 'stats':
        logger.warning("Stats call misrouted to migration endpoint - redirecting to stats")
        return redirect('/api/stats')
    
    try:
        logger.info(f"API proxy: Starting migration call for run_id: {run_id}")
        start_time = time.time()
        response = requests.get(f"{API_BASE_URL}/api/v1/migrations/{run_id}", timeout=5)
        api_time = time.time() - start_time
        logger.info(f"API proxy: Migration call completed in {api_time:.3f}s - Status: {response.status_code}")
        
        return jsonify(response.json()), response.status_code
    
    except Exception as e:
        logger.error(f"API proxy: Error in migration call for run_id {run_id}: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)