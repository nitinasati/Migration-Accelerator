"""
Flask web application for the Migration-Accelerators dashboard.
"""

import os
import requests
from flask import Flask, render_template, request, jsonify, redirect, url_for
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key')

# API base URL
API_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:8000')

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

@app.route('/')
def index():
    """Dashboard home page."""
    try:
        # Get migration statistics
        response = requests.get(f"{API_BASE_URL}/api/v1/migrations/stats")
        print(response.json())
        if response.status_code == 200:
            stats = response.json()
            # Ensure we have default values if any keys are missing
            stats = {
                'total_migrations': stats.get('total_migrations', 0),
                'status_breakdown': stats.get('status_breakdown', {}),
                'recent_24h': stats.get('recent_24h', 0)
            }
        else:
            stats = {
                'total_migrations': 0,
                'status_breakdown': {},
                'recent_24h': 0
            }
        
        # Get recent migrations
        response = requests.get(f"{API_BASE_URL}/api/v1/migrations/?limit=5")
        if response.status_code == 200:
            recent_migrations = response.json()
        else:
            recent_migrations = []
        
        return render_template('index.html', stats=stats, recent_migrations=recent_migrations)
    
    except Exception as e:
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
            
            response = requests.get(f"{API_BASE_URL}/api/v1/migrations/search/", params=search_params)
        else:
            # Get all migrations without filters
            response = requests.get(f"{API_BASE_URL}/api/v1/migrations/?limit={limit}&offset={offset}")
        
        if response.status_code == 200:
            migrations = response.json()
        else:
            migrations = []
        
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
        # Get migration details
        response = requests.get(f"{API_BASE_URL}/api/v1/migrations/{run_id}")
        if response.status_code != 200:
            return render_template(
                'migration_detail.html', 
                error="Migration not found",
                migration_run=None,
                workflow_states=[],
                workflow_checkpoints=[]
            )
        
        migration_run = response.json()
        
        # Get workflow states
        response = requests.get(f"{API_BASE_URL}/api/v1/migrations/{run_id}/states")
        if response.status_code == 200:
            workflow_states = response.json()
        else:
            workflow_states = []
        
        # Get workflow checkpoints
        response = requests.get(f"{API_BASE_URL}/api/v1/migrations/{run_id}/checkpoints")
        if response.status_code == 200:
            workflow_checkpoints = response.json()
        else:
            workflow_checkpoints = []
        
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
            
            response = requests.get(f"{API_BASE_URL}/api/v1/migrations/search/", params=params)
            if response.status_code == 200:
                results = response.json()
            else:
                results = []
        else:
            results = []
        
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
@app.route('/api/migrations')
def api_migrations():
    """API proxy for migrations."""
    try:
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        response = requests.get(f"{API_BASE_URL}/api/v1/migrations/?limit={limit}&offset={offset}")
        return jsonify(response.json()), response.status_code
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/migrations/<run_id>')
def api_migration(run_id):
    """API proxy for specific migration."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/migrations/{run_id}")
        return jsonify(response.json()), response.status_code
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/migrations/<run_id>/states')
def api_workflow_states(run_id):
    """API proxy for workflow states."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/migrations/{run_id}/states")
        return jsonify(response.json()), response.status_code
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/migrations/<run_id>/checkpoints')
def api_workflow_checkpoints(run_id):
    """API proxy for workflow checkpoints."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/migrations/{run_id}/checkpoints")
        return jsonify(response.json()), response.status_code
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/stats')
def api_stats():
    """API proxy for statistics."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/migrations/stats")
        return jsonify(response.json()), response.status_code
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)