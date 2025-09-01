# Migration-Accelerators Web Dashboard

A modern web dashboard for visualizing and monitoring data migration workflows.

## Features

- **Dashboard Overview**: Real-time statistics and recent migrations
- **Migration List**: View all migration runs with filtering capabilities
- **Workflow Visualization**: Pizza tracker showing workflow progress
- **Search Functionality**: Search across file paths, record types, and statuses
- **Responsive Design**: Mobile-friendly interface using Bootstrap 5

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set environment variables in `.env` file:
```env
API_BASE_URL=http://localhost:8000/api/v1
```

3. Run the application:
```bash
python app.py
```

The web application will be available at `http://localhost:5000`

## Usage

### Dashboard
- View migration statistics
- Monitor recent migrations
- Quick access to common actions

### Migrations List
- Filter by record type, file path, and status
- View migration details
- Click on run ID to see workflow states

### Workflow Visualization
- Pizza tracker showing step-by-step progress
- Detailed view of each workflow state
- Progress indicators and metadata

### Search
- Search across multiple fields
- Quick search suggestions
- Real-time results

## API Integration

The web application connects to the Migration-Accelerators API to fetch data. Ensure the API is running on the configured port before starting the web application.

## Technologies Used

- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **UI Framework**: Bootstrap 5
- **Icons**: Bootstrap Icons
- **HTTP Client**: Requests library
