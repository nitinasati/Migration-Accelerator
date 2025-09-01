# Migration-Accelerators Platform

A comprehensive platform for intelligent data migration from legacy systems to modern platforms using AI-powered agents and workflow orchestration.

## 🚀 Features

### Core Migration Engine
- **Multi-Agent Architecture**: File Reader, Mapping, Transformation, and API Integration agents
- **LLM-Powered Intelligence**: AI-driven file parsing, mapping selection, and data transformation
- **LangGraph Workflows**: Orchestrated migration processes with state persistence
- **Multiple File Formats**: Support for CSV, Excel, JSON, XML, and Fixed-width files
- **Real-time Monitoring**: Live progress tracking and error handling

### Web Dashboard & API
- **Modern Web Interface**: Responsive dashboard built with Flask and Bootstrap 5
- **RESTful API**: FastAPI-based backend with comprehensive endpoints
- **Workflow Visualization**: Pizza tracker showing step-by-step migration progress
- **Advanced Search**: Filter by record type, file path, and status
- **Real-time Statistics**: Live migration metrics and performance data

### Data Persistence
- **PostgreSQL Database**: Robust state storage and workflow history
- **State Management**: Complete workflow state persistence with JSONB support
- **Checkpoint System**: Workflow recovery and audit trails
- **Performance Metrics**: Duration tracking and success rate analysis

## 🏗️ Architecture

```
Migration-Accelerators/
├── agents/              # AI-powered migration agents
├── workflows/           # LangGraph workflow definitions
├── llm/                # LLM provider abstractions
├── config/             # Configuration and settings
├── memory/             # Database persistence layer
├── web/                # Web dashboard application
│   ├── static/         # CSS, JS, and assets
│   ├── templates/      # HTML templates
│   └── app.py          # Flask web application
├── api/                # RESTful API
│   ├── routes/         # API endpoint definitions
│   ├── models/         # Data models and validation
│   └── main.py         # FastAPI application
└── tests/              # Test suites
```

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.8+
- PostgreSQL 12+
- Required environment variables (see `.env.template`)

### 2. Installation
```bash
# Clone the repository
git clone <repository-url>
cd Migration-Accelerator

# Install dependencies
pip install -r requirements.txt

# Set up database
python main.py db-setup

# Verify database connection
python main.py db-status
```

### 3. Run Migration
```bash
# Execute a migration
python main.py migrate data/input/sample_disability_data.csv --dry-run

# Run with real API integration
python main.py migrate data/input/sample_disability_data.csv
```

### 4. Start Web Dashboard
```bash
# Terminal 1: Start the API
cd api
python main.py

# Terminal 2: Start the web dashboard
cd web
python app.py
```

- **API**: http://localhost:8000 (with docs at /docs)
- **Web Dashboard**: http://localhost:5000

## 🌐 Web Dashboard Features

### Dashboard Overview
- Real-time migration statistics
- Recent migration runs
- Quick action buttons
- Auto-refreshing data

### Migration Management
- List all migration runs
- Filter by record type, file path, and status
- View detailed migration information
- Export migration data

### Workflow Visualization
- **Pizza Tracker**: Visual workflow progress indicator
- Step-by-step execution details
- Progress bars and metadata
- Workflow state inspection

### Search & Analytics
- Full-text search across all fields
- Advanced filtering options
- Performance metrics and trends
- Success rate analysis

## 🔌 API Endpoints

### Core Endpoints
- `GET /api/v1/migrations` - List migrations with filters
- `GET /api/v1/migrations/{id}` - Get migration details
- `GET /api/v1/migrations/{id}/states` - Get workflow states
- `GET /api/v1/migrations/stats` - Get migration statistics
- `GET /api/v1/migrations/search` - Search migrations

### Query Parameters
- **Filtering**: `record_type`, `file_path`, `status`
- **Pagination**: `limit`, `offset`
- **Search**: `q` (full-text search)

## 🗄️ Database Schema

### Core Tables
- `migration_runs`: Migration execution records
- `workflow_states`: Individual workflow step states
- `workflow_checkpoints`: Workflow recovery points
- `workflow_metadata`: Additional workflow information

### Key Features
- **UUID Primary Keys**: Unique identification for all entities
- **JSONB Support**: Flexible metadata and state storage
- **Foreign Key Constraints**: Referential integrity
- **Indexing**: Optimized query performance
- **Triggers**: Automatic timestamp updates

## 🤖 AI Agents

### File Reader Agent
- **Intelligent Parsing**: LLM-powered file format detection
- **Multi-format Support**: CSV, Excel, JSON, XML, Fixed-width
- **Encoding Handling**: Automatic encoding detection and conversion
- **Error Recovery**: Graceful handling of malformed files

### Mapping Agent
- **Smart Selection**: AI-driven mapping configuration selection
- **Field Analysis**: Automatic field type and relationship detection
- **Business Logic**: Intelligent transformation rule generation
- **Validation**: Built-in data quality checks

### Transformation Agent
- **LLM Processing**: AI-powered data transformation
- **Type Conversion**: Automatic data type handling
- **Business Rules**: Application of domain-specific logic
- **Error Handling**: Comprehensive error reporting

### API Integration Agent
- **Multiple Modes**: Real API calls or file output
- **Authentication**: Bearer token, API key, and OAuth2 support
- **Batch Processing**: Efficient bulk data handling
- **Retry Logic**: Automatic retry with exponential backoff

## 🔧 Configuration

### Environment Variables
```env
# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=migration
DB_USER=postgres
DB_PASSWORD=your_password

# LLM Providers
OPENAI_API_KEY=your_openai_key
BEDROCK_ACCESS_KEY=your_aws_key
BEDROCK_SECRET_KEY=your_aws_secret

# LangSmith
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=your_project_name
```

### LLM Provider Support
- **OpenAI**: GPT-3.5, GPT-4 models
- **AWS Bedrock**: Claude, Llama, and other models
- **Anthropic**: Claude models
- **Google**: PaLM and Gemini models
- **Mock Provider**: Fallback for development

## 📊 Monitoring & Observability

### LangSmith Integration
- **LLM Call Tracing**: Monitor all AI model interactions
- **Workflow Execution**: Track complete migration workflows
- **Performance Metrics**: Response times and token usage
- **Error Analysis**: Detailed error tracking and debugging

### Structured Logging
- **JSON Format**: Machine-readable log output
- **Context Tracking**: Correlation IDs and request tracing
- **Performance Metrics**: Timing and resource usage
- **Error Reporting**: Comprehensive error context

## 🧪 Testing

### Test Coverage
- **Unit Tests**: Individual agent and utility testing
- **Integration Tests**: Workflow orchestration testing
- **API Tests**: Endpoint validation and error handling
- **Mock Providers**: Offline development and testing

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=.

# Run specific test categories
pytest tests/test_agents/
pytest tests/test_workflows/
```

## 🚀 Performance & Scalability

### Optimization Features
- **Async Operations**: Non-blocking I/O operations
- **Connection Pooling**: Efficient database connections
- **Batch Processing**: Bulk data operations
- **Caching**: Intelligent result caching
- **Resource Management**: Memory and connection cleanup

### Scalability Considerations
- **Horizontal Scaling**: Stateless agent design
- **Database Optimization**: Proper indexing and query optimization
- **Load Balancing**: API endpoint distribution
- **Monitoring**: Performance metrics and alerting

## 🔒 Security

### Best Practices
- **Environment Variables**: Secure credential management
- **Input Validation**: Pydantic model validation
- **SQL Injection Protection**: Parameterized queries
- **Authentication**: Secure API access control
- **Audit Logging**: Complete operation tracking

## 🤝 Contributing

### Development Guidelines
- **Code Style**: PEP 8 compliance with type hints
- **Documentation**: Comprehensive docstrings and README updates
- **Testing**: Maintain high test coverage
- **Code Review**: All changes require review
- **CI/CD**: Automated testing and deployment

### Development Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Set up pre-commit hooks
pre-commit install
```

## 📈 Roadmap

### Planned Features
- **Real-time Notifications**: WebSocket-based updates
- **Advanced Analytics**: Machine learning insights
- **Multi-tenant Support**: Organization and user management
- **Plugin System**: Extensible agent architecture
- **Cloud Deployment**: Kubernetes and Docker support
- **Mobile App**: React Native mobile dashboard

### Performance Improvements
- **Streaming Processing**: Large file handling
- **Distributed Processing**: Multi-node execution
- **Caching Layer**: Redis integration
- **CDN Integration**: Static asset optimization

## 📚 Documentation

### Additional Resources
- **API Documentation**: Interactive Swagger UI at `/docs`
- **Code Examples**: Sample implementations and use cases
- **Troubleshooting**: Common issues and solutions
- **Architecture Guide**: Detailed system design documentation

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

### Getting Help
- **Issues**: GitHub issue tracker
- **Discussions**: GitHub discussions
- **Documentation**: Comprehensive guides and examples
- **Community**: Active developer community

### Contact
- **Maintainers**: Core development team
- **Contributors**: Community contributors
- **Users**: Migration-Accelerators community

---

**Migration-Accelerators** - Transforming data migration with AI-powered intelligence! 🚀
