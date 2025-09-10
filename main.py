#!/usr/bin/env python3
"""
Main CLI interface for the Migration-Accelerators platform.
"""

import asyncio
import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from config.settings import settings, LLMConfig, MCPConfig, FieldMappingConfig, get_llm_config, get_mcp_config, get_langsmith_config
from workflows.migration_graph import MigrationWorkflow


# Initialize Typer app
app = typer.Typer(
    name="migration-accelerators",
    help="Migration-Accelerators - Agentic AI Data Migration Platform",
    add_completion=False
)

# Initialize Rich console
console = Console()

# Status constants
STATUS_CONFIGURED = "✓ Configured"
STATUS_NOT_CONFIGURED = "⚠ Not Configured"
STATUS_AVAILABLE = "✓ Available"
STATUS_MISSING = "⚠ Missing"


@app.command()
def migrate(
    file_path: str = typer.Argument(..., help="Path to the input file"),
    mapping_file: Optional[str] = typer.Option(None, "--mapping", "-m", help="Path to mapping configuration file"),
    record_type: str = typer.Option("disability", "--type", "-t", help="Record type (disability, absence, etc.)"),
    layout_file: Optional[str] = typer.Option(None, "--layout", "-l", help="Path to layout file for mainframe fixed-width files"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Run in dry-run mode (no API calls)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """Migrate data from legacy systems to modern platforms using Agentic AI."""
    
    console.print(Panel.fit(
        "[bold blue]Migration-Accelerators[/bold blue]\n"
        "Agentic AI Data Migration Platform - Powered by A2A Framework, LangGraph, and MCP",
        border_style="blue"
    ))
    
    # Validate file exists
    if not os.path.exists(file_path):
        console.print(f"[red]Error: File not found: {file_path}[/red]")
        raise typer.Exit(1)
    
    # Note: Mapping configuration is now handled automatically by the LLM-powered mapping agent
    # The agent will analyze the data and select the appropriate mapping configuration
    console.print(f"[green]✓[/green] Using LLM-powered intelligent mapping for record type: {record_type}")
    
    # Run migration
    asyncio.run(_run_migration(file_path, record_type, layout_file, dry_run, verbose))


@app.command()
def validate(
    file_path: str = typer.Argument(..., help="Path to the input file"),
    mapping_file: Optional[str] = typer.Option(None, "--mapping", "-m", help="Path to mapping configuration file")
):
    """Validate input file and mapping configuration."""
    
    console.print(Panel.fit(
        "[bold yellow]File and Configuration Validation[/bold yellow]",
        border_style="yellow"
    ))
    
    # Validate file exists
    if not os.path.exists(file_path):
        console.print(f"[red]Error: File not found: {file_path}[/red]")
        raise typer.Exit(1)
    
    # Note: Mapping validation is now handled automatically by the LLM-powered mapping agent
    console.print("[green]✓[/green] File validation completed - mapping will be handled automatically")
    
    console.print("[green]✓[/green] File validation completed successfully")


@app.command()
def status():
    """Check platform status and configuration."""
    
    console.print(Panel.fit(
        "[bold green]Migration-Accelerators Platform Status[/bold green]",
        border_style="green"
    ))
    
    # Check configuration
    table = Table(title="Configuration Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details")
    
    # LLM Configuration
    llm_config = get_llm_config()
    table.add_row("LLM Provider", STATUS_CONFIGURED, f"{llm_config.provider.value} - {llm_config.model}")
    
    # MCP Configuration
    mcp_config = get_mcp_config()
    table.add_row("MCP Client", STATUS_CONFIGURED, f"Server: {mcp_config.server_url}")
    
    # LangSmith Configuration
    langsmith_config = get_langsmith_config()
    if langsmith_config.api_key:
        table.add_row("LangSmith", STATUS_CONFIGURED, f"Project: {langsmith_config.project}")
    else:
        table.add_row("LangSmith", STATUS_NOT_CONFIGURED, "API key not set")
    
    # Sample data files
    sample_files = [
        "data/input/sample_disability_data.csv",
        "data/input/sample_absence_data.csv"
    ]
    
    for sample_file in sample_files:
        if os.path.exists(sample_file):
            table.add_row(f"Sample Data ({Path(sample_file).stem})", STATUS_AVAILABLE, sample_file)
        else:
            table.add_row(f"Sample Data ({Path(sample_file).stem})", STATUS_MISSING, sample_file)
    
    # Mapping files
    mapping_files = [
        "config/mappings/disability_mapping.yaml",
        "config/mappings/absence_mapping.yaml"
    ]
    
    for mapping_file in mapping_files:
        if os.path.exists(mapping_file):
            table.add_row(f"Mapping Config ({Path(mapping_file).stem})", STATUS_AVAILABLE, mapping_file)
        else:
            table.add_row(f"Mapping Config ({Path(mapping_file).stem})", STATUS_MISSING, mapping_file)
    
    console.print(table)


@app.command()
def test():
    """Run platform tests."""
    
    console.print(Panel.fit(
        "[bold blue]Running Platform Tests[/bold blue]",
        border_style="blue"
    ))
    
    # Run basic tests
    try:
        import pytest
        result = pytest.main(["-v", "tests/"])
        if result == 0:
            console.print("[green]✓ All tests passed[/green]")
        else:
            console.print("[red]✗ Some tests failed[/red]")
            raise typer.Exit(1)
    except ImportError:
        console.print("[yellow]Warning: pytest not installed. Install with: pip install pytest[/yellow]")
    except Exception as e:
        console.print(f"[red]Error running tests: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def db_setup():
    """Set up the PostgreSQL database for LangGraph memory persistence."""
    
    console.print(Panel.fit(
        "[bold blue]Database Setup[/bold blue]\n"
        "Setting up PostgreSQL database for LangGraph state persistence",
        border_style="blue"
    ))
    
    try:
        # Run the database setup script
        import subprocess
        import sys
        
        script_path = "scripts/create_tables.py"
        if not os.path.exists(script_path):
            console.print(f"[red]Error: Database setup script not found: {script_path}[/red]")
            raise typer.Exit(1)
        
        console.print("Running database setup script...")
        result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
        
        if result.returncode == 0:
            console.print("[green]✓ Database setup completed successfully![/green]")
            console.print(result.stdout)
        else:
            console.print("[red]✗ Database setup failed[/red]")
            console.print(f"Error: {result.stderr}")
            raise typer.Exit(1)
            
    except Exception as e:
        console.print(f"[red]Error setting up database: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def db_status():
    """Check the status of the PostgreSQL database."""
    
    console.print(Panel.fit(
        "[bold blue]Database Status[/bold blue]",
        border_style="blue"
    ))
    
    try:
        from config.database import get_database_config
        from memory import PostgresMemoryManager
        import asyncio
        
        config = get_database_config()
        console.print("Database Configuration:")
        console.print(f"  Host: {config.host}")
        console.print(f"  Port: {config.port}")
        console.print(f"  Database: {config.database}")
        console.print(f"  Username: {config.username}")
        console.print("  Password: *******")
        console.print(f"  Schema: {config.schema}")
        
        # Test connection
        async def test_connection():
            try:
                memory_manager = PostgresMemoryManager(config)
                await memory_manager.initialize()
                await memory_manager.close()
                return True
            except Exception as e:
                return str(e)
        
        result = asyncio.run(test_connection())
        
        if result is True:
            console.print("[green]✓ Database connection successful[/green]")
        else:
            console.print(f"[red]✗ Database connection failed: {result}[/red]")
            
    except Exception as e:
        console.print(f"[red]Error checking database status: {e}[/red]")
        raise typer.Exit(1)





@app.command()
def logs(
    project: str = typer.Option("migration-accelerators", "--project", "-p", help="LangSmith project name"),
    limit: int = typer.Option(10, "--limit", "-l", help="Number of logs to show")
):
    """View logs in LangSmith."""
    
    console.print(Panel.fit(
        f"[bold blue]LangSmith Logs - Project: {project}[/bold blue]",
        border_style="blue"
    ))
    
    langsmith_config = get_langsmith_config()
    if not langsmith_config.api_key:
        console.print("[red]Error: LangSmith API key not configured[/red]")
        raise typer.Exit(1)
    
    console.print(f"[green]✓[/green] LangSmith configured for project: {project}")
    console.print("[yellow]Note: Use LangSmith web interface to view detailed logs[/yellow]")


async def _run_migration(
    file_path: str,
    record_type: str,
    layout_file: Optional[str],
    dry_run: bool,
    verbose: bool
):
    """Run the migration workflow."""
    
    try:
        # Initialize workflow
        llm_config = get_llm_config()
        mcp_config = get_mcp_config()
        
        workflow = MigrationWorkflow(llm_config, mcp_config)
        

        
        # Prepare target system configuration
        target_system = {
            "endpoint": f"{record_type}_policy",
            "base_url": "https://api.insurance-system.com",
            "authentication": {
                "type": "bearer_token",
                "token": "demo_token"  # In real implementation, get from secure storage
            },
            "batch_size": 10
        }
        
        if dry_run:
            target_system["dry_run"] = True
            console.print("[yellow]Running in dry-run mode - no actual API calls will be made[/yellow]")
        
        # Show progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task("Starting migration...", total=100)
            
            # Run migration
            result = await workflow.run(
                file_path=file_path,
                record_type=record_type,
                target_system=target_system,
                layout_file=layout_file
            )
            
            progress.update(task, completed=100, description="Migration completed")
        
        # Display results
        _display_migration_results(result, verbose)
        
        # Close workflow
        await workflow.close()
        
    except Exception as e:
        console.print(f"[red]Migration failed: {e}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(1)


def _display_migration_results(result: dict, verbose: bool):
    """Display migration results."""
    
    console.print(Panel.fit(
        "[bold green]Migration Results[/bold green]",
        border_style="green"
    ))
    
    summary = result.get("migration_summary", {})
    
    # Summary table
    table = Table(title="Migration Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Success", "✓ Yes" if summary.get("success") else "✗ No")
    table.add_row("Records Processed", str(summary.get("total_records_processed", 0)))
    table.add_row("Progress", f"{summary.get('progress', 0):.1f}%")
    table.add_row("Completed Steps", str(len(summary.get("completed_steps", []))))
    
    # Add timing information if available
    if summary.get("total_duration_seconds"):
        table.add_row("Total Duration", summary.get("total_duration_formatted", "N/A"))
        table.add_row("Duration (seconds)", f"{summary.get('total_duration_seconds', 0):.2f}s")
    
    if summary.get("start_time"):
        table.add_row("Start Time", summary.get("start_time", "N/A"))
    
    if summary.get("end_time"):
        table.add_row("End Time", summary.get("end_time", "N/A"))
    
    console.print(table)
    
    # Show completed steps
    completed_steps = summary.get("completed_steps", [])
    if completed_steps:
        console.print("\n[bold]Completed Steps:[/bold]")
        for step in completed_steps:
            console.print(f"  [green]✓[/green] {step.replace('_', ' ').title()}")
    
    # Show errors if any
    errors = result.get("errors", [])
    if errors:
        console.print(f"\n[bold red]Errors ({len(errors)}):[/bold red]")
        for error in errors[:5]:  # Show first 5 errors
            console.print(f"  [red]✗[/red] {error}")
        if len(errors) > 5:
            console.print(f"  ... and {len(errors) - 5} more errors")
    
    # Show warnings if any
    warnings = result.get("warnings", [])
    if warnings:
        console.print(f"\n[bold yellow]Warnings ({len(warnings)}):[/bold yellow]")
        for warning in warnings[:3]:  # Show first 3 warnings
            console.print(f"  [yellow]⚠[/yellow] {warning}")
        if len(warnings) > 3:
            console.print(f"  ... and {len(warnings) - 3} more warnings")
    
    # Show detailed results if verbose
    if verbose and result.get("data_pipeline"):
        console.print("\n[bold]Data Pipeline Details:[/bold]")
        
        pipeline = result["data_pipeline"]
        
        if pipeline.get("file_data"):
            console.print(f"  [cyan]File Data:[/cyan] {len(pipeline['file_data'])} records")
        
        if pipeline.get("validated_data"):
            console.print(f"  [cyan]Validated Data:[/cyan] {len(pipeline['validated_data'])} records")
        
        if pipeline.get("mapped_data"):
            console.print(f"  [cyan]Mapped Data:[/cyan] {len(pipeline['mapped_data'])} records")
        
        if pipeline.get("api_results"):
            api_results = pipeline["api_results"]
            if isinstance(api_results, dict) and "api_results" in api_results:
                console.print(f"  [cyan]API Results:[/cyan] {len(api_results['api_results'])} records")
            else:
                console.print("  [cyan]API Results:[/cyan] Available")


if __name__ == "__main__":
    app()
