#!/usr/bin/env python3
"""
Main CLI interface for the Agentic Insurance Data Migration Platform.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm

from config.settings import settings, LLMConfig, MCPConfig, FieldMappingConfig
from llm.providers import LLMProviderFactory
from mcp.client import MCPToolManager
from workflows.migration_graph import MigrationWorkflow
from config.mappings import load_mapping_config

# Initialize Typer app
app = typer.Typer(
    name="migration-platform",
    help="Agentic Insurance Data Migration Platform",
    add_completion=False
)

# Initialize Rich console
console = Console()


@app.command()
def migrate(
    file_path: str = typer.Argument(..., help="Path to the input file"),
    mapping_file: Optional[str] = typer.Option(None, "--mapping", "-m", help="Path to mapping configuration file"),
    record_type: str = typer.Option("disability", "--type", "-t", help="Record type (disability, absence, etc.)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Run in dry-run mode (no API calls)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """Migrate insurance data from legacy mainframe to modern system."""
    
    console.print(Panel.fit(
        "[bold blue]Agentic Insurance Data Migration Platform[/bold blue]\n"
        "Powered by A2A Framework, LangGraph, and MCP",
        border_style="blue"
    ))
    
    # Validate file exists
    if not os.path.exists(file_path):
        console.print(f"[red]Error: File not found: {file_path}[/red]")
        raise typer.Exit(1)
    
    # Load mapping configuration
    mapping_config = None
    if mapping_file:
        if not os.path.exists(mapping_file):
            console.print(f"[red]Error: Mapping file not found: {mapping_file}[/red]")
            raise typer.Exit(1)
        
        try:
            mapping_config = load_mapping_config(mapping_file)
            console.print(f"[green]✓[/green] Loaded mapping configuration: {mapping_file}")
        except Exception as e:
            console.print(f"[red]Error loading mapping file: {e}[/red]")
            raise typer.Exit(1)
    
    # Run migration
    asyncio.run(_run_migration(file_path, mapping_config, record_type, dry_run, verbose))


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
    
    # Validate mapping file if provided
    if mapping_file:
        if not os.path.exists(mapping_file):
            console.print(f"[red]Error: Mapping file not found: {mapping_file}[/red]")
            raise typer.Exit(1)
        
        try:
            mapping_config = load_mapping_config(mapping_file)
            console.print(f"[green]✓[/green] Mapping configuration is valid")
        except Exception as e:
            console.print(f"[red]Error: Invalid mapping configuration: {e}[/red]")
            raise typer.Exit(1)
    
    # Validate file format
    file_format = _detect_file_format(file_path)
    console.print(f"[green]✓[/green] File format detected: {file_format}")
    
    # Basic file analysis
    try:
        import pandas as pd
        if file_format == "csv":
            df = pd.read_csv(file_path)
        elif file_format == "excel":
            df = pd.read_excel(file_path)
        else:
            console.print(f"[yellow]Warning: File format {file_format} not fully supported for validation[/yellow]")
            raise typer.Exit(0)
        
        console.print(f"[green]✓[/green] File contains {len(df)} records")
        console.print(f"[green]✓[/green] File has {len(df.columns)} columns")
        
        # Show column names
        table = Table(title="File Columns")
        table.add_column("Column Name", style="cyan")
        table.add_column("Data Type", style="magenta")
        table.add_column("Non-Null Count", style="green")
        
        for col in df.columns:
            table.add_row(
                col,
                str(df[col].dtype),
                str(df[col].count())
            )
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error analyzing file: {e}[/red]")
        raise typer.Exit(1)
    
    console.print("[green]✓[/green] Validation completed successfully")


@app.command()
def status():
    """Show platform status and configuration."""
    
    console.print(Panel.fit(
        "[bold green]Platform Status[/bold green]",
        border_style="green"
    ))
    
    # Configuration status
    config_table = Table(title="Configuration")
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="green")
    config_table.add_column("Status", style="yellow")
    
    # LLM Configuration
    llm_config = settings.get_llm_config()
    config_table.add_row(
        "LLM Provider",
        llm_config.provider.value,
        "[green]✓[/green]" if llm_config.api_key else "[red]✗[/red]"
    )
    config_table.add_row(
        "LLM Model",
        llm_config.model,
        "[green]✓[/green]"
    )
    
    # MCP Configuration
    mcp_config = settings.get_mcp_config()
    config_table.add_row(
        "MCP Server",
        mcp_config.server_url,
        "[green]✓[/green]"
    )
    
    # LangSmith Configuration
    langsmith_config = settings.get_langsmith_config()
    config_table.add_row(
        "LangSmith Project",
        langsmith_config.project,
        "[green]✓[/green]" if langsmith_config.api_key else "[yellow]⚠[/yellow]"
    )
    
    console.print(config_table)
    
    # Directory status
    dir_table = Table(title="Directory Status")
    dir_table.add_column("Directory", style="cyan")
    dir_table.add_column("Path", style="green")
    dir_table.add_column("Status", style="yellow")
    
    directories = [
        ("Input Directory", settings.input_dir),
        ("Output Directory", settings.output_dir),
        ("Temp Directory", settings.temp_dir)
    ]
    
    for name, path in directories:
        exists = os.path.exists(path)
        dir_table.add_row(
            name,
            path,
            "[green]✓[/green]" if exists else "[red]✗[/red]"
        )
    
    console.print(dir_table)


@app.command()
def test():
    """Run platform tests."""
    
    console.print(Panel.fit(
        "[bold purple]Platform Testing[/bold purple]",
        border_style="purple"
    ))
    
    # Test LLM provider
    console.print("[cyan]Testing LLM Provider...[/cyan]")
    try:
        llm_config = settings.get_llm_config()
        llm_provider = LLMProviderFactory.create(llm_config)
        console.print("[green]✓[/green] LLM provider initialized successfully")
    except Exception as e:
        console.print(f"[red]✗[/red] LLM provider failed: {e}")
    
    # Test MCP manager
    console.print("[cyan]Testing MCP Manager...[/cyan]")
    try:
        mcp_config = settings.get_mcp_config()
        mcp_manager = MCPToolManager(mcp_config)
        console.print("[green]✓[/green] MCP manager initialized successfully")
    except Exception as e:
        console.print(f"[red]✗[/red] MCP manager failed: {e}")
    
    # Test workflow
    console.print("[cyan]Testing Migration Workflow...[/cyan]")
    try:
        llm_config = settings.get_llm_config()
        mcp_config = settings.get_mcp_config()
        workflow = MigrationWorkflow(llm_config, mcp_config)
        console.print("[green]✓[/green] Migration workflow initialized successfully")
    except Exception as e:
        console.print(f"[red]✗[/red] Migration workflow failed: {e}")
    
    console.print("[green]✓[/green] All tests completed")


@app.command()
def logs(
    project: str = typer.Option("insurance-migration", "--project", "-p", help="LangSmith project name"),
    limit: int = typer.Option(10, "--limit", "-l", help="Number of logs to show")
):
    """Show recent LangSmith logs."""
    
    console.print(Panel.fit(
        "[bold orange]LangSmith Logs[/bold orange]",
        border_style="orange"
    ))
    
    try:
        from langsmith import Client
        
        client = Client()
        runs = client.list_runs(project_name=project, limit=limit)
        
        if not runs:
            console.print("[yellow]No logs found[/yellow]")
            return
        
        log_table = Table(title=f"Recent Logs - {project}")
        log_table.add_column("Run ID", style="cyan")
        log_table.add_column("Name", style="green")
        log_table.add_column("Status", style="yellow")
        log_table.add_column("Duration", style="magenta")
        log_table.add_column("Created", style="blue")
        
        for run in runs:
            status_style = "[green]✓[/green]" if run.status == "completed" else "[red]✗[/red]"
            duration = f"{run.end_time - run.start_time:.2f}s" if run.end_time and run.start_time else "N/A"
            
            log_table.add_row(
                run.id[:8],
                run.name or "N/A",
                status_style,
                duration,
                run.start_time.strftime("%Y-%m-%d %H:%M:%S") if run.start_time else "N/A"
            )
        
        console.print(log_table)
        
    except Exception as e:
        console.print(f"[red]Error accessing LangSmith logs: {e}[/red]")
        console.print("[yellow]Make sure LANGCHAIN_API_KEY is set correctly[/yellow]")


async def _run_migration(
    file_path: str,
    mapping_config: Optional[FieldMappingConfig],
    record_type: str,
    dry_run: bool,
    verbose: bool
):
    """Run the migration workflow."""
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        # Initialize components
        task = progress.add_task("Initializing components...", total=None)
        
        try:
            # Get configurations
            llm_config = settings.get_llm_config()
            mcp_config = settings.get_mcp_config()
            
            # Initialize workflow
            workflow = MigrationWorkflow(llm_config, mcp_config)
            
            progress.update(task, description="Running migration workflow...")
            
            # Run migration
            result = await workflow.run(file_path, mapping_config, record_type)
            
            progress.update(task, description="Migration completed")
            
            # Display results
            _display_migration_results(result, verbose)
            
            # Cleanup
            await workflow.cleanup()
            
        except Exception as e:
            progress.update(task, description="Migration failed")
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)


def _display_migration_results(result: dict, verbose: bool):
    """Display migration results."""
    
    if result["success"]:
        console.print("[green]✓[/green] Migration completed successfully")
    else:
        console.print("[red]✗[/red] Migration failed")
        if "error" in result:
            console.print(f"[red]Error: {result['error']}[/red]")
    
    # Display summary
    summary = result.get("summary", {})
    
    summary_table = Table(title="Migration Summary")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")
    
    summary_table.add_row("Total Records", str(summary.get("total_records", 0)))
    summary_table.add_row("Successful Records", str(summary.get("successful_records", 0)))
    summary_table.add_row("Failed Records", str(summary.get("failed_records", 0)))
    
    success_rate = summary.get("success_rate", 0)
    summary_table.add_row("Success Rate", f"{success_rate:.1%}")
    
    duration = summary.get("duration_seconds")
    if duration:
        summary_table.add_row("Duration", f"{duration:.2f} seconds")
    
    console.print(summary_table)
    
    # Display errors and warnings
    errors = summary.get("errors", [])
    warnings = summary.get("warnings", [])
    
    if errors:
        console.print("[red]Errors:[/red]")
        for error in errors:
            console.print(f"  • {error}")
    
    if warnings:
        console.print("[yellow]Warnings:[/yellow]")
        for warning in warnings:
            console.print(f"  • {warning}")
    
    # Display verbose information
    if verbose and "state" in result:
        state = result["state"]
        console.print(f"\n[cyan]Current Step:[/cyan] {state.get('current_step', 'unknown')}")
        console.print(f"[cyan]File Path:[/cyan] {state.get('file_path', 'unknown')}")
        console.print(f"[cyan]Record Type:[/cyan] {state.get('record_type', 'unknown')}")


def _detect_file_format(file_path: str) -> str:
    """Detect file format based on extension."""
    path = Path(file_path)
    extension = path.suffix.lower()
    
    if extension == '.csv':
        return "csv"
    elif extension in ['.xlsx', '.xls']:
        return "excel"
    elif extension == '.json':
        return "json"
    elif extension == '.xml':
        return "xml"
    elif extension in ['.txt', '.dat']:
        return "fixed_width"
    else:
        return "unknown"


if __name__ == "__main__":
    app()
