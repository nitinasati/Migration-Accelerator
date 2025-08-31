# Setup script to add Graphviz to PATH permanently
# Run this script as Administrator to add Graphviz to system PATH

Write-Host "Setting up Graphviz for Migration-Accelerators..." -ForegroundColor Green

# Check if Graphviz is installed
$graphvizPath = "C:\Program Files\Graphviz\bin"
if (Test-Path $graphvizPath) {
    Write-Host "✓ Graphviz found at: $graphvizPath" -ForegroundColor Green
    
    # Add to current session PATH
    $env:PATH += ";$graphvizPath"
    Write-Host "✓ Added Graphviz to current session PATH" -ForegroundColor Green
    
    # Test if dot command works
    try {
        $version = & dot -V 2>&1
        Write-Host "✓ Graphviz is working: $version" -ForegroundColor Green
        Write-Host ""
        Write-Host "You can now run: python main.py graph" -ForegroundColor Cyan
        Write-Host "Or: python main.py migrate data/input/sample_disability_data.csv --dry-run" -ForegroundColor Cyan
    }
    catch {
        Write-Host "✗ Graphviz dot command not working" -ForegroundColor Red
    }
}
else {
    Write-Host "✗ Graphviz not found at expected location" -ForegroundColor Red
    Write-Host "Please install Graphviz first:" -ForegroundColor Yellow
    Write-Host "  winget install graphviz" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Note: To make this permanent, add the following to your system PATH:" -ForegroundColor Yellow
Write-Host "  C:\Program Files\Graphviz\bin" -ForegroundColor Yellow
