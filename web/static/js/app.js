/**
 * Main JavaScript file for Migration-Accelerators Dashboard
 */

// Global variables
let refreshInterval;
let currentPage = 'dashboard';

// Initialize application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

/**
 * Initialize the application
 */
function initializeApp() {
    // Set up navigation highlighting
    setupNavigation();
    
    // Set up auto-refresh for dashboard
    if (currentPage === 'dashboard') {
        setupAutoRefresh();
    }
    
    // Set up form submissions
    setupFormHandling();
    
    // Set up tooltips
    setupTooltips();
    
    // Set up search functionality
    setupSearch();
}

/**
 * Set up navigation highlighting
 */
function setupNavigation() {
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.navbar-nav .nav-link');
    
    navLinks.forEach(link => {
        if (link.getAttribute('href') === currentPath) {
            link.classList.add('active');
        }
    });
}

/**
 * Set up auto-refresh for dashboard
 */
function setupAutoRefresh() {
    // DISABLED: This conflicts with our smart refresh logic in index.html
    // The dashboard now uses manual refresh and smart update prevention
    console.log('Auto-refresh disabled - using smart refresh logic instead');
    
    // Uncomment the line below if you want to re-enable auto-refresh
     refreshInterval = setInterval(() => {
         if (currentPage === 'dashboard') {
             loadDashboardData();
        }
     }, 5000);
}

/**
 * Set up form handling
 */
function setupFormHandling() {
    // Handle filter form submissions
    const filterForm = document.querySelector('form[action*="migrations"]');
    if (filterForm) {
        filterForm.addEventListener('submit', function(e) {
            // Add loading state
            const submitBtn = this.querySelector('button[type="submit"]');
            if (submitBtn) {
                submitBtn.innerHTML = '<i class="bi bi-hourglass-split"></i> Loading...';
                submitBtn.disabled = true;
            }
        });
    }
    
    // Handle search form submissions
    const searchForm = document.querySelector('form[action*="search"]');
    if (searchForm) {
        searchForm.addEventListener('submit', function(e) {
            const searchInput = this.querySelector('input[name="q"]');
            if (searchInput && searchInput.value.trim() === '') {
                e.preventDefault();
                showAlert('Please enter a search term', 'warning');
            }
        });
    }
}

/**
 * Set up tooltips
 */
function setupTooltips() {
    // Initialize Bootstrap tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

/**
 * Set up search functionality
 */
function setupSearch() {
    const searchInput = document.querySelector('input[name="q"]');
    if (searchInput) {
        // Add search suggestions
        searchInput.addEventListener('input', function() {
            const query = this.value.trim();
            if (query.length > 2) {
                // Could implement live search suggestions here
                console.log('Searching for:', query);
            }
        });
        
        // Add keyboard shortcuts
        searchInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && e.ctrlKey) {
                e.preventDefault();
                this.form.submit();
            }
        });
    }
}

/**
 * Show alert message
 */
function showAlert(message, type = 'info') {
    const alertContainer = document.createElement('div');
    alertContainer.className = `alert alert-${type} alert-dismissible fade show`;
    alertContainer.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    // Insert at the top of the main content
    const mainContent = document.querySelector('main .container');
    if (mainContent) {
        mainContent.insertBefore(alertContainer, mainContent.firstChild);
        
        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            if (alertContainer.parentNode) {
                alertContainer.remove();
            }
        }, 5000);
    }
}

/**
 * Format duration in seconds to human readable format
 */
function formatDuration(seconds) {
    if (!seconds || isNaN(seconds)) return 'N/A';
    
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    
    if (hours > 0) {
        return `${hours}h ${minutes}m ${secs}s`;
    } else if (minutes > 0) {
        return `${minutes}m ${secs}s`;
    } else {
        return `${secs}s`;
    }
}

/**
 * Format datetime string
 */
function formatDateTime(dateString) {
    if (!dateString) return 'N/A';
    
    try {
        const date = new Date(dateString);
        return date.toLocaleString();
    } catch (e) {
        return dateString;
    }
}

/**
 * Get status color class
 */
function getStatusColor(status) {
    const colors = {
        'running': 'primary',
        'completed': 'success',
        'failed': 'danger',
        'paused': 'warning'
    };
    return colors[status] || 'secondary';
}



/**
 * Load dashboard data
 */
async function loadDashboardData() {
    // Declare statsCards outside try block so it's available in finally
    let statsCards = null;
    
    try {
        // Auto-refresh in progress
        
        // Show loading state
        statsCards = document.getElementById('stats-cards');
        if (statsCards) {
            statsCards.classList.add('loading');
        }
        
        // Load statistics
        const statsResponse = await fetch('/api/migrations/stats');
        if (!statsResponse.ok) throw new Error('Failed to load statistics');
        const stats = await statsResponse.json();
        
        // Update statistics cards
        updateStatisticsCards(stats);
        
        // Load recent migrations
        const migrationsResponse = await fetch('/api/migrations?limit=5');
        if (!migrationsResponse.ok) throw new Error('Failed to load migrations');
        const migrations = await migrationsResponse.json();
        
        updateRecentMigrationsTable(migrations);
        

        
        // Auto-refresh completed successfully
        
    } catch (error) {
        console.error('Failed to load dashboard data:', error);
        showAlert('Failed to load dashboard data. Please refresh the page.', 'danger');
        // Auto-refresh failed
    } finally {
        // Remove loading state
        if (statsCards) {
            statsCards.classList.remove('loading');
        }
    }
}

/**
 * Update statistics cards
 */
function updateStatisticsCards(stats) {
    // Use smart update logic to prevent overwriting valid data
    const elements = {
        'total-runs': stats.total_migrations || stats.total_runs || 0,
        'completed-runs': stats.status_breakdown?.completed || stats.completed_runs || 0,
        'running-runs': stats.status_breakdown?.running || stats.running_runs || 0,
        'failed-runs': stats.status_breakdown?.failed || stats.failed_runs || 0,
        'recent-24h': stats.recent_24h || 0
    };
    
    Object.entries(elements).forEach(([id, newValue]) => {
        const element = document.getElementById(id);
        if (element) {
            // Check if element already has valid data (prevent overwriting)
            const currentValue = element.textContent;
            
            // Always update if:
            // 1. Current value is placeholder (-) or 0
            // 2. No lastUpdate timestamp (first time)
            // 3. It's been more than 30 seconds (allow more frequent updates)
            // 4. The new value is different from current value (data changed)
            const shouldUpdate = currentValue === '-' || currentValue === '0' || 
                               !element.dataset.lastUpdate || 
                               (Date.now() - parseInt(element.dataset.lastUpdate)) > 30000 || // 30 seconds instead of 5 minutes
                               currentValue !== newValue.toString(); // Update if value changed
            
            if (shouldUpdate) {
                element.textContent = newValue;
                element.dataset.lastUpdate = Date.now();
                console.log(`Updated ${id} with: ${newValue} (was: ${currentValue})`);
            } else {
                console.log(`Skipping ${id} update - no change needed: ${currentValue}`);
            }
        }
    });
}

/**
 * Update recent migrations table
 */
function updateRecentMigrationsTable(migrations) {
    const tbody = document.querySelector('#recent-migrations-table');
    if (!tbody) return;
    
    if (migrations.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="7" class="text-center text-muted">
                    <i class="bi bi-inbox"></i> No migrations found
                </td>
            </tr>
        `;
        return;
    }
    
    tbody.innerHTML = migrations.map(migration => `
        <tr>
            <td>
                <a href="/migrations/${migration.id}" class="text-decoration-none">
                    ${migration.id.substring(0, 8)}...
                </a>
            </td>
            <td>
                <span class="text-truncate d-inline-block" style="max-width: 200px;" title="${migration.file_path || 'N/A'}">
                    ${migration.file_path || 'N/A'}
                </span>
            </td>
            <td>
                <span class="badge bg-secondary">${migration.record_type || 'N/A'}</span>
            </td>
            <td>
                <span class="badge bg-${getStatusColor(migration.status)}">${migration.status || 'N/A'}</span>
            </td>
            <td>${formatDateTime(migration.created_at)}</td>
            <td>${formatDuration(migration.total_duration)}</td>
            <td>
                <a href="/migrations/${migration.id}" class="btn btn-sm btn-outline-primary">
                    <i class="bi bi-eye"></i> View
                </a>
            </td>
        </tr>
    `).join('');
}

/**
 * Export data to CSV
 */
function exportToCSV(data, filename) {
    if (!data || data.length === 0) {
        showAlert('No data to export', 'warning');
        return;
    }
    
    try {
        // Convert data to CSV format
        const headers = Object.keys(data[0]);
        const csvContent = [
            headers.join(','),
            ...data.map(row => headers.map(header => {
                const value = row[header];
                // Handle values that need quotes
                if (typeof value === 'string' && (value.includes(',') || value.includes('"'))) {
                    return `"${value.replace(/"/g, '""')}"`;
                }
                return value || '';
            }).join(','))
        ].join('\n');
        
        // Create download link
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        const url = URL.createObjectURL(blob);
        link.setAttribute('href', url);
        link.setAttribute('download', filename);
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        showAlert('Data exported successfully', 'success');
    } catch (error) {
        console.error('Export failed:', error);
        showAlert('Export failed. Please try again.', 'danger');
    }
}

/**
 * Refresh current page data
 */
function refreshPageData() {
    if (currentPage === 'dashboard') {
        loadDashboardData();
    } else if (currentPage === 'migrations') {
        // Reload the page to refresh filters
        window.location.reload();
    }
}

/**
 * Handle page visibility change
 */
document.addEventListener('visibilitychange', function() {
    if (document.hidden) {
        // Page is hidden, pause auto-refresh
        if (refreshInterval) {
            clearInterval(refreshInterval);
        }
    } else {
        // Page is visible, resume auto-refresh
        if (currentPage === 'dashboard') {
            setupAutoRefresh();
        }
    }
});

/**
 * Cleanup on page unload
 */
window.addEventListener('beforeunload', function() {
    if (refreshInterval) {
        clearInterval(refreshInterval);
    }
});

// Export functions for use in templates
window.MigrationDashboard = {
    formatDuration,
    formatDateTime,
    getStatusColor,
    showAlert,
    exportToCSV,
    refreshPageData
};
