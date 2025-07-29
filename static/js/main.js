// Sentiment Analyzer Main JavaScript

document.addEventListener('DOMContentLoaded', function() {
    console.log('Sentiment Analyzer loaded');
    
    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Auto-hide alerts after 5 seconds
    setTimeout(function() {
        var alerts = document.querySelectorAll('.alert');
        alerts.forEach(function(alert) {
            var bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        });
    }, 5000);
});

// Utility functions
function updateConfidenceBar(elementId, confidence) {
    const bar = document.getElementById(elementId);
    if (bar) {
        const fill = bar.querySelector('.confidence-fill');
        if (fill) {
            fill.style.width = (confidence * 100) + '%';
        }
    }
}

function formatConfidence(confidence) {
    return (confidence * 100).toFixed(1) + '%';
}
