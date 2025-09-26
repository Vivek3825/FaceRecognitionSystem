/**
 * AeroSecure - Main JavaScript
 * Futuristic Airport Security System
 */

class AeroSecure {
    constructor() {
        this.currentPage = 'dashboard';
        this.isLoading = true;
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.startLoadingSequence();
        this.updateDateTime();
        this.initializeSystem();
    }

    setupEventListeners() {
        // Navigation
        document.querySelectorAll('.nav-item').forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                const page = e.currentTarget.dataset.page;
                this.showPage(page);
            });
        });

        // Photo upload
        const photoUpload = document.getElementById('photo-upload');
        const photoInput = document.getElementById('photo-input');
        
        if (photoUpload && photoInput) {
            photoUpload.addEventListener('click', () => photoInput.click());
            photoUpload.addEventListener('dragover', this.handleDragOver);
            photoUpload.addEventListener('drop', this.handleDrop);
            photoInput.addEventListener('change', this.handleFileSelect);
        }

        // Form submissions
        document.querySelectorAll('form').forEach(form => {
            form.addEventListener('submit', this.handleFormSubmit);
        });

        // Camera controls
        document.querySelectorAll('.control-btn').forEach(btn => {
            btn.addEventListener('click', this.handleCameraControl);
        });

        // Camera list items
        document.querySelectorAll('.camera-item').forEach(item => {
            item.addEventListener('click', this.handleCameraSelect);
        });

        // Quick action buttons
        document.querySelectorAll('.action-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const page = e.currentTarget.onclick || 'dashboard';
                if (typeof page === 'string') {
                    this.showPage(page);
                }
            });
        });

        // Window resize
        window.addEventListener('resize', this.handleResize.bind(this));
    }

    startLoadingSequence() {
        const loadingScreen = document.getElementById('loading-screen');
        const mainContainer = document.getElementById('main-container');
        
        // Simulate system initialization
        setTimeout(() => {
            loadingScreen.classList.add('hidden');
            mainContainer.classList.remove('hidden');
            this.isLoading = false;
            this.playStartupSound();
        }, 3500);
    }

    playStartupSound() {
        // In a real implementation, you would play a futuristic startup sound
        console.log('🔊 System initialized - AeroSecure online');
    }

    updateDateTime() {
        const updateTime = () => {
            const now = new Date();
            const timeElement = document.getElementById('current-time');
            const dateElement = document.getElementById('current-date');
            
            if (timeElement) {
                timeElement.textContent = now.toLocaleTimeString('en-US', { 
                    hour12: false,
                    hour: '2-digit',
                    minute: '2-digit',
                    second: '2-digit'
                });
            }
            
            if (dateElement) {
                dateElement.textContent = now.toLocaleDateString('en-US', {
                    weekday: 'long',
                    year: 'numeric',
                    month: 'long',
                    day: 'numeric'
                });
            }
        };

        updateTime();
        setInterval(updateTime, 1000);
    }

    showPage(pageId) {
        // Hide all pages
        document.querySelectorAll('.page').forEach(page => {
            page.classList.remove('active');
        });

        // Show selected page
        const targetPage = document.getElementById(pageId);
        if (targetPage) {
            targetPage.classList.add('active');
            this.currentPage = pageId;
            
            // Update navigation
            document.querySelectorAll('.nav-item').forEach(item => {
                item.classList.remove('active');
            });
            
            const activeNavItem = document.querySelector(`[data-page="${pageId}"]`);
            if (activeNavItem) {
                activeNavItem.classList.add('active');
            }

            // Initialize page-specific features
            this.initializePage(pageId);
            
            // Play transition sound
            this.playTransitionSound();
        }
    }

    initializePage(pageId) {
        switch (pageId) {
            case 'camera-monitor':
                this.initializeCameraMonitor();
                break;
            case 'person-search':
                this.initializePersonSearch();
                break;
            case 'add-person':
                this.initializeAddPerson();
                break;
            case 'dashboard':
                this.initializeDashboard();
                break;
        }
    }

    initializeDashboard() {
        // Animate stats counters
        this.animateCounters();
        
        // Update system health metrics
        this.updateSystemHealth();
        
        // Refresh alerts
        this.refreshAlerts();
    }

    initializeCameraMonitor() {
        // Start camera feeds simulation
        this.startCameraFeeds();
        
        // Initialize detection overlays
        this.initializeDetectionOverlays();
    }

    initializePersonSearch() {
        // Setup search functionality
        this.setupSearchFeatures();
    }

    initializeAddPerson() {
        // Initialize camera capture
        this.initializeCameraCapture();
    }

    animateCounters() {
        document.querySelectorAll('.stat-number').forEach(counter => {
            const target = parseInt(counter.textContent.replace(/,/g, ''));
            let current = 0;
            const increment = target / 50;
            
            const updateCounter = () => {
                if (current < target) {
                    current += increment;
                    counter.textContent = Math.floor(current).toLocaleString();
                    requestAnimationFrame(updateCounter);
                } else {
                    counter.textContent = target.toLocaleString();
                }
            };
            
            updateCounter();
        });
    }

    updateSystemHealth() {
        const metrics = [
            { selector: '.metric-fill', values: [65, 42, 78, 95] }
        ];

        document.querySelectorAll('.metric').forEach((metric, index) => {
            const fill = metric.querySelector('.metric-fill');
            const value = metric.querySelector('.metric-value');
            
            if (fill && value) {
                const targetWidth = [65, 42, 78, 95][index] || 50;
                fill.style.width = `${targetWidth}%`;
                value.textContent = `${targetWidth}%`;
            }
        });
    }

    refreshAlerts() {
        // Simulate real-time alert updates
        const alerts = document.querySelectorAll('.alert-item');
        alerts.forEach((alert, index) => {
            setTimeout(() => {
                alert.style.animation = 'fadeInLeft 0.5s ease forwards';
            }, index * 200);
        });
    }

    startCameraFeeds() {
        // Simulate camera feed activity
        document.querySelectorAll('.camera-feed').forEach((feed, index) => {
            setTimeout(() => {
                this.simulateCameraActivity(feed);
            }, index * 500);
        });
    }

    simulateCameraActivity(feedElement) {
        const scanLine = feedElement.querySelector('.scan-line');
        const detectionBox = feedElement.querySelector('.detection-box');
        
        if (scanLine) {
            scanLine.style.animation = 'scanLine 3s linear infinite';
        }
        
        // Randomly show/hide detection boxes
        if (detectionBox && Math.random() > 0.5) {
            setTimeout(() => {
                detectionBox.style.display = 'block';
                detectionBox.style.animation = 'fadeInUp 0.5s ease forwards';
            }, Math.random() * 3000);
        }
    }

    initializeDetectionOverlays() {
        document.querySelectorAll('.scanning-overlay').forEach(overlay => {
            overlay.style.opacity = '1';
        });
    }

    setupSearchFeatures() {
        const searchBtn = document.querySelector('.search-btn');
        if (searchBtn) {
            searchBtn.addEventListener('click', (e) => {
                e.preventDefault();
                this.performSearch();
            });
        }
    }

    performSearch() {
        // Simulate search process
        const results = document.querySelector('.search-results');
        if (results) {
            results.style.opacity = '0.5';
            
            setTimeout(() => {
                results.style.opacity = '1';
                this.animateSearchResults();
                this.showNotification('Search completed - 3 matches found', 'success');
            }, 1500);
        }
    }

    animateSearchResults() {
        document.querySelectorAll('.result-item').forEach((item, index) => {
            item.style.opacity = '0';
            item.style.transform = 'translateY(20px)';
            
            setTimeout(() => {
                item.style.transition = 'all 0.5s ease';
                item.style.opacity = '1';
                item.style.transform = 'translateY(0)';
            }, index * 200);
        });
    }

    initializeCameraCapture() {
        const startCameraBtn = document.querySelector('.capture-controls .btn-secondary');
        const captureBtn = document.querySelector('.capture-controls .btn-primary');
        
        if (startCameraBtn) {
            startCameraBtn.addEventListener('click', this.startCamera.bind(this));
        }
        
        if (captureBtn) {
            captureBtn.addEventListener('click', this.capturePhoto.bind(this));
        }
    }

    async startCamera() {
        try {
            // In a real implementation, you would access the user's camera
            const cameraView = document.querySelector('.camera-view');
            if (cameraView) {
                cameraView.innerHTML = '<div class="camera-active">📹 Camera Active</div>';
                cameraView.style.background = 'linear-gradient(45deg, #0a0a0f, #1a1a2e)';
                
                this.showNotification('Camera activated successfully', 'success');
            }
        } catch (error) {
            console.error('Camera access failed:', error);
            this.showNotification('Camera access failed', 'error');
        }
    }

    capturePhoto() {
        const cameraView = document.querySelector('.camera-view');
        if (cameraView) {
            // Simulate photo capture
            cameraView.style.filter = 'brightness(2)';
            
            setTimeout(() => {
                cameraView.style.filter = 'brightness(1)';
                this.showNotification('Photo captured successfully', 'success');
            }, 200);
        }
    }

    initializeSystem() {
        // Simulate system checks
        setTimeout(() => this.systemCheck('cameras'), 500);
        setTimeout(() => this.systemCheck('database'), 1000);
        setTimeout(() => this.systemCheck('network'), 1500);
        setTimeout(() => this.systemCheck('security'), 2000);
    }

    systemCheck(component) {
        console.log(`✅ ${component.toUpperCase()} system online`);
    }

    handleDragOver(e) {
        e.preventDefault();
        e.currentTarget.style.borderColor = 'var(--primary-color)';
        e.currentTarget.style.background = 'rgba(0, 212, 255, 0.1)';
    }

    handleDrop(e) {
        e.preventDefault();
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.processUploadedFile(files[0]);
        }
        
        e.currentTarget.style.borderColor = 'var(--border-color)';
        e.currentTarget.style.background = 'rgba(255, 255, 255, 0.02)';
    }

    handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            this.processUploadedFile(file);
        }
    }

    processUploadedFile(file) {
        if (file && file.type.startsWith('image/')) {
            // Simulate file processing
            this.showNotification('Processing image...', 'info');
            
            setTimeout(() => {
                this.showNotification('Image processed successfully', 'success');
                // In a real implementation, you would upload to backend
            }, 2000);
        } else {
            this.showNotification('Please select a valid image file', 'error');
        }
    }

    handleFormSubmit(e) {
        e.preventDefault();
        const form = e.target;
        const formData = new FormData(form);
        
        // Simulate form submission
        this.showNotification('Processing request...', 'info');
        
        setTimeout(() => {
            this.showNotification('Person added successfully', 'success');
            form.reset();
        }, 2000);
    }

    handleCameraControl(e) {
        const control = e.currentTarget;
        const icon = control.querySelector('i');
        
        // Visual feedback
        control.style.transform = 'scale(0.95)';
        setTimeout(() => {
            control.style.transform = 'scale(1)';
        }, 150);
        
        // Determine action based on icon
        if (icon.classList.contains('fa-search-plus')) {
            this.showNotification('Zooming in...', 'info');
        } else if (icon.classList.contains('fa-search-minus')) {
            this.showNotification('Zooming out...', 'info');
        } else if (icon.classList.contains('fa-record-vinyl')) {
            this.showNotification('Recording started', 'success');
        }
    }

    handleCameraSelect(e) {
        // Remove active class from all camera items
        document.querySelectorAll('.camera-item').forEach(item => {
            item.classList.remove('active');
        });
        
        // Add active class to selected item
        e.currentTarget.classList.add('active');
        
        // Update main camera feed
        const cameraName = e.currentTarget.querySelector('.camera-name').textContent;
        this.showNotification(`Switched to ${cameraName}`, 'info');
    }

    handleResize() {
        // Handle responsive behavior
        const sidebar = document.querySelector('.sidebar');
        const mainContent = document.querySelector('.main-content');
        
        if (window.innerWidth < 768) {
            // Mobile layout adjustments
            if (sidebar) {
                sidebar.style.display = 'none';
            }
        } else {
            // Desktop layout
            if (sidebar) {
                sidebar.style.display = 'block';
            }
        }
    }

    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <i class="fas fa-${this.getNotificationIcon(type)}"></i>
            <span>${message}</span>
        `;
        
        // Style the notification
        Object.assign(notification.style, {
            position: 'fixed',
            top: '20px',
            right: '20px',
            padding: '15px 20px',
            borderRadius: '8px',
            color: 'white',
            fontWeight: '600',
            fontSize: '14px',
            zIndex: '10000',
            display: 'flex',
            alignItems: 'center',
            gap: '10px',
            minWidth: '300px',
            boxShadow: '0 4px 20px rgba(0, 0, 0, 0.3)',
            backdropFilter: 'blur(20px)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            animation: 'slideInRight 0.5s ease forwards'
        });
        
        // Set background based on type
        switch (type) {
            case 'success':
                notification.style.background = 'rgba(0, 255, 136, 0.2)';
                notification.style.borderColor = 'rgba(0, 255, 136, 0.5)';
                break;
            case 'error':
                notification.style.background = 'rgba(255, 51, 102, 0.2)';
                notification.style.borderColor = 'rgba(255, 51, 102, 0.5)';
                break;
            case 'warning':
                notification.style.background = 'rgba(255, 170, 0, 0.2)';
                notification.style.borderColor = 'rgba(255, 170, 0, 0.5)';
                break;
            default:
                notification.style.background = 'rgba(0, 212, 255, 0.2)';
                notification.style.borderColor = 'rgba(0, 212, 255, 0.5)';
        }
        
        // Add to DOM
        document.body.appendChild(notification);
        
        // Remove after 3 seconds
        setTimeout(() => {
            notification.style.animation = 'slideOutRight 0.5s ease forwards';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 500);
        }, 3000);
    }

    getNotificationIcon(type) {
        switch (type) {
            case 'success': return 'check-circle';
            case 'error': return 'exclamation-triangle';
            case 'warning': return 'exclamation-circle';
            default: return 'info-circle';
        }
    }

    playTransitionSound() {
        // In a real implementation, you would play a subtle UI sound
        console.log('🔊 Page transition');
    }

    // Public API methods
    switchToPage(pageId) {
        this.showPage(pageId);
    }

    refreshData() {
        this.animateCounters();
        this.updateSystemHealth();
        this.refreshAlerts();
        this.showNotification('Data refreshed', 'success');
    }

    toggleFullscreen() {
        if (!document.fullscreenElement) {
            document.documentElement.requestFullscreen();
        } else {
            document.exitFullscreen();
        }
    }
}

// Animation keyframes for notifications
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOutRight {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// Global functions for HTML onclick handlers
function showPage(pageId) {
    if (window.aeroSecure) {
        window.aeroSecure.switchToPage(pageId);
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.aeroSecure = new AeroSecure();
});

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AeroSecure;
}