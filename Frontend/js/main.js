/**
 * AeroSecure - Main JavaScript
 * Futuristic Airport Security System
 */

class AeroSecure {
    constructor() {
        this.currentPage = 'dashboard';
        this.isLoading = true;
        this.apiBaseUrl = 'http://localhost:5000'; // API server URL
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
        // Initialize person registration system
        this.initializePersonRegistration();
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

    initializePersonRegistration() {
        this.registrationData = {
            cameraStream: null,
            currentAngle: 'front',
            capturedImages: {},
            angles: ['front', 'left', 'right']
        };

        // Check camera permissions
        this.checkCameraPermissions();

        // Load available cameras
        this.loadAvailableCameras();

        // Event listeners
        this.setupRegistrationEventListeners();
    }

    async checkCameraPermissions() {
        try {
            if (navigator.permissions) {
                const permission = await navigator.permissions.query({ name: 'camera' });
                console.log('📷 Camera permission status:', permission.state);
                
                if (permission.state === 'denied') {
                    this.showNotification('Camera permission denied. Please enable camera access.', 'warning');
                } else if (permission.state === 'prompt') {
                    this.showNotification('Camera permission will be requested when you start the camera.', 'info');
                }
            }
        } catch (error) {
            console.log('Cannot check camera permissions:', error);
        }
    }

    setupRegistrationEventListeners() {
        // Refresh cameras button
        const refreshCamerasBtn = document.getElementById('refresh-cameras');
        if (refreshCamerasBtn) {
            refreshCamerasBtn.addEventListener('click', () => this.loadAvailableCameras());
        }

        // Add debug logging
        console.log('🔧 Person Registration System initialized');
        console.log('📡 API Base URL:', this.apiBaseUrl);
        
        // Check if getUserMedia is available
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            console.log('✅ Camera API available');
        } else {
            console.log('❌ Camera API not available');
            this.showNotification('Camera API not supported in this browser', 'warning');
        }

        // Test camera button
        const testCameraBtn = document.getElementById('test-camera');
        if (testCameraBtn) {
            testCameraBtn.addEventListener('click', () => this.testSelectedCamera());
        }

        // Start camera button
        const startCameraBtn = document.getElementById('start-camera');
        if (startCameraBtn) {
            startCameraBtn.addEventListener('click', () => {
                console.log('🎥 Start camera button clicked');
                this.startCameraStream();
            });
        }

        // Capture photo button
        const captureBtn = document.getElementById('capture-photo');
        if (captureBtn) {
            captureBtn.addEventListener('click', () => this.captureCurrentAngle());
        }

        // Retake photo button
        const retakeBtn = document.getElementById('retake-photo');
        if (retakeBtn) {
            retakeBtn.addEventListener('click', () => this.retakeCurrentAngle());
        }

        // Registration form
        const registrationForm = document.getElementById('person-registration-form');
        if (registrationForm) {
            registrationForm.addEventListener('submit', (e) => this.handleRegistrationSubmit(e));
        }

        // Cancel registration
        const cancelBtn = document.getElementById('cancel-registration');
        if (cancelBtn) {
            cancelBtn.addEventListener('click', () => this.cancelRegistration());
        }

        // Individual retake buttons
        const retakeButtons = document.querySelectorAll('.retake-angle-btn');
        retakeButtons.forEach(btn => {
            btn.addEventListener('click', (e) => {
                const angle = e.target.closest('.retake-angle-btn').dataset.angle;
                this.retakeSpecificAngle(angle);
            });
        });
    }

    async loadAvailableCameras() {
        try {
            // First try to get browser's media devices
            let browserCameras = [];
            
            if (navigator.mediaDevices && navigator.mediaDevices.enumerateDevices) {
                try {
                    // Request permissions first
                    await navigator.mediaDevices.getUserMedia({ video: true });
                    
                    const devices = await navigator.mediaDevices.enumerateDevices();
                    browserCameras = devices
                        .filter(device => device.kind === 'videoinput')
                        .map((device, index) => ({
                            id: device.deviceId,
                            name: device.label || `Camera ${index + 1}`,
                            status: 'available'
                        }));
                    
                    // Stop the temporary stream
                    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                    stream.getTracks().forEach(track => track.stop());
                    
                } catch (permissionError) {
                    console.log('Browser camera enumeration failed, falling back to server detection');
                }
            }
            
            // Fallback to server-side camera detection
            let serverCameras = [];
            try {
                const response = await fetch(`${this.apiBaseUrl}/api/cameras`);
                const data = await response.json();
                
                if (data.success) {
                    serverCameras = data.cameras;
                }
            } catch (serverError) {
                console.log('Server camera detection failed');
            }
            
            // Use browser cameras if available, otherwise use server cameras
            const availableCameras = browserCameras.length > 0 ? browserCameras : serverCameras;
            
            const cameraSelect = document.getElementById('camera-select');
            if (cameraSelect) {
                cameraSelect.innerHTML = '<option value="">Select a camera</option>';
                
                if (availableCameras.length > 0) {
                    availableCameras.forEach(camera => {
                        const option = document.createElement('option');
                        option.value = camera.id;
                        option.textContent = camera.name;
                        cameraSelect.appendChild(option);
                    });
                    
                    // Add a default option
                    const defaultOption = document.createElement('option');
                    defaultOption.value = '0';
                    defaultOption.textContent = 'Default Camera';
                    cameraSelect.appendChild(defaultOption);
                    
                    this.showNotification(`Found ${availableCameras.length} available cameras`, 'success');
                } else {
                    // Add a fallback option
                    const fallbackOption = document.createElement('option');
                    fallbackOption.value = '0';
                    fallbackOption.textContent = 'Default Webcam';
                    cameraSelect.appendChild(fallbackOption);
                    
                    this.showNotification('Using default camera', 'info');
                }
            }
        } catch (error) {
            console.error('Error loading cameras:', error);
            this.showNotification('Error detecting cameras. Using default camera.', 'warning');
            
            // Add fallback option
            const cameraSelect = document.getElementById('camera-select');
            if (cameraSelect) {
                cameraSelect.innerHTML = `
                    <option value="">Select a camera</option>
                    <option value="0">Default Webcam</option>
                `;
            }
        }
    }

    async testSelectedCamera() {
        const cameraSelect = document.getElementById('camera-select');
        const testResult = document.getElementById('camera-test-result');
        
        if (!cameraSelect.value) {
            this.showNotification('Please select a camera first', 'warning');
            return;
        }

        try {
            this.showNotification('Testing camera...', 'info');
            
            // For browser-detected cameras, do a client-side test
            if (cameraSelect.value !== '0' && cameraSelect.value.length > 10) {
                // This looks like a browser device ID
                try {
                    const constraints = {
                        video: {
                            deviceId: { exact: cameraSelect.value }
                        }
                    };
                    
                    const stream = await navigator.mediaDevices.getUserMedia(constraints);
                    
                    // Create a temporary video element to capture a frame
                    const video = document.createElement('video');
                    const canvas = document.createElement('canvas');
                    
                    video.srcObject = stream;
                    await video.play();
                    
                    // Wait a moment for the camera to initialize
                    await new Promise(resolve => setTimeout(resolve, 1000));
                    
                    // Capture a frame
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                    const ctx = canvas.getContext('2d');
                    ctx.drawImage(video, 0, 0);
                    
                    const imageData = canvas.toDataURL('image/jpeg', 0.8);
                    
                    // Clean up
                    stream.getTracks().forEach(track => track.stop());
                    
                    testResult.className = 'camera-test-result success';
                    testResult.innerHTML = `
                        <div><strong>✓ Browser Camera Test Successful</strong></div>
                        <div>${cameraSelect.options[cameraSelect.selectedIndex].text} is working properly</div>
                        <img src="${imageData}" style="max-width: 200px; margin-top: 10px; border-radius: 8px;">
                    `;
                    this.showNotification('Camera test successful', 'success');
                    return;
                    
                } catch (browserError) {
                    console.log('Browser camera test failed, trying server test:', browserError);
                }
            }
            
            // Fallback to server-side test
            const response = await fetch(`${this.apiBaseUrl}/api/capture-test/${cameraSelect.value}`);
            const data = await response.json();
            
            if (data.success) {
                testResult.className = 'camera-test-result success';
                testResult.innerHTML = `
                    <div><strong>✓ Server Camera Test Successful</strong></div>
                    <div>Camera ${cameraSelect.value} is working properly</div>
                    <img src="${data.image}" style="max-width: 200px; margin-top: 10px; border-radius: 8px;">
                `;
                this.showNotification('Camera test successful', 'success');
            } else {
                testResult.className = 'camera-test-result error';
                testResult.innerHTML = `
                    <div><strong>✗ Camera Test Failed</strong></div>
                    <div>${data.error}</div>
                `;
                this.showNotification('Camera test failed', 'error');
            }
        } catch (error) {
            console.error('Camera test error:', error);
            testResult.className = 'camera-test-result error';
            testResult.innerHTML = `
                <div><strong>✗ Camera Test Error</strong></div>
                <div>Unable to test camera: ${error.message}</div>
            `;
            this.showNotification('Camera test error', 'error');
        }
    }

    async startCameraStream() {
        const cameraSelect = document.getElementById('camera-select');
        
        if (!cameraSelect.value) {
            this.showNotification('Please select a camera first', 'warning');
            return;
        }

        try {
            this.showNotification('Starting camera...', 'info');
            
            const video = document.getElementById('camera-stream');
            const startBtn = document.getElementById('start-camera');
            
            // Stop any existing stream first
            if (this.registrationData.cameraStream) {
                this.stopCameraStream();
            }
            
            // Try different constraint approaches
            let constraints;
            const selectedCameraId = cameraSelect.value;
            
            // First try with exact device ID
            if (selectedCameraId !== '0') {
                constraints = {
                    video: {
                        deviceId: { exact: selectedCameraId }
                    }
                };
            } else {
                // For default camera, use basic constraints
                constraints = {
                    video: {
                        width: { ideal: 640 },
                        height: { ideal: 480 },
                        facingMode: 'user'
                    }
                };
            }

            console.log('Requesting camera with constraints:', constraints);
            
            // Request camera access
            this.registrationData.cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
            
            // Set video source
            video.srcObject = this.registrationData.cameraStream;
            
            // Wait for video to load
            await new Promise((resolve, reject) => {
                video.onloadedmetadata = () => {
                    video.play()
                        .then(resolve)
                        .catch(reject);
                };
                video.onerror = reject;
                
                // Timeout after 10 seconds
                setTimeout(() => reject(new Error('Video load timeout')), 10000);
            });
            
            // Enable capture button
            const captureBtn = document.getElementById('capture-photo');
            
            if (captureBtn) captureBtn.disabled = false;
            if (startBtn) {
                startBtn.innerHTML = '<i class="fas fa-stop"></i> Stop Camera';
                startBtn.onclick = () => this.stopCameraStream();
            }

            this.updateAngleInstruction();
            this.showNotification('Camera started successfully', 'success');
            
        } catch (error) {
            console.error('Error starting camera:', error);
            
            let errorMessage = 'Failed to start camera';
            
            if (error.name === 'NotFoundError') {
                errorMessage = 'Camera not found. Please check camera selection.';
            } else if (error.name === 'NotAllowedError') {
                errorMessage = 'Camera access denied. Please allow camera permissions.';
            } else if (error.name === 'NotReadableError') {
                errorMessage = 'Camera is being used by another application.';
            } else if (error.name === 'OverconstrainedError') {
                errorMessage = 'Camera constraints not supported. Trying fallback...';
                
                // Try with basic constraints as fallback
                try {
                    const basicConstraints = { 
                        video: true 
                    };
                    
                    console.log('Trying fallback constraints:', basicConstraints);
                    this.registrationData.cameraStream = await navigator.mediaDevices.getUserMedia(basicConstraints);
                    
                    const video = document.getElementById('camera-stream');
                    video.srcObject = this.registrationData.cameraStream;
                    
                    await video.play();
                    
                    const captureBtn = document.getElementById('capture-photo');
                    const startBtn = document.getElementById('start-camera');
                    
                    if (captureBtn) captureBtn.disabled = false;
                    if (startBtn) {
                        startBtn.innerHTML = '<i class="fas fa-stop"></i> Stop Camera';
                        startBtn.onclick = () => this.stopCameraStream();
                    }

                    this.updateAngleInstruction();
                    this.showNotification('Camera started with basic settings', 'success');
                    return;
                    
                } catch (fallbackError) {
                    console.error('Fallback camera start failed:', fallbackError);
                    errorMessage = 'Camera initialization failed completely.';
                }
            }
            
            this.showNotification(errorMessage, 'error');
            
            // Reset button state
            const startBtn = document.getElementById('start-camera');
            if (startBtn) {
                startBtn.innerHTML = '<i class="fas fa-video"></i> Start Camera';
                startBtn.onclick = () => this.startCameraStream();
            }
        }
    }

    stopCameraStream() {
        if (this.registrationData.cameraStream) {
            this.registrationData.cameraStream.getTracks().forEach(track => track.stop());
            this.registrationData.cameraStream = null;
        }

        const video = document.getElementById('camera-stream');
        const startBtn = document.getElementById('start-camera');
        const captureBtn = document.getElementById('capture-photo');

        if (video) video.srcObject = null;
        if (captureBtn) captureBtn.disabled = true;
        if (startBtn) {
            startBtn.innerHTML = '<i class="fas fa-video"></i> Start Camera';
            startBtn.onclick = () => this.startCameraStream();
        }

        this.showNotification('Camera stopped', 'info');
    }

    updateAngleInstruction() {
        const instruction = document.getElementById('angle-instruction');
        const currentAngleSpan = document.getElementById('current-angle');
        
        const angleMap = {
            'front': 'FRONT view - Look straight at the camera',
            'left': 'LEFT view - Turn your head to the left',
            'right': 'RIGHT view - Turn your head to the right'
        };

        if (instruction) {
            instruction.textContent = `Position your face for ${angleMap[this.registrationData.currentAngle]}`;
        }
        
        if (currentAngleSpan) {
            currentAngleSpan.textContent = this.registrationData.currentAngle.charAt(0).toUpperCase() + 
                                          this.registrationData.currentAngle.slice(1);
        }
    }

    captureCurrentAngle() {
        const video = document.getElementById('camera-stream');
        const canvas = document.getElementById('capture-canvas');
        const angle = this.registrationData.currentAngle;

        if (!video || !canvas) return;

        // Set up canvas
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        
        // Draw current frame
        ctx.drawImage(video, 0, 0);
        
        // Convert to base64
        const imageData = canvas.toDataURL('image/jpeg', 0.9);
        
        // Store captured image
        this.registrationData.capturedImages[angle] = imageData;
        
        // Update UI
        this.updateCapturedImageDisplay(angle, imageData);
        
        // Move to next angle or enable registration
        this.moveToNextAngle();
        
        this.showNotification(`${angle.charAt(0).toUpperCase() + angle.slice(1)} image captured`, 'success');
    }

    updateCapturedImageDisplay(angle, imageData) {
        const imageSlot = document.querySelector(`.image-slot[data-angle="${angle}"]`);
        const statusIndicator = document.getElementById(`${angle}-status`);
        const imageElement = document.getElementById(`${angle}-image`);
        const retakeBtn = document.getElementById(`retake-${angle}`);

        if (imageSlot) {
            imageSlot.classList.add('captured');
        }

        if (statusIndicator) {
            statusIndicator.textContent = 'Captured';
            statusIndicator.className = 'status-indicator captured';
        }

        if (imageElement) {
            imageElement.src = imageData;
            imageElement.style.display = 'block';
            imageElement.parentElement.querySelector('.image-placeholder').style.display = 'none';
        }

        if (retakeBtn) {
            retakeBtn.style.display = 'block';
        }
    }

    moveToNextAngle() {
        const currentIndex = this.registrationData.angles.indexOf(this.registrationData.currentAngle);
        
        if (currentIndex < this.registrationData.angles.length - 1) {
            // Move to next angle
            this.registrationData.currentAngle = this.registrationData.angles[currentIndex + 1];
            this.updateAngleInstruction();
            
            // Enable retake button
            const retakeBtn = document.getElementById('retake-photo');
            if (retakeBtn) retakeBtn.disabled = false;
        } else {
            // All angles captured
            this.showNotification('All angles captured! Ready to register person.', 'success');
            
            // Enable registration button
            const registerBtn = document.getElementById('register-person');
            if (registerBtn) registerBtn.disabled = false;
            
            // Stop camera
            this.stopCameraStream();
        }
    }

    retakeCurrentAngle() {
        const angle = this.registrationData.currentAngle;
        
        // Remove captured image
        delete this.registrationData.capturedImages[angle];
        
        // Reset UI for this angle
        this.resetAngleUI(angle);
        
        // Restart camera if needed
        if (!this.registrationData.cameraStream) {
            this.startCameraStream();
        }
        
        // Disable register button if not all angles captured
        const registerBtn = document.getElementById('register-person');
        if (registerBtn && Object.keys(this.registrationData.capturedImages).length < 3) {
            registerBtn.disabled = true;
        }

        this.showNotification(`${angle.charAt(0).toUpperCase() + angle.slice(1)} image removed - ready to retake`, 'info');
    }

    resetAngleUI(angle) {
        const imageSlot = document.querySelector(`.image-slot[data-angle="${angle}"]`);
        const statusIndicator = document.getElementById(`${angle}-status`);
        const imageElement = document.getElementById(`${angle}-image`);
        const retakeBtn = document.getElementById(`retake-${angle}`);

        if (imageSlot) imageSlot.classList.remove('captured');
        
        if (statusIndicator) {
            statusIndicator.textContent = 'Pending';
            statusIndicator.className = 'status-indicator pending';
        }

        if (imageElement) {
            imageElement.style.display = 'none';
            imageElement.parentElement.querySelector('.image-placeholder').style.display = 'flex';
        }

        if (retakeBtn) {
            retakeBtn.style.display = 'none';
        }
    }

    retakeSpecificAngle(angle) {
        // Switch to the specified angle
        this.registrationData.currentAngle = angle;
        
        // Remove the captured image for this angle
        delete this.registrationData.capturedImages[angle];
        
        // Reset UI for this angle
        this.resetAngleUI(angle);
        
        // Update instruction and current angle display
        this.updateAngleInstruction();
        
        // Start camera if not already running
        if (!this.registrationData.cameraStream) {
            this.startCameraStream();
        }
        
        // Enable capture and retake buttons
        const captureBtn = document.getElementById('capture-photo');
        const retakeBtn = document.getElementById('retake-photo');
        
        if (captureBtn) captureBtn.disabled = false;
        if (retakeBtn) retakeBtn.disabled = false;
        
        // Disable register button
        const registerBtn = document.getElementById('register-person');
        if (registerBtn) registerBtn.disabled = true;

        this.showNotification(`Ready to retake ${angle} image`, 'info');
    }

    async handleRegistrationSubmit(e) {
        e.preventDefault();
        
        const personName = document.getElementById('person-name').value.trim();
        
        if (!personName) {
            this.showNotification('Please enter person name', 'warning');
            return;
        }

        if (Object.keys(this.registrationData.capturedImages).length !== 3) {
            this.showNotification('Please capture all three angle images', 'warning');
            return;
        }

        // Show progress
        this.showRegistrationProgress(true);
        
        try {
            const registrationData = {
                name: personName,
                images: this.registrationData.capturedImages
            };

            this.updateProgress(20, 'Sending registration data...');
            
            const response = await fetch(`${this.apiBaseUrl}/api/register-person`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(registrationData)
            });

            this.updateProgress(80, 'Processing registration...');
            
            const result = await response.json();
            
            if (result.success) {
                this.updateProgress(100, 'Registration completed successfully!');
                
                setTimeout(() => {
                    this.showNotification(`Successfully registered ${result.person_name} with ID ${result.person_id}`, 'success');
                    this.resetRegistrationForm();
                    this.showRegistrationProgress(false);
                }, 2000);
            } else {
                throw new Error(result.error || 'Registration failed');
            }
            
        } catch (error) {
            console.error('Registration error:', error);
            this.showNotification(`Registration failed: ${error.message}`, 'error');
            this.showRegistrationProgress(false);
        }
    }

    showRegistrationProgress(show) {
        const progressElement = document.getElementById('registration-progress');
        if (progressElement) {
            progressElement.style.display = show ? 'block' : 'none';
        }
    }

    updateProgress(percentage, text) {
        const progressFill = document.getElementById('progress-fill');
        const progressText = document.getElementById('progress-text');
        
        if (progressFill) {
            progressFill.style.width = `${percentage}%`;
        }
        
        if (progressText) {
            progressText.textContent = text;
        }
    }

    resetRegistrationForm() {
        // Reset form
        const form = document.getElementById('person-registration-form');
        if (form) form.reset();
        
        // Reset registration data
        this.registrationData = {
            cameraStream: null,
            currentAngle: 'front',
            capturedImages: {},
            angles: ['front', 'left', 'right']
        };
        
        // Reset UI
        this.registrationData.angles.forEach(angle => {
            const imageSlot = document.querySelector(`.image-slot[data-angle="${angle}"]`);
            const statusIndicator = document.getElementById(`${angle}-status`);
            const imageElement = document.getElementById(`${angle}-image`);
            const retakeBtn = document.getElementById(`retake-${angle}`);

            if (imageSlot) imageSlot.classList.remove('captured');
            
            if (statusIndicator) {
                statusIndicator.textContent = 'Pending';
                statusIndicator.className = 'status-indicator pending';
            }

            if (imageElement) {
                imageElement.style.display = 'none';
                imageElement.parentElement.querySelector('.image-placeholder').style.display = 'flex';
            }

            if (retakeBtn) {
                retakeBtn.style.display = 'none';
            }
        });
        
        // Disable buttons
        const registerBtn = document.getElementById('register-person');
        const retakeBtn = document.getElementById('retake-photo');
        const captureBtn = document.getElementById('capture-photo');
        
        if (registerBtn) registerBtn.disabled = true;
        if (retakeBtn) retakeBtn.disabled = true;
        if (captureBtn) captureBtn.disabled = true;
        
        this.updateAngleInstruction();
    }

    cancelRegistration() {
        if (confirm('Are you sure you want to cancel? All captured images will be lost.')) {
            this.stopCameraStream();
            this.resetRegistrationForm();
            this.showNotification('Registration cancelled', 'info');
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

// Initialize the application when sections are loaded
function initializeAeroSecure() {
    if (window.sectionsLoaded || !window.sectionLoader) {
        window.aeroSecure = new AeroSecure();
    } else {
        // Wait for sections to load
        document.addEventListener('sectionsLoaded', () => {
            window.aeroSecure = new AeroSecure();
        });
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    initializeAeroSecure();
});

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AeroSecure;
}