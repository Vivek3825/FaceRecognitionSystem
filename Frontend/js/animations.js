/**
 * AeroSecure - Advanced Animations Controller
 * Handles complex animations and visual effects
 */

class AnimationController {
    constructor() {
        this.isAnimating = false;
        this.animationQueue = [];
        this.observers = new Map();
        this.init();
    }

    init() {
        this.setupIntersectionObservers();
        this.setupParticleSystem();
        this.setupMatrixEffect();
        this.setupHolographicEffects();
    }

    setupIntersectionObservers() {
        // Observe elements for scroll-triggered animations
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    this.triggerElementAnimation(entry.target);
                }
            });
        }, observerOptions);

        // Observe cards and animated elements
        document.querySelectorAll('.card, .stat-item, .alert-item').forEach(el => {
            observer.observe(el);
        });

        this.observers.set('scroll', observer);
    }

    triggerElementAnimation(element) {
        element.style.animation = 'fadeInUp 0.6s ease forwards';
        element.style.animationDelay = `${Math.random() * 0.3}s`;
    }

    setupParticleSystem() {
        this.createParticleBackground();
        this.animateParticles();
    }

    createParticleBackground() {
        const particleContainer = document.createElement('div');
        particleContainer.className = 'particle-container';
        particleContainer.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: -1;
            overflow: hidden;
        `;

        // Create floating particles
        for (let i = 0; i < 50; i++) {
            const particle = document.createElement('div');
            particle.className = 'particle';
            particle.style.cssText = `
                position: absolute;
                width: 2px;
                height: 2px;
                background: rgba(0, 212, 255, ${Math.random() * 0.5 + 0.2});
                border-radius: 50%;
                animation: floatParticle ${5 + Math.random() * 10}s linear infinite;
                animation-delay: ${Math.random() * 5}s;
            `;
            
            particle.style.left = Math.random() * 100 + '%';
            particle.style.animationDuration = (8 + Math.random() * 12) + 's';
            
            particleContainer.appendChild(particle);
        }

        document.body.appendChild(particleContainer);
    }

    animateParticles() {
        const style = document.createElement('style');
        style.textContent = `
            @keyframes floatParticle {
                0% {
                    transform: translateY(100vh) rotate(0deg);
                    opacity: 0;
                }
                10% {
                    opacity: 1;
                }
                90% {
                    opacity: 1;
                }
                100% {
                    transform: translateY(-100px) rotate(360deg);
                    opacity: 0;
                }
            }
        `;
        document.head.appendChild(style);
    }

    setupMatrixEffect() {
        // Create digital rain effect for loading screen
        const matrixContainer = document.createElement('div');
        matrixContainer.className = 'matrix-rain';
        matrixContainer.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: -2;
            opacity: 0.1;
        `;

        // Create rain columns
        for (let i = 0; i < 20; i++) {
            const column = document.createElement('div');
            column.className = 'matrix-column';
            column.style.cssText = `
                position: absolute;
                top: 0;
                width: 20px;
                height: 100%;
                left: ${i * 5}%;
                font-family: 'Courier New', monospace;
                font-size: 14px;
                color: var(--primary-color);
                animation: matrixRain ${3 + Math.random() * 2}s linear infinite;
                animation-delay: ${Math.random() * 2}s;
            `;
            
            // Add random characters
            const chars = '01アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン';
            for (let j = 0; j < 20; j++) {
                const char = document.createElement('div');
                char.textContent = chars[Math.floor(Math.random() * chars.length)];
                char.style.cssText = `
                    opacity: ${Math.random()};
                    margin-bottom: 5px;
                `;
                column.appendChild(char);
            }
            
            matrixContainer.appendChild(column);
        }

        document.body.appendChild(matrixContainer);
    }

    setupHolographicEffects() {
        // Add holographic shimmer to cards
        document.querySelectorAll('.card').forEach(card => {
            card.addEventListener('mouseenter', () => {
                this.addHolographicShimmer(card);
            });
        });
    }

    addHolographicShimmer(element) {
        const shimmer = document.createElement('div');
        shimmer.className = 'holographic-shimmer';
        shimmer.style.cssText = `
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(
                90deg,
                transparent,
                rgba(0, 212, 255, 0.2),
                rgba(255, 255, 255, 0.3),
                rgba(0, 212, 255, 0.2),
                transparent
            );
            animation: shimmerPass 1.5s ease-out;
            z-index: 10;
            pointer-events: none;
        `;

        element.style.position = 'relative';
        element.style.overflow = 'hidden';
        element.appendChild(shimmer);

        setTimeout(() => {
            if (shimmer.parentNode) {
                shimmer.parentNode.removeChild(shimmer);
            }
        }, 1500);
    }

    // Radar sweep animation for camera detection
    createRadarSweep(container) {
        const radar = document.createElement('div');
        radar.className = 'radar-sweep';
        radar.style.cssText = `
            position: absolute;
            top: 50%;
            left: 50%;
            width: 100px;
            height: 100px;
            transform: translate(-50%, -50%);
            border: 2px solid var(--primary-color);
            border-radius: 50%;
            opacity: 0.6;
        `;

        const sweepLine = document.createElement('div');
        sweepLine.style.cssText = `
            position: absolute;
            top: 50%;
            left: 50%;
            width: 50%;
            height: 2px;
            background: linear-gradient(90deg, transparent, var(--primary-color));
            transform-origin: left center;
            animation: radarSweep 2s linear infinite;
        `;

        radar.appendChild(sweepLine);
        container.appendChild(radar);

        return radar;
    }

    // Glitch effect for security alerts
    triggerGlitchEffect(element, duration = 1000) {
        element.style.animation = `glitch 0.3s infinite`;
        
        setTimeout(() => {
            element.style.animation = '';
        }, duration);
    }

    // Typing animation for text
    typeWriter(element, text, speed = 50) {
        element.textContent = '';
        let i = 0;
        
        const timer = setInterval(() => {
            if (i < text.length) {
                element.textContent += text.charAt(i);
                i++;
            } else {
                clearInterval(timer);
            }
        }, speed);
    }

    // Loading bar animation
    animateLoadingBar(element, targetPercent, duration = 2000) {
        let currentPercent = 0;
        const increment = targetPercent / (duration / 16);
        
        const animate = () => {
            if (currentPercent < targetPercent) {
                currentPercent += increment;
                element.style.width = Math.min(currentPercent, targetPercent) + '%';
                requestAnimationFrame(animate);
            }
        };
        
        animate();
    }

    // Face detection box animation
    animateFaceDetection(container) {
        const detectionBox = document.createElement('div');
        detectionBox.className = 'face-detection-box';
        detectionBox.style.cssText = `
            position: absolute;
            border: 2px solid var(--success-color);
            border-radius: 4px;
            animation: faceDetectionPulse 2s infinite;
            pointer-events: none;
        `;

        // Random position and size
        const size = 80 + Math.random() * 40;
        const x = Math.random() * (container.clientWidth - size);
        const y = Math.random() * (container.clientHeight - size);

        detectionBox.style.left = x + 'px';
        detectionBox.style.top = y + 'px';
        detectionBox.style.width = size + 'px';
        detectionBox.style.height = size + 'px';

        container.appendChild(detectionBox);

        // Remove after animation
        setTimeout(() => {
            if (detectionBox.parentNode) {
                detectionBox.parentNode.removeChild(detectionBox);
            }
        }, 3000);
    }

    // Data stream animation for processing
    createDataStream(container) {
        const stream = document.createElement('div');
        stream.className = 'data-stream';
        stream.style.cssText = `
            position: absolute;
            right: 10px;
            top: 0;
            width: 2px;
            height: 100%;
            background: linear-gradient(
                to bottom,
                transparent,
                var(--primary-color),
                transparent
            );
            animation: dataStream 1s ease-in-out infinite;
        `;

        container.style.position = 'relative';
        container.appendChild(stream);

        return stream;
    }

    // Security scan animation
    performSecurityScan(element) {
        const scanOverlay = document.createElement('div');
        scanOverlay.className = 'security-scan-overlay';
        scanOverlay.style.cssText = `
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: repeating-linear-gradient(
                90deg,
                transparent,
                transparent 4px,
                rgba(0, 212, 255, 0.1) 4px,
                rgba(0, 212, 255, 0.1) 8px
            );
            animation: securityScan 2s ease-in-out;
            pointer-events: none;
        `;

        element.style.position = 'relative';
        element.appendChild(scanOverlay);

        setTimeout(() => {
            if (scanOverlay.parentNode) {
                scanOverlay.parentNode.removeChild(scanOverlay);
            }
        }, 2000);
    }

    // Circuit board animation for system status
    createCircuitAnimation(container) {
        const circuit = document.createElement('div');
        circuit.className = 'circuit-board';
        circuit.innerHTML = `
            <svg width="100%" height="100%" style="position: absolute; top: 0; left: 0;">
                <defs>
                    <pattern id="circuit" x="0" y="0" width="40" height="40" patternUnits="userSpaceOnUse">
                        <path d="M0,20 L40,20 M20,0 L20,40" stroke="rgba(0, 212, 255, 0.2)" stroke-width="1"/>
                        <circle cx="20" cy="20" r="2" fill="rgba(0, 212, 255, 0.5)"/>
                        <circle cx="0" cy="20" r="1" fill="rgba(0, 212, 255, 0.3)"/>
                        <circle cx="40" cy="20" r="1" fill="rgba(0, 212, 255, 0.3)"/>
                        <circle cx="20" cy="0" r="1" fill="rgba(0, 212, 255, 0.3)"/>
                        <circle cx="20" cy="40" r="1" fill="rgba(0, 212, 255, 0.3)"/>
                    </pattern>
                </defs>
                <rect width="100%" height="100%" fill="url(#circuit)" opacity="0.3"/>
            </svg>
        `;

        container.style.position = 'relative';
        container.appendChild(circuit);

        // Animate circuit nodes
        const nodes = circuit.querySelectorAll('circle');
        nodes.forEach((node, index) => {
            setTimeout(() => {
                node.style.animation = 'circuitPulse 2s infinite';
            }, index * 200);
        });
    }

    // Status indicator animations
    animateStatusIndicator(element, status) {
        element.classList.remove('online', 'offline', 'warning');
        element.classList.add(status);

        switch (status) {
            case 'online':
                element.style.animation = 'pulse 2s infinite';
                element.style.backgroundColor = 'var(--success-color)';
                break;
            case 'offline':
                element.style.animation = 'none';
                element.style.backgroundColor = 'var(--danger-color)';
                break;
            case 'warning':
                element.style.animation = 'pulse 1s infinite';
                element.style.backgroundColor = 'var(--warning-color)';
                break;
        }
    }

    // Cleanup method
    destroy() {
        this.observers.forEach(observer => observer.disconnect());
        this.observers.clear();
        
        // Remove particle system
        const particleContainer = document.querySelector('.particle-container');
        if (particleContainer) {
            particleContainer.remove();
        }
        
        // Remove matrix effect
        const matrixRain = document.querySelector('.matrix-rain');
        if (matrixRain) {
            matrixRain.remove();
        }
    }
}

// Additional animation keyframes
const additionalStyles = document.createElement('style');
additionalStyles.textContent = `
    @keyframes shimmerPass {
        0% { left: -100%; }
        100% { left: 100%; }
    }

    @keyframes faceDetectionPulse {
        0%, 100% {
            transform: scale(1);
            opacity: 1;
        }
        50% {
            transform: scale(1.1);
            opacity: 0.7;
        }
    }

    @keyframes securityScan {
        0% {
            transform: translateX(-100%);
            opacity: 0;
        }
        50% {
            opacity: 1;
        }
        100% {
            transform: translateX(100%);
            opacity: 0;
        }
    }

    @keyframes matrixRain {
        0% {
            transform: translateY(-100%);
            opacity: 0;
        }
        10% {
            opacity: 1;
        }
        90% {
            opacity: 1;
        }
        100% {
            transform: translateY(100vh);
            opacity: 0;
        }
    }

    /* Responsive animation adjustments */
    @media (prefers-reduced-motion: reduce) {
        .particle-container,
        .matrix-rain {
            display: none;
        }
        
        * {
            animation-duration: 0.1s !important;
            transition-duration: 0.1s !important;
        }
    }

    @media (max-width: 768px) {
        .particle-container {
            display: none;
        }
        
        .holographic-shimmer {
            display: none;
        }
    }
`;

document.head.appendChild(additionalStyles);

// Initialize animation controller when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.animationController = new AnimationController();
});

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AnimationController;
}