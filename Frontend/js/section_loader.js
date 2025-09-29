/**
 * HTML Section Loader
 * Handles loading of HTML section files into the main index.html
 */

class HTMLSectionLoader {
    constructor() {
        this.loadedSections = new Map();
        this.basePath = '/frontend/html_section_files/';
    }

    async loadSection(sectionName) {
        // Check if already loaded
        if (this.loadedSections.has(sectionName)) {
            return this.loadedSections.get(sectionName);
        }

        try {
            const response = await fetch(`${this.basePath}${sectionName}.html`);
            if (!response.ok) {
                throw new Error(`Failed to load section: ${sectionName}`);
            }
            
            const html = await response.text();
            this.loadedSections.set(sectionName, html);
            return html;
        } catch (error) {
            console.error(`Error loading section ${sectionName}:`, error);
            return null;
        }
    }

    async loadAllSections() {
        const sections = [
            'dashboard',
            'camera_monitor',
            'person_search',
            'add_person',
            'watchlist',
            'access_control',
            'reports',
            'settings'
        ];

        const loadPromises = sections.map(section => this.loadSection(section));
        await Promise.all(loadPromises);
        
        return this.loadedSections;
    }

    insertSection(containerId, sectionName) {
        const container = document.getElementById(containerId);
        const html = this.loadedSections.get(sectionName);
        
        if (container && html) {
            container.innerHTML = html;
            return true;
        }
        
        return false;
    }

    async loadAndInsertSection(containerId, sectionName) {
        const html = await this.loadSection(sectionName);
        const container = document.getElementById(containerId);
        
        if (container && html) {
            container.innerHTML = html;
            return true;
        }
        
        return false;
    }
}

// Initialize section loader
window.sectionLoader = new HTMLSectionLoader();

// Load all sections when DOM is ready
document.addEventListener('DOMContentLoaded', async () => {
    console.log('🔄 Loading HTML sections...');
    
    const loadingIndicator = document.getElementById('section-loading');
    
    try {
        await window.sectionLoader.loadAllSections();
        console.log('✅ All HTML sections loaded successfully');
        
        // Insert sections into main content area
        const mainContent = document.querySelector('.main-content');
        if (mainContent) {
            let sectionsHTML = '';
            
            // Add each section to the main content
            const sections = ['dashboard', 'camera_monitor', 'person_search', 'add_person', 'watchlist', 'access_control', 'reports', 'settings'];
            
            for (const section of sections) {
                const html = window.sectionLoader.loadedSections.get(section);
                if (html) {
                    sectionsHTML += html;
                }
            }
            
            // Hide loading indicator
            if (loadingIndicator) {
                loadingIndicator.style.display = 'none';
            }
            
            // Add sections to main content (after loading indicator)
            mainContent.innerHTML += sectionsHTML;
            
            // Mark sections as loaded
            window.sectionsLoaded = true;
            
            // Dispatch custom event to notify main.js
            const sectionsLoadedEvent = new CustomEvent('sectionsLoaded');
            document.dispatchEvent(sectionsLoadedEvent);
        }
        
    } catch (error) {
        console.error('❌ Error loading HTML sections:', error);
        
        // Hide loading indicator
        if (loadingIndicator) {
            loadingIndicator.style.display = 'none';
        }
        
        // Fallback: show error message
        const mainContent = document.querySelector('.main-content');
        if (mainContent) {
            mainContent.innerHTML += `
                <div class="page active">
                    <div class="page-header">
                        <h1>Loading Error</h1>
                    </div>
                    <div class="coming-soon">
                        <i class="fas fa-exclamation-triangle"></i>
                        <h2>Failed to Load Sections</h2>
                        <p>Please refresh the page or check your internet connection.</p>
                        <button class="btn-primary" onclick="location.reload()">
                            <i class="fas fa-sync-alt"></i>
                            Reload Page
                        </button>
                    </div>
                </div>
            `;
        }
    }
});