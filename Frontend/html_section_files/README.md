# HTML Section Files Structure

This directory contains the modular HTML sections that have been extracted from the main `index.html` file to improve maintainability and organization.

## File Structure

```
Frontend/html_section_files/
├── dashboard.html          # Dashboard page content
├── camera_monitor.html     # Camera monitoring interface
├── person_search.html      # Person search functionality
├── add_person.html         # Person registration form
├── watchlist.html          # Security watchlist management
├── access_control.html     # Access control system
├── reports.html            # Analytics and reports
└── settings.html           # System settings
```

## How It Works

1. **Main Index File**: The `index.html` now serves as the main shell containing:
   - HTML head with meta tags, CSS imports, and fonts
   - Header with system status and user info
   - Navigation sidebar
   - Empty main content area (loaded dynamically)
   - Script imports

2. **Section Loader**: The `section_loader.js` handles:
   - Loading HTML section files asynchronously
   - Caching loaded sections for performance
   - Inserting sections into the main content area
   - Error handling for failed loads

3. **Dynamic Loading**: When the page loads:
   - Section loader fetches all HTML files
   - Sections are inserted into the main content area
   - Main application (AeroSecure) initializes after sections are loaded
   - Navigation and functionality work exactly as before

## Benefits

- **Maintainability**: Each page is now a separate file, making it easier to maintain
- **Organization**: Code is better organized and easier to navigate
- **Modularity**: Individual sections can be developed independently
- **Performance**: Sections are cached after first load
- **Scalability**: Easy to add new pages by creating new HTML files

## Adding New Sections

To add a new page/section:

1. Create a new HTML file in this directory (e.g., `new_feature.html`)
2. Add the section name to the `sections` array in `section_loader.js`
3. Add corresponding navigation item in `index.html`
4. Add page initialization logic in `main.js` if needed

## File Naming Convention

- Use lowercase with underscores: `section_name.html`
- Keep names descriptive and consistent
- Match with the page ID used in navigation

## Content Structure

Each section file should contain:
```html
<!-- Page Name -->
<div id="page-id" class="page">
    <div class="page-header">
        <h1>Page Title</h1>
        <div class="page-actions">
            <!-- Action buttons -->
        </div>
    </div>
    
    <!-- Page content -->
</div>
```

## Important Notes

- All existing functionality is preserved
- CSS classes and IDs remain the same
- JavaScript event handlers work unchanged
- The website behaves exactly as it did before the split
- No breaking changes to the user experience