/* MCPO Navigation Management - Page Navigation & Persistence */

// Page titles and subtitles for header
const PAGE_HEADERS = {
    'tools-page': { title: 'MCP Servers', subtitle: 'Manage your MCP server connections' },
    'chat-page': { title: 'Chat', subtitle: 'Test and refine models forwarded by OpenHubUI' },
    'skills-page': { title: 'Skills', subtitle: 'Create, enable, and maintain agent skills' },
    'logs-page': { title: 'Logs', subtitle: 'Server activity and debug information' },
    'config-page': { title: 'Configuration', subtitle: 'Server settings and client config export' },
    'settings-page': { title: 'Settings', subtitle: 'Code mode, authentication, and preferences' },
    'about-page': { title: 'About', subtitle: 'Information about OpenHubUI' }
};

// Update header title and subtitle
function updatePageHeader(pageId) {
    const header = PAGE_HEADERS[pageId] || { title: 'OpenHubUI', subtitle: '' };
    const titleEl = document.getElementById('page-title');
    const subtitleEl = document.getElementById('page-subtitle');
    if (titleEl) titleEl.textContent = header.title;
    if (subtitleEl) subtitleEl.textContent = header.subtitle;
}

// Navigation Management - Using single delegated event listener below to avoid conflicts

// Navigate to config page from tools button
function openConfigPage() {
    showPage('config-page');
}

// No legacy remapping: removed pages are not redirected to avoid ambiguity.

// Normalize legacy/alias page IDs to current ones
function normalizePageId(pageId) {
    // If missing or unknown, callers fall back generically in showPage.
    return pageId || 'tools-page';
}

// Page Navigation with Persistence
function showPage(pageId) {
    // Hide all pages
    document.querySelectorAll('.page').forEach(page => page.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(nav => nav.classList.remove('active'));

    // Resolve and validate target page
    let targetId = normalizePageId(pageId);
    let page = document.getElementById(targetId);
    if (!page) {
        console.warn(`[NAV] Unknown pageId "${pageId}", falling back to tools-page`);
        targetId = 'tools-page';
        page = document.getElementById(targetId);
    }

    // Show selected page
    const nav = document.querySelector(`[data-page="${targetId.replace('-page', '')}"]`);
    if (page) page.classList.add('active');
    if (nav) nav.classList.add('active');

    // Update header title
    updatePageHeader(targetId);

    // Save resolved page
    localStorage.setItem('mcpo-current-page', targetId);

    // Page-specific initializers with guard; show a soft banner on error
    try {
        if (targetId === 'about-page' && typeof window.loadAboutContent === 'function') {
            window.loadAboutContent();
        }
        if (targetId === 'chat-page' && typeof window.initChatPage === 'function') {
            window.initChatPage();
        }
        if (targetId === 'skills-page' && typeof window.initSkillsPage === 'function') {
            window.initSkillsPage();
        }
        if (targetId === 'settings-page' && typeof window.initSettingsPage === 'function') {
            window.initSettingsPage();
        }
    } catch (e) {
        console.warn('[NAV] Page initializer error:', e);
        const banner = document.createElement('div');
        banner.className = 'empty-state';
        banner.style.padding = '12px';
        banner.style.color = 'var(--text-secondary)';
        banner.textContent = 'Page loaded with limited functionality due to API error. UI remains available.';
        page?.prepend(banner);
    }
}

function loadCurrentPage() {
    const savedPage = localStorage.getItem('mcpo-current-page');
    const initial = normalizePageId(savedPage) || 'chat-page';
    showPage(initial);
}

// Update existing navigation clicks to use persistence
document.addEventListener('click', function(e) {
    const navItem = e.target.closest('.nav-item[data-page]');
    if (navItem) {
        const pageId = navItem.getAttribute('data-page') + '-page';
        console.log(`[NAV] Clicked nav item, showing page: ${pageId}`);
        showPage(pageId);
    }
});

// Expose for inline handlers
window.openConfigPage = openConfigPage;
