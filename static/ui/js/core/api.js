/* MCPO API Management - Core API Functions */

// Centralized fetch wrapper to inject Authorization and always return a safe shape
async function fetchJson(path, options = {}) {
    const headers = new Headers(options.headers || {});
    const apiKey = localStorage.getItem('mcpo-api-key');
    if (apiKey && !headers.has('Authorization')) {
        headers.set('Authorization', `Bearer ${apiKey}`);
    }
    try {
        const resp = await fetch(path, { ...options, headers });
        // Try to parse JSON consistently
        const contentType = resp.headers.get('content-type') || '';
        let data;
        if (contentType.includes('application/json')) {
            data = await resp.json();
        } else {
            const text = await resp.text();
            try { data = JSON.parse(text); } catch { data = { ok: resp.ok, text }; }
        }
        return { response: resp, data };
    } catch (err) {
        // Network error or fetch aborted; return a response-like object to avoid throwing in callers
        const fakeResp = {
            ok: false,
            status: 0,
            statusText: 'Network Error',
            headers: new Headers(),
            url: path,
        };
        const data = { ok: false, error: String(err), detail: 'Network error contacting API' };
        console.warn('[API] fetchJson network error:', err);
        return { response: fakeResp, data };
    }
}

// Real API Functions
async function fetchServers() {
    try {
        const { response, data } = await fetchJson('/_meta/servers');
        if (data.ok) {
            return data.servers;
        }
        console.error('Failed to fetch servers:', data);
        return [];
    } catch (error) {
        console.error('Error fetching servers:', error);
        return [];
    }
}

async function fetchServerTools(serverName) {
    try {
        const { response, data } = await fetchJson(`/_meta/servers/${serverName}/tools`);
        if (data.ok) {
            return data.tools;
        }
        console.error(`Failed to fetch tools for ${serverName}:`, data);
        return [];
    } catch (error) {
        console.error(`Error fetching tools for ${serverName}:`, error);
        return [];
    }
}

async function toggleServerEnabled(serverName, enabled) {
    try {
        const endpoint = enabled ? 'enable' : 'disable';
        const url = `/_meta/servers/${serverName}/${endpoint}`;
        console.log(`[API] Calling ${url}`);
        
        console.log(`[API] Starting fetch to ${url}...`);
        const { response, data } = await fetchJson(url, { method: 'POST' });
        
        console.log(`[API] Fetch completed. Response status: ${response.status}`);
        console.log(`[API] Response ok: ${response.ok}`);
        
        if (!response.ok) {
            console.error(`[API] HTTP error! status: ${response.status}`);
            return false;
        }
        
        console.log(`[API] Parsing JSON response...`);
        
        console.log(`[API] Response data:`, data);
        
        if (!data || data.ok !== true) {
            console.error(`[API] API returned ok=false for ${endpoint} server ${serverName}:`, data);
            return false;
        }
        
        console.log(`[API] Successfully ${endpoint}d server ${serverName}`);
        return true;
    } catch (error) {
        console.error(`[API] Error toggling server ${serverName}:`, error);
        return false;
    }
}

async function toggleToolEnabled(serverName, toolName, enabled) {
    try {
        const endpoint = enabled ? 'enable' : 'disable';
        const url = `/_meta/servers/${serverName}/tools/${toolName}/${endpoint}`;
        console.log(`[API] Calling ${url}`);
        
        console.log(`[API] Starting fetch to ${url}...`);
        const { response, data } = await fetchJson(url, { method: 'POST' });
        
        console.log(`[API] Fetch completed. Response status: ${response.status}`);
        console.log(`[API] Response ok: ${response.ok}`);
        
        if (!response.ok) {
            console.error(`[API] HTTP error! status: ${response.status}`);
            return false;
        }
        
        console.log(`[API] Parsing JSON response...`);
        
        console.log(`[API] Response data:`, data);
        
        if (!data || data.ok !== true) {
            console.error(`[API] API returned ok=false for ${endpoint} tool ${toolName}:`, data);
            return false;
        }
        
        console.log(`[API] Successfully ${endpoint}d tool ${toolName}`);
        return true;
    } catch (error) {
        console.error(`[API] Error toggling tool ${toolName}:`, error);
        return false;
    }
}

async function reloadConfig() {
    try {
        const { response, data } = await fetchJson('/_meta/reload', { method: 'POST' });
        if (data.ok) {
            console.log('Config reloaded successfully');
            await updateServerStates();
        } else {
            console.error('Failed to reload config:', data);
        }
        return data.ok;
    } catch (error) {
        console.error('Error reloading config:', error);
        return false;
    }
}

async function reinitServer(serverName) {
    try {
        const { response, data } = await fetchJson(`/_meta/reinit/${serverName}`, { method: 'POST' });
        if (data.ok) {
            console.log(`Server ${serverName} reinitialized successfully`);
            await updateServerStates();
        } else {
            console.error(`Failed to reinitialize server ${serverName}:`, data);
        }
        return data.ok;
    } catch (error) {
        console.error(`Error reinitializing server ${serverName}:`, error);
        return false;
    }
}

// Config management API
async function loadConfigContent() {
    try {
        const { response, data } = await fetchJson('/_meta/config/content');
        if (data.ok) {
            const editor = document.getElementById('server-config-editor');
            if (editor) {
                editor.value = data.content;
            }
        } else {
            console.error('Failed to load config:', data);
        }
    } catch (error) {
        console.error('Error loading config:', error);
    }
}

async function saveConfigContent() {
    const editor = document.getElementById('server-config-editor');
    if (!editor) return;
    
    try {
        const { response, data } = await fetchJson('/_meta/config/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ content: editor.value })
        });
        if (data.ok) {
            console.log('Config saved successfully');
            await updateServerStates();
        } else {
            console.error('Failed to save config:', data);
        }
    } catch (error) {
        console.error('Error saving config:', error);
    }
}

async function saveRequirements() {
    const editor = document.getElementById('requirements-editor');
    if (!editor) return;
    
    try {
        const { response, data } = await fetchJson('/_meta/requirements/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ content: editor.value })
        });
        if (data.ok) {
            console.log('Requirements saved successfully');
        } else {
            console.error('Failed to save requirements:', data);
        }
    } catch (error) {
        console.error('Error saving requirements:', error);
    }
}

async function installDependencies() {
    try {
        const { response, data } = await fetchJson('/_meta/install-dependencies', { method: 'POST' });
        if (data.ok) {
            console.log('Dependencies installation started');
        } else {
            console.error('Failed to install dependencies:', data);
            alert('Failed to install dependencies. Check logs for details.');
        }
    } catch (error) {
        console.error('Error installing dependencies:', error);
        alert('Error installing dependencies. Check logs for details.');
    }
}

async function loadRequirementsContent() {
    try {
        const { response, data } = await fetchJson('/_meta/requirements/content');
        if (data.ok) {
            const editor = document.getElementById('requirements-editor');
            if (editor) {
                editor.value = data.content;
            }
        } else {
            console.error('Failed to load requirements:', data);
        }
    } catch (error) {
        console.error('Error loading requirements:', error);
    }
}

async function loadAboutContent() {
    const aboutPage = document.getElementById('about-page');
    if (!aboutPage) return;
    
    // Only load once
    if (aboutPage.innerHTML.trim()) return;
    
    try {
        // Load version info
        const versionResp = await fetch('/_meta/metrics');
        let version = '1.0.0-rc1';
        if (versionResp.ok) {
            const metrics = await versionResp.json();
            version = metrics.version || version;
        }
        
        // Load about page content
        const resp = await fetch('/ui/about.html');
        if (resp.ok) {
            const content = await resp.text();
            aboutPage.innerHTML = content.trim() ? content : '<p>About page not found.</p>';
            
            // Update version in the about page
            const versionSpan = document.getElementById('about-version');
            if (versionSpan) {
                versionSpan.textContent = version;
            }
        } else {
            aboutPage.innerHTML = '<p>Failed to load about page.</p>';
        }
    } catch (error) {
        console.error('Error loading about page:', error);
        aboutPage.innerHTML = '<p>Error loading about page.</p>';
    }
}

async function fetchSkills() {
    try {
        const { data } = await fetchJson('/_meta/skills');
        if (data && data.ok && Array.isArray(data.skills)) {
            return data.skills;
        }
    } catch (error) {
        console.error('Error fetching skills:', error);
    }
    return [];
}

async function fetchSkill(skillId) {
    try {
        const { data } = await fetchJson(`/_meta/skills/${encodeURIComponent(skillId)}`);
        if (data && data.ok && data.skill) {
            return data.skill;
        }
    } catch (error) {
        console.error('Error fetching skill:', error);
    }
    return null;
}

async function saveSkill(payload) {
    try {
        const { data } = await fetchJson('/_meta/skills', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload || {}),
        });
        return !!(data && data.ok);
    } catch (error) {
        console.error('Error saving skill:', error);
        return false;
    }
}

async function deleteSkill(skillId) {
    try {
        const { data } = await fetchJson(`/_meta/skills/${encodeURIComponent(skillId)}`, {
            method: 'DELETE',
        });
        return !!(data && data.ok);
    } catch (error) {
        console.error('Error deleting skill:', error);
        return false;
    }
}

async function setSkillEnabled(skillId, enabled) {
    const suffix = enabled ? 'enable' : 'disable';
    try {
        const { data } = await fetchJson(`/_meta/skills/${encodeURIComponent(skillId)}/${suffix}`, {
            method: 'POST',
        });
        return !!(data && data.ok);
    } catch (error) {
        console.error('Error toggling skill:', error);
        return false;
    }
}

// Expose for inline handlers
window.saveConfigContent = saveConfigContent;
window.installDependencies = installDependencies;
window.saveRequirements = saveRequirements;
window.loadAboutContent = loadAboutContent;
window.fetchSkills = fetchSkills;
window.fetchSkill = fetchSkill;
window.saveSkill = saveSkill;
window.deleteSkill = deleteSkill;
window.setSkillEnabled = setSkillEnabled;
