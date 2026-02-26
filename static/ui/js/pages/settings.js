/* MCPO Settings Page */

const settingsState = {
    initialized: false,
    codeModeEnabled: false,
};

function getSettingsElements() {
    return {
        codeModeToggle: document.getElementById('settings-code-mode-toggle'),
        codeModeHint: document.getElementById('settings-code-mode-hint'),
        apiKeyInput: document.getElementById('settings-api-key-input'),
        apiKeyReveal: document.getElementById('settings-api-key-reveal'),
        apiKeySave: document.getElementById('settings-api-key-save'),
        apiKeyHint: document.getElementById('settings-api-key-hint'),
        themeBtn: document.getElementById('settings-theme-btn'),
        clearStorageBtn: document.getElementById('settings-clear-storage'),
    };
}

// --- Code Mode ---

async function loadCodeModeState() {
    try {
        const { data } = await fetchJson('/_meta/code-mode');
        if (data && data.ok !== undefined) {
            settingsState.codeModeEnabled = !!data.enabled;
        }
    } catch (e) {
        console.warn('[SETTINGS] Failed to load code mode state:', e);
    }
    renderCodeModeToggle();
}

function renderCodeModeToggle() {
    const els = getSettingsElements();
    if (!els.codeModeToggle) return;
    els.codeModeToggle.classList.toggle('on', settingsState.codeModeEnabled);
    if (els.codeModeHint) {
        els.codeModeHint.textContent = settingsState.codeModeEnabled
            ? 'Code mode is ON — clients see search_tools + execute_tool only.'
            : 'Code mode is OFF — clients see all individual tools.';
        els.codeModeHint.classList.toggle('hint-active', settingsState.codeModeEnabled);
    }
}

async function toggleCodeMode() {
    const newState = !settingsState.codeModeEnabled;
    const els = getSettingsElements();

    // Optimistic UI
    settingsState.codeModeEnabled = newState;
    renderCodeModeToggle();

    try {
        const { data } = await fetchJson('/_meta/code-mode', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ enabled: newState }),
        });
        if (!data || !data.ok) {
            // Revert
            settingsState.codeModeEnabled = !newState;
            renderCodeModeToggle();
            console.error('[SETTINGS] Failed to toggle code mode:', data);
        }
    } catch (e) {
        settingsState.codeModeEnabled = !newState;
        renderCodeModeToggle();
        console.error('[SETTINGS] Error toggling code mode:', e);
    }
}

// --- API Key ---

function loadApiKeyState() {
    const els = getSettingsElements();
    if (!els.apiKeyInput) return;
    const stored = localStorage.getItem('mcpo-api-key') || '';
    els.apiKeyInput.value = stored;
    updateApiKeyHint(stored);
}

function updateApiKeyHint(key) {
    const els = getSettingsElements();
    if (!els.apiKeyHint) return;
    if (key) {
        els.apiKeyHint.textContent = 'API key is set. All requests will include Bearer authentication.';
        els.apiKeyHint.classList.add('hint-active');
    } else {
        els.apiKeyHint.textContent = 'No API key configured. Requests are unauthenticated.';
        els.apiKeyHint.classList.remove('hint-active');
    }
}

function saveApiKey() {
    const els = getSettingsElements();
    if (!els.apiKeyInput) return;
    const key = els.apiKeyInput.value.trim();
    if (key) {
        localStorage.setItem('mcpo-api-key', key);
    } else {
        localStorage.removeItem('mcpo-api-key');
    }
    updateApiKeyHint(key);
    // Brief visual feedback
    if (els.apiKeySave) {
        const original = els.apiKeySave.textContent;
        els.apiKeySave.textContent = 'Saved';
        setTimeout(() => { els.apiKeySave.textContent = original; }, 1200);
    }
}

function toggleApiKeyReveal() {
    const els = getSettingsElements();
    if (!els.apiKeyInput || !els.apiKeyReveal) return;
    const isPassword = els.apiKeyInput.type === 'password';
    els.apiKeyInput.type = isPassword ? 'text' : 'password';
    els.apiKeyReveal.textContent = isPassword ? 'Hide' : 'Show';
}

// --- Clear Storage ---

function clearLocalStorage() {
    if (!confirm('Clear all local preferences? This removes your API key, theme, chat history, and page state.')) {
        return;
    }
    localStorage.clear();
    location.reload();
}

// --- Bind & Init ---

function bindSettingsHandlers() {
    const els = getSettingsElements();
    if (els.codeModeToggle) {
        els.codeModeToggle.addEventListener('click', toggleCodeMode);
    }
    if (els.apiKeySave) {
        els.apiKeySave.addEventListener('click', saveApiKey);
    }
    if (els.apiKeyReveal) {
        els.apiKeyReveal.addEventListener('click', toggleApiKeyReveal);
    }
    if (els.apiKeyInput) {
        els.apiKeyInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                saveApiKey();
            }
        });
    }
    if (els.themeBtn) {
        els.themeBtn.addEventListener('click', () => {
            if (typeof toggleTheme === 'function') toggleTheme();
        });
    }
    if (els.clearStorageBtn) {
        els.clearStorageBtn.addEventListener('click', clearLocalStorage);
    }
}

async function initSettingsPage() {
    if (settingsState.initialized) {
        // Re-entering: just refresh code mode state
        await loadCodeModeState();
        loadApiKeyState();
        return;
    }
    bindSettingsHandlers();
    settingsState.initialized = true;
    await loadCodeModeState();
    loadApiKeyState();
}

window.initSettingsPage = initSettingsPage;
