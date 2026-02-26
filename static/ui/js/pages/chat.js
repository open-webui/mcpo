/* MCPO Chat Page Logic - OpenRouter Agentic Chat */

const CHAT_SESSION_STORAGE_KEY = 'mcpo-chat-session-id';
const FAVORITES_STORAGE_KEY = 'mcpo-chat-favorite-models';
const MODEL_SEARCH_STORAGE_KEY = 'mcpo-chat-model-search';

const chatState = {
    initialized: false,
    sessionId: null,
    session: null,
    models: [],
    skills: [],
    selectedSkillIds: [],
    favorites: [],
    modelSearch: '',
    streaming: false,
    abortController: null,
    buffer: '',
    stepMessages: {},
    currentStepId: null,
    modelPanelOpen: false,
    // UI state that should persist across re-renders
    expandedReasoningIds: new Set(),  // Track which reasoning blocks are expanded
    expandedToolIds: new Set(),       // Track which tool cards are expanded
};

function getChatElements() {
    return {
        modelSelect: document.getElementById('chat-model-select'),
        modelList: document.getElementById('chat-model-list'),
        modelListItems: document.getElementById('chat-model-list-items'),
        modelSearchInput: document.getElementById('chat-model-search'),
        modelFavToggle: document.getElementById('chat-model-fav-toggle'),
        skillToggle: document.getElementById('chat-skill-toggle'),
        skillList: document.getElementById('chat-skill-list'),
        skillListItems: document.getElementById('chat-skill-list-items'),
        activeSkills: document.getElementById('chat-active-skills'),
        resetBtn: document.getElementById('chat-reset-session'),
        messagesContainer: document.getElementById('chat-messages'),
        alertsContainer: document.getElementById('chat-alerts'),
        form: document.getElementById('chat-input-form'),
        textarea: document.getElementById('chat-input-text'),
        streamToggle: document.getElementById('chat-stream-toggle'),
        stopBtn: document.getElementById('chat-stop-stream'),
        sessionMeta: document.getElementById('chat-session-meta'),
    };
}

async function initChatPage() {
    if (chatState.initialized) {
        return;
    }
    chatState.initialized = true;

    const els = getChatElements();
    if (!els.modelSelect || !els.form) {
        console.warn('[CHAT] Page elements missing; abort init');
        return;
    }

    attachEventHandlers(els);

    await loadModels();
    await loadSkills();
    await loadFavorites();
    populateModelSelect();
    renderSkillList();
    await ensureSession();
    renderSession();
}

function attachEventHandlers(els) {
    els.modelSelect.addEventListener('change', async (event) => {
        if (!chatState.sessionId) return;
        chatState.session.model = event.target.value;
        await persistSessionMeta();
    });

    if (els.resetBtn) {
        els.resetBtn.addEventListener('click', async () => {
            await resetSession();
        });
    }

    if (els.modelFavToggle) {
        els.modelFavToggle.addEventListener('click', () => {
            chatState.modelPanelOpen = !chatState.modelPanelOpen;
            renderModelList();
        });
    }

    if (els.skillToggle) {
        els.skillToggle.addEventListener('click', () => {
            if (!els.skillList) return;
            els.skillList.classList.toggle('hidden');
        });
    }

    if (els.modelSearchInput) {
        els.modelSearchInput.addEventListener('input', (e) => {
            chatState.modelSearch = e.target.value || '';
            localStorage.setItem(MODEL_SEARCH_STORAGE_KEY, chatState.modelSearch);
            renderModelList();
        });
    }

    els.form.addEventListener('submit', async (event) => {
        event.preventDefault();
        await sendChatMessage();
    });

    // Enter to send, Shift+Enter for newline
    els.textarea.addEventListener('keydown', async (event) => {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            await sendChatMessage();
        }
    });

    els.stopBtn.addEventListener('click', () => {
        abortStreaming();
    });
}

async function loadSkills() {
    try {
        chatState.skills = await fetchSkills();
    } catch (error) {
        console.warn('[CHAT] Failed to load skills', error);
        chatState.skills = [];
    }
}

function getSelectedSkillIds() {
    return (chatState.selectedSkillIds || []).filter(Boolean);
}

function renderSkillList() {
    const { skillListItems } = getChatElements();
    if (!skillListItems) return;
    skillListItems.innerHTML = '';
    const selected = new Set(getSelectedSkillIds());
    const enabledSkills = (chatState.skills || []).filter((item) => item.enabled !== false);
    if (!enabledSkills.length) {
        skillListItems.innerHTML = '<div class="empty-state">No enabled skills</div>';
        renderActiveSkills();
        return;
    }
    enabledSkills.forEach((skill) => {
        const row = document.createElement('label');
        row.className = 'model-row';
        row.style.cursor = 'pointer';
        row.innerHTML = `
            <input type="checkbox" ${selected.has(skill.id) ? 'checked' : ''} style="margin-right:8px;" />
            <div class="model-text">
                <div class="model-label">${escapeHtml(skill.title || skill.id)}</div>
                <div class="model-sub">${escapeHtml(skill.id)}</div>
            </div>
        `;
        const checkbox = row.querySelector('input[type="checkbox"]');
        checkbox?.addEventListener('change', async () => {
            const next = new Set(getSelectedSkillIds());
            if (checkbox.checked) {
                next.add(skill.id);
            } else {
                next.delete(skill.id);
            }
            chatState.selectedSkillIds = Array.from(next);
            if (chatState.session) {
                chatState.session.skillIds = Array.from(next);
                await persistSessionMeta();
            } else {
                renderActiveSkills();
            }
        });
        skillListItems.appendChild(row);
    });
    renderActiveSkills();
}

function renderActiveSkills() {
    const { activeSkills } = getChatElements();
    if (!activeSkills) return;
    const selected = getSelectedSkillIds();
    if (!selected.length) {
        activeSkills.textContent = 'All enabled skills (default)';
        return;
    }
    activeSkills.textContent = `Selected: ${selected.join(', ')}`;
}

async function loadModels() {
    try {
        const { response, data } = await fetchJson('/chat/sessions/models');
        if (!response.ok || !data.models) {
            console.warn('[CHAT] Failed to load models', data);
            return;
        }
        chatState.models = data.models || [];
        renderModelList();
    } catch (error) {
        console.error('[CHAT] Error loading models', error);
    }
}

async function loadFavorites() {
    // Load from server (primary source of truth)
    try {
        const { response, data } = await fetchJson('/chat/sessions/favorites');
        if (response.ok && Array.isArray(data.favorites)) {
            chatState.favorites = data.favorites;
            // Sync to localStorage for offline fallback
            localStorage.setItem(FAVORITES_STORAGE_KEY, JSON.stringify(chatState.favorites));
        } else {
            // Fallback to localStorage if server fails
            const raw = localStorage.getItem(FAVORITES_STORAGE_KEY);
            chatState.favorites = raw ? JSON.parse(raw) : [];
        }
    } catch (error) {
        console.warn('[CHAT] Failed to load favorites from server, using localStorage', error);
        const raw = localStorage.getItem(FAVORITES_STORAGE_KEY);
        chatState.favorites = raw ? JSON.parse(raw) : [];
    }

    const savedSearch = localStorage.getItem(MODEL_SEARCH_STORAGE_KEY);
    if (typeof savedSearch === 'string') {
        chatState.modelSearch = savedSearch;
        const els = getChatElements();
        if (els.modelSearchInput) {
            els.modelSearchInput.value = savedSearch;
        }
    }
}

async function saveFavorites() {
    // Save to localStorage immediately for UI responsiveness
    localStorage.setItem(FAVORITES_STORAGE_KEY, JSON.stringify(chatState.favorites || []));
    
    // Persist to server for cross-browser/device sync
    try {
        await fetchJson('/chat/sessions/favorites', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ models: chatState.favorites || [] }),
        });
    } catch (error) {
        console.warn('[CHAT] Failed to persist favorites to server', error);
    }
}

async function toggleFavorite(modelId) {
    if (!modelId) return;
    const set = new Set(chatState.favorites || []);
    if (set.has(modelId)) {
        set.delete(modelId);
    } else {
        set.add(modelId);
    }
    chatState.favorites = Array.from(set);
    await saveFavorites();
    populateModelSelect();
    renderModelList();
    ensureAllowedModel();
}

function getAllowedModels() {
    const favorites = new Set(chatState.favorites || []);
    if (favorites.size > 0) {
        const allowed = chatState.models.filter((m) => favorites.has(m.id));
        if (allowed.length > 0) return allowed;
    }
    return chatState.models;
}

function ensureAllowedModel() {
    const allowed = getAllowedModels();
    if (!allowed.length) return;
    if (!chatState.session) return;
    if (!allowed.find((m) => m.id === chatState.session.model)) {
        chatState.session.model = allowed[0].id;
        persistSessionMeta();
        populateModelSelect();
    }
}

function populateModelSelect() {
    const { modelSelect } = getChatElements();
    if (!modelSelect) return;

    modelSelect.innerHTML = '';
    const favoritesSet = new Set(chatState.favorites || []);
    const allowed = getAllowedModels();
    const favorites = chatState.models.filter((m) => favoritesSet.has(m.id));
    const others = chatState.models.filter((m) => !favoritesSet.has(m.id));

    if (favorites.length > 0) {
        const favGroup = document.createElement('optgroup');
        favGroup.label = '★ Favorites';
        favorites.forEach((model) => {
            const option = document.createElement('option');
            option.value = model.id;
            option.textContent = formatModelLabel(model);
            favGroup.appendChild(option);
        });
        modelSelect.appendChild(favGroup);

        if (others.length > 0) {
            const otherGroup = document.createElement('optgroup');
            otherGroup.label = 'All models (enable in favorites to use)';
            others.forEach((model) => {
                const option = document.createElement('option');
                option.value = model.id;
                option.textContent = formatModelLabel(model);
                option.disabled = true; // visible but cannot be selected
                otherGroup.appendChild(option);
            });
            modelSelect.appendChild(otherGroup);
        }
    } else {
        allowed.forEach((model) => {
            const option = document.createElement('option');
            option.value = model.id;
            option.textContent = formatModelLabel(model);
            modelSelect.appendChild(option);
        });
    }

    const targetModel = chatState.session?.model;
    if (targetModel && allowed.find((m) => m.id === targetModel)) {
        modelSelect.value = targetModel;
    } else if (allowed[0]) {
        modelSelect.value = allowed[0].id;
        if (chatState.session) {
            chatState.session.model = allowed[0].id;
            persistSessionMeta();
        }
    }
}

function renderModelList() {
    const { modelList, modelListItems, modelSearchInput } = getChatElements();
    if (!modelList || !modelListItems) return;
    if (!chatState.modelPanelOpen) {
        modelList.classList.add('hidden');
        modelListItems.innerHTML = '';
        return;
    }
    modelList.classList.remove('hidden');
    modelListItems.innerHTML = '';

    const favorites = new Set(chatState.favorites || []);
    const query = (chatState.modelSearch || '').toLowerCase();

    const models = chatState.models.filter((model) => {
        if (!query) return true;
        const haystack = `${model.id.toLowerCase()} ${(model.label || '').toLowerCase()} ${inferProvider(model.id).toLowerCase()}`;
        return haystack.includes(query);
    });

    models.forEach((model) => {
        const row = document.createElement('div');
        row.className = 'model-row';

        const star = document.createElement('button');
        star.type = 'button';
        star.className = `model-star ${favorites.has(model.id) ? 'active' : ''}`;
        star.title = favorites.has(model.id) ? 'Unfavorite' : 'Favorite';
        star.textContent = '★';
        star.addEventListener('click', () => toggleFavorite(model.id));

        const label = document.createElement('div');
        label.className = 'model-label';
        label.textContent = formatModelLabel(model);

        const sub = document.createElement('div');
        sub.className = 'model-sub';
        sub.textContent = model.id;

        const provider = document.createElement('div');
        provider.className = 'model-provider';
        provider.textContent = inferProvider(model.id);

        row.appendChild(star);
        const textWrap = document.createElement('div');
        textWrap.className = 'model-text';
        textWrap.appendChild(label);
        textWrap.appendChild(sub);
        textWrap.appendChild(provider);
        row.appendChild(textWrap);

        modelListItems.appendChild(row);
    });

    if (modelSearchInput && modelSearchInput.value !== chatState.modelSearch) {
        modelSearchInput.value = chatState.modelSearch;
    }
}

function inferProvider(modelId) {
    const id = (modelId || '').toLowerCase();
    if (id.includes('minimax') || id.startsWith('m2')) return 'MiniMax';
    if (id.startsWith('gemini') || id.includes('google')) return 'Google / Gemini';
    if (id.startsWith('claude') || id.includes('anthropic')) return 'Anthropic';
    if (id.startsWith('openrouter') || id.startsWith('router') || id.includes('/')) return 'OpenRouter';
    if (id.startsWith('gpt') || id.includes('openai')) return 'OpenAI';
    return 'Model';
}

function formatModelLabel(model) {
    const provider = inferProvider(model.id);
    const label = model.label || model.id;
    return `${provider}: ${label}`;
}

async function ensureSession() {
    const storedId = localStorage.getItem(CHAT_SESSION_STORAGE_KEY);
    if (storedId) {
        const session = await fetchSession(storedId);
        if (session) {
            chatState.sessionId = storedId;
            // Preserve locally-selected model if user changed it before sending
            const localModel = chatState.session?.model;
            chatState.session = session;
            chatState.selectedSkillIds = Array.isArray(session.skillIds) ? session.skillIds.slice() : [];
            if (localModel && localModel !== session.model) {
                // User changed model locally - preserve that choice
                chatState.session.model = localModel;
            }
            persistSessionMeta();
            return;
        }
    }
    await createSession();
}

async function createSession() {
    const allowed = getAllowedModels();
    const defaultModel = allowed?.[0]?.id;
    if (!defaultModel) {
        throw new Error('No OpenRouter models available');
    }
    try {
        const payload = {
            model: defaultModel,
            skill_ids: getSelectedSkillIds(),
            system_prompt: `You are HubUI Assistant, an embedded AI agent in OpenHubUI—a companion app for Open WebUI.

## Your Purpose
OpenHubUI solves a critical limitation: Open WebUI only supports MCP via Streamable HTTP, but most MCP servers use stdio or SSE transports. OpenHubUI proxies ANY MCP server (stdio, SSE, or HTTP) into a single aggregated Streamable HTTP endpoint that Open WebUI can consume.

Additionally, OpenAPI tool integrations don't work as reliably as native MCP—OpenHubUI provides native MCP protocol support for better tool execution.

## Your Role
Help users:
1. Add and configure MCP servers in OpenHubUI
2. Connect the aggregated MCP endpoint to Open WebUI
3. Troubleshoot connection and configuration issues

## Tool Usage Guidelines
When calling MCP tools:
- **Always provide ALL required parameters** - check the tool schema carefully
- **For timezone parameters**: Use IANA timezone format (e.g., "Europe/Paris", "America/New_York", "Asia/Tokyo")
- **Infer values from context**: If user says "Paris", use "Europe/Paris"; "New York" → "America/New_York"; "Tokyo" → "Asia/Tokyo"
- If a required parameter is ambiguous, ask the user for clarification before calling the tool

## Management Tools (Direct Access)
- mcpo.get_config / mcpo.post_config — Read and update MCP server configuration
- mcpo.get_requirements / mcpo.post_requirements — Manage Python dependencies  
- mcpo.install_python_package — Install packages needed by MCP servers
- mcpo.get_logs — View server logs for debugging

## How to Connect OpenHubUI to Open WebUI
The proxy on port 8001 exposes MCP servers in two ways:

**Option A: Single Aggregated Connection (Recommended)**
- URL: http://host.docker.internal:8001 (Docker) or http://localhost:8001 (local)
- All MCP servers appear as one connection in Open WebUI
- Tools from all servers available together

**Option B: Per-Server Connections**
- URL: http://host.docker.internal:8001/{server-name}
- Example: http://host.docker.internal:8001/perplexity
- Each server gets its own toggle in Open WebUI's External Tools
- Useful if user wants to enable/disable specific servers in Open WebUI

**Steps to add in Open WebUI:**
1. Go to ⚙️ Admin Settings → External Tools
2. Click + (Add Server)
3. Set Type to "MCP (Streamable HTTP)"
4. Enter the URL (aggregate or per-server)
5. Auth: None (unless API key configured)
6. Save

## Guidelines
- Always read current config (mcpo.get_config) before making changes
- Explain what you're doing and why
- After config changes, remind users to check the Configuration → Client Configuration tab for the connection JSON
- Be concise and action-oriented

The user is viewing the OpenHubUI admin interface.`,
        };
        const { response, data } = await fetchJson('/chat/sessions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        if (!response.ok || !data.session) {
            throw new Error(data?.error || 'Failed to create session');
        }
        chatState.sessionId = data.session.id;
        chatState.session = data.session;
        chatState.selectedSkillIds = Array.isArray(data.session.skillIds) ? data.session.skillIds.slice() : [];
        chatState.stepMessages = {};
        chatState.currentStepId = null;
        localStorage.setItem(CHAT_SESSION_STORAGE_KEY, chatState.sessionId);
        populateModelSelect();
        persistSessionMeta();
    } catch (error) {
        console.error('[CHAT] createSession failed', error);
        showChatAlert('Session creation failed', 'error');
    }
}

async function fetchSession(sessionId) {
    try {
        const { response, data } = await fetchJson(`/chat/sessions/${sessionId}`);
        if (response.ok && data.session) {
            return data.session;
        }
    } catch (error) {
        console.warn('[CHAT] fetchSession failed', error);
    }
    return null;
}

async function resetSession() {
    if (!chatState.sessionId) return;
    try {
        const { response, data } = await fetchJson(`/chat/sessions/${chatState.sessionId}/reset`, {
            method: 'POST',
        });
        if (!response.ok || !data.session) {
            throw new Error('Failed to reset session');
        }
        chatState.session = data.session;
        chatState.selectedSkillIds = Array.isArray(data.session.skillIds) ? data.session.skillIds.slice() : [];
        chatState.stepMessages = {};
        chatState.currentStepId = null;
        renderSession();
        showChatAlert('Session cleared', 'info');
    } catch (error) {
        console.error('[CHAT] resetSession failed', error);
        showChatAlert('Failed to reset session', 'error');
    }
}

async function persistSessionMeta() {
    const { modelSelect } = getChatElements();
    if (modelSelect && chatState.session?.model) {
        modelSelect.value = chatState.session.model;
    }
    renderSkillList();
    renderSession();
}

async function sendChatMessage() {
    const els = getChatElements();
    if (!els.textarea || chatState.streaming) return;

    const message = (els.textarea.value || '').trim();
    if (!message) {
        return;
    }

    await ensureSession();
    if (!chatState.sessionId) {
        showChatAlert('Cannot send message: no session', 'error');
        return;
    }

    const stream = els.streamToggle?.checked !== false;

    appendUserMessage(message);
    els.textarea.value = '';

    const payload = {
        message,
        stream,
        model: chatState.session.model,
        skill_ids: getSelectedSkillIds(),
    };

    if (stream) {
        await sendStreamingMessage(payload);
    } else {
        await sendStandardMessage(payload);
    }
}

async function sendStandardMessage(payload) {
    try {
        const { response, data } = await fetchJson(`/chat/sessions/${chatState.sessionId}/messages`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        if (!response.ok) {
            throw new Error(data?.detail || 'Chat request failed');
        }
        chatState.session = data.session;
        chatState.selectedSkillIds = Array.isArray(data.session.skillIds) ? data.session.skillIds.slice() : [];
        chatState.stepMessages = {};
        chatState.currentStepId = null;
        renderSession();
    } catch (error) {
        console.error('[CHAT] sendStandardMessage failed', error);
        showChatAlert(`Chat failed: ${error.message}`, 'error');
    }
}

async function sendStreamingMessage(payload) {
    abortStreaming();

    chatState.streaming = true;
    const els = getChatElements();
    els.stopBtn.disabled = false;
    toggleFormDisabled(true);

    const controller = new AbortController();
    chatState.abortController = controller;
    chatState.buffer = '';

    try {
        const response = await fetch(`/chat/sessions/${chatState.sessionId}/messages`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
            signal: controller.signal,
        });

        if (!response.ok || !response.body) {
            throw new Error('Streaming response failed');
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (chatState.streaming) {
            const { value, done } = await reader.read();
            if (done) break;
            if (!value) continue;
            const chunk = decoder.decode(value, { stream: true });
            chatState.buffer += chunk;
            await processSSEBuffer();
        }
    } catch (error) {
        if (controller.signal.aborted) {
            showChatAlert('Streaming aborted', 'info');
        } else {
            console.error('[CHAT] streaming error', error);
            showChatAlert(`Streaming error: ${error.message}`, 'error');
        }
    } finally {
        finalizeStreaming();
        await refreshSessionState();
    }
}

async function processSSEBuffer() {
    const messages = chatState.buffer.split('\n\n');
    chatState.buffer = messages.pop();

    for (const raw of messages) {
        if (!raw.trim()) continue;
        const line = raw.trim();
        if (!line.startsWith('data:')) continue;
        const payload = line.slice('data:'.length).trim();
        if (!payload || payload === '[DONE]') continue;
        try {
            const event = JSON.parse(payload);
            await handleStreamEvent(event);
        } catch (error) {
            console.warn('[CHAT] Failed to parse stream event', payload, error);
        }
    }
}

async function handleStreamEvent(event) {
    switch (event.type) {
        case 'session.updated':
            chatState.session = event.session;
            renderSession();
            break;
        case 'skills.loaded':
            displaySkillsLoaded(event.skills || []);
            break;
        case 'step.started':
            appendStepMessage(event.step);
            break;
        case 'step.completed':
            updateStepMessage(event.step, event.status);
            break;
        case 'tool.call.started':
            appendToolCall(event.toolCall);
            break;
        case 'tool.call.delta':
            updateToolCallDelta(event.toolCall);
            break;
        case 'tool.call.result':
            completeToolCall(event.toolCall);
            break;
        case 'message.delta':
            appendStreamingMessageDelta(event.text);
            break;
        case 'reasoning.delta':
            appendStreamingReasoningDelta(event.text);
            break;
        case 'message.completed':
            finalizeAssistantMessage(event.message);
            break;
        case 'error':
            showChatAlert(event.message || 'Streaming error', 'error');
            break;
        case 'done':
            finalizeStreaming();
            break;
        default:
            break;
    }
}

function displaySkillsLoaded(skills) {
    if (!skills.length) return;
    const container = document.getElementById('chat-messages');
    if (!container) return;
    const names = skills.map((s) => s.title || s.id).join(', ');
    const el = document.createElement('div');
    el.className = 'skills-loaded-indicator';
    el.innerHTML = `<span class="skills-loaded-icon">&#9889;</span> Skills loaded: <strong>${escapeHtml(names)}</strong>`;
    container.appendChild(el);
    el.scrollIntoView({ behavior: 'smooth', block: 'end' });
}

function appendUserMessage(content) {
    chatState.session = chatState.session || { messages: [] };
    chatState.session.messages = chatState.session.messages || [];
    chatState.session.messages.push({ role: 'user', content });
    renderMessages();
}

function appendStreamingMessageDelta(text) {
    if (!text) return;
    const messages = chatState.session.messages || [];
    let current = messages[messages.length - 1];

    // Create new message if current is a step, has tools, or is finished
    if (!current || current.role !== 'assistant' || current.streamingFinished || current.step || (current.tool_calls && current.tool_calls.length > 0)) {
        current = { role: 'assistant', content: '', reasoning: '' };
        messages.push(current);
    }
    current.content = (current.content || '') + text;
    renderMessages();
}

function appendStreamingReasoningDelta(text) {
    if (!text) return;
    const messages = chatState.session.messages || [];
    let current = messages[messages.length - 1];

    // Create new message if current is a step, has tools, or is finished
    if (!current || current.role !== 'assistant' || current.streamingFinished || current.step || (current.tool_calls && current.tool_calls.length > 0)) {
        current = { role: 'assistant', content: '', reasoning: '' };
        messages.push(current);
    }
    current.reasoning = (current.reasoning || '') + text;
    current.isThinking = true;
    renderMessages();
}

function finalizeAssistantMessage(message) {
    if (!chatState.session) return;
    const messages = chatState.session.messages || [];
    const current = messages[messages.length - 1];
    if (current && current.role === 'assistant') {
        current.content = message?.content || current.content || '';
        current.streamingFinished = true;
        current.isThinking = false;
        current.tool_calls = message?.tool_calls || current.tool_calls || [];
        current.reasoning = message?.reasoning || current.reasoning || '';
    } else {
        messages.push({
            role: 'assistant',
            content: message?.content || '',
            tool_calls: message?.tool_calls || [],
            reasoning: message?.reasoning || ''
        });
    }
    renderMessages();
}

function appendToolCall(toolCall) {
    if (!toolCall || !chatState.currentStepId) return;
    const message = chatState.stepMessages[chatState.currentStepId];
    if (!message) return;
    message.toolCalls = message.toolCalls || [];
    message.toolCalls.push({ ...toolCall, status: 'running' });
    renderMessages();
}

function updateToolCallDelta(toolCall) {
    if (!toolCall) return;
    const stepId = chatState.currentStepId;
    if (!stepId) return;
    const message = chatState.stepMessages[stepId];
    if (!message || !message.toolCalls) return;
    const target = message.toolCalls.find((c) => c.id === toolCall.id);
    if (!target) return;
    target.arguments = toolCall.arguments;
    renderMessages();
}

function completeToolCall(toolCall) {
    if (!toolCall) return;
    
    // First try current step
    const stepId = chatState.currentStepId;
    if (stepId) {
        const message = chatState.stepMessages[stepId];
        if (message?.toolCalls) {
            const target = message.toolCalls.find((c) => c.id === toolCall.id);
            if (target) {
                target.status = 'completed';
                target.result = toolCall.result;
                renderMessages();
                return;
            }
        }
    }
    
    // Fall back to searching all messages
    const messages = chatState.session?.messages || [];
    for (const msg of messages) {
        const calls = msg.toolCalls || msg.tool_calls || [];
        const target = calls.find((c) => c.id === toolCall.id);
        if (target) {
            target.status = 'completed';
            target.result = toolCall.result;
            renderMessages();
            return;
        }
    }
}

async function refreshSessionState() {
    if (!chatState.sessionId) return;
    const latest = await fetchSession(chatState.sessionId);
    if (latest) {
        // Preserve tool calls from streaming stepMessages into latest session messages
        // The server doesn't always return full tool call details, so merge local state
        const localMessages = chatState.session?.messages || [];
        const serverMessages = latest.messages || [];
        
        // Merge tool calls from local streaming state into server state
        for (let i = 0; i < serverMessages.length && i < localMessages.length; i++) {
            const local = localMessages[i];
            const server = serverMessages[i];
            
            // Preserve tool calls if server doesn't have them
            if (local.toolCalls && local.toolCalls.length > 0 && !server.tool_calls?.length) {
                server.tool_calls = local.toolCalls;
            }
            // Preserve reasoning expanded state
            if (local.reasoning && !server.reasoning) {
                server.reasoning = local.reasoning;
            }
        }
        
        // Also merge any step messages that have tool calls
        for (const [stepId, stepMsg] of Object.entries(chatState.stepMessages)) {
            if (stepMsg.toolCalls && stepMsg.toolCalls.length > 0) {
                // Find matching message in server state
                const idx = serverMessages.findIndex(m => m.stepId === stepId || m.step);
                if (idx >= 0) {
                    serverMessages[idx].toolCalls = stepMsg.toolCalls;
                    serverMessages[idx].tool_calls = stepMsg.toolCalls;
                }
            }
        }
        
        chatState.session = latest;
        chatState.selectedSkillIds = Array.isArray(latest.skillIds) ? latest.skillIds.slice() : [];
        renderSession();
    }
}

function finalizeStreaming() {
    if (!chatState.streaming) return;
    chatState.streaming = false;
    abortStreaming();
    toggleFormDisabled(false);
    renderSession();
}

function abortStreaming() {
    const els = getChatElements();
    if (chatState.abortController) {
        chatState.abortController.abort();
    }
    chatState.abortController = null;
    chatState.streaming = false;
    if (els.stopBtn) els.stopBtn.disabled = true;
    toggleFormDisabled(false);
}

function toggleFormDisabled(disabled) {
    const els = getChatElements();
    if (els.textarea) els.textarea.disabled = disabled;
    if (els.modelSelect) els.modelSelect.disabled = disabled;
    if (els.skillToggle) els.skillToggle.disabled = disabled;
    if (els.resetBtn) els.resetBtn.disabled = disabled;
}

function renderSession() {
    renderMessages();
    renderActiveSkills();
    renderSessionMeta();
    // Sync model dropdown with session state
    const { modelSelect } = getChatElements();
    if (modelSelect && chatState.session?.model) {
        modelSelect.value = chatState.session.model;
    }
}

function renderMessages() {
    const { messagesContainer } = getChatElements();
    if (!messagesContainer) return;
    messagesContainer.innerHTML = '';

    const messages = chatState.session?.messages || [];
    messages.forEach((message, msgIndex) => {
        const role = message.role || 'assistant';

        // Skip system messages - don't show in chat window
        if (role === 'system') return;
        
        // Skip tool role messages - results are shown in tool cards
        if (role === 'tool') return;

        // Create message block container
        const block = document.createElement('div');
        block.className = `chat-message-block chat-message-block--${role}`;

        // Generate unique IDs for state tracking
        const reasoningId = `reasoning-${msgIndex}`;
        
        // 1. Render reasoning/thinking (Collapsed by default, persistent via chatState)
        if (role === 'assistant' && message.reasoning) {
            // Use persistent state from chatState, default to collapsed (false)
            const isExpanded = chatState.expandedReasoningIds.has(reasoningId);
            const reasoningEl = document.createElement('div');
            // Use chat-tool-call classes for consistent card styling
            reasoningEl.className = `chat-tool-call chat-reasoning ${isExpanded ? 'expanded' : ''} ${message.isThinking ? 'chat-reasoning--streaming' : ''}`;

            reasoningEl.innerHTML = `
                <div class="chat-tool-call__header">
                    <div class="chat-tool-call__name">
                        <svg class="chat-tool-call__icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm0 18a8 8 0 1 1 8-8 8 8 0 0 1-8 8z"/>
                            <path d="M12 6v6l4 2"/>
                        </svg>
                        <span>Thinking</span>
                    </div>
                    <div class="chat-tool-call__status">
                         ${message.isThinking ? '<span class="chat-tool-call__dot"></span>' : ''}
                    </div>
                </div>
                <div class="chat-tool-call__body">
                    <div class="chat-reasoning__content">${escapeHtml(message.reasoning)}</div>
                </div>
            `;

            // Attach toggle listener using persistent state
            const header = reasoningEl.querySelector('.chat-tool-call__header');
            header.addEventListener('click', () => {
                if (chatState.expandedReasoningIds.has(reasoningId)) {
                    chatState.expandedReasoningIds.delete(reasoningId);
                } else {
                    chatState.expandedReasoningIds.add(reasoningId);
                }
                renderMessages();
            });

            block.appendChild(reasoningEl);
        }

        // 2. Render tool calls (Before content to allow [Tools] -> [Result/Text])
        const toolCalls = message.tool_calls || message.toolCalls || [];
        if (toolCalls.length) {
            const toolsContainer = document.createElement('div');
            toolsContainer.className = 'chat-tool-calls';

            toolCalls.forEach((call, callIndex) => {
                const toolEl = document.createElement('div');
                const status = call.status || (call.result ? 'success' : 'running');
                const isError = call.result?.ok === false || status === 'error';
                
                // Track tool expansion state
                const toolId = `tool-${msgIndex}-${callIndex}`;
                const isToolExpanded = chatState.expandedToolIds.has(toolId);

                toolEl.className = `chat-tool-call ${isError ? 'chat-tool-call--error' : status === 'running' ? 'chat-tool-call--running' : 'chat-tool-call--success'} ${isToolExpanded ? 'expanded' : ''}`;

                // Parse duration if available
                const duration = call.result?.duration_ms || call.duration_ms;
                const durationText = duration ? `${duration}ms` : '';

                toolEl.innerHTML = `
                    <div class="chat-tool-call__header">
                        <div class="chat-tool-call__name">
                            <svg class="chat-tool-call__icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/>
                            </svg>
                            <span>${escapeHtml(call.name || call.function?.name || 'Tool')}</span>
                        </div>
                        <div class="chat-tool-call__status">
                            <div class="chat-tool-call__indicator">
                                <span class="chat-tool-call__dot"></span>
                                <span class="chat-tool-call__dot"></span>
                                <span class="chat-tool-call__dot"></span>
                            </div>
                            ${durationText ? `<span class="chat-tool-call__duration">${durationText}</span>` : ''}
                            <svg class="chat-tool-call__chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <polyline points="6 9 12 15 18 9"></polyline>
                            </svg>
                        </div>
                    </div>
                    <div class="chat-tool-call__body">
                        <div class="chat-tool-call__section">
                            <div class="chat-tool-call__label">Arguments</div>
                            <pre class="chat-tool-call__args">${formatToolArgs(call.arguments || call.function?.arguments)}</pre>
                        </div>
                        ${call.result ? `
                        <div class="chat-tool-call__section">
                            <div class="chat-tool-call__label">Result</div>
                            <pre class="chat-tool-call__result">${formatToolResult(call.result)}</pre>
                        </div>
                        ` : ''}
                    </div>
                `;

                // Add click handler for tool expansion toggle
                const toolHeader = toolEl.querySelector('.chat-tool-call__header');
                toolHeader.addEventListener('click', () => {
                    if (chatState.expandedToolIds.has(toolId)) {
                        chatState.expandedToolIds.delete(toolId);
                    } else {
                        chatState.expandedToolIds.add(toolId);
                    }
                    renderMessages();
                });

                toolsContainer.appendChild(toolEl);
            });

            block.appendChild(toolsContainer);
        }

        // 3. Render main message bubble (After tools)
        // Skip content that looks like raw tool result JSON (already shown in tool cards)
        const content = message.content;
        const contentStr = typeof content === 'string' ? content.trim() : '';
        const isToolResultJson = contentStr.startsWith('{') && contentStr.endsWith('}') && 
            (contentStr.includes('"ok":') && contentStr.includes('"output":'));
        
        if (content && !isToolResultJson) {
            const wrapper = document.createElement('div');
            wrapper.className = `chat-message ${role}`;

            const bubble = document.createElement('div');
            bubble.className = 'chat-bubble';
            bubble.innerHTML = formatMessageContent(content);
            wrapper.appendChild(bubble);

            block.appendChild(wrapper);
        }

        messagesContainer.appendChild(block);
    });

    // Add streaming indicator if currently streaming
    if (chatState.streaming) {
        const indicator = document.createElement('div');
        indicator.className = 'chat-streaming-indicator';
        indicator.innerHTML = `
            <div class="chat-streaming-dots">
                <span class="chat-streaming-dot"></span>
                <span class="chat-streaming-dot"></span>
                <span class="chat-streaming-dot"></span>
            </div>
            <span>Generating...</span>
        `;
        messagesContainer.appendChild(indicator);
    }

    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function stripThinkTags(content) {
    // Strip <think>...</think> tags from content for display
    // These are preserved in message history for interleaved thinking
    if (!content) return content;
    return content.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
}

function formatMessageContent(content) {
    if (!content) return '';
    // Strip <think> tags for display (kept in history for thinking continuity)
    let cleanContent = stripThinkTags(content);
    // Basic markdown-like formatting
    let html = escapeHtml(cleanContent);
    // Code blocks
    html = html.replace(/```(\w*)\n?([\s\S]*?)```/g, '<pre><code>$2</code></pre>');
    // Inline code
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
    // Bold
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    // Line breaks
    html = html.replace(/\n/g, '<br>');
    return html;
}

function formatToolArgs(args) {
    if (!args) return escapeHtml('{}');
    try {
        let obj = args;
        if (typeof args === 'string') {
            obj = JSON.parse(args);
        }
        const json = JSON.stringify(obj, null, 2);
        return syntaxHighlightJson(json);
    } catch {
        return escapeHtml(String(args));
    }
}

function syntaxHighlightJson(json) {
    // Escape HTML first, then apply syntax highlighting
    const escaped = escapeHtml(json);
    // Apply syntax highlighting with spans
    return escaped
        // Strings (already escaped quotes)
        .replace(/(&quot;[^&]*&quot;)(\s*:)?/g, (match, str, colon) => {
            if (colon) {
                // Key
                return `<span class="json-key">${str}</span>${colon}`;
            }
            // Value string
            return `<span class="json-string">${str}</span>`;
        })
        // Numbers
        .replace(/\b(-?\d+\.?\d*)\b/g, '<span class="json-number">$1</span>')
        // Booleans and null
        .replace(/\b(true|false|null)\b/g, '<span class="json-boolean">$1</span>');
}

function formatToolResult(result) {
    if (!result) return '';
    try {
        if (typeof result === 'string') {
            // Plain string - escape and return
            return escapeHtml(result);
        }
        // Handle MCPO tool result format
        if (result.ok === false && result.error) {
            return `<span class="json-error">Error: ${escapeHtml(result.error)}</span>`;
        }
        if (result.output !== undefined) {
            if (typeof result.output === 'string') {
                // Check if it's JSON-like
                try {
                    const parsed = JSON.parse(result.output);
                    return syntaxHighlightJson(JSON.stringify(parsed, null, 2));
                } catch {
                    return escapeHtml(result.output);
                }
            }
            return syntaxHighlightJson(JSON.stringify(result.output, null, 2));
        }
        return syntaxHighlightJson(JSON.stringify(result, null, 2));
    } catch {
        return escapeHtml(String(result));
    }
}

function renderSteps() {
    const { stepsContainer } = getChatElements();
    if (!stepsContainer) return;

    stepsContainer.innerHTML = '';
    const steps = chatState.session?.steps || [];
    if (!steps.length) {
        stepsContainer.innerHTML = '<div class="empty-state">No agent steps yet.</div>';
        return;
    }

    steps.forEach((step) => {
        const card = document.createElement('div');
        card.className = 'chat-step-card';

        const title = document.createElement('div');
        title.className = 'chat-step-title';
        title.textContent = step.title || step.type || 'Step';
        card.appendChild(title);

        const meta = document.createElement('div');
        meta.className = 'chat-step-meta';
        const created = new Date(step.createdAt || Date.now());
        meta.innerHTML = `<span>${created.toLocaleTimeString()}</span>`;
        if (step.detail?.finishReason) {
            meta.innerHTML += `<span>Finish: ${step.detail.finishReason}</span>`;
        }
        if (step.detail?.summary) {
            meta.innerHTML += `<span>Summary: ${step.detail.summary}</span>`;
        }
        card.appendChild(meta);

        const toolCalls = step.detail?.toolCalls || [];
        toolCalls.forEach((call) => {
            const callNode = document.createElement('div');
            callNode.className = 'chat-step-tool';
            const status = call.status ? `status=${call.status}` : '';
            const args = call.arguments ? safeJson(call.arguments) : '';
            const result = call.result ? `<br><strong>Result:</strong> ${safeJson(call.result)}` : '';
            callNode.innerHTML = `<strong>${call.name}</strong> ${status}<br><strong>Args:</strong> ${args}${result}`;
            card.appendChild(callNode);
        });

        stepsContainer.appendChild(card);
    });
}

function renderToolCatalog() {
    const { toolsContainer } = getChatElements();
    if (!toolsContainer) return;

    toolsContainer.innerHTML = '';
    const entries = Object.entries(chatState.toolCatalog || {});
    if (!entries.length) {
        toolsContainer.innerHTML = '<div class="empty-state">No tools available</div>';
        return;
    }

    entries.forEach(([server, tools]) => {
        const wrapper = document.createElement('div');
        wrapper.className = 'chat-tool-item';

        const header = document.createElement('h3');
        header.textContent = server;
        wrapper.appendChild(header);

        if (!tools.length) {
            const empty = document.createElement('div');
            empty.textContent = 'No callable endpoints';
            empty.className = 'empty-state';
            wrapper.appendChild(empty);
        } else {
            const list = document.createElement('ul');
            tools.forEach((tool) => {
                const li = document.createElement('li');
                li.innerHTML = `<code>${tool}</code>`;
                list.appendChild(li);
            });
            wrapper.appendChild(list);
        }

        toolsContainer.appendChild(wrapper);
    });
}

function renderSessionMeta() {
    const { sessionMeta } = getChatElements();
    if (!sessionMeta) return;

    if (!chatState.session) {
        sessionMeta.textContent = 'No active session';
        return;
    }

    const created = new Date(chatState.session.createdAt || Date.now());
    sessionMeta.innerHTML = `Session <code>${chatState.session.id}</code> • Created ${created.toLocaleString()}`;
}

function showChatAlert(message, variant = 'info') {
    const { alertsContainer } = getChatElements();
    if (!alertsContainer) return;

    const node = document.createElement('div');
    node.className = `chat-alert ${variant}`;
    node.innerHTML = `<span>${message}</span>`;

    alertsContainer.prepend(node);
    setTimeout(() => {
        if (alertsContainer.contains(node)) {
            alertsContainer.removeChild(node);
        }
    }, 5000);
}

function safeJson(value) {
    if (!value) return '';
    try {
        if (typeof value === 'string') {
            JSON.parse(value);
            return value;
        }
        return JSON.stringify(value, null, 2);
    } catch (error) {
        return String(value);
    }
}

function appendStepMessage(step) {
    if (!step) return;
    const message = {
        role: 'assistant',
        step: true,
        stepId: step.id,
        title: step.title || step.type || 'Agent step',
        content: step.detail?.summary || step.title || 'Agent step in progress…',
        finishReason: step.detail?.finishReason || null,
        summary: step.detail?.summary || null,
        toolCalls: [],
    };
    chatState.session = chatState.session || { messages: [] };
    chatState.session.messages = chatState.session.messages || [];
    chatState.session.messages.push(message);
    chatState.stepMessages[step.id] = message;
    chatState.currentStepId = step.id;
    renderMessages();
}

function updateStepMessage(step, status) {
    if (!step) return;
    const message = chatState.stepMessages[step.id];
    if (!message) return;
    message.finishReason = step.detail?.finishReason || status || null;
    if (step.detail?.summary) {
        message.summary = step.detail.summary;
        message.content = step.detail.summary;
    }
    renderMessages();
}

window.initChatPage = initChatPage;
