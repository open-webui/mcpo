/* MCPO Skills Page — Split-panel layout */

const skillsState = {
    initialized: false,
    list: [],
    selectedId: null,
    isNew: false,
};

function getSkillsElements() {
    return {
        listContainer: document.getElementById('skills-list'),
        emptyState: document.getElementById('skills-empty-state'),
        editor: document.getElementById('skills-editor'),
        editorMode: document.getElementById('skills-editor-mode'),
        newBtn: document.getElementById('skills-new-btn'),
        deleteBtn: document.getElementById('skills-delete-btn'),
        saveBtn: document.getElementById('skills-save-btn'),
        idInput: document.getElementById('skills-id-input'),
        titleInput: document.getElementById('skills-title-input'),
        descInput: document.getElementById('skills-description-input'),
        contentInput: document.getElementById('skills-content-input'),
    };
}

// --- List rendering ---

function renderSkillsList() {
    const els = getSkillsElements();
    if (!els.listContainer) return;
    els.listContainer.innerHTML = '';

    if (!skillsState.list.length) {
        const empty = document.createElement('div');
        empty.className = 'sk-list-empty';
        empty.textContent = 'No skills yet';
        els.listContainer.appendChild(empty);
        return;
    }

    skillsState.list.forEach((skill) => {
        const card = document.createElement('div');
        card.className = 'sk-card';
        if (skill.id === skillsState.selectedId) card.classList.add('selected');

        const info = document.createElement('div');
        info.className = 'sk-card-info';
        info.addEventListener('click', () => selectSkill(skill.id));

        const title = document.createElement('div');
        title.className = 'sk-card-title';
        title.textContent = skill.title || skill.id;

        const desc = document.createElement('div');
        desc.className = 'sk-card-desc';
        desc.textContent = skill.description || skill.id;

        info.appendChild(title);
        info.appendChild(desc);

        const toggle = document.createElement('div');
        toggle.className = 'toggle' + (skill.enabled ? ' on' : '');
        toggle.title = skill.enabled ? 'Enabled — click to disable' : 'Disabled — click to enable';
        toggle.addEventListener('click', (e) => {
            e.stopPropagation();
            toggleSkillEnabled(skill.id, !skill.enabled, toggle);
        });

        card.appendChild(info);
        card.appendChild(toggle);
        els.listContainer.appendChild(card);
    });
}

// --- Skill selection ---

async function selectSkill(skillId) {
    const els = getSkillsElements();
    skillsState.selectedId = skillId;
    skillsState.isNew = false;
    renderSkillsList();
    showEditor(true);

    const skill = await fetchSkill(skillId);
    if (!skill) return;
    els.idInput.value = skill.id || '';
    els.idInput.readOnly = true;
    els.titleInput.value = skill.title || '';
    els.descInput.value = skill.description || '';
    els.contentInput.value = skill.content || '';
    els.editorMode.textContent = 'Editing';
    els.deleteBtn.style.display = '';
}

function startNewSkill() {
    const els = getSkillsElements();
    skillsState.selectedId = null;
    skillsState.isNew = true;
    renderSkillsList();
    showEditor(true);

    els.idInput.value = '';
    els.idInput.readOnly = false;
    els.titleInput.value = '';
    els.descInput.value = '';
    els.contentInput.value = '';
    els.editorMode.textContent = 'New Skill';
    els.deleteBtn.style.display = 'none';
    els.idInput.focus();
}

function showEditor(visible) {
    const els = getSkillsElements();
    if (els.editor) els.editor.style.display = visible ? '' : 'none';
    if (els.emptyState) els.emptyState.style.display = visible ? 'none' : '';
}

// --- Toggle enable/disable ---

async function toggleSkillEnabled(skillId, enabled, toggleEl) {
    // Optimistic
    toggleEl.classList.toggle('on', enabled);
    toggleEl.title = enabled ? 'Enabled — click to disable' : 'Disabled — click to enable';

    const ok = await setSkillEnabled(skillId, enabled);
    if (!ok) {
        toggleEl.classList.toggle('on', !enabled);
        toggleEl.title = !enabled ? 'Enabled — click to disable' : 'Disabled — click to enable';
        return;
    }
    // Update local list
    const item = skillsState.list.find((s) => s.id === skillId);
    if (item) item.enabled = enabled;
}

// --- Save ---

async function saveSkillFromEditor() {
    const els = getSkillsElements();
    if (!els.idInput) return;
    const payload = {
        id: (els.idInput.value || '').trim(),
        title: (els.titleInput.value || '').trim(),
        description: (els.descInput.value || '').trim(),
        content: els.contentInput.value || '',
    };
    if (!payload.id || !payload.title || !payload.content.trim()) {
        alert('ID, title, and content are required.');
        return;
    }
    const ok = await saveSkill(payload);
    if (!ok) {
        alert('Failed to save skill.');
        return;
    }
    skillsState.selectedId = payload.id;
    skillsState.isNew = false;
    await refreshSkillsPanel();
    // Visual feedback
    if (els.saveBtn) {
        const orig = els.saveBtn.textContent;
        els.saveBtn.textContent = 'Saved';
        setTimeout(() => { els.saveBtn.textContent = orig; }, 1200);
    }
}

// --- Delete ---

async function deleteSelectedSkill() {
    if (!skillsState.selectedId) return;
    const skill = skillsState.list.find((s) => s.id === skillsState.selectedId);
    const name = skill ? skill.title : skillsState.selectedId;
    if (!confirm(`Delete skill "${name}"? This cannot be undone.`)) return;

    const ok = await deleteSkill(skillsState.selectedId);
    if (!ok) {
        alert('Failed to delete skill.');
        return;
    }
    skillsState.selectedId = null;
    skillsState.isNew = false;
    showEditor(false);
    await refreshSkillsPanel();
}

// --- Refresh ---

async function refreshSkillsPanel() {
    skillsState.list = await fetchSkills();
    renderSkillsList();
    if (skillsState.selectedId) {
        await selectSkill(skillsState.selectedId);
    }
}

// --- Bind & Init ---

function bindSkillsHandlers() {
    const els = getSkillsElements();
    els.newBtn?.addEventListener('click', startNewSkill);
    els.saveBtn?.addEventListener('click', saveSkillFromEditor);
    els.deleteBtn?.addEventListener('click', deleteSelectedSkill);
}

function initSkillsPage() {
    if (skillsState.initialized) {
        refreshSkillsPanel().catch((e) => console.warn('[SKILLS] refresh failed:', e));
        return;
    }
    bindSkillsHandlers();
    skillsState.initialized = true;
    refreshSkillsPanel().catch((e) => console.warn('[SKILLS] init failed:', e));
}

window.initSkillsPage = initSkillsPage;
