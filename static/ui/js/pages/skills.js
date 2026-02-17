/* MCPO Skills Page */

const skillsState = {
    initialized: false,
    list: [],
    selectedId: null,
};

function getSkillsElements() {
    return {
        select: document.getElementById('skills-select'),
        refreshBtn: document.getElementById('skills-refresh-btn'),
        newBtn: document.getElementById('skills-new-btn'),
        toggleBtn: document.getElementById('skills-toggle-btn'),
        deleteBtn: document.getElementById('skills-delete-btn'),
        saveBtn: document.getElementById('skills-save-btn'),
        idInput: document.getElementById('skills-id-input'),
        titleInput: document.getElementById('skills-title-input'),
        descInput: document.getElementById('skills-description-input'),
        contentInput: document.getElementById('skills-content-input'),
    };
}

function renderSkillsSelect() {
    const els = getSkillsElements();
    if (!els.select) return;
    els.select.innerHTML = '';
    if (!skillsState.list.length) {
        const empty = document.createElement('option');
        empty.value = '';
        empty.textContent = 'No skills found';
        els.select.appendChild(empty);
        return;
    }
    skillsState.list.forEach((skill) => {
        const option = document.createElement('option');
        option.value = skill.id;
        option.textContent = `${skill.enabled ? 'ON' : 'OFF'} ${skill.title} (${skill.id})`;
        els.select.appendChild(option);
    });
    const target = skillsState.selectedId || skillsState.list[0].id;
    els.select.value = target;
}

async function loadSkillIntoEditor(skillId) {
    const els = getSkillsElements();
    if (!skillId || !els.idInput) return;
    const skill = await fetchSkill(skillId);
    if (!skill) return;
    skillsState.selectedId = skill.id;
    els.idInput.value = skill.id || '';
    els.titleInput.value = skill.title || '';
    els.descInput.value = skill.description || '';
    els.contentInput.value = skill.content || '';
}

async function refreshSkillsPanel() {
    const els = getSkillsElements();
    if (!els.select) return;
    skillsState.list = await fetchSkills();
    if (!skillsState.selectedId && skillsState.list.length) {
        skillsState.selectedId = skillsState.list[0].id;
    }
    renderSkillsSelect();
    if (skillsState.selectedId) {
        await loadSkillIntoEditor(skillsState.selectedId);
    }
}

function clearSkillEditor() {
    const els = getSkillsElements();
    if (!els.idInput) return;
    skillsState.selectedId = null;
    els.idInput.value = '';
    els.titleInput.value = '';
    els.descInput.value = '';
    els.contentInput.value = '';
}

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
        alert('Skill id, title, and content are required.');
        return;
    }
    const ok = await saveSkill(payload);
    if (!ok) {
        alert('Failed to save skill.');
        return;
    }
    skillsState.selectedId = payload.id;
    await refreshSkillsPanel();
}

async function toggleSelectedSkill() {
    const current = skillsState.list.find((item) => item.id === skillsState.selectedId);
    if (!current) return;
    const ok = await setSkillEnabled(current.id, !current.enabled);
    if (!ok) {
        alert('Failed to update skill state.');
        return;
    }
    await refreshSkillsPanel();
}

async function deleteSelectedSkill() {
    if (!skillsState.selectedId) return;
    const ok = await deleteSkill(skillsState.selectedId);
    if (!ok) {
        alert('Failed to delete skill.');
        return;
    }
    clearSkillEditor();
    await refreshSkillsPanel();
}

function bindSkillsHandlers() {
    const els = getSkillsElements();
    if (!els.select) return;
    els.select.addEventListener('change', async () => {
        skillsState.selectedId = els.select.value || null;
        if (skillsState.selectedId) {
            await loadSkillIntoEditor(skillsState.selectedId);
        }
    });
    els.refreshBtn?.addEventListener('click', async () => refreshSkillsPanel());
    els.newBtn?.addEventListener('click', () => clearSkillEditor());
    els.saveBtn?.addEventListener('click', async () => saveSkillFromEditor());
    els.toggleBtn?.addEventListener('click', async () => toggleSelectedSkill());
    els.deleteBtn?.addEventListener('click', async () => deleteSelectedSkill());
}

function initSkillsPage() {
    if (skillsState.initialized) {
        refreshSkillsPanel().catch((error) => {
            console.warn('[SKILLS] refresh failed:', error);
        });
        return;
    }
    bindSkillsHandlers();
    skillsState.initialized = true;
    refreshSkillsPanel().catch((error) => {
        console.warn('[SKILLS] init failed:', error);
    });
}

document.addEventListener('DOMContentLoaded', initSkillsPage);
window.initSkillsPage = initSkillsPage;
