from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from mcpo.services.state import get_state_manager


_BOOL_TRUE = {"1", "true", "yes", "on"}


@dataclass
class SkillDefinition:
    id: str
    title: str
    description: str
    content: str
    enabled: bool = True
    priority: int = 100
    scopes: List[str] | None = None
    providers: List[str] | None = None
    models: List[str] | None = None
    tags: List[str] | None = None
    source_path: str | None = None


def _skills_dir() -> Path:
    configured = (os.getenv("MCPO_SKILLS_DIR") or "skills").strip()
    return Path(configured).resolve()


def _to_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in _BOOL_TRUE


def _parse_list_value(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    text = str(raw).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    parts = [part.strip().strip("\"'") for part in text.split(",")]
    return [part for part in parts if part]


def _parse_frontmatter(raw: str) -> tuple[Dict[str, Any], str]:
    text = raw or ""
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---", 4)
    if end == -1:
        return {}, text
    meta_blob = text[4:end]
    body = text[end + 4 :].lstrip("\n")
    meta: Dict[str, Any] = {}
    for line in meta_blob.splitlines():
        if not line.strip() or line.strip().startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip()
    return meta, body


def _safe_skill_id(value: str) -> str:
    sid = re.sub(r"[^0-9A-Za-z_-]", "-", (value or "").strip().lower()).strip("-")
    return sid or "skill"


def _build_skill_from_file(path: Path) -> Optional[SkillDefinition]:
    if not path.exists() or not path.is_file():
        return None
    raw = path.read_text(encoding="utf-8")
    meta, body = _parse_frontmatter(raw)
    sid = _safe_skill_id(str(meta.get("id") or path.stem))
    title = str(meta.get("title") or sid)
    description = str(meta.get("description") or "")
    enabled = _to_bool(meta.get("enabled"), default=True)
    try:
        priority = int(str(meta.get("priority", "100")))
    except ValueError:
        priority = 100
    scopes = _parse_list_value(meta.get("scopes"))
    providers = _parse_list_value(meta.get("providers"))
    models = _parse_list_value(meta.get("models"))
    tags = _parse_list_value(meta.get("tags"))
    return SkillDefinition(
        id=sid,
        title=title,
        description=description,
        content=(body or "").strip(),
        enabled=enabled,
        priority=priority,
        scopes=scopes or None,
        providers=providers or None,
        models=models or None,
        tags=tags or None,
        source_path=str(path),
    )


def list_skills() -> List[SkillDefinition]:
    root = _skills_dir()
    if not root.exists():
        return []
    skills: List[SkillDefinition] = []
    for path in sorted(root.glob("*.md")):
        skill = _build_skill_from_file(path)
        if skill:
            skills.append(skill)
    state = get_state_manager()
    states = state.get_all_skill_states()
    for skill in skills:
        override = states.get(skill.id, {})
        if "enabled" in override:
            skill.enabled = bool(override["enabled"])
    skills.sort(key=lambda item: (item.priority, item.id))
    return skills


def get_skill(skill_id: str) -> Optional[SkillDefinition]:
    sid = _safe_skill_id(skill_id)
    for skill in list_skills():
        if skill.id == sid:
            return skill
    return None


def upsert_skill_file(*, skill_id: str, title: str, description: str, content: str) -> SkillDefinition:
    sid = _safe_skill_id(skill_id)
    root = _skills_dir()
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{sid}.md"
    payload = "\n".join(
        [
            "---",
            f"id: {sid}",
            f"title: {title or sid}",
            f"description: {description or ''}",
            "enabled: true",
            "priority: 100",
            "scopes: [chat, completions]",
            "---",
            (content or "").rstrip(),
            "",
        ]
    )
    path.write_text(payload, encoding="utf-8")
    skill = _build_skill_from_file(path)
    if not skill:
        raise ValueError(f"Failed to load saved skill: {sid}")
    return skill


def delete_skill_file(skill_id: str) -> bool:
    sid = _safe_skill_id(skill_id)
    path = _skills_dir() / f"{sid}.md"
    if not path.exists():
        return False
    path.unlink()
    return True


def _matches_scope(skill: SkillDefinition, scope: str) -> bool:
    if not skill.scopes:
        return True
    return scope in {s.strip().lower() for s in skill.scopes}


def _matches_provider(skill: SkillDefinition, provider: Optional[str]) -> bool:
    if not skill.providers or not provider:
        return True
    p = provider.strip().lower()
    allowed = {item.strip().lower() for item in skill.providers}
    return p in allowed


def _matches_model(skill: SkillDefinition, model: Optional[str]) -> bool:
    if not skill.models or not model:
        return True
    m = model.strip().lower()
    for rule in skill.models:
        r = rule.strip().lower()
        if not r:
            continue
        if r.endswith("*") and m.startswith(r[:-1]):
            return True
        if m == r:
            return True
    return False


def select_skills(
    *,
    scope: str,
    model: Optional[str],
    provider: Optional[str],
    requested_skill_ids: Optional[Sequence[str]] = None,
) -> List[SkillDefinition]:
    selected: List[SkillDefinition] = []
    requested = {_safe_skill_id(item) for item in (requested_skill_ids or []) if str(item).strip()}
    for skill in list_skills():
        if not skill.enabled:
            continue
        if requested and skill.id not in requested:
            continue
        if not _matches_scope(skill, scope):
            continue
        if not _matches_provider(skill, provider):
            continue
        if not _matches_model(skill, model):
            continue
        selected.append(skill)
    selected.sort(key=lambda item: (item.priority, item.id))
    return selected


def compile_skills_system_prompt(
    *,
    scope: str,
    model: Optional[str],
    provider: Optional[str],
    requested_skill_ids: Optional[Sequence[str]] = None,
) -> str:
    skills = select_skills(
        scope=scope,
        model=model,
        provider=provider,
        requested_skill_ids=requested_skill_ids,
    )
    if not skills:
        return ""
    lines: List[str] = ["Agent Skills (system-managed instructions):"]
    for skill in skills:
        lines.append(f"\n[Skill: {skill.title} | id={skill.id}]")
        lines.append((skill.content or "").strip())
    return "\n".join(lines).strip()

