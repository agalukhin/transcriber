from __future__ import annotations

import re
from pathlib import Path

from .config import Config
from .session import Meta, Rec, Session, atomic_write_text

APP_NAMES = {"kontur.talk": "Толк", "zoom.us": "Zoom"}


def app_label(app: str) -> str:
    return APP_NAMES.get(app, "")


APP_LOCATIVE = {"kontur.talk": "Толке", "zoom.us": "Zoom"}


def default_title(meta: Meta) -> str:
    name = APP_LOCATIVE.get(meta.app)
    return f"Встреча в {name}" if name else "Встреча"


def title_of(meta: Meta) -> str:
    return meta.title.strip() or default_title(meta)


def clean_title(title: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[/\\:|#^\[\]]", " ", title)).strip()


def slug(title: str) -> str:
    return re.sub(r"-+", "-", re.sub(r"\s+", "-", clean_title(title))).strip("-")


def stamp(meta: Meta) -> tuple[str, str]:
    return meta.started_at.strftime("%Y-%m-%d"), meta.started_at.strftime("%H%M")


def raw_filename(meta: Meta) -> str:
    date, hhmm = stamp(meta)
    return f"Local-{date}-{hhmm}-{slug(title_of(meta))}.md"


def note_filename(meta: Meta) -> str:
    date, hhmm = stamp(meta)
    return f"MTG-{date} {hhmm} · {clean_title(title_of(meta))}.md"


def mmss(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 60:02d}:{s % 60:02d}"


def hhmm(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}"


def render_raw(meta: Meta, records: list[Rec], n_speakers: int, status_text: str, marker_every_s: int = 300) -> str:
    date, _ = stamp(meta)
    title = title_of(meta)
    who = ", ".join(meta.participants) if meta.participants else "не получены"
    app = app_label(meta.app) or "ручной старт"
    lines = [
        "---", "type: raw", "source: local", f"source_date: {date}", "projects: []", "tags: [raw]", "---", "",
        f"# {title} — {meta.started_at.strftime('%Y-%m-%d %H:%M')}", "",
        f"Локальная запись ({app}). Участники: {who}. "
        f"Говорящих на канале собеседников: {n_speakers}. Статус: {status_text}. "
        "«Me» — Владислав (микрофон), «Speaker N» — собеседники (системный звук), метки по голосу.", "",
        "## Полный транскрипт", "",
    ]
    next_marker = marker_every_s
    for r in sorted(records, key=lambda r: r.start):
        while r.start >= next_marker:
            lines.append(f"— {hhmm(next_marker)} —")
            lines.append("")
            next_marker += marker_every_s
        text = r.text if r.text is not None else f"[не распознано {mmss(r.start)}–{mmss(r.end)}]"
        if not text:
            continue
        lines.append(f"{r.speaker}: {text}")
        lines.append("")
    return "\n".join(lines)


def render_stub(meta: Meta, raw_stem: str) -> str:
    date, hhmm = stamp(meta)
    title = clean_title(title_of(meta))
    participants = ", ".join(["Владислав"] + [p for p in meta.participants if p != "Владислав"])
    return "\n".join([
        "---", "type: meeting", f"id: MTG-{date}-{hhmm}", f"title: {title}", f"date: {date}",
        f"participants: [{participants}]", "projects: []", f'transcript: "[[{raw_stem}]]"', "tags: [meeting, stub]", "---", "",
        f"# {title}", "",
        f"**Дата:** {date} {meta.started_at.strftime('%H:%M')} · **Участники:** {participants}", "",
        "## Мысли Владислава", "", "", "<!-- дистиллят: ожидает дейли-рутин -->", "",
    ])


def raw_path(cfg: Config, meta: Meta) -> Path:
    return cfg.vault_root / cfg.raw_subdir / raw_filename(meta)


def note_path(cfg: Config, meta: Meta) -> Path:
    return cfg.vault_root / cfg.meetings_subdir / note_filename(meta)


def write_raw(cfg: Config, session: Session, status_text: str) -> Path:
    st = session.state
    n_speakers = len(set(st.speaker_of_emb)) if st.speaker_of_emb else 0
    p = raw_path(cfg, session.meta)
    p.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(p, render_raw(session.meta, st.records, n_speakers, status_text, cfg.marker_every_s))
    st.raw_path = str(p)
    return p


def ensure_stub(cfg: Config, session: Session) -> Path:
    p = note_path(cfg, session.meta)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(p, render_stub(session.meta, raw_path(cfg, session.meta).stem))
    session.state.note_path = str(p)
    return p
