from __future__ import annotations

import os
import re
from pathlib import Path

from .config import Config
from .session import Meta, Rec, Session, atomic_write_text

APPS: dict[str, tuple[str, str]] = {"kontur.talk": ("Толк", "Толке"), "zoom.us": ("Zoom", "Zoom")}

TITLE_CAP = 120

PRIVATE_DIR_MODE = 0o700        # приватный каталог и файлы видны только владельцу
PRIVATE_FILE_MODE = 0o600
PRIVATE_HEADER = "Приватная запись. Вне рабочего вольта."
THOUGHTS = "## Мысли Владислава"
STUB_TAG = re.compile(r"^tags:.*\bstub\b", re.M)


def app_label(app: str) -> str:
    pair = APPS.get(app)
    return pair[0] if pair else ""


def app_locative(app: str) -> str:
    pair = APPS.get(app)
    return pair[1] if pair else ""


def default_title(meta: Meta) -> str:
    name = app_locative(meta.app)
    return f"Встреча в {name}" if name else "Встреча"


def clean_title(title: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[/\\:|#^\[\]]", " ", title)).strip()


def title_of(meta: Meta) -> str:
    return meta.title.strip() if clean_title(meta.title) else default_title(meta)


def slug(title: str) -> str:
    return re.sub(r"-+", "-", re.sub(r"\s+", "-", clean_title(title))).strip("-")


def stamp(meta: Meta) -> tuple[str, str]:
    return meta.started_at.strftime("%Y-%m-%d"), meta.started_at.strftime("%H%M")


def raw_filename(meta: Meta) -> str:
    date, hhmm = stamp(meta)
    return f"Local-{date}-{hhmm}-{slug(title_of(meta))[:TITLE_CAP]}.md"


def note_filename(meta: Meta) -> str:
    date, hhmm = stamp(meta)
    return f"MTG-{date} {hhmm} · {clean_title(title_of(meta))[:TITLE_CAP]}.md"


def mmss(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 60:02d}:{s % 60:02d}"


def hhmm(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}"


def render_raw(meta: Meta, records: list[Rec], n_speakers: int, status_text: str, marker_every_s: int = 300,
               identified: list[str] | None = None) -> str:
    date, _ = stamp(meta)
    title = title_of(meta)
    who = ", ".join(meta.participants) if meta.participants else "не получены"
    app = app_label(meta.app) or "ручной старт"
    # у приватной записи нет фронт-маттера вольта: файл лежит вне вольта и в граф не попадает
    front = [] if meta.private else [
        "---", "type: raw", "source: local", f"source_date: {date}", "projects: []", "tags: [raw]", "---", ""]
    mark = [PRIVATE_HEADER, ""] if meta.private else []
    lines = [
        *front,
        f"# {title} — {meta.started_at.strftime('%Y-%m-%d %H:%M')}", "",
        *mark,
        f"Локальная запись ({app}). Участники: {who}. "
        f"Говорящих на канале собеседников: {n_speakers}. "
        + (f"Опознаны по голосу: {', '.join(identified)}. " if identified else "")
        + f"Статус: {status_text}. "
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


def render_private_note(meta: Meta) -> str:
    """Заметки к приватной записи — обычный файл: ни фронт-маттера, ни ссылок в вольт."""
    return f"# Заметки — {meta.started_at.strftime('%Y-%m-%d %H:%M')}\n\n"


def is_untouched_stub(text: str) -> bool:
    """Заглушка, в которую ещё не писали: тег ``stub`` на месте, секция мыслей пуста.

    Всё спорное считаем тронутым: снести чужую мысль хуже, чем оставить лишний файл.
    """
    head, sep, body = text.partition(THOUGHTS)
    if not sep or not STUB_TAG.search(head):
        return False
    return not [ln for ln in body.splitlines() if ln.strip() and not ln.strip().startswith("<!--")]


def private_dir(cfg: Config, meta: Meta) -> Path:
    return cfg.private_dir / meta.id


def raw_path(cfg: Config, meta: Meta) -> Path:
    if meta.private:
        return private_dir(cfg, meta) / "transcript.md"
    return cfg.vault_root / cfg.raw_subdir / raw_filename(meta)


def note_path(cfg: Config, meta: Meta) -> Path:
    if meta.private:
        return private_dir(cfg, meta) / "notes.md"
    return cfg.vault_root / cfg.meetings_subdir / note_filename(meta)


def write_text(cfg: Config, meta: Meta, path: Path, text: str) -> None:
    """Запись файла записи. Приватная — в каталог 0700 файлом 0600, и не в вольте.

    ``mkdir(mode=…)`` уважает umask и не трогает уже существующий каталог, поэтому права
    выставляются ``chmod`` после создания — и на корень приватного каталога тоже.
    """
    if not meta.private:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, text)
        return
    for d in (cfg.private_dir, path.parent):
        d.mkdir(parents=True, exist_ok=True)
        os.chmod(d, PRIVATE_DIR_MODE)
    atomic_write_text(path, text)
    os.chmod(path, PRIVATE_FILE_MODE)


def pinned_raw_path(cfg: Config, session: Session) -> Path:
    """Путь фиксируется первой записью: смена meta.title на ходу не должна плодить второй файл."""
    pinned = session.state.raw_path
    if pinned and Path(pinned).exists():
        return Path(pinned)
    return raw_path(cfg, session.meta)


def write_raw(cfg: Config, session: Session, status_text: str) -> Path:
    st = session.state
    sys_speakers = {r.speaker for r in st.records if r.channel == "sys"}
    n_speakers = max(len(set(st.speaker_of_emb)), len(sys_speakers))
    identified = [st.speaker_names[k] for k in sorted(st.speaker_names, key=lambda x: (len(x), x))]
    p = pinned_raw_path(cfg, session)
    text = render_raw(session.meta, st.records, n_speakers, status_text, cfg.marker_every_s, identified)
    write_text(cfg, session.meta, p, text)
    st.raw_path = str(p)
    return p


def ensure_stub(cfg: Config, session: Session) -> Path:
    """Заметка рядом с транскриптом: MTG-заглушка в вольте, у приватной записи — простой notes.md."""
    meta = session.meta
    pinned = session.state.note_path
    p = Path(pinned) if pinned and Path(pinned).exists() else note_path(cfg, meta)
    if not p.exists():
        text = render_private_note(meta) if meta.private else \
            render_stub(meta, pinned_raw_path(cfg, session).stem)
        write_text(cfg, meta, p, text)
    session.state.note_path = str(p)
    return p
