"""CLI воркера.

* ``transcriber run`` — демон, следит за каталогом сессий;
* ``transcriber from-wav MIC SYS`` — прогон пары WAV как одной сессии;
* ``transcriber finalize SESSION_ID`` — финализация сессии (нужны аудио и модели);
* ``transcriber refinalize SESSION_ID`` — постобработка уже сохранённого ``state.json``
  без аудио и без ASR: тексты через фильтры галлюцинаций, эхо-фильтр по всем mic-репликам,
  перекластеризация эмбеддингов текущими порогами, перезапись Raw. Нужна, когда пороги
  или фильтры поменялись, а переслушать встречу уже нечем — аудио удалено;
* ``transcriber retranscribe SESSION_ID`` — полный повторный прогон сессии по сохранённому
  ``mic.pcm``/``sys.pcm``: state сбрасывается (Raw и MTG остаются на своих путях) и вся
  дорожка проходит ASR заново с текущим конфигом. Нужна при ``keep_audio: true``, когда
  поменялись промпт, модель или пороги сегментации;
* ``transcriber learn SESSION_ID "Speaker 2=Ярослав Серов" …`` — положить отпечатки голоса
  кластера в библиотеку под именем человека. Метка берётся из финального Raw: ``Speaker N``
  или уже проставленное имя. Вместо пар можно указать ``--from-note ПУТЬ`` — ключ
  ``speakers:`` из фронт-маттера MTG-заметки. После ``learn`` полезен ``refinalize``:
  он переподпишет реплики именами;
* ``transcriber voices`` — что накоплено в библиотеке голосов: имя, число отпечатков, источники.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from .config import Config, load_config
from .daemon import Daemon
from .echo import is_echo
from .models import Models, load_models
from .pipeline import Pipeline, recluster_records
from .quality import junk_reason, prompt_vocabulary
from .speakers import label
from .session import Session, State
from .vault import write_raw
from .voices import VoiceLibrary

log = logging.getLogger("transcriber")


def decode_pcm(raw: bytes, width: int) -> np.ndarray:
    """PCM-сэмплы WAV в float32 [-1, 1]. 8 бит — беззнаковые, 24 бита — три байта little-endian."""
    if width == 1:
        return (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    if width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if width == 3:
        b = np.frombuffer(raw, dtype=np.uint8)
        b = b[: len(b) // 3 * 3].reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        v = np.where(v >= 1 << 23, v - (1 << 24), v)          # знаковое расширение
        return v.astype(np.float32) / float(1 << 23)
    if width == 4:
        return np.frombuffer(raw, dtype="<i4").astype(np.float32) / float(1 << 31)
    raise ValueError(f"unsupported WAV sample width: {width * 8}-bit")


def read_wav(path: Path, target_sr: int) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        sr, ch, width, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    try:
        audio = decode_pcm(raw, width)
    except ValueError as e:
        raise ValueError(f"{path}: {e}") from None
    if ch > 1:
        audio = audio.reshape(-1, ch).mean(axis=1)
    if sr != target_sr:
        from scipy.signal import resample_poly

        g = math.gcd(sr, target_sr)
        audio = resample_poly(audio, target_sr // g, sr // g).astype(np.float32)
    return audio


def from_wav(cfg: Config, mic: Path, sys_: Path, title: str, app: str, models: Models | None = None,
             started_at: datetime | None = None) -> Path:
    started_at = started_at or datetime.now().astimezone()
    base = started_at.strftime("%Y-%m-%d-%H%M%S")
    sid, n = base, 0
    while (cfg.sessions_dir / sid / "meta.json").exists():   # чужую сессию не переписываем
        n += 1
        sid = f"{base}-{n}"
    d = cfg.sessions_dir / sid
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(json.dumps({"id": sid, "started_at": started_at.isoformat(), "app": app,
                                             "title": title, "participants": []}, ensure_ascii=False))
    for name, src in (("mic.pcm", mic), ("sys.pcm", sys_)):
        (d / name).write_bytes((np.clip(read_wav(src, cfg.sample_rate), -1, 1) * 32767).astype(np.int16).tobytes())
    (d / "STOP").touch()
    m = models or load_models(cfg)
    pipe = Pipeline(cfg, Session.load(d), m.asr, m.embed, m.prob_factory)
    pipe.finalize()
    return Path(pipe.s.state.raw_path)


def refinalize(cfg: Config, session_id: str) -> Path:
    """Пересобрать результат сессии из сохранённого state.json — без аудио, без ASR."""
    session = Session.load(cfg.sessions_dir / session_id)
    st = session.state
    vocabulary = prompt_vocabulary(session.meta.participants, cfg.glossary)
    for r in st.records:
        reason = junk_reason(r.text, vocabulary, cfg.language) if r.text else None
        if reason:
            log.info("dropped (%s) %.1f-%.1f: %s", reason, r.start, r.end, r.text)
            r.text = ""
    sys_recs = [r for r in st.records if r.channel == "sys"]
    kept: list = []
    for r in st.records:
        if r.channel == "mic" and is_echo(r, sys_recs, cfg.echo_overlap, cfg.echo_similarity,
                                          cfg.echo_strong_similarity):
            log.info("echo dropped %.1f-%.1f: %s", r.start, r.end, r.text)
            continue
        kept.append(r)
    st.records = kept
    recluster_records(cfg, st)
    p = write_raw(cfg, session, f"финализирован повторно {datetime.now().strftime('%H:%M')}")
    session.save_state()
    return p


def retranscribe(cfg: Config, session_id: str, models: Models | None = None) -> Path:
    """Прогнать сессию заново по сохранённому аудио — с текущим промптом, моделью и порогами."""
    d = cfg.sessions_dir / session_id
    session = Session.load(d)
    if not (session.mic_pcm.exists() or session.sys_pcm.exists()):
        raise ValueError(f"{session_id}: аудио сессии не сохранено — помогает только refinalize")
    st = session.state
    st.status = "recording"                    # raw_path и note_path сохраняем: файлы вольта те же
    st.mic_offset = st.sys_offset = 0
    st.records, st.pending_mic, st.embeddings, st.speaker_of_emb, st.errors = [], [], [], [], []
    session.save_state()
    (d / "STOP").touch()
    m = models or load_models(cfg)
    pipe = Pipeline(cfg, session, m.asr, m.embed, m.prob_factory)
    while pipe.step() or pipe.has_backlog():
        pass
    pipe.finalize()
    return Path(pipe.s.state.raw_path)


SPEAKER_LABEL = re.compile(r"Speaker\s+(\d+)$")


def parse_pair(text: str) -> tuple[str, str]:
    """«Speaker 2=Ярослав Серов» → («Speaker 2», «Ярослав Серов»)."""
    label, sep, name = text.partition("=")
    if not sep or not label.strip() or not name.strip():
        raise ValueError(f'{text!r}: нужна пара вида "Speaker 2=Ярослав Серов" — со знаком =')
    return label.strip(), name.strip()


def speakers_from_note(path: Path) -> dict[str, str]:
    """Ключ ``speakers:`` из фронт-маттера MTG-заметки: {«Speaker 2»: «Ярослав Серов»}."""
    text = Path(path).read_text(encoding="utf-8")
    parts = text.split("---", 2)
    if not text.lstrip().startswith("---") or len(parts) < 3:
        raise ValueError(f"{path}: нет фронт-маттера")
    block = parts[1]
    try:
        data = yaml.safe_load(block) or {}
    except yaml.YAMLError:
        # чужие ключи заметки могут не пережить safe_load — нам нужна ровно одна строка
        line = next((x for x in block.splitlines() if x.startswith("speakers:")), "")
        data = yaml.safe_load(line) if line else {}
    speakers = (data or {}).get("speakers")
    if not isinstance(speakers, dict) or not speakers:
        raise ValueError(f"{path}: во фронт-маттере нет ключа speakers с парами «Speaker N: Имя»")
    return {str(k).strip(): str(v).strip() for k, v in speakers.items()}


def cluster_of_label(st: State, text: str) -> int | None:
    """Метка из финального Raw в индекс кластера: «Speaker N» → N-1, имя → его кластер."""
    m = SPEAKER_LABEL.match(text)
    if m:
        return int(m.group(1)) - 1
    for key, name in st.speaker_names.items():
        if name == text:
            return int(key)
    return None


def learn(cfg: Config, session_id: str, mapping: dict[str, str]) -> list[tuple[str, str, int]]:
    """Отпечатки названных кластеров сессии — в библиотеку голосов.

    Сверка идёт по всей карте до первой записи: половина применённого сопоставления хуже,
    чем понятная ошибка и повтор команды целиком.
    """
    if not mapping:
        raise ValueError("нечего учить: не задана ни одна пара «Speaker N=Имя»")
    st = Session.load(cfg.sessions_dir / session_id).state
    known = sorted({label(c) for c in st.speaker_of_emb} | set(st.speaker_names.values()))
    plan: list[tuple[str, str, list[list[float]]]] = []
    for text, name in mapping.items():
        cluster = cluster_of_label(st, text)
        embs = [st.embeddings[i] for i, c in enumerate(st.speaker_of_emb[:len(st.embeddings)])
                if int(c) == cluster] if cluster is not None else []
        if not embs:
            raise ValueError(f"{session_id}: метки {text!r} в сессии нет; есть {', '.join(known) or '—'}")
        plan.append((text, name, embs))
    lib = VoiceLibrary(cfg.voices_dir, cfg.voice_match_similarity)
    out: list[tuple[str, str, int]] = []
    for text, name, embs in plan:
        prof = lib.add(name, embs, source=f"{session_id}:{text}")
        print(f"{name} ← {text}: +{len(embs)} отпечатков, всего {prof.count}")
        out.append((name, text, len(embs)))
    return out


def show_voices(cfg: Config) -> None:
    lib = VoiceLibrary(cfg.voices_dir, cfg.voice_match_similarity)
    names = lib.names()
    if not names:
        print(f"Библиотека голосов пуста ({cfg.voices_dir})")
        return
    print(f"Библиотека голосов ({cfg.voices_dir}), порог {cfg.voice_match_similarity}:")
    for name in names:
        prof = lib.get(name)
        print(f"  {name}: {prof.count} отпечатков ({len(prof.samples)} хранится) · "
              f"{', '.join(prof.sources) or 'источник не указан'}")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(prog="transcriber")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run", help="daemon: watch sessions dir")
    fw = sub.add_parser("from-wav", help="transcribe two WAV files as one session")
    fw.add_argument("mic")
    fw.add_argument("sys")
    fw.add_argument("--title", default="")
    fw.add_argument("--app", default="manual")
    fz = sub.add_parser("finalize", help="finalize a session by id")
    fz.add_argument("session_id")
    rf = sub.add_parser("refinalize", help="redo post-processing over a saved session (no audio, no ASR)")
    rf.add_argument("session_id")
    rt = sub.add_parser("retranscribe", help="run the whole pipeline again over the session's saved audio")
    rt.add_argument("session_id")
    ln = sub.add_parser("learn", help='teach a voice: learn SESSION_ID "Speaker 2=Ярослав Серов" …')
    ln.add_argument("session_id")
    ln.add_argument("pairs", nargs="*", metavar="SPEAKER=NAME")
    ln.add_argument("--from-note", metavar="PATH", help="взять карту из ключа speakers: фронт-маттера MTG-заметки")
    sub.add_parser("voices", help="list the voice library")
    args = ap.parse_args(argv)
    cfg = load_config()
    if args.cmd == "run":
        Daemon(cfg).run_forever()
    elif args.cmd == "from-wav":
        try:
            print(from_wav(cfg, Path(args.mic), Path(args.sys), args.title, args.app))
        except ValueError as e:
            print(e, file=sys.stderr)
            raise SystemExit(2) from None
    elif args.cmd == "finalize":
        d = cfg.sessions_dir / args.session_id
        (d / "STOP").touch()
        m = load_models(cfg)
        pipe = Pipeline(cfg, Session.load(d), m.asr, m.embed, m.prob_factory)
        pipe.finalize()
        print(pipe.s.state.raw_path)
    elif args.cmd == "refinalize":
        print(refinalize(cfg, args.session_id))
    elif args.cmd == "retranscribe":
        try:
            print(retranscribe(cfg, args.session_id))
        except ValueError as e:
            print(e, file=sys.stderr)
            raise SystemExit(2) from None
    elif args.cmd == "learn":
        try:
            mapping = speakers_from_note(Path(args.from_note)) if args.from_note else {}
            mapping.update(dict(parse_pair(x) for x in args.pairs))
            learn(cfg, args.session_id, mapping)
        except (ValueError, OSError) as e:
            print(e, file=sys.stderr)
            raise SystemExit(2) from None
        print("готово; чтобы имена попали в Raw — transcriber refinalize " + args.session_id)
    elif args.cmd == "voices":
        show_voices(cfg)


if __name__ == "__main__":
    main(sys.argv[1:])
