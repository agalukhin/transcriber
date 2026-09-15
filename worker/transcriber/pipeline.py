from __future__ import annotations

import logging
from datetime import datetime
from typing import Callable

import numpy as np

from .asr import build_prompt
from .config import Config
from .echo import is_echo
from .pcm import PcmReader, read_range
from .quality import junk_reason, prompt_vocabulary
from .session import Rec, Session, State
from .speakers import OnlineClusterer, absorb_small_clusters, label, recluster
from .vad import Segmenter, Utterance
from .vault import ensure_stub, write_raw

log = logging.getLogger("transcriber")


def recluster_records(cfg: Config, st: State) -> None:
    """Глобальная перекластеризация по всем эмбеддингам сессии + перелейблинг sys-реплик.

    Общая для finalize() и `transcriber refinalize`, поэтому вынесена из класса.
    """
    if len(st.embeddings) < 2:
        return
    new = recluster(np.asarray(st.embeddings, dtype=np.float32), cfg.speaker_similarity)
    new = absorb_small_clusters(new, st.embeddings)
    st.speaker_of_emb = new
    sys_recs = [r for r in sorted(st.records, key=lambda r: r.start) if r.channel == "sys"]
    first_emb = next((r for r in sys_recs if r.emb is not None), None)
    # короткая реплика до первой опознанной наследует её метку, а не «Speaker 1» вслепую
    last = label(new[first_emb.emb]) if first_emb is not None else label(0)
    for r in sys_recs:
        if r.emb is not None:
            r.speaker = label(new[r.emb])
            last = r.speaker
        else:
            r.speaker = last

Asr = Callable[[np.ndarray, str], str]
Embed = Callable[[np.ndarray], np.ndarray]
ProbFactory = Callable[[], Callable[[np.ndarray], float]]


class Pipeline:
    def __init__(self, cfg: Config, session: Session, asr: Asr, embed: Embed, prob_factory: ProbFactory):
        self.cfg = cfg
        self.s = session
        self.asr = asr
        self.embed = embed
        st = session.state
        self.readers = {"mic": PcmReader(session.mic_pcm, st.mic_offset), "sys": PcmReader(session.sys_pcm, st.sys_offset)}
        pad = int(cfg.sample_rate * cfg.speech_pad_ms / 1000)
        self.segs: dict[str, Segmenter] = {}
        for ch, off, pcm in (("mic", st.mic_offset, session.mic_pcm), ("sys", st.sys_offset, session.sys_pcm)):
            # on resume the pad before the offset is already on disk — otherwise the first
            # utterance after a restart starts up to speech_pad_ms late
            preroll = read_range(pcm, off - pad, off) if off > 0 and pad > 0 else None
            self.segs[ch] = Segmenter(prob_factory(), cfg.sample_rate, cfg.vad_threshold, cfg.min_silence_ms,
                                      cfg.speech_pad_ms, cfg.max_utterance_s, start_sample=off, preroll=preroll)
        self.clusterer = OnlineClusterer.from_history(cfg.speaker_similarity, st.embeddings, st.speaker_of_emb)
        self.prompt = build_prompt(session.meta.participants, cfg.glossary)
        self.prompt_words = prompt_vocabulary(session.meta.participants, cfg.glossary)
        self.last_sys_speaker = next((r.speaker for r in reversed(st.records) if r.channel == "sys"), label(0))
        ensure_stub(cfg, session)
        if st.raw_path is None:
            write_raw(cfg, session, "идёт запись")
            session.save_state()

    # ---- live -------------------------------------------------------------

    def step(self) -> bool:
        changed = False
        cap = max(1, int(self.cfg.sample_rate * self.cfg.read_chunk_s))
        for ch in ("mic", "sys"):
            audio = self.readers[ch].read_all(max_samples=cap)   # бэклог за сутки не тянем в память целиком
            if audio.size:
                for u in self.segs[ch].feed(audio):
                    self._handle(ch, u)
                    changed = True
        changed |= self._flush_pending_mic(force=False)
        st = self.s.state
        offsets = (self.segs["mic"].boundary(), self.segs["sys"].boundary())
        moved = offsets != (st.mic_offset, st.sys_offset)
        st.mic_offset, st.sys_offset = offsets
        if changed:
            write_raw(self.cfg, self.s, "идёт запись")
        if changed or moved:
            self.s.save_state()
        return changed

    def has_backlog(self) -> bool:
        """Есть ли непрочитанное аудио — step() читает порциями, демон должен продолжать крутить."""
        return any(r.available() > 0 for r in self.readers.values())

    def _sec(self, samples: int) -> float:
        return samples / self.cfg.sample_rate

    def _transcribe(self, u: Utterance) -> str | None:
        try:
            return self.asr(u.audio, self.prompt)
        except Exception as e:  # noqa: BLE001 — any model failure becomes a marker in the transcript
            msg = f"asr failed at {self._sec(u.start):.1f}-{self._sec(u.end):.1f}: {e!r}"
            log.warning(msg)
            self.s.state.errors.append(msg)
            return None

    def _drop_junk(self, text: str | None, u: Utterance) -> str | None:
        """Галлюцинация — это не «не распознано», а пустая реплика: в транскрипте её нет."""
        reason = junk_reason(text, self.prompt_words, self.cfg.language) if text else None
        if reason is None:
            return text
        log.info("dropped (%s) %.1f-%.1f: %s", reason, self._sec(u.start), self._sec(u.end), text)
        return ""

    def _handle(self, ch: str, u: Utterance) -> None:
        text = self._drop_junk(self._transcribe(u), u)
        rec = Rec(channel=ch, start=self._sec(u.start), end=self._sec(u.end), text=text, speaker="Me", emb=None)
        if ch == "mic":
            self.s.state.pending_mic.append(rec)
            return
        st = self.s.state
        if self._sec(u.samples) >= self.cfg.min_embed_s:
            emb = self.embed(u.audio)
            idx = self.clusterer.assign(emb)
            st.embeddings.append([float(x) for x in emb])
            st.speaker_of_emb.append(idx)
            rec.emb = len(st.embeddings) - 1
            rec.speaker = label(idx)
            self.last_sys_speaker = rec.speaker
        else:
            rec.speaker = self.last_sys_speaker
        st.records.append(rec)

    def _flush_pending_mic(self, force: bool) -> bool:
        st = self.s.state
        seg = self.segs["sys"]
        # горизонт — позиция sys-канала; если sys.pcm нет или он отстаёт, ориентируемся на mic,
        # иначе реплики владельца висят в pending до самой финализации
        horizon = max(self._sec(seg.pos), self._sec(self.segs["mic"].pos) - self.cfg.echo_wait_s)
        open_start = self._sec(seg.open_start) if seg.open_start is not None else None
        sys_recs = [r for r in st.records if r.channel == "sys"]
        keep: list[Rec] = []
        changed = False
        for rec in st.pending_mic:
            ready = force or (horizon >= rec.end + self.cfg.echo_wait_s and (open_start is None or open_start > rec.end))
            if not ready:
                keep.append(rec)
                continue
            if is_echo(rec, sys_recs, self.cfg.echo_overlap, self.cfg.echo_similarity,
                       self.cfg.echo_strong_similarity):
                log.info("echo dropped %.1f-%.1f: %s", rec.start, rec.end, rec.text)
            else:
                st.records.append(rec)
            changed = True
        st.pending_mic = keep
        return changed

    # ---- finalization -----------------------------------------------------

    def finalize(self) -> None:
        while self.step() or self.has_backlog():   # step() читает порциями — дочитываем весь хвост
            pass
        for ch in ("mic", "sys"):
            for u in self.segs[ch].flush():       # recording may have stopped mid-speech
                self._handle(ch, u)
        self.s.state.mic_offset = self.segs["mic"].boundary()   # flushed audio must not be re-read
        self.s.state.sys_offset = self.segs["sys"].boundary()
        self._flush_pending_mic(force=True)
        st = self.s.state
        recluster_records(self.cfg, st)
        st.status = "finalized"
        write_raw(self.cfg, self.s, f"финализирован {datetime.now().strftime('%H:%M')}")
        self.s.save_state()
        if not self.cfg.keep_audio:
            for p in (self.s.mic_pcm, self.s.sys_pcm):
                p.unlink(missing_ok=True)
