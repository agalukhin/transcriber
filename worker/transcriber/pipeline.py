from __future__ import annotations

import logging
from datetime import datetime
from typing import Callable

import numpy as np

from .asr import build_prompt
from .config import Config
from .echo import is_echo
from .pcm import PcmReader
from .session import Rec, Session
from .speakers import OnlineClusterer, label, recluster
from .vad import Segmenter, Utterance
from .vault import ensure_stub, write_raw

log = logging.getLogger("transcriber")

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
        self.segs = {
            ch: Segmenter(prob_factory(), cfg.sample_rate, cfg.vad_threshold, cfg.min_silence_ms, cfg.speech_pad_ms,
                          cfg.max_utterance_s, start_sample=off)
            for ch, off in (("mic", st.mic_offset), ("sys", st.sys_offset))
        }
        self.clusterer = OnlineClusterer.from_history(cfg.speaker_similarity, st.embeddings, st.speaker_of_emb)
        self.prompt = build_prompt(session.meta.participants, cfg.glossary)
        self.last_sys_speaker = next((r.speaker for r in reversed(st.records) if r.channel == "sys"), label(0))
        ensure_stub(cfg, session)
        if st.raw_path is None:
            write_raw(cfg, session, "идёт запись")
            session.save_state()

    # ---- live -------------------------------------------------------------

    def step(self) -> bool:
        changed = False
        for ch in ("mic", "sys"):
            audio = self.readers[ch].read_all()
            if audio.size:
                for u in self.segs[ch].feed(audio):
                    self._handle(ch, u)
                    changed = True
        changed |= self._flush_pending_mic(force=False)
        self.s.state.mic_offset = self.segs["mic"].boundary()
        self.s.state.sys_offset = self.segs["sys"].boundary()
        if changed:
            write_raw(self.cfg, self.s, "идёт запись")
        self.s.save_state()
        return changed

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

    def _handle(self, ch: str, u: Utterance) -> None:
        text = self._transcribe(u)
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
        horizon = self._sec(seg.pos)
        open_start = self._sec(seg.open_start) if seg.open_start is not None else None
        sys_recs = [r for r in st.records if r.channel == "sys"]
        keep: list[Rec] = []
        changed = False
        for rec in st.pending_mic:
            ready = force or (horizon >= rec.end + self.cfg.echo_wait_s and (open_start is None or open_start > rec.end))
            if not ready:
                keep.append(rec)
                continue
            if is_echo(rec, sys_recs, self.cfg.echo_overlap, self.cfg.echo_similarity):
                log.info("echo dropped %.1f-%.1f: %s", rec.start, rec.end, rec.text)
            else:
                st.records.append(rec)
            changed = True
        st.pending_mic = keep
        return changed

    # ---- finalization -----------------------------------------------------

    def finalize(self) -> None:
        while self.step():
            pass
        self._flush_pending_mic(force=True)
        st = self.s.state
        if len(st.embeddings) >= 2:
            new = recluster(np.asarray(st.embeddings, dtype=np.float32), self.cfg.speaker_similarity)
            st.speaker_of_emb = new
            last = label(0)
            for r in sorted(st.records, key=lambda r: r.start):
                if r.channel != "sys":
                    continue
                if r.emb is not None:
                    r.speaker = label(new[r.emb])
                    last = r.speaker
                else:
                    r.speaker = last
        st.status = "finalized"
        write_raw(self.cfg, self.s, f"финализирован {datetime.now().strftime('%H:%M')}")
        self.s.save_state()
        if not self.cfg.keep_audio:
            for p in (self.s.mic_pcm, self.s.sys_pcm):
                p.unlink(missing_ok=True)
