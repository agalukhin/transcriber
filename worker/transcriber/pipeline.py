from __future__ import annotations

import logging
from datetime import datetime
from typing import Callable

import numpy as np

from .aec import EchoSuppressor
from .asr import Segment, build_prompt
from .blocks import Block, BlockBuilder
from .config import Config
from .echo import is_acoustic_residue, is_echo, is_residue
from .pcm import PcmReader, read_range
from .quality import junk_reason, prompt_vocabulary, strip_leak_prefix
from .session import Rec, Session, State
from .speakers import OnlineClusterer, absorb_small_clusters, label, recluster
from .vad import Segmenter
from .vault import ensure_stub, write_raw
from .voices import VoiceLibrary, centroid_of

log = logging.getLogger("transcriber")

PROMPT_LEAK = "prompt leak"     # имя правила из quality.junk_reason, которое лечится повтором
CARRY_MAX = 200                 # столько символов прошлого блока канала уносим в следующий промпт
KEPT_MIN_ENERGY = 1e-8          # тише — сырой микрофон почти молчит, доля энергии ничего не значит
KEPT_MIN_S = 0.1                # короче — доля по паре кадров STFT, ей верить нельзя


def name_clusters(cfg: Config, st: State, library: VoiceLibrary | None = None) -> dict[int, str]:
    """Сверить центроиды кластеров с библиотекой голосов: {индекс кластера: имя}.

    Кластер короче ``voice_min_samples`` не именуем — по одной-двум репликам отпечаток врёт.
    Одно имя достаётся одному кластеру: при споре побеждает большее сходство, проигравший
    остаётся «Speaker N». Ошибиться именем хуже, чем не опознать.
    """
    lib = library if library is not None else VoiceLibrary(cfg.voices_dir, cfg.voice_match_similarity)
    st.speaker_names = {}
    if not lib.names():
        return {}
    by_cluster: dict[int, list[list[float]]] = {}
    for i, cluster in enumerate(st.speaker_of_emb[:len(st.embeddings)]):
        by_cluster.setdefault(int(cluster), []).append(st.embeddings[i])
    claims: list[tuple[float, int, str]] = []
    for cluster, embs in by_cluster.items():
        if len(embs) < cfg.voice_min_samples:
            continue
        centroid = centroid_of(embs)
        hit = lib.match(centroid) if centroid else None
        if hit is not None:
            claims.append((hit[1], cluster, hit[0]))
    taken: set[str] = set()
    mapping: dict[int, str] = {}
    for similarity, cluster, name in sorted(claims, key=lambda c: (-c[0], c[1])):
        if name in taken:
            log.info("voices: %s уже занят более похожим кластером — %s остаётся без имени",
                     name, label(cluster))
            continue
        taken.add(name)
        mapping[cluster] = name
        log.info("voices: %s → %s (%.3f, %d реплик)", label(cluster), name, similarity,
                 len(by_cluster[cluster]))
    st.speaker_names = {str(c): mapping[c] for c in sorted(mapping)}
    return mapping


def recluster_records(cfg: Config, st: State, library: VoiceLibrary | None = None,
                      private: bool = False) -> None:
    """Глобальная перекластеризация по всем эмбеддингам сессии + перелейблинг sys-реплик.

    Общая для finalize() и `transcriber refinalize`, поэтому вынесена из класса.
    После перекластеризации кластеры сверяются с библиотекой голосов: опознанные получают
    имя человека вместо «Speaker N», индекс кластера остаётся в ``speaker_of_emb``.

    У приватной записи имён нет вовсе: библиотека не читается, все остаются «Speaker N» —
    приватный разговор не должен опознаваться по рабочим голосам и попадать в библиотеку.
    """
    if len(st.embeddings) >= 2:
        new = recluster(np.asarray(st.embeddings, dtype=np.float32), cfg.speaker_similarity)
        st.speaker_of_emb = absorb_small_clusters(new, st.embeddings)
    if private:
        st.speaker_names = {}
    names: dict[int, str] = {} if private else name_clusters(cfg, st, library)
    if not st.embeddings:
        return

    def speaker_of(emb: int) -> str:
        cluster = int(st.speaker_of_emb[emb])
        return names.get(cluster) or label(cluster)

    sys_recs = [r for r in sorted(st.records, key=lambda r: r.start) if r.channel == "sys"]
    first_emb = next((r for r in sys_recs if r.emb is not None), None)
    # короткая реплика до первой опознанной наследует её метку, а не «Speaker 1» вслепую
    last = speaker_of(first_emb.emb) if first_emb is not None else label(0)
    for r in sys_recs:
        if r.emb is not None:
            r.speaker = speaker_of(r.emb)
            last = r.speaker
        else:
            r.speaker = last


Asr = Callable[[np.ndarray, str], list[Segment]]
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
        self.builders: dict[str, BlockBuilder] = {
            ch: BlockBuilder(cfg.sample_rate, cfg.block_min_s, cfg.block_max_s,
                             cfg.block_gap_s, cfg.block_gap_keep_s, cfg.block_idle_s)
            for ch in ("mic", "sys")}
        # подавитель живёт вместе с конвейером: после перезапуска воркера он стартует холодным и
        # первые секунды записи идут без подавления, пока не наберётся статистика тракта
        self.aec = EchoSuppressor(cfg.sample_rate, beta=cfg.echo_beta,
                                  floor=cfg.echo_floor) if cfg.echo_suppress else None
        self._finalizing = False
        self.carry = {"mic": "", "sys": ""}   # хвост прошлого блока канала; после перезапуска пуст
        self.clusterer = OnlineClusterer.from_history(cfg.speaker_similarity, st.embeddings, st.speaker_of_emb)
        self.prompt = build_prompt(session.meta.participants, cfg.glossary,
                                   include_participants=bool(session.meta.participants))
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
            audio = self._read(ch, cap)   # бэклог за сутки не тянем в память целиком
            blocks: list[Block] = []
            if audio.size:
                for u in self.segs[ch].feed(audio):
                    blocks.extend(self.builders[ch].add(u))
            # позиция чтения — «сколько времени уже прошло»: по ней блок закрывается по простою
            blocks.extend(self.builders[ch].idle(self.readers[ch].offset))
            for b in blocks:
                self._handle_block(ch, b)
                changed = True
        changed |= self._flush_pending_mic(force=False)
        st = self.s.state
        offsets = (self._resume_point("mic"), self._resume_point("sys"))
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

    def _read(self, ch: str, cap: int) -> np.ndarray:
        """Порция канала. Микрофон при включённом подавлении читается только вместе с опорой.

        Опора — тот же отрезок дорожки динамиков. Если sys ещё не дописан (живая запись, mic
        успел раньше), хвост микрофона возвращаем читателю и заберём следующим шагом: пускать
        микрофон через подавитель без опоры нельзя — он примет эхо за речь владельца. На
        финализации (и когда дорожки динамиков нет вовсе) недостающее добиваем тишиной,
        иначе хвост записи завис бы навсегда.
        """
        reader = self.readers[ch]
        if ch != "mic" or self.aec is None:
            return reader.read_all(max_samples=cap)
        start = reader.offset
        audio = reader.read_all(max_samples=cap)
        if not audio.size:
            return audio
        ref = read_range(self.s.sys_pcm, start, start + len(audio))
        if len(ref) < len(audio):
            if self._finalizing or not self.s.sys_pcm.exists():
                ref = np.concatenate([ref, np.zeros(len(audio) - len(ref), dtype=np.float32)])
            else:
                reader.offset = start + len(ref)
                audio = audio[:len(ref)]
                if not audio.size:
                    return audio
        return self.aec.process(audio, ref[:len(audio)])

    def _resume_point(self, ch: str) -> int:
        """Точка продолжения после падения: начало открытого блока, иначе граница сегментатора.

        Заново распознаётся максимум один блок, и дубля не будет: всё, что было до этой точки,
        уже лежит в records.
        """
        open_start = self.builders[ch].open_start
        return open_start if open_start is not None else self.segs[ch].boundary()

    def _sec(self, samples: int) -> float:
        return samples / self.cfg.sample_rate

    def _prompt_for(self, ch: str) -> str:
        """Термины плюс хвост прошлого блока этого канала — штатный способ whisper тянуть контекст."""
        tail = self.carry.get(ch, "")
        if not (self.cfg.carry_context and tail):
            return self.prompt
        return f"{self.prompt} {tail[-CARRY_MAX:]}".strip()

    def _transcribe(self, block: Block, prompt: str) -> list[Segment] | None:
        try:
            return self.asr(block.audio, prompt)
        except Exception as e:  # noqa: BLE001 — any model failure becomes a marker in the transcript
            msg = f"asr failed at {self._sec(block.start):.1f}-{self._sec(block.end):.1f}: {e!r}"
            log.warning(msg)
            self.s.state.errors.append(msg)
            return None

    def _reason(self, text: str | None) -> str | None:
        return junk_reason(text, self.prompt_words, self.cfg.language) if text else None

    def _retry_leaky_block(self, block: Block, segments: list[Segment], prompt: str) -> list[Segment]:
        """Если промпт пересказан в половине сегментов и больше — блок переспрашиваем целиком без промпта.

        Под пересказом обычно живая речь, а решать посегментно нельзя: без промпта модель
        нарежет блок иначе, и склеить две нарезки в одну ленту не выйдет.
        """
        if not (prompt and segments):
            return segments
        leaks = sum(1 for s in segments if self._reason(s.text) == PROMPT_LEAK)
        if leaks * 2 < len(segments):
            return segments
        log.info("retry without prompt %.1f-%.1f: %s", self._sec(block.start), self._sec(block.end),
                 segments[0].text[:60])
        retry = self._transcribe(block, "")
        return retry if retry is not None else segments

    def _handle_block(self, ch: str, block: Block) -> None:
        prompt = self._prompt_for(ch)
        segments = self._transcribe(block, prompt)
        if segments is None:
            self._place(ch, Rec(channel=ch, start=self._sec(block.start), end=self._sec(block.end),
                                text=None, speaker="Me", emb=None), block, None)
            return
        kept: list[str] = []
        for seg in self._retry_leaky_block(block, segments, prompt):
            start = block.map_to_original(seg.start)
            end = max(start, block.map_to_original(seg.end))
            text = strip_leak_prefix(seg.text)   # приставка-утечка срезается, речь под ней остаётся
            reason = self._reason(text)
            if reason:
                # Галлюцинация — это не «не распознано», а пустая реплика: в транскрипте её нет.
                log.info("dropped (%s) %.1f-%.1f: %s", reason, start, end, text)
                text = ""
            else:
                kept.append(text)
            rec = Rec(channel=ch, start=start, end=end, text=text, speaker="Me", emb=None)
            if ch == "mic" and self.aec is not None:
                rec.kept = self._kept(block, seg, start, end)
            self._place(ch, rec, block, seg)
        if self.cfg.carry_context:
            self.carry[ch] = " ".join(kept)[-CARRY_MAX:]

    def _place(self, ch: str, rec: Rec, block: Block, seg: Segment | None) -> None:
        """Реплику микрофона — в очередь на эхо-проверку, реплику собеседников — с меткой говорящего."""
        if ch == "mic":
            self.s.state.pending_mic.append(rec)
            return
        st = self.s.state
        audio = self._segment_audio(block, seg) if seg is not None else np.zeros(0, dtype=np.float32)
        if len(audio) >= int(self.cfg.min_embed_s * self.cfg.sample_rate):
            emb = self.embed(audio)
            idx = self.clusterer.assign(emb)
            st.embeddings.append([float(x) for x in emb])
            st.speaker_of_emb.append(idx)
            rec.emb = len(st.embeddings) - 1
            rec.speaker = label(idx)
            self.last_sys_speaker = rec.speaker
        else:
            rec.speaker = self.last_sys_speaker
        st.records.append(rec)

    def _kept(self, block: Block, seg: Segment, start: float, end: float) -> float | None:
        """Доля энергии микрофона, которую подавитель эха оставил на отрезке реплики.

        Очищенный звук — из склейки блока (координаты блока), сырой — с диска по исходному
        отрезку. Выход подавителя совпадает со входом по номерам сэмплов, так что сравнивать
        можно напрямую. Слишком короткая или почти беззвучная реплика доли не получает.
        """
        if end - start < KEPT_MIN_S:
            return None
        sr = self.cfg.sample_rate
        raw = read_range(self.s.mic_pcm, int(start * sr), int(end * sr))
        clean = self._segment_audio(block, seg)
        if not raw.size or not clean.size:
            return None
        raw_energy = float(np.mean(raw.astype(np.float64) ** 2))
        if raw_energy < KEPT_MIN_ENERGY:
            return None
        return round(float(np.mean(clean.astype(np.float64) ** 2)) / raw_energy, 4)

    def _segment_audio(self, block: Block, seg: Segment) -> np.ndarray:
        sr = self.cfg.sample_rate
        return block.audio[max(0, int(seg.start * sr)):int(seg.end * sr)]

    def _sys_open_start(self) -> float | None:
        """Начало самого раннего незакрытого куска sys: до него mic-реплику судить об эхе рано."""
        open_starts = [x for x in (self.builders["sys"].open_start, self.segs["sys"].open_start) if x is not None]
        return self._sec(min(open_starts)) if open_starts else None

    def _flush_pending_mic(self, force: bool) -> bool:
        st = self.s.state
        # горизонт — позиция sys-канала; если sys.pcm нет или он отстаёт, ориентируемся на mic,
        # иначе реплики владельца висят в pending до самой финализации
        horizon = max(self._sec(self.segs["sys"].pos), self._sec(self.segs["mic"].pos) - self.cfg.echo_wait_s)
        open_start = self._sys_open_start()
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
            elif is_residue(rec, sys_recs):
                log.info("echo residue dropped %.1f-%.1f: %s", rec.start, rec.end, rec.text)
            elif is_acoustic_residue(rec, sys_recs, self.cfg.echo_kept_max, self.cfg.echo_kept_min_cover):
                log.info("echo residue (acoustic) dropped %.1f-%.1f kept=%.2f: %s",
                         rec.start, rec.end, rec.kept, rec.text)
            else:
                st.records.append(rec)
            changed = True
        st.pending_mic = keep
        return changed

    # ---- finalization -----------------------------------------------------

    def finalize(self) -> None:
        self._finalizing = True                    # опору под хвост микрофона добиваем тишиной
        while self.step() or self.has_backlog():   # step() читает порциями — дочитываем весь хвост
            pass
        if self.aec is not None:
            for u in self.segs["mic"].feed(self.aec.flush()):   # хвост короче окна STFT
                for b in self.builders["mic"].add(u):
                    self._handle_block("mic", b)
        for ch in ("mic", "sys"):
            for u in self.segs[ch].flush():        # recording may have stopped mid-speech
                for b in self.builders[ch].add(u):
                    self._handle_block(ch, b)
            for b in self.builders[ch].flush():    # и недобранный блок уходит в ASR как есть
                self._handle_block(ch, b)
        self.s.state.mic_offset = self._resume_point("mic")   # flushed audio must not be re-read
        self.s.state.sys_offset = self._resume_point("sys")
        self._flush_pending_mic(force=True)
        st = self.s.state
        recluster_records(self.cfg, st, private=self.s.meta.private)
        st.status = "finalized"
        write_raw(self.cfg, self.s, f"финализирован {datetime.now().strftime('%H:%M')}")
        self.s.save_state()
        if not self.cfg.keep_audio:
            for p in (self.s.mic_pcm, self.s.sys_pcm):
                p.unlink(missing_ok=True)
