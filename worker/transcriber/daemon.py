from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from .config import Config
from .models import Models, load_models
from .pipeline import Pipeline
from .session import Session, lock_holder
from .vault import ensure_stub, write_raw

log = logging.getLogger("transcriber")

FAIL_LIMIT = 5          # столько подряд падений — и сессию откладываем
BACKOFF_START_S = 5.0
BACKOFF_MAX_S = 300.0
DEAD = ("finalized", "failed")


class Daemon:
    def __init__(self, cfg: Config, models: Models | None = None):
        self.cfg = cfg
        self._models = models
        self.pipes: dict[Path, Pipeline] = {}
        self.failures: dict[Path, int] = {}      # подряд идущие падения по каталогу
        self.retry_after: dict[Path, float] = {}

    @property
    def models(self) -> Models:
        if self._models is None:
            log.info("loading models")
            self._models = load_models(self.cfg)
        return self._models

    def once(self) -> bool:
        """One pass over all sessions. Returns True if any session did work."""
        worked = False
        if not self.cfg.sessions_dir.exists():
            return False
        self._forget_vanished()
        for d in sorted(p for p in self.cfg.sessions_dir.iterdir() if (p / "meta.json").exists()):
            if self._sleeping(d):
                continue
            if lock_holder(d) is not None:       # сессию взял CLI (retranscribe и т.п.) — не мешаем
                self.pipes.pop(d, None)
                continue
            try:
                pipe = self.pipes.get(d)
                if pipe is not None and _status_on_disk(d) in DEAD:
                    del self.pipes[d]            # финализировали снаружи (cli finalize)
                    continue
                if pipe is None:
                    session = Session.load(d)
                    if session.state.status in DEAD:
                        continue
                    if session.state.raw_path is None:
                        # заметка и Raw должны существовать до первой загрузки моделей (10+ с),
                        # иначе «открыть заметку» в приложении бьётся в пустой note_path
                        ensure_stub(self.cfg, session)
                        write_raw(self.cfg, session, "идёт запись")
                        session.save_state()
                    m = self.models
                    pipe = Pipeline(self.cfg, session, m.asr, m.embed, m.prob_factory)
                    self.pipes[d] = pipe
                worked |= pipe.step() or pipe.has_backlog()    # бэклог читается порциями
                if pipe.s.stop_requested():
                    log.info("finalizing %s", d.name)
                    pipe.finalize()
                    del self.pipes[d]
                    worked = True
                    if not self.pipes:
                        self.models.release()    # встреч больше нет — кеш буферов MLX отдаём системе
                self.failures.pop(d, None)
                self.retry_after.pop(d, None)
            except Exception:  # noqa: BLE001 — one broken session must not stop the others
                self._note_failure(d)
        return worked

    # ---- отказоустойчивость ----------------------------------------------

    def _forget_vanished(self) -> None:
        for d in [p for p in self.pipes if not (p / "meta.json").exists()]:
            log.info("session %s vanished, dropping", d.name)
            del self.pipes[d]

    def _sleeping(self, d: Path) -> bool:
        return self.failures.get(d, 0) >= FAIL_LIMIT and time.monotonic() < self.retry_after.get(d, 0.0)

    def _note_failure(self, d: Path) -> None:
        self.pipes.pop(d, None)
        n = self.failures.get(d, 0) + 1
        self.failures[d] = n
        if n < FAIL_LIMIT:
            log.exception("session %s failed (%d/%d)", d.name, n, FAIL_LIMIT)
            return
        delay = min(BACKOFF_MAX_S, BACKOFF_START_S * 2 ** (n - FAIL_LIMIT))
        self.retry_after[d] = time.monotonic() + delay
        log.warning("session %s failed %d times, skipping for %.0fs", d.name, n, delay)
        self._mark_failed(d)

    def _mark_failed(self, d: Path) -> None:
        """Если state.json читается — помечаем сессию failed, чтобы once() обходил её как finalized."""
        try:
            session = Session.load(d)
        except Exception:  # noqa: BLE001 — сломанная meta.json: пометить негде, хватит бэкоффа
            return
        if session.state.status not in DEAD:
            session.state.status = "failed"
            session.save_state()

    def run_forever(self) -> None:
        log.info("worker started, sessions at %s", self.cfg.sessions_dir)
        while True:
            worked = self.once()
            time.sleep(0.2 if worked else 1.0)


def _status_on_disk(d: Path) -> str:
    try:
        return str(json.loads((d / "state.json").read_text()).get("status", ""))
    except Exception:  # noqa: BLE001 — нечитаемый state.json разберёт обычный путь обработки
        return ""
