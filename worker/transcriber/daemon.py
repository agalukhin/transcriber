from __future__ import annotations

import logging
import time
from pathlib import Path

from .config import Config
from .models import Models, load_models
from .pipeline import Pipeline
from .session import Session

log = logging.getLogger("transcriber")


class Daemon:
    def __init__(self, cfg: Config, models: Models | None = None):
        self.cfg = cfg
        self._models = models
        self.pipes: dict[Path, Pipeline] = {}

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
        for d in sorted(p for p in self.cfg.sessions_dir.iterdir() if (p / "meta.json").exists()):
            try:
                pipe = self.pipes.get(d)
                if pipe is None:
                    session = Session.load(d)
                    if session.state.status == "finalized":
                        continue
                    m = self.models
                    pipe = Pipeline(self.cfg, session, m.asr, m.embed, m.prob_factory)
                    self.pipes[d] = pipe
                worked |= pipe.step()
                if pipe.s.stop_requested():
                    log.info("finalizing %s", d.name)
                    pipe.finalize()
                    del self.pipes[d]
                    worked = True
            except Exception:  # noqa: BLE001 — one broken session must not stop the others
                log.exception("session %s failed", d.name)
                self.pipes.pop(d, None)
        return worked

    def run_forever(self) -> None:
        log.info("worker started, sessions at %s", self.cfg.sessions_dir)
        while True:
            worked = self.once()
            time.sleep(0.2 if worked else 1.0)
