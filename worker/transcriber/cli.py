from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import Config, load_config
from .daemon import Daemon
from .models import Models, load_models
from .pipeline import Pipeline
from .session import Session


def read_wav(path: Path, target_sr: int) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        sr, ch, width, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    if width != 2:
        raise SystemExit(f"{path}: only 16-bit WAV supported")
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
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
    sid = started_at.strftime("%Y-%m-%d-%H%M%S")
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
    args = ap.parse_args(argv)
    cfg = load_config()
    if args.cmd == "run":
        Daemon(cfg).run_forever()
    elif args.cmd == "from-wav":
        print(from_wav(cfg, Path(args.mic), Path(args.sys), args.title, args.app))
    elif args.cmd == "finalize":
        d = cfg.sessions_dir / args.session_id
        (d / "STOP").touch()
        m = load_models(cfg)
        pipe = Pipeline(cfg, Session.load(d), m.asr, m.embed, m.prob_factory)
        pipe.finalize()
        print(pipe.s.state.raw_path)


if __name__ == "__main__":
    main(sys.argv[1:])
