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


if __name__ == "__main__":
    main(sys.argv[1:])
