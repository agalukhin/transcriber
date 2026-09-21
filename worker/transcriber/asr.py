from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

PROMPT_MAX = 600  # chars; whisper keeps ~224 tokens of prompt
GLOSSARY_MAX = 12       # чем длиннее промпт, тем охотнее модель пересказывает его вместо речи
NO_SPEECH_MAX = 0.6     # сегмент с такой вероятностью тишины — не речь, а домысел модели
LOGPROB_MIN = -1.2      # и сегмент, в который модель сама не верит


def clean_text(text: str, hallucinations: list[str]) -> str:
    out = text
    for h in hallucinations:
        out = re.sub(re.escape(h) + r"[\s.…!,;:—-]*", "", out, flags=re.IGNORECASE)
    out = re.sub(r"\s*[,;:]\s*(?=[,;:.]|$)", "", out)
    return re.sub(r"\s+", " ", out).strip()


def build_prompt(participants: list[str], glossary: list[str], include_participants: bool = False) -> str:
    """Короткий промпт из терминов, без ярлыков: заголовки вроде «Термины:» whisper принимает за начало
    реплики и продолжает их («Термины. Термины…», «Термина, Слава, …»). Имена — только когда
    участники действительно известны."""
    parts = []
    if include_participants and participants:
        names = ["Владислав"] + [p for p in participants if p and p != "Владислав"]
        parts.append(f"{', '.join(names)}.")
    if glossary:
        parts.append(f"{', '.join(glossary[:GLOSSARY_MAX])}.")
    prompt = " ".join(parts)
    return prompt if len(prompt) <= PROMPT_MAX else prompt[:PROMPT_MAX - 1].rsplit(",", 1)[0] + "."


@dataclass
class Segment:
    """Кусок блока, как его нарезал whisper. Секунды — от начала блока, не от начала встречи."""
    start: float
    end: float
    text: str


def kept_segments(result: dict) -> list[dict]:
    """Сегменты без тех, которым сама модель не верит."""
    return [s for s in (result.get("segments") or [])
            if float(s.get("no_speech_prob", 0.0) or 0.0) <= NO_SPEECH_MAX
            and float(s.get("avg_logprob", 0.0) or 0.0) >= LOGPROB_MIN]


class Transcriber:
    def __init__(self, model_name: str, language: str, hallucinations: list[str], sample_rate: int = 16000):
        import mlx_whisper

        self.mlx_whisper = mlx_whisper
        self.model_name = model_name
        self.language = language
        self.hallucinations = hallucinations
        self.sample_rate = sample_rate

    def __call__(self, audio: np.ndarray, prompt: str = "") -> list[Segment]:
        """Блок на 45-90 с → сегменты с временами внутри блока.

        ``condition_on_previous_text`` включён: внутри блока это тот самый контекст,
        ради которого реплики и склеиваются, а утечка предыдущего текста ограничена блоком.
        """
        result = self.mlx_whisper.transcribe(
            np.ascontiguousarray(audio, dtype=np.float32),
            path_or_hf_repo=self.model_name,
            language=self.language,
            initial_prompt=prompt or None,
            fp16=True,
            condition_on_previous_text=True,
        )
        segments = kept_segments(result)
        if not (result.get("segments") or []):
            text = clean_text(result.get("text", ""), self.hallucinations)
            return [Segment(0.0, len(audio) / self.sample_rate, text)] if text else []
        out = []
        for s in segments:
            text = clean_text(s.get("text", ""), self.hallucinations)
            if text:
                out.append(Segment(float(s.get("start", 0.0) or 0.0), float(s.get("end", 0.0) or 0.0), text))
        return out
