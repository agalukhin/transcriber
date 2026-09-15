from __future__ import annotations

import re

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
    """Короткий промпт из терминов: имена whisper охотнее всего пересказывает вместо речи,
    поэтому «Участники: …» добавляется, только когда участники действительно известны."""
    parts = []
    if include_participants and participants:
        names = ["Владислав"] + [p for p in participants if p and p != "Владислав"]
        parts.append(f"Участники: {', '.join(names)}.")
    if glossary:
        parts.append(f"Термины: {', '.join(glossary[:GLOSSARY_MAX])}.")
    prompt = " ".join(parts)
    return prompt if len(prompt) <= PROMPT_MAX else prompt[:PROMPT_MAX - 1].rsplit(",", 1)[0] + "."


def join_segments(result: dict) -> str:
    """Текст без сегментов, которым сама модель не верит; если чистить нечего — исходный text."""
    segments = result.get("segments") or []
    text = result.get("text", "")
    if not segments:
        return text
    kept = [s for s in segments
            if float(s.get("no_speech_prob", 0.0) or 0.0) <= NO_SPEECH_MAX
            and float(s.get("avg_logprob", 0.0) or 0.0) >= LOGPROB_MIN]
    return text if len(kept) == len(segments) else "".join(s.get("text", "") for s in kept)


class Transcriber:
    def __init__(self, model_name: str, language: str, hallucinations: list[str]):
        import mlx_whisper

        self.mlx_whisper = mlx_whisper
        self.model_name = model_name
        self.language = language
        self.hallucinations = hallucinations

    def __call__(self, audio: np.ndarray, prompt: str = "") -> str:
        result = self.mlx_whisper.transcribe(
            np.ascontiguousarray(audio, dtype=np.float32),
            path_or_hf_repo=self.model_name,
            language=self.language,
            initial_prompt=prompt or None,
            fp16=True,
            condition_on_previous_text=False,
        )
        return clean_text(join_segments(result), self.hallucinations)
